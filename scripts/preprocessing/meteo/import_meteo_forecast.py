from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import geopandas as gpd
from rasterio.transform import Affine
from rasterstats import zonal_stats
from rasterio.transform import from_origin

import xarray as xr

from dotenv import dotenv_values

from neural_hydrology.scripts.preprocessing.meteo.knmi_open_data import KnmiOpenDataClient, KnmiRequestBudgetExceeded

LOGGER = logging.getLogger(__name__)

MAX_API_REQUESTS_PER_RUN = 100
HARMONIE_VERSION = "1.0"


def _knmi_latest_filenames(client: KnmiOpenDataClient, *, dataset: str, n: int, version: str = HARMONIE_VERSION) -> list[str]:
    files = client.list_files(dataset=dataset, version=version, max_keys=n, order_by="created", sorting="desc")
    return [f["filename"] for f in files if isinstance(f.get("filename"), str)]


def _read_polders_4326() -> gpd.GeoDataFrame:
    nh_root = Path(__file__).resolve().parents[3]
    attr_path = nh_root / "data_ens" / "attributes" / "polders_data_aangevuld.csv"
    df = pd.read_csv(attr_path)
    geom_series = gpd.GeoSeries.from_wkt(df["geom_simple"])
    gdf = gpd.GeoDataFrame(df[["SHAPE_ID"]].copy(), geometry=geom_series, crs="EPSG:28992")
    return gdf.to_crs("EPSG:4326")


def _harmonie_affine_transform() -> Affine:
    # Consistent with existing deterministic example for HARMONIE Cy43.
    # Grid is regular lat/lon with origin (0E, 56.002N).
    return from_origin(0.0, 56.0020, 0.0296, 0.0184)


def _safe_extract_flattened_member(tf: tarfile.TarFile, member: tarfile.TarInfo, out_dir: Path) -> Path:
    """
    Extract a tar member to out_dir using the basename only.

    This avoids path traversal issues and makes outputs deterministic.
    """
    if not member.isfile():
        raise ValueError(f"Expected regular file in tar, got type={member.type} name={member.name!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(member.name).name
    src = tf.extractfile(member)
    if src is None:
        raise RuntimeError(f"Could not extract file content for member {member.name!r}")
    with src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out_path


def _parse_member_and_step(filename: str) -> tuple[int, datetime, int]:
    """
    Parse member id, init time (UTC) and lead time (hours) from KNMI tar member name.

    Examples:
    - harm43_v1_ned_uwcw_meteo_030_202604071100_00300_GB
    - harm43_v1_ned_uwcw_renew_001_202604071200_00000_GB
    """
    m = re.search(r"_(meteo|renew)_(\d{3})_(\d{12})_(\d{5})_GB$", filename)
    if not m:
        raise ValueError(f"Unexpected HARMONIE member filename format: {filename}")
    member_id = int(m.group(2))
    init = datetime.strptime(m.group(3), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    step_code = int(m.group(4))
    lead_hours = step_code // 100
    return member_id, init, lead_hours


def _list_tar_members(tar_path: Path, tag: str) -> list[str]:
    # tag in {"meteo","renew"}
    pat = re.compile(rf"harm43_v1_ned_uwcw_{re.escape(tag)}_\d{{3}}_\d{{12}}_\d{{5}}_GB$")
    with tarfile.open(tar_path) as tf:
        return sorted([m.name for m in tf if pat.search(m.name)])


def _unique_member_ids(tar_members: list[str], tag: str) -> list[int]:
    pat = re.compile(rf"harm43_v1_ned_uwcw_{re.escape(tag)}_(\d{{3}})_")
    ids = sorted({int(pat.search(m).group(1)) for m in tar_members if pat.search(m)})
    return ids


def _unique_lead_hours(tar_members: list[str]) -> list[int]:
    leads = sorted({_parse_member_and_step(m)[2] for m in tar_members})
    return leads


def _parse_tar_init_time(tar_filename: str) -> datetime:
    m = re.search(r"harm43_v1_P2[ab]_(\d{10})\.tar$", tar_filename)
    if not m:
        raise ValueError(f"Unexpected tar filename format: {tar_filename}")
    return datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)


def _parse_tar_init_key(tar_filename: str) -> str:
    m = re.search(r"harm43_v1_P2[ab]_(\d{10})\.tar$", tar_filename)
    if not m:
        raise ValueError(f"Unexpected tar filename format: {tar_filename}")
    return m.group(1)


def _expected_batch_for_hour_offset(offset_h: int) -> set[int]:
    """
    KNMI rolling ensemble batches (by hour offset within the 6-hour composition window):
    offset 0 -> members 26..30
    offset 1 -> members 21..25
    offset 2 -> members 16..20
    offset 3 -> members 11..15
    offset 4 -> members 6..10
    offset 5 -> members 1..5
    """
    if not (0 <= offset_h <= 5):
        raise ValueError(offset_h)
    start = 26 - 5 * offset_h
    end = start + 4
    return set(range(start, end + 1))


def _expected_batch_for_hour_number(hour_number: int) -> set[int]:
    """
    Hour numbering as in KNMI description (and as requested by user):
    hour 1 -> members 1..5
    hour 2 -> members 6..10
    ...
    hour 6 -> members 26..30
    """
    if not (1 <= hour_number <= 6):
        raise ValueError(hour_number)
    start = 1 + 5 * (hour_number - 1)
    return set(range(start, start + 5))


def _members_in_tar(tar_path: Path, tag: str) -> set[int]:
    members = _list_tar_members(tar_path, tag=tag)
    ids = set(_unique_member_ids(members, tag=tag))
    # member 0 is always present but not part of the 30-member EPS.
    ids.discard(0)
    return ids


def _extract_selected_members(tar_path: Path, out_dir: Path, member_names: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        # Extract only what we need.
        existing = {p.name for p in out_dir.glob("*_GB")}
        names = set(member_names)
        to_extract = [m for m in tf.getmembers() if m.name in names and Path(m.name).name not in existing]
        if not to_extract:
            return
        for m in to_extract:
            _safe_extract_flattened_member(tf, m, out_dir)


def _open_grib_field(
    path: Path,
    *,
    indicator_of_parameter: int,
    type_of_level: str,
    level: int,
    time_range_indicator: int,
) -> xr.DataArray:
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "indicatorOfParameter": indicator_of_parameter,
                "typeOfLevel": type_of_level,
                "level": level,
                "timeRangeIndicator": time_range_indicator,
            }
        },
    )
    # cfgrib exposes variable name as 'unknown' for GRIB1 from this centre.
    if "unknown" not in ds:
        raise KeyError(f"Expected 'unknown' var not found in {path.name}. vars={list(ds.data_vars)}")
    return ds["unknown"]


def _zonal_mean_for_all_polders(grid_values: np.ndarray, polders_4326: gpd.GeoDataFrame, affine) -> np.ndarray:
    # Match deterministic approach: flip the array vertically.
    grid_values = grid_values[::-1]
    stats = zonal_stats(
        polders_4326,
        grid_values,
        affine=affine,
        stats=["mean"],
        all_touched=True,
        nodata=-999,
    )
    return np.array([d.get("mean", np.nan) for d in stats], dtype=float)


def load_harmonie_ensemble_forecast_by_basin() -> dict[str, dict[str, object]]:
    """
    Download/process latest HARMONIE p2a/p2b tar files and return ensembles per basin.

    Returns a mapping:
    {
      "<SHAPE_ID>": {
        "date": pd.DatetimeIndex (naive datetime64[ns], valid times),
        "neerslag": np.ndarray shape (time,30) mm/hour,
        "temperatuur": np.ndarray shape (time,30) Celsius,
        "u": np.ndarray shape (time,30) m/s,
        "v": np.ndarray shape (time,30) m/s,
        "straling": np.ndarray shape (time,30) J/cm2 per hour
      },
      ...
    }
    """
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("import_harmonie_ensemble_forecast: start")

    nh_root = Path(__file__).resolve().parents[3]
    data_dir = nh_root / "data_ens"
    tmp_dir = data_dir / "_tmp_harmonie"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = dotenv_values(nh_root / ".env")
    client = KnmiOpenDataClient.from_neural_hydrology_env().with_request_budget(max_requests=MAX_API_REQUESTS_PER_RUN)

    # ENSEMBLE settings:
    # - ENSEMBLE_STARTTIME: YYYYMMDDHH (UTC). This is the OLDEST run in the 6-hour rolling window ("hour 1"),
    #   which should contain members 1..5.
    # - DOWNLOAD_ENSEMBLE: 1/0. If 0, only use locally present tars in tmp_dir.
    # Precedence: .env first, then process env, then defaults.
    compose_start_init = env.get("ENSEMBLE_STARTTIME") or os.environ.get("ENSEMBLE_STARTTIME")  # e.g. "2026040718"
    download_raw = env.get("DOWNLOAD_ENSEMBLE") or os.environ.get("DOWNLOAD_ENSEMBLE") or "1"
    download_enabled = download_raw.lower() not in {"0", "false", "no"}

    if "ENSEMBLE_STARTTIME" not in env:
        LOGGER.warning(
            "ENSEMBLE_STARTTIME not found in neural_hydrology/.env; "
            "falling back to process env and then script defaults."
        )

    if download_enabled:
        LOGGER.info("DOWNLOAD_ENSEMBLE enabled: clearing %s", tmp_dir)
        for p in tmp_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    latest_p2a: list[str] = []
    latest_p2b: list[str] = []
    if download_enabled:
        try:
            latest_p2a = _knmi_latest_filenames(client, dataset="harmonie_arome_cy43_p2a", n=48)
            latest_p2b = _knmi_latest_filenames(client, dataset="harmonie_arome_cy43_p2b", n=48)
        except KnmiRequestBudgetExceeded as e:
            raise RuntimeError(f"KNMI request budget exceeded while listing latest files ({e}). Rerun in chunks.") from e

    p2a_by_key = {_parse_tar_init_key(f): f for f in latest_p2a}
    p2b_by_key = {_parse_tar_init_key(f): f for f in latest_p2b}

    if compose_start_init is None:
        if not download_enabled:
            raise RuntimeError("Set ENSEMBLE_STARTTIME when DOWNLOAD_ENSEMBLE=0.")
        common_keys = sorted(set(p2a_by_key.keys()).intersection(set(p2b_by_key.keys())))
        if len(common_keys) < 6:
            raise RuntimeError(f"Not enough common p2a/p2b runs available to compose ensemble: {len(common_keys)}")
        # Default to the oldest of the last 6 common hours (hour 1)
        compose_start_init = common_keys[-6]

    start_dt = datetime.strptime(compose_start_init, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    keys_asc = [(start_dt + timedelta(hours=i)).strftime("%Y%m%d%H") for i in range(0, 6)]
    inits = [datetime.strptime(k, "%Y%m%d%H").replace(tzinfo=timezone.utc) for k in keys_asc]
    for a, b in zip(inits, inits[1:]):
        if b - a != timedelta(hours=1):
            raise RuntimeError(f"Selected runs are not consecutive ascending hours: {inits}")
    LOGGER.info("Composition start init (hour 1, UTC, oldest): %s", compose_start_init)
    LOGGER.info("Composition keys oldest->newest (hour1..hour6, UTC): %s", keys_asc)

    p2a_tars: list[Path] = []
    p2b_tars: list[Path] = []
    for hour_number, k in enumerate(keys_asc, start=1):
        if download_enabled:
            if k not in p2a_by_key or k not in p2b_by_key:
                raise RuntimeError(f"Cannot find p2a/p2b tar for init {k} in API list.")
            try:
                ta = client.download_file(
                    dataset="harmonie_arome_cy43_p2a",
                    version=HARMONIE_VERSION,
                    filename=p2a_by_key[k],
                    out_dir=tmp_dir,
                    log_each_download=True,
                )
                tb = client.download_file(
                    dataset="harmonie_arome_cy43_p2b",
                    version=HARMONIE_VERSION,
                    filename=p2b_by_key[k],
                    out_dir=tmp_dir,
                    log_each_download=True,
                )
                p2a_tars.append(ta)
                p2b_tars.append(tb)
            except KnmiRequestBudgetExceeded as e:
                raise RuntimeError(f"KNMI request budget exceeded while downloading tar files ({e}). Rerun in chunks.") from e
        else:
            p2a_tars.append(tmp_dir / f"harm43_v1_P2a_{k}.tar")
            p2b_tars.append(tmp_dir / f"harm43_v1_P2b_{k}.tar")
            if not p2a_tars[-1].exists() or not p2b_tars[-1].exists():
                raise FileNotFoundError(f"Missing local tar(s) for init {k}: {p2a_tars[-1].name}, {p2b_tars[-1].name}")
        # Validate immediately after downloading/locating this hour's pair.
        expected = _expected_batch_for_hour_number(hour_number)
        ma = _members_in_tar(p2a_tars[-1], tag="meteo")
        mb = _members_in_tar(p2b_tars[-1], tag="renew")
        if ma != expected or mb != expected:
            msg = (
                "Rolling ensemble misalignment: "
                f"hour={hour_number} init={k} UTC expected={sorted(expected)} "
                f"got p2a={sorted(ma)} ({p2a_tars[-1].name}) "
                f"p2b={sorted(mb)} ({p2b_tars[-1].name}). "
                "Pick ENSEMBLE_STARTTIME as the oldest hour in the 6-hour window ('hour 1')."
            )
            LOGGER.error(msg)
            raise RuntimeError(msg)
        LOGGER.info(
            "Validated hour %d (compose init=%s UTC): members=%s",
            hour_number,
            k,
            sorted(expected),
        )

    # At this point we've validated all 6 hour pairs in order.
    polders = _read_polders_4326()
    basin_ids = polders["SHAPE_ID"].astype(str).to_list()
    affine = _harmonie_affine_transform()

    # Build member->tar mapping (1..30)
    member_to_p2a: dict[int, Path] = {}
    member_to_p2b: dict[int, Path] = {}
    for ta, tb in zip(p2a_tars, p2b_tars):
        for m in _members_in_tar(ta, tag="meteo"):
            member_to_p2a[m] = ta
        for m in _members_in_tar(tb, tag="renew"):
            member_to_p2b[m] = tb

    members = list(range(1, 31))
    if set(member_to_p2a.keys()) != set(members) or set(member_to_p2b.keys()) != set(members):
        raise RuntimeError("Could not build full 30-member mapping from the 6 runs.")

    slot_count = 30

    # Determine valid_time overlap across all members based on init times + lead hours
    # We assume each tar contains the same set of lead hours.
    example_members = _list_tar_members(p2a_tars[0], tag="meteo")
    leads = _unique_lead_hours(example_members)
    times_per_member: dict[int, np.ndarray] = {}
    for m in members:
        # pick any member file name from tar list to parse init, but easier: parse tar init (hour) and assume minute 00.
        init = _parse_tar_init_time(member_to_p2a[m].name).replace(minute=0)
        times_per_member[m] = np.array([(init + timedelta(hours=h)).replace(tzinfo=None) for h in leads], dtype="datetime64[ns]")

    # Common overlap only (keep everything as datetime64[ns] to avoid dtype mismatches)
    common_times = set(times_per_member[members[0]].astype("datetime64[ns]").tolist())
    for m in members[1:]:
        common_times &= set(times_per_member[m].astype("datetime64[ns]").tolist())
    times = np.array(sorted(common_times), dtype="datetime64[ns]")
    t_len = len(times)
    if t_len == 0:
        raise RuntimeError("No common overlap in valid times across all members.")

    # KNMI lagged-ensemble documentation: a "full" 30-member ensemble yields 54 forecast hours.
    # With 6 hourly-staggered runs and lead times typically spanning +00..+60 (inclusive),
    # the pure intersection often results in 56 valid times (e.g. +05..+60). We trim one hour
    # on both ends to match the documented 54-hour horizon (e.g. +06..+59).
    if t_len == 56:
        times = times[1:-1]
        t_len = len(times)
    if t_len != 54:
        LOGGER.warning("Expected 54 time steps per KNMI docs, got %d. Proceeding anyway.", t_len)

    # Prepare per-basin arrays (time, slot)
    def make_var():
        return np.full((t_len, slot_count), np.nan, dtype=float)

    per_basin: dict[str, dict[str, np.ndarray]] = {bid: {} for bid in basin_ids}
    for bid in basin_ids:
        per_basin[bid]["neerslag"] = make_var()
        per_basin[bid]["temperatuur"] = make_var()
        per_basin[bid]["u"] = make_var()
        per_basin[bid]["v"] = make_var()
        per_basin[bid]["straling_j_m2_h"] = make_var()  # internal hourly increment J/m²

    extract_p2a = tmp_dir / "extract_p2a_composed"
    extract_p2b = tmp_dir / "extract_p2b_composed"
    extract_p2a.mkdir(parents=True, exist_ok=True)
    extract_p2b.mkdir(parents=True, exist_ok=True)

    # Extract only files needed for members 1..30 and lead hours in the overlap window.
    # First find needed lead hours for overlap (relative to each member init).
    lead_set = set(leads)

    # Read and zonal-average per lead time for each available member.
    for mid in members:
        slot = mid - 1
        LOGGER.info("Processing composed member %s into slot %s", mid, slot + 1)

        # Collect cumulative precipitation (mm) and temperature, u, v per lead
        # We'll build in local lead order then subset to overlap times.
        precip_cum_full = np.full((len(leads), len(basin_ids)), np.nan, dtype=float)
        temp_c_full = np.full((len(leads), len(basin_ids)), np.nan, dtype=float)
        wind_u_full = np.full((len(leads), len(basin_ids)), np.nan, dtype=float)
        wind_v_full = np.full((len(leads), len(basin_ids)), np.nan, dtype=float)
        rad_cum_full = np.full((len(leads), len(basin_ids)), np.nan, dtype=float)

        # Extract needed GB files for this member from its tar(s)
        ta = member_to_p2a[mid]
        tb = member_to_p2b[mid]
        members_a = _list_tar_members(ta, tag="meteo")
        members_b = _list_tar_members(tb, tag="renew")
        need_names_a: list[str] = []
        for n in members_a:
            member_id, _, lead = _parse_member_and_step(n)
            if member_id == mid and lead in lead_set:
                need_names_a.append(n)
        need_names_b: list[str] = []
        for n in members_b:
            member_id, _, lead = _parse_member_and_step(n)
            if member_id == mid and lead in lead_set:
                need_names_b.append(n)
        _extract_selected_members(ta, extract_p2a, need_names_a)
        _extract_selected_members(tb, extract_p2b, need_names_b)

        # Index extracted files for this member
        idx_a: dict[tuple[int, int], Path] = {}
        for p in extract_p2a.glob("*_GB"):
            mid2, _, lead = _parse_member_and_step(p.name)
            if mid2 == mid:
                idx_a[(mid2, lead)] = p
        idx_b: dict[tuple[int, int], Path] = {}
        for p in extract_p2b.glob("*_GB"):
            mid2, _, lead = _parse_member_and_step(p.name)
            if mid2 == mid:
                idx_b[(mid2, lead)] = p

        for li, lead in enumerate(leads):
            f_meteo = idx_a.get((mid, lead))
            f_renew = idx_b.get((mid, lead))
            if f_meteo is None or f_renew is None:
                continue

            # precipitation cumulative (indicator 181, TRI 4, heightAboveGround level 0)
            da_p = _open_grib_field(
                f_meteo, indicator_of_parameter=181, type_of_level="heightAboveGround", level=0, time_range_indicator=4
            )
            # temperature at 2m (indicator 11, TRI 0)
            da_t = _open_grib_field(
                f_meteo, indicator_of_parameter=11, type_of_level="heightAboveGround", level=2, time_range_indicator=0
            )
            # u/v at 10m (33/34, TRI 0)
            da_u = _open_grib_field(
                f_meteo, indicator_of_parameter=33, type_of_level="heightAboveGround", level=10, time_range_indicator=0
            )
            da_v = _open_grib_field(
                f_meteo, indicator_of_parameter=34, type_of_level="heightAboveGround", level=10, time_range_indicator=0
            )
            # global radiation cumulative (indicator 117, TRI 4) from p2b
            da_r = _open_grib_field(
                f_renew, indicator_of_parameter=117, type_of_level="heightAboveGround", level=0, time_range_indicator=4
            )

            # Diagnostics: NaNs can originate from source fields or be introduced by zonal-mean/missing leads.
            p_vals = da_p.values
            if np.isnan(p_vals).any():
                nan_frac = float(np.isnan(p_vals).mean())
                LOGGER.warning(
                    "Source precip field contains NaNs: member=%s lead=+%02dh file=%s nan_frac=%.4f",
                    mid,
                    lead,
                    f_meteo.name,
                    nan_frac,
                )
            precip_mean = _zonal_mean_for_all_polders(p_vals, polders, affine)
            if np.isnan(precip_mean).any():
                nan_frac = float(np.isnan(precip_mean).mean())
                LOGGER.warning(
                    "Zonal-mean precip has NaNs: member=%s lead=+%02dh file=%s nan_frac=%.4f",
                    mid,
                    lead,
                    f_meteo.name,
                    nan_frac,
                )
            precip_cum_full[li, :] = precip_mean
            tvals = _zonal_mean_for_all_polders(da_t.values, polders, affine)
            # Convert K -> C if values look like Kelvin
            if np.nanmean(tvals) > 100:
                tvals = tvals - 273.15
            temp_c_full[li, :] = tvals
            wind_u_full[li, :] = _zonal_mean_for_all_polders(da_u.values, polders, affine)
            wind_v_full[li, :] = _zonal_mean_for_all_polders(da_v.values, polders, affine)
            r_vals = da_r.values
            if np.isnan(r_vals).any():
                nan_frac = float(np.isnan(r_vals).mean())
                LOGGER.warning(
                    "Source radiation field contains NaNs: member=%s lead=+%02dh file=%s nan_frac=%.4f",
                    mid,
                    lead,
                    f_renew.name,
                    nan_frac,
                )
            rad_mean = _zonal_mean_for_all_polders(r_vals, polders, affine)
            if np.isnan(rad_mean).any():
                nan_frac = float(np.isnan(rad_mean).mean())
                LOGGER.warning(
                    "Zonal-mean radiation has NaNs: member=%s lead=+%02dh file=%s nan_frac=%.4f",
                    mid,
                    lead,
                    f_renew.name,
                    nan_frac,
                )
            rad_cum_full[li, :] = rad_mean

        # Convert cumulative -> hourly increments (time axis is hourly)
        precip_inc_full = np.vstack([np.full((1, len(basin_ids)), np.nan), np.diff(precip_cum_full, axis=0)])
        rad_inc_full = np.vstack([np.full((1, len(basin_ids)), np.nan), np.diff(rad_cum_full, axis=0)])
        # Negative increments can occur due to cumulative resets/misalignment.
        # For downstream consumers we clip these to 0 instead of marking them missing.
        precip_inc_full = np.where(precip_inc_full < 0, 0.0, precip_inc_full)
        rad_inc_full = np.where(rad_inc_full < 0, 0.0, rad_inc_full)

        # Diagnostics: if we missed lead files, cumulative arrays will remain NaN.
        missing_leads = int(np.isnan(precip_cum_full).all(axis=1).sum())
        if missing_leads:
            LOGGER.warning("Missing lead files for member=%s: %d/%d leads unresolved", mid, missing_leads, len(leads))

        # Subset to overlap times (use numpy datetime64 comparisons)
        member_times_full = times_per_member[mid].astype("datetime64[ns]")
        keep_mask = np.isin(member_times_full, times)
        precip_inc = precip_inc_full[keep_mask, :]
        rad_inc = rad_inc_full[keep_mask, :]
        temp_c = temp_c_full[keep_mask, :]
        wind_u = wind_u_full[keep_mask, :]
        wind_v = wind_v_full[keep_mask, :]

        # Store into per-basin arrays for this slot
        for bi, bid in enumerate(basin_ids):
            per_basin[bid]["neerslag"][:, slot] = precip_inc[:, bi]
            per_basin[bid]["temperatuur"][:, slot] = temp_c[:, bi]
            per_basin[bid]["u"][:, slot] = wind_u[:, bi]
            per_basin[bid]["v"][:, slot] = wind_v[:, bi]
            per_basin[bid]["straling_j_m2_h"][:, slot] = rad_inc[:, bi]

    out: dict[str, dict[str, object]] = {}
    date_index = pd.DatetimeIndex(pd.to_datetime(times))
    for bid in basin_ids:
        data = per_basin[bid]
        out[bid] = {
            "date": date_index,
            "neerslag": data["neerslag"].astype(np.float32),
            "temperatuur": data["temperatuur"].astype(np.float32),
            "u": data["u"].astype(np.float32),
            "v": data["v"].astype(np.float32),
            "straling": (data["straling_j_m2_h"] / 1e4).astype(np.float32),
        }

    LOGGER.info("Done. Built HARMONIE ensembles for %d basins (time=%d, members=30).", len(out), len(date_index))
    return out


def import_harmonie_ensemble_forecast() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    LOGGER.setLevel(logging.INFO)
    res = load_harmonie_ensemble_forecast_by_basin()
    any_bid = next(iter(res.keys()), None)
    if any_bid is None:
        raise RuntimeError("No basins produced by forecast loader.")
    LOGGER.info("Example basin=%s variables=%s", any_bid, sorted([k for k in res[any_bid].keys() if k != "date"]))


if __name__ == "__main__":
    import_harmonie_ensemble_forecast()
