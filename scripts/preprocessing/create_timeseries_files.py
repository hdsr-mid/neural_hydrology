from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neural_hydrology.scripts.preprocessing.meteo.import_knmi_station_cabauw import (
    load_knmi_station_cabauw_hourly,
)
from neural_hydrology.scripts.preprocessing.meteo.import_meteo_forecast import (
    load_harmonie_ensemble_forecast_by_basin,
)
from neural_hydrology.scripts.preprocessing.meteo.import_precip_radar_hourly import (
    load_precip_radar_hourly_by_shape,
)
from neural_hydrology.scripts.preprocessing.meteo.radar_hdf5 import read_polders

LOGGER = logging.getLogger(__name__)

# Max number of KNMI Open Data file downloads in import_rtcor_from per run (5-min RTCOR slots).
# Lower = sneller en minder API-belasting; hoger = meer gaten kunnen worden opgevuld.
RTCOR_MAX_DOWNLOADS = 100

# Max. gap (uren) voor korte interpolatie meteo; ook drempel voor WARNING bij lange ontbrekende periodes neerslag.
METEO_INTERP_LIMIT_HOURS = 3


def _dedupe_subsumed_nan_segments(
    segments: list[tuple[int, int, int, pd.Timestamp, pd.Timestamp]],
) -> list[tuple[int, int, int, pd.Timestamp, pd.Timestamp]]:
    """
    Drop segments whose index range [s,e] lies strictly inside another segment's range.
    Avoids logging a short gap when a longer reported gap already covers the same timeline.
    Each segment is (s, e, n_slots, t_start, t_end).
    """
    if len(segments) <= 1:
        return segments
    # Deduplicate identical (s, e)
    by_key: dict[tuple[int, int], tuple[int, int, int, pd.Timestamp, pd.Timestamp]] = {}
    for seg in segments:
        s, e = seg[0], seg[1]
        by_key[(s, e)] = seg
    uniq = list(by_key.values())

    kept: list[tuple[int, int, int, pd.Timestamp, pd.Timestamp]] = []
    for i, seg_i in enumerate(uniq):
        s_i, e_i = seg_i[0], seg_i[1]
        inside_wider_other = False
        for j, seg_j in enumerate(uniq):
            if i == j:
                continue
            s_j, e_j = seg_j[0], seg_j[1]
            # j covers i on the time index and is strictly wider than i
            if s_j <= s_i and e_j >= e_i and (s_j < s_i or e_j > e_i):
                inside_wider_other = True
                break
        if not inside_wider_other:
            kept.append(seg_i)
    return kept


def _warn_nan_gaps_exceed_limit(
    master: pd.DatetimeIndex,
    arr_time_ens: np.ndarray,
    *,
    variable: str,
    limit_hours: float,
    basin_label: str | None,
) -> None:
    """
    Log one WARNING per (variable, basin): all contiguous NaN runs on member 1 that exceed
    limit_hours (hourly index), summarized in a single message.
    Used for neerslag, temperatuur, u, v, straling.
    """
    a = np.asarray(arr_time_ens, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != len(master):
        return
    mask = np.isnan(a[:, 0])
    if not np.any(mask):
        return
    m = mask.astype(np.int8)
    edges = np.diff(np.r_[0, m, 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    basin = basin_label or "?"

    segments: list[tuple[int, int, int, pd.Timestamp, pd.Timestamp]] = []
    for s, e in zip(starts, ends):
        if s > e:
            continue
        n_slots = e - s + 1
        if float(n_slots) <= limit_hours:
            continue
        segments.append((s, e, n_slots, master[s], master[e]))

    segments = _dedupe_subsumed_nan_segments(segments)
    if not segments:
        return

    parts = [
        f"{n_slots} tijdstappen ({t0.isoformat()} — {t1.isoformat()})"
        for _, _, n_slots, t0, t1 in segments
    ]
    LOGGER.warning(
        "Missende data (%s) gebied=%s: %d periode(s) zonder data (> limiet %s u): %s.",
        variable,
        basin,
        len(segments),
        limit_hours,
        "; ".join(parts),
    )


@dataclass(frozen=True)
class ExampleEncoding:
    date_units: str = "hours since 2014-01-01 01:00:00"
    date_calendar: str = "standard"
    date_dtype: str = "float64"


@dataclass(frozen=True)
class MissingDataConfig:
    # neerslag: ontbrekende waarden als 0 mm/u (historisch + forecast)
    neerslag_fill_nan_with_zero: bool = True


def _to_naive_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        # assume already UTC
        return pd.DatetimeIndex(idx).tz_localize(None)
    return pd.DatetimeIndex(idx.tz_convert("UTC").tz_localize(None))


def _interpolate_meteo_2d_short_gaps(arr: np.ndarray, *, limit: int) -> np.ndarray:
    """
    Per ensemble lid: lineair interpoleren langs de tijd-as; alleen korte gaten (limit uren).
    Werkt op de volledige gemergde reeks (historisch + forecast).
    """
    a = np.asarray(arr, dtype=np.float64, order="C")
    _, n_ens = a.shape
    for j in range(n_ens):
        s = pd.Series(a[:, j])
        a[:, j] = s.interpolate(method="linear", limit=limit, limit_direction="both").to_numpy()
    return a.astype(np.float32)


def _fill_neerslag_nan_zero(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return np.where(np.isnan(a), np.float32(0.0), a)


def _expand_det_to_ens(values: np.ndarray, n_ens: int = 30) -> np.ndarray:
    """
    Expand deterministic (time,) to (time, n_ens) by repetition.
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return np.repeat(v, repeats=n_ens, axis=1)


def _new_ds(date: np.ndarray) -> xr.Dataset:
    enc = ExampleEncoding()
    ds = xr.Dataset(coords={"date": ("date", date.astype("datetime64[ns]"))})
    ds["date"].attrs["long_name"] = "date"
    ds["date"].encoding.update({"units": enc.date_units, "calendar": enc.date_calendar, "dtype": enc.date_dtype})
    return ds


def _add_member_vars(ds: xr.Dataset, base: str, arr_time_ens: np.ndarray, *, units: str) -> None:
    if arr_time_ens.ndim != 2 or arr_time_ens.shape[1] != 30:
        raise ValueError(f"Expected (time,30) for {base}, got shape={arr_time_ens.shape}")
    for k in range(30):
        name = f"{base}_{k+1}"
        ds[name] = ("date", arr_time_ens[:, k].astype("float32"))
        if units:
            ds[name].attrs["units"] = units


def _merge_hist_and_fc(
    *,
    hist_date: pd.DatetimeIndex,
    hist_det: dict[str, pd.Series],
    fc_date: pd.DatetimeIndex,
    fc_ens: dict[str, np.ndarray],
    missing_cfg: MissingDataConfig,
    basin_label: str | None = None,
) -> xr.Dataset:
    """
    Build a single continuous dataset with 30 ensemble members.

    Merge on the union of historical and forecast timestamps. Where both have a value, historical
    wins; otherwise use forecast or historical whichever is non-NaN (overlap uses historical first).

    Missing data (na merge):
    - temperatuur, straling, u, v: korte lineaire interpolatie (METEO_INTERP_LIMIT_HOURS).
    - neerslag: zelfde gap-logging als meteo; NaN -> 0 mm/u indien neerslag_fill_nan_with_zero.
    """
    # master date as naive UTC
    hist_date_n = _to_naive_utc_index(hist_date)
    fc_date_n = pd.DatetimeIndex(fc_date).tz_localize(None) if pd.DatetimeIndex(fc_date).tz is not None else pd.DatetimeIndex(fc_date)
    master = pd.DatetimeIndex(sorted(set(hist_date_n.to_pydatetime()).union(set(fc_date_n.to_pydatetime()))))
    master = pd.DatetimeIndex(master).tz_localize(None)

    ds = _new_ds(master.to_numpy(dtype="datetime64[ns]"))

    # historical: align to master, expand to 30; merge with fc below
    for base in ("neerslag", "temperatuur", "u", "v", "straling"):
        s = hist_det.get(base)
        if s is None:
            # allow missing (e.g., if a source was skipped), keep NaN
            det = pd.Series(index=hist_date_n, data=np.nan, dtype=float)
        else:
            det = pd.Series(s.values, index=_to_naive_utc_index(pd.DatetimeIndex(s.index))).sort_index()

        det_master = det.reindex(master, copy=False)
        det_arr = det_master.to_numpy(dtype=np.float32)
        hist_ens = _expand_det_to_ens(det_arr, n_ens=30)

        fc_arr = np.full((len(master), 30), np.nan, dtype=np.float32)
        fc_src = fc_ens.get(base)
        if fc_src is not None and len(fc_date_n) > 0:
            fc_df = pd.DataFrame(fc_src, index=fc_date_n, columns=list(range(30)))
            fc_df = fc_df.reindex(master, copy=False)
            fc_arr = fc_df.to_numpy(dtype=np.float32)

        # Prefer historical where present; fill from forecast elsewhere (overlap: historical first).
        out = np.where(np.isnan(hist_ens), fc_arr, hist_ens).astype(np.float32)

        if base == "neerslag":
            _warn_nan_gaps_exceed_limit(
                master,
                out,
                variable=base,
                limit_hours=float(METEO_INTERP_LIMIT_HOURS),
                basin_label=basin_label,
            )
            if missing_cfg.neerslag_fill_nan_with_zero:
                out = _fill_neerslag_nan_zero(out)
        elif base in {"temperatuur", "u", "v", "straling"}:
            _warn_nan_gaps_exceed_limit(
                master,
                out,
                variable=base,
                limit_hours=float(METEO_INTERP_LIMIT_HOURS),
                basin_label=basin_label,
            )
            out = _interpolate_meteo_2d_short_gaps(out, limit=METEO_INTERP_LIMIT_HOURS)

        units = {
            "neerslag": "mm",
            "temperatuur": "graden Celsius",
            "u": "m s-1",
            "v": "m s-1",
            "straling": "J/cm2",
        }[base]
        _add_member_vars(ds, base, out, units=units)

    return ds


def create_timeseries_files(
    *,
    days: int = 365,
    missing_cfg: MissingDataConfig | None = None,
    basin_ids: list[str] | None = None,
) -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    LOGGER.setLevel(logging.INFO)

    if missing_cfg is None:
        missing_cfg = MissingDataConfig()

    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "neural_hydrology" / "data_ens"
    out_dir = data_dir / "time_series"
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading historical Cabauw meteo (days=%s)...", days)
    df_meteo = load_knmi_station_cabauw_hourly(days=days)
    if df_meteo.empty:
        raise RuntimeError("Cabauw historical meteo loader returned empty DataFrame.")

    LOGGER.info(
        "Loading historical radar precipitation (days=%s, RTCOR_MAX_DOWNLOADS=%s)...",
        days,
        RTCOR_MAX_DOWNLOADS,
    )
    df_precip_by_shape = load_precip_radar_hourly_by_shape(
        days=days,
        rtcor_max_downloads=RTCOR_MAX_DOWNLOADS,
    )
    if df_precip_by_shape.empty:
        LOGGER.warning("Radar precipitation loader returned empty DataFrame; neerslag will be NaN for all basins.")

    LOGGER.info("Loading HARMONIE ensemble forecast (this can take a while)...")
    fc = load_harmonie_ensemble_forecast_by_basin()

    polders = read_polders()
    all_basin_ids = polders["SHAPE_ID"].astype(str).to_list()
    if basin_ids is None:
        basin_ids = all_basin_ids
    else:
        requested = [str(x) for x in basin_ids]
        missing = sorted(set(requested) - set(all_basin_ids))
        if missing:
            raise RuntimeError(f"Requested basin_ids not found in attributes CSV: {missing[:10]}")
        basin_ids = requested
    LOGGER.info("Writing per-basin netCDF for %d gebieden to %s", len(basin_ids), out_dir)

    # Historical deterministic series (same for all basins except precipitation)
    hist_date = pd.DatetimeIndex(df_meteo.index)

    for bid in basin_ids:
        # neerslag deterministic is basin-specific
        if df_precip_by_shape.empty or bid not in df_precip_by_shape.columns.astype(str).tolist():
            neerslag = pd.Series(index=hist_date, data=np.nan, dtype=float)
        else:
            neerslag = df_precip_by_shape[bid]

        hist_det = {
            "neerslag": neerslag,
            "temperatuur": df_meteo["temperatuur"],
            "u": df_meteo["u"],
            "v": df_meteo["v"],
            "straling": df_meteo["straling"],
        }

        fc_basin = fc.get(bid)
        if fc_basin is None:
            raise RuntimeError(f"Forecast does not contain basin {bid}.")

        ds = _merge_hist_and_fc(
            hist_date=hist_date,
            hist_det=hist_det,
            fc_date=fc_basin["date"],
            fc_ens={
                "neerslag": fc_basin["neerslag"],
                "temperatuur": fc_basin["temperatuur"],
                "u": fc_basin["u"],
                "v": fc_basin["v"],
                "straling": fc_basin["straling"],
            },
            missing_cfg=missing_cfg,
            basin_label=str(bid),
        )

        out_path = out_dir / f"{bid}.nc"
        ds.to_netcdf(out_path)

    LOGGER.info("Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create per-basin netCDF time series with 30-member ensembles.")
    p.add_argument("--days", type=int, default=370, help="Historical window length (days).")
    p.add_argument(
        "--basin-id",
        action="append",
        default=None,
        help="Optional: restrict to a specific SHAPE_ID (repeatable).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_timeseries_files(
        days=int(args.days),
        missing_cfg=MissingDataConfig(),
        basin_ids=args.basin_id,
    )
