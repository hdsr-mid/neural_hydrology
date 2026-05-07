from __future__ import annotations

import logging
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neural_hydrology.scripts.preprocessing.meteo.knmi_open_data import (
    HISTORICAL_FETCH_END_OFFSET_HOURS,
    KnmiOpenDataClient,
    KnmiRequestBudgetExceeded,
    load_env_file,
)
LOGGER = logging.getLogger(__name__)

UURGEGEVENS_BASE_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
UURGEGEVENS_STN = 348
UURGEGEVENS_VARS = "DD:FF:T:Q"

TENMIN_DATASET = "10-minute-in-situ-meteorological-observations"
DATASET_VERSION = "1.0"


def _now_utc_hour_end() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _end_utc_from_env_or_now() -> datetime:
    """
    Determine endtime (UTC) for deterministic runs.

    If `ENSEMBLE_STARTTIME` (YYYYMMDDHH) is set in `neural_hydrology/.env`, use it.
    Otherwise, fall back to current UTC hour endtime.
    """
    nh_root = Path(__file__).resolve().parents[3]
    env = load_env_file(nh_root / ".env")
    est = env.get("ENSEMBLE_STARTTIME")
    if est:
        try:
            return datetime.strptime(est, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return _now_utc_hour_end()


def _parse_endtime_from_filename(filename: str) -> pd.Timestamp | None:
    """
    Try to parse the *end timestamp* (UTC) from a KNMI Open Data filename.

    Supports common patterns documented by KNMI. If parsing fails, return None.
    """
    name = Path(filename).name

    # hourly-observations-YYYYMMDD-HH.nc  -> endtime is next hour
    m = re.search(r"(\d{8})-(\d{2})\.nc$", name)
    if m:
        ymd = m.group(1)
        hh = int(m.group(2))
        try:
            start = pd.to_datetime(f"{ymd}{hh:02d}", format="%Y%m%d%H", utc=True)
            return start + pd.Timedelta(hours=1)
        except Exception:
            return None

    # Generic 12 digits (YYYYMMDDHHMM) interpreted as endtime
    m = re.search(r"(\d{12})", name)
    if m:
        raw = m.group(1)
        try:
            return pd.to_datetime(raw, format="%Y%m%d%H%M", utc=True)
        except Exception:
            return None

    # Generic 10 digits (YYYYMMDDHH) interpreted as endtime at full hour
    m = re.search(r"(\d{10})", name)
    if m:
        raw = m.group(1)
        try:
            return pd.to_datetime(raw, format="%Y%m%d%H", utc=True)
        except Exception:
            return None

    return None


def _iter_filenames_old_to_new(
    client: KnmiOpenDataClient,
    *,
    dataset: str,
    version: str,
    page_size: int = 250,
) -> Iterable[str]:
    """
    Iterate filenames from old -> new (ascending).

    Uses paging via start_after_filename to avoid scanning the full catalog in one response.
    """
    start_after: str | None = None
    while True:
        try:
            files = client.list_files(
                dataset=dataset,
                version=version,
                max_keys=page_size,
                order_by="filename",
                sorting="asc",
                start_after_filename=start_after,
            )
        except KnmiRequestBudgetExceeded:
            return
        names = [f.get("filename") for f in files if isinstance(f.get("filename"), str)]
        if not names:
            return
        for n in names:
            yield n
        start_after = names[-1]


def _select_files_by_timerange(
    filenames: Iterable[str], *, start_inclusive: pd.Timestamp, end_inclusive: pd.Timestamp
) -> tuple[list[str], list[str]]:
    """
    Select files whose parsed endtime is within [start, end].

    Returns (selected, unparsed) where unparsed couldn't be parsed from filename.
    """
    selected: list[str] = []
    unparsed: list[str] = []
    for n in filenames:
        ts = _parse_endtime_from_filename(n)
        if ts is None:
            unparsed.append(n)
            continue
        if start_inclusive <= ts <= end_inclusive:
            selected.append(n)
    return selected, unparsed


def _find_time_coord(ds: xr.Dataset) -> str:
    for c in ds.coords:
        if np.issubdtype(ds[c].dtype, np.datetime64):
            return c
    for c in ("time", "datetime", "date"):
        if c in ds.coords and np.issubdtype(ds[c].dtype, np.datetime64):
            return c
    raise RuntimeError(f"Cannot find datetime coordinate in dataset. coords={list(ds.coords)}")


def _find_station_dim_and_index(ds: xr.Dataset, *, station_name_contains: str = "cabauw") -> tuple[str | None, int | None]:
    """
    Return (station_dim, station_index) if a multi-station dataset, else (None, None).
    """
    target = station_name_contains.casefold()

    # Identify candidate station dimensions
    candidate_dims = [d for d in ds.dims if "station" in d.casefold() or d.casefold() in {"id", "stid"}]

    # Search string-like variables/coords that reference a station dimension.
    for var_name in list(ds.coords) + list(ds.data_vars):
        v = ds[var_name]
        if v.ndim != 1:
            continue
        if not any(dim in candidate_dims for dim in v.dims):
            continue
        # Convert to list of strings safely
        try:
            vals = [str(x) for x in v.values.tolist()]
        except Exception:
            continue
        for i, s in enumerate(vals):
            if target in s.casefold():
                station_dim = v.dims[0]
                return station_dim, i

    # If there is a station dim but we cannot find names, we cannot select reliably.
    if candidate_dims:
        return candidate_dims[0], None
    return None, None


def _select_station(ds: xr.Dataset, *, station_name_contains: str = "cabauw") -> xr.Dataset:
    station_dim, idx = _find_station_dim_and_index(ds, station_name_contains=station_name_contains)
    if station_dim is None:
        return ds
    if idx is None:
        raise RuntimeError(
            "Dataset seems to contain a station dimension, but Cabauw could not be identified by name. "
            f"dims={dict(ds.dims)} vars={list(ds.data_vars)[:30]} coords={list(ds.coords)[:30]}"
        )
    return ds.isel({station_dim: idx})


def _score_var(v: xr.DataArray, keywords: list[str]) -> int:
    score = 0
    hay = " ".join(
        [
            v.name or "",
            str(v.attrs.get("standard_name", "")),
            str(v.attrs.get("long_name", "")),
            str(v.attrs.get("units", "")),
        ]
    ).casefold()
    for k in keywords:
        if k.casefold() in hay:
            score += 1
    return score


def _pick_var(ds: xr.Dataset, *, keywords: list[str], must_have_time: str) -> str:
    best: tuple[int, str] | None = None
    for name, v in ds.data_vars.items():
        if must_have_time not in v.dims:
            continue
        s = _score_var(v, keywords)
        if s <= 0:
            continue
        if best is None or s > best[0]:
            best = (s, name)
    if best is None:
        raise RuntimeError(
            f"Could not find variable for keywords={keywords}. "
            f"Available vars={list(ds.data_vars)}"
        )
    return best[1]


def _wind_components_from_speed_dir(speed_ms: pd.Series, wind_from_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Convert wind speed and meteorological wind-from direction (degrees) to u/v.
    u: positive eastward, v: positive northward.
    """
    # Meteorological convention: direction is where wind *comes from*, clockwise from north.
    # Convert to radians and compute components of wind *towards*:
    rad = np.deg2rad(wind_from_deg.astype(float))
    u = -speed_ms.astype(float) * np.sin(rad)
    v = -speed_ms.astype(float) * np.cos(rad)
    return u.astype(float), v.astype(float)


def _wind_components_from_knmi_dd_ff(
    *, wind_from_deg: pd.Series, speed_tenths_ms: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Convert KNMI klimatologie wind (DD, FF) to u/v.

    - DD: wind direction (degrees, wind-from). 0=calm, 990=variable.
    - FF: wind speed in 0.1 m/s.
    """
    sp_ms = pd.to_numeric(speed_tenths_ms, errors="coerce").astype(float) / 10.0
    dd = pd.to_numeric(wind_from_deg, errors="coerce").astype(float)

    calm = dd.eq(0) | sp_ms.le(0)
    variable = dd.eq(990)

    dd_clean = dd.mask(variable).mask(calm)
    u, v = _wind_components_from_speed_dir(sp_ms, dd_clean)
    u = u.mask(variable).where(~calm, other=0.0)
    v = v.mask(variable).where(~calm, other=0.0)
    return u, v


def _to_knmi_uurgegevens_yyyymmddhh(dt: pd.Timestamp) -> str:
    d = pd.Timestamp(dt).tz_convert("UTC")
    return d.strftime("%Y%m%d%H00")


def _fetch_knmi_uurgegevens_text(*, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> str:
    # KNMI uurgegevens: if start and end use the same wall-clock hour (e.g. both 09:00), the
    # service returns one row per day (that hour only) instead of 24 hourly rows. Floor the
    # start to UTC midnight so the range always requests full days. (End may keep the run's hour.)
    start_utc = pd.Timestamp(start_utc).tz_convert("UTC").floor("D")
    query = {
        "stns": str(UURGEGEVENS_STN),
        "vars": UURGEGEVENS_VARS,
        "start": _to_knmi_uurgegevens_yyyymmddhh(start_utc),
        "end": _to_knmi_uurgegevens_yyyymmddhh(end_utc),
    }
    url = f"{UURGEGEVENS_BASE_URL}?{urllib.parse.urlencode(query)}"
    with urllib.request.urlopen(url, timeout=120) as resp:  # nosec - fixed KNMI endpoint + encoded query
        return resp.read().decode("utf-8", errors="replace")


def _parse_knmi_uurgegevens_text(text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        if parts[0].upper() == "STN":
            continue
        try:
            stn = int(parts[0])
            if stn != UURGEGEVENS_STN:
                continue
            ymd = parts[1]
            hh = int(parts[2])
            dd = float(parts[3])
            ff = float(parts[4])
            t = float(parts[5])
            q = float(parts[6])
        except Exception:
            continue

        base = pd.to_datetime(ymd, format="%Y%m%d", utc=True)
        if hh == 24:
            ts = base + pd.Timedelta(days=1)
        else:
            ts = base + pd.Timedelta(hours=hh)

        rows.append({"date": ts, "DD": dd, "FF": ff, "T": t, "Q": q})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    df.index.name = "date"

    temperatuur = pd.to_numeric(df["T"], errors="coerce").astype(float) / 10.0
    straling = pd.to_numeric(df["Q"], errors="coerce").astype(float)

    u, v = _wind_components_from_knmi_dd_ff(wind_from_deg=df["DD"], speed_tenths_ms=df["FF"])

    out = pd.DataFrame(
        {
            "temperatuur": temperatuur,
            "u": u.astype(float),
            "v": v.astype(float),
            "straling": straling,
        },
        index=df.index,
    )
    return out.sort_index()


def _to_series(da: xr.DataArray, time_coord: str) -> pd.Series:
    # Ensure we get a 1D time series indexed by timestamps.
    if da.ndim != 1 or da.dims[0] != time_coord:
        da = da.squeeze()
    idx = pd.DatetimeIndex(pd.to_datetime(da[time_coord].values, utc=True))
    return pd.Series(da.values, index=idx).sort_index()


def _convert_temperature_to_celsius(s: pd.Series, units: str | None) -> pd.Series:
    if units:
        u = units.strip().casefold()
        if u in {"k", "kelvin"}:
            return s.astype(float) - 273.15
    # heuristic
    if float(pd.to_numeric(s, errors="coerce").mean()) > 100:
        return s.astype(float) - 273.15
    return s.astype(float)


def _convert_radiation_to_j_cm2_per_interval(
    s: pd.Series, *, units: str | None, interval_seconds: int, is_flux_mean: bool
) -> pd.Series:
    """
    Convert radiation series to energy per interval in J/cm2.

    - If units are W/m2 and is_flux_mean=True: interpret s as mean flux over interval.
    - If units are J/m2 or J/cm2: interpret s as energy over interval already.
    """
    if units is None:
        raise RuntimeError("Radiation units are unknown; cannot convert safely.")
    u = units.replace(" ", "").casefold()
    # KNMI NetCDF may use Unicode superscripts (e.g. J/cm²)
    u = u.replace("²", "2").replace("³", "3")
    vals = s.astype(float)

    if "w" in u and ("/m2" in u or "m-2" in u):
        if not is_flux_mean:
            # still treat it as mean flux; safest assumption for station products
            is_flux_mean = True
        j_m2 = vals * float(interval_seconds)  # W/m2 * s = J/m2
        return j_m2 / 1e4

    if "j" in u and ("/m2" in u or "m-2" in u):
        return vals / 1e4

    if "j" in u and ("/cm2" in u or "cm-2" in u):
        return vals

    raise RuntimeError(f"Unsupported radiation units: {units!r}")


def _load_station_timeseries_from_files(
    paths: list[Path],
    *,
    station_name_contains: str,
    expected_interval: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Read multiple NetCDF files and return a DataFrame indexed by UTC endtime.

    expected_interval in {"1h","10min"} used for radiation interval conversion.
    Returns: (df, units_by_var)
    """
    if not paths:
        return pd.DataFrame(), {}

    series_parts: list[pd.DataFrame] = []
    units: dict[str, str] = {}

    for p in sorted(paths):
        ds = xr.open_dataset(p)
        try:
            ds = _select_station(ds, station_name_contains=station_name_contains)
            t = _find_time_coord(ds)

            sp_name = _pick_var(ds, keywords=["wind_speed", "windspeed", "ff"], must_have_time=t)
            wd_name = _pick_var(ds, keywords=["wind_from_direction", "winddirection", "dd"], must_have_time=t)
            sp = _to_series(ds[sp_name], t).astype(float)
            wd = _to_series(ds[wd_name], t).astype(float)

            wd = wd.mask(wd.eq(990.0))
            u, v = _wind_components_from_speed_dir(sp, wd)
            u = u.where(sp.gt(0), other=0.0)
            v = v.where(sp.gt(0), other=0.0)
            units.setdefault("u", "m s-1")
            units.setdefault("v", "m s-1")

            temp_name = _pick_var(ds, keywords=["air_temperature", "temperature", "ta"], must_have_time=t)
            temp_raw = _to_series(ds[temp_name], t)
            temp_units = str(ds[temp_name].attrs.get("units", "")).strip()
            temp_c = _convert_temperature_to_celsius(temp_raw, temp_units or None)
            units.setdefault("temperatuur", "graden Celsius")

            rad_name = _pick_var(ds, keywords=["global_radiation", "radiation", "shortwave", "solar", "qg"], must_have_time=t)
            rad_raw = _to_series(ds[rad_name], t)
            rad_units = str(ds[rad_name].attrs.get("units", "")).strip() or None

            if expected_interval == "1h":
                rad_j_cm2 = _convert_radiation_to_j_cm2_per_interval(
                    rad_raw, units=rad_units, interval_seconds=3600, is_flux_mean=True
                )
            elif expected_interval == "10min":
                rad_j_cm2 = _convert_radiation_to_j_cm2_per_interval(
                    rad_raw, units=rad_units, interval_seconds=600, is_flux_mean=True
                )
            else:
                raise ValueError(expected_interval)
            units.setdefault("straling", "J/cm2")

            part = pd.DataFrame(
                {
                    "temperatuur": temp_c.astype(float),
                    "u": u.astype(float),
                    "v": v.astype(float),
                    "straling_j_cm2_interval": rad_j_cm2.astype(float),
                }
            )
            series_parts.append(part)
        finally:
            ds.close()

    if not series_parts:
        return pd.DataFrame(), units

    df = pd.concat(series_parts).sort_index()
    # De-duplicate exact duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]
    return df, units


def _agg_10min_to_hour(df_10m: pd.DataFrame, *, min_steps_per_hour: int = 5) -> pd.DataFrame:
    if df_10m.empty:
        return df_10m
    # Endtime convention: bin to hourly endtime by ceil('h')
    hour_end = df_10m.index.to_series().dt.ceil("h")
    grouped = df_10m.groupby(hour_end)

    # Means for meteo, sums for interval energy
    means = grouped[["temperatuur", "u", "v"]].mean()
    sums = grouped[["straling_j_cm2_interval"]].sum(min_count=min_steps_per_hour)
    counts = grouped.count()

    ok = counts[["temperatuur", "u", "v", "straling_j_cm2_interval"]] >= min_steps_per_hour
    out = pd.concat([means, sums], axis=1)
    out = out.where(ok)
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    out.index.name = "date"
    out = out.rename(columns={"straling_j_cm2_interval": "straling"})
    return out.sort_index()


def load_knmi_station_cabauw_hourly(*, days: int = 365, station_name_contains: str = "cabauw") -> pd.DataFrame:
    """
    Load deterministic historical hourly meteo from KNMI for Cabauw.

    Returns a DataFrame indexed by UTC hour endtime with columns:
    - temperatuur (Celsius)
    - u, v (m/s)
    - straling (J/cm2 per hour)
    """
    end_utc = _end_utc_from_env_or_now()
    end_utc = end_utc + timedelta(hours=HISTORICAL_FETCH_END_OFFSET_HOURS)
    start_utc = end_utc - timedelta(days=days)
    start_ts = pd.Timestamp(start_utc).tz_convert("UTC")
    end_ts = pd.Timestamp(end_utc).tz_convert("UTC")

    nh_root = Path(__file__).resolve().parents[3]
    tmp_dir = nh_root / "data_ens" / "_tmp_knmi_station"
    tmp_10m = tmp_dir / "tenmin"
    tmp_10m.mkdir(parents=True, exist_ok=True)

    client = KnmiOpenDataClient.from_neural_hydrology_env().with_request_budget(max_requests=100)

    uur_text = _fetch_knmi_uurgegevens_text(start_utc=start_ts, end_utc=end_ts)
    df_hourly = _parse_knmi_uurgegevens_text(uur_text)
    if df_hourly.empty:
        return pd.DataFrame()

    last_endtime_utc = df_hourly.index.max().to_pydatetime()
    start_recent = pd.Timestamp(last_endtime_utc).tz_convert("UTC")
    start_recent_excl = start_recent + pd.Timedelta(seconds=1)
    end_recent = pd.Timestamp(end_utc - timedelta(minutes=10)).tz_convert("UTC")

    tenmin_paths: list[Path] = []
    if start_recent_excl <= end_recent:
        for n in _iter_filenames_old_to_new(client, dataset=TENMIN_DATASET, version=DATASET_VERSION, page_size=250):
            ts = _parse_endtime_from_filename(n)
            if ts is None:
                continue
            if ts <= start_recent:
                continue
            if ts > end_recent:
                break
            out_path = tmp_10m / n
            if out_path.exists():
                tenmin_paths.append(out_path)
                continue
            try:
                tenmin_paths.append(
                    client.download_file(dataset=TENMIN_DATASET, version=DATASET_VERSION, filename=n, out_dir=tmp_10m)
                )
            except KnmiRequestBudgetExceeded:
                break

    df_10m_raw, _ = _load_station_timeseries_from_files(
        tenmin_paths, station_name_contains=station_name_contains, expected_interval="10min"
    )
    df_recent_hourly = _agg_10min_to_hour(df_10m_raw, min_steps_per_hour=5)

    if df_recent_hourly.empty:
        merged = df_hourly
    else:
        merged = df_hourly.combine_first(df_recent_hourly[["temperatuur", "u", "v", "straling"]]).sort_index()

    return merged[["temperatuur", "u", "v", "straling"]].sort_index()


def import_knmi_station_cabauw(*, days: int = 365, station_name_contains: str = "cabauw") -> None:
    logging.basicConfig(level="INFO")
    LOGGER.setLevel(logging.INFO)
    df = load_knmi_station_cabauw_hourly(days=days, station_name_contains=station_name_contains)
    if df.empty:
        raise RuntimeError("No KNMI hourly data could be loaded for Cabauw.")
    LOGGER.info("Loaded Cabauw hourly meteo: rows=%d cols=%s", df.shape[0], list(df.columns))


if __name__ == "__main__":
    import_knmi_station_cabauw(days=365, station_name_contains="cabauw")

