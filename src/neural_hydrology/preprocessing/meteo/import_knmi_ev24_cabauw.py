from __future__ import annotations

import logging
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from neural_hydrology.paths import load_env
from neural_hydrology.preprocessing.meteo.knmi_open_data import HISTORICAL_FETCH_END_OFFSET_HOURS

LOGGER = logging.getLogger(__name__)

DAGGEGEVENS_BASE_URL = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
CABAUW_STN = 348
EV24_VARS = "EV24"

# Unit chain: KNMI EV24 is 0.1 mm/day → /10 → mm/day → /24 → mm/hour (matches training NetCDF).
EV24_TO_MM_PER_DAY = 10.0
HOURS_PER_DAY = 24


def _now_utc_hour_end() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _end_utc_from_env_or_now() -> datetime:
    """
    Endtime (UTC) for deterministic historical fetch.

    Uses ENSEMBLE_STARTTIME (YYYYMMDDHH) from neural_hydrology/.env when set;
    otherwise current UTC hour end.
    """
    env = load_env()
    est = env.get("ENSEMBLE_STARTTIME")
    if est:
        try:
            return datetime.strptime(est, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return _now_utc_hour_end()


def _fetch_knmi_ev24_text(*, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> str:
    start_d = pd.Timestamp(start_utc).tz_convert("UTC").strftime("%Y%m%d")
    end_d = pd.Timestamp(end_utc).tz_convert("UTC").strftime("%Y%m%d")
    query = {
        "stns": str(CABAUW_STN),
        "vars": EV24_VARS,
        "start": start_d,
        "end": end_d,
    }
    url = f"{DAGGEGEVENS_BASE_URL}?{urllib.parse.urlencode(query)}"
    with urllib.request.urlopen(url, timeout=120) as resp:  # nosec - fixed KNMI endpoint + encoded query
        return resp.read().decode("utf-8", errors="replace")


def _parse_knmi_ev24_text(text: str) -> pd.Series:
    """
    Parse KNMI daggegevens CSV for Cabauw EV24.

    Returns daily Series (UTC midnight index) in mm/day (raw EV24 / 10).
    """
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        if parts[0].upper() == "STN":
            continue
        try:
            stn = int(parts[0])
            if stn != CABAUW_STN:
                continue
            ymd = parts[1]
            ev24 = float(parts[2])
        except Exception:
            continue
        day = pd.to_datetime(ymd, format="%Y%m%d", utc=True).normalize()
        rows.append({"date": day, "ev24_tenths_mm": ev24})

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).set_index("date").sort_index()
    mm_day = pd.to_numeric(df["ev24_tenths_mm"], errors="coerce").astype(float) / EV24_TO_MM_PER_DAY
    mm_day.name = "verdamping_mm_day"
    return mm_day


def _daily_mm_to_hourly_mm(daily_mm: pd.Series) -> pd.Series:
    """
    Expand daily mm to hourly mm using KNMI day convention (hours 1..24).

    For calendar day D, assign daily_mm/24 to timestamps D 01:00 .. (D+1) 00:00 UTC.
    Matches the flat-within-day pattern in training NetCDFs.
    """
    if daily_mm.empty:
        return pd.Series(dtype=float)

    timestamps: list[pd.Timestamp] = []
    values: list[float] = []
    for day, val in daily_mm.items():
        day_utc = pd.Timestamp(day).tz_convert("UTC").normalize()
        if pd.isna(val):
            continue
        hourly = float(val) / float(HOURS_PER_DAY)
        for h in range(1, HOURS_PER_DAY + 1):
            timestamps.append(day_utc + pd.Timedelta(hours=h))
            values.append(hourly)

    if not timestamps:
        return pd.Series(dtype=float)

    out = pd.Series(values, index=pd.DatetimeIndex(timestamps, tz="UTC"), dtype=np.float64)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.name = "verdamping"
    out.index.name = "date"
    return out


def load_knmi_ev24_cabauw_hourly(*, days: int = 365) -> pd.Series:
    """
    Load Cabauw Makkink reference evaporation (EV24) as an hourly series.

    Unit chain:
      EV24 (0.1 mm/day) → /10 → mm/day → /24 → mm/hour

    Window matches Cabauw hourly meteo: ENSEMBLE_STARTTIME (or now) plus
    HISTORICAL_FETCH_END_OFFSET_HOURS, looking back ``days``.

    Returns
    -------
    pd.Series
        ``verdamping`` in mm per hour, UTC-aware hourly index (hour ends 01:00..00:00
        per KNMI day). Empty Series if the KNMI response has no usable rows.
    """
    end_utc = _end_utc_from_env_or_now()
    end_utc = end_utc + timedelta(hours=HISTORICAL_FETCH_END_OFFSET_HOURS)
    start_utc = end_utc - timedelta(days=days)
    start_ts = pd.Timestamp(start_utc).tz_convert("UTC")
    end_ts = pd.Timestamp(end_utc).tz_convert("UTC")

    LOGGER.info(
        "Loading KNMI EV24 Cabauw (stn=%s) from %s to %s (days=%s)...",
        CABAUW_STN,
        start_ts.strftime("%Y-%m-%d"),
        end_ts.strftime("%Y-%m-%d"),
        days,
    )
    text = _fetch_knmi_ev24_text(start_utc=start_ts, end_utc=end_ts)
    daily_mm = _parse_knmi_ev24_text(text)
    if daily_mm.empty:
        LOGGER.warning("KNMI EV24 Cabauw: no daily rows parsed.")
        return pd.Series(dtype=float, name="verdamping")

    n_nan = int(daily_mm.isna().sum())
    if n_nan:
        LOGGER.warning("KNMI EV24 Cabauw: %d daily NaN value(s) after parse.", n_nan)

    hourly = _daily_mm_to_hourly_mm(daily_mm)
    LOGGER.info(
        "KNMI EV24 Cabauw: %d daily rows → %d hourly values (mm/h), mean=%.4f max=%.4f.",
        len(daily_mm),
        len(hourly),
        float(hourly.mean()) if len(hourly) else float("nan"),
        float(hourly.max()) if len(hourly) else float("nan"),
    )
    return hourly
