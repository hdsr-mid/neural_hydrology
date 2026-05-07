from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neural_hydrology.scripts.preprocessing.meteo.import_radar_mfbs_historical import import_mfbs_historical_last_year
from neural_hydrology.scripts.preprocessing.meteo.import_radar_rtcor_recent import import_rtcor_from
from neural_hydrology.scripts.preprocessing.meteo.knmi_open_data import (
    HISTORICAL_FETCH_END_OFFSET_HOURS,
    load_env_file,
)

LOGGER = logging.getLogger(__name__)


def _merge_hourly(mfbs: pd.DataFrame, rtcor: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two hourly DataFrames (index UTC endtime, columns SHAPE_ID).

    MFBS is treated as leading (gauge-adjusted climatological); RTCOR fills gaps.
    """
    if mfbs.empty:
        return rtcor
    if rtcor.empty:
        return mfbs
    merged = mfbs.combine_first(rtcor)
    merged = merged.sort_index()
    return merged


def load_precip_radar_hourly_by_shape(*, days: int = 365, rtcor_max_downloads: int = 100) -> pd.DataFrame:
    """
    Load deterministic historical hourly precipitation (mm) per SHAPE_ID.

    Returns a DataFrame indexed by UTC hour endtime with columns SHAPE_ID (as strings).
    """
    # Optional deterministic end time (UTC) using ENSEMBLE_STARTTIME (YYYYMMDDHH) from neural_hydrology/.env,
    # then extend by HISTORICAL_FETCH_END_OFFSET_HOURS (HARMONIE 6-hour ensemble window).
    nh_root = Path(__file__).resolve().parents[3]
    env = load_env_file(nh_root / ".env")
    end_utc: datetime | None = None
    est = env.get("ENSEMBLE_STARTTIME")
    if est:
        try:
            end_utc = datetime.strptime(est, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            end_utc = None
    if end_utc is None:
        end_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_utc = end_utc + timedelta(hours=HISTORICAL_FETCH_END_OFFSET_HOURS)

    try:
        mfbs_res = import_mfbs_historical_last_year(days=days, end_utc=end_utc)
        mfbs_df = mfbs_res.hourly_mm_by_shape
        start_recent = mfbs_res.last_endtime_utc
    except RuntimeError as e:
        # MFBS can be missing for a requested window; fall back to RTCOR only.
        LOGGER.warning("MFBS unavailable for requested window (days=%s): %s", days, str(e))
        mfbs_df = pd.DataFrame()
        start_recent = (end_utc - timedelta(days=days)).astimezone(timezone.utc)

    # Fetch RTCOR newer than MFBS end (start_exclusive uses MFBS last timestep when available).
    rtcor_res = import_rtcor_from(
        start_exclusive_utc=start_recent,
        end_inclusive_utc=end_utc,
        max_downloads=rtcor_max_downloads,
    )

    merged = _merge_hourly(mfbs_df, rtcor_res.hourly_mm_by_shape)
    merged.columns = merged.columns.astype(str)
    return merged.sort_index()


def import_precip_radar_hourly(*, days: int = 365) -> None:
    """
    Orchestrator:
    - MFBS yearly zip -> hourly polder mean (mm)
    - RTCOR 5m -> hourly polder mean (sum per hour, mm)
    - Merge to a continuous hourly UTC endtime series
    - Write into data_ens/time_series/<SHAPE_ID>.nc as variable neerslag_radar
      (appends if file exists).
    """
    logging.basicConfig(level="INFO")
    LOGGER.setLevel(logging.INFO)

    merged = load_precip_radar_hourly_by_shape(days=days)
    LOGGER.info(
        "Loaded radar hourly precipitation: gebieden=%d uren=%d",
        merged.shape[1],
        merged.shape[0],
    )


if __name__ == "__main__":
    import_precip_radar_hourly(days=365)

