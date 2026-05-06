from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import urllib.error

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neural_hydrology.scripts.preprocessing.meteo.knmi_open_data import KnmiOpenDataClient, KnmiRequestBudgetExceeded
from neural_hydrology.scripts.preprocessing.meteo.radar_hdf5 import (
    read_knmi_radar_h5,
    read_polders,
    zonal_mean_precip_mm,
)

LOGGER = logging.getLogger(__name__)


RTCOR_DATASET = "nl_rdr_data_rtcor_5m"
RTCOR_VERSION = "1.0"


@dataclass(frozen=True)
class RtcorHourlyResult:
    hourly_mm_by_shape: pd.DataFrame  # index=UTC hour endtime, cols=SHAPE_ID
    source_h5_count: int


def _hour_end_utc(ts_utc: pd.Timestamp) -> pd.Timestamp:
    # Endtime convention: files are labeled by end of their accumulation interval.
    # For 5-min steps we want hourly endtime, so we bin by ceil('h').
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    return ts_utc.ceil("h")


def import_rtcor_from(
    *,
    start_exclusive_utc: datetime,
    min_steps_per_hour: int = 10,
    end_inclusive_utc: datetime | None = None,
    max_downloads: int = 100,
) -> RtcorHourlyResult:
    logging.basicConfig(level="INFO")
    LOGGER.setLevel(logging.INFO)

    start_ts = pd.Timestamp(start_exclusive_utc).tz_convert("UTC")
    if end_inclusive_utc is None:
        now = datetime.now(timezone.utc) - timedelta(minutes=10)
        end_ts = pd.Timestamp(now).tz_convert("UTC")
    else:
        end_ts = pd.Timestamp(end_inclusive_utc).tz_convert("UTC")

    nh_root = Path(__file__).resolve().parents[3]
    data_dir = nh_root / "data_ens"
    tmp_dir = data_dir / "_tmp_radar_rtcor"
    h5_dir = tmp_dir / "h5"
    h5_dir.mkdir(parents=True, exist_ok=True)

    # Budget: keep each run within 100 KNMI Open Data requests.
    client = KnmiOpenDataClient.from_neural_hydrology_env().with_request_budget(max_requests=100)

    # Start downloading from MFBS end (or last written output) forward, without relying on file listings.
    start_5m = start_ts.ceil("5min")
    end_5m = end_ts.floor("5min")
    if end_5m <= start_5m:
        LOGGER.warning("RTCOR: empty 5-min window after rounding: (%s, %s].", start_ts, end_ts)
        return RtcorHourlyResult(hourly_mm_by_shape=pd.DataFrame(), source_h5_count=0)

    expected_times = pd.date_range(start=start_5m, end=end_5m, freq="5min", tz="UTC")
    LOGGER.info(
        "RTCOR: %d expected 5-min slots from %s to %s "
        "(oldest-first; skip existing locals; max_downloads=%d)",
        len(expected_times),
        start_5m,
        end_5m,
        max_downloads,
    )

    paths: list[Path] = []
    missing = 0
    from_cache = 0
    downloads_ok = 0
    skipped_no_budget = 0
    budget_notice_logged = False
    # Oldest first: reuse locals; attempt download only for gaps, at most max_downloads times.
    for i, t in enumerate(expected_times, start=1):
        fname = f"RAD_NL25_RAC_RT_{t.strftime('%Y%m%d%H%M')}.h5"
        local_path = h5_dir / fname
        if local_path.is_file():
            paths.append(local_path)
            from_cache += 1
            continue
        if downloads_ok >= max_downloads:
            skipped_no_budget += 1
            continue
        try:
            p = client.download_file(
                dataset=RTCOR_DATASET,
                version=RTCOR_VERSION,
                filename=fname,
                out_dir=h5_dir,
                log_each_download=False,
            )
            paths.append(p)
            downloads_ok += 1
            if downloads_ok == max_downloads and not budget_notice_logged:
                LOGGER.info(
                    "RTCOR: reached max_downloads=%d; continuing scan for local files only (no more API downloads).",
                    max_downloads,
                )
                budget_notice_logged = True
        except KnmiRequestBudgetExceeded as e:
            LOGGER.warning("RTCOR: stopping downloads due to request budget (%s).", str(e))
            break
        except (FileNotFoundError, urllib.error.HTTPError):
            missing += 1
        if i % 300 == 0:
            LOGGER.info(
                "RTCOR: progress %d/%d (paths=%d from_cache=%d downloads=%d missing=%d skipped_no_budget=%d)",
                i,
                len(expected_times),
                len(paths),
                from_cache,
                downloads_ok,
                missing,
                skipped_no_budget,
            )

    LOGGER.info(
        "RTCOR: slot scan done — downloads=%d, from_cache=%d, missing_remote=%d, skipped_after_max_downloads=%d",
        downloads_ok,
        from_cache,
        missing,
        skipped_no_budget,
    )

    if downloads_ok == 0 and from_cache > 0:
        LOGGER.info(
            "RTCOR: stop na 0 downloads — alle benodigde slots waren al lokaal (cached=%d).",
            from_cache,
        )

    if not paths:
        LOGGER.warning("RTCOR: no 5-min files downloaded in window (%s, %s].", start_ts, end_ts)
        return RtcorHourlyResult(hourly_mm_by_shape=pd.DataFrame(), source_h5_count=0)

    polders = read_polders()

    per_step: list[pd.Series] = []
    per_step_time: list[pd.Timestamp] = []
    for p in sorted(paths):
        grid = read_knmi_radar_h5(p)
        ts = pd.Timestamp(grid.timestamp_utc)
        if ts <= start_ts or ts > end_ts:
            continue
        s = zonal_mean_precip_mm(grid, polders)
        per_step.append(s)
        per_step_time.append(ts)

    if not per_step:
        return RtcorHourlyResult(hourly_mm_by_shape=pd.DataFrame(), source_h5_count=len(paths))

    df_5m = pd.DataFrame(per_step, index=pd.DatetimeIndex(per_step_time, tz="UTC")).sort_index()
    df_5m.index.name = "date_5m_end"

    hour_end = df_5m.index.to_series().apply(_hour_end_utc).astype("datetime64[ns, UTC]")
    grouped = df_5m.groupby(hour_end)

    sums = grouped.sum(min_count=min_steps_per_hour)
    counts = grouped.count()
    # Require at least min_steps_per_hour for each SHAPE_ID; otherwise NaN.
    ok = counts >= min_steps_per_hour
    hourly = sums.where(ok)
    hourly.index.name = "date"

    return RtcorHourlyResult(hourly_mm_by_shape=hourly, source_h5_count=len(paths))


if __name__ == "__main__":
    # Example: last 7 days
    res = import_rtcor_from(start_exclusive_utc=datetime.now(timezone.utc) - timedelta(days=7))

