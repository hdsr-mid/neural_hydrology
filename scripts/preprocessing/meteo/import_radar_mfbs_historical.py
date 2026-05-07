from __future__ import annotations

import logging
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import re

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neural_hydrology.scripts.preprocessing.meteo.knmi_open_data import KnmiOpenDataClient
from neural_hydrology.scripts.preprocessing.meteo.radar_hdf5 import (
    read_knmi_radar_h5,
    read_polders,
    zonal_mean_precip_mm,
)

LOGGER = logging.getLogger(__name__)


MFBS_DATASET = "rad_nl25_rac_mfbs_01h"
MFBS_VERSION = "2.0"


@dataclass(frozen=True)
class MfbsHourlyResult:
    hourly_mm_by_shape: pd.DataFrame  # index=UTC endtime, cols=SHAPE_ID
    last_endtime_utc: datetime
    source_files: list[Path]


def _select_yearly_zips(files: list[dict], start_utc: datetime, end_utc: datetime) -> list[str]:
    """
    MFBS is published as yearly zip archives:
    RADNL_CLIM____MFBSNL25_01H_<start>_<end>_0002.zip
    """
    names = [f.get("filename") for f in files if isinstance(f.get("filename"), str)]
    selected: list[str] = []
    for n in names:
        if not n.endswith(".zip"):
            continue
        m = re.search(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_", n)
        if not m:
            continue
        sdt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        edt = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        if edt <= start_utc or sdt >= end_utc:
            continue
        selected.append(n)
    return sorted(set(selected))


def _extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".h5"):
                continue
            target = out_dir / member
            if target.exists():
                extracted.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted


def import_mfbs_historical_last_year(*, days: int = 365, end_utc: datetime | None = None) -> MfbsHourlyResult:
    logging.basicConfig(level="INFO")
    LOGGER.setLevel(logging.INFO)

    if end_utc is None:
        now = datetime.now(timezone.utc)
        end_utc = now.replace(minute=0, second=0, microsecond=0)
    else:
        end_utc = end_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_utc = end_utc - timedelta(days=days)

    nh_root = Path(__file__).resolve().parents[3]
    data_dir = nh_root / "data_ens"
    tmp_dir = data_dir / "_tmp_radar_mfbs"
    zip_dir = tmp_dir / "zips"
    extract_dir = tmp_dir / "extract"
    zip_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    client = KnmiOpenDataClient.from_neural_hydrology_env()

    # MFBS file listing is small; pull many and select overlaps.
    files = client.list_files(dataset=MFBS_DATASET, version=MFBS_VERSION, max_keys=20, sorting="desc")
    zip_names = _select_yearly_zips(files, start_utc=start_utc, end_utc=end_utc)
    if not zip_names:
        raise RuntimeError("No MFBS zip archives found that overlap requested window.")

    LOGGER.info("MFBS: selected %d zip(s): %s", len(zip_names), zip_names)

    zip_paths: list[Path] = []
    for name in zip_names:
        zip_paths.append(
            client.download_file(dataset=MFBS_DATASET, version=MFBS_VERSION, filename=name, out_dir=zip_dir)
        )

    all_h5: list[Path] = []
    for zp in zip_paths:
        out = extract_dir / zp.stem
        all_h5.extend(_extract_zip(zp, out))

    if not all_h5:
        raise RuntimeError("MFBS: no .h5 files extracted from selected zip archives.")

    polders = read_polders()

    rows: list[pd.Series] = []
    times: list[pd.Timestamp] = []
    for p in sorted(all_h5):
        grid = read_knmi_radar_h5(p)
        t = pd.Timestamp(grid.timestamp_utc)
        if t < pd.Timestamp(start_utc) or t > pd.Timestamp(end_utc):
            continue
        s = zonal_mean_precip_mm(grid, polders)
        rows.append(s)
        times.append(t)

    if not rows:
        raise RuntimeError("MFBS: no hourly fields found within requested time window after parsing.")

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(times, tz="UTC")).sort_index()
    df.index.name = "date"
    last_end = df.index.max().to_pydatetime()

    return MfbsHourlyResult(hourly_mm_by_shape=df, last_endtime_utc=last_end, source_files=zip_paths)


if __name__ == "__main__":
    res = import_mfbs_historical_last_year(days=365)
    print(res.hourly_mm_by_shape.shape, res.last_endtime_utc)

