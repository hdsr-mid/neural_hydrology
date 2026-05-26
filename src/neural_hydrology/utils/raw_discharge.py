import pandas as pd
from typing import Tuple

from pathlib import Path

DISCHARGE_CSV_COL_NAMES = {
    "gemaal": "debiet_x_IB",
    "stuw": "debiet",
    "adcp": "debiet",
}


def find_discharge_file_by_code(folder: Path, code) -> Tuple[Path, str] | Tuple[None, None]:
    folder = Path(folder)

    for file in folder.glob("*.csv"):
        _, _, filename_code, filename_structure_type, _ = file.stem.split("_")
        if code == filename_code:
            return file, filename_structure_type

    return None, None


def read_raw_discharge(
        csv_path: Path,
        variable: str,
):
    """
    Clips on 0 (values below zero are set to 0)
    Converts from m3/s to m3/h
    """
    df = pd.read_csv(
        csv_path,
        sep=",",
        usecols=["datetime", variable],
        parse_dates=["datetime"],
        dayfirst=True  # important for DD/MM/YYYY
    )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df[variable] = df[variable].clip(0)
    df[variable] = (df[variable] * 3600).astype(float)
    df[variable] = df[variable].astype(float)
    return df