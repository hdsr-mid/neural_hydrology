from pathlib import Path

from typing import List

import pandas as pd

from src.neural_hydrology.utils.raw_discharge import find_discharge_file_by_code, read_raw_discharge
from src.neural_hydrology.viz.time_series import plot_time_series, plot_time_series_by_area

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def afvg_ids_to_gis_object_ids(afvg_ids: List[str]) -> List[str]:
    df = pd.read_excel(DATA_DIR / "kunstwerkcodes_fewswis_dhydro_vertaaltabel.xlsx")
    mapping = dict(zip(df["AFVG_id"], df["GIS_object_id"]))

    result = []
    for afvg_id in afvg_ids:
        result.append(mapping.get(afvg_id))
    return result


def cumulative(df: pd.DataFrame, col_name: str):
    dt = df.index.to_series().diff().dt.total_seconds().fillna(0)

    return pd.DataFrame({
        col_name: (df[col_name] * dt).cumsum()
    })


def dhydro_compare_plot(dhydro_results_path: Path, afvg_ids: List[str]):

    gis_object_ids = afvg_ids_to_gis_object_ids(afvg_ids)
    plot_data = dict()

    for i in range(len(afvg_ids)):
        afvg_id = afvg_ids[i]
        gis_object_id = gis_object_ids[i]

        # DHYDRO
        dhydro_discharge = pd.read_csv(dhydro_results_path / f"hdsr_ge_{gis_object_id}.txt", parse_dates=["time"])
        dhydro_discharge = dhydro_discharge[["time", "pump_structure_discharge"]].set_index("time") * 3600

        # RAW
        raw_discharge_file, _ = find_discharge_file_by_code(
            folder=DATA_DIR / "raw_discharge_data",
            code=afvg_id
        )
        raw_discharge = read_raw_discharge(
            csv_path=raw_discharge_file,
            variable="debiet_x_IB"
        )
        raw_discharge_subset = raw_discharge.loc[dhydro_discharge.index.min():dhydro_discharge.index.max()]
        plot_data[afvg_id] = {
            "DHYDRO Discharge [m3]": cumulative(dhydro_discharge, col_name="pump_structure_discharge"),
            "Measured discharge [m3]": cumulative(raw_discharge_subset, col_name="debiet_x_IB"),
        }

    plot_time_series_by_area(
        data= plot_data,
        title=f"Cumulative pump discharge [m3]",
    )

dhydro = Path(r"C:\Users\leendert.vanwolfswin\Documents\hdsr\dhydro_results")
dhydro_compare_plot(dhydro, afvg_ids=["AFVG50", "AFVG77"])
