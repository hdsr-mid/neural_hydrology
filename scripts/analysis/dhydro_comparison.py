from pathlib import Path

from typing import List

import pandas as pd

from src.neural_hydrology.utils.raw_discharge import find_discharge_file_by_code, read_raw_discharge
from src.neural_hydrology.viz.comparison_dashboard import afvg_comparison_dashboard

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def afvg_ids_to_gis_object_ids(afvg_ids: List[str]) -> List[str]:
    """Map AFGV ids (as used in the LSTM model) to GIS object ids (as used in the DHYDRO model).

    Identifiers that are not found in the translation table are returned as
    ``None`` in the corresponding position.

    Args:
    afvg_ids: A list of AFGV identifiers.

    Returns:
    A list of GIS object identifiers in the same order as the input.
    """
    df = pd.read_excel(DATA_DIR / "kunstwerkcodes_fewswis_dhydro_vertaaltabel.xlsx")
    mapping = dict(zip(df["AFVG_id"], df["GIS_object_id"]))

    result = []
    for afvg_id in afvg_ids:
        result.append(mapping.get(afvg_id))
    return result


def cumulative(df: pd.DataFrame, col_name: str):
    """Calculate the cumulative volume from a flow-rate time series.

    The values are integrated over time using the interval between consecutive
    timestamps in the DataFrame index.

    Args:
    df: DataFrame containing the time series.
    col_name: Name of the column to integrate.

    Returns:
    A DataFrame containing the cumulative volume for the specified column.
    """
    dt = df.index.to_series().diff().dt.total_seconds().fillna(0)

    return pd.DataFrame({
        col_name: (df[col_name] * dt).cumsum()
    })


def comparison_dashboard_with_io(
        dhydro_results_path: Path,
        lstm_results_path: Path,
        afvg_ids: List[str]
    ):
    """Compare measured pump discharge with DHYDRO and LSTM simulations.

    For each AFGV, measured discharge data, DHYDRO results, and LSTM
    predictions are processed into raw, cumulative, and weekly discharge
    series. The results are visualized in an interactive Plotly dashboard
    that includes performance metrics and uncertainty bands for the LSTM
    predictions.

    Args:
    dhydro_results_path: Directory containing DHYDRO discharge result files.
    lstm_results_path: Path to the CSV file containing LSTM predictions.
    afvg_ids: A list of AFGV identifiers to include in the comparison.

    Returns:
    None. The comparison dashboard is displayed interactively.
    """
    gis_object_ids = afvg_ids_to_gis_object_ids(afvg_ids)
    plot_data = dict()
    lstm_results = pd.read_csv(
        lstm_results_path,
        parse_dates=["datetime"]
    )

    for i in range(len(afvg_ids)):
        afvg_id = afvg_ids[i]
        gis_object_id = gis_object_ids[i]

        # DHYDRO
        try:
            dhydro_discharge = pd.read_csv(dhydro_results_path / f"hdsr_ge_{gis_object_id}.txt", parse_dates=["time"])
        except FileNotFoundError:
            print(f"DHYDRO discharge not found for afvg_id {afvg_id}!")
            continue
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

        # LSTM results
        lstm_discharge = (
            lstm_results[lstm_results["afvg_id"] == afvg_id]
            .set_index("datetime")[["sim_min", "sim_med", "sim_max"]]
        ) * 3600  # convert from m3/s to m3 per time step (hour)
        lstm_discharge_subset = lstm_discharge.loc[
                                dhydro_discharge.index.min():dhydro_discharge.index.max()
                                ]

        # Add all to plot_data
        plot_data[afvg_id] = {
            "raw": {
                "DHYDRO": dhydro_discharge,
                "Measured": raw_discharge_subset,
                "LSTM": {
                    "median": lstm_discharge_subset[["sim_med"]],
                    "lower": lstm_discharge_subset[["sim_min"]],
                    "upper": lstm_discharge_subset[["sim_max"]],
                },
            },
            "cumulative": {
                "DHYDRO": cumulative(
                    dhydro_discharge,
                    "pump_structure_discharge",
                ),
                "Measured": cumulative(
                    raw_discharge_subset,
                    "debiet_x_IB",
                ),
                "LSTM": {
                    "median": cumulative(
                        lstm_discharge_subset[["sim_med"]],
                        "sim_med",
                    ),
                    "lower": cumulative(
                        lstm_discharge_subset[["sim_min"]],
                        "sim_min",
                    ),
                    "upper": cumulative(
                        lstm_discharge_subset[["sim_max"]],
                        "sim_max",
                    ),
                },
            },

            "weekly": {
                "DHYDRO": dhydro_discharge.resample("W").sum(),
                "Measured": raw_discharge_subset.resample("W").sum(),
                "LSTM": (
                    lstm_discharge_subset[["sim_med"]]
                    .resample("W")
                    .sum()
                ),
            },
        }

    afvg_comparison_dashboard(
        data=plot_data,
        title="Pump discharge comparison",
    )


if __name__ == "__main__":
    dhydro = Path(r"C:\Users\leendert.vanwolfswin\Documents\hdsr\dhydro_results")
    lstm = Path(r"C:\Users\leendert.vanwolfswin\Documents\hdsr\lstm_results\simulatie_debieten_test_2021-2022.csv")
    comparison_dashboard_with_io(dhydro, lstm, afvg_ids=[
        'AFVG13',
        'AFVG15',
        'AFVG16',
        # 'AFVG18',
        # 'AFVG22',
        # 'AFVG23',
        # 'AFVG24',
        # 'AFVG26',
        # 'AFVG28',
        # 'AFVG29',
        # 'AFVG30',
        # 'AFVG31',
        # 'AFVG33',
        # 'AFVG34',
        # 'AFVG36',
        # 'AFVG37',
        # 'AFVG39',
        # 'AFVG41',
        # 'AFVG42',
        # 'AFVG44',
        # 'AFVG45',
        # 'AFVG46',
        # 'AFVG47',
        # 'AFVG5',
        # 'AFVG50',
        # 'AFVG54',
        # 'AFVG55',
        # 'AFVG56',
        # 'AFVG57',
        # 'AFVG58',
        # 'AFVG6',
        # 'AFVG62',
        # 'AFVG64',
        # 'AFVG65',
        # 'AFVG67',
        # 'AFVG70',
        # 'AFVG77',
        # 'AFVG78',
        # 'AFVG8',
    ]
)
