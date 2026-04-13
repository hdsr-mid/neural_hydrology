import math
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr

from utils.results import evaluate

DISCHARGE_CSV_COL_NAMES = {
    "gemaal": "debiet_x_IB",
    "stuw": "debiet",
    "adcp": "debiet",
}


def weekly_totals_from_netcdf(
        netcdf_path: Path,
        area: float,
        variable: str,
) -> pd.DataFrame:
    """
    Also converts from m/s to m3/h
    """
    ds = xr.open_dataset(netcdf_path)
    df = ds[variable].to_dataframe()
    df = df * area * 3600  # m/s -> m3/h
    df_weekly = df.resample('W').sum()
    return df_weekly


def weekly_totals_from_csv(
        csv_path: Path,
        variable: str,
):
    """
    Also converts from m3/s to m3/h
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
    df_weekly = df.resample('W').sum()
    return df_weekly


def get_area(basin: str) -> float:
    """Get the area of given `basin` from the static attributes csv"""
    csv_path = Path(__file__).parent.parent.parent / "data" / "attributes" / "polders_data_aangevuld.csv"
    df = pd.read_csv(csv_path)
    row = df.loc[df["SHAPE_ID"] == basin, "oppervlak"]
    if row.empty:
        return None
    return row.iloc[0]


def get_overlap(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df1_old_column_name: str,
        df1_new_column_name: str,
        df2_old_column_name: str,
        df2_new_column_name: str,
) -> pd.DataFrame:
    """Return a DataFrame that has the """
    df1.index = pd.to_datetime(df1["date"])
    df2.index = pd.to_datetime(df2["date"])

    df1_selection = df1[[df1_old_column_name]].rename(columns={df1_old_column_name: df1_new_column_name})
    df2_selection = df2[[df2_old_column_name]].rename(columns={df2_old_column_name: df2_new_column_name})

    overlap = df1_selection.join(df2_selection, how="inner")
    return overlap


def find_discharge_file_by_code(folder: Path, code) -> Tuple[Path, str] | Tuple[None, None]:
    folder = Path(folder)

    for file in folder.glob("*.csv"):
        _, _, filename_code, filename_structure_type, _ = file.stem.split("_")
        if code == filename_code:
            return file, filename_structure_type

    return None, None


def mean_sim_per_measured_quantile(x, y, n_quantiles):
    """
    Compute average simulated discharge per measured quantile.

    Parameters
    ----------
    x : array-like
        Measured discharge (x-axis)
    y : array-like
        Simulated discharge (y-axis)
    n_quantiles : int
        Number of quantile bins

    Returns
    -------
    x_mean : ndarray
        Mean measured value per quantile
    y_mean : ndarray
        Mean simulated value per quantile
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Quantile edges in measured space
    q_edges = np.quantile(x, np.linspace(0, 1, n_quantiles + 1))

    x_mean = []
    y_mean = []

    for lo, hi in zip(q_edges[:-1], q_edges[1:]):
        mask = (x >= lo) & (x <= hi)
        if np.any(mask):
            x_mean.append(x[mask].mean())
            y_mean.append(y[mask].mean())

    return np.array(x_mean), np.array(y_mean)


def qq_plot_subplot(
    fig: go.Figure,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    row: int,
    col: int,
    ncols: int,
):
    # determine axis limits
    min_val = min(df[xcol].min(), df[ycol].min())
    max_val = max(df[xcol].max(), df[ycol].max())

    # scatter points
    fig.add_trace(
        go.Scatter(
            x=df[xcol],
            y=df[ycol],
            mode="markers",
            marker=dict(size=6, color="blue", opacity=0.4),
            showlegend=False
        ),
        row=row,
        col=col
    )

    # x = y line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="black", width=1, dash="dot"),
            showlegend=False
        ),
        row=row,
        col=col
    )

    # quantile-mean line
    x_q, y_q = mean_sim_per_measured_quantile(
        x=df[xcol], y=df[ycol], n_quantiles=10
    )

    fig.add_trace(
        go.Scatter(
            x=x_q,
            y=y_q,
            mode="lines+markers",
            line=dict(color="red", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=row, col=col
    )

    # axis settings
    fig.update_xaxes(
        range=[0, max_val],
        title_text=xcol,
        row=row,
        col=col,
    )

    fig.update_yaxes(
        range=[0, max_val],
        scaleanchor=f"x{(row - 1) * ncols + col}",
        scaleratio=1,
        title_text=ycol,
        row=row,
        col=col,
    )


def qq_plots(
        dataframes: List[pd.DataFrame],
        titles: List[str],
        xcol: str,
        ycol: str,
):

    n_plots = len(dataframes)

    max_cols = 4
    n_cols = min(max_cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
        subplot_titles=titles
    )

    for i, df in enumerate(dataframes):
        row = i // n_cols + 1
        col = i % n_cols + 1

        qq_plot_subplot(
            fig,
            df,
            xcol=xcol,
            ycol=ycol,
            row=row,
            col=col,
            ncols=n_cols,
        )

    fig.update_layout(
        width=600 * n_cols,
        height=600 * n_rows,
        title="Weekly Discharge Q–Q Plots",
        showlegend=False
    )

    fig.show()


def dummy_dataset():
    dates1 = pd.date_range(start="2020-01-01", end="2020-06-01", freq="W")
    df1 = pd.DataFrame({
        "date": dates1,
        "q": np.random.normal(loc=10, scale=2, size=len(dates1))
    })

    # -------------------------------
    # Create example weekly DataFrame 2
    # Overlapping, but different date range
    # -------------------------------
    dates2 = pd.date_range(start="2020-03-01", end="2020-09-01", freq="W")
    df2 = pd.DataFrame({
        "date": dates2,
        "q": np.random.normal(loc=10, scale=2, size=len(dates2))
    })
    merged = get_overlap(
        df1=df1,
        df2=df2,
        df1_old_column_name="q",
        df1_new_column_name="measured",
        df2_old_column_name="q",
        df2_new_column_name="simulated"
    )
    return merged


if __name__ == "__main__":

    run_dir = "C:/Users/leendert.vanwolfswin/Documents/hdsr/runs/runs/development_run_23_2503_122253"
    netcdf_output_dir = Path(run_dir) / "netcdf"
    netcdf_output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).parent.parent.parent / "data"
    raw_discharges_dir = data_dir / "raw_discharge_data"
    basins_file = data_dir / "hdsr_polders.txt"
    with basins_file.open("r") as f:
        lines = f.readlines()
        basins = [line.strip() for line in lines]
    # basins = ["AFVG41"]
    print(basins)

    # NetCDFs maken per polder
    for basin in basins:
        netcdf_output = netcdf_output_dir / f"simulation_output_{basin}.nc"
        evaluate(
            run_dir=run_dir,
            period="test",
            basin=basin,
            time_resolution="1h",
            netcdf_output_file=netcdf_output,
            config_overrides={
                "device": "cpu",
                "data_dir": str(data_dir),
            }
        )

    weekly_totals_dfs = []
    titles = []
    for basin in basins:
        if basin == "AFVG41":
            pass
        area = get_area(basin=basin)
        csv_path, structure_type = find_discharge_file_by_code(folder=raw_discharges_dir, code=basin)
        if csv_path is None:
            warnings.warn(f"CSV voor {basin} niet gevonden!")
            continue
        measured_variable_name = DISCHARGE_CSV_COL_NAMES[structure_type]
        input_weekly_totals = weekly_totals_from_csv(
            csv_path=csv_path,
            variable=measured_variable_name
        )
        output_weekly_totals = weekly_totals_from_netcdf(
            netcdf_path=netcdf_output_dir / f"simulation_output_{basin}.nc",
            area=area,
            variable="afvoer_sim"
        )
        weekly_totals_joined = input_weekly_totals.join(
            output_weekly_totals,
            how="inner"  # inner join keeps only weeks present in both
        )
        weekly_totals_joined.rename(
            columns={
                measured_variable_name: "measured",
                "afvoer_sim": "simulated"
            },
            inplace=True
        )
        if len(weekly_totals_joined) == 0:
            warnings.warn(f"Geen overlappende weken gevonden voor {basin}")
            continue

        weekly_totals_dfs.append(weekly_totals_joined)
        title = f"{basin}: {structure_type} ({round(area/1e6, 2):,} km²)"
        titles.append(title)

    qq_plots(
        dataframes=weekly_totals_dfs,
        xcol="measured",
        ycol="simulated",
        titles=titles
    )

    print("Klaar")
