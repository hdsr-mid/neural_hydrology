from pathlib import Path

import numpy as np
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config
import pandas as pd
import plotly.graph_objects as go
import xarray as xr


def evaluate(
        run_dir: str | Path,
        period: str,
        basin: str,
        time_resolution: str,
        netcdf_output_file: Path
) -> None:
    """
    Evaluate the model for the given run directory and period, and save results to a NetCDF file.
    period must be "train" or "test"

    Arguments
    ---------
        run_dir (str | pathlib.Path): Path to the run directory containing ``config.yml``.
        period (str): Evaluation split; must be ``"train"`` or ``"test"``.
        basin (str): Basin identifier used to select the basin-specific results.
        time_resolution (str): Temporal resolution key ("1h" or "1d") used to select results.
        netcdf_output_file (pathlib.Path): Output path for the resulting NetCDF file.
    """
    run_dir = Path(run_dir)
    config = Config(run_dir / "config.yml")
    model = get_tester(cfg=config, run_dir=run_dir, period=period, init_model=True)
    results = model.evaluate(save_results=True, metrics=config.metrics)
    results_xr_dataset: xr.Dataset = results[basin][time_resolution]["xr"]
    results_xr_dataset = results_xr_dataset. \
        isel(time_step=slice(-24, None)). \
        stack(datetime=['date', 'time_step'])
    results_xr_dataset = results_xr_dataset.reset_index("datetime")
    results_xr_dataset["datetime"] = results_xr_dataset["date"] + pd.to_timedelta(
        results_xr_dataset["time_step"],
        unit="h"
    )
    results_xr_dataset = results_xr_dataset.set_index({"datetime": "datetime"})
    results_xr_dataset = results_xr_dataset.drop_vars(['date', 'time_step'])
    results_xr_dataset.to_netcdf(netcdf_output_file)


def weekly_totals(netcdf_path: Path, area: float):
    ds = xr.open_dataset(netcdf_path)
    df = ds["afvoer"].to_dataframe()
    df = df * area * 3600  # m/s -> m3/h
    df_weekly = df.resample('W').sum()
    return df_weekly


def get_area(basin: str) -> float:
    """Get the area of given `basin` from the static attributes csv"""
    csv_path = Path(__file__).parent.parent / "data" / "attributes" / "polders_data_aangevuld.csv"
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


def qq_plot(df: pd.DataFrame, xcol: str, ycol: str):
    # Determine axis limits (same for x and y)
    min_val = min(df[xcol].min(), df[ycol].min())
    max_val = max(df[xcol].max(), df[ycol].max())

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=df[xcol],
        y=df[ycol],
        mode="markers",
        name="Weektotaal afvoer [m3]",
        marker=dict(size=6, color="blue", opacity=0.7)
    ))

    # Diagonal line: x = y
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="x=y",
        line=dict(color="black", width=2, dash="dot")
    ))

    # Axis settings (equal spacing)
    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",  # makes axes use same scale
            scaleratio=1,
            range=[min_val, max_val],
            title=xcol
        ),
        yaxis=dict(
            range=[min_val, max_val],
            title=ycol
        ),
        # width=700,
        # height=700,
        title="Q/Q plot"
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
    # basin = "AFVG13"
    # data_dir = Path(__file__).parent.parent / "data" / "time_series"
    # area = get_area(basin=basin)
    # input_weekly_totals = weekly_totals(netcdf_path=data_dir / f"{basin}.nc", area=area)
    #
    # run_dir = "C:/Users/leendert.vanwolfswin/Documents/hdsr/development_run_2103_053442/development_run_2103_053442"
    # netcdf_output_dir = Path("C:/Users/leendert.vanwolfswin/Documents/hdsr/development_run_2103_053442/development_run_2103_053442/netcdf/")
    # netcdf_output_dir.mkdir(parents=True, exist_ok=True)
    # netcdf_output = netcdf_output_dir / f"simulation_output_{basin}.nc"
    # evaluate(run_dir=run_dir, period="test", basin=basin, time_resolution="h", netcdf_output_file=netcdf_output_dir / f"simulation_output_{basin}.nc")
    #
    # output_weekly_totals = weekly_totals(netcdf_path=netcdf_output, area=area)

    dummy = dummy_dataset()
    qq_plot(df=dummy, xcol="measured", ycol="simulated")

    print("Klaar")
