from pathlib import Path
import os
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config
import pandas as pd
import xarray as xr

from src.plot import plot_time_series_interactive_single_plot


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
    model = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period=period, init_model=True)
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


if __name__ == "__main__":
    ## Inladen test en traingegevens

    TARGET_VARIABLE_NAME = "Q_pump"

    workdir = Path(f"C:/Users/leendert.vanwolfswin/Documents/slim malen nijkerk putten/lstm_{AREA_NAME.lower()}")
    os.chdir(workdir)
    runs_dir = workdir / "runs"
    graphs_dir = workdir / "graphs"
    run_dir = runs_dir / "groene_gemalen_1612_143408_PRUNED"  # Putten met grondwater
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    netcdf_output_file = run_dir / "results" / f"{TARGET_VARIABLE_NAME}_train_test.nc"
    config = Config(run_dir / "config.yml")

    basin = GEMAAL_CODE
    time_resolution = "1h"

    for period in ["train", "test"]:

        netcdf_output_file = run_dir / "results" / f"{TARGET_VARIABLE_NAME}_{period}.nc"
        if not netcdf_output_file.is_file():
            evaluate(run_dir=run_dir,
                     period=period,
                     basin=basin,
                     time_resolution=time_resolution,
                     netcdf_output_file=netcdf_output_file,
                     )

        plot_time_series_interactive_single_plot(
            file_path=netcdf_output_file,
            variable=[
                f"{TARGET_VARIABLE_NAME}_obs",
                f"{TARGET_VARIABLE_NAME}_sim",
            ],
            datetime_coord="datetime",
            title=f"{period.capitalize()} period results",
            transformation=lambda x: x ** 2 * 3600,  # undo the square root and convert m3/s to m3
            reference_variable=f"{TARGET_VARIABLE_NAME}_obs",
            compare_variable=f"{TARGET_VARIABLE_NAME}_sim",
            html_output_path=graphs_dir / f"{AREA_NAME}_{run_dir.name}_{TARGET_VARIABLE_NAME}_{period}_results.html",
        )
