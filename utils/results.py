import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config
import xarray as xr
import geopandas as gpd

from utils.attributes import read_attributes


def evaluate(
        run_dir: str | Path,
        period: str,
        basins: List[str],
        config_overrides: dict = None,
) -> Dict:
    """
    Evaluate the model for the given run directory and period, and return a results dict.
    period must be "train" or "test"

    Arguments
    ---------
        run_dir (str | pathlib.Path): Path to the run directory containing ``config.yml``.
        period (str): Evaluation split; must be ``"train"`` or ``"test"``.
        basin (str): Basin identifier used to select the basin-specific results.
        time_resolution (str): Temporal resolution key ("1h" or "1D") used to select results.
        netcdf_output_file (pathlib.Path): Output path for the resulting NetCDF file.
    """
    config_overrides = config_overrides or {}
    run_dir = Path(run_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_basin_file:
        temp_basin_file.write("\n".join(basins))
        temp_basin_file.flush()
        temp_basin_file_path = temp_basin_file.name

    config_overrides["test_basin_file"] = temp_basin_file_path
    config_overrides["train_basin_file"] = temp_basin_file_path
    config_overrides["validation_basin_file"] = temp_basin_file_path

    with open(run_dir / "config.yml") as config:
        config_yaml = yaml.safe_load(config)

    for key, value in config_overrides.items():
        config_yaml[key] = value

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp_config:
        yaml.dump(config_yaml, temp_config)
        temp_config.flush()
        temp_config_path = Path(temp_config.name)
        config = Config(temp_config_path)

    model = get_tester(cfg=config, run_dir=run_dir, period=period, init_model=True)
    results = model.evaluate(save_results=True, metrics=config.metrics)
    temp_config_path.unlink()
    Path(temp_basin_file_path).unlink()
    return results


def to_netcdf(
    results_dict: Dict,
    basin: str,
    time_resolution: str,
    netcdf_output_file: Path,
) -> None:
    """
    Arguments
    ---------
        basin: Basin identifier used to select the basin-specific results.
        time_resolution: Temporal resolution key ("1h" or "1D") used to select results.
        netcdf_output_file: Output path for the resulting NetCDF file.
    """
    results_xr_dataset: xr.Dataset = results_dict[basin][time_resolution]["xr"]
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


def get_nse(
        results_dict: Dict,
        basin: str,
        time_resolution: str,
) -> float | None:
    """
    Returns requested NSE value or None if not found

    Arguments
    ---------
        basin: Basin identifier used to select the basin-specific results.
        time_resolution: Temporal resolution key ("1h" or "1D") used to select results.
        netcdf_output_file: Output path for the resulting NetCDF file.
    """
    results_dataset: xr.Dataset = results_dict[basin][time_resolution]
    nse_key = f"NSE_{time_resolution}"
    try:
        nse_value = results_dataset[nse_key]
    except KeyError:
        return np.nan
    return nse_value


def get_nse_values_gdf(
        results_dict: Dict,
        basins: List[str],
) -> gpd.GeoDataFrame:
    nse_1h = {}
    nse_1d = {}
    for basin in basins:
        nse_1h[basin] = get_nse(
            results_dict=results_dict,
            basin=basin,
            time_resolution="1h"
        )
        nse_1d[basin] = get_nse(
            results_dict=results_dict,
            basin=basin,
            time_resolution="1D"
        )

    gdf = read_attributes()[["SHAPE_ID", "geometry"]].copy()
    gdf["nse_1h"] = gdf["SHAPE_ID"].map(nse_1h)
    gdf["nse_1d"] = gdf["SHAPE_ID"].map(nse_1d)
    gdf["nse_avg"] = (gdf["nse_1h"] + gdf["nse_1d"]) / 2
    return gdf
