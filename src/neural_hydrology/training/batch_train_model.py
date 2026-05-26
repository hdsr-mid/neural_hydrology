import os

from pathlib import Path
import shutil
import yaml
import torch
from neural_hydrology.paths import get_env, get_path, load_env
from neural_hydrology.utils.training import (
    get_run_folder_by_name_timestamp,
    load_validated_tensorboard_scalars,
    log_tensorboard_metrics_to_mlflow,
    run_neural_hydrology_model,
)
from neural_hydrology.utils.results import evaluate, to_netcdf
from neuralhydrology.nh_run import eval_run
import warnings
warnings.filterwarnings("ignore", message="'H' is deprecated and will be removed in a future version")
import mlflow
import numpy as np
import xarray as xr
import datetime

load_env()
mlflow_uri = get_env("MLFLOW_TRACKING_URI", "databricks")
if mlflow_uri:
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

EXPERIMENT_NAME = "runs" # "LSTM_wonderful_williamson_20260407_124224"
TRIAL_NAME = "trial_28"
PATH_HPO = get_path("HPO_OUTPUT_DIR") / EXPERIMENT_NAME
NUMBER_OF_RETRAININGS = 4

RETRAIN_BASE_DIR = get_path("RETRAIN_BASE_DIR")
DESTINATION_DIR = RETRAIN_BASE_DIR / f"{EXPERIMENT_NAME}_{TRIAL_NAME}"
COPIED_TRIAL_DIR = DESTINATION_DIR / TRIAL_NAME
EVAL_OUTPUT_DIR = DESTINATION_DIR / "eval_results"
MLFLOW_EXPERIMENT_NAME = f"/Shared/{EXPERIMENT_NAME}_{TRIAL_NAME}_retrain"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def resolve_source_run_dir(source_trial_dir: Path) -> Path:
    """Resolve the actual NeuralHydrology run directory within a trial folder.

    A trial folder may contain the config.yml directly, or it may contain a
    single subfolder (created by NeuralHydrology with a timestamp suffix) that
    holds the config.yml. This function handles both cases.

    Parameters
    ----------
    source_trial_dir : Path
        Path to the trial directory to inspect. This is typically
        PATH_HPO / TRIAL_NAME.

    Returns
    -------
    Path
        The directory that contains config.yml (either source_trial_dir itself
        or its single subfolder).

    Raises
    ------
    RuntimeError
        If the directory doesn't exist, is not a directory, contains no
        config.yml, or contains multiple subfolders with config.yml.
    """
    if not source_trial_dir.exists():
        raise RuntimeError(f"Source trial folder does not exist: {source_trial_dir}")
    if not source_trial_dir.is_dir():
        raise RuntimeError(f"Source trial path is not a directory: {source_trial_dir}")

    if (source_trial_dir / "config.yml").exists():
        return source_trial_dir

    subfolders_with_config = [
        f for f in source_trial_dir.iterdir()
        if f.is_dir() and (f / "config.yml").exists()
    ]
    if not subfolders_with_config:
        raise RuntimeError(
            f"No config.yml found in source trial folder or its direct subfolders: {source_trial_dir}"
        )
    if len(subfolders_with_config) > 1:
        raise RuntimeError(
            f"Multiple subfolders with config.yml found in {source_trial_dir}: "
            f"{[f.name for f in subfolders_with_config]}"
        )

    return subfolders_with_config[0]


def copy_trial_folder(source_run_dir: Path, destination_dir: Path) -> Path:
    """Copy an entire trial folder to a new destination.

    Creates the parent directories if needed. Fails if the destination already
    exists to prevent accidental overwrites of previous retrain runs.

    Parameters
    ----------
    source_run_dir : Path
        The source trial/run directory to copy.
    destination_dir : Path
        The target path where the folder will be copied to.

    Returns
    -------
    Path
        The destination directory path (same as input destination_dir).

    Raises
    ------
    RuntimeError
        If destination_dir already exists.
    """
    if destination_dir.exists():
        raise RuntimeError(
            f"Destination folder already exists: {destination_dir}. "
            "Remove it first or change EXPERIMENT_NAME/TRIAL_NAME."
        )

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, destination_dir)
    return destination_dir


def prepare_retrain_config(base_config_path: Path, retrain_dir: Path, i_retrain: int):
    """Create a modified NeuralHydrology config for a retrain run.

    Loads the base config, changes the experiment name, run directory, and seed
    to create a unique retrain configuration. Removes stale path keys
    (img_log_dir, train_dir) that would point to the original run's directories.

    Parameters
    ----------
    base_config_path : Path
        Path to the original config.yml to use as template.
    retrain_dir : Path
        Directory where the retrain run output will be stored.
    i_retrain : int
        Zero-based retrain index (0 for first retrain, 1 for second, etc.).

    Returns
    -------
    tuple[Path, str, int]
        - retrain_config_path: Path to the saved retrain config YAML file.
        - experiment_name: The new experiment name (e.g. 'trial_28_retrain_1').
        - seed: The modified random seed for this retrain.
    """
    with open(base_config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    seed = i_retrain + 1
    experiment_name = f"{TRIAL_NAME}_retrain_{i_retrain + 1}"
    config["experiment_name"] = experiment_name
    config["run_dir"] = str(retrain_dir)
    config["seed"] = config["seed"] + seed

    for stale_path_key in ["img_log_dir", "train_dir"]:
        config.pop(stale_path_key, None)

    retrain_dir.mkdir(parents=True, exist_ok=True)
    retrain_config_path = retrain_dir / f"config_retrain_{i_retrain + 1}.yml"
    with open(retrain_config_path, "w") as file:
        yaml.dump(config, file)

    return retrain_config_path, experiment_name, seed


def get_basins_from_config(config_path: Path) -> list[str]:
    """Read the list of test basins from a NeuralHydrology config file.

    Reads the 'test_basin_file' field from the config, resolves the path
    (relative to the config's directory if not absolute), and returns all
    non-empty lines as basin identifiers.

    Parameters
    ----------
    config_path : Path
        Path to the NeuralHydrology config.yml file.

    Returns
    -------
    list[str]
        List of basin identifiers (e.g. ['AFVG1', 'AFVG2', ...]).
    """
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    basin_file = Path(config["test_basin_file"])
    if not basin_file.is_absolute():
        basin_file = config_path.parent / basin_file
    return [line.strip() for line in basin_file.read_text().splitlines() if line.strip()]


def evaluate_and_save(run_dir: Path, config_path: Path, best_epoch: int, run_label: str):
    """Evaluate the model on train, validation, and test sets and save NetCDF results.

    Runs the NeuralHydrology evaluation for all basins at the specified epoch
    for each period (train, validation, test), then writes per-basin NetCDF
    files for both time resolutions (1h and 1D) to
    EVAL_OUTPUT_DIR / run_label / period /.

    Parameters
    ----------
    run_dir : Path
        The NeuralHydrology run directory containing the trained model and
        config.yml.
    config_path : Path
        Path to the config file (used to determine which basins to evaluate).
    best_epoch : int
        The 1-indexed epoch number to evaluate (typically the epoch with the
        highest validation NSE).
    run_label : str
        Label for this run's output subfolder (e.g. 'original',
        'trial_28_retrain_1').
    """
    basins = get_basins_from_config(config_path)

    for period in ("train", "validation", "test"):
        output_dir = EVAL_OUTPUT_DIR / run_label / period
        output_dir.mkdir(parents=True, exist_ok=True)

        results = evaluate(
            run_dir=run_dir,
            period=period,
            basins=basins,
            epoch=best_epoch,
        )

        for basin in basins:
            for time_resolution in ("1h", "1D"):
                netcdf_path = output_dir / f"{basin}_{time_resolution}.nc"
                to_netcdf(
                    results_dict=results,
                    basin=basin,
                    time_resolution=time_resolution,
                    netcdf_output_file=netcdf_path,
                )

        print(f"Evaluation results ({period}) saved to {output_dir}")


def compute_median_ensemble(run_labels: list[str], basins: list[str]):
    """Compute the median ensemble prediction from individual model NetCDF outputs.

    For each basin and time resolution (1h, 1D), loads the prediction NetCDFs
    from all run labels, concatenates them along a new 'model' dimension, and
    computes the element-wise median across models. The median prediction is
    saved as a new NetCDF file. NSE is computed by comparing the median
    prediction against the observations.

    Parameters
    ----------
    run_labels : list[str]
        List of run label strings corresponding to subdirectories in
        EVAL_OUTPUT_DIR (e.g. ['original', 'trial_28_retrain_1', ...]).
    basins : list[str]
        List of basin identifiers to process.

    Returns
    -------
    dict[tuple[str, str], float]
        Dictionary mapping (basin, time_resolution) to the NSE value of the
        median ensemble prediction. Values may be np.nan if observations are
        missing or variance is zero.
    """
    ensemble_dir = EVAL_OUTPUT_DIR / "median_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    nse_results = {}

    for time_resolution in ("1h", "1D"):
        for basin in basins:
            # Load predictions from all models
            datasets = []
            for label in run_labels:
                nc_path = EVAL_OUTPUT_DIR / label / "test" / f"{basin}_{time_resolution}.nc"
                if nc_path.exists():
                    datasets.append(xr.open_dataset(nc_path))

            if not datasets:
                print(f"WARNING: No NetCDF files found for {basin} {time_resolution}")
                continue

            # Stack predictions along a new 'model' dimension and take median
            # Identify the prediction variable (typically ends with '_obs' for obs, rest is prediction)
            pred_vars = [v for v in datasets[0].data_vars if 'obs' not in v.lower()]
            obs_vars = [v for v in datasets[0].data_vars if 'obs' in v.lower()]

            # Concatenate along new model dimension and compute median
            ensemble_ds = xr.concat(datasets, dim="model")
            median_ds = ensemble_ds[pred_vars].median(dim="model")

            # Keep observations from the first dataset
            for obs_var in obs_vars:
                median_ds[obs_var] = datasets[0][obs_var]

            # Save median ensemble NetCDF
            nc_out = ensemble_dir / f"{basin}_{time_resolution}.nc"
            median_ds.to_netcdf(nc_out)

            # Compute NSE: 1 - sum((pred - obs)^2) / sum((obs - mean(obs))^2)
            if pred_vars and obs_vars:
                pred = median_ds[pred_vars[0]].values
                obs = median_ds[obs_vars[0]].values

                # Remove NaN pairs
                mask = ~(np.isnan(pred) | np.isnan(obs))
                pred_clean = pred[mask]
                obs_clean = obs[mask]

                if len(obs_clean) > 0:
                    ss_res = np.sum((pred_clean - obs_clean) ** 2)
                    ss_tot = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
                    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                else:
                    nse = np.nan

                nse_results[(basin, time_resolution)] = nse

            # Close datasets
            for ds in datasets:
                ds.close()

    print(f"Median ensemble saved to {ensemble_dir}")
    return nse_results


def main():
    """Orchestrate the batch retraining and ensemble evaluation pipeline.

    Performs the following steps:
    1. Copies the source HPO trial folder to a new destination directory.
    2. Logs validation metrics of the original model to MLflow.
    3. Evaluates the original model on the test set and saves NetCDF results.
    4. Retrains the model NUMBER_OF_RETRAININGS times with different random
       seeds, logging metrics and saving evaluation results for each.
    5. Computes the median ensemble prediction across all 5 models (original +
       4 retrains) and logs the ensemble NSE to MLflow.
    """
    source_trial_dir = PATH_HPO / TRIAL_NAME
    copied_trial_dir = copy_trial_folder(source_trial_dir, COPIED_TRIAL_DIR)
    copied_run_dir = resolve_source_run_dir(copied_trial_dir)
    copied_config_path = copied_run_dir / "config.yml"

    if not copied_config_path.exists():
        raise RuntimeError(f"Copied config.yml not found: {copied_config_path}")

    with open(copied_config_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    print(f"Source trial folder: {source_trial_dir}")
    print(f"Copied trial folder: {copied_trial_dir}")
    print(f"Resolved copied run folder: {copied_run_dir}")
    print(f"Destination folder: {DESTINATION_DIR}")
    print(f"Copied config path: {copied_config_path}")
    print(f"Copied config experiment_name: {config_dict.get('experiment_name')}")
    print(f"Copied config model: {config_dict.get('model')}")

    del config_dict

    with mlflow.start_run(run_name=DESTINATION_DIR.name) as parent_run:
        mlflow.log_params(
            {
                "source_experiment_name": EXPERIMENT_NAME,
                "source_trial_name": TRIAL_NAME,
                "number_of_retrainings": NUMBER_OF_RETRAININGS,
            }
        )

        with mlflow.start_run(run_name=TRIAL_NAME, nested=True):
            mlflow.set_tag("run_type", "copied_original")
            mlflow.log_artifact(str(copied_config_path), artifact_path="config")
            data = load_validated_tensorboard_scalars(copied_run_dir)
            _, best_epoch = log_tensorboard_metrics_to_mlflow(data, run_folder=copied_run_dir)
            evaluate_and_save(copied_run_dir, copied_config_path, best_epoch, "original")

        for i_retrain in range(NUMBER_OF_RETRAININGS):
            retrain_dir = DESTINATION_DIR / f"retrain_{i_retrain + 1}"
            config_path, experiment_name, seed = prepare_retrain_config(
                base_config_path=copied_config_path,
                retrain_dir=retrain_dir,
                i_retrain=i_retrain,
            )

            with mlflow.start_run(
                run_name=experiment_name,
                nested=True,
            ):
                mlflow.set_tag("run_type", "retrain")
                mlflow.log_params(
                    {
                        "retrain_index": i_retrain + 1,
                        "experiment_name": experiment_name,
                        "seed": seed,
                    }
                )
                mlflow.log_artifact(str(config_path), artifact_path="config")

                gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
                device_mode = "GPU" if gpu_available else "CPU"
                mlflow.set_tag("device_mode", device_mode)

                run_neural_hydrology_model(config_path)

                run_folder = get_run_folder_by_name_timestamp(
                    trial_dir=retrain_dir,
                    experiment_name=experiment_name,
                )
                print(f"Selected run folder: {Path(run_folder).name}")

                data = load_validated_tensorboard_scalars(run_folder)
                _, best_epoch = log_tensorboard_metrics_to_mlflow(data, run_folder=run_folder)
                mlflow.log_artifact(str(config_path), artifact_path="config")
                evaluate_and_save(run_folder, config_path, best_epoch, experiment_name)

        # Compute median ensemble from all 5 models
        run_labels = ["original"] + [
            f"{TRIAL_NAME}_retrain_{i + 1}" for i in range(NUMBER_OF_RETRAININGS)
        ]
        basins = get_basins_from_config(copied_config_path)
        nse_results = compute_median_ensemble(run_labels, basins)

        # Log ensemble NSE to MLflow
        for (basin, resolution), nse in nse_results.items():
            mlflow.log_metric(
                f"ensemble_nse_{basin}_{resolution}", float(nse)
            )

        # Log mean ensemble NSE
        for resolution in ("1h", "1D"):
            nse_values = [
                v for (b, r), v in nse_results.items()
                if r == resolution and not np.isnan(v)
            ]
            if nse_values:
                mlflow.log_metric(
                    f"ensemble_mean_nse_{resolution}", float(np.mean(nse_values))
                )
                mlflow.log_metric(
                    f"ensemble_median_nse_{resolution}", float(np.median(nse_values))
                )


if __name__ == "__main__":
    main()
