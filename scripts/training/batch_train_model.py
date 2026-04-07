# this script is meant to take from an experiment


import os
os.environ["MLFLOW_TRACKING_URI"] = "databricks"

from pathlib import Path
import shutil
import yaml
import torch
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import warnings
warnings.filterwarnings("ignore", message="'H' is deprecated and will be removed in a future version")
import os
import optuna
import mlflow
import numpy as np
import datetime

EXPERIMENT_NAME = "runs" # "LSTM_wonderful_williamson_20260407_124224"
TRIAL_NAME = "trial_28"
PATH_HPO = Path(f"/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output/HPO/{EXPERIMENT_NAME}")
RETRAIN_NAME = "example_retrain"
NUMBER_OF_RETRAININGS = 2

RETRAIN_BASE_DIR = Path("/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output/BATCH_RETRAIN")
DESTINATION_DIR = RETRAIN_BASE_DIR / f"{TRIAL_NAME}_{RETRAIN_NAME}"
MLFLOW_EXPERIMENT_NAME = f"/{TRIAL_NAME}_{RETRAIN_NAME}"

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def run_neural_hydrology_model(config_name):
    run_config = Config(Path(config_name))
    print('model:\t\t', run_config.model)
    print('use_frequencies:', run_config.use_frequencies)
    print('seq_length:\t', run_config.seq_length)

    if torch.cuda.is_available():
        start_run(config_file=Path(config_name))
    else:
        start_run(config_file=Path(config_name), gpu=-1)


def extract_tensorboard_scalars(logdir):
    """Extract all TensorBoard scalars, searching subdirectories for event files."""
    scalars = {}

    for root, dirs, files in os.walk(logdir):
        event_files = [f for f in files if f.startswith('events.out.tfevents')]
        if event_files:
            event_acc = EventAccumulator(root)
            event_acc.Reload()
            for tag in event_acc.Tags().get('scalars', []):
                scalars[tag] = [(e.step, e.value) for e in event_acc.Scalars(tag)]

    return scalars


def find_tag(data, pattern):
    """Find a TensorBoard tag matching the pattern (case-insensitive)."""
    pattern_lower = pattern.lower()
    for tag in data.keys():
        if tag.lower() == pattern_lower:
            return tag
    raise KeyError(f"No tag matching '{pattern}' found. Available tags: {list(data.keys())}")


def resolve_source_run_dir(source_trial_dir: Path) -> Path:
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
    if destination_dir.exists():
        raise RuntimeError(
            f"Destination folder already exists: {destination_dir}. "
            "Remove it first or change RETRAIN_NAME/TRIAL_NAME."
        )

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, destination_dir)
    return destination_dir


def load_config(config_path: Path):
    with open(config_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_object = Config(config_path)
    return config_dict, config_object


def prepare_retrain_config(base_config_path: Path, trial_dir: Path, i_retrain: int):
    with open(base_config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    experiment_name = f"{TRIAL_NAME}_retrain_{RETRAIN_NAME}_{i_retrain + 1}"
    config["experiment_name"] = experiment_name
    config["run_dir"] = str(trial_dir)

    for stale_path_key in ["img_log_dir", "train_dir"]:
        config.pop(stale_path_key, None)

    retrain_config_path = trial_dir / f"config_retrain_{i_retrain + 1}.yml"
    with open(retrain_config_path, "w") as file:
        yaml.dump(config, file)

    return retrain_config_path, experiment_name


def main():
    source_trial_dir = PATH_HPO / TRIAL_NAME
    source_run_dir = resolve_source_run_dir(source_trial_dir)

    copied_run_dir = copy_trial_folder(source_run_dir, DESTINATION_DIR)
    copied_config_path = copied_run_dir / "config.yml"

    if not copied_config_path.exists():
        raise RuntimeError(f"Copied config.yml not found: {copied_config_path}")

    config_dict, config_object = load_config(copied_config_path)

    print(f"Source trial folder: {source_trial_dir}")
    print(f"Resolved source run folder: {source_run_dir}")
    print(f"Destination folder: {copied_run_dir}")
    print(f"Copied config path: {copied_config_path}")
    print(f"Copied config experiment_name: {config_dict.get('experiment_name')}")
    print(f"Copied config model: {config_object.model}")

    trial_dir = copied_run_dir

    with mlflow.start_run(run_name=DESTINATION_DIR.name) as parent_run:
        mlflow.log_params(
            {
                "source_trial_name": TRIAL_NAME,
                "retrain_name": RETRAIN_NAME,
                "number_of_retrainings": NUMBER_OF_RETRAININGS,
            }
        )

        for i_retrain in range(NUMBER_OF_RETRAININGS):
            config_path, experiment_name = prepare_retrain_config(
                base_config_path=copied_config_path,
                trial_dir=trial_dir,
                i_retrain=i_retrain,
            )

            with mlflow.start_run(
                run_name=f"retrain_{i_retrain + 1}",
                nested=True,
            ):
                mlflow.log_params(
                    {
                        "retrain_index": i_retrain + 1,
                        "experiment_name": experiment_name,
                    }
                )
                mlflow.log_artifact(str(config_path), artifact_path="config")

                gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
                device_mode = "GPU" if gpu_available else "CPU"
                mlflow.set_tag("device_mode", device_mode)

                run_neural_hydrology_model(config_path)

                folders_in_trial = [os.path.join(trial_dir, folder) for folder in os.listdir(trial_dir)]
                folders_in_trial = [f for f in folders_in_trial if os.path.isdir(f)]

                matching_folders = [
                    f for f in folders_in_trial
                    if os.path.basename(f).startswith(experiment_name + '_')
                ]
                if not matching_folders:
                    raise RuntimeError(
                        f"No run folder found starting with '{experiment_name}_' in {trial_dir}. "
                        f"Available folders: {[os.path.basename(f) for f in folders_in_trial]}"
                    )
                run_folder = max(matching_folders, key=os.path.getmtime)
                print(f"Selected run folder: {os.path.basename(run_folder)}")

                data = extract_tensorboard_scalars(run_folder)
                if not any('valid' in tag for tag in data.keys()):
                    print("WARNING: No validation tags found in TensorBoard logs.")
                    print("Run folder contents:")
                    for root, dirs, files in os.walk(run_folder):
                        level = root.replace(str(run_folder), '').count(os.sep)
                        indent = ' ' * 2 * level
                        print(f"{indent}{os.path.basename(root)}/")
                        sub_indent = ' ' * 2 * (level + 1)
                        for file in files:
                            print(f"{sub_indent}{file}")
                    raise RuntimeError(
                        f"No validation TensorBoard tags found in {run_folder}. "
                        f"Available tags: {list(data.keys())}. "
                        "Check that NeuralHydrology is configured to log validation metrics to TensorBoard "
                        "(config: log_tensorboard: true, validate_every: 1)."
                    )

                tag_nse_1d = find_tag(data, 'valid/mean_nse_1D')
                tag_nse_1h = find_tag(data, 'valid/mean_nse_1h')

                validation_NSE_scores_1d = np.array([loss for epoch, loss in data[tag_nse_1d]])
                validation_NSE_scores_1h = np.array([loss for epoch, loss in data[tag_nse_1h]])
                validation_NSE_scores_mean_1d_1h = (validation_NSE_scores_1d + validation_NSE_scores_1h) / 2

                max_validation_NSE_score = float(np.max(validation_NSE_scores_mean_1d_1h))

                for (epoch_nse_1d, loss_nse_1d), (epoch_nse_1h, loss_nse_1h) in zip(
                    data[tag_nse_1d],
                    data[tag_nse_1h],
                ):
                    mlflow.log_metric("val_nse_1d", float(loss_nse_1d), step=int(epoch_nse_1d))
                    mlflow.log_metric("val_nse_1h", float(loss_nse_1h), step=int(epoch_nse_1h))
                    mlflow.log_metric(
                        "val_nse_1h_1d",
                        (float(loss_nse_1d) + float(loss_nse_1h)) / 2,
                        step=int(epoch_nse_1h),
                    )

                mlflow.log_metric("max_validation_nse_1d_1h", max_validation_NSE_score)
                mlflow.log_artifact(str(config_path), artifact_path="config")


if __name__ == "__main__":
    main()
