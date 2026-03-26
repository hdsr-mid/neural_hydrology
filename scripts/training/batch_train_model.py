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

PATH_HPO = Path("/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output/HPO/runs")
TRIAL_NR = "trial_28"
RETRAIN_NAME = "example"


RETRAIN_BASE_DIR = Path("/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output/BATCH_RETRAIN/")
DESTINATION_DIR = RETRAIN_BASE_DIR / f"RETRAIN_{TRIAL_NR}_{RETRAIN_NAME}"


def validate_source_run(source_run_dir: Path) -> Path:
    if not source_run_dir.exists():
        raise RuntimeError(f"Source trial folder does not exist: {source_run_dir}")
    if not source_run_dir.is_dir():
        raise RuntimeError(f"Source trial path is not a directory: {source_run_dir}")

    source_config_path = source_run_dir / "config.yml"
    if not source_config_path.exists():
        raise RuntimeError(f"No config.yml found in source trial folder: {source_run_dir}")

    return source_config_path


def copy_trial_folder(source_run_dir: Path, destination_dir: Path) -> Path:
    if destination_dir.exists():
        raise RuntimeError(
            f"Destination folder already exists: {destination_dir}. "
            "Remove it first or change RETRAIN_NAME/TRIAL_NR."
        )

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, destination_dir)
    return destination_dir


def load_config(config_path: Path):
    with open(config_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_object = Config(config_path)
    return config_dict, config_object


def main():
    source_run_dir = PATH_HPO / TRIAL_NR 

    # check which folders there are and select the path of the first folder in the list
    # check which folders there are and select the path of the first folder in the list
    subfolders = [f for f in source_run_dir.iterdir() if f.is_dir()]
    if not subfolders:
        raise RuntimeError(f"No subfolders found in source trial folder: {source_run_dir}")
    source_run_dir = subfolders[0]
    
    validate_source_run(source_run_dir)

    copied_run_dir = copy_trial_folder(source_run_dir, DESTINATION_DIR)
    copied_config_path = copied_run_dir / "config.yml"

    if not copied_config_path.exists():
        raise RuntimeError(f"Copied config.yml not found: {copied_config_path}")

    config_dict, config_object = load_config(copied_config_path)

    print(f"Source trial folder: {source_run_dir}")
    print(f"Destination folder: {copied_run_dir}")
    print(f"Copied config path: {copied_config_path}")
    print(f"Config experiment_name: {config_dict.get('experiment_name')}")
    print(f"Config model: {config_dict.get('model')}")
    print(f"Config object model: {config_object.model}")


if __name__ == "__main__":
    main()


 