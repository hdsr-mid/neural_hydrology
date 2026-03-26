# this script is meant to take from an experiment 


import os
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
 
from pathlib import Path
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

EXPERIMENT_NAME = "/Shared/hdsr_lstm_optuna_example"
TRIAL_RUN_NAME = "trial_23"
BEST_MODEL_RETRAINING_NAME = EXPERIMENT_NAME + "/" + TRIAL_RUN_NAME + "/BEST_RETRAINING"
DOWNLOAD_DIR = Path("/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/batch_train_model")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("databricks")


def get_experiment_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' was not found.")
    return experiment.experiment_id


def get_single_run(experiment_id: str, run_name: str):
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        output_format="list",
    )

    if not runs:
        raise RuntimeError(
            f"No MLflow run named '{run_name}' found in experiment id {experiment_id}."
        )
    if len(runs) > 1:
        raise RuntimeError(
            f"Multiple MLflow runs named '{run_name}' found in experiment id {experiment_id}: "
            f"{[run.info.run_id for run in runs]}"
        )
    return runs[0]


def get_config_artifact_path(run_id: str) -> str:
    artifacts = mlflow.artifacts.list_artifacts(run_id, artifact_path="config")
    config_files = [artifact.path for artifact in artifacts if artifact.path.endswith((".yml", ".yaml"))]

    if not config_files:
        raise RuntimeError(
            f"No config YAML artifact found for run {run_id}. "
            f"Artifacts under 'config': {[artifact.path for artifact in artifacts]}"
        )
    if len(config_files) > 1:
        raise RuntimeError(
            f"Multiple config YAML artifacts found for run {run_id}: {config_files}"
        )
    return config_files[0]


def download_config_artifact(run_id: str, artifact_path: str, download_dir: Path) -> Path:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=str(download_dir),
    )
    return Path(local_path)


def load_config(config_path: Path):
    with open(config_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_object = Config(config_path)
    return config_dict, config_object


def main():
    experiment_id = get_experiment_id(EXPERIMENT_NAME)
    run = get_single_run(experiment_id=experiment_id, run_name=TRIAL_RUN_NAME)
    artifact_path = get_config_artifact_path(run.info.run_id)
    local_config_path = download_config_artifact(
        run_id=run.info.run_id,
        artifact_path=artifact_path,
        download_dir=DOWNLOAD_DIR,
    )
    config_dict, config_object = load_config(local_config_path)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run name: {TRIAL_RUN_NAME}")
    print(f"Run id: {run.info.run_id}")
    print(f"Downloaded config: {local_config_path}")
    print(f"Config experiment_name: {config_dict.get('experiment_name')}")
    print(f"Config model: {config_dict.get('model')}")
    print(f"Config object model: {config_object.model}")


if __name__ == "__main__":
    main()

