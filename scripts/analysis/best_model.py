import argparse
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from mlflow import MlflowClient
from neuralhydrology.nh_run import start_run
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator




# NOTE: WIP NOT READY!!!!!!!!!


os.environ["MLFLOW_TRACKING_URI"] = "databricks"

SEEDS = [0, 1, 2, 3, 4]


def get_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]

    notebook_root = Path("/Workspace/Shared/neural_hydrology_fork")
    if notebook_root.exists():
        return notebook_root

    return Path.cwd()


PROJECT_ROOT = get_project_root()
BASE_CONFIG = PROJECT_ROOT / "config.yml"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "configs_best_model"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain one Optuna trial multiple times with different seeds."
    )
    parser.add_argument("--experiment-name", help="MLflow experiment name, with or without /Shared/ prefix.")
    parser.add_argument("--trial-number", type=int, help="Optuna trial number to retrain.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="List of seeds to retrain with. Defaults to 0 1 2 3 4.",
    )
    return parser.parse_args()


def get_user_inputs(args: argparse.Namespace) -> tuple[str, int, list[int]]:
    experiment_name = args.experiment_name or input("Experiment name: ").strip()
    trial_number = args.trial_number if args.trial_number is not None else int(input("Trial number: ").strip())
    return experiment_name, trial_number, args.seeds


def normalize_experiment_name(experiment_name: str) -> str:
    if experiment_name.startswith("/"):
        return experiment_name
    return f"/Shared/{experiment_name}"


def extract_tensorboard_scalars(logdir: Path) -> dict[str, list[tuple[int, float]]]:
    event_acc = EventAccumulator(str(logdir))
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags()["scalars"]:
        scalars[tag] = [(event.step, event.value) for event in event_acc.Scalars(tag)]
    return scalars


def run_neural_hydrology_model(config_path: Path) -> None:
    if torch.cuda.is_available():
        start_run(config_file=config_path)
    else:
        start_run(config_file=config_path, gpu=-1)


def get_trial_run(client: MlflowClient, experiment_id: str, trial_number: int):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = 'trial_{trial_number}'",
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"Could not find MLflow run for trial_{trial_number}.")
    return runs[0]


def find_config_artifact(client: MlflowClient, run_id: str) -> str:
    artifact_infos = client.list_artifacts(run_id, "config")
    yaml_artifacts = [
        artifact.path for artifact in artifact_infos if artifact.path.endswith((".yml", ".yaml"))
    ]
    if not yaml_artifacts:
        raise RuntimeError(f"No YAML config artifact found under 'config' for run {run_id}.")
    if len(yaml_artifacts) > 1:
        raise RuntimeError(f"Expected one config artifact, found {len(yaml_artifacts)} for run {run_id}.")
    return yaml_artifacts[0]


def load_trial_config(client: MlflowClient, run_id: str) -> dict:
    artifact_path = find_config_artifact(client, run_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_config_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=tmp_dir,
            )
        )
        with open(local_config_path) as file:
            return yaml.load(file, Loader=yaml.FullLoader)


def get_run_folder(experiment_prefix: str) -> Path:
    matching_runs = [
        run_dir for run_dir in RUNS_DIR.iterdir() if run_dir.is_dir() and run_dir.name.startswith(experiment_prefix)
    ]
    if not matching_runs:
        raise RuntimeError(f"No NH run folder found for experiment name prefix '{experiment_prefix}'.")
    return max(matching_runs, key=lambda path: path.stat().st_ctime)


def log_validation_metrics(run_folder: Path) -> float:
    data = extract_tensorboard_scalars(run_folder)
    validation_nse_scores_1d = np.array([loss for _, loss in data["valid/mean_nse_1d"]])
    validation_nse_scores_1h = np.array([loss for _, loss in data["valid/mean_nse_1h"]])
    validation_nse_scores_mean = (validation_nse_scores_1d + validation_nse_scores_1h) / 2
    max_validation_nse_score = float(np.max(validation_nse_scores_mean))

    for (epoch_nse_1d, loss_nse_1d), (epoch_nse_1h, loss_nse_1h) in zip(
        data["valid/mean_nse_1d"],
        data["valid/mean_nse_1h"],
    ):
        mlflow.log_metric("val_nse_1d", float(loss_nse_1d), step=int(epoch_nse_1d))
        mlflow.log_metric("val_nse_1h", float(loss_nse_1h), step=int(epoch_nse_1h))
        mlflow.log_metric(
            "val_nse_1h_1d",
            (float(loss_nse_1d) + float(loss_nse_1h)) / 2,
            step=int(epoch_nse_1h),
        )

    mlflow.log_metric("max_validation_nse_1d_1h", max_validation_nse_score)
    return max_validation_nse_score


def train_repeat(source_experiment_name: str, trial_number: int, seed: int, base_config: dict) -> None:
    config = dict(base_config)
    repeat_experiment_name = f"{source_experiment_name}_trial_{trial_number}_seed_{seed}"
    config["experiment_name"] = repeat_experiment_name
    config["run_dir"] = str(RUNS_DIR)
    config["seed"] = seed

    config_path = CONFIG_DIR / f"{repeat_experiment_name}.yml"
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    with mlflow.start_run(run_name=f"seed_{seed}", nested=True):
        mlflow.log_param("source_experiment_name", source_experiment_name)
        mlflow.log_param("source_trial_number", trial_number)
        mlflow.log_param("seed", seed)
        mlflow.log_artifact(str(config_path), artifact_path="config")

        run_neural_hydrology_model(config_path)
        run_folder = get_run_folder(repeat_experiment_name)
        log_validation_metrics(run_folder)
        mlflow.log_param("nh_run_folder", str(run_folder))


def main() -> None:
    args = parse_args()
    experiment_name, trial_number, seeds = get_user_inputs(args)

    source_experiment_name = experiment_name.strip("/")
    normalized_experiment_name = normalize_experiment_name(experiment_name)
    rerun_experiment_name = f"/Shared/{source_experiment_name.split('/')[-1]}_best_model"

    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(normalized_experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{normalized_experiment_name}' not found.")

    trial_run = get_trial_run(client, experiment.experiment_id, trial_number)
    trial_config = load_trial_config(client, trial_run.info.run_id)
    if not isinstance(trial_config, dict):
        raise RuntimeError(f"Expected dict-like YAML config, got {type(trial_config).__name__}.")

    mlflow.set_experiment(rerun_experiment_name)
    with mlflow.start_run(run_name=f"best_model_trial_{trial_number}") as parent_run:
        mlflow.log_param("source_experiment_name", normalized_experiment_name)
        mlflow.log_param("source_trial_number", trial_number)
        mlflow.log_param("source_run_id", trial_run.info.run_id)
        mlflow.log_param("base_config_path", str(BASE_CONFIG))
        mlflow.log_param("runs_dir", str(RUNS_DIR))
        mlflow.set_tag("rerun_type", "best_model_retrain")

        for seed in seeds:
            train_repeat(source_experiment_name.split("/")[-1], trial_number, seed, trial_config)


if __name__ == "__main__":
    main()
