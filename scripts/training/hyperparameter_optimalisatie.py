import os
os.environ["MLFLOW_TRACKING_URI"] = "databricks"

from pathlib import Path
import yaml
import torch
from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import start_run
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import warnings
warnings.filterwarnings("ignore", message="'H' is deprecated and will be removed in a future version")
import os 
import optuna
import mlflow
import numpy as np
import datetime
from names_generator import generate_name
from collections import defaultdict

NAME = generate_name()
EXPERIMENT_NAME = f"LSTM_{NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
N_TRIALS = 50
BASE_CONFIG = "/Workspace/Shared/neural_hydrology_fork/config.yml"
OUTPUT_DIR = Path("/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output")
RUNS_DIR = OUTPUT_DIR / f"HPO/{EXPERIMENT_NAME}"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)

mlflow.set_tracking_uri("databricks")  # if running on Databricks this is often already configured
mlflow.set_experiment(f"/Shared/{EXPERIMENT_NAME}")

def get_run_folder_by_name_timestamp(trial_dir, experiment_name):
    trial_dir = Path(trial_dir)

    matching_dirs = [
        p for p in trial_dir.iterdir()
        if p.is_dir() and p.name.startswith(f"{experiment_name}_")
    ]

    if not matching_dirs:
        raise RuntimeError(
            f"No run folders found for '{experiment_name}_' in {trial_dir}"
        )

    def parse_run_time(folder: Path):
        suffix = folder.name.removeprefix(f"{experiment_name}_")
        return datetime.datetime.strptime(suffix, "%d%m_%H%M%S")

    matching_dirs.sort(key=parse_run_time, reverse=True)
    return matching_dirs[0]

def run_neural_hydrology_model(config_name):
    run_config = Config(Path(config_name))
    print('model:\t\t', run_config.model)
    print('use_frequencies:', run_config.use_frequencies)
    print('seq_length:\t', run_config.seq_length)

    # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available():
        start_run(config_file=Path(config_name))

    # fall back to CPU-only mode
    else:
        start_run(config_file=Path(config_name), gpu=-1)

def extract_tensorboard_scalars(logdir):
    """Extract all TensorBoard scalars, searching subdirectories for event files."""
    scalars = {}

    # Walk logdir and all subdirectories to find every event file
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

def objective(trial):
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):

        with open(BASE_CONFIG) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        config = dict(config)
        experiment_name = config['experiment_name']
        experiment_name = experiment_name + '_' + str(trial.number)
        config['experiment_name'] = experiment_name

        # Create a per-trial parent folder that holds both config and run output
        trial_dir = RUNS_DIR / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        config["run_dir"] = str(trial_dir)
        config['hidden_size'] = 64
        # config['train_start_date'] = '01/01/2017'
        # config['epochs'] = 20

        # dropout, also apply to the embedding networks
        dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
        config['output_dropout'] = dropout
        config['learning_rate'] = {0: trial.suggest_categorical('learning_rate', [0.001, 0.0005, 0.0001, 0.00005, 0.00001])}


        # STATIC VARIABLES SELECTION OF THE HPO BELOW
        static_variables = [
            'water_percentage',
            'stedelijk_percentage',
            'oppervlak',
            'water_opp',
            'stedelijk_opp',
        ]

        maaiveldhoogte_mean_median_options = {
            'none': [],
            'mean': ['maaiveldhoogte'],
            'median': ['maaiveldhoogte_median'],
            'mean_median': ['maaiveldhoogte', 'maaiveldhoogte_median'],
        }
        maaiveldhoogte_mean_median_choice = trial.suggest_categorical(
            'static_variables_maaiveldhoogte_mean_median',
            list(maaiveldhoogte_mean_median_options.keys()),
        )
        static_variables_maaiveldhoogte_mean_median = maaiveldhoogte_mean_median_options[
            maaiveldhoogte_mean_median_choice
        ]

        maaiveldhoogte_iqr_p95_p05_options = {
            'none': [],
            'iqr': ['maaiveldhoogte_iqr'],
            'p95_p05': ['maaiveldhoogte_p95_minus_p05'],
            'iqr_p95_p05': ['maaiveldhoogte_iqr', 'maaiveldhoogte_p95_minus_p05'],
        }
        maaiveldhoogte_iqr_p95_p05_choice = trial.suggest_categorical(
            'static_variables_maaiveldhoogte_iqr_p95_p05',
            list(maaiveldhoogte_iqr_p95_p05_options.keys()),
        )
        static_variables_maaiveldhoogte_iqr_p95_p05 = maaiveldhoogte_iqr_p95_p05_options[
            maaiveldhoogte_iqr_p95_p05_choice
        ]

        kwel_options = {
            'none': [],
            'kwel_mean': ['kwel_mean'],
        }
        kwel_choice = trial.suggest_categorical(
            'static_variables_kwel',
            list(kwel_options.keys()),
        )
        static_variables_kwel = kwel_options[kwel_choice]

        peil_options = {
            'none': [],
            'peil_range': ['peil_range'],
        }
        peil_choice = trial.suggest_categorical(
            'static_variables_peil',
            list(peil_options.keys()),
        )
        static_variables_peil = peil_options[peil_choice]

        infiltratie_permeabiliteit_options = {
            'none': [],
            'infiltratie': ['infiltratie'],
            'permabiliteit': ['permabiliteit'],
            'infiltratie_permabiliteit': ['infiltratie', 'permabiliteit'],
        }
        infiltratie_permeabiliteit_choice = trial.suggest_categorical(
            'static_variables_infiltratie_permeabiliteit',
            list(infiltratie_permeabiliteit_options.keys()),
        )
        static_variables_infiltratie_permeabiliteit = infiltratie_permeabiliteit_options[
            infiltratie_permeabiliteit_choice
        ]

        static_variables = (
            static_variables
            + static_variables_maaiveldhoogte_mean_median
            + static_variables_maaiveldhoogte_iqr_p95_p05
            + static_variables_kwel
            + static_variables_peil
            + static_variables_infiltratie_permeabiliteit
        )

        config['static_attributes'] = static_variables

        # Save config inside the trial folder
        config_name = f'config_simulatie_nr_{trial.number}.yml'
        config_path = trial_dir / config_name

        with open(config_path, "w") as file:
            yaml.dump(config, file)

        # Log only Optuna trial params (prefixed to avoid collision with NH's internal logging)
        # Full config is already saved as artifact below
        mlflow.log_params({f"optuna/{k}": str(v) for k, v in trial.params.items()})
        mlflow.log_artifact(config_path, artifact_path="config")

        # Determine device mode
        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        device_mode = "GPU" if gpu_available else "CPU"
        mlflow.set_tag("device_mode", device_mode)

        # draai het model met de nieuwe config
        run_neural_hydrology_model(config_path)

        run_folder = get_run_folder_by_name_timestamp(
            trial_dir=trial_dir,
            experiment_name=experiment_name,
        )

        # now we need to log the metric we want to optimize from the neural hydrology model using the latest folder
        # we optimize the hyperparameters to maximize the mean NSE score
        data = extract_tensorboard_scalars(run_folder)
        # Check for validation tags before accessing them
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

        # we take the maximum validation NSE score
        max_validation_NSE_score = float(np.max(validation_NSE_scores_mean_1d_1h))

        for (epoch_nse_1d, loss_nse_1d), (epoch_nse_1h, loss_nse_1h) in zip(
            data[tag_nse_1d],
            data[tag_nse_1h],
        ):
            mlflow.log_metric("val_nse_1d", float(loss_nse_1d), step=int(epoch_nse_1d))
            mlflow.log_metric("val_nse_1h", float(loss_nse_1h), step=int(epoch_nse_1h))
            mlflow.log_metric("val_nse_1h_1d", (float(loss_nse_1d) + float(loss_nse_1h))/2, step=int(epoch_nse_1h))

        mlflow.log_metric("max_validation_nse_1d_1h", max_validation_NSE_score)
        mlflow.log_artifact(str(config_path), artifact_path="config")

    return max_validation_NSE_score


if __name__ == "__main__":
    study = optuna.create_study(
        direction='maximize',
        study_name=EXPERIMENT_NAME,
        storage=f'sqlite:////local_disk0/tmp/{EXPERIMENT_NAME}.db',
        load_if_exists=True
    )

    with mlflow.start_run(run_name=EXPERIMENT_NAME) as parent_run:
        mlflow.set_tag("study_name", EXPERIMENT_NAME)

        study.optimize(objective, n_trials=N_TRIALS)

        importances = optuna.importance.get_param_importances(study)
        mlflow.log_dict(importances, "optuna/param_importances.json")

        fig = optuna.visualization.plot_param_importances(study)
        mlflow.log_figure(fig, "optuna/param_importances.html")

        hist_fig = optuna.visualization.plot_optimization_history(study)
        mlflow.log_figure(hist_fig, "optuna/optimization_history.html")
        mlflow.log_metric("best_value", study.best_value)
        mlflow.log_params({f"best/{k}": str(v) for k, v in study.best_trial.params.items()})
        mlflow.set_tag("best_trial_number", study.best_trial.number)
 