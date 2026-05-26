import os

# Prevent TensorFlow (loaded as side-effect on Databricks ML runtime) from
# grabbing all GPU memory, which conflicts with PyTorch's CUDA context.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import yaml
import torch
from neural_hydrology.paths import get_env, get_path, load_env
from neural_hydrology.utils.training import (
    load_validated_tensorboard_scalars,
    log_tensorboard_metrics_to_mlflow,
    run_neural_hydrology_model,
)
from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import start_run
import warnings
warnings.filterwarnings("ignore", message="'H' is deprecated and will be removed in a future version")
import os 
import optuna
import mlflow
import numpy as np
import datetime
from names_generator import generate_name
from collections import defaultdict

load_env()
mlflow_uri = get_env("MLFLOW_TRACKING_URI", "databricks")
if mlflow_uri:
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

NAME = generate_name()
EXPERIMENT_NAME = f"LSTM_{NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
N_TRIALS = 50
BASE_CONFIG = str(get_path("BASE_CONFIG"))
OUTPUT_DIR = get_path("OUTPUT_DIR")
RUNS_DIR = get_path("HPO_OUTPUT_DIR") / EXPERIMENT_NAME
RUNS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(f"/Shared/{EXPERIMENT_NAME}")

def get_run_folder_by_name_timestamp(trial_dir, experiment_name):
    """Find the most recent NeuralHydrology run folder for a given experiment.

    NeuralHydrology creates a subfolder for each run with the naming convention
    '{experiment_name}_{DDMM_HHMMSS}'. This function finds all such folders in
    the given trial directory and returns the one with the most recent timestamp.

    Parameters
    ----------
    trial_dir : str or Path
        Directory to search for run folders. Typically the per-trial output
        directory (e.g. RUNS_DIR / 'trial_0').
    experiment_name : str
        The experiment name prefix used by NeuralHydrology when creating the
        run folder (matches config['experiment_name']).

    Returns
    -------
    Path
        The Path to the most recently created run folder.

    Raises
    ------
    RuntimeError
        If no folders matching '{experiment_name}_*' are found in trial_dir.
    """
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


def generate_static_attributes_HPO(trial):
    """Generate the list of static catchment attributes for a hyperparameter trial.

    Uses Optuna's suggest_categorical to select which groups of static attributes
    to include in the model. A base set of attributes is always included

    Parameters
    ----------
    trial : optuna.trial.Trial
        The current Optuna trial object, used to suggest categorical choices
        for each attribute group.

    Returns
    -------
    list[str]
        Combined list of static attribute column names to be used in the
        NeuralHydrology config under 'static_attributes'.
    """
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

def objective(trial):
    """Optuna objective function for hyperparameter optimization of the LSTM model.

    Executes one full trial of the HPO loop:
    1. Loads the base NeuralHydrology config from BASE_CONFIG.
    2. Modifies hyperparameters (dropout, learning_rate, static_attributes)
       based on Optuna suggestions.
    3. Saves the modified config to a per-trial directory.
    4. Logs trial parameters and config as an MLflow artifact.
    5. Trains the NeuralHydrology model with the modified config.
    6. Extracts validation metrics from TensorBoard logs and logs them to MLflow.
    7. Returns the optimization target (max mean NSE across 1d and 1h resolutions).

    Parameters
    ----------
    trial : optuna.trial.Trial
        The current Optuna trial object providing suggest_* methods for
        hyperparameter sampling.

    Returns
    -------
    float
        The maximum validation NSE score (average of 1-day and 1-hour mean NSE)
        achieved during training. Optuna maximizes this value.
    """
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
        config['static_attributes'] = generate_static_attributes_HPO(trial)

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

        data = load_validated_tensorboard_scalars(run_folder)
        max_validation_NSE_score, best_epoch = log_tensorboard_metrics_to_mlflow(data)

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
 