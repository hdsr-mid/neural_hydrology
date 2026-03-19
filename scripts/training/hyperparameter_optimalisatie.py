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

EXPERIMENT_NAME = f"hdsr_lstm_optuna_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
N_TRIALS = 5
BASE_CONFIG = "../../config.yml" #, "config_simulatie_2.yml", "config_simulatie_3.yml", "config_simulatie_5.yml"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("databricks")  # if running on Databricks this is often already configured
mlflow.set_experiment(f"/Shared/{EXPERIMENT_NAME}")

# make config folder
config_folder = Path(f'../../configs_{EXPERIMENT_NAME}')
config_folder.mkdir(parents=True, exist_ok=True)



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
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()  # Load the TensorBoard logs

    scalars = {}
    for tag in event_acc.Tags()['scalars']:  # Get all scalar tags
        scalars[tag] = [(e.step, e.value) for e in event_acc.Scalars(tag)]
    
    return scalars
    
def objective(trial):
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):

        with open(BASE_CONFIG) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        config = dict(config)
        experiment_name = config['experiment_name']
        experiment_name = experiment_name + '_' + str(trial.number)
        config['experiment_name'] = experiment_name

        config["run_dir"] = str(RUNS_DIR)
        config['hidden_size'] = 16 
        config['train_start_date'] = '01/01/2017'
        config['epochs'] = 20


        # dropout, also apply to the embedding networks
        dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
        config['output_dropout'] = dropout
        config['learning_rate'] = {0: trial.suggest_categorical('learning_rate', [0.001, 0.0005, 0.0001, 0.00005, 0.00001])}
        static_variables = ['water_percentage',
                            'stedelijk_percentage',
                            'oppervlak',
                            'water_opp',
                            'stedelijk_opp',
                            'zand',
                            'klei',
                            'veen',
                            'stuw']  

        # maaiveldhoogteopties mean/mediaan
        static_variable_maaiveld_hoogte_mean_median_options = [[], ['maaiveldhoogte'],
    ['maaiveldhoogte_median'], ['maaiveldhoogte', 'maaiveldhoogte_median']]
        static_mv_options1 = trial.suggest_categorical('static_variables_maaiveldhoogte_mean_median', [0, 1, 2, 3])
        static_variables_maaiveldhoogte_mean_median = static_variable_maaiveld_hoogte_mean_median_options[static_mv_options1]

        # willen we de iqr en/of p95-p05 mee van het maaiveldhoogte
        static_variables_maaiveldhoogte_iqr_p95_p05_options = [[], ['maaiveldhoogte_iqr'],
        ['maaiveldhoogte_p95_minus_p05'], ['maaiveldhoogte_iqr', 'maaiveldhoogte_p95_minus_p05']]
        static_nv_options2 = trial.suggest_categorical('static_variables_maaiveldhoogte_iqr_p95_p05', [0, 1, 2, 3])
        static_variables_maaiveldhoogte_iqr_p95_p05 = static_variables_maaiveldhoogte_iqr_p95_p05_options[static_nv_options2]

        # willen we de infiltratie en/of permeabiliteit meeneemen in de statische variabelen
        static_variables_infiltratie_permeabiliteit_options = [[], ['infiltratie'], ['permabiliteit'], ['infiltratie', 'permabiliteit']]
        static_inf_perm_options = trial.suggest_categorical('static_variables_infiltratie_permeabiliteit', [0, 1, 2, 3])
        static_variables_infiltratie_permeabiliteit = static_variables_infiltratie_permeabiliteit_options[static_inf_perm_options]

        # alle statische variabelen bij elkaar mergen
        static_variables = \
                    static_variables + \
                    static_variables_maaiveldhoogte_mean_median + \
                    static_variables_maaiveldhoogte_iqr_p95_p05 + \
                    static_variables_infiltratie_permeabiliteit

        config['static_attributes'] = static_variables

        # maak bestandsnaam voor de nieuwe config
        config_name = f'config_simulatie_nr_{trial.number}.yml'
        config_path = config_folder / config_name

        with open(config_path, "w") as file:
            yaml.dump(config, file)

        # schrijf de nieuwe config naar een nieuw bestand
        mlflow.log_params(
            config
        )
        mlflow.log_artifact(config_path, artifact_path="config")

        # Determine device mode
        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        device_mode = "GPU" if gpu_available else "CPU"
        mlflow.set_tag("device_mode", device_mode)

        # draai het model met de nieuwe config
        run_neural_hydrology_model(config_path)

        # we assume the latest folder in the runs directory is of past training above
        folders_in_runs = [os.path.join(RUNS_DIR, folder) for folder in os.listdir(RUNS_DIR)]

        # Filter only directories 
        folders_in_runs = [f for f in folders_in_runs if os.path.isdir(f)]
        run_folder = [f for f in folders_in_runs if os.path.basename(f).startswith(experiment_name)][0]

        # now we need to log the metric we want to optimize from the neural hydrology model using the latest folder
        # we optimize the hyperparameters to maximize the mean NSE score
        data = extract_tensorboard_scalars(run_folder)
        validation_NSE_scores_1d = np.array([loss for epoch, loss in data['valid/mean_nse_1d']])
        validation_NSE_scores_1h = np.array([loss for epoch, loss in data['valid/mean_nse_1h']])
        validation_NSE_scores_mean_1d_1h = (validation_NSE_scores_1d + validation_NSE_scores_1h) / 2

        # we take the maximum validation NSE score
        max_validation_NSE_score = float(np.max(validation_NSE_scores_mean_1d_1h))

        for (epoch_nse_1d, loss_nse_1d), (epoch_nse_1h, loss_nse_1h) in zip(
            data["valid/mean_nse_1d"],
            data["valid/mean_nse_1h"],
        ):
            mlflow.log_metric("val_nse_1d", float(loss_nse_1d), step=int(epoch_nse_1d))
            mlflow.log_metric("val_nse_1h", float(loss_nse_1h), step=int(epoch_nse_1h))
            mlflow.log_metric("val_nse_1h_1d", (float(loss_nse_1d) + float(loss_nse_1h))/2, step=int(epoch_nse_1h))

        mlflow.log_params(trial.params)
        mlflow.log_metric("max_validation_nse_1d_1h", max_validation_NSE_score)
        mlflow.log_artifact(str(config_path), artifact_path="config")

    return max_validation_NSE_score


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
    mlflow.log_params(study.best_trial.params)
    mlflow.set_tag("best_trial_number", study.best_trial.number)
