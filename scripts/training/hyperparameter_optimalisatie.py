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

runs_dir = "../../runs"
# mlflow.set_tracking_uri("databricks")  # if running on Databricks this is often already configured
mlflow.set_experiment("/Shared/hdsr_lstm_optuna")
base_config = "../../config.yml"#, "config_simulatie_2.yml", "config_simulatie_3.yml", "config_simulatie_5.yml"]

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
        print("CPU mode")
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

        with open(base_config) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        config = dict(base_config)

        # override with optuna some of the config values we want to optimize
        config['epochs'] = 75 #100
        config['log_interval'] = 1
        config['save_weights_every'] = 1
        # config['train_end_date'] = '01/01/2015' # weghalen

        config['hidden_size'] = 64 

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

        # kies of je maaiveldhoogte mee wil nemen in de statische variabelen en zo ja, welke keuzes

        # maaiveldhoogteopties mean/mediaan
        static_variable_maaiveld_hoogte_mean_median_options = [[], ['maaiveldhoogte'],
    ['maaiveldhoogte_median'], ['maaiveldhoogte', 'maaiveldhoogte_median']]
        static_mv_options1 = trial.suggest_categorical('static_variables_maaiveldhoogte_mean_median', [0,1,2,3 ])
        static_variables_maaiveldhoogte_mean_median = static_variable_maaiveld_hoogte_mean_median_options[static_mv_options1]

        # willen we de iqr en/of p95-p05 mee van het maaiveldhoogte
        static_variables_maaiveldhoogte_iqr_p95_p05_options = [[], ['maaiveldhoogte_iqr'],
        ['maaiveldhoogte_p95_minus_p05'], ['maaiveldhoogte_iqr', 'maaiveldhoogte_p95_minus_p05']]
        static_nv_options2 = trial.suggest_categorical('static_variables_maaiveldhoogte_iqr_p95_p05', [0,1,2,3])
        static_variables_maaiveldhoogte_iqr_p95_p05 = static_variables_maaiveldhoogte_iqr_p95_p05_options[static_nv_options2]

        # willen we de infiltratie en/of permeabiliteit meeneemen in de statische variabelen
        static_variables_infiltratie_permeabiliteit_options = [[], ['infiltratie'], ['permabiliteit'], ['infiltratie', 'permabiliteit']]
        static_inf_perm_options = trial.suggest_categorical('static_variables_infiltratie_permeabiliteit', [0,1,2, 3])
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
        
        # schrijf de nieuwe config naar een nieuw bestand
        with open(config_name, 'w') as file:
            yaml.dump(config, file)

        # draai het model met de nieuwe config
        run_neural_hydrology_model(config_name)

        # we assume the latest folder in the runs directory is of past training above
        folders_in_runs = [os.path.join(runs_dir, folder) for folder in os.listdir(runs_dir)]

        # Filter only directories
        folders_in_runs = [f for f in folders_in_runs if os.path.isdir(f)]

        if folders_in_runs:  # Ensure the list is not empty
            latest_folder = max(folders_in_runs, key=os.path.getctime)
            print("Latest folder:", latest_folder)
        else:
            print("No folders found in 'runs'")

        # now we need to log the metric we want to optimize from the neural hydrology model using the latest folder
        # we optimize the hyperparameters to maximize the mean NSE score
        data = extract_tensorboard_scalars(latest_folder)
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
            mlflow.log_metric("val_nse_1h_1d", (float(loss_nse_1d), float(loss_nse_1h))/2, step=int(epoch_nse_1h))


        mlflow.log_params(trial.params)
        mlflow.log_metric("max_validation_nse_1d_1h", max_validation_NSE_score)
        mlflow.log_artifact(config_name, artifact_path="config")

    return max_validation_NSE_score


study = optuna.create_study(
    direction='maximize',
    study_name='nh_test999',
    storage='sqlite:///nh_test1_study.db',
    load_if_exists=True
)

study.optimize(objective, n_trials=5)

# Output the best result.
print("Best trial:")
print("  Accuracy: {:.4f}".format(study.best_trial.value))
print("  Hyperparameters: ")
for key, value in study.best_trial.params.items():
    print("    {}: {}".format(key, value))
