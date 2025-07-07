from pathlib import Path
import yaml
import torch
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import warnings
warnings.filterwarnings("ignore", message="'H' is deprecated and will be removed in a future version")
import os 
import numpy as np
import optuna

runs_dir = "runs"

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
        print('No GPU available')
        exit()
        # print("CPU mode")
        # start_run(config_file=Path(config_name), gpu=-1)

def extract_tensorboard_scalars(logdir):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()  # Load the TensorBoard logs

    scalars = {}
    for tag in event_acc.Tags()['scalars']:  # Get all scalar tags
        scalars[tag] = [(e.step, e.value) for e in event_acc.Scalars(tag)]
    
    return scalars


initial_static_features = ['maaiveldhoogte',
                            'maaiveldhoogte_iqr',
                            'water_percentage',
                            'stedelijk_percentage',
                            'oppervlak',
                            'stuw'
                            ]

static_feature_maaiveld_median = ['maaiveldhoogte_median']
static_feature_p95_minus_p05 = ['maaiveldhoogte_p95_minus_p05']
static_feature_infiltratie = ['infiltratie']
static_feature_permabiliteit = ['permabiliteit']
static_feature_stedelijk_opp_water_opp = ['stedelijk_opp', 'water_opp']
static_feature_klei_zand_veen = ['klei', 'zand', 'veen']


# LET OP ALS JE EEN LIST IN DE SUGGESTIONS HEBT,  BIJ TRIAL.SUGGEST_CATEGORICAL, DAN VINDT OPTUNA HET NIET LEUK ALS JE DE RESULTATEN INZICHTELIJK WIL MAKEN
# BIJ OPTUNA DASHBOARD (in de terminal, bij anaconda in de environment optuna-dashboard sqlite:///nh_2_study.db) DAT IS DAN NIET MOGELIJK!!!!!!!!!!!
def objective(trial):

    base_config = "config_simulatie_basis.yml"#, "config_simulatie_2.yml", "config_simulatie_3.yml", "config_simulatie_5.yml"]
    # load the yaml file manually using the yaml package
    with open(base_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # override with optuna some of the config values we want to optimize
    config['epochs'] = 75 


    config['hidden_size'] = 128 # 128 moet het zijn

    # dropout, also apply to the embedding networks
    dropout = 0.2 #trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
    config['output_dropout'] = dropout
    config['statics_embedding']['dropout'] = dropout
    config['dynamics_embedding']['dropout'] = dropout

    # statics embedding
    config['statics_embedding']['hiddens'] = [64,64]#trial.suggest_categorical('statics_embedding_hiddens', [[32], [64], [32, 32], [64, 64], [128]])
    
    # dynamics embedding 
    dynamics_embedding  = [64] #trial.suggest_categorical('dynamics_embedding', ['not', [32], [64], [32, 32], [64, 64], [128]])
    # if dynamics_embedding == 'not':
    #     config.pop('dynamics_embedding')
    # else:
    config['dynamics_embedding']['hiddens'] = dynamics_embedding


    # verdere fine-tune op basis van de eerdere resultaten van de hyperparameter optimalisatie
    config['learning_rate'] = {0: trial.suggest_categorical('learning_rate', [0.0001, 0.00025, 0.0005, 0.00075, 0.001])}

    # static features
    static_features = initial_static_features.copy()

    if trial.suggest_categorical('static_feature_maaiveld_median', [True, False]):
        static_features += static_feature_maaiveld_median
    
    if trial.suggest_categorical('static_feature_p95_minus_p05', [True, False]):
        static_features += static_feature_p95_minus_p05
    
    if trial.suggest_categorical('static_feature_infiltratie', [True, False]):
        static_features += static_feature_infiltratie
    
    if trial.suggest_categorical('static_feature_permabiliteit', [True, False]):
        static_features += static_feature_permabiliteit

    if trial.suggest_categorical('static_feature_stedelijk_opp_water_opp', [True, False]):
        static_features += static_feature_stedelijk_opp_water_opp
    
    if trial.suggest_categorical('static_feature_klei_zand_veen', [True, False]):
        static_features += static_feature_klei_zand_veen
    
    config['static_attributes'] = static_features

    
    # maak bestandsnaam voor de nieuwe config
    trial_nr = trial.number
    config_name = f'config_simulatie_nr_{trial_nr}.yml'
    
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

    # validation NSE scores
    validation_NSE_scores_1d = np.array([loss for epoch, loss in data['valid/median_nse_1d']])
    validation_NSE_scores_1h = np.array([loss for epoch, loss in data['valid/median_nse_1h']])
    validation_NSE_scores_median_1d_1h = (validation_NSE_scores_1d + validation_NSE_scores_1h) / 2

    max_validation_NSE_score = np.max(validation_NSE_scores_median_1d_1h)
        # max_valid_NSE_scores.append(max_validation_NSE_score)

    # save the info dict to a file, this file has the information about where each run is stored, just for another way to be safe
    info_dict = {'run_folder': latest_folder,'config_file':config_name, 'trial_nr':trial_nr,'max_NSE_median_1d_1h': max_validation_NSE_score, 'nse_1d':validation_NSE_scores_1d, 'nse_1h': validation_NSE_scores_1h, 'nse_median_1d_1h': validation_NSE_scores_median_1d_1h}
    info_dict_file = f'info_dict_nr_feature_optimization{trial.number}.yml'
    with open(info_dict_file, 'w') as file:
        yaml.dump(info_dict, file)

    # final_validation_NSE_score = np.mean(max_valid_NSE_scores)

    return max_validation_NSE_score


study = optuna.create_study(
    direction='maximize',
    study_name='nh_feature_optimization_3',
    storage='sqlite:///nh_3_study.db',
    load_if_exists=True
)

study.optimize(objective, n_trials=100)

# Output the best result.
print("Best trial:")
print("  Accuracy: {:.4f}".format(study.best_trial.value))
print("  Hyperparameters: ")
for key, value in study.best_trial.params.items():
    print("    {}: {}".format(key, value))
