import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import start_run,eval_run
from neuralhydrology.evaluation import metrics, get_tester


def run_neural_hydrology_model(config_name):
    run_config = Config(Path(config_name))
    print('model:\t\t', run_config.model)
    print('use_frequencies:', run_config.use_frequencies)
    print('seq_length:\t', run_config.seq_length)

    # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        print("GPU mode")
        start_run(config_file=Path(config_name))
        
        # evaluate the model on the test set
        run_dir = sorted([d for d in Path("runs").iterdir() if d.is_dir()], key=lambda x: x.stat().st_ctime)[-1]
        tester_config = Config(run_dir / "config.yml")
        tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
        results = tester.evaluate(save_results=True, metrics=tester_config.metrics)

        # save figures
        fig_path = f"{run_dir}/figures"
        os.makedirs(fig_path, exist_ok=True)

        # loop over all polders and create a figure for each polder
        polders = results.keys()

        for polder in polders:
            # get hourly data
            hourly_xr = results[polder]["1h"]["xr"]
            hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
            hourly_xr['datetime2'] = hourly_xr.coords['date'] + hourly_xr.coords['time_step']
            qobs = hourly_xr["afvoer_obs"]
            qsim = hourly_xr["afvoer_sim"]

            # check if results[polder]['1h']['NSE'] is available and set as NSE variable else set to -999
            if 'NSE_1h' in results[polder]['1h']:
                NSE = results[polder]['1h']['NSE_1h']
            elif 'NSE' in results[polder]['1h']:
                NSE = results[polder]['1h']['NSE']
            else:
                NSE = -999

            fig, ax = plt.subplots(figsize=(16,8))
            ax.plot(qobs["date"], qobs, label="Observed")
            ax.plot(qsim["date"], qsim, label="Simulated")
            ax.legend()
            ax.set_ylabel("Discharge (mm/d)")
            ax.set_title(f"Polder: {polder} - Test period - hourly NSE {NSE:.3f}")
            fig.savefig(f"{fig_path}/{polder}_1h.png")
            plt.close(fig)

    # fall back to CPU-only mode
    else:
        print("CPU mode")
        start_run(config_file=Path(config_name), gpu=-1)
        
        # evaluate the model on the test set
        run_dir = sorted([d for d in Path("runs").iterdir() if d.is_dir()])[-1]
        tester_config = Config(run_dir / "config.yml")
        tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
        results = tester.evaluate(save_results=True, metrics=tester_config.metrics)

        # save figures
        fig_path = f"{run_dir}/figures"
        os.makedirs(fig_path, exist_ok=True)

        # Loop over all polders and create a figure for each polder
        polders = results.keys()

        for polder in polders:
            # Get data
            hourly_xr = results[polder]["1h"]["xr"]
            hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
            hourly_xr['datetime2'] = hourly_xr.coords['date'] + hourly_xr.coords['time_step']
            qobs = hourly_xr["afvoer_obs"]
            qsim = hourly_xr["afvoer_sim"]

            # check if results[polder]['1h']['NSE'] is available and set as NSE variable else set to -999
            if 'NSE_1h' in results[polder]['1h']:
                NSE = results[polder]['1h']['NSE_1h']
            elif 'NSE' in results[polder]['1h']:
                NSE = results[polder]['1h']['NSE']
            else:
                NSE = -999

            fig, ax = plt.subplots(figsize=(16,8))
            ax.plot(qobs["date"], qobs, label="Observed")
            ax.plot(qsim["date"], qsim, label="Simulated")
            ax.legend()
            ax.set_ylabel("Discharge (mm/d)")
            ax.set_title(f"Polder: {polder} - Test period - hourly NSE {NSE:.3f}")
            fig.savefig(f"{fig_path}/{polder}_1h.png")
            plt.close(fig)
    
# config_name_list = ["config_simulatie_1.yml", "config_simulatie_2.yml", "config_simulatie_3.yml", "config_simulatie_4.yml", "config_simulatie_5.yml"]
# config_name_list = ["final_config.yml"]
config_name_list = ["1_mtslstm_emb.yml", "2_mtslstm_emb_ohe.yml", "3_mtslstm.yml", "4_mtslstm_ohe.yml", 
                    "5_smtslstm.yml", "6_smtslstm_emb.yml", "7_smtslstm_ohe.yml", "8_smtslstm_emb_ohe.yml", 
                    "9_mtslstm_emb_stat.yml", "10_mtslstm_emb_dyn.yml"]

# code om batch aan configs te draaien
for config_name in config_name_list:
    print("start of run with config", config_name)
    run_neural_hydrology_model(config_name)
    print("run finished")