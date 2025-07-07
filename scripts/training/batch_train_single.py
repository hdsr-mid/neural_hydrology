import os
from pathlib import Path
import torch
from neuralhydrology.evaluation import metrics,get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def create_graph(run_name):
    run_path = os.path.join(runs_dir, run_name)
    print(run_path)

    # Create folder for figures
    fig_path = os.path.join(run_path, "figures")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    run_dir = Path(run_path)

    tester_config = Config(run_dir / "config.yml")

    tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
    results = tester.evaluate(save_results=True, metrics=tester_config.metrics)

    # Loop over all polders and create a figure for each polder
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

        return NSE

# Make list of all polders
skip = [
    "Prins_Hendrikpolder",
]

# Read txt file with the basin names
with open("hdsr_polders.txt") as f:
    polders = f.readlines()
polders = [x.strip() for x in polders]

# Remove polders from the list
polders = [x for x in polders if x not in skip]

# Create empty dictionary to store NSE values
NSE_dict = {}

# Loop over all polders for training
for polder in polders:
    print("Training model for", polder)
    with open("hdsr_single.txt", "w") as f:
        f.write(polder)


    try:
    # train model
        config_path = "template_single_lstm.yml" 
        start_run(config_file=Path(config_path))

        # evaluate model
        script_path = os.getcwd()
        runs_dir = os.path.join(script_path, "runs")
        run_list = os.listdir(runs_dir)
        run_list = [run_name for run_name in run_list if "single" in run_name]
        run_list.sort(key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)), reverse=True)
        run_name = run_list[0]
        NSE = create_graph(run_name)

        # store NSE value in dictionary
        NSE_dict[polder] = NSE

    except:
        print(f"Error in {polder}")
        NSE_dict[polder] = np.nan

    # time.sleep(60)

# print NSE values
print(NSE_dict)

# write NSE values to csv
df = pd.DataFrame(list(NSE_dict.items()), columns=['Polder', 'NSE'])
df.to_csv("hhnk_single_lstm_NSE.csv", index=False)

    