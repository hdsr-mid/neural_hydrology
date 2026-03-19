import os
os.environ["MLFLOW_TRACKING_URI"] = "databricks"

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import start_run, eval_run
from neuralhydrology.evaluation import metrics, get_tester

import mlflow

# Output directory to Unity Catalog Volume
OUTPUT_DIR = Path("/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output")

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/hdsr_batch_single_lstm")


def run_neural_hydrology_model(config_name):
    run_config = Config(Path(config_name))
    print('model:\t\t', run_config.model)
    print('use_frequencies:', run_config.use_frequencies)
    print('seq_length:\t', run_config.seq_length)

    # Write outputs to run folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_DIR)
    runs_dir = OUTPUT_DIR / "runs"

    # Load config for logging parameters to MLflow
    with open(config_name) as f:
        config_dict = yaml.safe_load(f)

    with mlflow.start_run(run_name="run_model"):
        # Log configuration parameters
        mlflow.log_params({
            "model": config_dict.get("model"),
            "epochs": config_dict.get("epochs"),
            "batch_size": config_dict.get("batch_size"),
            "hidden_size": config_dict.get("hidden_size"),
            "learning_rate": str(config_dict.get("learning_rate")),
            "loss": config_dict.get("loss"),
            "optimizer": config_dict.get("optimizer"),
            "seq_length": str(config_dict.get("seq_length")),
            "use_frequencies": str(config_dict.get("use_frequencies")),
        })
        mlflow.log_artifact(str(config_name), artifact_path="config")
        mlflow.set_tag("config_file", str(config_name))

        # Determine device mode
        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        device_mode = "GPU" if gpu_available else "CPU"
        mlflow.set_tag("device_mode", device_mode)
        print(f"{device_mode} mode")

        # Train model
        if gpu_available:
            start_run(config_file=Path(config_name))
        else:
            start_run(config_file=Path(config_name), gpu=-1)

        # Evaluate the model on the test set
        run_dir = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_ctime,
        )[-1]
        tester_config = Config(run_dir / "config.yml")
        tester = get_tester(
            cfg=Config(run_dir / "config.yml"),
            run_dir=run_dir,
            period="test",
            init_model=True,
        )
        results = tester.evaluate(save_results=True, metrics=tester_config.metrics)

        # Save figures
        fig_path = f"{run_dir}/figures"
        os.makedirs(fig_path, exist_ok=True)

        # Loop over all polders and create a child run for each
        polders = results.keys()
        nse_values = {}

        for polder in polders:
            with mlflow.start_run(run_name=polder, nested=True) as child_run:
                mlflow.set_tag("polder", polder)

                # Get hourly data
                hourly_xr = results[polder]["1h"]["xr"]
                hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(
                    datetime=["date", "time_step"]
                )
                hourly_xr["datetime2"] = (
                    hourly_xr.coords["date"] + hourly_xr.coords["time_step"]
                )
                qobs = hourly_xr["afvoer_obs"]
                qsim = hourly_xr["afvoer_sim"]

                # Check if NSE is available
                if "NSE_1h" in results[polder]["1h"]:
                    NSE = results[polder]["1h"]["NSE_1h"]
                elif "NSE" in results[polder]["1h"]:
                    NSE = results[polder]["1h"]["NSE"]
                else:
                    NSE = -999

                nse_values[polder] = NSE

                # Log NSE metric to child run
                mlflow.log_metric("nse_1h", float(NSE))

                fig, ax = plt.subplots(figsize=(16, 8))
                ax.plot(qobs["date"], qobs, label="Observed")
                ax.plot(qsim["date"], qsim, label="Simulated")
                ax.legend()
                ax.set_ylabel("Discharge (mm/d)")
                ax.set_title(
                    f"Polder: {polder} - Test period - hourly NSE {NSE:.3f}"
                )
                figure_file = f"{fig_path}/{polder}_1h.png"
                fig.savefig(figure_file)
                mlflow.log_artifact(figure_file, artifact_path="figures")
                plt.close(fig)

        # Log summary metrics on the parent run
        valid_nse = [v for v in nse_values.values() if v != -999]
        if valid_nse:
            mlflow.log_metric("mean_nse", float(np.mean(valid_nse)))
            mlflow.log_metric("median_nse", float(np.median(valid_nse)))
            mlflow.log_metric("min_nse", float(np.min(valid_nse)))
            mlflow.log_metric("max_nse", float(np.max(valid_nse)))
        mlflow.log_metric("num_polders", len(nse_values))

        mlflow.set_tag("status", "success")
        print(f"MLflow run logged. NSE values: {nse_values}")


config_name_list = [Path("/Workspace/Shared/neural_hydrology/config.yml").resolve()]

# code om batch aan configs te draaien
for config_name in config_name_list:
    print("start of run with config", config_name)
    run_neural_hydrology_model(config_name)
    print("run finished")