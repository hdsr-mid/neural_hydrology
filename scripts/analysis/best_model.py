# Loop over all runs and get best run based on NSE_1h
# best_run = None

import pandas as pd
from pathlib import Path

# Get all runs and order by file creation time
runs = sorted([d for d in Path("runs").iterdir() if d.is_dir()], key=lambda x: x.stat().st_ctime)

for run in runs:
    if run.is_dir():
        try:
            test_metrics = pd.read_csv(run / "test/model_epoch100/test_metrics.csv")
            # print name of file and statistics of column NSE_1h
            print(f"{run}: {test_metrics['NSE_1h'].describe()}")
        except FileNotFoundError:
            try:
                test_metrics = pd.read_csv(run / "test/model_epoch075/test_metrics.csv")
                # print name of file and statistics of column NSE_1h
                print(f"{run}: {test_metrics['NSE_1h'].describe()}")
            except FileNotFoundError:
                try:
                    test_metrics = pd.read_csv(run / "test/model_epoch005/test_metrics.csv")
                    # print name of file and statistics of column NSE_1h
                    print(f"{run}: {test_metrics['NSE_1h'].describe()}")
                except FileNotFoundError:
                    print(f"File not found in {run}")
                    continue
        except FileNotFoundError:
            print(f"File not found in {run}")
            continue

#%%