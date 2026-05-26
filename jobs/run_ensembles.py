#!/usr/bin/env python3
"""
Databricks Job entrypoint: 30-member ensemble inference with 5-model median bagging.

Runs inference for each ``BEST_MODEL_DIR_1`` … ``BEST_MODEL_DIR_5`` in ``.env``,
takes the median per HARMONIE ensemble member, and writes grouped NetCDF to
``INFERENCE_RUNS_DIR``.

Databricks (Jobs → Python script):
  Path: /Workspace/Shared/neural_hydrology/jobs/run_ensembles.py

Local (from repo-root, after ``pip install -e .``):
  python jobs/run_ensembles.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

_DATABRICKS_ENV_CANDIDATES = (
    "/Workspace/Shared/neural_hydrology_fork/.env",
    "/Workspace/Shared/neural_hydrology/.env",
)
for _env_path in _DATABRICKS_ENV_CANDIDATES:
    if Path(_env_path).is_file():
        load_dotenv(_env_path, override=False)
        break

from neural_hydrology.inference.bagging import BAGGING_N_MODELS, median_bag_predictions
from neural_hydrology.inference.run_model import (
    _parse_ensemble_starttime,
    run_ensemble_inference,
    write_inference_netcdf,
)
from neural_hydrology.paths import (
    get_env,
    get_path,
    get_project_root,
    load_env,
    resolve_bagging_model_dirs,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_N_ENSEMBLES = 30


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    run_dirs = resolve_bagging_model_dirs()
    data_dir = get_path("DATA_ENS_DIR")
    out_dir = get_path("INFERENCE_RUNS_DIR")
    n_ensembles = int(get_env("N_ENSEMBLES", str(DEFAULT_N_ENSEMBLES)) or DEFAULT_N_ENSEMBLES)
    ensemble_starttime = _parse_ensemble_starttime(load_env().get("ENSEMBLE_STARTTIME"))

    LOGGER.info(
        "Starting bagged ensemble inference (bagging_n_models=%s, n_ensembles=%s, "
        "data_dir=%s, out_dir=%s, root=%s)",
        BAGGING_N_MODELS,
        n_ensembles,
        data_dir,
        out_dir,
        get_project_root(),
    )

    model_outputs = []
    t0 = time.perf_counter()
    for i, run_dir in enumerate(run_dirs, start=1):
        t_model = time.perf_counter()
        LOGGER.info("Model %s/%s inference: %s", i, BAGGING_N_MODELS, run_dir)
        output = run_ensemble_inference(
            run_dir,
            data_dir,
            n_ensembles,
            ensemble_starttime=ensemble_starttime,
        )
        model_outputs.append(output)
        LOGGER.info(
            "Model %s/%s finished in %.1fs",
            i,
            BAGGING_N_MODELS,
            time.perf_counter() - t_model,
        )

    bagged, bagged_dt = median_bag_predictions(model_outputs)
    meta = model_outputs[0][2]

    write_inference_netcdf(
        out_dir,
        bagged,
        bagged_dt,
        global_attrs={
            "bagging_n_models": str(BAGGING_N_MODELS),
            "bagging_method": "median",
            "run_dirs": ",".join(str(p) for p in run_dirs),
            "n_ensembles": str(n_ensembles),
            "start_date": str(meta["start"]),
            "end_date": str(meta["end"]),
        },
    )

    LOGGER.info(
        "Bagged ensemble inference complete in %.1fs -> %s",
        time.perf_counter() - t0,
        out_dir,
    )


if __name__ == "__main__":
    main()
