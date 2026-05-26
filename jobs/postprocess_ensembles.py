#!/usr/bin/env python3
"""
Databricks Job entrypoint: export ensemble inference NetCDF to Unity Catalog.

Reads ``polders_hdsr_1h.nc`` from ``INFERENCE_RUNS_DIR``, truncates
``ENSEMBLE_FORECAST_TABLE``, and reloads all basins/ensemble members.

Databricks (Jobs → Python script):
  Path: /Workspace/Shared/neural_hydrology/jobs/postprocess_ensembles.py

Local (from repo-root, after ``pip install -e .``, requires Spark/Databricks):
  python jobs/postprocess_ensembles.py
"""

from __future__ import annotations

import logging
import os
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

from neural_hydrology.paths import get_env, get_path, get_project_root
from neural_hydrology.postprocessing.export_ensemble_table import (
    ENSEMBLE_NETCDF_NAME,
    load_ensemble_netcdf_long,
    write_ensemble_table,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    table_name = get_env("ENSEMBLE_FORECAST_TABLE")
    if not table_name or not str(table_name).strip():
        raise RuntimeError(
            "ENSEMBLE_FORECAST_TABLE must be set in .env "
            "(e.g. dbw_datascience_tst_weu_001.default.ensemble_forecast_1h)"
        )

    nc_path = get_path("INFERENCE_RUNS_DIR") / ENSEMBLE_NETCDF_NAME
    LOGGER.info(
        "Postprocessing ensemble output (nc=%s, table=%s, root=%s)",
        nc_path,
        table_name,
        get_project_root(),
    )

    df = load_ensemble_netcdf_long(nc_path)

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    write_ensemble_table(spark, df, str(table_name).strip())


if __name__ == "__main__":
    main()
