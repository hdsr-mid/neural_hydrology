#!/usr/bin/env python3
"""
Databricks Job entrypoint: build ensemble time series NetCDFs for all HDSR polders.

Runs preprocessing for every SHAPE_ID in the attributes CSV over a 365-day
historical window (plus HARMONIE ensemble forecast merge).

Databricks (Jobs → Python script):
  Path: /Workspace/Shared/neural_hydrology/jobs/preprocess_ensembles.py

Local (from repo-root, after ``pip install -e .``):
  python jobs/preprocess_ensembles.py
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

from neural_hydrology.paths import get_project_root
from neural_hydrology.preprocessing.create_timeseries_files import (
    MissingDataConfig,
    create_timeseries_files,
)

LOGGER = logging.getLogger(__name__)
DAYS = 365


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    days = int(os.environ.get("PREPROCESSING_DAYS", str(DAYS)))
    LOGGER.info(
        "Starting preprocessing for all polders (days=%s, root=%s)",
        days,
        get_project_root(),
    )

    create_timeseries_files(
        days=days,
        missing_cfg=MissingDataConfig(),
        basin_ids=None,
    )


if __name__ == "__main__":
    main()
