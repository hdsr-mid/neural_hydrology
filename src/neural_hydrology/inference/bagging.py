"""Median bagging across multiple trained model inference outputs."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PerFreqByBasin = dict[str, dict[str, dict[str, np.ndarray]]]
PerFreqDatetime = dict[str, pd.DatetimeIndex]
InferenceMeta = dict[str, Any]
ModelOutput = tuple[PerFreqByBasin, PerFreqDatetime, InferenceMeta]

BAGGING_N_MODELS = 5


def median_bag_predictions(model_outputs: list[ModelOutput]) -> tuple[PerFreqByBasin, PerFreqDatetime]:
    """
    Aggregate per-model inference outputs by taking the median per variable and timestep.

    Datetimes are aligned per frequency using the intersection across all models.
    """
    if len(model_outputs) != BAGGING_N_MODELS:
        raise ValueError(f"Expected {BAGGING_N_MODELS} model outputs, got {len(model_outputs)}")

    frequency_sets = [frozenset(meta["frequencies"]) for _, _, meta in model_outputs]
    if len(set(frequency_sets)) != 1:
        raise ValueError("All models must use the same use_frequencies configuration.")

    frequencies = list(model_outputs[0][2]["frequencies"])
    bagged: PerFreqByBasin = {f: {} for f in frequencies}
    bagged_dt: PerFreqDatetime = {}

    for freq in frequencies:
        dt_common: pd.DatetimeIndex | None = None
        for _, per_dt, _ in model_outputs:
            dt = per_dt.get(freq)
            if dt is None or len(dt) == 0:
                raise ValueError(f"Missing datetime index for frequency {freq!r}")
            dt_common = dt if dt_common is None else dt_common.intersection(dt)

        if dt_common is None or len(dt_common) == 0:
            raise ValueError(f"Empty datetime intersection for frequency {freq!r}")

        bagged_dt[freq] = dt_common
        logger.info(
            "Bagging datetime intersection for %s: %s steps (%s .. %s)",
            freq,
            len(dt_common),
            dt_common.min(),
            dt_common.max(),
        )

        basins: set[str] | None = None
        for per_basin, _, _ in model_outputs:
            basin_set = set(per_basin.get(freq, {}).keys())
            basins = basin_set if basins is None else basins & basin_set
        if not basins:
            raise ValueError(f"No common basins for frequency {freq!r}")

        for basin in sorted(basins):
            var_names: set[str] | None = None
            for per_basin, _, _ in model_outputs:
                names = set(per_basin.get(freq, {}).get(basin, {}).keys())
                var_names = names if var_names is None else var_names & names
            if not var_names:
                raise ValueError(f"No common simulation variables for basin {basin!r}, freq {freq!r}")

            bagged[freq].setdefault(basin, {})
            for var_name in sorted(var_names):
                stacks: list[np.ndarray] = []
                for per_basin, per_dt, _ in model_outputs:
                    dt = per_dt[freq]
                    values = per_basin[freq][basin][var_name]
                    aligned = pd.Series(values, index=dt).reindex(dt_common).to_numpy()
                    stacks.append(aligned)
                bagged[freq][basin][var_name] = np.median(np.stack(stacks, axis=0), axis=0)

    return bagged, bagged_dt
