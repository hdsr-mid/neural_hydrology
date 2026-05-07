from copy import copy

import warnings

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.viz.time_series import plot_time_series, plot_time_series_by_area
from utils.attributes import get_area
from utils.raw_discharge import DISCHARGE_CSV_COL_NAMES, find_discharge_file_by_code, read_raw_discharge


def find_missing_breakpoints(breakpoints: np.ndarray, max_time_steps: int) -> np.ndarray:
    """
    For an integer array ``breakpoints``, fill all gaps larger than max_time_steps, moving backwards from the upper
    limit of the gap.

    Returns:
        breakpoints, with filled gaps
    """
    breakpoints = np.asarray(breakpoints)
    a = breakpoints[:-1]  # a values
    b = breakpoints[1:]  # b values

    # Compute number of steps for each pair
    steps = (b - a) // max_time_steps

    # Create 2D array of all steps for each pair
    max_steps = steps.max()
    if max_steps == 0:
        return np.array([], dtype=int)

    # Create array of shape (num_pairs, max_steps) with steps 1..max_steps
    step_matrix = np.arange(1, max_steps + 1)
    step_matrix = np.broadcast_to(step_matrix, (len(steps), max_steps))

    # Mask out steps larger than actual steps for each pair
    mask = step_matrix <= steps[:, None]

    # Compute sequences: b - step*n
    seq_matrix = b[:, None] - step_matrix * max_time_steps

    # Apply mask
    seq_matrix = np.where(mask, seq_matrix, 0)

    # Flatten and remove zeros
    seq_matrix = seq_matrix[seq_matrix > 0]
    result = np.concatenate([breakpoints, seq_matrix])
    result.sort()
    return result


def average_series_over_preceding_non_pump_period(time_series: pd.Series, max_time_steps: int):
    """
    Smooth out the pumped volume over the preceding period (including the time step with pumping).
    The length of the preceding period will never be longer than ``max_time_steps``.
    The preceding period ends when another positive value is encountered.
    """
    # treat the series as an array for indexing ops
    values = time_series.values

    # find indices of positive values where the next value is zero
    is_positive = (values > 0)

    # next-is-zero mask, except last element handled separately
    next_zero = np.zeros_like(values, dtype=bool)
    next_zero[:-1] = (values[1:] == 0)

    breakpoints = np.where(is_positive & next_zero)[0]
    breakpoints = find_missing_breakpoints(breakpoints, max_time_steps)

    averaged = np.zeros(len(values), dtype=float)

    for i, breakpoint in enumerate(breakpoints):
        if i == 0:
            start_idx = 0
        else:
            start_idx = np.max(
                [
                    0,  # prevent negative indexing
                    breakpoints[i - 1] + 1, # start after the previous breakpoint
                    breakpoint - (max_time_steps - 1),  # never average over window larger than max_time_steps
                ]
            )

        end_idx = breakpoint + 1   # inclusive of the positive value

        window = values[start_idx:end_idx]
        averaged[start_idx:end_idx] = window.mean()

    # Wrap result back into a Series with same index
    return pd.Series(averaged, index=time_series.index)


if __name__ == "__main__":
    import xarray as xr

    data_dir = Path(__file__).parent.parent.parent / "data"
    raw_discharges_dir = data_dir / "raw_discharge_data"
    basins_file = data_dir / "hdsr_polders.txt"
    with basins_file.open("r") as f:
        lines = f.readlines()
        basins = [line.strip() for line in lines]
    # basins = ["AFVG41"]
    print(basins)
    results = dict()
    num_plots = 3
    plot_i = 1
    max_time_steps = 24 * 3
    for basin in basins:
        if plot_i > num_plots:
            break
        csv_path, structure_type = find_discharge_file_by_code(folder=raw_discharges_dir, code=basin)
        if csv_path is None:
            warnings.warn(f"CSV voor {basin} niet gevonden!")
            continue
        measured_variable_name = DISCHARGE_CSV_COL_NAMES[structure_type]
        raw_discharge = read_raw_discharge(
            csv_path=csv_path,
            variable=measured_variable_name
        )
        smooth_discharge = copy(raw_discharge)
        smooth_discharge[measured_variable_name] = average_series_over_preceding_non_pump_period(
            smooth_discharge[measured_variable_name],
            max_time_steps=max_time_steps
        )
        netcdf_path = data_dir / "time_series" / f"{basin}.nc"
        old_smoothing_method = xr.open_dataset(netcdf_path)["afvoer"].to_dataframe()
        area = get_area(basin=basin)
        old_smoothing_method["afvoer"] = old_smoothing_method["afvoer"] * area * 3600
        results[basin] = {
                "Raw discharge [m3/h]": raw_discharge,
                "Smoothed discharge [m3/h]": smooth_discharge,
                "Old smoothing method [m3/h]": old_smoothing_method,
            }
        plot_i += 1

    plot_time_series_by_area(
        title=f"Raw vs. smooth discharge, max time steps: {max_time_steps}",
        data=results,
    )

