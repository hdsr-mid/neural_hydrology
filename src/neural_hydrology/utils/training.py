
import os
import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config


def run_neural_hydrology_model(config_name):
    """Train a NeuralHydrology model using the specified configuration file.

    Loads the config, prints the model type, and launches the training run.
    Automatically selects GPU if CUDA is available, otherwise falls back to CPU.

    Parameters
    ----------
    config_name : str or Path
        Path to the NeuralHydrology YAML configuration file.

    Returns
    -------
    None
        The trained model and results are written to the run directory
        specified in the config file.
    """
    run_config = Config(Path(config_name))
    print('model:\t\t', run_config.model)

    # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available():
        start_run(config_file=Path(config_name))

    # fall back to CPU-only mode
    else:
        start_run(config_file=Path(config_name), gpu=-1)

def extract_tensorboard_scalars(logdir):
    """Extract all scalar metrics from TensorBoard event files in a directory tree.

    Recursively walks the given directory and all subdirectories looking for
    TensorBoard event files (files starting with 'events.out.tfevents'). For
    each event file found, it loads all scalar tags and their values.

    Parameters
    ----------
    logdir : str or Path
        Root directory to search for TensorBoard event files. This is typically
        the NeuralHydrology run folder which contains subdirectories like
        'train/' and 'valid/' each with their own event files.

    Returns
    -------
    dict[str, list[tuple[int, float]]]
        Dictionary mapping tag names (e.g. 'valid/mean_nse_1d', 'train/avg_loss')
        to a list of (step, value) tuples, where step is the epoch number and
        value is the scalar metric value at that epoch.
    """
    scalars = {}

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Walk logdir and all subdirectories to find every event file
    for root, dirs, files in os.walk(logdir):
        event_files = [f for f in files if f.startswith('events.out.tfevents')]
        if event_files:
            event_acc = EventAccumulator(root)
            event_acc.Reload()
            for tag in event_acc.Tags().get('scalars', []):
                scalars[tag] = [(e.step, e.value) for e in event_acc.Scalars(tag)]

    return scalars

def find_tag(data, pattern):
    """Find a TensorBoard tag matching a pattern using case-insensitive comparison.

    Iterates over the keys of the scalar data dictionary and returns the first
    tag whose lowercase form exactly matches the lowercase pattern.

    Parameters
    ----------
    data : dict[str, list[tuple[int, float]]]
        Dictionary of TensorBoard scalars as returned by
        extract_tensorboard_scalars().
    pattern : str
        The tag name to search for (e.g. 'valid/mean_nse_1d'). Matching is
        case-insensitive.

    Returns
    -------
    str
        The original (case-preserved) tag name from the data dictionary.

    Raises
    ------
    KeyError
        If no tag in the data matches the pattern.
    """
    pattern_lower = pattern.lower()
    for tag in data.keys():
        if tag.lower() == pattern_lower:
            return tag
    raise KeyError(f"No tag matching '{pattern}' found. Available tags: {list(data.keys())}")


def load_validated_tensorboard_scalars(run_folder):
    """Extract TensorBoard scalars and verify that validation metrics are present.

    Calls extract_tensorboard_scalars() and then checks that at least one tag
    contains the substring 'valid'. This guards against cases where the model
    training completed but validation was not performed (e.g. due to
    misconfiguration of validate_every or log_tensorboard in the config).

    Parameters
    ----------
    run_folder : str or Path
        The NeuralHydrology run folder containing TensorBoard event files.

    Returns
    -------
    dict[str, list[tuple[int, float]]]
        Dictionary of TensorBoard scalars (same format as
        extract_tensorboard_scalars()), guaranteed to contain at least one
        validation tag.

    Raises
    ------
    RuntimeError
        If no tags containing 'valid' are found in the extracted scalars.
    """
    data = extract_tensorboard_scalars(run_folder)
    if any('valid' in tag for tag in data):
        return data
    raise RuntimeError(
        f"No validation TensorBoard tags found in {run_folder}. "
        f"Available tags: {list(data.keys())}. "
        "Check that NeuralHydrology is configured to log validation metrics "
        "(config: log_tensorboard: true, validate_every: 1)."
    )


def log_tensorboard_metrics_to_mlflow(data, run_folder=None):
    """Log all relevant training and validation metrics from TensorBoard to MLflow.

    Extracts the following metrics from the TensorBoard scalar data and logs
    them to the active MLflow run:
    - val_nse_1d: mean NSE at 1-day resolution per epoch
    - val_nse_1h: mean NSE at 1-hour resolution per epoch
    - val_nse_1h_1d: average of the 1d and 1h NSE per epoch
    - val_median_nse_1d: median NSE at 1-day resolution per epoch
    - val_median_nse_1h: median NSE at 1-hour resolution per epoch
    - val_loss: validation loss per epoch
    - train_loss: training loss per epoch
    - max_validation_nse_1d_1h: best mean NSE (average of 1d and 1h) across all epochs
    - epoch_largest_NSE: epoch index at which the best NSE was achieved

    Parameters
    ----------
    data : dict[str, list[tuple[int, float]]]
        Dictionary of TensorBoard scalars as returned by
        extract_tensorboard_scalars() or load_validated_tensorboard_scalars().
        Must contain the tags: valid/mean_nse_1d, valid/mean_nse_1h,
        valid/median_nse_1d, valid/median_nse_1h, valid/avg_loss,
        train/avg_loss.
    run_folder : str or Path, optional
        If provided, logs the run folder path as an MLflow parameter
        ('nh_run_folder').

    Returns
    -------
    float
        The maximum validation NSE score, computed as the highest epoch-wise
        average of the 1-day and 1-hour mean NSE scores.
    """
    tag_nse_1d_valid = find_tag(data, f'valid/mean_nse_1d')
    tag_nse_1h_valid = find_tag(data, f'valid/mean_nse_1h')
    tag_median_nse_1d_valid = find_tag(data, f'valid/median_nse_1d')
    tag_median_nse_1h_valid = find_tag(data, f'valid/median_nse_1h')
    tag_loss_valid = find_tag(data, 'valid/avg_loss')
    tag_loss_train = find_tag(data, 'train/avg_loss')

    validation_NSE_scores_1d = np.array([loss for epoch, loss in data[tag_nse_1d_valid]])
    validation_NSE_scores_1h = np.array([loss for epoch, loss in data[tag_nse_1h_valid]])
    validation_NSE_scores_mean_1d_1h = (validation_NSE_scores_1d + validation_NSE_scores_1h) / 2

    max_validation_NSE_score = float(np.max(validation_NSE_scores_mean_1d_1h))

    for (epoch_nse_1d, loss_nse_1d), (epoch_nse_1h, loss_nse_1h) in zip(
        data[tag_nse_1d_valid],
        data[tag_nse_1h_valid],
    ):
        mlflow.log_metric("val_nse_1d", float(loss_nse_1d), step=int(epoch_nse_1d))
        mlflow.log_metric("val_nse_1h", float(loss_nse_1h), step=int(epoch_nse_1h))
        mlflow.log_metric("val_nse_1h_1d", (float(loss_nse_1d) + float(loss_nse_1h)) / 2, step=int(epoch_nse_1h))

    for (epoch_med_1d, med_nse_1d), (epoch_med_1h, med_nse_1h) in zip(
        data[tag_median_nse_1d_valid],
        data[tag_median_nse_1h_valid],
    ):
        mlflow.log_metric("val_median_nse_1d", float(med_nse_1d), step=int(epoch_med_1d))
        mlflow.log_metric("val_median_nse_1h", float(med_nse_1h), step=int(epoch_med_1h))

    for epoch_val_loss, val_loss in data[tag_loss_valid]:
        mlflow.log_metric("val_loss", float(val_loss), step=int(epoch_val_loss))

    for epoch_train_loss, train_loss in data[tag_loss_train]:
        mlflow.log_metric("train_loss", float(train_loss), step=int(epoch_train_loss))

    mlflow.log_metric("max_validation_nse_1d_1h", max_validation_NSE_score)
    mlflow.log_metric("epoch_largest_NSE", int(np.argmax(validation_NSE_scores_mean_1d_1h)))

    if run_folder is not None:
        mlflow.log_param("nh_run_folder", str(run_folder))

    best_epoch = int(np.argmax(validation_NSE_scores_mean_1d_1h)) + 1
    return max_validation_NSE_score, best_epoch