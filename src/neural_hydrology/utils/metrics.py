import numpy as np
import pandas as pd


def nse(observed: pd.Series, simulated: pd.Series) -> float:
    """Compute the Nash-Sutcliffe Efficiency (NSE) between observed and simulated values.

    Missing values are ignored by comparing only indices where both series contain
    data. An NSE of 1 indicates a perfect match, 0 indicates performance equal to
    using the mean of the observations, and values below 0 indicate worse performance.

    Args:
        observed: Observed values.
        simulated: Simulated or modeled values.

    Returns:
        The Nash-Sutcliffe Efficiency score.
    """
    mask = observed.notna() & simulated.notna()

    obs = observed[mask]
    sim = simulated[mask]

    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def pbias(observed: pd.Series, simulated: pd.Series) -> float:
    """Compute the Percent Bias (PBIAS) between observed and simulated values.

    Missing values are ignored by comparing only indices where both series contain
    data. Positive values indicate overestimation by the simulation, while
    negative values indicate underestimation. A value of 0 represents a perfect
    match in total volume.

    Args:
        observed: Observed values.
        simulated: Simulated or modeled values.

    Returns:
        The percent bias, expressed as a percentage.
    """
    mask = observed.notna() & simulated.notna()

    obs = observed[mask]
    sim = simulated[mask]

    return 100 * (sim.sum() - obs.sum()) / obs.sum()
