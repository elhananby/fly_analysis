from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from helpers import _calculate_mean_and_std


def plot_trajectory(df: pd.DataFrame, ax: plt.Axes = None, **kwargs):
    """
    Plot the trajectory of a fly.

    Parameters:
        df (pd.DataFrame): DataFrame containing the trajectory.
        ax (plt.Axes): Axes to plot on. If None, a new figure is created.
        **kwargs: Keyword arguments to pass to plt.plot.

    Returns:
        plt.Axes: Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(df.x, df.y, **kwargs)

    # plot first point of trajectory
    ax.plot(df.x.iloc[0], df.y.iloc[0], "ro")
    return ax


def plot_mean_and_std(arr: np.ndarray, ax: plt.Axes = None, **kwargs):
    """
    Plot the mean and standard deviation of an array.

    Parameters:
        arr (np.ndarray): Array to plot.
        ax (plt.Axes): Axes to plot on. If None, a new figure is created.
        **kwargs: Keyword arguments to pass to plt.plot.

    Returns:
        plt.Axes: Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    mean, std = _calculate_mean_and_std(arr)
    X = np.arange(len(mean))
    ax.plot(X, mean, **kwargs)
    ax.fill_between(X, mean - std, mean + std, alpha=0.5)
    return ax
