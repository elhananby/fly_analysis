from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from .helpers import _calculate_mean_and_std
from matplotlib import colormaps
from typing import List


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


def plot_mean_and_std(arr: np.ndarray, ax: plt.Axes = None, abs_value: bool = False, shaded_area: list = [None, None], **kwargs):
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
    mean, std = _calculate_mean_and_std(arr, abs_value)
    X = np.arange(len(mean))
    ax.plot(X, mean, **kwargs)
    ax.fill_between(X, mean - std, mean + std, alpha=0.5)
    if shaded_area[0] is not None and shaded_area[1] is not None:
        ax.axvspan(shaded_area[0], shaded_area[1], alpha=0.2, color="gray")
    return ax


def plot_dispersions(groups: List[List[np.array]], labels: List[str], ax=None):
    """
    Plots the XY parts of trajectories for multiple groups in a 2D plot.

    Args:
    - groups: A list of groups, where each group is a list of np.array trajectories (each of shape (n, 3)).
    - labels: A list of strings representing the label for each group.

    """
    if len(groups) != len(labels):
        raise ValueError("Number of groups must match the number of labels.")

    if ax is None:
        ax = plt.gca()

    # Define a set of colors for different groups
    colors = colormaps["tab10"]  # Use a colormap with sufficient distinct colors

    # Plot each group
    for i, group in enumerate(groups):
        for traj in group:
            ax.plot(traj[:, 0], traj[:, 1], color=colors(i), alpha=0.5)

        # Calculate and plot the mean trajectory for the group
        mean_traj = np.mean(np.stack(group), axis=0)
        ax.plot(
            mean_traj[:, 0],
            mean_traj[:, 1],
            color=colors(i),
            linewidth=3,
            label=labels[i],
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trajectory Dispersion Comparison")
    ax.legend()
    ax.grid(True)
