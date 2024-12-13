from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from numba import jit


def filter_by_distance(df: pd.DataFrame, threshold: float = 0.5) -> List[int]:
    """
    Filters objects in a DataFrame based on the total distance traveled.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the object data.
    threshold (float, optional): The minimum distance required for an object to be included. Defaults to 0.5.

    Returns:
    List[int]: A list of object IDs that meet the distance threshold.
    """

    def _distance(grp: pd.DataFrame) -> float:
        dist = np.linalg.norm(grp[["x", "y", "z"]].diff().values, axis=1).sum()
        return dist

    good_obj_ids = []

    for obj_id, grp in df.groupby("obj_id"):
        dist = _distance(grp)
        if dist > threshold:
            good_obj_ids.append(obj_id)

    return good_obj_ids


def filter_by_length(df: pd.DataFrame, threshold: int = 300) -> List[int]:
    """
    Filters objects in a DataFrame based on the length of the trajectory.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the object data.
    threshold (int, optional): The minimum length required for an object to be included. Defaults to 5.

    Returns:
    List[int]: A list of object IDs that meet the length threshold.
    """
    good_obj_ids = []
    for obj_id, grp in df.groupby("obj_id"):
        if len(grp) > threshold:
            good_obj_ids.append(obj_id)
    return good_obj_ids


def filter_by_duration(df: pd.DataFrame, threshold: float = 5) -> List[int]:
    """
    Filters objects in a DataFrame based on the duration of activity.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the object data.
    threshold (float, optional): The minimum duration required for an object to be included. Defaults to 5.

    Returns:
    List[int]: A list of object IDs that meet the duration threshold.
    """
    good_obj_ids = []
    # durations = df.groupby("obj_id")['timestamp'].apply(lambda x: x.max() - x.min())
    for obj_id, grp in df.groupby("obj_id"):
        durations = grp["timestamp"].max() - grp["timestamp"].min()
        if durations > threshold:
            good_obj_ids.append(obj_id)
    return good_obj_ids


def filter_by_median_position(
    df: pd.DataFrame,
    xlim: Tuple[float, float] = (-0.2, 0.2),
    ylim: Tuple[float, float] = (-0.2, 0.2),
    zlim: Tuple[float, float] = (0.1, 0.2),
) -> List[int]:
    """
    Filters a DataFrame based on the median position of each object.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        xlim (Tuple[float, float]): The range of x values to filter.
        ylim (Tuple[float, float]): The range of y values to filter.
        zlim (Tuple[float, float]): The range of z values to filter.

    Returns:
        List[int]: A list of object IDs that satisfy the filter conditions.
    """
    good_obj_ids = []
    for obj_id, grp in df.groupby("obj_id"):
        mx, my, mz = grp["x"].median(), grp["y"].median(), grp["z"].median()
        if (
            xlim[0] <= mx <= xlim[1]
            and ylim[0] <= my <= ylim[1]
            and zlim[0] <= mz <= zlim[1]
        ):
            good_obj_ids.append(obj_id)

    return good_obj_ids


def filter_by_velocity(df: pd.DataFrame, threshold: float = 1.0) -> List[int]:
    """
    Filters a DataFrame based on the average velocity of each object.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        threshold (float, optional): The minimum average velocity threshold to filter objects. Defaults to 1.0.

    Returns:
        List[int]: A list of object IDs that have an average velocity greater than the threshold.
    """
    good_obj_ids = []

    for obj_id, grp in df.groupby("obj_id"):
        linear_velocity = np.sqrt(
            grp["xvel"].to_numpy() ** 2
            + grp["yvel"].to_numpy() ** 2
            + grp["zvel"].to_numpy() ** 2
        )

        if np.mean(linear_velocity) > threshold:
            good_obj_ids.append(obj_id)

    return good_obj_ids


def filter_by_acceleration(df: pd.DataFrame, threshold: float = 0.5) -> List[int]:
    """
    Filters a DataFrame based on the average acceleration of each object.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        threshold (float, optional): The minimum average acceleration threshold to filter objects. Defaults to 0.5.

    Returns:
        List[int]: A list of object IDs that have an average acceleration greater than the threshold.
    """
    good_obj_ids = []

    for obj_id, grp in df.groupby("obj_id"):
        xacc = np.diff(grp["xvel"].to_numpy())
        yacc = np.diff(grp["yvel"].to_numpy())
        zacc = np.diff(grp["zvel"].to_numpy())
        linear_acceleration = np.sqrt(xacc**2 + yacc**2 + zacc**2)

        if np.mean(linear_acceleration) > threshold:
            good_obj_ids.append(obj_id)
    return good_obj_ids


def apply_filters(
    df: pd.DataFrame, filters: List[Callable], *args
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Apply a list of filters to a DataFrame and return a filtered DataFrame and a list of object IDs that pass the filters.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        filters (List[Callable]): A list of filter functions to apply to the DataFrame. Each filter function should take a DataFrame and any additional arguments and return a list of object IDs that pass the filter.
        *args: Additional arguments to pass to the filter functions.

    Returns:
        Tuple[pd.DataFrame, List[int]]: A tuple containing the filtered DataFrame and a list of object IDs that pass the filters.

    """
    filtered_ids = set(df["obj_id"].unique())
    for filter_func, filter_args in zip(filters, args):
        filtered_ids &= set(filter_func(df, *filter_args))
    good_obj_ids = list(filtered_ids)
    filtered_df = df[df["obj_id"].isin(good_obj_ids)]
    return filtered_df, good_obj_ids


@jit(nopython=True)
def _calculate_distance(positions: np.ndarray) -> float:
    """Calculate total distance traveled from position differences."""
    diffs = positions[1:] - positions[:-1]
    distances = np.sqrt(np.sum(diffs * diffs, axis=1))
    return np.sum(distances)


@jit(nopython=True)
def _get_group_data(
    data: np.ndarray, obj_ids: np.ndarray, target_id: int
) -> np.ndarray:
    """Extract data for a specific object ID."""
    mask = data[:, 0] == target_id
    return data[mask]


@jit(nopython=True)
def _calculate_median(arr: np.ndarray) -> float:
    """Calculate median of array."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_arr[mid - 1] + sorted_arr[mid]) / 2
    return sorted_arr[mid]


@jit(nopython=True)
def filter_by_distance_numba(data: np.ndarray, threshold: float = 0.5) -> List[int]:
    """
    Filter objects based on total distance traveled.
    data columns: [obj_id, x, y, z, ...]
    """
    obj_ids = np.unique(data[:, 0])
    good_obj_ids = []

    for obj_id in obj_ids:
        group_data = _get_group_data(data, obj_ids, obj_id)
        positions = group_data[:, 1:4]  # x, y, z columns
        if len(positions) > 1:
            dist = _calculate_distance(positions)
            if dist > threshold:
                good_obj_ids.append(int(obj_id))

    return good_obj_ids


@jit(nopython=True)
def filter_by_length_numba(data: np.ndarray, threshold: int = 300) -> List[int]:
    """
    Filter objects based on trajectory length.
    data columns: [obj_id, ...]
    """
    obj_ids = np.unique(data[:, 0])
    good_obj_ids = []

    for obj_id in obj_ids:
        group_data = _get_group_data(data, obj_ids, obj_id)
        if len(group_data) > threshold:
            good_obj_ids.append(int(obj_id))

    return good_obj_ids


@jit(nopython=True)
def filter_by_duration_numba(data: np.ndarray, threshold: float = 5.0) -> List[int]:
    """
    Filter objects based on duration.
    data columns: [obj_id, x, y, z, timestamp, ...]
    """
    obj_ids = np.unique(data[:, 0])
    good_obj_ids = []
    timestamp_col = 4  # Assuming timestamp is the 5th column

    for obj_id in obj_ids:
        group_data = _get_group_data(data, obj_ids, obj_id)
        timestamps = group_data[:, timestamp_col]
        duration = np.max(timestamps) - np.min(timestamps)
        if duration > threshold:
            good_obj_ids.append(int(obj_id))

    return good_obj_ids


@jit(nopython=True)
def filter_by_median_position_numba(
    data: np.ndarray,
    xlim: Tuple[float, float] = (-0.2, 0.2),
    ylim: Tuple[float, float] = (-0.2, 0.2),
    zlim: Tuple[float, float] = (0.1, 0.2),
) -> List[int]:
    """
    Filter objects based on median position.
    data columns: [obj_id, x, y, z, ...]
    """
    obj_ids = np.unique(data[:, 0])
    good_obj_ids = []

    for obj_id in obj_ids:
        group_data = _get_group_data(data, obj_ids, obj_id)
        mx = _calculate_median(group_data[:, 1])  # x
        my = _calculate_median(group_data[:, 2])  # y
        mz = _calculate_median(group_data[:, 3])  # z

        if (
            xlim[0] <= mx <= xlim[1]
            and ylim[0] <= my <= ylim[1]
            and zlim[0] <= mz <= zlim[1]
        ):
            good_obj_ids.append(int(obj_id))

    return good_obj_ids


@jit(nopython=True)
def filter_by_velocity_numba(data: np.ndarray, threshold: float = 1.0) -> List[int]:
    """
    Filter objects based on average velocity.
    data columns: [obj_id, x, y, z, timestamp, xvel, yvel, zvel, ...]
    """
    obj_ids = np.unique(data[:, 0])
    good_obj_ids = []
    xvel_col, yvel_col, zvel_col = 5, 6, 7  # Assuming velocity columns positions

    for obj_id in obj_ids:
        group_data = _get_group_data(data, obj_ids, obj_id)
        velocities = np.sqrt(
            group_data[:, xvel_col] ** 2
            + group_data[:, yvel_col] ** 2
            + group_data[:, zvel_col] ** 2
        )
        if np.mean(velocities) > threshold:
            good_obj_ids.append(int(obj_id))

    return good_obj_ids


@jit(nopython=True)
def filter_by_acceleration_numba(data: np.ndarray, threshold: float = 0.5) -> List[int]:
    """
    Filter objects based on average acceleration.
    data columns: [obj_id, x, y, z, timestamp, xvel, yvel, zvel, ...]
    """
    obj_ids = np.unique(data[:, 0])
    good_obj_ids = []
    xvel_col, yvel_col, zvel_col = 5, 6, 7  # Assuming velocity columns positions

    for obj_id in obj_ids:
        group_data = _get_group_data(data, obj_ids, obj_id)
        if len(group_data) > 1:
            xacc = np.diff(group_data[:, xvel_col])
            yacc = np.diff(group_data[:, yvel_col])
            zacc = np.diff(group_data[:, zvel_col])
            accelerations = np.sqrt(xacc**2 + yacc**2 + zacc**2)
            if np.mean(accelerations) > threshold:
                good_obj_ids.append(int(obj_id))

    return good_obj_ids


def prepare_data_for_numba(df: pd.DataFrame) -> np.ndarray:
    """
    Convert pandas DataFrame to numpy array format required by the numba functions.

    Expected columns: ['obj_id', 'x', 'y', 'z', 'timestamp', 'xvel', 'yvel', 'zvel']
    """
    required_columns = ["obj_id", "x", "y", "z", "timestamp", "xvel", "yvel", "zvel"]
    return df[required_columns].values
