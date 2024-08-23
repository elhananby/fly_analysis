import pandas as pd
import numpy as np
from typing import List, Tuple, Callable


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
        dist = np.linalg.norm(grp[['x', 'y', 'z']].diff().values, axis=1).sum()
        return dist
    
    good_obj_ids = []

    for obj_id, grp in df.groupby("obj_id"):
        dist = _distance(grp)
        if dist > threshold:
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
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
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