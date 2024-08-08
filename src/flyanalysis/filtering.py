import pandas as pd
import numpy as np
from typing import List, Tuple, Callable
from .braidz import read_braidz

def filter_by_distance(df: pd.DataFrame, threshold: float = 0.5) -> List[int]:
    """
    Calculate the total distance traveled by each object in the DataFrame grouped by 'obj_id'.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the positional data of objects.
        threshold (float, optional): The minimum distance threshold to filter objects. Defaults to 0.5.
    
    Returns:
        List[int]: A list of object IDs that have a total distance greater than the threshold.
    """
    def total_distance(group):
        return np.sqrt((group['x'].diff() ** 2 + group['y'].diff() ** 2 + group['z'].diff() ** 2).sum())
    
    distances = df.groupby("obj_id").apply(total_distance)
    return distances[distances > threshold].index.tolist()

def filter_by_duration(df: pd.DataFrame, threshold: float = 5) -> List[int]:
    """
    Calculate the duration of each object's activity based on timestamp data in the DataFrame grouped by 'obj_id'.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing timestamp data of objects.
        threshold (float, optional): The minimum duration threshold to filter objects. Defaults to 10.
    
    Returns:
        List[int]: A list of object IDs that have a duration greater than the threshold.
    """
    durations = df.groupby("obj_id")['timestamp'].apply(lambda x: x.max() - x.min())
    return durations[durations > threshold].index.tolist()

def filter_by_median_position(
    df: pd.DataFrame,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float]
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
    medians = df.groupby("obj_id")[['x', 'y', 'z']].median()
    mask = (
        (medians['x'].between(*xlim)) &
        (medians['y'].between(*ylim)) &
        (medians['z'].between(*zlim))
    )
    return medians[mask].index.tolist()

def filter_by_velocity(df: pd.DataFrame, threshold: float = 1.0) -> List[int]:
    """
    Filters a DataFrame based on the average velocity of each object.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        threshold (float, optional): The minimum average velocity threshold to filter objects. Defaults to 1.0.

    Returns:
        List[int]: A list of object IDs that have an average velocity greater than the threshold.
    """
    def avg_velocity(group):
        time_diff = group['timestamp'].diff()
        dist = np.sqrt(group['x'].diff() ** 2 + group['y'].diff() ** 2 + group['z'].diff() ** 2)
        return (dist / time_diff).mean()
    
    velocities = df.groupby("obj_id").apply(avg_velocity)
    return velocities[velocities > threshold].index.tolist()

def filter_by_acceleration(df: pd.DataFrame, threshold: float = 0.5) -> List[int]:
    """
    Filters a DataFrame based on the average acceleration of each object.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        threshold (float, optional): The minimum average acceleration threshold to filter objects. Defaults to 0.5.

    Returns:
        List[int]: A list of object IDs that have an average acceleration greater than the threshold.
    """
    def avg_acceleration(group):
        time_diff = group['timestamp'].diff()
        velocity = np.sqrt(group['x'].diff() ** 2 + group['y'].diff() ** 2 + group['z'].diff() ** 2) / time_diff
        return velocity.diff().mean()
    
    accelerations = df.groupby("obj_id").apply(avg_acceleration)
    return accelerations[accelerations > threshold].index.tolist()

def filter_by_direction_changes(df: pd.DataFrame, threshold: int = 3) -> List[int]:
    """
    Filters a DataFrame based on the number of direction changes for each object.

    Args:
        df (pd.DataFrame): The DataFrame to filter. It should have columns 'x', 'y', and 'z', representing the coordinates of the objects.
        threshold (int, optional): The minimum number of direction changes required to keep an object. Defaults to 3.

    Returns:
        List[int]: A list of object IDs that have more than the threshold number of direction changes.

    This function calculates the direction change for each object in the DataFrame by calculating the difference in 'x', 'y', and 'z' coordinates between consecutive rows. It then calculates the absolute difference in direction between consecutive rows and counts the number of times the absolute difference exceeds pi/4. Finally, it filters the DataFrame based on the number of direction changes and returns a list of object IDs that satisfy the filter condition.
    """
    def count_direction_changes(group):
        dx = group['x'].diff()
        dy = group['y'].diff()
        dz = group['z'].diff()
        direction = np.arctan2(np.sqrt(dx**2 + dy**2), dz)
        return (direction.diff().abs() > np.pi/4).sum()
    
    direction_changes = df.groupby("obj_id").apply(count_direction_changes)
    return direction_changes[direction_changes > threshold].index.tolist()


def apply_filters(df: pd.DataFrame, filters: List[Callable], *args) -> Tuple[pd.DataFrame, List[int]]:
    """
    Apply a list of filters to a DataFrame and return a filtered DataFrame and a list of object IDs that pass the filters.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        filters (List[Callable]): A list of filter functions to apply to the DataFrame. Each filter function should take a DataFrame and any additional arguments and return a list of object IDs that pass the filter.
        *args: Additional arguments to pass to the filter functions.

    Returns:
        Tuple[pd.DataFrame, List[int]]: A tuple containing the filtered DataFrame and a list of object IDs that pass the filters.

    """
    filtered_ids = set(df['obj_id'].unique())
    for filter_func, filter_args in zip(filters, args):
        filtered_ids &= set(filter_func(df, *filter_args))
    good_obj_ids = list(filtered_ids)
    filtered_df = df[df['obj_id'].isin(good_obj_ids)]
    return filtered_df, good_obj_ids


if __name__ == "__main__":
    df, csvs = read_braidz("20230906_155507.braidz")

    # Assume df is already loaded
    print("Original DataFrame shape:", df.shape)
    print("Number of unique objects:", df['obj_id'].nunique())
    
    # Example of using each filter separately
    print("\nUsing filters separately:")
    
    distance_filtered = filter_by_distance(df, threshold=0.5)
    print(f"Objects passing distance filter: {len(distance_filtered)}")
    
    duration_filtered = filter_by_duration(df, threshold=10)
    print(f"Objects passing duration filter: {len(duration_filtered)}")
    
    position_filtered = filter_by_median_position(df, xlim=(0, 10), ylim=(0, 10), zlim=(0, 10))
    print(f"Objects passing median position filter: {len(position_filtered)}")
    
    velocity_filtered = filter_by_velocity(df, threshold=1.0)
    print(f"Objects passing velocity filter: {len(velocity_filtered)}")
    
    acceleration_filtered = filter_by_acceleration(df, threshold=0.5)
    print(f"Objects passing acceleration filter: {len(acceleration_filtered)}")
    
    direction_filtered = filter_by_direction_changes(df, threshold=3)
    print(f"Objects passing direction changes filter: {len(direction_filtered)}")
    
    # Example of using apply_filters to combine multiple filters
    print("\nUsing apply_filters to combine multiple filters:")
    
    filtered_df, good_obj_ids = apply_filters(
        df,
        [filter_by_distance, filter_by_duration, filter_by_median_position, filter_by_velocity],
        [0.5],  # args for filter_by_distance
        [10],   # args for filter_by_duration
        [(0, 10), (0, 10), (0, 10)],  # args for filter_by_median_position
        [1.0]   # args for filter_by_velocity
    )
    
    print(f"Number of objects that passed all filters: {len(good_obj_ids)}")
    print(f"Shape of filtered DataFrame: {filtered_df.shape}")
    print("Object IDs that passed all filters:", good_obj_ids[:10], "..." if len(good_obj_ids) > 10 else "")
    
    # Example of chaining filters manually
    print("\nChaining filters manually:")
    
    manual_filtered_ids = set(df['obj_id'].unique())
    manual_filtered_ids &= set(filter_by_distance(df, 0.5))
    manual_filtered_ids &= set(filter_by_duration(df, 10))
    manual_filtered_ids &= set(filter_by_median_position(df, (0, 10), (0, 10), (0, 10)))
    manual_filtered_ids &= set(filter_by_velocity(df, 1.0))
    
    manual_filtered_df = df[df['obj_id'].isin(manual_filtered_ids)]
    
    print(f"Number of objects after manual filtering: {len(manual_filtered_ids)}")
    print(f"Shape of manually filtered DataFrame: {manual_filtered_df.shape}")
    
    # Verify that manual filtering and apply_filters give the same result
    assert set(good_obj_ids) == manual_filtered_ids, "Manual filtering and apply_filters give different results"
    print("Manual filtering and apply_filters give the same result.")
