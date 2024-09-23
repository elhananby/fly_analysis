import numpy as np
import pandas as pd

# from pybind11_rdp import rdp
from typing import Union
from scipy.signal import find_peaks
from .helpers import sg_smooth, circular_median, angdiff, unwrap_ignore_nan
from scipy.signal import savgol_filter
from pynumdiff.smooth_finite_difference import butterdiff

def time(df: pd.DataFrame) -> float:
    """
    Calculate the total time of the trajectory.

    Parameters:
        df (pd.DataFrame): Input dataframe with 't' column.

    Returns:
        float: Total time of the trajectory.
    """
    return df.timestamp.iloc[-1] - df.timestamp.iloc[0]


def distance(df: pd.DataFrame, axes: str = "xyz") -> float:
    """
    Calculate the total distance of the trajectory.

    Parameters:
        df (pd.DataFrame): Input dataframe with 'x', 'y', and 'z' columns.
        axes (str): Axes to consider for distance calculation. Default is 'xyz'.

    Returns:
        float: Total distance of the trajectory.
    """
    return sum(
        (
            (df.x.values[1:] - df.x.values[:-1]) ** 2
            + (df.y.values[1:] - df.y.values[:-1]) ** 2
            + (df.z.values[1:] - df.z.values[:-1]) ** 2
        )
        ** (1 / 2)
    )


def get_angular_velocity(
    df: pd.DataFrame, dt: float = 0.01, degrees: bool = True, smooth: bool = True,
) -> np.ndarray:
    """
    Calculate the angular velocity of the trajectory.

    Parameters:
        df (pd.DataFrame): Input dataframe with 'xvel' and 'yvel' columns.
        dt (float): Time step size. Default is 0.01.
        degrees (bool): If True, the result is converted to degrees. Default is True.

    Returns:
        np.ndarray: Angular velocity of the trajectory.
    """


    thetas = np.arctan2(df.yvel.values, df.xvel.values)
    thetas_u = unwrap_ignore_nan(thetas)
    if smooth:
        angular_velocity = np.copy(thetas_u)
        _, angular_velocity[~np.isnan(angular_velocity)] = butterdiff(angular_velocity[~np.isnan(angular_velocity)], dt, [1, 0.1])
    else:
        angular_velocity = np.gradient(thetas_u, dt)
    return np.rad2deg(angular_velocity) if degrees else angular_velocity

def get_linear_velocity(
    df: pd.DataFrame, dt: float = 0.01, axes: str = "xy"
) -> np.ndarray:
    """
    Calculate the linear velocity of the trajectory.

    Parameters:
        df (pd.DataFrame): Input dataframe with 'xvel' and 'yvel' columns.
        dt (float): Time step size. Default is 0.01.
        axes (str): Axes to consider for velocity calculation. Default is 'xy'.

    Returns:
        np.ndarray: Linear velocity of the trajectory.
    """
    pos = df[[ax + "vel" for ax in axes]].values
    return np.sqrt((pos**2).sum(axis=1)) / dt


def detect_saccades(
    df: pd.DataFrame, height: float = 500, distance: int = 10
) -> np.ndarray:
    """
    Detect saccades in the trajectory.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        height (float): Minimum height of peaks. Default is 500.
        distance (int): Minimum number of samples separating peaks. Default is 10.

    Returns:
        np.ndarray: Indices of detected saccades.
    """
    angvel = get_angular_velocity(df)

    neg_sac_idx, _ = find_peaks(-angvel, height=height, distance=distance)
    pos_sac_idx, _ = find_peaks(angvel, height=height, distance=distance)

    return np.concatenate((neg_sac_idx, pos_sac_idx)), angvel


def get_turn_angle(
    df: pd.DataFrame,
    idx: Union[np.ndarray, list, None] = None,
    axes: str = "xy",
    degrees: bool = True,
) -> np.ndarray:
    """
    Calculate the turn angle at each point in the trajectory.

    Parameters:
        df (pd.DataFrame): Input dataframe with 'x' and 'y' columns.
        idx (np.ndarray | list | None): Indices to consider for angle calculation.
            If None, all indices are considered. Default is None.
        axes (str): Axes to consider for angle calculation. Default is 'xy'.
        degrees (bool): If True, the result is converted to degrees. Default is True.

    Returns:
        np.ndarray: Turn angles at each point in the trajectory.
    """
    if idx is None:
        idx = np.arange(len(df))
    angles = np.zeros((len(idx),))

    pos = df.loc[:, [ax for ax in axes]].to_numpy()

    for i in range(1, len(idx) - 1):
        p1 = pos[idx[i - 1], :]
        p2 = pos[idx[i], :]
        p3 = pos[idx[i + 1], :]

        v1 = p1 - p2
        v2 = p3 - p2

        angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))

        if degrees:
            angles[i] = np.rad2deg(angle)
        else:
            angles[i] = angle

    return angles

def extract_flying_sections(df, z_min=0.05, z_max=0.25, r_max=0.23, min_flight_duration=100):
    """
    Extracts sections of the trajectory where the fly is flying.
    
    Args:
    df (pd.DataFrame): DataFrame containing trajectory data for a single obj_id.
    z_min (float): Minimum z-coordinate to be considered flying (default: 0.05).
    z_max (float): Maximum z-coordinate to be considered flying (default: 0.3).
    r_max (float): Maximum radius to be considered inside the arena (default: 0.24).
    min_flight_duration (int): Minimum number of consecutive frames to be considered a flight (default: 10).
    
    Returns:
    list: List of DataFrames, each containing a section of flight.
    """
    # Calculate radius
    df['r'] = np.sqrt(df['x']**2 + df['y']**2)
    
    # Create a boolean mask for flying conditions
    flying_mask = (df['z'] > z_min) & (df['z'] < z_max) & (df['r'] < r_max) & (df['x'] < r_max) & (df['y'] < r_max)
    
    # Find the start and end indices of flying sections
    flying_changes = flying_mask.astype(int).diff()
    flight_starts = df.index[flying_changes == 1].tolist()
    flight_ends = df.index[flying_changes == -1].tolist()
    
    # Handle edge cases
    if flying_mask.iloc[0]:
        flight_starts.insert(0, df.index[0])
    if flying_mask.iloc[-1]:
        flight_ends.append(df.index[-1])
    
    # Extract flying sections
    flying_sections = []
    for start, end in zip(flight_starts, flight_ends):
        section = df.loc[start:end]
        if len(section) >= min_flight_duration:
            flying_sections.append(section)
    
    return flying_sections

def advanced_trajectory_filter(df, max_velocity=1.0, max_acceleration=20.0, max_angle_change=np.pi/2, window_size=5):
    """
    Apply advanced filtering to remove unrealistic movements from fly trajectories.
    
    Args:
    df (pd.DataFrame): DataFrame containing trajectory data for a single obj_id.
    max_velocity (float): Maximum allowed velocity in m/s.
    max_acceleration (float): Maximum allowed acceleration in m/s^2.
    max_angle_change (float): Maximum allowed angle change in radians.
    window_size (int): Window size for Savitzky-Golay filter.
    
    Returns:
    pd.DataFrame: Filtered DataFrame with unrealistic movements removed.
    """
    # Calculate time step
    df['dt'] = df['timestamp'].diff()
    
    # Calculate velocity
    df['vx'] = df['x'].diff() / df['dt']
    df['vy'] = df['y'].diff() / df['dt']
    df['vz'] = df['z'].diff() / df['dt']
    df['velocity'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    
    # Calculate acceleration
    df['ax'] = df['vx'].diff() / df['dt']
    df['ay'] = df['vy'].diff() / df['dt']
    df['az'] = df['vz'].diff() / df['dt']
    df['acceleration'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    
    # Calculate angle changes
    df['angle'] = np.arctan2(df['vy'], df['vx'])
    df['angle_change'] = df['angle'].diff().abs()
    df['angle_change'] = df['angle_change'].apply(lambda x: min(x, 2*np.pi - x))  # Handle circular nature of angles
    
    # Apply Savitzky-Golay filter to smooth the trajectory
    df['x_smooth'] = savgol_filter(df['x'], window_size, 3)
    df['y_smooth'] = savgol_filter(df['y'], window_size, 3)
    df['z_smooth'] = savgol_filter(df['z'], window_size, 3)
    
    # Create a mask for realistic movements
    realistic_mask = (
        (df['velocity'] <= max_velocity) &
        (df['acceleration'] <= max_acceleration) &
        (df['angle_change'] <= max_angle_change)
    )
    
    # Apply the mask and return the filtered DataFrame
    df_filtered = df[realistic_mask].copy()
    
    # Update positions with smoothed values
    df_filtered['x'] = df_filtered['x_smooth']
    df_filtered['y'] = df_filtered['y_smooth']
    df_filtered['z'] = df_filtered['z_smooth']
    
    # Drop temporary columns
    columns_to_drop = ['dt', 'vx', 'vy', 'vz', 'velocity', 'ax', 'ay', 'az', 'acceleration', 'angle', 'angle_change', 'x_smooth', 'y_smooth', 'z_smooth']
    df_filtered = df_filtered.drop(columns=columns_to_drop)
    
    return df_filtered

def heading_direction_diff(
    pos: Union[np.ndarray, pd.DataFrame], origin: int = 50, end: int = 80, n: int = 1
) -> np.ndarray:
    """
    Calculate the difference in heading direction between two points in a trajectory.

    Parameters:
        pos (Union[np.ndarray, pd.DataFrame]): The position data, either as a numpy array or a pandas DataFrame.
        origin (int, optional): The index of the origin point. Defaults to 50.
        end (int, optional): The index of the end point. Defaults to 80.
        n (int, optional): The number of steps to look back/forward. Defaults to 1.

    Returns:
        np.ndarray: The difference in heading direction in degrees.
    """
    from spatialmath.base import angdiff

    if isinstance(pos, pd.DataFrame):
        pos = pos[["x", "y"]].to_numpy()

    p1 = np.arctan2(
        pos[origin, 1] - pos[origin - n, 1], pos[origin, 0] - pos[origin - n, 0]
    )
    p2 = np.arctan2(pos[end + n, 1] - pos[end, 1], pos[end + n, 0] - pos[end, 0])

    return np.rad2deg(angdiff(p1, p2))


def smooth_columns(
    df: pd.DataFrame, columns: list = ["x", "y", "z", "xvel", "yvel", "zvel"], **kwargs
) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter to the input columns.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    columns (list): Columns to smooth. Default is ['x', 'y', 'z', 'xvel', 'yvel', 'zvel'].

    Returns:
    pd.DataFrame: Smoothed dataframe.
    """
    df_copy = df.copy()
    arrs = df[columns].to_numpy()
    smoothed_arrs = np.apply_along_axis(sg_smooth, 0, arrs, **kwargs)
    df_copy[columns + "_raw"] = df[columns]
    df_copy[columns] = smoothed_arrs

    return df_copy


def mGSD(trajectory, delta=5, threshold=0.001):
    """
    Modified Geometric Saccade Detection Algorithm

    Parameters:
    trajectory: numpy array of shape (n, 2) where n is the number of frames
                and each row is [x, y] position
    delta: time step in frames (default 5, which is 50ms at 100Hz)
    threshold: threshold for saccade detection (default 0.001)

    Returns:
    saccades: list of frame indices where saccades were detected
    """

    n = len(trajectory)
    scores = np.zeros(n)

    for k in range(delta, n - delta):
        # Redefine coordinate system
        centered = trajectory - trajectory[k]

        # Calculate angles for before and after intervals
        before = centered[k - delta : k]
        after = centered[k + 1 : k + delta + 1]

        theta_before = circular_median(np.arctan2(before[:, 1], before[:, 0]))
        theta_after = circular_median(np.arctan2(after[:, 1], after[:, 0]))

        # Calculate amplitude score
        A = np.abs(angdiff(theta_after, theta_before))

        # Calculate dispersion score
        window = centered[k - delta : k + delta + 1]
        D = np.std(np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2))

        # Calculate mGSD score
        scores[k] = A * D

    # Detect saccades
    above_threshold = scores > threshold
    saccades = []
    in_saccade = False
    start = 0

    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_saccade:
            in_saccade = True
            start = i
        elif not above_threshold[i] and in_saccade:
            if i - start > 5:  # Only count as saccade if it lasts more than 5 frames
                saccades.append(start + (i - start) // 2)  # Median position
            in_saccade = False

    return saccades
