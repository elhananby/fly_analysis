import numpy as np
from scipy.signal import savgol_filter
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import pandas as pd

def _calculate_mean_and_std(arr: np.ndarray):
    """
    Calculate the mean and standard deviation of an array.

    Parameters:
        arr (np.ndarray): Input array.

    Returns:
        tuple: Mean and standard deviation of the array.
    """
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def sg_smooth(arr: np.array, **kwargs) -> np.array:
    """
    Apply Savitzky-Golay filter to the input array.

    Parameters:
    arr (np.array): Input array.
    **kwargs: Keyword arguments for scipy.signal.savgol_filter.

    Returns:
    np.array: Filtered array.
    """
    return savgol_filter(arr, **kwargs)

def create_ekf(dt: float = 0.01) -> ExtendedKalmanFilter:
    # Create Extended Kalman Filter
    ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)

    # Initial state [x, y, z, x_vel, y_vel, z_vel]
    ekf.x = np.array([0, 0, 0, 0, 0, 0])

    # State transition matrix
    ekf.F = np.array([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

    # Custom Jacobian function
    ekf.HJacobian = np.eye(3, 6)

    # Custom measurement function
    ekf.Hx = lambda x: x[:3]

    # Process noise covariance
    ekf.Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

    # Measurement noise covariance
    ekf.R = np.diag([0.1, 0.1, 0.1])

    return ekf

def smooth_trajectory(data: pd.DataFrame) -> np.array:
    # Extract time step from the data
    dt = data['timestamp'].diff().mean()
    print(dt)
    # Create Extended Kalman Filter
    ekf = create_ekf(dt)

    # Set initial state to the first point in the data
    ekf.x[:3] = data.iloc[0][['x', 'y', 'z']]

    smoothed_trajectory = []

    for _, measurement in data.iterrows():
        measurement_array = measurement[['x', 'y', 'z', 'xvel', 'yvel', 'zvel']].values

        # Manually perform the predict step
        ekf.x = np.dot(ekf.F, ekf.x)
        ekf.P = np.dot(ekf.F, np.dot(ekf.P, ekf.F.T)) + ekf.Q

        # Manually perform the update step
        y = measurement_array[:3] - ekf.Hx(ekf.x)
        S = np.dot(ekf.HJacobian, np.dot(ekf.P, ekf.HJacobian.T)) + ekf.R
        K = np.dot(np.dot(ekf.P, ekf.HJacobian.T), np.linalg.inv(S))
        ekf.x = ekf.x + np.dot(K, y)
        ekf.P = np.dot((np.eye(6) - np.dot(K, ekf.HJacobian)), ekf.P)

        # Save smoothed state
        smoothed_trajectory.append(ekf.x.copy())

    return np.array(smoothed_trajectory)