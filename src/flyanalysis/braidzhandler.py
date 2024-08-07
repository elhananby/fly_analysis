import numpy as np
import pandas as pd
from pybind11_rdp import rdp
from typing import Union
from scipy.signal import savgol_filter, find_peaks
from helpers import sg_smooth
import os
import zipfile


class BraidzHandler:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._validate()
        self._open_file()

    def _validate(self):
        # check if filename exists and if it has the right extension (.braidz)
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} not found.")
        if not self.filename.endswith(".braidz"):
            raise ValueError(f"File {self.filename} is not a .braidz file.")

    def _open_file(self) -> None:
        # check if file exists and extensions in .braidz
        self._validate()

        # open the file for reading
        with zipfile.ZipFile(file=self.filename, mode="r") as archive:
            try:
                self.df = pd.read_csv(
                    archive.open("kalman_estimates.csv.gz"),
                    comment="#",
                    compression="gzip",
                )
            except pd.errors.EmptyDataError:
                raise ValueError(
                    f"File {self.filename} does not contain 'kalman_estimates.csv.gz'."
                )

            # check if any other csv files exist in the braidz file, and load them
            csv_files = [csv for csv in archive.namelist() if csv.endswith(".csv")]
            self.csvs = {}
            for csv_file in csv_files:
                key, _ = os.path.splitext(csv_file)
                try:
                    self.csvs[key] = pd.read_csv(archive.open(csv_file))
                except pd.errors.EmptyDataError:
                    continue

    def time(self):
        return self.df.timestamp.iloc[-1] - self.df.timestamp.iloc[0]

    def distance(self):
        return sum(
            (
                (self.df.x.values[1:] - self.df.x.values[:-1]) ** 2
                + (self.df.y.values[1:] - self.df.y.values[:-1]) ** 2
                + (self.df.z.values[1:] - self.df.z.values[:-1]) ** 2
            )
            ** (1 / 2)
        )

    def linear_velocity(self, dt: float = 0.01, axes: str = "xy") -> np.ndarray:
        """
        Calculate the linear velocity of the trajectory.

        Parameters:
            df (pd.DataFrame): Input dataframe with 'xvel' and 'yvel' columns.
            dt (float): Time step size. Default is 0.01.
            axes (str): Axes to consider for velocity calculation. Default is 'xy'.

        Returns:
            np.ndarray: Linear velocity of the trajectory.
        """
        pos = self.df[[ax + "vel" for ax in axes]].values
        return np.sqrt((pos**2).sum(axis=1)) / dt

    def angular_velocity(self, dt: float = 0.01, degrees: bool = True) -> np.ndarray:
        """
        Calculate the angular velocity of the trajectory.

        Parameters:
            df (pd.DataFrame): Input dataframe with 'xvel' and 'yvel' columns.
            dt (float): Time step size. Default is 0.01.
            degrees (bool): If True, the result is converted to degrees. Default is True.

        Returns:
            np.ndarray: Angular velocity of the trajectory.
        """
        thetas = np.arctan2(self.df.yvel.values, self.df.xvel.values)
        thetas_u = np.unwrap(thetas)
        angular_velocity = np.gradient(thetas_u) / (1 * dt)
        return np.rad2deg(angular_velocity) if degrees else angular_velocity

    def detect_saccades(self, height: float = 500, distance: int = 10) -> np.ndarray:
        """
        Detect saccades in the trajectory.

        Parameters:
            df (pd.DataFrame): Input dataframe.
            height (float): Minimum height of peaks. Default is 500.
            distance (int): Minimum number of samples separating peaks. Default is 10.

        Returns:
            np.ndarray: Indices of detected saccades.
        """
        angvel = get_angular_velocity(self.df)

        neg_sac_idx, _ = find_peaks(-angvel, height=height, distance=distance)
        pos_sac_idx, _ = find_peaks(angvel, height=height, distance=distance)

        return np.concatenate((neg_sac_idx, pos_sac_idx))

    def turn_angle(
        self,
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
            idx = np.arange(len(self.df))
        angles = np.zeros((len(idx),))

        pos = self.df.loc[:, [ax for ax in axes]].to_numpy()

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

    def simplify_trajectory(self, epsilon: float = 0.001) -> np.ndarray:
        """
        Simplify the trajectory using Ramer-Douglas-Peucker algorithm.

        Parameters:
            df (pd.DataFrame): Input dataframe with 'x' and 'y' columns.
            epsilon (float): The maximum permissible deviation from the line.
                Points with greater deviation will be included in the output.

        Returns:
            np.array: Indices of the simplified trajectory.
        """
        pos = self.df.loc[:, ["x", "y"]].to_numpy()
        simplified = rdp(pos, epsilon=epsilon, return_mask=True)
        return np.where(simplified)[0]

    def heading_direction_diff(
        self,
        origin: int = 50,
        end: int = 80,
        n: int = 1,
    ) -> np.ndarray:
        from spatialmath.base import angdiff

        pos = self.df[["x", "y"]].to_numpy()

        p1 = np.arctan2(
            pos[origin, 1] - pos[origin - n, 1], pos[origin, 0] - pos[origin - n, 0]
        )
        p2 = np.arctan2(pos[end + n, 1] - pos[end, 1], pos[end + n, 0] - pos[end, 0])

        return np.rad2deg(angdiff(p1, p2))

    def smooth_columns(
        self,
        columns: list = ["x", "y", "z", "xvel", "yvel", "zvel"],
        **kwargs,
    ):
        """
        Apply Savitzky-Golay filter to the input columns.

        Parameters:
        df (pd.DataFrame): Input dataframe.
        columns (list): Columns to smooth. Default is ['x', 'y', 'z', 'xvel', 'yvel', 'zvel'].

        Returns:
        pd.DataFrame: Smoothed dataframe.
        """
        for col in columns:
            self.df[f"{col}_raw"] = self.df[col].copy()
            self.df[col] = sg_smooth(self.df[col].to_numpy(), **kwargs)


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
    df: pd.DataFrame, dt: float = 0.01, degrees: bool = True
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
    thetas_u = np.unwrap(thetas)
    angular_velocity = np.gradient(thetas_u) / (1 * dt)
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

    return np.concatenate((neg_sac_idx, pos_sac_idx))


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


def get_simplified_trajectory(df: pd.DataFrame, epsilon: float = 0.001) -> np.ndarray:
    """
    Simplify the trajectory using Ramer-Douglas-Peucker algorithm.

    Parameters:
        df (pd.DataFrame): Input dataframe with 'x' and 'y' columns.
        epsilon (float): The maximum permissible deviation from the line.
            Points with greater deviation will be included in the output.

    Returns:
        np.array: Indices of the simplified trajectory.
    """
    pos = df.loc[:, ["x", "y"]].to_numpy()
    simplified = rdp(pos, epsilon=epsilon, return_mask=True)
    return np.where(simplified)[0]


def heading_direction_diff(
    pos: Union[np.ndarray, pd.DataFrame], origin: int = 50, end: int = 80, n: int = 1
) -> np.ndarray:
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
    for col in columns:
        df_copy[f"{col}_raw"] = df[col].copy()
        df_copy[col] = sg_smooth(df[col].to_numpy(), **kwargs)

    return df_copy
