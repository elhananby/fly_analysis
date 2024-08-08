from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import directed_hausdorff



class TrackingDataAnalyzer:
    def __init__(self, data_source):
        """
        Initializes the TrackingDataAnalyzer object with the provided data_source.

        Parameters:
            data_source: Union[pd.DataFrame, str] - A pandas DataFrame or a file path to initialize the object.

        Raises:
            ValueError: If the input data_source is not a pandas DataFrame or a valid file path.

        Returns:
            None
        """
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        elif isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        else:
            raise ValueError("Input must be a pandas DataFrame or a file path")

        self.original_df = self.df.copy()

    def filter_by_time(self, threshold):
        """
        Filter out objects with total time less than the threshold.

        Parameters:
            threshold (int or float): The minimum time threshold in seconds.

        Returns:
            self: The instance of the TrackingDataAnalyzer class with the filtered DataFrame.

        This function calculates the total time for each object in the DataFrame grouped by 'obj_id'. It then filters out the objects with total time less than the threshold. The filtered DataFrame is assigned back to the instance variable 'df'.

        Note:
            The 'timestamp' column in the DataFrame should be in seconds.
        """
        time_per_obj = self.df.groupby("obj_id")["timestamp"].agg(["min", "max"])
        time_per_obj["total_time"] = time_per_obj["max"] - time_per_obj["min"]
        valid_objs = time_per_obj[time_per_obj["total_time"] >= threshold].index
        self.df = self.df[self.df["obj_id"].isin(valid_objs)]
        return self

    def filter_by_distance(self, threshold):
        """
        Filter out objects with total distance traveled less than the threshold.

        Parameters:
            threshold (float): The minimum distance threshold to filter objects.

        Returns:
            self: The modified TrackingDataAnalyzer object.

        This method calculates the total distance traveled by each object in the DataFrame grouped by 'obj_id'.
        It then filters out objects with total distance less than the threshold. The filtered DataFrame is stored
        in the 'df' attribute of the TrackingDataAnalyzer object.

        The 'calculate_distance' function calculates the total distance traveled by each object in a group. It
        calculates the difference in x, y, and z coordinates for each consecutive pair of rows in the group and
        calculates the square of the sum of these differences. The square root of the sum is returned as the total
        distance traveled by the object.

        The 'distances' variable stores the total distance traveled by each object. The 'valid_objs' variable
        stores the object IDs that have a total distance greater than or equal to the threshold. The 'df' attribute
        of the TrackingDataAnalyzer object is updated to only include rows with 'obj_id' values in 'valid_objs'.

        Example usage:
        ```
        analyzer = TrackingDataAnalyzer(data_source)
        filtered_analyzer = analyzer.filter_by_distance(10.0)
        ```
        """
        def calculate_distance(group):
            x, y, z = group[["x", "y", "z"]].diff().pow(2).sum().values.tolist()
            return np.sqrt(x + y + z)

        distances = self.df.groupby("obj_id").apply(calculate_distance)
        valid_objs = distances[distances >= threshold].index
        self.df = self.df[self.df["obj_id"].isin(valid_objs)]
        return self

    def calculate_linear_velocity(self):
        """
        Calculate the linear velocity for each object in the DataFrame and add it as a new column.

        This function calculates the linear velocity for each object in the DataFrame by taking the square root of the sum of the squares of the x, y, and z velocity components. The calculated linear velocity is added as a new column to the DataFrame.

        Parameters:
            self (TrackingDataAnalyzer): The instance of the TrackingDataAnalyzer class.

        Returns:
            TrackingDataAnalyzer: The updated instance of the TrackingDataAnalyzer class with the linear velocity column added.
        """
        self.df["linear_velocity"] = np.sqrt(
            self.df["xvel"] ** 2 + self.df["yvel"] ** 2 + self.df["zvel"] ** 2
        )
        return self

    def calculate_angular_velocity(self):
        """
        Calculate the angular velocity for each object in the DataFrame and add it as a new column.

        This function calculates the angular velocity for each object in the DataFrame by taking the derivative of the x, y, and z coordinates with respect to time. The angular velocity is then calculated as the ratio of the velocity to the radius of the object.

        Returns:
            self: The instance of the TrackingDataAnalyzer class.

        """
        def calc_angular_vel(group):
            dx = group["x"].diff()
            dy = group["y"].diff()
            dz = group["z"].diff()
            dt = group["timestamp"].diff()
            r = np.sqrt(group["x"] ** 2 + group["y"] ** 2 + group["z"] ** 2)
            v = np.sqrt(dx**2 + dy**2 + dz**2) / dt
            return v / r

        self.df["angular_velocity"] = (
            self.df.groupby("obj_id")
            .apply(calc_angular_vel)
            .reset_index(level=0, drop=True)
        )
        return self

    def smooth_trajectory(self, window_length=11, polyorder=3):
        """
        Apply Savitzky-Golay filter to smooth the trajectory.

        This method applies the Savitzky-Golay filter to the 'x', 'y', and 'z' columns of the DataFrame,
        smoothing the trajectory of each object. The filter is applied using the specified window length
        and polynomial order.

        Parameters:
            window_length (int, optional): The window length for the filter. Defaults to 11.
            polyorder (int, optional): The polynomial order for the filter. Defaults to 3.

        Returns:
            self: The TrackingDataAnalyzer instance with the smoothed trajectory.
        """
        for col in ["x", "y", "z"]:
            self.df[f"{col}_smooth"] = self.df.groupby("obj_id")[col].transform(
                lambda x: savgol_filter(x, window_length, polyorder)
            )
        return self

    def find_peaks(self, column, prominence=1):
        """
        Find peaks in a specified column.

        Args:
            column (str): The name of the column to find peaks in.
            prominence (float, optional): The minimum prominence of the peaks. Defaults to 1.

        Returns:
            self: The updated instance of the class.

        This function groups the data by 'obj_id' and applies the 'find_obj_peaks' function to each group.
        The 'find_obj_peaks' function finds the peaks in the specified column using the 'find_peaks' function.
        The indices of the peaks are stored in a pandas Series and returned.
        The 'is_peak' column is then updated in the dataframe to indicate which rows are peaks.
        Finally, the updated instance of the class is returned.

        Example usage:
        ```
        analyzer = TrackingDataAnalyzer(df)
        analyzer.find_peaks('column_name')
        ```
        """
        def find_obj_peaks(group):
            peaks, _ = find_peaks(group[column], prominence=prominence)
            return pd.Series(group.iloc[peaks].index, name="peak_indices")

        peak_indices = self.df.groupby("obj_id").apply(find_obj_peaks)
        self.df["is_peak"] = self.df.index.isin(peak_indices.explode())
        return self

    def plot_trajectory(self, obj_id):
        """
        Plot the trajectory of a specific object.

        Parameters:
            obj_id (int): The ID of the object to plot the trajectory for.

        Returns:
            None

        This function plots the trajectory of a specific object identified by its ID. It retrieves the data for the object from the DataFrame `self.df` and creates a scatter plot of the object's X and Y coordinates, with the color of each point representing the timestamp. The plot is displayed with a colorbar indicating the timestamp.

        Example usage:
            traj_analyzer = TrackingDataAnalyzer(df)
            traj_analyzer.plot_trajectory(123)
        """
        obj_data = self.df[self.df["obj_id"] == obj_id]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            obj_data["x"], obj_data["y"], c=obj_data["timestamp"], cmap="viridis"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Trajectory of Object {obj_id}")
        plt.colorbar(scatter, label="Timestamp")
        plt.show()

    def reset(self):
        """
        Resets the DataFrame to its original state.

        Parameters:
            None

        Returns:
            tracking_data_analyzer: The instance itself.
        """
        self.df = self.original_df.copy()
        return self

    def extract_window(
        self,
        idx: int,
        n_before: int = 50,
        n_after: int = 100,
        columns: Union[str, List[str]] = None,
        padding: bool = True,
    ) -> pd.DataFrame:
        """
        Extracts a window of data around a specific index from one or more columns.

        Parameters:
            idx (int): The central index around which to extract the window.
            n_before (int, optional): The number of rows to include before the central index. Defaults to 50.
            n_after (int, optional): The number of rows to include after the central index. Defaults to 100.
            columns (Union[str, List[str]], optional): The column(s) to extract. If None, all columns are extracted.
                Can be a single column name or a list of column names. Defaults to None.
            padding (bool, optional): If True, pad the extracted window with NaNs to ensure
                it always has length n_before + n_after + 1. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted window.
        """
        if columns is None:
            columns = self.df.columns
        elif isinstance(columns, str):
            columns = [columns]

        start_idx = max(0, idx - n_before)
        end_idx = min(len(self.df), idx + n_after + 1)

        window = self.df.loc[start_idx:end_idx, columns].copy()

        actual_before = idx - start_idx
        actual_after = end_idx - idx - 1

        window["relative_position"] = range(-actual_before, actual_after + 1)

        if padding and (len(window) < n_before + n_after + 1):
            pad_df = pd.DataFrame(
                np.nan, index=range(n_before + n_after + 1), columns=window.columns
            )
            pad_before = n_before - actual_before
            pad_df.iloc[pad_before : pad_before + len(window)] = window.values
            pad_df["relative_position"] = range(-n_before, n_after + 1)
            window = pad_df

        return window

    def animate_trajectory(self, obj_id, interval=50):
        """
        Generate an animation of the trajectory of a specific object.

        Parameters:
            obj_id (int): The unique identifier of the object.
            interval (int, optional): The time interval between frames in milliseconds. Defaults to 50.

        Returns:
            FuncAnimation: The animation object.

        This function retrieves the data for the specified object from the DataFrame and creates a matplotlib figure and axis. It then sets the x and y limits of the axis based on the minimum and maximum values of the object's x and y coordinates.

        The function defines two helper functions, `init()` and `animate(i)`, which are used to initialize and update the animation.

        The `init()` function sets the initial state of the animation by setting the data of the line object to empty lists.

        The `animate(i)` function updates the animation by setting the data of the line object to the x and y coordinates of the object up to the specified index `i`.

        The `FuncAnimation` object is created with the figure, animation function, initialization function, number of frames, interval, and blit option.

        The function returns the animation object.
        """
        obj_data = self.df[self.df["obj_id"] == obj_id].sort_values("timestamp")

        fig, ax = plt.subplots()
        ax.set_xlim(obj_data["x"].min(), obj_data["x"].max())
        ax.set_ylim(obj_data["y"].min(), obj_data["y"].max())
        (line,) = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i):
            line.set_data(obj_data["x"][:i], obj_data["y"][:i])
            return (line,)

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(obj_data),
            interval=interval,
            blit=True,
        )

        return anim

    def generate_trajectory_heatmap(self, resolution=100):
        """
        Generate a heatmap of trajectory density.

        Parameters:
            resolution (int, optional): The resolution of the heatmap grid. Defaults to 100.

        Returns:
            tuple: A tuple containing the heatmap and the figure object.
                - heatmap (ndarray): The 2D array representing the heatmap.
                - fig (Figure): The figure object containing the heatmap.

        This function calculates the heatmap of trajectory density based on the x and y coordinates of the tracking data.
        It uses the `np.histogram2d` function to calculate the histogram of x and y coordinates and then transposes the heatmap.
        The extent of the heatmap is determined by the minimum and maximum values of the x and y edges.
        The heatmap is plotted using `imshow` and a colorbar is added to the plot.
        The title, x-label, and y-label of the plot are set accordingly.
        The figure object containing the heatmap is returned along with the heatmap array.
        """
        x = self.df["x"]
        y = self.df["y"]

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=resolution)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        fig, ax = plt.subplots()
        im = ax.imshow(heatmap.T, extent=extent, origin="lower", cmap="hot")
        ax.set_title("Trajectory Density Heatmap")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="Density")

        return heatmap, fig

    def detect_stopping_points(self, speed_threshold=0.1, min_duration=5):
        """
        Detects stopping points in trajectories based on a speed threshold and minimum duration.

        Parameters:
            speed_threshold (float, optional): The speed threshold below which an object is considered stopped. Defaults to 0.1.
            min_duration (int, optional): The minimum duration (in seconds) to consider a stop. Defaults to 5.

        Returns:
            pandas.DataFrame: A DataFrame containing the detected stopping points. Each row represents a stop and includes the following columns:
                - obj_id (int): The object ID.
                - start_time (datetime): The start time of the stop.
                - end_time (datetime): The end time of the stop.
                - duration (timedelta): The duration of the stop.
                - x (float): The average x-coordinate of the object during the stop.
                - y (float): The average y-coordinate of the object during the stop.
                - z (float): The average z-coordinate of the object during the stop.
        """
        self.df["speed"] = np.sqrt(
            self.df["xvel"] ** 2 + self.df["yvel"] ** 2 + self.df["zvel"] ** 2
        )
        self.df["is_stopped"] = self.df["speed"] < speed_threshold

        stops = []
        for obj_id, group in self.df.groupby("obj_id"):
            stop_starts = group.index[
                group["is_stopped"] & ~group["is_stopped"].shift(1).fillna(False)
            ]
            stop_ends = group.index[
                group["is_stopped"] & ~group["is_stopped"].shift(-1).fillna(False)
            ]

            for start, end in zip(stop_starts, stop_ends):
                duration = group.loc[end, "timestamp"] - group.loc[start, "timestamp"]
                if duration >= min_duration:
                    stops.append(
                        {
                            "obj_id": obj_id,
                            "start_time": group.loc[start, "timestamp"],
                            "end_time": group.loc[end, "timestamp"],
                            "duration": duration,
                            "x": group.loc[start:end, "x"].mean(),
                            "y": group.loc[start:end, "y"].mean(),
                            "z": group.loc[start:end, "z"].mean(),
                        }
                    )

        return pd.DataFrame(stops)

    def calculate_trajectory_similarity(self, obj_id1, obj_id2):
        """
        Calculate the similarity between two trajectories using the Hausdorff distance.

        Parameters:
            obj_id1 (int): The ID of the first object to compare.
            obj_id2 (int): The ID of the second object to compare.

        Returns:
            float: The maximum Hausdorff distance between the two trajectories.
        """
        traj1 = self.df[self.df["obj_id"] == obj_id1][["x", "y", "z"]].values
        traj2 = self.df[self.df["obj_id"] == obj_id2][["x", "y", "z"]].values

        return max(
            directed_hausdorff(traj1, traj2)[0], directed_hausdorff(traj2, traj1)[0]
        )

    def parallel_process(self, func, *args, n_jobs=-1):
        """
        Parallelize a function across object IDs in the DataFrame.

        Parameters:
            func (function): The function to parallelize. It should take a pandas DataFrame group as its first argument.
            *args: Any additional arguments to pass to the function.
            n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1, which means using all available CPU cores.

        Returns:
            list: A list of results from the parallelized function calls.
        """
        return Parallel(n_jobs=n_jobs)(
            delayed(func)(group, *args) for _, group in self.df.groupby("obj_id")
        )


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame for demonstration
    data = pd.DataFrame(
        {
            "obj_id": np.repeat(range(1, 6), 100),
            "frame": np.tile(range(100), 5),
            "timestamp": np.tile(np.arange(0, 10, 0.1), 5),
            "x": np.random.rand(500) * 100,
            "y": np.random.rand(500) * 100,
            "z": np.random.rand(500) * 10,
            "xvel": np.random.randn(500),
            "yvel": np.random.randn(500),
            "zvel": np.random.randn(500),
        }
    )

    # Initialize the analyzer
    analyzer = TrackingDataAnalyzer(data)

    # Filter trajectories
    analyzer.filter_by_time(5).filter_by_distance(50)

    # Calculate and smooth velocities
    analyzer.calculate_linear_velocity().calculate_angular_velocity().smooth_trajectory()

    # Find peaks in linear velocity
    analyzer.find_peaks("linear_velocity", prominence=0.5)

    # Plot trajectory for object 1
    analyzer.plot_trajectory(1)

    # Extract a window of data
    window = analyzer.extract_window(
        250, n_before=50, n_after=50, columns=["x", "y", "z", "linear_velocity"]
    )
    print("Extracted window:")
    print(window)

    # Generate and display trajectory heatmap
    heatmap, fig = analyzer.generate_trajectory_heatmap()
    plt.show()

    # Detect stopping points
    stopping_points = analyzer.detect_stopping_points()
    print("\nDetected stopping points:")
    print(stopping_points)

    # Calculate similarity between two trajectories
    similarity = analyzer.calculate_trajectory_similarity(1, 2)
    print(f"\nSimilarity between trajectories 1 and 2: {similarity}")

    # Use parallel processing for a custom function
    def custom_analysis(group):
        return group["obj_id"].iloc[0], group["linear_velocity"].mean()

    results = analyzer.parallel_process(custom_analysis)
    print("\nResults of parallel processing (Object ID, Mean Linear Velocity):")
    print(results)

    # Animate trajectory for object 1
    anim = analyzer.animate_trajectory(1)
    plt.show()

    print("\nAnalysis complete!")
