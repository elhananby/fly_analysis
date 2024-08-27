# Fly Analysis

This repository contains a Python package for analyzing fly behavior data. The package provides tools for processing and visualizing data from experiments tracking fly movements.

## Table of Contents

- [Fly Analysis](#fly-analysis)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
    - [Cloning the Repository](#cloning-the-repository)
    - [Creating the Environment](#creating-the-environment)
    - [Installing the Package](#installing-the-package)
  - [Module Descriptions](#module-descriptions)
    - [braidz.py](#braidzpy)
    - [filtering.py](#filteringpy)
    - [helpers.py](#helperspy)
    - [plotting.py](#plottingpy)
    - [processing.py](#processingpy)
    - [trajectory.py](#trajectorypy)
    - [video.py](#videopy)
  - [Usage](#usage)
  - [Data Format](#data-format)
    - [.braidz File Structure](#braidz-file-structure)
    - [kalman\_estimates.csv.gz](#kalman_estimatescsvgz)
    - [Additional CSV Files](#additional-csv-files)
  - [Working with .braidz Data](#working-with-braidz-data)
    - [Extracting Object Trajectories](#extracting-object-trajectories)
    - [Plotting Trajectories](#plotting-trajectories)
    - [Working with Stimulus Data](#working-with-stimulus-data)

## Repository Structure

The repository is structured as follows:

```
fly_analysis/
├── src/
│   └── fly_analysis/
│       ├── braidz.py
│       ├── filtering.py
│       ├── helpers.py
│       ├── plotting.py
│       ├── processing.py
│       ├── trajectory.py
│       └── video.py
├── notebooks/
│   └── (Jupyter notebooks)
├── scripts/
│   └── (Python scripts)
├── environment.yaml
├── README.md
├── setup.py
└── .gitignore
```

## Installation

### Cloning the Repository

To clone the repository, open a terminal and run the following command:

```bash
git clone https://github.com/your-username/fly_analysis.git
cd fly_analysis
```

Replace `your-username` with the actual GitHub username or organization name where the repository is hosted.

### Creating the Environment

This project uses Conda for environment management. To create the environment, follow these steps:

1. Make sure you have Anaconda or Miniconda installed on your system.
2. Open a terminal and navigate to the project directory.
3. Run the following command to create the environment:

```bash
conda env create -f environment.yaml
```

4. Activate the environment:

```bash
conda activate fly_analysis
```

### Installing the Package

After activating the environment, install the package in editable mode:

```bash
pip install -e .
```

This command installs the package in editable mode, allowing you to modify the source code and immediately see the effects without reinstalling.

## Module Descriptions

### braidz.py

This module provides functionality for reading and processing .braidz files, which contain fly tracking data.

Key functions:
- `read_braidz(filename, extract_first=False)`: Reads data from a .braidz file or a folder.
- `_read_from_file(filename, parser="pyarrow")`: Reads data from a .braidz file using either PyArrow or pandas for CSV parsing.
- `_extract_to_folder(filename, read_extracted=False)`: Extracts a .braidz file to a folder.

Usage example:
```python
from fly_analysis import braidz

# Read data from a .braidz file
df, csvs = braidz.read_braidz("path/to/your/file.braidz")
```

### filtering.py

This module contains functions for filtering fly trajectory data based on various criteria.

Key functions:
- `filter_by_distance(df, threshold=0.5)`: Filters objects based on the total distance traveled.
- `filter_by_duration(df, threshold=5)`: Filters objects based on the duration of activity.
- `filter_by_median_position(df, xlim, ylim, zlim)`: Filters objects based on their median position.
- `filter_by_velocity(df, threshold=1.0)`: Filters objects based on their average velocity.
- `filter_by_acceleration(df, threshold=0.5)`: Filters objects based on their average acceleration.
- `apply_filters(df, filters, *args)`: Applies multiple filters to a DataFrame.

Usage example:
```python
from fly_analysis import filtering

# Filter trajectories by distance
good_obj_ids = filtering.filter_by_distance(df, threshold=1.0)

# Apply multiple filters
filtered_df, good_obj_ids = filtering.apply_filters(
    df,
    [filtering.filter_by_distance, filtering.filter_by_duration],
    1.0,  # distance threshold
    10    # duration threshold
)
```

### helpers.py

This module provides utility functions for data processing and analysis.

Key functions:
- `sg_smooth(arr, **kwargs)`: Applies Savitzky-Golay smoothing to an array.
- `process_sequences(arr, func)`: Processes sequences in a given array by applying a function to each non-NaN sequence.
- `unwrap_with_nan(arr, placeholder=0)`: Unwraps an array while handling NaN values.
- `circular_median(angles, degrees=False)`: Calculates the circular median of a set of angles.
- `angdiff(theta1, theta2)`: Calculates the angular difference between two angles.

Usage example:
```python
from fly_analysis import helpers
import numpy as np

# Smooth a noisy signal
noisy_signal = np.random.randn(100)
smooth_signal = helpers.sg_smooth(noisy_signal, window_length=11, polyorder=3)

# Calculate circular median of angles
angles = np.random.uniform(0, 2*np.pi, 100)
median_angle = helpers.circular_median(angles)
```

### plotting.py

This module contains functions for visualizing fly trajectory data.

Key functions:
- `plot_trajectory(df, ax=None, **kwargs)`: Plots the trajectory of a fly.
- `plot_mean_and_std(arr, ax=None, **kwargs)`: Plots the mean and standard deviation of an array.
- `plot_dispersions(groups, labels, ax=None)`: Plots the XY parts of trajectories for multiple groups in a 2D plot.

Usage example:
```python
from fly_analysis import plotting
import matplotlib.pyplot as plt

# Plot a single trajectory
fig, ax = plt.subplots()
plotting.plot_trajectory(df, ax=ax, color='blue', linewidth=1)

# Plot dispersions for multiple groups
groups = [group1_trajectories, group2_trajectories]
labels = ['Group 1', 'Group 2']
plotting.plot_dispersions(groups, labels)
plt.show()
```

### processing.py

This module provides functions for processing fly trajectory data, particularly for extracting stimulus-centered data.

Key functions:
- `extract_stimulus_centered_data(df, csv, n_before=50, n_after=100, columns=["angular_velocity", "linear_velocity", "position"], padding=None)`: Extracts stimulus-centered data from a DataFrame and a CSV file.

Usage example:
```python
from fly_analysis import processing

# Extract stimulus-centered data
stimulus_data = processing.extract_stimulus_centered_data(df, csv_data, n_before=100, n_after=200)
```

### trajectory.py

This module provides functions for analyzing and processing fly trajectories.

Key functions:
- `time(df)`: Calculate the total time of the trajectory.
- `distance(df, axes="xyz")`: Calculate the total distance of the trajectory.
- `get_angular_velocity(df, dt=0.01, degrees=True)`: Calculate the angular velocity of the trajectory.
- `get_linear_velocity(df, dt=0.01, axes="xy")`: Calculate the linear velocity of the trajectory.
- `detect_saccades(df, height=500, distance=10)`: Detect saccades in the trajectory.
- `get_turn_angle(df, idx=None, axes="xy", degrees=True)`: Calculate the turn angle at each point in the trajectory.
- `heading_direction_diff(pos, origin=50, end=80, n=1)`: Calculate the difference in heading direction between two points.
- `smooth_columns(df, columns=["x", "y", "z", "xvel", "yvel", "zvel"], **kwargs)`: Apply Savitzky-Golay filter to the input columns.
- `mGSD(trajectory, delta=5, threshold=0.001)`: Modified Geometric Saccade Detection Algorithm.

Usage example:
```python
from fly_analysis import trajectory
import pandas as pd

# Assuming 'df' is your trajectory DataFrame
total_time = trajectory.time(df)
total_distance = trajectory.distance(df)
angular_velocity = trajectory.get_angular_velocity(df)
saccades = trajectory.detect_saccades(df)

# Smooth trajectory data
smoothed_df = trajectory.smooth_columns(df, window_length=11, polyorder=3)
```

### video.py

This module provides functions for processing video data, particularly useful for analyzing fly behavior in video recordings.

Key functions:
- `threshold(frame, threshold=127)`: Threshold a frame.
- `gray(frame)`: Convert a frame to grayscale.
- `find_contours(frame)`: Find contours in a frame.
- `get_largest_contour(contours)`: Get the largest contour in a list of contours.
- `get_contour_parameters(contour)`: Get parameters of a contour (centroid, area, perimeter, ellipse).
- `read_video(filename)`: Read a video file and yield frames.

Usage example:
```python
from fly_analysis import video
import numpy as np

# Read a video file
for frame in video.read_video("path/to/your/video.mp4"):
    # Process each frame
    gray_frame = video.gray(frame)
    thresholded_frame = video.threshold(gray_frame)
    contours = video.find_contours(thresholded_frame)
    
    if contours:
        largest_contour = video.get_largest_contour(contours)
        centroid, area, perimeter, ellipse = video.get_contour_parameters(largest_contour)
        
        # Use these parameters for further analysis
        print(f"Fly centroid: {centroid}, Area: {area}")
```

## Usage

To use the fly_analysis package in your Python scripts or Jupyter notebooks, first import the necessary modules:

```python
from fly_analysis import braidz, filtering, helpers, plotting, processing, trajectory, video
```

Then you can use the functions provided by each module as described in the module descriptions above.


## Data Format

### .braidz File Structure

The fly_analysis package works with .braidz files, which are actually ZIP files with a specific structure. These files contain various data related to fly tracking experiments. Here's an example of the contents of a .braidz file:

```
  Length      Date    Time    Name
---------  ---------- -----   ----
       97  2019-11-25 09:33   README.md
      155  2019-11-25 09:33   braid_metadata.yml
        0  2019-11-25 09:33   images/
   308114  2019-11-25 09:33   images/Basler_22005677.png
   233516  2019-11-25 09:33   images/Basler_22139107.png
   283260  2019-11-25 09:33   images/Basler_22139109.png
   338040  2019-11-25 09:33   images/Basler_22139110.png
       78  2019-11-25 09:33   cam_info.csv.gz
     2469  2019-11-25 09:33   calibration.xml
      397  2019-11-25 09:33   textlog.csv.gz
   108136  2019-11-25 09:33   kalman_estimates.csv.gz
      192  2019-11-25 09:33   trigger_clock_info.csv.gz
       30  2019-11-25 09:33   experiment_info.csv.gz
     2966  2019-11-25 09:33   data_association.csv.gz
   138783  2019-11-25 09:33   data2d_distorted.csv.gz
---------                     -------
  1416233                     15 files
```

The most important file for analysis is `kalman_estimates.csv.gz`, which contains the actual tracking data.

### kalman_estimates.csv.gz

This file contains the main tracking data for the flies. Each row in this file represents a single observation of a fly at a specific time point. The columns in this file are defined as follows:

```rust
pub struct KalmanEstimatesRow {
    pub obj_id: u32,
    pub frame: SyncFno,
    pub timestamp: Option<FlydraFloatTimestampLocal<Triggerbox>>,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub xvel: f64,
    pub yvel: f64,
    pub zvel: f64,
    pub P00: f64,
    pub P01: f64,
    pub P02: f64,
    pub P11: f64,
    pub P12: f64,
    pub P22: f64,
    pub P33: f64,
    pub P44: f64,
    pub P55: f64,
}
```

Key information about these fields:

- `obj_id`: A unique identifier assigned to each detected object (fly). It's important to note that if an object is lost for a few frames and then re-detected, it will be assigned a new `obj_id`. This means the system does not maintain object identity if there are multiple objects in the arena.
- `frame`: The synchronized detection frame across all cameras.
- `timestamp`: The raw timestamp in UNIX time.
- `x`, `y`, `z`: The position of the detected object in 3D space.
- `xvel`, `yvel`, `zvel`: The velocity of the detected object in each dimension.
- The remaining fields (`P00` to `P55`) contain debug data related to the Kalman filter updating process.

### Additional CSV Files

The .braidz file may contain additional CSV files that provide supplementary information about the experiment. One such file is `stim.csv`, which contains information about stimuli presented during the experiment.

When you use the `read_braidz` function, it extracts all CSV files into a dictionary, making them easily accessible for further analysis.

The `stim.csv` file typically contains:
- A copy of the row from `kalman_estimates.csv.gz` at the time of stimulus presentation
- Additional columns with information about the stimulus, such as duration, position, size, etc.

This information is crucial for analyzing fly behavior in response to specific stimuli.

## Working with .braidz Data

### Extracting Object Trajectories

To work with the data from a .braidz file, you first need to read it using the `braidz.py` module, and then you can apply various filters to extract specific object trajectories. Here's an example of how to do this:

```python
from fly_analysis import braidz, filtering, plotting
import matplotlib.pyplot as plt

# Read the .braidz file
df, csvs = braidz.read_braidz("path/to/your/file.braidz")

# Apply filters to extract specific object trajectories
# For example, let's filter objects that have moved more than 10 units and lasted for more than 5 seconds
distance_filtered = filtering.filter_by_distance(df, threshold=10)
duration_filtered = filtering.filter_by_duration(df, threshold=5)

# Combine the filters
good_obj_ids = list(set(distance_filtered) & set(duration_filtered))

# Extract the trajectories of the filtered objects
filtered_df = df[df['obj_id'].isin(good_obj_ids)]

# Now filtered_df contains only the trajectories of objects that meet our criteria
```

### Plotting Trajectories

After extracting the trajectories you're interested in, you can use the `plotting.py` module to visualize them. Here's an example of how to plot multiple trajectories:

```python
# Create a new figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each trajectory
for obj_id in good_obj_ids:
    obj_df = filtered_df[filtered_df['obj_id'] == obj_id]
    plotting.plot_trajectory(obj_df, ax=ax, label=f'Object {obj_id}')

# Customize the plot
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Fly Trajectories')
ax.legend()

# Show the plot
plt.show()
```

This code will create a 2D plot showing the trajectories of all the objects that passed our filtering criteria. Each trajectory will be plotted in a different color and labeled with its `obj_id`.

You can further customize these plots or create different types of visualizations (e.g., 3D trajectories, velocity plots) using the functions in the `plotting.py` module and matplotlib's extensive capabilities.

Remember to explore the other modules in the package, such as `trajectory.py` and `processing.py`, which offer additional tools for analyzing the extracted trajectories, such as calculating angular velocities, detecting saccades, or extracting stimulus-centered data.

### Working with Stimulus Data

The `extract_stimulus_centered_data` function in the `processing.py` module is designed to work with both the main tracking data (`kalman_estimates.csv.gz`) and the stimulus data (`stim.csv`). Here's how it works and how you can use it:

1. **How `extract_stimulus_centered_data` works**:
   - It takes the main DataFrame (`df`) from `kalman_estimates.csv.gz` and the CSV data (including `stim.csv`) as inputs.
   - For each stimulus event in `stim.csv`, it extracts a window of data from the main DataFrame, centered around the stimulus presentation time.
   - It can extract various types of data (e.g., position, velocity) for a specified number of frames before and after the stimulus.

2. **Using `extract_stimulus_centered_data`**:

```python
from fly_analysis import braidz, processing
import pandas as pd

# Read the .braidz file
df, csvs = braidz.read_braidz("path/to/your/file.braidz")

# Extract stimulus-centered data
stimulus_data = processing.extract_stimulus_centered_data(
    df,
    csvs['stim'],
    n_before=50,  # Number of frames before stimulus
    n_after=100,  # Number of frames after stimulus
    columns=["position", "linear_velocity"]
)

# Now stimulus_data contains the extracted data centered around each stimulus event
```

3. **Extracting data for specific object IDs**:
   If you need to extract all data for specific object IDs mentioned in the `stim.csv` file, you can do so using the following approach:

```python
# Get unique object IDs from the stim.csv file
stim_obj_ids = csvs['stim']['obj_id'].unique()

# Extract all data for these object IDs from the main DataFrame
stim_objects_data = df[df['obj_id'].isin(stim_obj_ids)]

# If you want to separate data for each object:
stim_objects_dict = {obj_id: stim_objects_data[stim_objects_data['obj_id'] == obj_id] for obj_id in stim_obj_ids}

# Now stim_objects_dict is a dictionary where each key is an obj_id, and the value is a DataFrame containing all the data for that object
```

This approach allows you to extract and analyze the complete trajectories of objects that received stimuli, which can be useful for studying behavior before, during, and after stimulus presentation.

By combining the stimulus information from `stim.csv` with the detailed trajectory data from `kalman_estimates.csv.gz`, you can perform in-depth analyses of how flies respond to specific stimuli in your experiments.
