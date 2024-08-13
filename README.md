# FlyAnalysis

FlyAnalysis is a comprehensive Python package designed to perform advanced analysis on `braid` output data. It provides a suite of tools for processing and analyzing fly trajectory data stored in `.braidz` files, as well as functions for video processing and trajectory analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
  - [braidz.py](#braidzpy)
  - [filtering.py](#filteringpy)
  - [helpers.py](#helperspy)
  - [plotting.py](#plottingpy)
  - [trajectory.py](#trajectorypy)
  - [video.py](#videopy)
- [Example Data](#example-data)
- [Documentation](#documentation)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/elhananby/FlyAnalysis
   ```
   or just directly download it as a zip file.

2. Change to the repository directory:
   ```
   cd FlyAnalysis
   ```

3. Create a **`mamba`** (preferred) or `conda` environment:
   ```
   mamba env create -f environment.yaml
   ```
   or
   ```
   conda env create -f environment.yaml
   ```

4. Activate the environment:
   ```
   conda activate flyanalysis
   ```

5. Install the package in developer mode:
   ```
   pip install -e .
   ```

This installation method allows any changes made to the code to reflect directly after re-importing the modules.

## Usage

Here's a basic example of how to use FlyAnalysis:

```python
import flyanalysis as fa

# Read a .braidz file
df, csvs = fa.braidz.read_braidz("path/to/your/file.braidz")

```

## Modules

### braidz.py

This module contains functions to load and process `.braidz` data files. Key functions include:

- `read_braidz(filename, extract_first=False)`: Reads data from a .braidz file or folder.
- `validate_file(filename)`: Validates that the file is of the expected .braidz type.
- `validate_directory(filename)`: Validates that the directory contains the expected files.

### filtering.py

This module provides various filtering functions to process the trajectory data:

- `filter_by_distance(df, threshold=0.5)`: Filters objects based on total distance traveled.
- `filter_by_duration(df, threshold=5)`: Filters objects based on duration of activity.
- `filter_by_median_position(df, xlim, ylim, zlim)`: Filters objects based on their median position.
- `filter_by_velocity(df, threshold=1.0)`: Filters objects based on average velocity.
- `filter_by_acceleration(df, threshold=0.5)`: Filters objects based on average acceleration.
- `filter_by_direction_changes(df, threshold=3)`: Filters objects based on number of direction changes.
- `apply_filters(df, filters, *args)`: Applies multiple filters to a DataFrame.

### helpers.py

This module contains utility functions used across the package:

- `sg_smooth(arr, **kwargs)`: Applies Savitzky-Golay smoothing to an array.
- `process_sequences(arr, func)`: Processes sequences in an array by applying a function to each non-NaN sequence.
- `unwrap_with_nan(arr, placeholder=0)`: Unwraps an array while handling NaN values.

### plotting.py

This module provides functions for visualizing the data:

- `plot_trajectory(df, ax=None, **kwargs)`: Plots the trajectory of a fly.
- `plot_mean_and_std(arr, ax=None, **kwargs)`: Plots the mean and standard deviation of an array.

### trajectory.py

This module offers advanced trajectory analysis functions:

- `time(df)`: Calculates the total time of the trajectory.
- `distance(df, axes="xyz")`: Calculates the total distance of the trajectory.
- `get_angular_velocity(df, dt=0.01, degrees=True)`: Calculates the angular velocity of the trajectory.
- `get_linear_velocity(df, dt=0.01, axes="xy")`: Calculates the linear velocity of the trajectory.
- `detect_saccades(df, height=500, distance=10)`: Detects saccades in the trajectory.
- `get_turn_angle(df, idx=None, axes="xy", degrees=True)`: Calculates the turn angle at each point in the trajectory.
- `get_simplified_trajectory(df, epsilon=0.001)`: Simplifies the trajectory using the Ramer-Douglas-Peucker algorithm.
- `heading_direction_diff(pos, origin=50, end=80, n=1)`: Calculates the difference in heading direction between two points.
- `smooth_columns(df, columns=["x", "y", "z", "xvel", "yvel", "zvel"], **kwargs)`: Applies Savitzky-Golay filter to specified columns.

### video.py

This module provides functions for video processing:

- `threshold(frame, threshold=127)`: Applies thresholding to a frame.
- `gray(frame)`: Converts a frame to grayscale.
- `find_contours(frame)`: Finds contours in a frame.
- `get_contour_parameters(contour)`: Gets parameters of a contour (centroid, area, perimeter, ellipse).
- `read_frame(video_reader, frame_num)`: Reads a specific frame from a video.
- `read_video(filename)`: Reads a video file and yields frames.

## Example Data

You can download an example `.braidz` file for analysis from [insert link here].

## Documentation

For more in-depth information about the `braid` output data format, please refer to the official [braid documentation](https://strawlab.github.io/strand-braid/braidz-files.html).
