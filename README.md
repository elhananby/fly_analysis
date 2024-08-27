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

To use the package, you must first install it. It is best to first create it inside a virtual environment (conda or mamba). Then you need to go to the repository folder, and install it using pip. 

1. Clone the repository:
   ```
   git clone https://github.com/elhananby/fly_analysis
   ```
   or just directly download it as a zip file.

2. Change to the repository directory:
   ```
   cd fly_analysis
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
   conda activate flyanalysis-env
   ```

5. Install the package (first go to the repo root in terminal):
   ```
   pip install -e .
   ```
   This will install the package in "editable" mode, which allows you to make changes to the code and have them reflected in the installed package without needing to reinstall.

## Usage

Here's a basic example of how to use FlyAnalysis:

```python
import braidz

# Read a .braidz file
df, csvs = braidz.read_braidz("path/to/your/file.braidz")

```

## Modules

### braidz.py

The `braidz.py` module provides functionality for reading and processing `.braidz` files, which are compressed archives containing CSV data files. This module is particularly useful for handling Kalman estimates and other related CSV data.

#### read_braidz(filename: str, extract_first: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]

This is the primary function for reading data from a `.braidz` file or a folder containing extracted `.braidz` contents.

###### Parameters:
- `filename` (str): The path to the `.braidz` file or folder.
- `extract_first` (bool, optional): Whether to first extract the `.braidz` file to a folder. Default is False.

###### Returns:
- A tuple containing:
  1. A pandas DataFrame with data from 'kalman_estimates.csv.gz'
  2. A dictionary of pandas DataFrames from other CSV files in the archive

##### Usage:
```python
from fly_analysis.braidz import read_braidz

# Reading from a .braidz file
df, csv_dict = read_braidz("path/to/your/file.braidz")

# Reading from a .braidz file, extracting it first
df, csv_dict = read_braidz("path/to/your/file.braidz", extract_first=True)

# Reading from an already extracted folder
df, csv_dict = read_braidz("path/to/extracted/folder")
```

### _read_from_file(filename: str, parser: Literal["pandas", "pyarrow"] = "pyarrow") -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]

This internal function reads data directly from a `.braidz` file using either PyArrow or pandas for CSV parsing.

#### Parameters:
- `filename` (str): The path to the `.braidz` file.
- `parser` (str, optional): The parser to use for reading CSV files. Either "pandas" or "pyarrow". Default is "pyarrow".

#### Returns:
- A tuple containing:
  1. A pandas DataFrame with data from 'kalman_estimates.csv.gz' (or None if not found)
  2. A dictionary of pandas DataFrames from other CSV files in the archive

### _read_from_folder(filename: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]

This internal function reads data from a folder containing extracted `.braidz` contents.

#### Parameters:
- `filename` (str): The path to the folder.

#### Returns:
- A tuple containing:
  1. A pandas DataFrame with data from 'kalman_estimates.csv.gz'
  2. A dictionary of pandas DataFrames from other CSV files in the folder

### _extract_to_folder(filename: str, read_extracted: bool = False) -> str

This internal function extracts a `.braidz` file to a folder and optionally reads the extracted files.

#### Parameters:
- `filename` (str): The path to the `.braidz` file.
- `read_extracted` (bool, optional): Whether to read the extracted files. Default is False.

#### Returns:
- The path to the folder where the files were extracted.

## Helper Functions

The module also includes several helper functions for reading CSV files and validating input:

- `read_csv_pyarrow(file_obj) -> Optional[pd.DataFrame]`: Reads a CSV file using PyArrow.
- `read_csv_pandas(file_obj) -> Optional[pd.DataFrame]`: Reads a CSV file using pandas.
- `validate_file(filename: str)`: Validates that the file is of the expected `.braidz` type.
- `validate_directory(filename: str)`: Validates that the directory contains the expected CSV files.

## Dependencies

This module requires the following Python libraries:
- pandas
- pyarrow
- tqdm
- zipfile
- gzip

Ensure these dependencies are installed before using the module.

## Error Handling

The functions in this module include error handling for common issues such as:
- Invalid file types
- Missing or empty files
- Parsing errors

When errors occur, appropriate exceptions are raised with informative error messages.
```


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
