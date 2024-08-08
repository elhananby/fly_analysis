# FlyAnalysis

This package contains some functions and classes to perform basic analysis on `braid` output data.

I would highly recommend going over the official [**braid documentation**](https://strawlab.github.io/strand-braid/braidz-files.html) for more in-depth information.

You can get an example file for analysis from here: 
## Installation

* Clone the repository using:
  ```
  git clone https://github.com/elhananby/FlyAnalysis
  ```
* cd into the repo folder and create a **`mamba`** (preferable)/`conda` environment using:
  ```
  mamba env create -f environment.yaml
  ```
  or 
  ```
  conda env create -f environment.yaml
  ```
* Install the flyanalysis package using:
  ```
  pip install -e .
  ```
  This installs it in developer mode, so any changes made to the code should reflect directly after re-importing to modules.

## Modules

### braidz.py

Contains all the functions required to load a `.braidz` data file.
```python
import flyanalysis as fa
df, csvs = fa.braidz.read_braidz("20240120_110636.braidz")
```
`df` contains the `kalman_estimates.csv.gz`, which contains the raw tracking data (you can read more about it here <https://strawlab.github.io/strand-braid/braidz-files.html>)
`csvs` contains all the additionally recorded data from the system, including optogenetic triggers (`opto.csv`) or stimuli triggers and information (`stim.csv`)

### helpers.py

Some basic helper data processing functions. Usually not called directly.

### trajectory.py

This is the main analysis module. Contains multiple functions to perform filtering and analysis.
Refer to the `docstring` for additional information about each function.

### plotting.py

Basic plotting functions, including a simple 2D trajectory plotting (which should optimally be used per `obj_id`) and a function to plot the mean and std of a time series (can be used for example for angular velocity windows).

### tracking_data_analyzer.py

Work-in-progress class to encapsulate the entire `braidz` processing pipeline.