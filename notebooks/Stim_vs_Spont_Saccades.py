import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from typing import List, Tuple, Optional
import argparse
import fly_analysis as fa

# Utility functions

def angdiff(theta1: float, theta2: float) -> float:
    """Calculate the angular difference between two angles."""
    return np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2))

def smooth_columns(df: pd.DataFrame, columns: List[str], window: int = 21, polyorder: int = 3) -> pd.DataFrame:
    """Apply Savitzky-Golay filter to specified columns of a DataFrame."""
    df_smoothed = df.copy()
    for col in columns:
        df_smoothed[f"{col}_raw"] = df_smoothed[col]
        df_smoothed[col] = savgol_filter(df_smoothed[col], window, polyorder)
    return df_smoothed

def calculate_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate angular and linear velocities."""
    df['theta'] = np.arctan2(df.yvel, df.xvel)
    df['theta_u'] = np.unwrap(df.theta)
    df['angular_velocity'] = np.gradient(df.theta_u, 0.01)
    df['linear_velocity'] = np.sqrt(df.xvel**2 + df.yvel**2)
    return df

def detect_saccades(angular_velocity: np.ndarray, threshold: float = 500, distance: int = 10) -> np.ndarray:
    """Detect saccades based on angular velocity peaks."""
    peaks, _ = find_peaks(np.abs(angular_velocity), height=np.deg2rad(threshold), distance=distance)
    saccades = np.zeros(len(angular_velocity), dtype=bool)
    saccades[peaks] = True
    return saccades

def filter_trajectories(df: pd.DataFrame, min_length: int = 300, max_z_speed: float = 0.3, min_speed: float = 0.05) -> pd.DataFrame:
    """Filter fly trajectories based on various criteria."""
    def trajectory_filter(gdf):
        if len(gdf) > min_length:
            speed = np.linalg.norm(np.vstack([gdf.zvel, gdf.xvel, gdf.yvel]), axis=0)
            return (gdf.z.min() > 0.05 and
                    gdf.z.max() < 0.5 and
                    np.max(np.abs(gdf.zvel)) < max_z_speed and
                    np.min(speed[5:]) > min_speed)
        return False

    return df[df.groupby("obj_id").apply(trajectory_filter)]

def split_on_jumps(df: pd.DataFrame, column: str = "frame", k: int = 1, n: int = 300) -> List[pd.DataFrame]:
    """Split DataFrame on jumps in a specified column."""
    diffs = df[column].diff()
    split_indices = diffs[diffs > k].index
    result = []
    
    if not split_indices.empty:
        start_idx = 0
        for idx in split_indices:
            chunk = df.iloc[start_idx:idx]
            if len(chunk) > n:
                result.append(chunk)
            start_idx = idx
        
        chunk = df.iloc[start_idx:]
        if len(chunk) > n:
            result.append(chunk)
    else:
        result = [df]
    
    return result

def calculate_heading_difference(x: np.ndarray, y: np.ndarray, loom_idx: int) -> float:
    """Calculate the heading difference before and after a loom event."""
    before_vector = np.array([x[loom_idx+15] - x[loom_idx-10], y[loom_idx+15] - y[loom_idx-10]])
    after_vector = np.array([x[loom_idx + 50] - x[loom_idx+15], y[loom_idx + 50] - y[loom_idx+15]])
    
    dot_product = np.dot(before_vector, after_vector)
    magnitudes = np.linalg.norm(before_vector) * np.linalg.norm(after_vector)
    
    cos_angle = dot_product / magnitudes
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    cross_product = np.cross(before_vector, after_vector)
    angle *= np.sign(cross_product)
    
    return np.degrees(angle)

# Data processing functions

def process_fly_data(df: pd.DataFrame, stim: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Process fly data and extract relevant information."""
    angvels = []
    linvels = []
    heading_differences = []

    for _, row in stim.iterrows():
        obj_id, frame, exp_num = row.obj_id, row.frame, row.exp_num
        
        grp_new = df[(df.obj_id == obj_id) & (df.exp_num == exp_num)]
        
        if len(grp_new) < 150:
            continue
        
        grp_new = smooth_columns(grp_new, ["x", "y", "z", "xvel", "yvel", "zvel"])
        grp_new = calculate_velocities(grp_new)
        grp_new['saccade'] = detect_saccades(grp_new.angular_velocity)
        
        grp_new = grp_new[
            (grp_new.x.between(-0.1, 0.1)) &
            (grp_new.y.between(-0.1, 0.1)) &
            (grp_new.z.between(0.05, 0.3)) &
            (grp_new.linear_velocity >= 0.01)
        ]
        
        for subgrp in split_on_jumps(grp_new):
            stim_idx = np.where(subgrp['frame'].values == frame)[0]
            stim_idx = stim_idx[0] if len(stim_idx) > 0 else np.nan
            
            x, y = subgrp['x'].values, subgrp['y'].values
            angular_velocity = subgrp['angular_velocity'].values
            linear_velocity = subgrp['linear_velocity'].values
            saccades = np.where(subgrp['saccade'].values)[0]
            
            try:
                stim_sac = next(sac for sac in saccades if 25 < sac - stim_idx < 45)
            except StopIteration:
                stim_sac = np.nan
            
            for sac in saccades:
                if sac - 25 < 0 or sac + 25 >= len(angular_velocity):
                    continue
                angvels.append(angular_velocity[sac - 25 : sac + 25])
                linvels.append(linear_velocity[sac - 25 : sac + 25])
                
                heading_before = np.arctan2(y[sac - 10], x[sac - 10])
                heading_after = np.arctan2(y[sac + 10], x[sac + 10])
                heading_differences.append(angdiff(heading_before, heading_after))
    
    return angvels, linvels, heading_differences

# Visualization functions

def plot_mean_and_std(data: np.ndarray, title: str, xlabel: str, ylabel: str):
    """Plot mean and standard deviation of data."""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean, label='Mean')
    plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3, label='Std Dev')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_trajectory(x: np.ndarray, y: np.ndarray, loom_idx: int, window: int = 25):
    """Plot fly trajectory with loom point and heading vectors."""
    plt.figure(figsize=(10, 10))
    
    plt.plot(x, y, '-', color='gray', alpha=0.5, label='Full trajectory')
    plt.plot(x[loom_idx-window:loom_idx+window+1], y[loom_idx-window:loom_idx+window+1], '-b', label='Relevant trajectory')
    plt.plot(x[loom_idx-window], y[loom_idx-window], 'ko', markersize=10, label='Start')
    plt.plot(x[loom_idx], y[loom_idx], 'rx', markersize=10, label='Loom point')
    
    before_vector = np.array([x[loom_idx] - x[loom_idx - window], y[loom_idx] - y[loom_idx - window]])
    after_vector = np.array([x[loom_idx + window] - x[loom_idx], y[loom_idx + window] - y[loom_idx]])
    
    plot_range = max(x.max() - x.min(), y.max() - y.min())
    scale = 0.1 * plot_range / max(np.linalg.norm(before_vector), np.linalg.norm(after_vector))
    before_vector_scaled = before_vector * scale
    after_vector_scaled = after_vector * scale
    
    plt.arrow(x[loom_idx], y[loom_idx], before_vector_scaled[0], before_vector_scaled[1], 
              color='green', width=0.001*plot_range, head_width=0.02*plot_range, head_length=0.02*plot_range, label='Before vector')
    plt.arrow(x[loom_idx], y[loom_idx], after_vector_scaled[0], after_vector_scaled[1], 
              color='red', width=0.001*plot_range, head_width=0.02*plot_range, head_length=0.02*plot_range, label='After vector')
    
    angle = calculate_heading_difference(x, y, loom_idx)
    
    plt.title('Trajectory with Loom Point and Heading Vectors')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.text(0.05, 0.95, f'Heading difference: {angle:.2f}°', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.show()


def load_data(file_paths):
    braidz_data = fa.braidz.read_multiple_braidz(file_paths, root_folder="/home/buchsbaum/mnt/md0/Experiments/")
    df = braidz_data["df"]
    stim = braidz_data["stim"]
    return df, stim

# Main function

def main(df: pd.DataFrame, stim: pd.DataFrame):

    """Main function to process fly data and generate visualizations."""
    parser = argparse.ArgumentParser(description='Process fly data and generate visualizations.')
    parser.add_argument('files', nargs='+', help='Path to the files containing fly data and stimulus information.')
    args = parser.parse_args()
    
    angvels, linvels, heading_differences = process_fly_data(df, stim)
    
    angvels = np.abs(np.asarray(angvels))
    linvels = np.abs(np.asarray(linvels))
    heading_differences = np.asarray(heading_differences)
    
    plot_mean_and_std(angvels, "Mean Angular Velocity", "Time steps", "Angular velocity (rad/s)")
    plot_mean_and_std(linvels, "Mean Linear Velocity", "Time steps", "Linear velocity (m/s)")
    
    plt.figure(figsize=(10, 6))
    plt.hist(heading_differences, bins=20, alpha=0.5)
    plt.title("Distribution of Heading Differences")
    plt.xlabel("Heading difference (degrees)")
    plt.ylabel("Frequency")
    plt.show()


# Example usage
if __name__ == "__main__":
    main()