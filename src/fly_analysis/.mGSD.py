import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mGSD(df, delta=5, threshold=0.001):
    """
    Modified Geometric Saccade Detection Algorithm (Stupski and van Breugel, 2024)

    Parameters:
    df (pd.DataFrame): Input trajectory DataFrame.
    delta (int): Time step in frames (default 5).
    threshold (float): Threshold for saccade detection (default 0.001).

    Returns:
    tuple: A tuple containing the results DataFrame and a list of saccade event frames.

    The results DataFrame has the following columns:
    - 'frame': The frame index.
    - 'score': The modified geometric saccade detection score.

    The saccade event frames are the frames where there is a saccade, determined by the threshold.
    """

    def calculate_angle(x, y):
        return np.arctan2(y, x)

    def calculate_distance(x, y):
        return np.sqrt(x**2 + y**2)

    results = []
    n = len(df)

    for k in range(delta, n - delta):
        # Redefine origin
        x_center, y_center = df.iloc[k]["x"], df.iloc[k]["y"]
        x = df["x"] - x_center
        y = df["y"] - y_center

        # Calculate angles for before and after intervals
        before_angles = calculate_angle(x[k - delta : k], y[k - delta : k])
        after_angles = calculate_angle(
            x[k + 1 : k + delta + 1], y[k + 1 : k + delta + 1]
        )

        # Calculate amplitude score
        theta_before = np.median(before_angles)
        theta_after = np.median(after_angles)
        A = abs(theta_after - theta_before)

        # Calculate dispersion score
        distances = calculate_distance(
            x[k - delta : k + delta + 1], y[k - delta : k + delta + 1]
        )
        D = np.std(distances)

        # Calculate mGSD score
        S = A * D

        results.append({"frame": k, "score": S})

    results_df = pd.DataFrame(results)

    # Detect saccade events
    saccade_events = []
    in_saccade = False
    saccade_start = 0
    below_threshold_count = 0

    for i, row in results_df.iterrows():
        if row["score"] > threshold:
            if not in_saccade:
                in_saccade = True
                saccade_start = i
            below_threshold_count = 0
        else:
            if in_saccade:
                below_threshold_count += 1
                if below_threshold_count > 5:
                    saccade_events.append(
                        int(results_df.loc[saccade_start:i, "frame"].median())
                    )
                    in_saccade = False

    if in_saccade:
        saccade_events.append(int(results_df.loc[saccade_start:, "frame"].median()))

    return results_df, saccade_events

def plot_trajectory_with_saccades(df, saccade_events, output_file=None):
    plt.figure(figsize=(12, 8))
    
    # Plot the full trajectory
    plt.plot(df['x'], df['y'], 'b-', alpha=0.5, label='Trajectory')
    
    # Mark the start and end points
    plt.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', markersize=10, label='Start')
    plt.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'ro', markersize=10, label='End')
    
    # Mark saccade events
    for event in saccade_events:
        plt.plot(df['x'].iloc[event], df['y'].iloc[event], 'y*', markersize=15, label='Saccade')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title('Fly Trajectory with Detected Saccades')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()