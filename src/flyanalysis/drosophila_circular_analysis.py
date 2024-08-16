# File: drosophila_trajectory_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import circmean, circvar


class DrosophilaTrajectoryAnalysis:
    def __init__(self, exp_trajectories, ctrl_trajectories):
        self.exp_trajectories = exp_trajectories
        self.ctrl_trajectories = ctrl_trajectories
        self.exp_angles = self.calculate_angles(exp_trajectories)
        self.ctrl_angles = self.calculate_angles(ctrl_trajectories)
        self.exp_turns = self.calculate_turn_magnitudes(self.exp_angles)
        self.ctrl_turns = self.calculate_turn_magnitudes(self.ctrl_angles)

    @staticmethod
    def calculate_angles(trajectories):
        """Calculate angles for each trajectory at each frame."""
        dx = np.diff(trajectories[:, :, 0], axis=1)
        dy = np.diff(trajectories[:, :, 1], axis=1)
        angles = np.arctan2(dy, dx)
        return np.hstack((angles[:, :1], angles))

    @staticmethod
    def calculate_turn_magnitudes(angles):
        """Calculate turn magnitudes from angles."""
        return np.abs(np.diff(angles, axis=1, append=angles[:, :1]))

    @staticmethod
    def segment_trajectories(data, pre_stim=50, stim_duration=30):
        """Segment trajectories into pre-stimulus, stimulus, and post-stimulus periods."""
        pre = data[:, :pre_stim]
        stim = data[:, pre_stim : pre_stim + stim_duration]
        post = data[:, pre_stim + stim_duration :]
        return pre, stim, post

    def analyze_turn_response(self):
        """Analyze turn response for experimental and control groups."""
        exp_pre, exp_stim, exp_post = self.segment_trajectories(self.exp_turns)
        ctrl_pre, ctrl_stim, ctrl_post = self.segment_trajectories(self.ctrl_turns)

        exp_mean = exp_post.mean(axis=1)
        ctrl_mean = ctrl_post.mean(axis=1)

        t_stat, p_value = stats.ttest_ind(exp_mean, ctrl_mean)

        return t_stat, p_value

    def watson_williams_test(self, sample1, sample2):
        """Perform Watson-Williams test for two samples of circular data."""
        n1, n2 = len(sample1), len(sample2)
        r1, r2 = np.sum(np.exp(1j * sample1)), np.sum(np.exp(1j * sample2))
        r = np.abs(r1) + np.abs(r2)

        mean1, mean2 = circmean(sample1), circmean(sample2)
        var1, var2 = circvar(sample1), circvar(sample2)

        # Pooled variance
        var_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

        # Test statistic
        F = (
            (n1 + n2 - 2)
            * ((n1 * n2) / (n1 + n2))
            * (2 * (1 - r / (n1 + n2)))
            / var_pooled
        )

        # Degrees of freedom
        df1, df2 = 1, n1 + n2 - 2

        # P-value
        p_value = 1 - stats.f.cdf(F, df1, df2)

        return F, p_value

    def circular_analysis(self):
        """Perform circular statistical analysis on angles."""
        _, _, exp_post = self.segment_trajectories(self.exp_angles)
        _, _, ctrl_post = self.segment_trajectories(self.ctrl_angles)

        exp_mean = circmean(exp_post, axis=1)
        ctrl_mean = circmean(ctrl_post, axis=1)

        watson_williams_stat, ww_pvalue = self.watson_williams_test(exp_mean, ctrl_mean)

        return watson_williams_stat, ww_pvalue

    def plot_results(self):
        """Plot mean turn magnitudes over time for both groups."""
        exp_mean = self.exp_turns.mean(axis=0)
        ctrl_mean = self.ctrl_turns.mean(axis=0)
        exp_sem = self.exp_turns.std(axis=0) / np.sqrt(self.exp_turns.shape[0])
        ctrl_sem = self.ctrl_turns.std(axis=0) / np.sqrt(self.ctrl_turns.shape[0])

        frames = np.arange(150)
        plt.figure(figsize=(12, 6))
        plt.plot(frames, exp_mean, label="Experimental")
        plt.fill_between(frames, exp_mean - exp_sem, exp_mean + exp_sem, alpha=0.3)
        plt.plot(frames, ctrl_mean, label="Control")
        plt.fill_between(frames, ctrl_mean - ctrl_sem, ctrl_mean + ctrl_sem, alpha=0.3)
        plt.axvspan(50, 80, color="yellow", alpha=0.3, label="Stimulus")
        plt.xlabel("Frame")
        plt.ylabel("Turn Magnitude")
        plt.legend()
        plt.title("Mean Turn Magnitude Over Time")
        plt.show()

    def plot_polar_histogram(self, angles, title, num_bins=36):
        """
        Plot a polar histogram of the given angles.

        :param angles: Array of angles in radians
        :param title: Title for the plot
        :param num_bins: Number of bins for the histogram (default 36, i.e., 10-degree bins)
        """
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

        # Convert angles to degrees for binning
        angles_deg = np.degrees(angles) % 360

        # Create histogram
        counts, bins = np.histogram(angles_deg, bins=num_bins, range=(0, 360))

        # Get the centers of the bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Convert bin centers back to radians for plotting
        bin_centers_rad = np.radians(bin_centers)

        # Plot
        bars = ax.bar(
            bin_centers_rad,
            counts,
            width=np.radians(360 / num_bins),
            bottom=0.0,
            alpha=0.8,
        )

        # Customize the plot
        ax.set_theta_zero_location("N")  # 0 degrees at the top
        ax.set_theta_direction(-1)  # clockwise
        ax.set_title(title)

        # Add a legend with the total count
        ax.legend([f"n = {np.sum(counts)}"])

        plt.show()

    def plot_polar_comparisons(self):
        """Plot polar histograms for experimental and control groups."""
        # Segment the trajectories
        _, _, exp_post = self.segment_trajectories(self.exp_angles)
        _, _, ctrl_post = self.segment_trajectories(self.ctrl_angles)

        # Calculate mean angles for each trajectory in the post-stimulus period
        exp_mean_angles = circmean(exp_post, axis=1)
        ctrl_mean_angles = circmean(ctrl_post, axis=1)

        # Plot
        self.plot_polar_histogram(exp_mean_angles, "Experimental Group - Post-stimulus")
        self.plot_polar_histogram(ctrl_mean_angles, "Control Group - Post-stimulus")

    def run_analysis(self):
        """Run the complete analysis and print results."""
        t_stat, p_value = self.analyze_turn_response()
        print(f"Turn response analysis: t-statistic = {t_stat}, p-value = {p_value}")

        ww_stat, ww_pvalue = self.circular_analysis()
        print(f"Watson-Williams test: statistic = {ww_stat}, p-value = {ww_pvalue}")

        self.plot_results()
        self.plot_polar_comparisons()


def main():
    # Example usage
    exp_trajectories = np.random.rand(10, 150, 3)  # Replace with actual data
    ctrl_trajectories = np.random.rand(10, 150, 3)  # Replace with actual data

    analysis = DrosophilaTrajectoryAnalysis(exp_trajectories, ctrl_trajectories)
    analysis.run_analysis()


if __name__ == "__main__":
    main()
