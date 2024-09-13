import numpy as np
from typing import List, Tuple

class MGSDAlgorithm:
    def __init__(self, delta: int = 5, threshold: float = 0.001):
        self.delta = delta
        self.threshold = threshold

    def detect_saccades(self, trajectory: List[Tuple[float, float]]) -> List[int]:
        n = len(trajectory)
        scores = []

        for k in range(self.delta, n - self.delta):
            score = self._calculate_mgsd_score(trajectory, k)
            scores.append(score)

        return self._find_saccade_events(scores)

    def _calculate_mgsd_score(self, trajectory: List[Tuple[float, float]], k: int) -> float:
        # Step 2: Redefine origin
        origin = trajectory[k]
        
        # Step 3: Define before and after intervals
        before_interval = trajectory[k - self.delta : k]
        after_interval = trajectory[k + 1 : k + self.delta + 1]
        
        # Step 4 & 5: Calculate angles and median angles
        before_angles = [self._calculate_angle(p, origin) for p in before_interval]
        after_angles = [self._calculate_angle(p, origin) for p in after_interval]
        theta_before = np.median(before_angles)
        theta_after = np.median(after_angles)
        
        # Calculate amplitude score
        A = np.abs(theta_after - theta_before)
        
        # Step 6: Calculate dispersion score
        full_interval = trajectory[k - self.delta : k + self.delta + 1]
        distances = [self._euclidean_distance(p, origin) for p in full_interval]
        D = np.std(distances)
        
        # Step 7: Calculate mGSD score
        return A * D

    def _find_saccade_events(self, scores: List[float]) -> List[int]:
        saccade_events = []
        in_saccade = False
        start_frame = 0
        
        for i, score in enumerate(scores):
            if score > self.threshold and not in_saccade:
                in_saccade = True
                start_frame = i
            elif score <= self.threshold and in_saccade:
                if i - start_frame > 5:
                    saccade_events.append(start_frame + self.delta + (i - start_frame) // 2)
                in_saccade = False

        return saccade_events

    @staticmethod
    def _calculate_angle(point: Tuple[float, float], origin: Tuple[float, float]) -> float:
        return np.arctan2(point[1] - origin[1], point[0] - origin[0])

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate a sample trajectory with more pronounced saccades
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    x = np.cumsum(np.where(np.random.rand(1000) > 0.98, np.random.normal(0, 1), np.random.normal(0, 0.1)))
    y = np.cumsum(np.where(np.random.rand(1000) > 0.98, np.random.normal(0, 1), np.random.normal(0, 0.1)))
    trajectory = list(zip(x, y))

    # Create MGSD algorithm instance
    mgsd = MGSDAlgorithm(delta=5, threshold=0.005)  # Adjusted threshold for demonstration

    # Detect saccades
    saccade_events = mgsd.detect_saccades(trajectory)

    print(f"Detected {len(saccade_events)} saccade events at frames: {saccade_events}")

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'b-', alpha=0.5)
    plt.plot([x[i] for i in saccade_events], [y[i] for i in saccade_events], 'ro', markersize=8)
    plt.title('Trajectory with Detected Saccades')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.show()

    # Plot mGSD scores
    scores = [mgsd._calculate_mgsd_score(trajectory, k) for k in range(mgsd.delta, len(trajectory) - mgsd.delta)]
    plt.figure(figsize=(12, 4))
    plt.plot(range(mgsd.delta, len(trajectory) - mgsd.delta), scores)
    plt.axhline(y=mgsd.threshold, color='r', linestyle='--')
    plt.title('mGSD Scores')
    plt.xlabel('Frame')
    plt.ylabel('mGSD Score')
    plt.show()