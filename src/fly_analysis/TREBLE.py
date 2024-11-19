import numpy as np
import pandas as pd
from umap import UMAP
from scipy.spatial.distance import euclidean
from scipy.stats import fisher_exact
from scipy.spatial import procrustes
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import seaborn as sns
import warnings
from tqdm.auto import tqdm
from itertools import combinations

class TREBLE:
    def __init__(self, parallel: bool = False):
        """Initialize TREBLE analysis framework
        
        Args:
            parallel: If True, enables parallel processing but loses reproducibility
        """
        if parallel:
            self.umap = UMAP(n_components=2, n_jobs=-1)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", 
                    message="n_jobs value.*overridden.*", 
                    category=UserWarning)
                self.umap = UMAP(n_components=2, random_state=42)
        
    def get_windows(self, 
                    df: pd.DataFrame, 
                    window_size: int = 10, 
                    step_size: int = 1,
                    show_progress: bool = False) -> np.ndarray:
        """Extract sliding windows from velocity data."""
        windows = []
        total_windows = (len(df) - window_size) // step_size
        
        iterator = range(0, len(df) - window_size, step_size)
        if show_progress:
            iterator = tqdm(iterator, 
                          desc=f"Extracting windows (size={window_size})", 
                          total=total_windows)
            
        for i in iterator:
            window = df.iloc[i:i+window_size]
            features = np.concatenate([
                window['xvel'].values,
                window['yvel'].values,
                window['zvel'].values,
                window['linear_velocity'].values,
                window['angular_velocity'].values,
            ])
            windows.append(features)
        return np.array(windows)

    def iterative_umap(self,
                      dfs: List[pd.DataFrame],
                      window_sizes: List[int],
                      step_size: int = 1) -> Dict:
        """Run UMAP on multiple window sizes."""
        results = {}
        
        # Overall progress bar for window sizes
        for window_size in tqdm(window_sizes, desc="Processing window sizes"):
            # Get windows for all trajectories
            all_windows = []
            for idx, df in enumerate(tqdm(dfs, desc=f"Processing trajectories", leave=False)):
                windows = self.get_windows(df, window_size, step_size, show_progress=False)
                all_windows.append(windows)
                
            # Run UMAP on each trajectory's windows
            embeddings = []
            for idx, windows in enumerate(tqdm(all_windows, desc="Running UMAP", leave=False)):
                embedding = self.umap.fit_transform(windows)
                embeddings.append(embedding)
                
            results[window_size] = {
                'windows': all_windows,
                'embeddings': embeddings
            }
            
        return results

    def calculate_recurrence(self,
                           embeddings: List[np.ndarray],
                           n_bins: int = 16,
                           threshold: float = 0.05) -> Dict:
        """Calculate recurrence statistics for embeddings."""
        results = []
        
        for idx, embedding in enumerate(tqdm(embeddings, desc="Calculating recurrence")):
            # Grid the space
            x_bins = np.linspace(embedding[:,0].min(), embedding[:,0].max(), n_bins)
            y_bins = np.linspace(embedding[:,1].min(), embedding[:,1].max(), n_bins)
            
            # Find recurrences
            recurrences = []
            for i in range(len(embedding)):
                point = embedding[i]
                distances = np.sqrt(np.sum((embedding - point)**2, axis=1))
                recurring_points = np.where(distances < threshold)[0]
                if len(recurring_points) > 1:
                    recurrence_times = np.diff(recurring_points)
                    recurrences.extend(recurrence_times)
                    
            results.append({
                'recurrences': recurrences,
                'mean_recurrence': np.mean(recurrences) if recurrences else 0,
                'total_proportion': len(recurrences)/len(embedding)
            })
            
        return results

    def run_procrustes(self, embeddings: List[np.ndarray]) -> Dict:
        """Calculate Procrustes distances between embeddings."""
        n = len(embeddings)
        procrustes_distances = []
        euclidean_distances = []
        
        # Calculate total number of comparisons for progress bar
        total_comparisons = sum(1 for _ in combinations(range(n), 2))
        
        # Use combinations to get unique pairs
        pairs = combinations(range(n), 2)
        for i, j in tqdm(pairs, desc="Computing distances", total=total_comparisons):
            # Procrustes distance
            _, _, proc_dist = procrustes(embeddings[i], embeddings[j])
            procrustes_distances.append(proc_dist)
            
            # Euclidean distance
            euc_dist = np.mean([euclidean(p1, p2) 
                              for p1, p2 in zip(embeddings[i], embeddings[j])])
            euclidean_distances.append(euc_dist)
                
        return {
            'procrustes': procrustes_distances,
            'euclidean': euclidean_distances
        }

    def plot_results(self, results: Dict, window_sizes: List[int]):
        """Plot analysis results."""
        print("Generating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot Procrustes distances
        proc_means = [np.mean(results[w]['procrustes']) for w in window_sizes]
        proc_stds = [np.std(results[w]['procrustes']) for w in window_sizes]
        axes[0,0].errorbar(window_sizes, proc_means, yerr=proc_stds)
        axes[0,0].set_xlabel('Window Size')
        axes[0,0].set_ylabel('Procrustes RMSD')
        
        # Plot Euclidean distances
        euc_means = [np.mean(results[w]['euclidean']) for w in window_sizes]
        euc_stds = [np.std(results[w]['euclidean']) for w in window_sizes]
        axes[0,1].errorbar(window_sizes, euc_means, yerr=euc_stds)
        axes[0,1].set_xlabel('Window Size')
        axes[0,1].set_ylabel('Mean Euclidean Distance')
        
        # Plot recurrence statistics
        rec_means = [np.mean([r['mean_recurrence'] for r in results[w]['recurrence']]) 
                    for w in window_sizes]
        axes[1,0].plot(window_sizes, rec_means)
        axes[1,0].set_xlabel('Window Size')
        axes[1,0].set_ylabel('Mean Recurrence Time')
        
        # Plot recurrence proportions
        prop_means = [np.mean([r['total_proportion'] for r in results[w]['recurrence']]) 
                     for w in window_sizes]
        axes[1,1].plot(window_sizes, prop_means)
        axes[1,1].set_xlabel('Window Size')
        axes[1,1].set_ylabel('Proportion Recurrent')
        
        plt.tight_layout()
        plt.show()

    def analyze_trajectories(self,
                           dfs: List[pd.DataFrame],
                           window_sizes: List[int] = None,
                           step_size: int = 1) -> Dict:
        """Complete TREBLE analysis pipeline."""
        if window_sizes is None:
            window_sizes = [1] + list(range(5, 51, 5))
            
        print(f"Starting TREBLE analysis with {len(dfs)} trajectories")
        print(f"Testing window sizes: {window_sizes}")
            
        # Run iterative UMAP analysis
        umap_results = self.iterative_umap(dfs, window_sizes, step_size)
        
        # Calculate metrics for each window size
        results = {}
        for window_size in tqdm(window_sizes, desc="Computing metrics"):
            embeddings = umap_results[window_size]['embeddings']
            
            # Calculate distances
            distances = self.run_procrustes(embeddings)
            
            # Calculate recurrence
            recurrence = self.calculate_recurrence(embeddings)
            
            results[window_size] = {
                'procrustes': distances['procrustes'],
                'euclidean': distances['euclidean'],
                'recurrence': recurrence
            }
            
        # Plot results
        self.plot_results(results, window_sizes)
        
        print("Analysis complete!")
        return results

# Example usage:
"""
# Load your trajectory data into DataFrames
dfs = [pd.read_csv('trajectory1.csv'), pd.read_csv('trajectory2.csv')]

# Initialize and run analysis
treble = TREBLE()
results = treble.analyze_trajectories(dfs)
"""