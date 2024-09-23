from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FileGroup:
    """Represents a group of files and their associated data."""
    name: str
    files: List[str] = field(default_factory=list)
    df: Any = None
    stim: Any = None
    opto: Any = None
    processed: Any = None
    plots: Dict[str, Any] = field(default_factory=dict)

class FileGroupManager:
    """Manages groups of files and their associated data."""

    def __init__(self, base_path: str):
        """
        Initialize the FileGroupManager.

        Args:
            base_path (str): The base path where the files are located.
        """
        self.base_path = Path(base_path)
        self.groups: Dict[str, FileGroup] = {}

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """
        Allows dictionary-like access to data across all groups.

        Args:
            key (str): The type of data to retrieve (e.g., 'df', 'stim', 'opto', 'processed').

        Returns:
            Dict[str, Any]: A dictionary mapping group names to the requested data type.

        Raises:
            KeyError: If the requested data type doesn't exist.
        """
        valid_keys = {'df', 'stim', 'opto', 'processed', 'files', 'plots'}
        if key not in valid_keys:
            raise KeyError(f"Invalid key '{key}'. Valid keys are: {', '.join(valid_keys)}")

        return {group.name: getattr(group, key) for group in self.groups.values()}
    
    def add_group(self, name: str, files: Optional[List[str]] = None) -> None:
        """
        Add a new group, optionally with files.

        Args:
            name (str): The name of the new group.
            files (Optional[List[str]]): List of files to add to the group.

        Raises:
            ValueError: If the group already exists.
        """
        if name in self.groups:
            raise ValueError(f"Group '{name}' already exists.")
        
        self.groups[name] = FileGroup(name, files or [])
        logger.info(f"Added new group '{name}' with {len(self.groups[name].files)} file(s)")

    def remove_group(self, name: str) -> None:
        """
        Remove a group.

        Args:
            name (str): The name of the group to remove.

        Raises:
            KeyError: If the group doesn't exist.
        """
        if name not in self.groups:
            raise KeyError(f"Group '{name}' does not exist.")
        
        del self.groups[name]
        logger.info(f"Removed group '{name}'")

    def add_files_to_group(self, name: str, files: List[str]) -> None:
        """
        Add files to an existing group.

        Args:
            name (str): The name of the group.
            files (List[str]): List of files to add.

        Raises:
            KeyError: If the group doesn't exist.
        """
        if name not in self.groups:
            raise KeyError(f"Group '{name}' does not exist.")
        
        self.groups[name].files.extend(files)
        logger.info(f"Added {len(files)} file(s) to group '{name}'")

    def read_group_data(self, name: str) -> None:
        """
        Read data for a specific group.

        Args:
            name (str): The name of the group.

        Raises:
            KeyError: If the group doesn't exist.
            FileNotFoundError: If any file in the group is not found.
        """
        if name not in self.groups:
            raise KeyError(f"Group '{name}' does not exist.")

        group = self.groups[name]
        full_paths = [self.base_path / file for file in group.files]

        for path in full_paths:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

        try:
            from . import braidz
            # Convert PosixPath objects to strings
            str_paths = [str(path) for path in full_paths]
            group.df, group.stim, group.opto = braidz.read_multiple_braidz(str_paths)
            logger.info(f"Successfully read data for group '{name}'")
        except Exception as e:
            logger.error(f"Failed to read data for group '{name}': {str(e)}")
            raise

    def read_all_group_data(self) -> None:
        """
        Read data for all groups concurrently.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.read_group_data, name): name for name in self.groups}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error reading data for group '{name}': {str(e)}")

    def process_group_data(self, groups: Optional[Union[str, List[str]]] = None, csv_to_process: str = "opto") -> None:
        """
        Process data for specified group(s) or all groups if none specified.

        Args:
            groups (Optional[Union[str, List[str]]]): The name of a single group, 
                a list of group names, or None to process all groups.

        Raises:
            KeyError: If a specified group doesn't exist.
            ValueError: If the group's data hasn't been read.
        """
        from . import processing

        def process_single_group(name: str) -> None:
            if name not in self.groups:
                raise KeyError(f"Group '{name}' does not exist.")
            
            group = self.groups[name]
            if group.df is None:
                raise ValueError(f"Data for group '{name}' hasn't been read.")
            
            try:
                group.processed = processing.extract_stimulus_centered_data(group.df, getattr(group, csv_to_process))
                logger.info(f"Successfully processed data for group '{name}'")
            except Exception as e:
                logger.error(f"Failed to process data for group '{name}': {str(e)}")
                raise

        if groups is None:
            groups_to_process = list(self.groups.keys())
        elif isinstance(groups, str):
            groups_to_process = [groups]
        elif isinstance(groups, list):
            groups_to_process = groups
        else:
            raise TypeError("'groups' must be a string, a list of strings, or None")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_single_group, name): name for name in groups_to_process}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing data for group '{name}': {str(e)}")


    def plot_group_data(self, name: str, key: str = "angular_velocity") -> None:
        """
        Plot data for a specific group.

        Args:
            name (str): The name of the group.
            plot_type (str): The type of plot to generate.

        Raises:
            KeyError: If the group doesn't exist or hasn't been processed.
        """
        if name not in self.groups or self.groups[name].processed is None:
            raise KeyError(f"Processed data for group '{name}' is not available.")

        if key not in self.groups[name].processed:
            raise KeyError(f"Key '{key}' not found in processed data for group '{name}'.")
        
        try:
            from . import plotting
            plot = plotting.plot_mean_and_std(self.groups[name].processed[key])
            self.groups[name].plots[key] = plot
            logger.info(f"Successfully created {key} plot for group '{name}'")
        except Exception as e:
            logger.error(f"Failed to create {key} plot for group '{name}': {str(e)}")
            raise

    def plot_all_groups(self, key: str, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot data from all groups on the same plot for comparison.

        Args:
            plot_type (str): The type of plot to generate (e.g., 'time_series', 'histogram').
            save_path (Optional[str]): Path to save the plot. If None, the plot is not saved.

        Returns:
            plt.Figure: The matplotlib Figure object containing the plot.

        Raises:
            ValueError: If no groups have processed data or if the plot type is not supported.
        """
        groups_with_data = [group for group in self.groups.values() if group.processed is not None]
        if not groups_with_data:
            raise ValueError("No groups have processed data available for plotting.")

        _, axs = plt.subplots()
        
        if key not in ["angular_velocity", "linear_velocity"]:
            raise ValueError(f"Key '{key}' not supported.")
        
        from . import plotting
        for group in groups_with_data:
            plotting.plot_mean_and_std(group.processed[key], ax=axs, label=group.name, shaded_area=[50, 80], abs_value=True)

        plt.xlabel('Frame')
        plt.ylabel(key)
        plt.title('Comparative Time Series Across Groups')
        plt.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Comparative plot saved to {save_path}")

        #return plt.gcf()

    def get_group_data(self, name: str) -> FileGroup:
        """
        Get data for a specific group.

        Args:
            name (str): The name of the group.

        Returns:
            FileGroup: The FileGroup object containing all group data.

        Raises:
            KeyError: If the group doesn't exist.
        """
        if name not in self.groups:
            raise KeyError(f"Group '{name}' does not exist.")
        
        return self.groups[name]

    def get_all_groups(self) -> List[str]:
        """
        Get a list of all group names.

        Returns:
            List[str]: A list of all group names.
        """
        return list(self.groups.keys())

# Example usage
if __name__ == "__main__":
    base_path = Path("/home/buchsbaum/mnt/md0/Experiments/")
    manager = FileGroupManager(str(base_path))

    # Add groups with files
    manager.add_group("J21", ["20230906_155507.braidz"])
    manager.add_group("J73", ["20240910_140319.braidz"])
    manager.add_group("J74", ["20240911_151201.braidz"])

    # Add a group without files, then add files to it
    manager.add_group("J75")
    manager.add_files_to_group("J75", ["20240912_133045.braidz"])

    # Read and process all group data
    manager.read_all_group_data()
    
    for group_name in manager.get_all_groups():
        manager.process_group_data(group_name)
        manager.plot_group_data(group_name, "time_series")

    # Print out the data structure
    for group_name in manager.get_all_groups():
        group = manager.get_group_data(group_name)
        print(f"Group: {group.name}")
        print(f"  Files: {group.files}")
        print(f"  Data loaded: {group.df is not None and group.stim is not None and group.opto is not None}")
        print(f"  Data processed: {group.processed is not None}")
        print(f"  Plots created: {', '.join(group.plots.keys())}")
        print()

    # Demonstrate removing a group
    manager.remove_group("J74")
    print(f"Remaining groups: {manager.get_all_groups()}")