import numpy as np
from scipy.signal import savgol_filter


def _calculate_mean_and_std(arr: np.ndarray, abs_value: bool = False):
    """
    Calculate the mean and standard deviation of an array.

    Parameters:
        arr (np.ndarray): Input array.

    Returns:
        tuple: Mean and standard deviation of the array.
    """
    mean = np.nanmean(arr, axis=0) if not abs_value else np.nanmean(np.abs(arr), axis=0)
    std = np.nanstd(arr, axis=0) if not abs_value else np.nanstd(np.abs(arr), axis=0)
    return mean, std


def sg_smooth(arr: np.array, **kwargs) -> np.array:
    """
    Apply Savitzky-Golay smoothing to an array.

    Args:
        arr (np.array): The input array to be smoothed.
        **kwargs: Additional keyword arguments to be passed to the `savgol_filter` function.

    Returns:
        np.array: The smoothed array.

    This function uses the `savgol_filter` function from the `scipy.signal` module to apply Savitzky-Golay smoothing to the input array. The smoothing parameters are specified using the `**kwargs` parameter. The resulting smoothed array is returned.
    """
    return savgol_filter(arr, **kwargs)


def process_sequences(arr: np.ndarray, func):
    """
    Process sequences in a given array by applying a function to each non-NaN sequence.

    Parameters:
        arr (np.ndarray): The input array containing sequences.
        func (function): The function to apply to each non-NaN sequence.

    Returns:
        np.ndarray: The processed array with the function applied to each non-NaN sequence.
    """
    nan_indices = np.isnan(arr)
    non_nan_sequences = np.ma.clump_unmasked(np.ma.masked_array(arr, nan_indices))

    new_arr = np.copy(arr)

    for seq in non_nan_sequences:
        new_arr[seq] = func(arr[seq])

    return new_arr


def unwrap_ignore_nan(arr):
    """
    Unwrap an array of angles, ignoring NaN values.

    Parameters:
        arr (array-like): An array of angles.

    Returns:
        array-like: The unwrapped array of angles.

    This function uses the `np.unwrap` function to unwrap the input array of angles, but ignores NaN values. This is useful for unwrapping arrays that contain NaN values, which would otherwise cause `np.unwrap` to raise an error.
    """
    
    arr[~np.isnan(arr)] = np.unwrap(arr[~np.isnan(arr)])
    return arr


def circular_median(angles, degrees=False):
    """
    Calculate the circular median of a set of angles.

    Parameters:
    angles (array-like): An array of angles.
    degrees (bool): If True, angles are in degrees. If False, angles are in radians.

    Returns:
    float: The circular median angle in the same units as the input.
    """
    if degrees:
        angles = np.deg2rad(angles)

    # Convert angles to unit vectors
    x = np.cos(angles)
    y = np.sin(angles)

    # Calculate the mean of the vectors
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Convert mean vector back to angle
    median_angle = np.arctan2(mean_y, mean_x)

    if degrees:
        median_angle = np.rad2deg(median_angle)
        # Ensure the result is in the range [0, 360)
        return (median_angle + 360) % 360
    else:
        # Ensure the result is in the range [0, 2π)
        return (median_angle + 2 * np.pi) % (2 * np.pi)


def find_intersection(*lists: list) -> list:
    """
    Find the intersection of any number of lists.

    Args:
    *lists: Variable number of lists to find the intersection of.

    Returns:
    List[T]: A list containing elements that appear in all input lists.

    Raises:
    ValueError: If no lists are provided.

    Example:
    >>> find_intersection([1, 2, 3], [2, 3, 4], [3, 4, 5])
    [3]
    >>> find_intersection(['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e'])
    ['c']
    """
    if not lists:
        raise ValueError("At least one list must be provided")

    # Convert all lists to sets
    set_list = [set(lst) for lst in lists]

    # Use set intersection
    intersection = set.intersection(*set_list)

    # Convert back to list and return
    return list(intersection)

def angdiff(theta1, theta2):
    return ((theta2 - theta1) + np.pi) % (2 * np.pi) - np.pi