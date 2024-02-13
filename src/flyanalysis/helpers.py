import numpy as np
from scipy.signal import savgol_filter


def _calculate_mean_and_std(arr: np.ndarray):
    """
    Calculate the mean and standard deviation of an array.

    Parameters:
        arr (np.ndarray): Input array.

    Returns:
        tuple: Mean and standard deviation of the array.
    """
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def sg_smooth(arr: np.array, **kwargs) -> np.array:
    """
    Apply Savitzky-Golay filter to the input array.

    Parameters:
    arr (np.array): Input array.
    **kwargs: Keyword arguments for scipy.signal.savgol_filter.

    Returns:
    np.array: Filtered array.
    """
    return savgol_filter(arr, **kwargs)


def process_sequences(arr: np.ndarray, func):
    nan_indices = np.isnan(arr)
    non_nan_sequences = np.ma.clump_unmasked(np.ma.masked_array(arr, nan_indices))

    new_arr = np.copy(arr)

    for seq in non_nan_sequences:
        new_arr[seq] = func(arr[seq])

    return new_arr


def unwrap_with_nan(arr, placeholder=0):
    # Replace NaN values with a placeholder
    arr_no_nan = np.where(np.isnan(arr), placeholder, arr)

    # Perform the unwrap
    unwrapped_arr = np.unwrap(arr_no_nan)

    # Replace the placeholder values with NaN again
    unwrapped_arr = np.where(unwrapped_arr == placeholder, np.nan, unwrapped_arr)

    return unwrapped_arr