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


def unwrap_with_nan(arr, placeholder=0):
    """
    Replaces NaN values in the input array with a specified placeholder value, unwraps the array using np.unwrap(),
    and then replaces the placeholder values with NaN again.

    Parameters:
        arr (np.ndarray): The input array.
        placeholder (int, optional): The value to replace NaN values with. Defaults to 0.

    Returns:
        np.ndarray: The unwrapped array with NaN values replaced by the placeholder value.
    """
    # Replace NaN values with a placeholder
    arr_no_nan = np.where(np.isnan(arr), placeholder, arr)

    # Perform the unwrap
    unwrapped_arr = np.unwrap(arr_no_nan)

    # Replace the placeholder values with NaN again
    unwrapped_arr = np.where(unwrapped_arr == placeholder, np.nan, unwrapped_arr)

    return unwrapped_arr
