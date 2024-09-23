import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import logging

from scipy.signal import savgol_filter

from .trajectory import get_angular_velocity, get_linear_velocity


def extract_stimulus_centered_data(
    df: pd.DataFrame,
    csv: pd.DataFrame,
    n_before: int = 50,
    n_after: int = 100,
    columns: List[str] = ["angular_velocity", "linear_velocity", "position"],
    smooth: bool = True,
    padding: Optional[int] = 20,
    min_length: int = 150,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extracts stimulus-centered data from a DataFrame and a CSV file.

    This function takes in a DataFrame `df` and a CSV file `csv`, and extracts data
    centered around a stimulus event. The data is extracted for a specified number
    of frames before and after the stimulus event, and can be padded with NaN values
    if the stimulus event occurs near the start or end of the data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    csv (pd.DataFrame): The CSV file containing the stimulus event information.
    n_before (int, optional): The number of frames to extract before the stimulus event. Defaults to 50.
    n_after (int, optional): The number of frames to extract after the stimulus event. Defaults to 100.
    columns (List[str], optional): The columns to extract data for. Defaults to ["angular_velocity", "linear_velocity", "position"].
    smooth (bool, optional): Whether to apply smoothing to the data. Defaults to True.
    padding (Optional[int], optional): The number of frames to pad the data with if the stimulus event occurs near the start or end of the data. Defaults to 20.
    min_length (int, optional): The minimum length of data required to process a stimulus event. Defaults to 150.

    Returns:
    Dict[str, Dict[str, np.ndarray]]: A dictionary containing the extracted data for each column, separated into 'real' and 'sham' stimuli.
    """

    def pad_dataframe(df: pd.DataFrame, n: int, fill_value: Any = np.nan) -> pd.DataFrame:
        pad_df = pd.DataFrame(fill_value, index=range(n), columns=df.columns)
        return pd.concat([pad_df, df, pad_df], ignore_index=True)

    def smooth_columns(df: pd.DataFrame, columns: List[str] = ["x", "y", "z", "xvel", "yvel", "zvel"]) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                arr = df[col].to_numpy()
                df[f"original_{col}"] = arr.copy()
                df[col] = savgol_filter(arr, 21, 3)
        return df

    def initialize_data_dict() -> Dict[str, List[Any]]:
        return {col: [] for col in columns + ["timestamps", "exp_num"]}

    def process_segment(grp: pd.DataFrame, stim_idx: int, idx_before: int, idx_after: int) -> Tuple[Dict[str, Any], bool]:
        segment_data = {}
        is_valid = True

        for col in columns:
            if col == "angular_velocity":
                angvel = get_angular_velocity(grp, degrees=False)
                segment = angvel[idx_before:idx_after]
            elif col == "linear_velocity":
                linvel = get_linear_velocity(grp)
                segment = linvel[idx_before:idx_after]
            elif col == "position":
                segment = grp[["x", "y", "z"]].iloc[idx_before:idx_after].values
            else:
                logging.warning(f"Column {col} not found")
                is_valid = False
                break

            segment_data[col] = segment

        if is_valid:
            segment_data["timestamps"] = grp["timestamp"].iloc[stim_idx]
            segment_data["exp_num"] = grp["exp_num"].iloc[stim_idx] if "exp_num" in grp.columns else None

        return segment_data, is_valid

    data_dict = {"real": initialize_data_dict(), "sham": initialize_data_dict()}

    skipped_trajectory = {
        "group_too_short": 0,
        "no_index": 0,
        "insufficient_data": 0,
    }

    for _, row in csv.iterrows():
        obj_id = int(row["obj_id"])
        frame = int(row["frame"])
        grp = df[df.obj_id == obj_id].copy()

        if smooth:
            grp = smooth_columns(grp)

        if padding is not None:
            grp = pad_dataframe(grp, padding)

        if len(grp) < min_length:
            skipped_trajectory["group_too_short"] += 1
            logging.info(f"Skipping {obj_id} - too short. (length: {len(grp)})")
            continue

        try:
            stim_idx = np.where(grp.frame == frame)[0][0]
        except IndexError:
            skipped_trajectory["no_index"] += 1
            logging.warning(f"Frame {frame} not found for object {obj_id}")
            continue
        
        idx_before = stim_idx - n_before
        idx_after = stim_idx + n_after

        if idx_before < 0 or idx_after >= len(grp):
            skipped_trajectory["insufficient_data"] += 1
            logging.debug(f"Skipping {obj_id} - insufficient data around stimulus. ({idx_before}, {idx_after}, {len(grp)})")
            continue

        segment_data, is_valid = process_segment(grp, stim_idx, idx_before, idx_after)

        if is_valid:
            if "sham" in row:
                if row["sham"] == True:
                    stimulus_type = "sham"
                elif row["sham"] == False or row["sham"] == np.nan:
                    stimulus_type = "real"
            else:
                stimulus_type = "real"
            for key, value in segment_data.items():
                data_dict[stimulus_type][key].append(value)

    # Convert all key-items to numpy arrays
    for stimulus_type in ["real", "sham"]:
        for k, v in data_dict[stimulus_type].items():
            data_dict[stimulus_type][k] = np.array(v)

    total_skipped = skipped_trajectory["group_too_short"] + skipped_trajectory["no_index"] + skipped_trajectory["insufficient_data"]
    logging.info(f"Skipped a total of {total_skipped} trajectories.")
    logging.info(f"Skipped {skipped_trajectory['group_too_short']} trajectories because they were too short.")
    logging.info(f"Skipped {skipped_trajectory['no_index']} trajectories because the frame index was not found.")
    logging.info(f"Skipped {skipped_trajectory['insufficient_data']} trajectories because there was insufficient data around the stimulus.")
    return data_dict, skipped_trajectory