from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import trajectory
import logging

def extract_stimulus_centered_data(
    df: pd.DataFrame,
    csv: pd.DataFrame,
    n_before: int = 50,
    n_after: int = 100,
    columns: List[str] = ["angular_velocity", "linear_velocity", "position"],
    padding: Optional[int] = None,
) -> Dict[str, List[Any]]:
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
    padding (Optional[int], optional): The number of frames to pad the data with if the stimulus event occurs near the start or end of the data. Defaults to None.

    Returns:
    Dict[str, List[Any]]: A dictionary containing the extracted data for each column.
    """
    
    def pad_dataframe(df, n, fill_value=np.nan):
        pad_df = pd.DataFrame(np.nan, index=range(n), columns=df.columns)
        df_padded = pd.concat([pad_df, df, pad_df], ignore_index=True)
        return df_padded
    
    data_dict = {}
    for col in columns:
        data_dict[col] = []

    for idx, row in csv.iterrows():
        # extract identifier and frame number
        obj_id = int(row["obj_id"])
        frame = int(row["frame"])

        # filter dataframe based on identifier
        grp = df[df.obj_id == obj_id]
        
        # pad dataframe if necessary
        if padding is not None:
            grp = pad_dataframe(grp, padding)

        # skip if length is less than 150
        if len(grp) < 150:
            continue

        # find index of stimulus in main df
        try:
            stim_idx = np.where(grp.frame == frame)[0][0]
        except IndexError:
            continue

        # set indices and check boundaries
        idx_before = stim_idx - n_before
        idx_after = stim_idx + n_after

        if idx_before < 0 or idx_after >= len(grp):
            logging.info(f"Skipping {obj_id}, {frame} - too short.")
            continue

        # Get data and apply padding if necessary
        for col in columns:
            if col == "angular_velocity":
                angvel = trajectory.get_angular_velocity(grp, degrees=False)
                segment = angvel[idx_before:idx_after]
                data_dict[col].append(segment)

            elif col == "linear_velocity":
                linvel = trajectory.get_linear_velocity(grp)
                segment = linvel[idx_before:idx_after]
                data_dict[col].append(segment)

            elif col == "position":
                position_segment = (
                    grp[["x", "y", "z"]].iloc[idx_before:idx_after].values
                )
                data_dict[col].append(position_segment)

            else:
                print(f"Column {col} not found")

    return data_dict
