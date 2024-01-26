import glob
import os
import pathlib
import zipfile

import pandas as pd
from tqdm import tqdm


def _read_from_file(filename: str) -> tuple[pd.DataFrame, dict]:
    """
    Reads data from a .braidz file.

    This function opens a .braidz file, reads the 'kalman_estimates.csv.gz' file into a pandas DataFrame,
    and also reads any other .csv files present in the .braidz file into a dictionary of pandas DataFrames.

    Parameters:
    filename (str): The path to the .braidz file.

    Returns:
    tuple: A tuple containing the DataFrame from 'kalman_estimates.csv.gz' and a dictionary of DataFrames from other .csv files.
    """
    filepath = pathlib.Path(filename)
    with zipfile.ZipFile(file=filepath, mode="r") as archive:
        print(f"Reading {filename}")
        # read df from braidz file
        try:
            df = pd.read_csv(
                archive.open("kalman_estimates.csv.gz"), comment="#", compression="gzip"
            )
        except pd.errors.EmptyDataError:
            return None, None

        # check if any other csv files exist in the braidz file, and load them
        # this is to handle the opto.csv, stim.csv, or something_else.csv that may
        # get created during the recording
        csv_files = [csv for csv in archive.namelist() if csv.endswith(".csv")]

        csv_s = {}
        for csv_file in csv_files:
            key, _ = os.path.splitext(csv_file)
            try:
                csv_s[key] = pd.read_csv(archive.open(csv_file))
            except pd.errors.EmptyDataError:
                continue
    return df, csv_s


def _read_from_folder(filename: str) -> tuple[pd.DataFrame, dict]:
    """
    Reads data from a folder.

    This function reads the 'kalman_estimates.csv.gz' file from a specified folder into a pandas DataFrame,
    and also reads any other .csv files present in the folder into a dictionary of pandas DataFrames.

    Parameters:
    filename (str): The path to the folder.

    Returns:
    tuple: A tuple containing the DataFrame from 'kalman_estimates.csv.gz' and a dictionary of DataFrames from other .csv files.
    """
    df = pd.read_csv(
        os.path.join(filename, "kalman_estimates.csv.gz"),
        comment="#",
        compression="gzip",
    )
    csv_files = glob.glob(f"{filename}/*.csv")

    csv_s = {}
    for csv_file in csv_files:
        key, _ = os.path.splitext(csv_file)
        try:
            csv_s[key] = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            continue

    return df, csv_s


def _extract_to_folder(filename: str, read_extracted: bool = False) -> str:
    """
    Extracts a .braidz file to a folder.

    This function extracts all the files from a .braidz file to a folder with the same name as the .braidz file.
    If the folder already exists, it does nothing. If read_extracted is True, it also reads the extracted files.

    Parameters:
    filename (str): The path to the .braidz file.
    read_extracted (bool): Whether to read the extracted files. Default is False.

    Returns:
    str: The path to the folder where the files were extracted.
    """
    extract_path = os.path.splitext(filename)[0]
    if not os.path.exists(extract_path):
        os.mkdir(extract_path)
    # else:
    # return _read_from_folder(extract_path)

    with zipfile.ZipFile(filename, "r") as zip_ref:
        # Get the total number of files inside the zip
        total_files = len(zip_ref.infolist())

        # Loop through each file, extract it, and update the tqdm progress bar
        with tqdm(total=total_files, unit="file") as pbar:
            for member in zip_ref.infolist():
                # Extract a single file
                zip_ref.extract(member, extract_path)
                pbar.update(1)

    if read_extracted:
        return _read_from_folder(extract_path)

    return extract_path


def validate_file(filename: str):
    """
    Validates that the file is of the expected type.

    Parameters:
    filename (str): The path to the file.

    Raises:
    ValueError: If the file is not of the expected type.
    """
    if not filename.endswith(".braidz"):
        raise ValueError(f"File {filename} is not of the expected '.braidz' type")


def validate_directory(filename: str):
    """
    Validates that the directory contains the expected files.

    Parameters:
    filename (str): The path to the directory.

    Raises:
    ValueError: If the directory does not contain the expected files.
    """
    if not any(fname.endswith(".csv.gz") for fname in os.listdir(filename)):
        raise ValueError(f"Directory {filename} does not contain any '.csv.gz' files")


def read_braidz(
    filename: str, extract_first: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Reads data from a .braidz file or a folder.

    This function reads the 'kalman_estimates.csv.gz' file from a .braidz file or a folder into a pandas DataFrame,
    and also reads any other .csv files present into a dictionary of pandas DataFrames.
    If extract_first is True, it first extracts the .braidz file to a folder.

    Parameters:
    filename (str): The path to the .braidz file or folder.
    extract_first (bool): Whether to first extract the .braidz file to a folder. Default is False.

    Returns:
    tuple: A tuple containing the DataFrame from 'kalman_estimates.csv.gz' and a dictionary of DataFrames from other .csv files.
    """
    if os.path.isfile(filename):
        validate_file(filename)
        if extract_first:
            df, csvs = _extract_to_folder(filename, read_extracted=True)
        else:
            df, csvs = _read_from_file(filename)

    elif os.path.isdir(filename):
        validate_directory(filename)
        df, csvs = _read_from_folder(filename)

    else:
        raise FileNotFoundError(f"Path {filename} does not exist")

    return df, csvs
