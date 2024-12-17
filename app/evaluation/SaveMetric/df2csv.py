"""
This module provides a function to save a pandas DataFrame to a CSV file with options to either
reset (clear) the file or append to it.

Functions:
    save_dataframe_to_csv(df, csv_path, reset_csv):
        Save a DataFrame to a CSV file, with options to clear the file before saving or append to it.
"""

import pandas as pd
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)


def save_dataframe_to_csv(df, csv_path, reset_csv=0):
    """
    Save a DataFrame to a CSV file.

    This function saves the given DataFrame to the specified CSV file. If the 'reset_csv' parameter
    is set to True, the contents of the CSV file are cleared before saving the DataFrame. If
    'reset_csv' is set to False, the DataFrame is appended to the existing CSV file if it exists.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        csv_path (str): The path to the CSV file.
        reset_csv (int): If 1, reset (clear) the CSV file before saving. If 0, append to the CSV file.

    Returns:
        None
    """
    if reset_csv == 1:
        # Clear the CSV file by writing an empty DataFrame to it
        pd.DataFrame().to_csv(csv_path, index=False)
        logger.info(f"CSV FILE AT '{csv_path}' HAS BEEN CLEARED.")
    if os.path.exists(csv_path) and reset_csv == 0:
        # Append to the existing CSV file
        df.to_csv(csv_path, mode='a', header=False, index=False)
        logger.info(f"DATAFRAME HAS BEEN APPENDED TO THE EXISTING CSV FILE AT '{csv_path}'.")
    else:
        # Write the DataFrame to the CSV file (header included)
        df.to_csv(csv_path, mode='w', header=True, index=False)
        logger.info(f"DATAFRAME HAS BEEN WRITTEN TO A NEW CSV FILE AT '{csv_path}'.")
