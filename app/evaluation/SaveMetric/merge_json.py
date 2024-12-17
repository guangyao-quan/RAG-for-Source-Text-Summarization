"""
This module provides a function to merge two JSON files based on a common `app_id` field and save
the merged result to a new file.

Functions:
----------

1. `merge_json_by_app_id(json1_path, json2_path)`:
    - Merges two JSON files based on the `app_id` field and saves the merged result to a new file.

Usage Example:
--------------

    merged_file_path = merge_json_by_app_id('path/to/first.json', 'path/to/second.json')
    print(f"Merged file saved to: {merged_file_path}")
"""

import os
import json
import logging

# Configure logger
logger = logging.getLogger(__name__)


def merge_json_by_app_id(json1_path, json2_path):
    """
    Merges two JSON files based on the app_id field and saves the merged result to a new file.

    The function reads two JSON files, each containing a list of dictionaries. Each dictionary
    must have an `app_id` field. The function merges these dictionaries based on the `app_id`
    field. If a dictionary with the same `app_id` exists in both files, their contents are merged.
    If a dictionary exists in only one file, it is included as is. The merged result is saved to a
    new JSON file.

    Parameters:
    -----------
    json1_path : str
        Path to the first JSON file.
    json2_path : str
        Path to the second JSON file.

    Returns:
    --------
    str
        Path to the merged JSON file.
    """
    output_path = "data/outputs/rag_metric.json"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load JSON data from files
    with open(json1_path, 'r') as f1:
        json1 = json.load(f1)

    with open(json2_path, 'r') as f2:
        json2 = json.load(f2)

    # Convert list of dictionaries to dictionary indexed by app_id
    json1_dict = {item['app_id']: item for item in json1}
    json2_dict = {item['app_id']: item for item in json2}

    # Merge the dictionaries based on app_id
    merged_dict = {}
    for app_id in json1_dict:
        if app_id in json2_dict:
            merged_dict[app_id] = {**json1_dict[app_id], **json2_dict[app_id]}
        else:
            merged_dict[app_id] = json1_dict[app_id]

    for app_id in json2_dict:
        if app_id not in merged_dict:
            merged_dict[app_id] = json2_dict[app_id]

    # Convert merged dictionary back to a list
    merged_json = list(merged_dict.values())

    # Save the merged JSON to a file
    with open(output_path, 'w') as outfile:
        json.dump(merged_json, outfile, indent=4)

    logger.info(f"MERGED JSON SAVED TO '{output_path}'")

    return output_path
