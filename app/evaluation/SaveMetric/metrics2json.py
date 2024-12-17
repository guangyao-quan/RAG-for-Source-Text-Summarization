"""
This module defines a function for saving benchmark metrics to a JSON file.
It is designed to handle the serialization of benchmark results, which include
metrics such as BLEU, METEOR, and ROUGE scores, into a JSON format. The function
appends new metrics to an existing file or creates a new one if it does not exist.

The module uses standard Python libraries, os and json, to manage file
operations and JSON serialization.

Functions:
    save_metrics_to_json(average_scores, filepath, reset_json): Appends or creates a JSON file
    with the given metrics.
"""

import json
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)


def save_metrics_to_json(average_scores, filepath="data/metrics.json", reset_json=0):
    """
    Appends or saves the average metrics from benchmarks to a JSON file. If the file
    already exists, the function appends the new scores; if not, it creates a new file.

    This function is particularly useful in environments where benchmark results are accumulated
    over time, or when running multiple benchmarks in succession where each output needs to be
    preserved.

    Args:
        average_scores (dict): A dictionary containing the average metrics such as BLEU, METEOR,
                                ROUGE scores, and the 'app_id'. Each key should correspond to a
                                metric name, and the values should reflect the scores.
        filepath (str): The path where the JSON file will be saved. The default path is
                        'data/metrics.json', but it can be changed to specify a different
                        location or filename.
        reset_json (int): If 1, the existing JSON file will be cleared before saving the
                           new average scores. If 0, the new scores will be appended to the
                           existing content.

    Raises:
        IOError: An error occurred during the file write operation. This could be due to issues like
                 insufficient permissions or disk space.
        json.JSONDecodeError: An error occurred if the existing file contains invalid JSON and
                              cannot be decoded. This will cause the function to start a new file instead.

    Examples:
        scores = {"BLEU": 0.75, "METEOR": 0.77, "ROUGE": {"rouge1": 0.80, "rouge2": 0.75, "rougeL": 0.78}, "app_id": "run1"}
        save_metrics_to_json(scores, 'output/average_metrics.json', reset_json=True)
    """
    all_metrics = []

    if os.path.exists(filepath) and reset_json == 0:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                all_metrics = json.load(file)
        except json.JSONDecodeError:
            logger.error(f"ERROR DECODING JSON FROM {filepath}. STARTING A NEW FILE.")
            all_metrics = []

    if reset_json == 1:
        all_metrics = [average_scores]
    else:
        all_metrics.append(average_scores)

    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(all_metrics, file, indent=4)
        logger.info(f"METRICS SUCCESSFULLY SAVED OR UPDATED IN {filepath}.")
    except IOError as e:
        logger.error(f"FAILED TO SAVE OR UPDATE METRICS: {e}.")
