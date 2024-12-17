"""
This module provides a command-line interface for visualizing NLP evaluation scores and RAGAS metrics.
It reads traditional and RAG metrics from JSON files, processes the data, and generates plots to visualize
the evaluation scores.

Functions:
----------

1. `main(trad_file, ragas_file, mlflow_file, csv_file_paths, output_dir)`:
    - Main function to read evaluation scores, process them, and generate plots.

Usage Example:
--------------

    python visualize_scores.py trad_metric.json ragas_metric.json mlflow_metric.json output_dir
"""

import argparse
import os
from score_visualizer import ScoreVisualizer, kde_analysis
from app.evaluation.SaveMetric import merge_json


def main(trad_file, ragas_file, mlflow_file, csv_file_paths, output_dir):
    """
    Main function to read evaluation scores, process them, and generate plots.

    Args:
        trad_file (str): Path to the traditional metrics JSON file.
        ragas_file (str): Path to the RAGAS metrics JSON file.
        mlflow_file (str): Path to the MLflow metrics JSON file.
        csv_file_paths (list): List of paths to CSV files containing additional metrics.
        output_dir (str): Directory to save the generated plots.

    Raises:
        FileNotFoundError: If any of the provided file paths do not exist.
    """
    # Check if the provided file paths exist
    if not os.path.exists(trad_file):
        raise FileNotFoundError(f"Traditional metrics file not found: {trad_file}")
    if not os.path.exists(ragas_file):
        raise FileNotFoundError(f"RAGAS metrics file not found: {ragas_file}")
    if not os.path.exists(mlflow_file):
        raise FileNotFoundError(f"MLflow metrics file not found: {mlflow_file}")
    for csv_file in csv_file_paths:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot KDE analysis and save the figure
    xsum_trad_kde_fig_path = os.path.join(output_dir, "xsum_trad_kde_analysis_gpt.png")
    xsum_rag_kde_fig_path = os.path.join(output_dir, "xsum_rag_kde_analysis_gpt.png")
    kde_analysis(csv_file_paths[0], csv_file_paths[1], xsum_trad_kde_fig_path, xsum_rag_kde_fig_path,
                 dataset='xsum', title="KDE Analysis on XSUM with GPT")
    xsum_trad_kde_fig_path = os.path.join(output_dir, "xsum_trad_kde_analysis_bart.png")
    xsum_rag_kde_fig_path = os.path.join(output_dir, "xsum_rag_kde_analysis_bart.png")
    kde_analysis(csv_file_paths[0], csv_file_paths[1], xsum_trad_kde_fig_path, xsum_rag_kde_fig_path,
                 dataset='xsum-bart', title="KDE Analysis on XSUM with BART")
    multi_trad_kde_fig_path = os.path.join(output_dir, "multi_trad_kde_analysis.png")
    multi_rag_kde_fig_path = os.path.join(output_dir, "multi_rag_kde_analysis.png")
    kde_analysis(csv_file_paths[0], csv_file_paths[1], multi_trad_kde_fig_path, multi_rag_kde_fig_path,
                 dataset='multi-news', title="KDE Analysis on MULTI-NEWS with GPT")

    # Initialize the ScoreVisualizer
    sv = ScoreVisualizer()

    # Format and store the scores from the JSON files
    rag_file = merge_json.merge_json_by_app_id(mlflow_file, ragas_file)
    sv.format_scores_data(trad_file, metric_type="traditional")
    sv.format_scores_data(rag_file, metric_type="rag")
    sv.normalize_flesch_ari()

    # Plot traditional scores and save the figure
    trad_fig_path = os.path.join(output_dir, "xsum_trad_metric_gpt.png")
    trad_fig_path2 = os.path.join(output_dir, "xsum_trad_metric_bart.png")
    trad_fig_path3 = os.path.join(output_dir, "multi_trad_metric.png")
    ordered_list = ['gpt_baseline', 'default', 'sentence_window_3', 'sentence_window_5',
                    'auto_merging_1024_256_64', 'auto_merging_2048_512_128', 'extractive', 'bertopic', 'lda',
                    'knowledge_graph']
    ordered_list2 = ['bart_baseline', 'bart_sentence_window', 'bart_auto_merging', 'bart_extractive',
                     'bart_knowledge_graph']
    ordered_list3 = ['multi_news_default', 'multi_news_sw', 'multi_news_am', 'multi_news_ext',
                     'multi_news_lda', 'multi_news_bertopic']
    sv.plot_scores(scores_data=sv.trad_scores_data, title="Traditional Evaluation Metrics by Approach (Mean) on "
                                                          "XSUM with GPT",
                   save_path=trad_fig_path, ordered_approaches=ordered_list, figsize=(10, 8), rotation=45)
    sv.plot_scores(scores_data=sv.trad_scores_data, title="Traditional Evaluation Metrics by Approach (Mean) on "
                                                          "XSUM with BART",
                   save_path=trad_fig_path2, ordered_approaches=ordered_list2, figsize=(10, 8), rotation=45)
    sv.plot_scores(scores_data=sv.trad_scores_data, title="Traditional Evaluation Metrics by Approach (Mean) on "
                                                          "Multi-News",
                   save_path=trad_fig_path3, ordered_approaches=ordered_list3, figsize=(10, 8), rotation=45)

    # Plot RAG score and save the figure
    rag_fig_path = os.path.join(output_dir, "xsum_rag_metric_gpt.png")
    rag_fig_path2 = os.path.join(output_dir, "xsum_rag_metric_bart.png")
    rag_fig_path3 = os.path.join(output_dir, "multi_rag_metric.png")
    sv.plot_scores(scores_data=sv.rag_scores_data, title="RAG Evaluation Metrics by Approach (Mean) on XSUM with GPT",
                   save_path=rag_fig_path, ordered_approaches=ordered_list, figsize=(10, 8), rotation=90)
    sv.plot_scores(scores_data=sv.rag_scores_data, title="RAG Evaluation Metrics by Approach (Mean) on XSUM with BART",
                   save_path=rag_fig_path2, ordered_approaches=ordered_list2, figsize=(10, 8), rotation=90)
    sv.plot_scores(scores_data=sv.rag_scores_data, title="RAG Evaluation Metrics by Approach (Mean) on Multi-News",
                   save_path=rag_fig_path3, ordered_approaches=ordered_list3, figsize=(10, 8), rotation=90)


if __name__ == "__main__":
    """
    Command-line interface for the ScoreVisualizer script.

    This script reads traditional and RAG metrics from JSON files, processes
    the data, and generates plots for visualizing the evaluation scores.

    Example usage:
        python visualize_scores.py trad_metric.json ragas_metric.json mlflow_metric.json output_dir
    """
    parser = argparse.ArgumentParser(description="Visualize benchmarking results.")
    parser.add_argument("trad_file", type=str, help="Path to the traditional metrics JSON file.")
    parser.add_argument("ragas_file", type=str, help="Path to the RAGAS metrics JSON file.")
    parser.add_argument("mlflow_file", type=str, help="Path to the MLflow metrics JSON file.")
    parser.add_argument("csv_file_paths", type=str, nargs='+', help="List of paths to CSV files with "
                                                                    "additional metrics.")
    parser.add_argument("output_dir", type=str, help="Directory to save the plots.")

    args = parser.parse_args()
    main(args.trad_file, args.ragas_file, args.mlflow_file, args.csv_file_paths, args.output_dir)
