"""
This module provides utilities for visualizing and analyzing NLP evaluation scores and RAGAS metrics.
It includes classes and functions to read, process, and plot score data from JSON and CSV files.

Classes:
--------

1. `ScoreVisualizer`:
    - A utility class to visualize NLP traditional evaluation scores and RAGAS metrics.

Functions:
----------

1. `reorder_rows(df, app_id_column, order_dict)`:
    - Reorders the rows of a DataFrame based on a custom order provided in `order_dict`.

2. `plot_kde(df, save_path, id_column, order_dict, title=None)`:
    - Plots KDE for the given DataFrame and saves the plot.

3. `kde_analysis(csv_file_path1, csv_file_path2, save_path1, save_path2, dataset='xsum', title=None)`:
    - Analyzes and compares the distributions of scores across two dataframes read from CSV files independently.

Usage Example:
--------------

    # Initialize the visualizer
    visualizer = ScoreVisualizer()

    # Format traditional scores data
    visualizer.format_scores_data('path/to/traditional_scores.json', metric_type='traditional')

    # Format RAG scores data
    visualizer.format_scores_data('path/to/rag_scores.json', metric_type='rag')

    # Normalize scores
    visualizer.normalize_flesch_ari()

    # Plot traditional scores
    visualizer.plot_scores(visualizer.trad_scores_data, 'Traditional Scores', 'path/to/save/traditional_scores.png')

    # Plot RAG scores
    visualizer.plot_scores(visualizer.rag_scores_data, 'RAG Scores', 'path/to/save/rag_scores.png')

    # Perform KDE analysis
    kde_analysis('path/to/csv1.csv', 'path/to/csv2.csv', 'path/to/save1.png', 'path/to/save2.png', dataset='xsum')
"""

import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List


class ScoreVisualizer:
    """
    A utility class to visualize NLP traditional evaluation scores and RAGAS metrics.

    Attributes:
        trad_scores_data (dict): A dictionary to store traditional scores for different approaches.
        rag_scores_data (dict): A dictionary to store RAG scores for different approaches.

    Methods:
        format_scores_data(file_path, metric_type):
            Reads, formats, and stores scores from a JSON file.
        normalize_flesch_ari():
            Normalizes the 'flesch_kincaid_grade_level' and 'ari_grade_level' scores in the RAG data.
        plot_scores(scores_data, title, save_path, figsize, rotation):
            Generates and saves a bar chart visualizing the scores.
    """

    def __init__(self):
        """
        Initializes the ScoreVisualizer with empty dictionaries to hold score data.
        """
        self.trad_scores_data = {}
        self.rag_scores_data = {}

    def format_scores_data(self, file_path, metric_type="traditional"):
        """
        Reads, processes, and stores scores from a JSON file into the internal dictionary.

        Args:
            file_path (str): Path to the JSON file containing evaluation scores.
            metric_type (str): Type of metrics, either 'traditional' or 'rag'.

        Raises:
            ValueError: If the metric_type is not 'traditional' or 'rag'.
        """
        with open(file_path, "r") as f:
            evaluator_outputs = json.load(f)

        for evaluator_output in evaluator_outputs:
            approach_id = evaluator_output["app_id"]

            if metric_type == "traditional":
                bleu_score = evaluator_output.get("BLEU", 0)
                meteor_score = evaluator_output.get("METEOR", 0)
                rouge_scores = evaluator_output.get("ROUGE", {"rouge1": 0, "rouge2": 0, "rougeL": 0})
                bert_scores = evaluator_output.get("BERT", {"precision": 0, "recall": 0, "f1": 0})
                ldfacts_score = evaluator_output.get("LDFACTS", 0)
                bart_score = evaluator_output.get("BART", 0)

                formatted_scores = {
                    "bleu": round(bleu_score, 5),
                    "meteor": round(meteor_score, 5),
                    "rouge1": round(rouge_scores["rouge1"], 5),
                    "rouge2": round(rouge_scores["rouge2"], 5),
                    "rougel": round(rouge_scores["rougeL"], 5),
                    "bert_precision": round(bert_scores["precision"], 5),
                    "bert_recall": round(bert_scores["recall"], 5),
                    "bert_f1": round(bert_scores["f1"], 5),
                    "ldfacts": round(np.exp(ldfacts_score), 5),
                    "bart": round(np.exp(bart_score), 5),
                }

                if approach_id in self.trad_scores_data:
                    self.trad_scores_data[approach_id].update(formatted_scores)
                else:
                    self.trad_scores_data[approach_id] = formatted_scores

            elif metric_type == "rag":
                toxicity = evaluator_output.get("toxicity/v1/mean", 0)
                flesch_kincaid_grade_level = evaluator_output.get("flesch_kincaid_grade_level/v1/mean", 0)
                ari_grade_level = evaluator_output.get("ari_grade_level/v1/mean", 0)
                rouge1 = evaluator_output.get("rouge1/v1/mean", 0)
                rouge2 = evaluator_output.get("rouge2/v1/mean", 0)
                rougel = evaluator_output.get("rougeL/v1/mean", 0)
                rougelsum = evaluator_output.get("rougeLsum/v1/mean", 0)
                answer_relevancy = evaluator_output.get("answer_relevance/v1/mean", 0)
                context_relevancy = evaluator_output.get("context_relevancy", 0)
                groundedness = evaluator_output.get("faithfulness/v1/mean", 0)

                formatted_scores = {
                    "toxicity": round(toxicity, 5),
                    "flesch_kincaid_grade_level\n(normalized to (15, 17))": flesch_kincaid_grade_level,
                    "ari_grade_level\n(normalized to (19, 21))": ari_grade_level,
                    "rouge1": round(rouge1, 5),
                    "rouge2": round(rouge2, 5),
                    "rougel": round(rougel, 5),
                    "rougelsum": round(rougelsum, 5),
                    "answer_relevancy": (answer_relevancy - 1) / 4,
                    "context_relevancy": round(context_relevancy, 5),
                    "groundedness": (groundedness - 1) / 4,
                }

                if approach_id in self.rag_scores_data:
                    self.rag_scores_data[approach_id].update(formatted_scores)
                else:
                    self.rag_scores_data[approach_id] = formatted_scores

            else:
                raise ValueError("Unknown metric_type. Use 'traditional' or 'rag'.")

    def normalize_flesch_ari(self):
        """
        Normalizes the 'flesch_kincaid_grade_level' and 'ari_grade_level' scores in the RAG data.
        """
        # Define the new ranges for flesch and ari
        flesch_min, flesch_max = 15, 17
        ari_min, ari_max = 19, 21

        for scores in self.rag_scores_data.values():
            # Clip the scores to ensure they are within the expected range
            flesch_score = max(min(scores["flesch_kincaid_grade_level\n(normalized to (15, 17))"],
                                   flesch_max),
                               flesch_min)
            ari_score = max(min(scores["ari_grade_level\n(normalized to (19, 21))"],
                                ari_max),
                            ari_min)

            scores["flesch_kincaid_grade_level\n(normalized to (15, 17))"] = ((flesch_score - flesch_min) /
                                                                              (flesch_max - flesch_min))
            scores["ari_grade_level\n(normalized to (19, 21))"] = ((ari_score - ari_min) /
                                                                   (ari_max - ari_min))

    @staticmethod
    def plot_scores(scores_data, title, save_path, ordered_approaches: Optional[List[str]] = None,
                    figsize: Optional[Tuple[float, float]] = (10, 6), rotation=45):
        """
        Generates and saves a bar chart visualizing the scores for each approach.

        Args:
            scores_data (dict): The scores data to plot.
            title (str): The title of the plot.
            save_path (str): The path to save the plot.
            ordered_approaches (list): The list of approaches to plot in order
                                       (default is None, which plots all approaches).
            figsize (tuple): The size of the figure (default is (10, 6)).
            rotation (int): The rotation angle for x-axis labels (default is 45).
        """
        if scores_data:
            # Filter and order approaches if ordered_approaches is provided
            if ordered_approaches:
                filtered_scores_data = {approach: scores_data[approach] for approach in ordered_approaches if
                                        approach in scores_data}
            else:
                filtered_scores_data = scores_data

            score_types = sorted(list(filtered_scores_data[list(filtered_scores_data.keys())[0]].keys()))
            data = []

            for approach, scores in filtered_scores_data.items():
                for score_type in score_types:
                    data.append({
                        "approach": approach,
                        "score_type": score_type,
                        "value": scores[score_type]
                    })

            df = pd.DataFrame(data)

            plt.figure(figsize=figsize)
            palette = sns.color_palette("Blues", n_colors=len(filtered_scores_data))
            sns.barplot(x="score_type", y="value", hue="approach", data=df, palette=palette)
            plt.title(title)
            plt.xlabel("Metrics")
            plt.ylabel("Values")
            plt.xticks(rotation=rotation)
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="app_id")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")

            sota_values = {'rouge1': 0.50950, 'rouge2': 0.21930, 'rougel': 0.45610}
            for score_type, sota_value in sota_values.items():
                plt.scatter(x=[score_type], y=[sota_value], color='red', s=100, marker='*', zorder=10)
                plt.plot([score_type, score_type], [0, sota_value], color='red', linestyle=':', linewidth=1)

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            """
            ---------------------------Temporary Usage for Visualization of only RAG Metrics----------------------------
            import json
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            # Load the JSON data
            file_path = 'data/outputs/rag_metric.json'
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # List of approaches to include
            included_approaches = ["bart_sentence_window", "bart_auto_merging", "bart_knowledge_graph"]
            
            # Initialize a list to hold the processed data
            processed_data = []
            
            # Process the JSON data
            for entry in data:
                if entry["app_id"] in included_approaches:
                    processed_data.append({
                        "approach": entry["app_id"],
                        "answer_relevance": (entry["answer_relevance/v1/mean"] - 1) / 4,
                        "groundedness": (entry["faithfulness/v1/mean"] - 1) / 4,
                        "context_relevancy": entry["context_relevancy"]
                    })
            
            # Convert processed data to DataFrame
            df = pd.DataFrame(processed_data)
            
            # Melt the DataFrame for easier plotting with seaborn
            df_melted = df.melt(id_vars=["approach"], var_name="score_type", value_name="value")
            
            # Plotting
            palette = sns.color_palette("Blues", n_colors=len(df["approach"].unique()))
            sns.barplot(x="score_type", y="value", hue="approach", data=df_melted, palette=palette)
            plt.title("RAG Evaluation Metrics by Approach (Mean) on XSUM with BART")
            plt.xlabel("Metrics")
            plt.ylabel("Values")
            plt.xticks(rotation=45)
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Approach")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
            plt.show()
            ---------------------------Temporary Usage for Visualization of only RAG Metrics----------------------------
            """


def reorder_rows(df, app_id_column, order_dict):
    """
    Reorder the rows of a DataFrame based on the custom order provided in order_dict.

    Args:
        df (pd.DataFrame): The DataFrame to reorder.
        app_id_column (str): The name of the column containing the app_id values.
        order_dict (dict): A dictionary specifying the order and number of rows for each app_id value.

    Returns:
        pd.DataFrame: The reordered DataFrame.
    """
    ordered_df = pd.DataFrame()
    for app_id_value, count in order_dict.items():
        filtered_df = df[df[app_id_column] == app_id_value]
        ordered_df = pd.concat([ordered_df, filtered_df.head(count)], ignore_index=True)
    return ordered_df


def plot_kde(df, save_path, id_column, order_dict, title=None):
    """
    Plot KDE for the given DataFrame and save the plot.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        save_path (str): The path to save the plot.
        id_column (str): The column to reorder the rows by.
        order_dict (dict): The desired order and count for the rows.
        title (str): The title of the plot.
    """
    df = reorder_rows(df, id_column, order_dict)
    sns.set_style("whitegrid")
    num_cols = len(df.columns) - 1  # Subtract 1 to ignore the 'app_id' column
    num_cols_per_row = num_cols // 2  # Calculate the number of columns per row

    fig, axs = plt.subplots(2, num_cols_per_row, figsize=(5 * num_cols_per_row, 10), squeeze=False)  # Create subplots
    axs = axs.flatten()  # Flatten the axs array for easier indexing

    for i, col in enumerate(df.columns[1:]):  # Exclude the 'app_id' column
        if col in ['LDFACTS', 'BART']:
            df[col] = np.exp(df[col])  # Apply exponential transformation
        if col in ['groundedness', 'answer_relevancy']:
            df[col] = (df[col] - 1) / 4
        sns.kdeplot(data=df, x=col, hue='app_id', ax=axs[i], fill=True, warn_singular=False)
        axs[i].set_title(f'kde analysis ({col})')
        axs[i].set_ylabel('density')

    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)  # Adjust the top to fit the title
    plt.savefig(save_path)
    plt.close()


def kde_analysis(csv_file_path1, csv_file_path2, save_path1, save_path2, dataset='xsum', title=None):
    """
    Analyze and compare the distributions of scores across two dataframes read from CSV files independently.

    Args:
        title (str): The title of the plot.
        csv_file_path1 (str): The file path to the first CSV file.
        csv_file_path2 (str): The file path to the second CSV file.
        save_path1 (str): The path to save the plot for the first CSV file.
        save_path2 (str): The path to save the plot for the second CSV file.
        dataset (str): The dataset to analyze the scores.
    """
    df1 = pd.read_csv(csv_file_path1)
    df2 = pd.read_csv(csv_file_path2)

    if dataset == 'xsum':
        order_dict = {
            'default': 100,
            'sentence_window_3': 100,
            'sentence_window_5': 100,
            'auto_merging_1024_256_64': 100,
            'auto_merging_2048_512_128': 100,
            'extractive': 100,
            'bertopic': 20,
            'lda': 20,
            'knowledge_graph': 50
        }
    elif dataset == 'xsum-bart':
        order_dict = {
            'bart_sentence_window': 20,
            'bart_auto_merging': 20,
            'bart_extractive': 20,
            'bart_knowledge_graph': 20
        }
    elif dataset == 'multi-news':
        order_dict = {
            'multi_news_default': 10,
            'multi_news_sw': 10,
            'multi_news_am': 10,
            'multi_news_ext': 10,
            'multi_news_lda': 10,
            'multi_news_bertopic': 10
        }
    else:
        return 'INVALID DATASET'

    plot_kde(df1, save_path1, 'app_id', order_dict, title)
    plot_kde(df2, save_path2, 'app_id', order_dict, title)
