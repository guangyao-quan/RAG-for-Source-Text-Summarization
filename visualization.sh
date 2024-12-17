# Define variables for paths
TRAD_FILE="data/outputs/trad_metric.json"
RAGAS_FILE="data/outputs/ragas_metric.json"
MLFLOW_FILE="data/outputs/mlflow_metric.json"
CSV_FILE_PATHS=(
                "data/outputs/trad_kde_analysis.csv"
                "data/outputs/rag_kde_analysis.csv"
                )
OUTPUT_DIR="data/outputs/plots"

# Convert the CSV file paths and approach names arrays to space-separated strings
CSV_FILE_PATHS_STR="${CSV_FILE_PATHS[@]}"

# Run the visualization script
python app/evaluation/VisualizeMetric/visualization.py "$TRAD_FILE" "$RAGAS_FILE" "$MLFLOW_FILE" $CSV_FILE_PATHS_STR "$OUTPUT_DIR"
