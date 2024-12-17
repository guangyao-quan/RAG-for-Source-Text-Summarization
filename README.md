# RAG for Source Text Summarization
RAG (retrieval augmented generation) is a big hype right now to infuse LLM outputs with relevant context. But can we leverage the approaches for the summarization of source texts? How is the performance compared to traditional summarization models?

## Team members
* [GÖÇMEN, Hamdi Berke (M.Sc. Informatics Student, TUM)](https://www.linkedin.com/in/berkegocmen/)
* [QUAN, Guangyao (M.Sc. Informatics Student, TUM)](https://www.linkedin.com/in/guangyaoquan)
* [WIEHE, Luca Mattes (M.Sc. Robotics, Cognition, Intelligence Student, TUM)](https://www.linkedin.com/in/lucawiehe/)

## Supervisor
* [ANSCHÜTZ, Miriam (M.Sc./PhD Candidate, Social Computing Group, TUM)](https://www.linkedin.com/in/miriam-anschütz/)

## Task description
Your task is to use RAG systems and evaluate their performance on text simplification benchmarks like [CNN-daily](https://huggingface.co/datasets/cnn_dailymail) or [XSUM](https://huggingface.co/datasets/EdinburghNLP/xsum). The steps comprise:
1. Get familiar with RAG. A possible starting point is [this tutorial](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/).
2. Set up a pipeline to run RAG for summarization. What are good questions to yield a summary?
3. Extend your experiments with different retrievers, LLMS, and benchmark datasets.

## Project structure
This project is organized as follows:

- `app/`
    - `evaluation/`
        - `EvaluateSummary`
        - `ExternalHelper`
        - `SaveMetric`
        - `VisualizeMetric`
        - `__init__.py`
        - `benchmark.py`
        - `time_tracking.py`
    - `ingestion/`
        - `__init__.py`
        - `huggingface_datasets_ingestor.py`
        - `ingestor.py`
    - `llms/`
        - `__init__.py`
        - `bart.py`
    - `retrieval/`
        - `__init__.py`
        - `auto_merging_retriever.py`
        - `extractive_retriever.py`
        - `kg_retriever.py`
        - `retriever.py`
        - `sentence_window_retriever.py`
        - `simple_query_engine.py`
        - `topic_extractor.py`
        - `topic_retriever.py`
    - `__init__.py`
    - `configs.py`
    - `rag.py`
- `data/`
    - `indexes/`
    - `presentations/`
    - `report/`
    - `source-texts/`
    - `test-folder/`
- `.gitattributes`
- `.gitignore`
- `.pre-commit-config.yaml`
- `benchmark.sh`
- `visualization.sh`
- `poetry.lock`
- `pyproject.toml`
- `rag_pipeline_demo.ipynb`
- `README.md`

Each directory and file plays a specific role in the project:

- `/app/` - Holds the main RAG application code.
  - `/evaluation/` - Includes the evaluation module for assessing model performance.
  - `/ingestion/` - Comprises the data ingestion module, responsible for ingesting datasets.
  - `/llms/` - Contains the implementation of using BART as LLM.
  - `/retrieval/` - Features the retrieval module for fetching contexts.
  - `/configs.py` - Defines different types for components inside a RAGBuilder.
  - `/rag.py` - Implements the RAGBuilder returning the specific query engine.
- `/data/` - The main directory for dataset storage, indexes, and source texts.
  - `/indexes/` - Where index files for the datasets are stored.
  - `/outputs/` - Saves all json, csv and png file outputs by pipeline. (auto-generated while benchmarking)
  - `/presentations/` - A directory for storing midterm presentations and poster.
  - `/report/` - A folder for saving final report in pdf and all latex source files.
  - `/source-texts/` - The location for raw source texts used in the project.
  - `/test-folder/` - A directory for testing purposes.
- `/.gitattributes` - Specifies attributes for Git repositories.
- `/.gitignore` - Specifies intentionally untracked files to ignore.
- `/.pre-commit-config.yaml` - Configurations for pre-commit hooks.
- `/benchmark.sh` - Shell script for running benchmark with configurations.
- `/visualization.sh` - Shell script for visualizing benchmark results.
- `/poetry.lock` - The lock file for Poetry, pinning specific versions of dependencies.
- `/pyproject.toml` - The configuration file for Poetry, defining the project and its dependencies.
- `/rag_pipeline_demo.ipynb` - Jupyter notebook for the RAG pipeline demonstration.
- `/README.md` - The readme file for the project, providing an overview and documentation.

## Environment setup

### Pre-requisites
Python package and environment management install:
- [Python 3.11](https://www.python.org/downloads/)
- [Poetry](https://www.python-poetry.org/)

### Setting Up the Environment
1. **Install Dependencies**

   Use Poetry to install the necessary dependencies:
   ```console
   poetry install
   ```
2. **Activate the Virtual Environment**

   Activate the virtual environment created by Poetry:
   ```console
   poetry shell
   ```

### Dependency management
We use `poetry` for dependency management.

#### Synchronize virtual environment
To synchronize the virtual environment with the requirements, you can use the following:
```console
poetry install --sync
```

#### Add new package
To add a new package to the project, use:
```console
poetry add <package-name>
```

### Pre-Commit hooks
Install pre-commit hooks.
```console
pre-commit install
```
Check if pre-commit hooks are installed correctly.
```console
pre-commit run --all-files
```

## How to Run Benchmarking
To run benchmarking for different approaches using the provided script, follow the steps below:

### Running the Benchmark
Use the following command to execute the benchmark with the specified parameters:
```console
./benchmark.sh
```

### Parameters
Please refer to `app/configs.py` for different parameters.
- `--llm_type`: The type of the language model to use. Example: `GPT35_TURBO`.
- `--embedding_type`: The type of embedding to use. Example: `BGE_SMALL_EN`.
- `--index_type`: The type of index to use. Example: `VECTOR_INDEX`.
- `--retrieval_type`: The type of retrieval to use. Example: `DEFAULT`.
- `--evaluation_mode`: The mode of evaluation. Options: `"both" (for most settings)`, `"traditional" (only for extractive setting)`, `"rag" (in general not useful)`.
- `--eval_path`: The path to the evaluation dataset. Example: `"EdinburghNLP/xsum"`.
- `--app_id`: The specific and distinct application ID. Example: `"default" (set it for distinguishing different approaches)`.
- `--num_samples`: The number of samples to use for the benchmark. Example: `100`.
- `--reset_json`: Whether to reset the JSON results file. Example: `1` (yes), `0` (no).
- `--reset_csv`: Whether to reset the CSV results file. Example: `1` (yes), `0` (no).

### Example
To run the benchmark with 100 samples and reset both JSON and CSV results files, use:
```console
python app/evaluation/benchmark.py --llm_type GPT35_TURBO --embedding_type BGE_SMALL_EN --index_type VECTOR_INDEX --retrieval_type DEFAULT --evaluation_mode "both" --eval_path "EdinburghNLP/xsum" --app_id "default" --num_samples 100 --reset_json 1 --reset_csv 1
```
```console
./benchmark.sh &
```
This command will start the benchmarking process using the above-specified configurations in the background.

### Additional Information
* If you are trying to use OpenAI models, please first remember to save your API key in `.env` at the root.
* If you are trying to use local models instead, please also first ensure you have enough GPU memory with at least 16GB.
