"""
This module provides an implementation for ingesting datasets from Hugging Face into a search index.

Classes:
    HuggingFaceDatasetsIngestor: An ingestor for loading and indexing datasets from Hugging Face repositories.

Dependencies:
    logging: Used for logging information during the ingestion process.
    datasets: Used for loading and manipulating datasets from Hugging Face.
    llama_index.core: Contains core functionality for document management and indexing.
    llama_index.core.base.embeddings.base: Provides the base class for embedding models.
    app.configs: Custom module for configuration constants.
    app.ingestion.ingestor: Base ingestor class for handling document ingestion and indexing.

Usage:
    This module is designed to facilitate the ingestion of datasets from Hugging Face, allowing them to be
    indexed for efficient search and retrieval. It can be initialized with specific language and embedding models,
    load datasets from Hugging Face, and create an indexed database.

Example:
    Creating a HuggingFaceDatasetsIngestor instance with VectorStoreIndex and OpenAI embeddings, loading datasets,
    and initializing an index:

        from app.ingestion.huggingface_ingestor import HuggingFaceDatasetsIngestor
        from app.configs import IndexType
        from llama_index.core.base.embeddings.openai import OpenAIEmbedding

        embedding_model = OpenAIEmbedding()
        paths = [(("dataset_name", "config_name"), ["column1", "column2"])]
        ingestor = HuggingFaceDatasetsIngestor(IndexType.VECTOR_INDEX, embedding_model, paths)
        ingestor._load_sources(paths)
        index = ingestor.initialize_index()

Environment Variables:
    None
"""

import logging
from datasets import concatenate_datasets, load_dataset
from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from app.configs import IndexType
from app.ingestion.ingestor import Ingestor

logger = logging.getLogger(__name__)


class HuggingFaceDatasetsIngestor(Ingestor):
    """
    An ingestor for loading and indexing datasets from Hugging Face repositories.

    Attributes:
        paths (list): A list of tuples where each tuple contains a dataset path and configuration name,
                      and a list of keys specifying the columns to be ingested.

    Parameters:
        index_type (IndexType): The type of index to create.
        embed_model (BaseEmbedding): The embedding model to initialize.
        paths (list[tuple[tuple[str, str], list[str]]]): Paths of datasets in Hugging Face with keys to dataset columns
        to be ingested. Format: [((path, config_name), [key1, key2, key3, ...]), ...]
    """

    def __init__(
        self,
        index_type: IndexType,
        embed_model: BaseEmbedding,
        paths: list[tuple[tuple[str, str], list[str]]],
    ):
        """
        Initializes the HuggingFaceDatasetsIngestor with an index type and embedding model, and sets the paths.

        Parameters:
            index_type (IndexType): The type of index to create.
            embed_model (BaseEmbedding): The embedding model to initialize.
            paths (list[tuple[tuple[str, str], list[str]]]): Paths of datasets in Hugging Face with keys to dataset
            columns to be ingested.
        """
        super().__init__(index_type=index_type, embed_model=embed_model, paths=None)
        self.paths = paths

    def _load_sources(
        self,
        paths: list[tuple[tuple[str, str], list[str]]],
    ) -> None:
        """
        Loads the source documents from the specified paths and keys.

        Parameters:
            paths (list[tuple[tuple[str, str], list[str]]]): Paths of datasets in Hugging Face with keys to dataset
            columns to be ingested. Format: [((path, config_name), [key1, key2, key3, ...]), ...]
        """
        datasets = []
        for (path, config_name), keys in paths:
            dataset = load_dataset(path, config_name, split="train[:100]")
            # dataset = concatenate_datasets([dataset[split] for split in dataset.keys()])
            datasets.append((dataset, keys))

        logger.info("FINISHED LOADING DATASETS FROM HUGGINGFACE")

        self.documents = []
        for dataset, columns in datasets:
            for col in columns:
                for text in dataset[col]:
                    self.documents.append(Document(text=text)) if text != "" else None

        logger.info("FINISHED APPENDING NECESSARY COLUMNS")

        del datasets

        logger.info("FINISHED DELETING DATASETS")
