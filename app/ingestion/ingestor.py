"""
This module handles the ingestion and indexing of documents for retrieval purposes using
language models and embedding technologies. It facilitates the setup of language model and
embedding model configurations, loads documents from specified sources, and creates an
indexed repository that can be used for efficient search and retrieval tasks.

Classes:
    Ingestor: Manages the loading of source documents, initialization of the language model
              and embedding model, and the creation of a searchable index.

Dependencies:
    os: Used to fetch environment variables.
    pathlib: Used for handling file system paths.
    typing: Used for type annotations.
    openai: Used to interact with OpenAI's APIs for language model functionalities.
    llama_index.core: Contains core functionality for document management and indexing.
    llama_index.core.base.embeddings.base: Provides the base class for embedding models.
    llama_index.core.indices.base: Provides the base class for indices.
    llama_index.llms.openai: Provides an interface to OpenAI language models.
    ..configs: Custom module for configuration constants.

Usage:
    The module is designed to be used as a part of a larger system that requires document
    retrieval capabilities. It can be initialized with specific language and embedding models,
    load documents from the filesystem, and create an indexed database that allows for
    quick retrieval based on various search criteria.

Example:
    Creating an Ingestor instance with VectorStoreIndex and OpenAI embeddings, loading default sources,
    and initializing an index:

        from retrieval_module import Ingestor, IndexType, EmbeddingModel

        embedding_model = OpenAIEmbedding()
        ingestor = Ingestor(IndexType.VECTOR_INDEX, embedding_model)
        ingestor.load_sources(["/path/to/documents"])
        index = ingestor.initialize_index()

    This instance can now be used to interact with the indexed data through various defined methods
    or further customized operations.

Environment Variables:
    OPENAI_API_KEY: The API key required to authenticate requests to OpenAI's services.
"""

import os
from pathlib import Path
from typing import List, Union

import openai
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.base import BaseIndex

from ..configs import IndexType

# Set the OpenAI API key from environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY")


class Ingestor:
    """
    Manages the ingestion of source documents, initialization of embedding models,
    and the creation of a document index for querying.

    This class is designed to be initialized with specific types of embedding models,
    which it then uses to process and index the provided documents.

    Attributes:
        embed_model (BaseEmbedding): The embedding model initialized based on
            the specified embedding type.
        documents (List[Document]): The aggregated documents created from the loaded sources.
        index_type (IndexType): The type of index to be created.
        paths (List[str]): A list of directories pointing to the documents to be loaded.

    Parameters:
        index_type (IndexType): The enum value specifying which index type to use.
        embed_model (BaseEmbedding): The embedding model to initialize.
        paths (List[str], optional): A list of directories pointing to the documents to be loaded.
    """

    def __init__(
        self,
        index_type: IndexType,
        embed_model: BaseEmbedding,
        paths: Union[List[str], None] = None,
    ):
        """
        Initializes the Ingestor with an embedding model and index type, loads source documents,
        and creates an index for them.

        Parameters:
            index_type (IndexType): The type of index to create, supported values are from the IndexType enum.
            embed_model (BaseEmbedding): The embedding model to initialize.
            paths (List[str], optional): A list of directories pointing to the documents to be loaded.
        """
        self.embed_model = embed_model
        self.documents = None
        self.index_type = index_type
        self.paths = paths

    def _load_sources(
        self,
        paths: List[str],
    ) -> None:
        """
        Loads documents from specified file paths and merges them into a single document
        attribute in the Ingestor instance.

        Parameters:
            paths (List[str]): A list of directories pointing to the documents to be loaded.
        """
        if len(paths) == 1:
            self.documents = SimpleDirectoryReader(
                input_dir=paths[0]
            ).load_data()
        else:
            self.documents = SimpleDirectoryReader(
                input_files=[
                    str(file)
                    for data_dir in paths
                    for file in Path(data_dir).iterdir()
                    if file.is_file()
                ]
            ).load_data()

    def initialize_index(self) -> BaseIndex:
        """
        Initializes the document index using the previously loaded documents and the
        initialized embedding model. This index is used for subsequent retrieval tasks.

        Returns:
            BaseIndex: The created index.
        """
        self._load_sources(self.paths)

        if self.index_type == IndexType.VECTOR_INDEX:
            index = VectorStoreIndex.from_documents(
                documents=self.documents, embed_model=self.embed_model
            )
        elif self.index_type == IndexType.KG_INDEX:
            index = KnowledgeGraphIndex.from_documents(
                documents=self.documents,
                max_triplets_per_chunk=5,
                include_embeddings=True,
            )
        else:
            index = None
        return index

    def get_documents(self) -> List[Document]:
        """
        Retrieves the loaded documents.

        Returns:
            List[Document]: The list of loaded documents.
        """
        if not self.documents:
            self._load_sources(self.paths)
        return self.documents
