"""
This module provides an implementation for a retriever that uses a language model (LLM)
and an embedding model to index documents and perform retrieval operations.

Classes:
    Retriever: Initializes the retrieval system and provides methods for indexing documents
    and querying them using the configured language model and embedding model.
"""

from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM


class Retriever:
    """
    A retriever that uses a language model (LLM) and an embedding model to index documents
    and perform retrieval operations.

    Attributes:
        query_engine (QueryEngine): The query engine used to perform retrieval operations.

    Methods:
        __init__(documents: list[Document], llm: BaseLLM, embed_model: BaseEmbedding):
            Initializes the retrieval system by setting up the embedding model,
            language model, and indexing the documents.
        initialize_index(documents: list[Document], embed_model: BaseEmbedding) -> VectorStoreIndex:
            Selects and initializes the appropriate index type based on the retrieval type specified.
    """

    def __init__(
        self,
        documents: list[Document],
        llm: BaseLLM,
        embed_model: BaseEmbedding,
    ):
        """
        Initializes the retrieval system by setting up the embedding model,
        language model, and indexing the documents.

        Parameters:
            documents (list[Document]): The list of documents to be indexed.
            llm (BaseLLM): The large language model to use for processing queries.
            embed_model (BaseEmbedding): The embedding model to use for creating vector representations of documents.
        """
        index = self.initialize_index(documents, embed_model)

        self.query_engine = index.as_query_engine(llm=llm)

    @staticmethod
    def initialize_index(
        documents: list[Document],
        embed_model: BaseEmbedding,
    ) -> VectorStoreIndex:
        """
        Selects and initializes the appropriate index type based on the retrieval type specified.

        Parameters:
            documents (list[Document]): The list of documents to index.
            embed_model (BaseEmbedding): The embedding model to use for creating vector representations of documents.

        Returns:
            VectorStoreIndex: The appropriately configured index based on the retrieval type.
        """
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        return VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
