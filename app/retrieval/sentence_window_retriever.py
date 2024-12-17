"""
This module provides an implementation for a retriever that uses a sentence window approach
to improve contextual understanding by considering sentences within a specified window size.
It utilizes a language model (LLM) and an embedding model to index documents and perform
retrieval operations.

Classes:
    SentenceWindowRetriever: Initializes the retrieval system using sentence windows and
    provides methods for indexing documents and querying them using the configured
    language model and embedding model.
"""

from llama_index.core import Document, ServiceContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)


class SentenceWindowRetriever:
    """
    A retriever that uses a sentence window approach to improve contextual understanding
    by considering sentences within a specified window size. It uses a language model (LLM)
    and an embedding model to index documents and perform retrieval operations.

    Attributes:
        query_engine (QueryEngine): The query engine used to perform retrieval operations.

    Methods:
        __init__(documents: list[Document], llm: BaseLLM, embed_model: BaseEmbedding, window_size: int = 3,
        similarity_top_k: int = 5, rerank_top_n: int = 5):
            Initializes the retrieval system by setting up the embedding model, language model, and indexing
            the documents using sentence windows.
        initialize_index(documents: list[Document], window_size: int, llm: BaseLLM, embed_model: BaseEmbedding) ->
        VectorStoreIndex:
            Selects and initializes the appropriate index type based on the retrieval type specified.
        initialize_query_engine(index: VectorStoreIndex, similarity_top_k: int, rerank_top_n: int) -> QueryEngine:
            Configures and returns a query engine designed for sentence window retrieval, including dynamic result
            merging and reranking.
    """

    def __init__(
        self,
        documents: list[Document],
        llm: BaseLLM,
        embed_model: BaseEmbedding,
        window_size: int = 3,
        similarity_top_k: int = 5,
        rerank_top_n: int = 5,
    ):
        """
        Initializes the retrieval system by setting up the embedding model,
        language model, and indexing the documents using sentence windows.

        Parameters:
            documents (list[Document]): The list of documents to be indexed.
            llm (BaseLLM): The large language model to use for processing queries.
            embed_model (BaseEmbedding): The embedding model to use for creating vector representations of documents.
            window_size (int, optional): Number of sentences to include in each window; defaults to 3.
            similarity_top_k (int, optional): The number of top similar results to retrieve initially; defaults to 5.
            rerank_top_n (int, optional): The number of results to rerank to refine the search results; defaults to 5.
        """
        index = self.initialize_index(
            documents=documents,
            window_size=window_size,
            llm=llm,
            embed_model=embed_model,
        )
        self.query_engine = self.initialize_query_engine(
            index=index, similarity_top_k=similarity_top_k, rerank_top_n=rerank_top_n
        )

    @staticmethod
    def initialize_index(
        documents: list[Document],
        window_size: int,
        llm: BaseLLM,
        embed_model: BaseEmbedding,
    ) -> VectorStoreIndex:
        """
        Selects and initializes the appropriate index type based on the retrieval type specified.

        Parameters:
            documents (list[Document]): The list of documents to index.
            window_size (int): Number of sentences to include in each window.
            llm (BaseLLM): The large language model to use.
            embed_model (BaseEmbedding): The embedding model to use for creating vector representations of documents.

        Returns:
            VectorStoreIndex: The appropriately configured index based on the retrieval type.
        """
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        sentence_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            node_parser=node_parser,
        )

        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context, show_progress=True,
        )

        return sentence_index

    @staticmethod
    def initialize_query_engine(
        index: VectorStoreIndex, similarity_top_k: int, rerank_top_n: int
    ):
        """
        Configures and returns a query engine designed for sentence window retrieval,
        including dynamic result merging and reranking.

        Parameters:
            index (VectorStoreIndex): The index to be used for querying.
            similarity_top_k (int): The number of top similar results to retrieve initially.
            rerank_top_n (int): The number of results to rerank to refine the search results.

        Returns:
            QueryEngine: A query engine that includes capabilities for sentence window retrieval
            and reranking results.
        """
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )

        return index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
        )
