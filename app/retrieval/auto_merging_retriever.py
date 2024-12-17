"""
This module provides a class for creating and utilizing an automerging retriever query engine with hierarchical node
parsing. It leverages large language models (LLMs) and embedding models to create a structured retrieval system that
can dynamically merge and rerank results based on query relevance.

Classes:
    AutoMergingRetrieverQueryEngine: A class to initialize and configure an automerging retriever query engine.

Functions:
    initialize_index(documents: list[Document], chunk_sizes: list[int], llm: BaseLLM, embed_model: BaseEmbedding) ->
    VectorStoreIndex:
        Builds an automerging index using hierarchical node parsing to create a structured retrieval system.

    initialize_query_engine(index: VectorStoreIndex, similarity_top_k: int, rerank_top_n: int) -> RetrieverQueryEngine:
        Configures and returns a query engine designed for automerging retrieval, which includes dynamic result merging
        and reranking.
"""

from llama_index.core import Document, ServiceContext, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever


class AutoMergingRetrieverQueryEngine:
    """
    A class to create and utilize an automerging retriever query engine with hierarchical node parsing.

    This class initializes an index that can dynamically merge results based on query relevance
    and rerank the results using a sentence transformer model.
    """

    def __init__(
            self,
            documents: list[Document],
            llm: BaseLLM,
            embed_model: BaseEmbedding,
            chunk_sizes: list[int] = [2048, 512, 128],
            similarity_top_k: int = 5,
            rerank_top_n: int = 5,
    ):
        """
        Initializes the AutoMergingRetrieverQueryEngine with the provided documents, language model,
        and embedding model.

        Parameters:
            documents (list[Document]): The documents to be indexed.
            llm (BaseLLM): The large language model to use for text generation and processing.
            embed_model (BaseEmbedding): The embedding model to use for creating vector representations.
            chunk_sizes (list[int], optional): Sizes of document chunks for hierarchical parsing.
            Defaults to [2048, 512, 128].
            similarity_top_k (int, optional): The number of top similar results to retrieve initially. Defaults to 5.
            rerank_top_n (int, optional): The number of results to rerank to refine the search results. Defaults to 5.
        """
        index = self.initialize_index(
            documents=documents,
            chunk_sizes=chunk_sizes,
            llm=llm,
            embed_model=embed_model,
        )
        self.query_engine = self.initialize_query_engine(
            index=index,
            similarity_top_k=similarity_top_k,
            rerank_top_n=rerank_top_n,
        )

    @staticmethod
    def initialize_index(
            documents: list[Document],
            chunk_sizes: list[int],
            llm: BaseLLM,
            embed_model: BaseEmbedding,
    ) -> VectorStoreIndex:
        """
        Builds an automerging index that utilizes hierarchical node parsing to create
        a structured retrieval system, which can merge results dynamically based on query relevance.

        Parameters:
            documents (list[Document]): The documents to be indexed.
            llm (BaseLLM): The large language model to use.
            embed_model (BaseEmbedding): The embedding model to use.
            chunk_sizes (list[int]): Sizes of document chunks for hierarchical parsing; defaults to [2048, 512, 128].

        Returns:
            VectorStoreIndex: An index designed for automerging retrieval.
        """
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)

        merging_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
        )

        # Provide non-leaves in storage for lookup
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            service_context=merging_context,
            show_progress=True,
        )

        return automerging_index

    @staticmethod
    def initialize_query_engine(
            index: VectorStoreIndex,
            similarity_top_k: int,
            rerank_top_n: int,
    ) -> RetrieverQueryEngine:
        """
        Configures and returns a query engine designed for automerging retrieval,
        which uses a base retriever and supports dynamic result merging and reranking.

        Parameters:
            index (VectorStoreIndex): The index to be used for querying.
            similarity_top_k (int): The number of top similar results to retrieve initially.
            rerank_top_n (int): The number of results to rerank to refine the search results.

        Returns:
            RetrieverQueryEngine: A query engine that includes capabilities for
            automerging and reranking results.
        """
        base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = AutoMergingRetriever(base_retriever, index.storage_context)
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )

        return RetrieverQueryEngine.from_args(retriever, node_postprocessors=[rerank])
