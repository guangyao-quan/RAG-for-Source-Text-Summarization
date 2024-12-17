"""
This module provides an implementation for a simple query engine that uses
a retriever which returns no retrieval results. It also includes an example
of an extractive retriever that uses a transformer-based model for text summarization
to retrieve sentences relevant to a specific query from indexed documents.

Classes:
    NoRetrivalRetriever: A retriever that returns no retrieval results.
    SimpleQueryEngine: Initializes a simple query engine using a retriever that performs no retrieval operations.
"""

from typing import List

from langchain_core.language_models import BaseLLM
from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore


class NoRetrivalRetriever(BaseRetriever):
    """
    A retriever that returns no retrieval results.

    Methods:
        _retrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
            Processes a query and returns an empty list of NodeWithScore.
    """

    def __init__(self):
        """
        Initializes the NoRetrivalRetriever.
        """
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Processes a query and returns an empty list of NodeWithScore.

        Parameters:
            query_bundle (QueryBundle): The query bundle containing the query string.

        Returns:
            List[NodeWithScore]: An empty list as this retriever performs no retrieval operations.
        """
        return []


class SimpleQueryEngine:
    """
    A query engine that utilizes a retriever which performs no retrieval operations.

    Attributes:
        query_engine (RetrieverQueryEngine): The query engine used to perform retrieval operations.

    Methods:
        __init__(llm: BaseLLM):
            Initializes the SimpleQueryEngine with the specified language model and a retriever
            that returns no retrieval results.
    """

    def __init__(
        self,
        llm: BaseLLM,
    ):
        """
        Initializes the SimpleQueryEngine with the specified language model and a retriever
        that returns no retrieval results.

        Parameters:
            llm (BaseLLM): The large language model to use for processing queries.
        """
        retriever = NoRetrivalRetriever()
        self.query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
