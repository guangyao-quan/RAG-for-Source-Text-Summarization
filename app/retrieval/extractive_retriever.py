"""
This module provides an implementation for an extractive retriever that uses
a transformer-based model for text summarization and retrieves sentences
relevant to a specific query from indexed documents.

Classes:
    ExtractiveRetriever: Retrieves relevant text snippets based on a query
    using extractive summarization.
    ExtractiveRetrievalQueryEngine: Initializes and configures an extractive retriever query engine.
"""

from typing import List

from langchain_core.language_models import BaseLLM
from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from nltk import sent_tokenize
from summarizer import Summarizer as Extractor


class ExtractiveRetriever(BaseRetriever):
    """
    An extractive retriever that uses a transformer-based model to perform extractive
    summarization on text data and retrieves sentences that are relevant to a specific
    query from an index.

    Attributes:
        transformer_name (str): Name of the transformer model used for summarization.
        extraction_ratio (float): Ratio of the text to be extracted as a summary.

    Methods:
        _retrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
            Processes a query and returns a list of extracted sentences as NodeWithScore.
    """

    def __init__(
        self,
        transformer_name: str = "bert-base-uncased",
        extraction_ratio: float = 0.3,
    ):
        """
        Initializes the ExtractiveRetriever with the specified transformer model and extraction ratio.

        Parameters:
            transformer_name (str, optional): The name of the transformer model to use for summarization.
                                               Defaults to "bert-base-uncased".
            extraction_ratio (float, optional): The ratio of the text to be extracted as a summary.
                                                Defaults to 0.3.
        """
        super().__init__()
        self.extractor = Extractor(model=transformer_name)
        self.extraction_ratio = extraction_ratio

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve relevant text snippets from the index based on the provided query.

        Parameters:
            query_bundle (QueryBundle): The query bundle containing the query string.

        Returns:
            List[NodeWithScore]: A list of NodeWithScore, each containing a relevant
                                 extracted text snippet.
        """
        extracts = sent_tokenize(
            self.extractor(query_bundle.query_str, ratio=self.extraction_ratio)
        )
        return [NodeWithScore(node=TextNode(text=text)) for text in extracts]


class ExtractiveRetrievalQueryEngine:
    """
    A query engine that utilizes an extractive retriever to process queries and
    return relevant text snippets using a transformer-based summarization model.

    Attributes:
        query_engine (RetrieverQueryEngine): The configured query engine for extractive retrieval.

    Methods:
        __init__(llm: BaseLLM, extractive_transformer_name: str, extraction_ratio: float = 0.3):
            Initializes the ExtractiveRetrievalQueryEngine with the specified parameters.
    """

    def __init__(
        self,
        llm: BaseLLM,
        extractive_transformer_name: str = "bert-base-uncased",
        extraction_ratio: float = 0.3,
    ):
        """
        Initializes the ExtractiveRetrievalQueryEngine with the specified parameters.

        Parameters:
            llm (BaseLLM): The large language model to use for processing queries.
            extractive_transformer_name (str): The name of the transformer model to use for summarization.
            extraction_ratio (float, optional): The ratio of the text to be extracted as a summary.
                                                Defaults to 0.3.
        """
        retriever = ExtractiveRetriever(
            transformer_name=extractive_transformer_name,
            extraction_ratio=extraction_ratio,
        )
        self.query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
