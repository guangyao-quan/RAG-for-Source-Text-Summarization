"""
This module defines various enumerations used in a system designed for leveraging and evaluating
Language Learning Models (LLMs). These enums categorize different types of models, embeddings,
indexing mechanisms, and retrieval strategies to be used within the system. The module enables
a standardized approach to configure and operate complex NLP functionalities.

Example:
    config = {
        'llm': LLMType.GPT35_TURBO,
        'embedding': EmbeddingType.OPENAI,
        'index_type': IndexType.VECTOR_INDEX,
        'retrieval_type': RetrievalType.SENTENCE_WINDOW
    }
"""

from enum import Enum


class LLMType(Enum):
    """
    Defines the types of Language Learning Models (LLM) available for use.
    """

    DEFAULT = 0
    GPT35_TURBO = 1
    ZEPHIR_7B_ALPHA = 2
    BART = 3


class EmbeddingType(Enum):
    """
    Enumerates different types of embeddings that can be utilized within the system.
    """

    DEFAULT = 0
    BGE_SMALL_EN = 1
    OPENAI = 2


class IndexType(Enum):
    """
    Specifies the type of indexing mechanism to be used by the system.
    """

    DEFAULT = 0  # no index needed for extractive retrieval
    VECTOR_INDEX = 1  # vector index
    KG_INDEX = 2  # knowledge graph index


class RetrievalType(Enum):
    """
    Specifies the types of retrieval mechanisms supported by the system.
    """

    DEFAULT = 0
    SENTENCE_WINDOW = 1
    AUTO_MERGING = 2
    EXTRACTIVE_RETRIEVAL = 3
    TOPIC_EXTRACTION = 4
    KG_RETRIEVAL = 5


class IngestionSource(Enum):
    """
    Specifies the source of data to be ingested by the system.
    """
    DEFAULT = 0
    HF = 1
    LOCAL = 2


class TopicExtractorType(Enum):
    """
    Specifies the type of topic extraction mechanism to be used by the system.
    """
    LDA = 0
    BERTOPIC = 1
