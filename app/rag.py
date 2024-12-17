"""
This module provides the implementation of a RAG (Retrieval-Augmented Generation) Builder. The RAGBuilder
class allows the setup and initialization of various components required for a retrieval-augmented
generation system, including the ingestion of documents, selection of language models (LLMs), embedding
models, and retrieval types.

Classes:
    RAGBuilder: A builder class to set up and initialize a retrieval-augmented generation system.

Dependencies:
    typing: Used for type annotations.
    llama_index.core.base.embeddings.base: Provides the base class for embedding models.
    llama_index.core.base.llms.base: Provides the base class for language models.
    llama_index.embeddings.huggingface: Provides Hugging Face-based embedding models.
    llama_index.embeddings.openai: Provides OpenAI-based embedding models.
    llama_index.llms.huggingface: Provides Hugging Face-based language models.
    llama_index.llms.openai: Provides OpenAI-based language models.
    app.configs: Custom module for configuration constants.
    app.ingestion.huggingface_datasets_ingestor: Custom Hugging Face datasets ingestor.
    app.ingestion.ingestor: Base ingestor class for handling document ingestion and indexing.
    app.retrieval.auto_merging_retriever: Custom auto-merging retriever query engine.
    app.retrieval.extractive_retriever: Custom extractive retriever query engine.
    app.retrieval.retriever: Base retriever class for handling document retrieval.
    app.retrieval.sentence_window_retriever: Custom sentence window retriever query engine.
    app.retrieval.simple_query_engine: Simple query engine for retrieval.

Example:
    Creating a RAGBuilder instance and setting up the components for retrieval-augmented generation:

        from app.builder import RAGBuilder
        from app.configs import EmbeddingType, LLMType, IndexType, IngestionSource, RetrievalType

        builder = RAGBuilder()
        builder.set_embedding_type(EmbeddingType.OPENAI)
               .set_llm_type(LLMType.GPT35_TURBO)
               .set_index_type(IndexType.VECTOR_INDEX)
               .set_ingestion_source_and_path(IngestionSource.LOCAL, ["/path/to/documents"])
               .set_retrieval_type(RetrievalType.DEFAULT)
        rag_system = builder.build()

Environment Variables:
    None
"""

from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI

from app.configs import (
    EmbeddingType,
    IndexType,
    IngestionSource,
    LLMType,
    RetrievalType,
    TopicExtractorType
)
from app.ingestion.huggingface_datasets_ingestor import HuggingFaceDatasetsIngestor
from app.ingestion.ingestor import Ingestor
from app.llms.bart import BartHuggingFaceLLM
from app.retrieval.auto_merging_retriever import AutoMergingRetrieverQueryEngine
from app.retrieval.extractive_retriever import ExtractiveRetrievalQueryEngine
from app.retrieval.topic_retriever import TopicRetrievalQueryEngine
from app.retrieval.kg_retriever import KGRetrievalQueryEngine
from app.retrieval.retriever import Retriever
from app.retrieval.sentence_window_retriever import SentenceWindowRetriever
from app.retrieval.simple_query_engine import SimpleQueryEngine
from app.evaluation.EvaluateSummary.time_tracker import time_tracker


class RAGBuilder:
    """
    A builder class to set up and initialize a retrieval-augmented generation (RAG) system.

    Attributes:
        _ingestion_source_and_path (dict | None): Dictionary to store the source and paths for ingestion.
        _retrieval_kwargs (dict | None): Dictionary to store additional keyword arguments for retrieval.
        _embedding_type (EmbeddingType | None): Enum value specifying the embedding type.
        _llm_type (LLMType | None): Enum value specifying the language model type.
        _index_type (IndexType | None): Enum value specifying the index type.
        _ingestor (Ingestor | None): Ingestor instance for handling document ingestion.
        _retrieval_type (RetrievalType | None): Enum value specifying the retrieval type.

    Methods:
        set_embedding_type(embedding_type: EmbeddingType): Sets the embedding type.
        _initialize_embedding_model() -> BaseEmbedding: Initializes the embedding model.
        set_llm_type(llm_type: LLMType): Sets the language model type.
        _initialize_llm() -> BaseLLM: Initializes the language model.
        set_index_type(index_type: IndexType): Sets the index type.
        set_ingestion_source_and_path(source: IngestionSource, paths: Any): Sets the ingestion source and paths.
        _initialize_ingestor(embed_model): Initializes the ingestor.
        set_retrieval_type(retrieval_type: RetrievalType, **kwargs): Sets the retrieval type and additional arguments.
        build(): Builds and returns the configured RAG system.
    """

    def __init__(self):
        """
        Initializes the RAGBuilder with default values for its attributes.
        """
        self._ingestion_source_and_path: dict | None = None
        self._retrieval_kwargs = None
        self._embedding_type: EmbeddingType | None = None
        self._llm_type: LLMType | None = None
        self._index_type: IndexType | None = None
        self._ingestor: Ingestor | None = None
        self._retrieval_type: RetrievalType | None = None

    def set_embedding_type(self, embedding_type: EmbeddingType):
        """
        Sets the embedding type for the RAG system.

        Parameters:
            embedding_type (EmbeddingType): The embedding type to set.

        Returns:
            RAGBuilder: The current instance of RAGBuilder.

        Raises:
            ValueError: If the provided embedding type is not supported.
        """
        if not isinstance(embedding_type, EmbeddingType):
            raise ValueError(
                "Given Embedding type is not supported. Use one of the embeddings from EmbeddingType Enum"
            )
        self._embedding_type = embedding_type
        return self

    def _initialize_embedding_model(self) -> BaseEmbedding:
        """
        Initializes the embedding model based on the set embedding type.

        Returns:
            BaseEmbedding: The initialized embedding model.

        Raises:
            ValueError: If the embedding type is not set before initialization.
        """
        if not self._embedding_type:
            raise ValueError("Set Embedding type before initialize embedding model")
        elif self._embedding_type == EmbeddingType.BGE_SMALL_EN:
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        elif self._embedding_type == EmbeddingType.OPENAI:
            embed_model = OpenAIEmbedding()
        return embed_model

    def set_llm_type(self, llm_type: LLMType):
        """
        Sets the language model type for the RAG system.

        Parameters:
            llm_type (LLMType): The language model type to set.

        Returns:
            RAGBuilder: The current instance of RAGBuilder.

        Raises:
            ValueError: If the provided language model type is not supported.
        """
        if not isinstance(llm_type, LLMType):
            raise ValueError(
                "Given LLM type is not supported. Use one of the LLMs from LLMType Enum"
            )
        self._llm_type = llm_type
        return self

    def _initialize_llm(self) -> BaseLLM:
        """
        Initializes the language model based on the set language model type.

        Returns:
            BaseLLM: The initialized language model.

        Raises:
            ValueError: If the language model type is not set before initialization.
        """
        if not self._llm_type:
            raise ValueError("Set LLM type before the initialization")
        elif self._llm_type == LLMType.GPT35_TURBO:
            llm = OpenAI(model="gpt-3.5-turbo")
        elif self._llm_type == LLMType.ZEPHIR_7B_ALPHA:
            llm = HuggingFaceLLM(
                model_name="HuggingFaceH4/zephyr-7b-alpha",
                tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
            )
        elif self._llm_type == LLMType.BART:
            llm = BartHuggingFaceLLM()
        else:
            llm = OpenAI(model="gpt-3.5-turbo")
        return llm

    def set_index_type(self, index_type: IndexType):
        """
        Sets the index type for the RAG system.

        Parameters:
            index_type (IndexType): The index type to set.

        Returns:
            RAGBuilder: The current instance of RAGBuilder.

        Raises:
            ValueError: If the provided index type is not supported.
        """
        if not isinstance(index_type, IndexType):
            raise ValueError(
                "Given Index type is not supported. Use one of the Index from IndexType Enum"
            )
        self._index_type = index_type
        return self

    def set_ingestion_source_and_path(self, source: IngestionSource, paths: Any):
        """
        Sets the ingestion source and paths for the RAG system.

        Parameters:
            source (IngestionSource): The ingestion source to set.
            paths (Any): The paths for ingestion.

        Returns:
            RAGBuilder: The current instance of RAGBuilder.

        Raises:
            ValueError: If the provided ingestion source is not supported.
        """
        if not isinstance(source, IngestionSource):
            raise ValueError("Given source is not supported")
        self._ingestion_source_and_path = {"source": source, "paths": paths}
        return self

    def _initialize_ingestor(self, embed_model):
        """
        Initializes the ingestor based on the ingestion source and paths.

        Parameters:
            embed_model (BaseEmbedding): The embedding model to use for ingestion.

        Returns:
            Ingestor: The initialized ingestor.
        """
        source = self._ingestion_source_and_path["source"]
        paths = self._ingestion_source_and_path["paths"]

        if source == IngestionSource.LOCAL:
            ingestor = Ingestor(
                index_type=self._index_type, embed_model=embed_model, paths=paths
            )
        elif source == IngestionSource.HF:
            ingestor = HuggingFaceDatasetsIngestor(
                index_type=self._index_type, embed_model=embed_model, paths=paths
            )
        return ingestor

    def set_retrieval_type(self, retrieval_type: RetrievalType, **kwargs):
        """
        Sets the retrieval type for the RAG system.

        Parameters:
            retrieval_type (RetrievalType): The retrieval type to set.
            **kwargs: Additional keyword arguments for retrieval configuration.

        Returns:
            RAGBuilder: The current instance of RAGBuilder.

        Raises:
            ValueError: If the provided retrieval type is not supported.
        """
        if not isinstance(retrieval_type, RetrievalType):
            raise ValueError("Given Retrieval Type is not supported")
        self._retrieval_type = retrieval_type
        self._retrieval_kwargs = kwargs
        return self

    def set_debug_mode(self, debug_mode: bool):
        """
        Sets the debug mode for the RAG object.

        Parameters:
            debug_mode (bool): A boolean value indicating whether to enable debug mode or not.

        Returns:
            self: The RAG object with the updated debug mode.

        """
        self._debug_mode = debug_mode
        return self

    def set_topic_extractor_type(self, topic_extractor_type: TopicExtractorType):
        """
        Sets the type of topic extractor to be used.

        Parameters:
        - topic_extractor_type (TopicExtractorType): The type of topic extractor.

        Returns:
        - self: The current instance of the class.

        """
        self._topic_extractor_type = topic_extractor_type
        return self

    @time_tracker
    def build(self):
        """
        Builds and returns the configured RAG system.

        Returns:
            QueryEngine: The configured query engine based on the specified retrieval type and components.
        """
        llm = self._initialize_llm()

        if self._retrieval_type == RetrievalType.EXTRACTIVE_RETRIEVAL:
            return ExtractiveRetrievalQueryEngine(
                llm=llm, **self._retrieval_kwargs
            ).query_engine

        embed_model = self._initialize_embedding_model()
        ingestor = self._initialize_ingestor(embed_model=embed_model)

        if self._retrieval_type == RetrievalType.DEFAULT:
            return Retriever(
                documents=ingestor.get_documents(),
                llm=llm,
                embed_model=embed_model,
            ).query_engine
        if self._retrieval_type == RetrievalType.SENTENCE_WINDOW:
            return SentenceWindowRetriever(
                documents=ingestor.get_documents(),
                llm=llm,
                embed_model=embed_model,
                **self._retrieval_kwargs,
            ).query_engine
        if self._retrieval_type == RetrievalType.AUTO_MERGING:
            return AutoMergingRetrieverQueryEngine(
                documents=ingestor.get_documents(),
                llm=llm,
                embed_model=embed_model,
                **self._retrieval_kwargs,
            ).query_engine
        if self._retrieval_type == RetrievalType.TOPIC_EXTRACTION:
            return TopicRetrievalQueryEngine(
                llm=llm,
                index=ingestor.initialize_index(),
                topic_extractor=self._topic_extractor_type,
                embed_model=embed_model,
                verbose=self._debug_mode
            ).query_engine
        if self._retrieval_type == RetrievalType.KG_RETRIEVAL:
            return KGRetrievalQueryEngine(
                llm=llm,
                index=ingestor.initialize_index(),
                embed_model=embed_model,
                verbose=self._debug_mode
            ).query_engine

        return SimpleQueryEngine(llm=llm).query_engine
