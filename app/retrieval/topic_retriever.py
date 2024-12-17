"""
This module provides implementations for a topic retriever and a corresponding query engine.
The retriever uses a topic extraction method (either LDA or BERTopic) to extract topics from
documents and retrieve relevant chunks based on these topics.

Classes:
    TopicRetriever: Uses topic extraction methods to retrieve relevant text chunks from indexed documents.
    TopicRetrievalQueryEngine: Initializes a query engine that utilizes the TopicRetriever.

Dependencies:
    numpy, sklearn.metrics.pairwise, langchain_core.language_models, llama_index.core, app.retrieval.topic_extractor,
    ..configs: Libraries and modules used for topic extraction, language model integration, and retrieval operations.
"""

import time

import numpy as np
from langchain_core.language_models import BaseLLM
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from sklearn.metrics.pairwise import cosine_similarity

from app.retrieval.topic_extractor import LDATopicExtractor, BERTopicExtractor
from ..configs import TopicExtractorType


class TopicRetriever(BaseRetriever):
    """
    A retriever that uses topic extraction methods (LDA or BERTopic) to retrieve relevant text chunks
    from indexed documents based on extracted topics.

    Attributes:
        index: The index used for topic retrieval.
        n_topics (int): The number of top topics to retrieve.
        topic_extractor: The topic extraction method (LDA or BERTopic).
        embedding_model: The embedding model used for document representation.
        chunks_per_topic (int): The number of chunks to retrieve per topic.
        verbose (bool): Whether to print verbose output.

    Methods:
        __init__(index, topic_extractor: TopicExtractorType, embed_model, verbose, n_topics=5, chunks_per_topic=1):
            Initializes the TopicRetriever with the specified parameters.
        _retrieve(query_bundle) -> list:
            Retrieves relevant chunks based on extracted topics.
        init_topic_extractor(topic_extractor: TopicExtractorType):
            Initializes and returns the appropriate topic extractor based on the specified type.
    """

    def __init__(self,
                 index,
                 topic_extractor: TopicExtractorType,
                 embed_model,
                 verbose,
                 n_topics=5,
                 chunks_per_topic=1):
        """
        Initializes the TopicRetriever with the specified parameters.

        Parameters:
            index: The index used for topic retrieval.
            topic_extractor (TopicExtractorType): The topic extractor used for extracting topics from documents.
            embed_model: The embedding model used for document representation.
            verbose (bool): Whether to print verbose output.
            n_topics (int, optional): The number of top topics to retrieve (default is 5).
            chunks_per_topic (int, optional): The number of chunks to retrieve per topic (default is 1).
        """
        super().__init__()
        self.index = index
        self.n_topics = n_topics
        self.topic_extractor = self.init_topic_extractor(topic_extractor)
        self.embedding_model = embed_model
        self.chunks_per_topic = chunks_per_topic
        self.verbose = verbose

    def _retrieve(self, query_bundle):
        """
        Retrieves relevant chunks by ignoring the query_bundle and using information from extracted topics instead.

        Parameters:
            query_bundle (object): A query bundle that is necessary to match the signature of the parent class.

        Returns:
            list: A list of NodeWithScore objects representing the most relevant chunks along with their scores.
        """
        if self.verbose:
            start_time = time.time()

        doc_values = self.index.docstore.docs.values()
        vector_store = self.index.vector_store

        chunks = [value.get_content() for value in doc_values]

        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done retrieving {len(chunks)} chunks. Time: {duration:.6f}s")

        # Extract topics for each chunk
        if self.verbose:
            start_time = time.time()

        topics = self.topic_extractor.extract_topics(chunks)

        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done extracting Topics. Time: {duration:.6f}s")

        if self.verbose:
            start_time = time.time()

        # Get top n topics
        top_topics = topics.head(self.n_topics)

        # Get topic summaries
        topic_summaries = []
        for i, row in top_topics.iterrows():
            topic_keywords = row["Representation"]
            concat_keywords = " ".join(topic_keywords)
            topic_summaries.append(concat_keywords)

        # Duration of obtaining topic summaries
        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done obtaining Topic Summaries. Time: {duration:.6f}s")

        if self.verbose:
            start_time = time.time()

        # Obtain embeddings
        topic_embeddings = [self.embedding_model.get_text_embedding(topic_summary) for topic_summary in topic_summaries]
        chunk_keys = list(self.index.docstore.docs.keys())
        chunk_embeddings = list(vector_store.to_dict()["embedding_dict"].values())

        # Convert to numpy arrays for easier manipulation with cosine_similarity
        topic_embeddings = np.array(topic_embeddings)
        chunk_embeddings = np.array(chunk_embeddings)

        # Reshape embeddings if necessary
        if topic_embeddings.ndim == 1:
            topic_embeddings = topic_embeddings.reshape(1, -1)
        if chunk_embeddings.ndim == 1:
            chunk_embeddings = chunk_embeddings.reshape(1, -1)

        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done obtaining embeddings for chunks and topic. Time: {duration:.6f}s")

        if self.verbose:
            start_time = time.time()

        # Calculate cosine similarity between topic summaries and chunks
        similarities = cosine_similarity(topic_embeddings, chunk_embeddings)

        # Collect the most similar chunks for each topic summary
        nodes_with_scores = []
        for i, summary in enumerate(topic_summaries):
            # obtain index of most similar chunk
            most_similar_index = similarities[i].argmax()

            # obtain corresponding hash
            chunk_hash = chunk_keys[most_similar_index]

            # access text of most similar chunk
            node = self.index.docstore.docs[chunk_hash]

            # create node with score
            score = similarities[i][most_similar_index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        # Sort nodes by score in descending order
        nodes_with_scores.sort(key=lambda x: x.score, reverse=True)

        # Remove duplicates (if multiple topics return same chunk)
        unique_nodes_with_scores = {node.node.text: node for node in nodes_with_scores}.values()

        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done calculating scores. Time: {duration:.6f}s")

        return list(unique_nodes_with_scores)

    @staticmethod
    def init_topic_extractor(topic_extractor: TopicExtractorType):
        """
        Initializes and returns the appropriate topic extractor based on the specified type.

        Parameters:
            topic_extractor (TopicExtractorType): The type of topic extractor to initialize.

        Returns:
            TopicExtractor: The initialized topic extractor.
        """
        if topic_extractor == TopicExtractorType.LDA:
            return LDATopicExtractor()
        elif topic_extractor == TopicExtractorType.BERTOPIC:
            return BERTopicExtractor()


class TopicRetrievalQueryEngine:
    """
    Class representing a query engine for topic retrieval.

    Attributes:
        query_engine (RetrieverQueryEngine): The query engine used to perform retrieval operations.

    Methods:
        __init__(llm: BaseLLM, index, topic_extractor: TopicExtractorType, embed_model, verbose=False):
            Initializes the TopicRetrievalQueryEngine with the specified parameters.
    """

    def __init__(
            self,
            llm: BaseLLM,
            index,
            topic_extractor: TopicExtractorType,
            embed_model,
            verbose=False
    ):
        """
        Initializes the TopicRetrievalQueryEngine with the specified parameters.

        Parameters:
            llm (BaseLLM): The language model used for query generation.
            index: The index used for topic retrieval.
            topic_extractor (TopicExtractorType): The topic extractor used for extracting topics from documents.
            embed_model: The embedding model used for document representation.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        retriever = TopicRetriever(index, topic_extractor=topic_extractor, embed_model=embed_model, verbose=verbose)
        self.query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
