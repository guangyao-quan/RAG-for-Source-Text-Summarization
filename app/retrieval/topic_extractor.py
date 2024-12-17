"""
This module provides tools for extracting topics from text using Latent Dirichlet Allocation (LDA)
and BERTopic, further refining these topics using large language models via the LangChain library.
It includes functionalities for preprocessing text, performing LDA topic modeling, and generating textual
summaries of the identified topics.

Classes:
    TopicExtractor: A base class for topic extraction that defines a common interface.
    LDATopicExtractor: Implements topic extraction using the LDA model.
    BERTopicExtractor: Implements topic extraction using the BERTopic model with SentenceTransformer.

Dependencies:
    nltk, gensim, pandas, bertopic, sentence_transformers: Libraries used for text processing, topic modeling,
    and integrating with large language models for natural language generation, respectively.
"""

from collections import Counter

import nltk
import pandas as pd
from bertopic import BERTopic
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer


class TopicExtractor:
    """
    A base class for extracting topics from texts.
    """

    def extract_topics(self, texts):
        """
        Extracts topics from the given texts.

        Args:
            texts (list): A list of texts from which to extract topics.

        Returns:
            list: A list of extracted topics.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


class LDATopicExtractor(TopicExtractor):
    """
    A class for extracting topics using the Latent Dirichlet Allocation (LDA) model.

    Methods:
        extract_topics(texts, num_topics=5, words_per_topic=10): Extracts topics from a list of texts using LDA.
        preprocess(text): Preprocesses the given text by removing stopwords and tokens with a length less than
        or equal to 3.
    """

    def extract_topics(self, texts, num_topics=5, words_per_topic=10):
        """
        Extracts topics from a list of texts using the Latent Dirichlet Allocation (LDA) model.

        Args:
            texts (list): A list of texts to extract topics from.
            num_topics (int): The number of topics to extract (default is 5).
            words_per_topic (int): The number of words to include per topic (default is 10).

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted topics, their counts, and word representations.
        """
        # Preprocess the texts
        texts = [self.preprocess(text) for text in texts]

        # Create a dictionary and a corpus
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=10000)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Create the LDA model
        lda_model = LdaModel(corpus=corpus, id2word=dictionary,
                             num_topics=num_topics,
                             random_state=100,
                             update_every=1,
                             chunksize=100,
                             passes=10,
                             alpha='auto',
                             per_word_topics=True)

        doc_topics = [lda_model.get_document_topics(bow) for bow in corpus]

        topic_count = Counter()
        topic_words_map = {}

        for doc_distribution in doc_topics:
            doc_distribution.sort(key=lambda x: -x[1])
            top_topic, _ = doc_distribution[0]
            topic_count[top_topic] += 1

            if top_topic not in topic_words_map:
                topic_words = lda_model.show_topic(top_topic, topn=words_per_topic)
                top_words = [word for word, _ in topic_words]
                topic_words_map[top_topic] = top_words

        data = {
            "Topic": list(topic_words_map.keys()),
            "Count": [topic_count[topic] for topic in topic_words_map.keys()],
            "Representation": [topic_words_map[topic] for topic in topic_words_map.keys()]
        }
        df_topics = pd.DataFrame(data)

        # Sort topics by count in descending order
        df_topics_sorted = df_topics.sort_values(by='Count', ascending=False).reset_index(drop=True)
        return df_topics_sorted

    @staticmethod
    def preprocess(text):
        """
        Preprocesses the given text by removing stopwords and tokens with a length less than or equal to 3.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            list: A list of preprocessed tokens.
        """
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

        preprocessed_text = []

        for token in simple_preprocess(text, deacc=True):
            if token not in stop_words and len(token) > 3:
                preprocessed_text.append(token)

        return preprocessed_text


class BERTopicExtractor(TopicExtractor):
    """
    A topic extractor that uses BERTopic with SentenceTransformer for topic extraction from texts.

    Methods:
        extract_topics(texts): Extracts topics from a list of texts using BERTopic.
        preprocess(text): Preprocesses the given text by removing stopwords and tokens with length less than 3.
    """

    def extract_topics(self, texts):
        """
        Extracts topics from the given texts using BERTopic with SentenceTransformer.

        Args:
            texts (list): A list of texts to extract topics from.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted topics, their counts, and representations.
        """
        # Preprocess texts
        texts = [self.preprocess(text) for text in texts]

        # Use BERTopic with SentenceTransformer
        sentence_model = SentenceTransformer("all-mpnet-base-v2")
        topic_model = BERTopic(embedding_model=sentence_model, calculate_probabilities=True)
        _, _ = topic_model.fit_transform(texts)

        # Retrieve topic information as a DataFrame
        topic_info = topic_model.get_topic_info()

        # Only select relevant columns
        selected_columns = topic_info[['Topic', 'Count', 'Representation']]

        return selected_columns

    @staticmethod
    def preprocess(text):
        """
        Preprocesses the given text by removing stopwords and tokens with length less than 3.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

        preprocessed_text = []

        for token in simple_preprocess(text, deacc=True):
            if token not in stop_words and len(token) > 3:
                preprocessed_text.append(token)

        return " ".join(preprocessed_text)
