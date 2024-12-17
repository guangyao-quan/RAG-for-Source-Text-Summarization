"""
This module benchmarks the build and processing times for different configurations of a RAG (Retrieval-Augmented
Generation) system. The benchmark process includes data ingestion, retrieval, and evaluation using both traditional
and modern evaluation strategies.
"""

import os
import warnings
import logging
import pandas as pd
from dotenv import load_dotenv
from app.rag import RAGBuilder
from app.configs import LLMType, EmbeddingType, IndexType, RetrievalType, IngestionSource, TopicExtractorType
from app.evaluation.EvaluateSummary import complete_evaluator, data_loader

# Suppress user warnings and enable parallelism for tokenizers
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='data/outputs/benchmark.log',
                    filemode='w')
logger = logging.getLogger(__name__)


def build_query_engine(index_type, retrieval_type, topic_extractor_type=None):
    """
    Build a query engine based on specified indexing and retrieval types, and optionally a topic extractor type.

    Args:
        index_type (IndexType): The type of indexing mechanism.
        retrieval_type (RetrievalType): The type of retrieval mechanism.
        topic_extractor_type (TopicExtractorType, optional): The type of topic extraction mechanism.

    Returns:
        tuple: A tuple containing the query engine instance and the time taken to build it.
    """
    builder = RAGBuilder()

    if retrieval_type == RetrievalType.TOPIC_EXTRACTION:
        query_engine, build_time = (builder
                                    .set_llm_type(LLMType.GPT35_TURBO)
                                    .set_embedding_type(EmbeddingType.BGE_SMALL_EN)
                                    .set_index_type(index_type)
                                    .set_ingestion_source_and_path(
                                        source=IngestionSource.HF,
                                        paths=[(("EdinburghNLP/xsum", "default"), ["document"])])
                                    .set_retrieval_type(RetrievalType.TOPIC_EXTRACTION)
                                    .set_topic_extractor_type(topic_extractor_type)
                                    .set_debug_mode(True)
                                    .build())
    elif retrieval_type == RetrievalType.EXTRACTIVE_RETRIEVAL:
        query_engine, build_time = (builder
                                    .set_llm_type(LLMType.GPT35_TURBO)
                                    .set_retrieval_type(RetrievalType.EXTRACTIVE_RETRIEVAL)
                                    .build())
    elif retrieval_type == RetrievalType.KG_RETRIEVAL:
        query_engine, build_time = (builder
                                    .set_llm_type(LLMType.GPT35_TURBO)
                                    .set_embedding_type(EmbeddingType.BGE_SMALL_EN)
                                    .set_index_type(index_type)
                                    .set_ingestion_source_and_path(
                                        source=IngestionSource.HF,
                                        paths=[(("EdinburghNLP/xsum", "default"), ["document"])])
                                    .set_retrieval_type(RetrievalType.KG_RETRIEVAL)
                                    .set_debug_mode(True)
                                    .build())
    else:
        query_engine, build_time = (builder
                                    .set_llm_type(LLMType.GPT35_TURBO)
                                    .set_embedding_type(EmbeddingType.BGE_SMALL_EN)
                                    .set_index_type(index_type)
                                    .set_ingestion_source_and_path(
                                        source=IngestionSource.HF,
                                        paths=[(("EdinburghNLP/xsum", "default"), ["document"])])
                                    .set_retrieval_type(retrieval_type)
                                    .build())

    return query_engine, build_time


def benchmark_build_times():
    """
    Benchmark the build and processing times for different RAG system configurations.

    This function iterates over different combinations of index and retrieval types, builds the query engine,
    and evaluates a sample batch. It records the build and processing times for each configuration.

    Returns:
        list: A list of tuples containing index type, retrieval type, topic extractor type, build time,
        and process time.
    """
    data_generator = data_loader.EvaluationDataGenerator(path="EdinburghNLP/xsum", configuration_name="default")
    batch = data_generator.generate(num_samples=1)

    results = []

    for index_type in [IndexType.VECTOR_INDEX, IndexType.KG_INDEX]:
        for retrieval_type in RetrievalType:
            if index_type == IndexType.VECTOR_INDEX and retrieval_type == RetrievalType.KG_RETRIEVAL:
                continue
            if index_type == IndexType.KG_INDEX and retrieval_type == RetrievalType.TOPIC_EXTRACTION:
                continue
            if retrieval_type == RetrievalType.TOPIC_EXTRACTION:
                topic_extractor_type = TopicExtractorType.LDA
                query_engine, build_time = build_query_engine(index_type, retrieval_type, topic_extractor_type)
                logger.info(f"IndexType: {index_type.name}, RetrievalType: {retrieval_type.name}, "
                            f"TopicExtractorType: {topic_extractor_type.name}, Build time: {build_time:.4f} seconds.")
                comp_evaluator = complete_evaluator.CompleteEvaluator(
                    query_engine=query_engine,
                    app_id=f"{index_type.name}_{retrieval_type.name}_{topic_extractor_type.name}",
                    eval_batch=batch,
                    reset_json=True,
                    reset_csv=True
                )
                batch_results, process_time = comp_evaluator.process_batch(batch)
                results.append((index_type.name, retrieval_type.name, topic_extractor_type.name, build_time,
                                process_time))
            else:
                query_engine, build_time = build_query_engine(index_type, retrieval_type)
                logger.info(f"IndexType: {index_type.name}, RetrievalType: {retrieval_type.name}, "
                            f"Build time: {build_time:.4f} seconds.")

                comp_evaluator = complete_evaluator.CompleteEvaluator(
                    query_engine=query_engine,
                    app_id=f"{index_type.name}_{retrieval_type.name}",
                    eval_batch=batch,
                    reset_json=True,
                    reset_csv=True
                )
                batch_results, process_time = comp_evaluator.process_batch(batch)
                results.append((index_type.name, retrieval_type.name, None, build_time, process_time))

    return results


def main():
    """
    Main function to run the RAG system benchmark.

    This function executes the benchmark, collects the results, and logs them.
    The results include the build and processing times for different system configurations.
    """
    # Run the benchmark function
    benchmark_results = benchmark_build_times()

    # Create a DataFrame from the results
    df = pd.DataFrame(benchmark_results, columns=['IndexType', 'RetrievalType', 'TopicExtractorType', 'BuildTime',
                                                  'ProcessTime'])

    # Display the DataFrame
    logger.info("\n" + df.to_string())

    # Optionally, save the DataFrame to a CSV file
    df.to_csv('data/outputs/time_benchmark_results.csv', index=False)


if __name__ == "__main__":
    main()
