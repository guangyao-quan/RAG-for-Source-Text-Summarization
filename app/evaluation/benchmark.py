"""
This module provides the functionality to run a benchmark for a RAG (Retrieval-Augmented Generation) system.
The benchmark process includes data ingestion, retrieval, and evaluation using both traditional and modern
evaluation strategies.
"""

import argparse
import os
import warnings
import logging
from dotenv import load_dotenv
from app.configs import (
    EmbeddingType,
    IndexType,
    IngestionSource,
    LLMType,
    RetrievalType,
    TopicExtractorType
)
from app.evaluation.EvaluateSummary import complete_evaluator, data_loader
from app.rag import RAGBuilder

# Ignore user warnings and set parallelism for tokenizers
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='data/outputs/benchmark.log',
                    filemode='w')
logger = logging.getLogger(__name__)


def str_to_enum(enum_type):
    """
    Convert a string to a specific enumeration type.

    Args:
        enum_type (Enum): The enumeration type to convert to.

    Returns:
        function: A function that converts a string to the specified enumeration type.
    """
    def convert(value):
        try:
            return enum_type[value]
        except KeyError:
            raise argparse.ArgumentTypeError(f"Invalid {enum_type.__name__} value: {value}")
    return convert


def main():
    """
    The main method to run the RAG system benchmark.

    This function sets up the argument parser to accept various configurations and paths, creates instances of data
    loader, ingestor, retriever, and EvaluateSummary classes, and runs the evaluation process.

    Args (parsed from command line):
        --llm_type (str): Type of the Language Model.
        --embedding_type (str): Type of embedding used.
        --index_type (str): Type of indexing mechanism.
        --retrieval_type (str): Type of retrieval mechanism.
        --topic_extractor_type (str, optional): Type of topic extraction mechanism.
        --evaluation_mode (str): Mode of evaluation, one of 'traditional', 'mlflow', or 'both'.
        --eval_path (str): Path to the dataset to be evaluated.
        --app_id (str): Application identifier.
        --num_samples (int): Number of samples to evaluate. Defaults to 100.
        --reset_json (bool): Flag to reset the JSON file before saving metrics.
        --reset_csv (bool): Flag to reset the CSV file before saving the summary.

    Returns:
        dict: Average metrics from the traditional evaluation.
    """

    parser = argparse.ArgumentParser(description="Run RAG system benchmark")
    parser.add_argument(
        "--llm_type",
        type=str_to_enum(LLMType),
        required=True,
        help="Language Model type",
    )
    parser.add_argument(
        "--embedding_type",
        type=str_to_enum(EmbeddingType),
        required=False,
        help="Embedding type",
    )
    parser.add_argument(
        "--index_type",
        type=str_to_enum(IndexType),
        required=False,
        help="Indexing mechanism type",
    )
    parser.add_argument(
        "--retrieval_type",
        type=str_to_enum(RetrievalType),
        required=True,
        help="Retrieval mechanism type",
    )
    parser.add_argument(
        "--topic_extractor_type",
        type=str_to_enum(TopicExtractorType),
        required=False,
        help="Topic extractor type",
    )
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        required=True,
        help="One of 'traditional', 'mlflow', or 'both'",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        required=True,
        help="Path to dataset to be evaluated",
    )
    parser.add_argument(
        "--app_id",
        type=str,
        required=True,
        help="Application identifier",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        default=100,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--reset_json",
        type=int,
        required=True,
        help="Flag to reset the JSON before saving metrics",
    )
    parser.add_argument(
        "--reset_csv",
        type=int,
        required=True,
        help="Flag to reset the CSV before saving the summary",
    )
    args = parser.parse_args()

    logger.info("STARTING BENCHMARKING")

    logger.info("CREATING DATA GENERATOR")
    data_generator = data_loader.EvaluationDataGenerator(path=args.eval_path, configuration_name="default")
    logger.info("FINISHED CREATING DATA GENERATOR")

    logger.info("GENERATING EVALUATION BATCH")
    batch = data_generator.generate(num_samples=args.num_samples, max_length=900)
    logger.info("FINISHED GENERATING EVALUATION BATCH")

    logger.info("CREATING QUERY ENGINE")
    builder = RAGBuilder()
    if args.retrieval_type == RetrievalType.TOPIC_EXTRACTION:
        query_engine, build_time = (builder
                                    .set_llm_type(args.llm_type)
                                    .set_embedding_type(args.embedding_type)
                                    .set_index_type(args.index_type)
                                    .set_ingestion_source_and_path(
                                        source=IngestionSource.HF,
                                        paths=[(("EdinburghNLP/xsum", "default"), ["document"])])
                                    .set_retrieval_type(RetrievalType.TOPIC_EXTRACTION)
                                    .set_topic_extractor_type(args.topic_extractor_type)
                                    .set_debug_mode(True)
                                    .build())
    elif args.retrieval_type == RetrievalType.EXTRACTIVE_RETRIEVAL:
        query_engine, build_time = (builder
                                    .set_llm_type(args.llm_type)
                                    .set_retrieval_type(RetrievalType.EXTRACTIVE_RETRIEVAL)
                                    .build())
    elif args.retrieval_type == RetrievalType.KG_RETRIEVAL:
        query_engine, build_time = (builder
                                    .set_llm_type(args.llm_type)
                                    .set_embedding_type(args.embedding_type)
                                    .set_index_type(args.index_type)
                                    .set_ingestion_source_and_path(
                                        source=IngestionSource.HF,
                                        paths=[(("EdinburghNLP/xsum", "default"), ["document"])])
                                    .set_retrieval_type(RetrievalType.KG_RETRIEVAL)
                                    .set_debug_mode(True)
                                    .build())
    else:
        query_engine, build_time = (builder
                                    .set_llm_type(args.llm_type)
                                    .set_embedding_type(args.embedding_type)
                                    .set_index_type(args.index_type)
                                    .set_ingestion_source_and_path(
                                        source=IngestionSource.HF,
                                        paths=[(("EdinburghNLP/xsum", "default"), ["document"])])
                                    .set_retrieval_type(args.retrieval_type)
                                    .build())
    logger.info("FINISHED CREATING QUERY ENGINE IN {}".format(build_time))

    logger.info("CREATING COMPLETE EVALUATOR")
    comp_evaluator = complete_evaluator.CompleteEvaluator(
        query_engine=query_engine,
        app_id=args.app_id,
        eval_batch=batch,
        reset_json=args.reset_json,
        reset_csv=args.reset_csv)
    logger.info("FINISHED CREATING COMPLETE EVALUATOR")

    logger.info("RUNNING EVALUATION")
    comp_evaluator.evaluate(evaluation_mode=args.evaluation_mode)
    logger.info("FINISHED RUNNING EVALUATION")

    logger.info("FINISHED RUNNING BENCHMARKING")


if __name__ == "__main__":
    """
    Command-line interface for the RAG benchmarking script.

    This script reads various configurations and paths from command-line arguments, sets up
    the RAG system components, and runs the benchmark evaluation.

    Example usage:
        python benchmark_rag.py --llm_type gpt --embedding_type bert --index_type faiss --retrieval_type dense 
        --evaluation_mode both --eval_path path/to/dataset --app_id my_app --num_samples 100 --reset_json 1 
        --reset_csv 1
    """
    main()
