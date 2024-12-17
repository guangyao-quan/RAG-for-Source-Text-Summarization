"""
This module provides the CompleteEvaluator class for evaluating text generation models using traditional metrics
and RAGAS.
It includes methods for calculating various evaluation metrics such as BLEU, METEOR, ROUGE, BERTScore,
LDFACTS, and BARTScore.
It also provides functionality to create RAGAS evaluation examples, extract context from
nodes with scores, and perform comprehensive evaluation using both traditional metrics and RAGAS.
"""

from typing import Dict
import logging
import nltk
from datasets import Dataset
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import pandas as pd
import mlflow
from mlflow.metrics.genai import answer_relevance, faithfulness
from app.evaluation.ExternalHelper.bart_score import BARTScorer
from app.evaluation.ExternalHelper.ldfacts import LDFACTS
from app.evaluation.SaveMetric import df2csv, metrics2json
from app.evaluation.EvaluateSummary.time_tracker import time_tracker
from app.evaluation.EvaluateSummary.prompt_wrapper import with_prompt_template
from ragas import evaluate
from ragas.metrics import context_relevancy

logger = logging.getLogger(__name__)


class TraditionalEvaluator:
    """
    A class for evaluating text generation models using traditional metrics.
    """

    def __init__(self, app_id):
        self.app_id = app_id
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    @staticmethod
    def calculate_bleu_score(candidate: str, reference: str) -> float:
        candidate = nltk.word_tokenize(candidate)
        reference = nltk.word_tokenize(reference)
        return sentence_bleu([reference], candidate)

    @staticmethod
    def calculate_meteor_score(candidate: str, reference: str) -> float:
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        return meteor_score([reference_tokens], candidate_tokens)

    @staticmethod
    def calculate_rouge_scores(candidate: str, reference: str) -> Dict[str, float]:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {rouge: scores[rouge].fmeasure for rouge in ["rouge1", "rouge2", "rougeL"]}

    @staticmethod
    def calculate_bert_score(candidate: str, reference: str) -> Dict[str, float]:
        p, r, f1 = score([candidate], [reference], lang="en", rescale_with_baseline=True)
        return {"precision": p.mean().item(), "recall": r.mean().item(), "f1": f1.mean().item()}

    @staticmethod
    def calculate_ldfacts_score(candidate: str, source_text: str) -> float:
        ldfacts_scorer = LDFACTS(device="cpu")
        ldfacts_score = ldfacts_scorer.score_src_hyp_long([source_text], [candidate])
        return ldfacts_score[0]

    @staticmethod
    def calculate_bart_score(candidate: str, reference: str) -> float:
        bart_scorer = BARTScorer(device="cpu", checkpoint="facebook/bart-large-cnn")
        # Uncomment this line only after you downloaded and saved this file into this destination
        # Download link: https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view
        # bart_scorer.load(path='app/evaluation/bart_score.pth')
        bart_score = bart_scorer.score([candidate], [reference])
        return bart_score[0]

    @time_tracker
    def evaluate(self, data_records, count):
        total_bleu = 0
        total_meteor = 0
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougel = 0
        total_bert_precision = 0
        total_bert_recall = 0
        total_bert_f1 = 0
        total_ldfacts = 0
        total_bart = 0

        individual_scores = []

        for i in range(count):
            if i % 10 == 0:
                logger.info(f"TRADITIONALLY EVALUATING DOCUMENT: {i}")
            candidate = data_records["answer"][i]
            reference = data_records["ground_truth"][i]
            document = data_records["question"][i]

            bleu = self.calculate_bleu_score(candidate, reference)
            meteor = self.calculate_meteor_score(candidate, reference)
            rouge = self.calculate_rouge_scores(candidate, reference)
            bert = self.calculate_bert_score(candidate, reference)
            ldfacts_score = self.calculate_ldfacts_score(candidate, document)
            bart_score = self.calculate_bart_score(candidate, reference)

            total_bleu += bleu
            total_meteor += meteor
            total_rouge1 += rouge["rouge1"]
            total_rouge2 += rouge["rouge2"]
            total_rougel += rouge["rougeL"]
            total_bert_precision += bert["precision"]
            total_bert_recall += bert["recall"]
            total_bert_f1 += bert["f1"]
            total_ldfacts += ldfacts_score
            total_bart += bart_score

            individual_scores.append({
                "app_id": self.app_id,
                "BLEU": bleu,
                "METEOR": meteor,
                "ROUGE1": rouge["rouge1"],
                "ROUGE2": rouge["rouge2"],
                "ROUGEL": rouge["rougeL"],
                "BERT_precision": bert["precision"],
                "BERT_recall": bert["recall"],
                "BERT_f1": bert["f1"],
                "LDFACTS": ldfacts_score,
                "BART": bart_score
            })

        overall_metrics = {
            "BLEU": total_bleu / count,
            "METEOR": total_meteor / count,
            "ROUGE": {
                "rouge1": total_rouge1 / count,
                "rouge2": total_rouge2 / count,
                "rougeL": total_rougel / count,
            },
            "BERT": {
                "precision": total_bert_precision / count,
                "recall": total_bert_recall / count,
                "f1": total_bert_f1 / count,
            },
            "LDFACTS": total_ldfacts / count,
            "BART": total_bart / count,
            "app_id": self.app_id
        }

        df_individual_scores = pd.DataFrame(individual_scores)

        return {'overall_metrics': overall_metrics, 'df_individual_scores': df_individual_scores}


class RAGASEvaluator:
    """
    A class for evaluating text generation models using RAGAS metrics.
    """

    def __init__(self, app_id):
        self.app_id = app_id

    @time_tracker
    def evaluate(self, dataset: Dataset):
        ragas = evaluate(dataset, metrics=[context_relevancy])
        ragas["app_id"] = self.app_id
        return ragas


class MLflowEvaluator:
    """
    A class to evaluate the performance of text summarization models using MLflow.
    """

    def __init__(self, app_id):
        self.app_id = app_id

    @time_tracker
    def evaluate(self, dataframe):
        answer_relevancy = answer_relevance(model="openai:/gpt-3.5-turbo")
        groundedness = faithfulness(model="openai:/gpt-3.5-turbo")

        with mlflow.start_run():
            evals = mlflow.evaluate(
                data=dataframe,
                targets="ground_truth",
                predictions='answer',
                model_type='text-summarization',
                extra_metrics=[answer_relevancy, groundedness],
                evaluator_config={'col_mapping': {'inputs': 'question', 'context': 'contexts'}},
            )

        evals.metrics["app_id"] = self.app_id
        return evals


class CompleteEvaluator:
    """
    A class for evaluating text generation models using both traditional metrics and RAGAS.
    """

    def __init__(self, query_engine, app_id, eval_batch=None, reset_json=1, reset_csv=1):
        self.query_engine = query_engine
        self.app_id = app_id
        self.eval_batch = eval_batch
        self.reset_json = reset_json
        self.reset_csv = reset_csv

        self.traditional_evaluator = TraditionalEvaluator(app_id)
        self.ragas_evaluator = RAGASEvaluator(app_id)
        self.mlflow_evaluator = MLflowEvaluator(app_id)

    @staticmethod
    def extract_context(nodes_with_scores):
        extracted_texts = []
        for node_with_score in nodes_with_scores:
            node = node_with_score.node
            if 'text' in node.__dict__:
                extracted_texts.append(node.__dict__['text'])
        return extracted_texts

    @with_prompt_template
    def query_with_prompt(self, text):
        return self.query_engine.query(text)

    @time_tracker
    def process_batch(self, batch):
        data_records = {
            'app_id': [],
            'question': [],
            'contexts': [],
            'ground_truth': [],
            'answer': []
        }
        count = 0

        for data in batch:
            if count % 10 == 0:
                logger.info(f"GENERATING SUMMARY FOR DOCUMENT: {count}")
            document = data["document"]
            response = self.query_with_prompt(document)
            context = self.extract_context(response.source_nodes)
            reference = data["summary"]
            candidate = response.response

            data_records['app_id'].append(self.app_id)
            data_records['question'].append(document)
            data_records['contexts'].append(context)
            data_records['ground_truth'].append(reference)
            data_records['answer'].append(candidate)
            count += 1

        return {'data_records': data_records, 'count': count}

    def evaluate(self, evaluation_mode="both"):
        logger.info("PROCESSING BATCH DATA")
        batch_result, batch_process_time = self.process_batch(self.eval_batch)
        logger.info("TIME OF PROCESSING BATCH DATA: {}".format(batch_process_time))
        data_records = batch_result['data_records']
        count = batch_result['count']
        logger.info("FINISHED PROCESSING BATCH DATA")

        dataset = Dataset.from_dict(data_records)
        dataframe = pd.DataFrame(data_records)

        trad_oa = None
        ragas = None
        mlf = None
        trad_path = f"data/outputs/trad_metric.json"
        ragas_path = f"data/outputs/ragas_metric.json"
        mlf_path = f"data/outputs/mlflow_metric.json"
        kde_rag_path = f"data/outputs/rag_kde_analysis.csv"
        kde_trad_path = f"data/outputs/trad_kde_analysis.csv"
        df_path = f"data/outputs/summary.csv"

        df2csv.save_dataframe_to_csv(df=dataframe, csv_path=df_path, reset_csv=self.reset_csv)

        if evaluation_mode in ["traditional", "both"]:
            logger.info("EVALUATING TRADITIONAL METRICS")
            trad_result, trad_eval_time = self.traditional_evaluator.evaluate(data_records, count)
            logger.info("TIME OF EVALUATING TRADITIONAL METRICS: {}".format(trad_eval_time))
            trad_oa = trad_result["overall_metrics"]
            trad_idv = trad_result["df_individual_scores"]
            logger.info("FINISHED EVALUATING TRADITIONAL METRICS")
            df2csv.save_dataframe_to_csv(df=trad_idv, csv_path=kde_trad_path, reset_csv=self.reset_csv)
            metrics2json.save_metrics_to_json(average_scores=trad_oa, filepath=trad_path, reset_json=self.reset_json)
            del trad_oa["app_id"]

        if evaluation_mode in ["rag", "both"]:
            logger.info("EVALUATING RAGAS METRICS")
            ragas, ragas_eval_time = self.ragas_evaluator.evaluate(dataset)
            logger.info("TIME OF EVALUATING RAGAS METRICS: {}".format(ragas_eval_time))
            logger.info("FINISHED EVALUATING RAGAS METRICS")
            logger.info("EVALUATING MLFLOW METRICS")
            mlf, mlf_eval_time = self.mlflow_evaluator.evaluate(dataframe)
            logger.info("TIME OF EVALUATING MLF METRICS: {}".format(mlf_eval_time))
            logger.info("FINISHED EVALUATING MLFLOW METRICS")
            metrics2json.save_metrics_to_json(average_scores=ragas, filepath=ragas_path, reset_json=self.reset_json)
            metrics2json.save_metrics_to_json(average_scores=mlf.metrics, filepath=mlf_path, reset_json=self.reset_json)

            del ragas["app_id"]
            del mlf.metrics["app_id"]
            ragas_df = ragas.to_pandas()[["context_relevancy"]]
            mlf_df = mlf.tables["eval_results_table"][["toxicity/v1/score",
                                                       "flesch_kincaid_grade_level/v1/score",
                                                       "ari_grade_level/v1/score",
                                                       "faithfulness/v1/score",
                                                       "answer_relevance/v1/score"]]
            df_kde_analysis = pd.concat([mlf_df, ragas_df], axis=1)
            df_kde_analysis["app_id"] = self.app_id
            new_order = [
                "app_id",
                "toxicity/v1/score",
                "flesch_kincaid_grade_level/v1/score",
                "ari_grade_level/v1/score",
                "faithfulness/v1/score",
                "answer_relevance/v1/score",
                "context_relevancy",
            ]
            df_kde_analysis = df_kde_analysis[new_order]
            df_kde_analysis.columns = df_kde_analysis.columns.str.replace('/v1/score', '')
            df_kde_analysis = df_kde_analysis.rename(columns={"faithfulness": "groundedness",
                                                              "answer_relevance": "answer_relevancy"})
            """
            # Normalization doesn't make so much sense for kde analysis, but for histogram
            # Optional normalization methods: min-max, z-score, decimal scaling, log transformation
            df_kde_analysis["answer_relevancy"] = (df_kde_analysis["answer_relevancy"] - 1) / 4
            df_kde_analysis["groundedness"] = (df_kde_analysis["groundedness"] - 1) / 4
            df_kde_analysis["ari_grade_level"] = ((df_kde_analysis["ari_grade_level"] -
                                                   df_kde_analysis["ari_grade_level"].min()) /
                                                  (df_kde_analysis["ari_grade_level"].max() -
                                                   df_kde_analysis["ari_grade_level"].min())).round(2)
            df_kde_analysis["flesch_kincaid_grade_level"] = (((df_kde_analysis["flesch_kincaid_grade_level"] -
                                                              df_kde_analysis["flesch_kincaid_grade_level"].min()) /
                                                             (df_kde_analysis["flesch_kincaid_grade_level"].max() -
                                                              df_kde_analysis["flesch_kincaid_grade_level"].min())).
                                                             round(2))
            df_kde_analysis["toxicity"] = df_kde_analysis["toxicity"].round(5)
            df_kde_analysis["context_relevancy"] = df_kde_analysis["context_relevancy"].round(2)
            """
            df2csv.save_dataframe_to_csv(df=df_kde_analysis, csv_path=kde_rag_path, reset_csv=self.reset_csv)

        if evaluation_mode == "traditional":
            return trad_oa
        elif evaluation_mode == "rag":
            return ragas, mlf.metrics
        else:
            return trad_oa, ragas, mlf.metrics
