import logging
import os
import random
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizer
from official_eval import official_f1
import logging
logger = logging.getLogger(__name__)

# Define additional special tokens for entity markers
ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

def get_label(args):
    """Loads relation labels from the label file."""
    label_path = Path(args.data_dir) / args.label_file

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with label_path.open("r", encoding="utf-8") as f:
        return [label.strip() for label in f]


def load_tokenizer(args):
    """Loads the BERT tokenizer and adds special tokens for entity markers."""
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


# def write_prediction(args, output_file, preds):
#     """
#     Writes model predictions to a file for official evaluation.
#
#     :param output_file: Path to save predictions (e.g., eval/proposed_answers.txt)
#     :param preds: List of predicted labels (e.g., [0,1,0,2,18,...])
#     """
#     relation_labels = get_label(args)
#     output_path = Path(output_file)
#
#     with output_path.open("w", encoding="utf-8") as f:
#         for idx, pred in enumerate(preds):
#             f.write(f"{8001 + idx}\t{relation_labels[pred]}\n")
def write_prediction(args, output_file, preds):
    """
    Writes model predictions or true labels to a file for evaluation.

    :param output_file: Path to save predictions (e.g., eval/proposed_answers.txt)
    :param preds: List of predicted labels or true labels
    """
    relation_labels = get_label(args)  # Get Labels
    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write(f"{8001 + idx}\t{relation_labels[pred]}\n")

    logger.info(f"Predictions written to {output_file}, Total: {len(preds)}")


def init_logger():
    """Initialises the logger for logging model training and evaluation."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    """Sets the random seed for reproducibility across different runs."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.info(f"Random seed set to: {args.seed}")


def compute_metrics(preds, labels):
    """
    Computes accuracy and F1-score based on model predictions and true labels.

    :param preds: Predicted labels
    :param labels: Ground truth labels
    :return: Dictionary containing accuracy and F1-score
    """
    assert len(preds) == len(labels), "Predictions and labels must have the same length."
    return acc_and_f1(preds, labels)

def simple_accuracy(preds, labels):
    """Computes the accuracy of predictions."""
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro"):
    """
    Computes both accuracy and F1-score.

    :param preds: Predicted labels
    :param labels: Ground truth labels
    :param average: Averaging method for F1-score (default: macro)
    :return: Dictionary containing accuracy and F1-score
    """
    acc = simple_accuracy(preds, labels)
    f1 = official_f1()
    return {
        "accuracy": acc,
        "f1_score": f1,
    }
