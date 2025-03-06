import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from model import RBERT, DuoClassifier
from utils import get_label, init_logger, load_tokenizer

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))


def load_model(pred_config, args, device):
    """
    Load model from directory, prioritizing best model if available.

    Args:
        pred_config: Prediction configuration
        args: Training arguments
        device: Device to run inference on

    Returns:
        Loaded model
    """
    # Try to find the best model first
    best_dir = os.path.join(pred_config.model_dir, "best")
    if os.path.exists(best_dir) and os.path.exists(os.path.join(best_dir, "config.json")):
        try:
            logger.info("Loading best model...")
            model = RBERT.from_pretrained(best_dir, args=args)
            model.to(device)
            model.eval()
            logger.info("***** Best Model Loaded *****")
            return model
        except Exception as e:
            logger.warning(f"Failed to load best model: {e}")

    # Try to find the final model
    final_dir = os.path.join(pred_config.model_dir, "epoch_final")
    if os.path.exists(final_dir) and os.path.exists(os.path.join(final_dir, "config.json")):
        try:
            logger.info("Loading final model...")
            model = RBERT.from_pretrained(final_dir, args=args)
            model.to(device)
            model.eval()
            logger.info("***** Final Model Loaded *****")
            return model
        except Exception as e:
            logger.warning(f"Failed to load final model: {e}")

    # Check for any fold's best model (if using CV)
    for fold in range(5):  # Try up to 5 folds
        fold_best_dir = os.path.join(pred_config.model_dir, f"fold_{fold}_best")
        if os.path.exists(fold_best_dir) and os.path.exists(os.path.join(fold_best_dir, "config.json")):
            try:
                logger.info(f"Loading best model from fold {fold}...")
                model = RBERT.from_pretrained(fold_best_dir, args=args)
                model.to(device)
                model.eval()
                logger.info(f"***** Best Model for Fold {fold} Loaded *****")
                return model
            except Exception as e:
                logger.warning(f"Failed to load best model for fold {fold}: {e}")

    # If no best model found, try original path directly
    if os.path.exists(pred_config.model_dir) and os.path.exists(os.path.join(pred_config.model_dir, "config.json")):
        try:
            logger.info(f"Loading model from direct path: {pred_config.model_dir}")
            model = RBERT.from_pretrained(pred_config.model_dir, args=args)
            model.to(device)
            model.eval()
            logger.info("***** Model Loaded from Direct Path *****")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from direct path: {e}")

    # If we get here, no model found
    raise Exception("No model files found. Please check your model directory.")


def load_duo_classifier(pred_config, args, device):
    """
    Load both binary and relation classifiers for duo-classifier prediction.
    Prioritizes loading the best models for both classifiers.

    Args:
        pred_config: Prediction configuration
        args: Training arguments
        device: Device to use

    Returns:
        A DuoClassifier instance
    """
    # Check if binary model path is specified
    if not pred_config.binary_model_dir:
        raise Exception("Binary model path not specified for duo-classifier")

    # Check if binary model exists
    if not os.path.exists(pred_config.binary_model_dir):
        raise Exception("Binary model doesn't exist! Train binary model first!")

    # Load relation model (using the updated load_model function)
    relation_model = load_model(pred_config, args, device)

    # Set binary mode for loading binary model
    binary_args = argparse.Namespace(**vars(args))
    binary_args.binary_mode = True

    # Try to find the best binary model first
    binary_model = None
    model_paths_to_try = [
        os.path.join(pred_config.binary_model_dir, "best"),  # Best model
        os.path.join(pred_config.binary_model_dir, "epoch_final"),  # Final epoch
        pred_config.binary_model_dir  # Direct path
    ]

    # Also check fold-specific best models
    for fold in range(5):  # Try up to 5 folds
        model_paths_to_try.append(os.path.join(pred_config.binary_model_dir, f"fold_{fold}_best"))

    # Try each path in order
    for binary_path in model_paths_to_try:
        if os.path.exists(binary_path) and os.path.exists(os.path.join(binary_path, "config.json")):
            try:
                logger.info(f"Loading binary model from: {binary_path}")
                binary_model = RBERT.from_pretrained(binary_path, args=binary_args)
                binary_model.to(device)
                binary_model.eval()
                logger.info("***** Binary Model Loaded *****")
                break
            except Exception as e:
                logger.warning(f"Failed to load binary model from {binary_path}: {e}")

    if binary_model is None:
        # If still no model found, check if there are any epoch directories
        try:
            epoch_dirs = [d for d in os.listdir(pred_config.binary_model_dir) if d.startswith("epoch_")]
            if epoch_dirs:
                # Sort and use latest epoch
                epoch_nums = [int(d.split("_")[1]) for d in epoch_dirs if d.split("_")[1].isdigit()]
                if epoch_nums:
                    latest_epoch = max(epoch_nums)
                    latest_path = os.path.join(pred_config.binary_model_dir, f"epoch_{latest_epoch}")

                    logger.info(f"Loading binary model from latest epoch: {latest_path}")
                    binary_model = RBERT.from_pretrained(latest_path, args=binary_args)
                    binary_model.to(device)
                    binary_model.eval()
                    logger.info("***** Binary Model Loaded from Latest Epoch *****")
        except Exception as e:
            logger.error(f"Error looking for epoch directories: {e}")

    if binary_model is None:
        raise Exception("Could not find any binary model to load")

    # Create duo classifier
    duo_classifier = DuoClassifier(
        binary_model=binary_model,
        relation_model=relation_model,
        device=device,
        binary_threshold=pred_config.binary_threshold
    )
    logger.info("***** Duo-Classifier Created *****")
    logger.info(f"Binary threshold: {pred_config.binary_threshold}")

    return duo_classifier


def convert_input_file_to_tensor_dataset(
        pred_config,
        args,
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    tokenizer = load_tokenizer(args)

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_e1_mask = []
    all_e2_mask = []

    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            tokens = tokenizer.tokenize(line)

            e11_p = tokens.index("<e1>")  # the start position of entity1
            e12_p = tokens.index("</e1>")  # the end position of entity1
            e21_p = tokens.index("<e2>")  # the start position of entity2
            e22_p = tokens.index("</e2>")  # the end position of entity2

            # Replace the token
            tokens[e11_p] = "$"
            tokens[e12_p] = "$"
            tokens[e21_p] = "#"
            tokens[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            if args.add_sep_token:
                special_tokens_count = 2
            else:
                special_tokens_count = 1
            if len(tokens) > args.max_seq_len - special_tokens_count:
                tokens = tokens[: (args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            if args.add_sep_token:
                tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
            all_e1_mask.append(e1_mask)
            all_e2_mask.append(e2_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_e1_mask = torch.tensor(all_e1_mask, dtype=torch.long)
    all_e2_mask = torch.tensor(all_e2_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_e1_mask, all_e2_mask)

    return dataset


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)

    # Check whether to use duo-classifier
    if pred_config.duo_classifier:
        logger.info("Using duo-classifier for prediction")
        duo_classifier = load_duo_classifier(pred_config, args, device)
    else:
        logger.info("Using standard relation classifier for prediction")
        model = load_model(pred_config, args, device)

    logger.info(args)

    # Convert input file to TensorDataset
    dataset = convert_input_file_to_tensor_dataset(pred_config, args)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    if pred_config.duo_classifier:
        logger.info("Predicting relations using duo-classifier approach...")
        binary_preds = []
        relation_preds = []

        for batch in tqdm(data_loader, desc="Predicting"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "e1_mask": batch[3],
                    "e2_mask": batch[4],
                }

                # Get predictions from duo classifier
                duo_results = duo_classifier.predict(inputs)
                has_relation = duo_results["has_relation"].cpu().numpy()
                relations = duo_results["relation_preds"].cpu().numpy()

                binary_preds.extend(has_relation.astype(int))
                relation_preds.extend(relations)

        # Convert to numpy arrays
        binary_preds = np.array(binary_preds)
        preds = np.array(relation_preds)

        # Write binary predictions to a separate file
        binary_labels = ["No-Relation", "Has-Relation"]
        binary_output_file = os.path.join(os.path.dirname(pred_config.output_file),
                                          f"binary_{os.path.basename(pred_config.output_file)}")
        with open(binary_output_file, "w", encoding="utf-8") as writer:
            for pred in binary_preds:
                writer.write(f"{binary_labels[pred]}\n")
        logger.info(f"Binary predictions saved to {binary_output_file}")

    else:
        # Standard single-classifier prediction
        preds = None

        for batch in tqdm(data_loader, desc="Predicting"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": None,
                    "e1_mask": batch[3],
                    "e2_mask": batch[4],
                }
                outputs = model(**inputs)
                logits = outputs[0]

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)

    # Write relation predictions to output file
    label_lst = get_label(args)
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write("{}\n".format(label_lst[pred]))

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        default="sample_pred_in.txt",
        type=str,
        help="Input file for prediction",
    )
    parser.add_argument(
        "--output_file",
        default="sample_pred_out.txt",
        type=str,
        help="Output file for prediction",
    )
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # Duo-classifier options
    parser.add_argument("--duo_classifier", action="store_true", help="Use duo-classifier approach")
    parser.add_argument("--binary_model_dir", default="", type=str, help="Path to binary classifier model")
    parser.add_argument("--binary_threshold", default=0.5, type=float, help="Threshold for binary classifier")

    pred_config = parser.parse_args()
    predict(pred_config)