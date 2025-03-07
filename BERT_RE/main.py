import argparse
import os
import logging
from pathlib import Path
import sys
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset
from data_loader import load_and_cache_examples, SemEvalProcessor
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, cleanup_models
from transformers import BertConfig
from model import RBERT

logger = logging.getLogger(__name__)


def filter_dataset_with_binary_classifier(args, binary_model, dataset, device):
    """
    Filter a dataset using predictions from a binary classifier.
    Only samples predicted as having relations (class 1) are kept.

    Args:
        args: Command-line arguments
        binary_model: Trained binary classifier model
        dataset: Dataset to filter
        device: Device to run inference on

    Returns:
        Filtered dataset containing only samples predicted to have relations
    """
    from torch.utils.data import DataLoader, SequentialSampler

    logger.info(f"Filtering dataset with binary classifier")
    logger.info(f"Original dataset size: {len(dataset)}")

    # Set model to evaluation mode
    binary_model.eval()

    # Create dataloader
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)

    # Get predictions from binary classifier
    binary_preds = []
    original_indices = []

    for i, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)

        # Extract inputs
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": None,
            "e1_mask": batch[4],
            "e2_mask": batch[5],
        }

        # Get predictions
        with torch.no_grad():
            outputs = binary_model(**inputs)
            logits = outputs[0]
            probs = torch.softmax(logits, dim=1)

            # Store probabilities for potential threshold adjustment
            positive_probs = probs[:, 1].cpu().numpy()

            # Apply threshold
            predictions = (positive_probs > args.binary_threshold).astype(bool)

            # Store predictions and original indices
            binary_preds.extend(predictions)
            batch_indices = list(range(i * args.eval_batch_size, min((i + 1) * args.eval_batch_size, len(dataset))))
            original_indices.extend(batch_indices)

    # Find indices of positive samples (predicted to have relations)
    positive_indices = [idx for idx, pred in zip(original_indices, binary_preds) if pred]

    # Get original labels for all samples
    all_original_labels = [dataset[idx][3].item() for idx in range(len(dataset))]
    label_distribution_original = {}
    for label in all_original_labels:
        if label not in label_distribution_original:
            label_distribution_original[label] = 0
        label_distribution_original[label] += 1

    # Get labels of filtered samples for statistics
    filtered_labels = [dataset[idx][3].item() for idx in positive_indices]
    label_distribution_filtered = {}
    for label in filtered_labels:
        if label not in label_distribution_filtered:
            label_distribution_filtered[label] = 0
        label_distribution_filtered[label] += 1

    # Log detailed class distribution
    logger.info(f"Original class distribution:")
    for label, count in sorted(label_distribution_original.items()):
        percentage = (count / len(dataset)) * 100
        logger.info(f"  Class {label}: {count} samples ({percentage:.2f}%)")

    logger.info(f"Filtered class distribution:")
    for label, count in sorted(label_distribution_filtered.items()):
        if label in label_distribution_original and label_distribution_original[label] > 0:
            retention = (count / label_distribution_original[label]) * 100
            logger.info(f"  Class {label}: {count} samples (retained {retention:.2f}% of original)")
        else:
            logger.info(f"  Class {label}: {count} samples")

    binary_truth = [(label > 0) for label in all_original_labels]

    # Calculate statistics
    total_samples = len(dataset)
    positive_samples = len(positive_indices)
    negative_samples = total_samples - positive_samples
    true_relations = sum(1 for label in all_original_labels if label > 0)

    logger.info(f"Binary classifier statistics:")
    logger.info(
        f"  Predicted positive (has relation): {positive_samples} ({positive_samples / total_samples * 100:.2f}%)")
    logger.info(
        f"  Predicted negative (no relation): {negative_samples} ({negative_samples / total_samples * 100:.2f}%)")
    logger.info(f"  True positive (has relation): {true_relations} ({true_relations / total_samples * 100:.2f}%)")
    logger.info(
        f"  True negative (no relation): {total_samples - true_relations} ({(total_samples - true_relations) / total_samples * 100:.2f}%)")

    # Check if we're losing too many samples
    if positive_samples < 100:
        logger.warning(f"⚠️ Very few samples ({positive_samples}) remain after filtering! Model training may fail.")

        # If too few positive samples, lower the threshold and refilter
        if positive_samples < 50 and args.binary_threshold > 0.3:
            logger.warning(f"Automatically lowering binary threshold from {args.binary_threshold} to 0.3")
            lowered_threshold = 0.3
            new_positive_indices = [idx for idx, prob in zip(original_indices, positive_probs) if
                                    prob > lowered_threshold]
            if len(new_positive_indices) > positive_samples:
                logger.info(f"After lowering threshold: {len(new_positive_indices)} samples (was {positive_samples})")
                positive_indices = new_positive_indices
                positive_samples = len(positive_indices)

    # Check class diversity
    unique_classes = set(filtered_labels)
    if len(unique_classes) < 3:
        logger.warning(f"⚠️ Only {len(unique_classes)} classes in filtered data! Multi-class training may fail.")

        # If we lost too many classes, add some examples from missing classes
        original_classes = set(all_original_labels)
        missing_classes = original_classes - unique_classes
        if missing_classes and positive_samples > 0:
            logger.info(f"Adding samples from {len(missing_classes)} missing classes")

            # Add at least 5 examples from each missing class
            additional_indices = []
            for class_id in missing_classes:
                class_indices = [idx for idx in range(len(dataset))
                                 if dataset[idx][3].item() == class_id]
                if class_indices:
                    # Take up to 5 examples or 10% of the class, whichever is greater
                    samples_to_add = min(len(class_indices),
                                         max(5, int(label_distribution_original.get(class_id, 0) * 0.1)))
                    additional_indices.extend(class_indices[:samples_to_add])
                    logger.info(f"  Added {samples_to_add} samples from class {class_id}")

            if additional_indices:
                # Add indices, avoiding duplicates
                additional_indices = [idx for idx in additional_indices if idx not in positive_indices]
                positive_indices.extend(additional_indices)
                logger.info(f"After adding samples: {len(positive_indices)} total samples")

                # Recalculate filtered labels
                filtered_labels = [dataset[idx][3].item() for idx in positive_indices]
                unique_classes = set(filtered_labels)
                logger.info(f"Final class count: {len(unique_classes)} classes")

    # Create subset with only positive samples
    filtered_dataset = torch.utils.data.Subset(dataset, positive_indices)
    logger.info(
        f"Filtered dataset size: {len(filtered_dataset)} ({len(filtered_dataset) / len(dataset) * 100:.2f}% of original)")

    return filtered_dataset


def main(args):
    """Main function to initialise training and evaluation."""
    init_logger()
    set_seed(args)

    try:
        tokenizer = load_tokenizer(args)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # Create binary dataset if duo-classifier mode is enabled
    if args.duo_classifier:
        binary_data_dir = os.path.join(args.data_dir, "binary")
        processor = SemEvalProcessor(args)
        processor.create_binary_dataset(binary_data_dir)
        logger.info(f"Binary dataset created in {binary_data_dir}")

        # Update args for later use with binary mode
        args.binary_data_dir = binary_data_dir

    # Handle evaluation-only mode (no training)
    if args.do_eval and not args.do_train:
        logger.info("Evaluation mode only (no training)")

        try:
            # Load datasets
            test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

            # Try to find dev dataset
            dev_dataset = None
            if args.do_dev_eval:
                # Check if there's a specified dev file
                if hasattr(args, 'dev_file') and args.dev_file:
                    try:
                        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
                        logger.info(f"Using specified dev file: {args.dev_file}")
                    except FileNotFoundError as e:
                        logger.warning(f"Specified dev file not found: {e}")

                # If no dev dataset yet, try cross-validation dev files
                if dev_dataset is None:
                    for i in range(5):  # Try up to 5 CV dev files
                        dev_file = f"dev_k_{i}.tsv"
                        try:
                            args.dev_file = dev_file
                            dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
                            logger.info(f"Using CV dev file: {dev_file}")
                            break
                        except Exception:
                            continue

            # Initialize Trainer
            trainer = Trainer(args, dev_dataset=dev_dataset, test_dataset=test_dataset)

            # Load model
            logger.info(f"Loading model from {args.model_dir}")
            try:
                trainer.load_model()
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return

            # Evaluate on dev set if available
            if args.do_dev_eval and dev_dataset:
                logger.info("Evaluating on validation set")
                trainer.evaluate("dev", prefix="evaluation", save_cm=True)

            # Evaluate on test set
            logger.info("Evaluating on test set")
            trainer.evaluate("test", prefix="evaluation", save_cm=True)

            # If duo-classifier mode is enabled, evaluate with the duo-classifier approach
            if args.duo_classifier and args.binary_model_dir:
                logger.info("Evaluating with duo-classifier approach")
                trainer.evaluate_duo_classifier("test", prefix="duo_evaluation", save_cm=True)

            return
        except Exception as e:
            logger.error(f"Error in evaluation mode: {e}")
            return

    try:
        processor = SemEvalProcessor(args)

        # If K-Fold cross-validation is enabled, split the dataset accordingly
        if hasattr(args, 'k_folds') and args.k_folds > 1:
            logger.info(f"Setting up {args.k_folds}-fold cross-validation")
            try:
                processor.split_kfold(args.k_folds)
            except Exception as e:
                logger.error(f"Failed to split dataset for cross-validation: {e}")
                return

            # Load full dataset for CV
            train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
            test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

            # If using duo-classifier in CV mode
            if args.duo_classifier:
                logger.info("Training duo-classifier with cross-validation")

                # Save original args for later
                orig_args = copy.deepcopy(args)

                # Set up binary classifier training
                binary_args = copy.deepcopy(args)
                binary_args.binary_mode = True
                binary_args.data_dir = args.binary_data_dir
                binary_args.model_dir = os.path.join(args.model_dir, "binary")
                binary_args.train_file = f"binary_{args.train_file}"
                binary_args.test_file = f"binary_{args.test_file}"
                binary_args.label_file = "binary_label.txt"

                # Load binary datasets
                binary_train_dataset = load_and_cache_examples(binary_args, tokenizer, mode="train")
                binary_test_dataset = load_and_cache_examples(binary_args, tokenizer, mode="test")

                # Train binary classifier with CV
                binary_trainer = Trainer(binary_args, train_dataset=binary_train_dataset,
                                         test_dataset=binary_test_dataset)
                if args.do_train:
                    binary_trainer.train()

                # Debug: Verify the label distribution in train_dataset
                train_labels = [train_dataset[i][3].item() for i in range(min(1000, len(train_dataset)))]
                unique_labels, counts = np.unique(train_labels, return_counts=True)
                logger.info(f"Training dataset contains {len(unique_labels)} unique labels: {unique_labels}")
                logger.info(f"Label distribution: {dict(zip(unique_labels, counts))}")

                # Load the best binary model for filtering during inference
                device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

                # Search for the most likely binary model directory based on the CV fold structure
                binary_model_dir = None

                # First try fold_X_best directories
                for fold_idx in range(args.k_folds):
                    best_dir = os.path.join(binary_args.model_dir, f"fold_{fold_idx}_best")
                    if os.path.exists(best_dir) and os.path.exists(os.path.join(best_dir, "config.json")):
                        logger.info(f"Found binary model at best fold directory: {best_dir}")
                        binary_model_dir = best_dir
                        break

                # If no best model found, try epoch_final in fold directories
                if binary_model_dir is None:
                    for fold_idx in range(args.k_folds):
                        epoch_final_dir = os.path.join(binary_args.model_dir, f"fold_{fold_idx}", "epoch_final")
                        if os.path.exists(epoch_final_dir) and os.path.exists(
                                os.path.join(epoch_final_dir, "config.json")):
                            logger.info(f"Found binary model at epoch_final directory: {epoch_final_dir}")
                            binary_model_dir = epoch_final_dir
                            break

                # If still not found, try any available fold directory with epoch subdirectories
                if binary_model_dir is None:
                    for fold_idx in range(args.k_folds):
                        fold_dir = os.path.join(binary_args.model_dir, f"fold_{fold_idx}")
                        if os.path.exists(fold_dir):
                            # Look for epoch_X subdirectories
                            try:
                                epoch_dirs = [d for d in os.listdir(fold_dir) if
                                              d.startswith("epoch_") and os.path.isdir(os.path.join(fold_dir, d))]
                                if epoch_dirs:
                                    # Use the first found epoch directory
                                    epoch_dir = os.path.join(fold_dir, epoch_dirs[0])
                                    if os.path.exists(os.path.join(epoch_dir, "config.json")):
                                        logger.info(f"Found binary model at: {epoch_dir}")
                                        binary_model_dir = epoch_dir
                                        break
                            except Exception as e:
                                logger.warning(f"Error checking fold directory {fold_dir}: {e}")
                                continue

                if binary_model_dir is None:
                    logger.error("Could not find any usable binary model directory")
                    return

                # Now use the found model directory
                try:
                    logger.info(f"Loading binary model from: {binary_model_dir}")
                    binary_config = BertConfig.from_pretrained(binary_model_dir, num_labels=2)
                    binary_model = RBERT.from_pretrained(binary_model_dir, config=binary_config, args=binary_args)
                    binary_model.to(device)
                except Exception as e:
                    logger.error(f"Error loading binary model: {e}")
                    return

                # Set up relation classifier parameters
                relation_args = copy.deepcopy(orig_args)
                relation_args.binary_mode = False  # Ensure multi-class mode
                relation_args.model_dir = os.path.join(args.model_dir, "relation")
                relation_args.binary_model_dir = binary_args.model_dir

                # Train relation classifier using standard CV approach
                relation_trainer = Trainer(relation_args, train_dataset=train_dataset, test_dataset=test_dataset)

                if args.do_train:
                    logger.info("Training relation classifier with cross-validation")

                    # Use standard train() method which handles cross-validation internally
                    relation_trainer.train()

                    # After training, evaluate with duo-classifier approach
                    if args.do_eval:
                        logger.info("Evaluating duo-classifier on test set")
                        relation_trainer.evaluate_duo_classifier("test", prefix="duo_final", save_cm=True)
            else:
                # Standard cross-validation (single classifier)
                # Initialize trainer with full training dataset
                trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset)

                # Train with k-fold cross-validation
                if args.do_train:
                    trainer.train()

                # Evaluate on test set
                if args.do_eval:
                    trainer.evaluate("test", prefix="final_test", save_cm=True)
        else:
            # Standard train/dev/test procedure (no cross-validation)
            logger.info("Standard training procedure (no cross-validation)")

            # Fix path construction
            train_file = Path(args.data_dir) / args.train_file

            # Check for dev file
            dev_file = None
            if hasattr(args, 'dev_file') and args.dev_file:
                dev_path = Path(args.data_dir) / args.dev_file
                if dev_path.exists():
                    dev_file = dev_path
                else:
                    logger.warning(f"Dev file {dev_path} not found.")

            # If no dev file specified or not found, try standard cross-validation files
            if dev_file is None and hasattr(args, 'do_dev_eval') and args.do_dev_eval:
                for i in range(5):  # Try up to 5 possible CV dev files
                    test_path = Path(args.data_dir) / f"dev_k_{i}.tsv"
                    if test_path.exists():
                        dev_file = test_path
                        logger.info(f"Found dev file: {dev_file}")
                        break

            # Ensure file paths are correct
            train_file = train_file.resolve()
            if dev_file:
                dev_file = dev_file.resolve()

            # Log file paths
            logger.info(f"Using training file: {train_file}")
            logger.info(f"Using validation file: {dev_file if dev_file else 'None'}")

            # Load datasets
            train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
            dev_dataset = None
            if dev_file:
                try:
                    # Save dev file path to args for loading
                    args.dev_file = str(dev_file.relative_to(args.data_dir))
                    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
                except Exception as e:
                    logger.warning(f"Failed to load dev dataset: {e}")

            test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

            # If duo-classifier is enabled
            if args.duo_classifier:
                logger.info("Training duo-classifier")

                # Save original args for later
                orig_args = copy.deepcopy(args)

                # Set up binary classifier training
                binary_args = copy.deepcopy(args)
                binary_args.binary_mode = True
                binary_args.data_dir = args.binary_data_dir
                binary_args.model_dir = os.path.join(args.model_dir, "binary")
                binary_args.train_file = f"binary_{args.train_file}"
                binary_args.test_file = f"binary_{args.test_file}"
                binary_args.label_file = "binary_label.txt"

                # Load binary datasets
                binary_train_dataset = load_and_cache_examples(binary_args, tokenizer, mode="train")
                binary_test_dataset = load_and_cache_examples(binary_args, tokenizer, mode="test")

                # Train binary classifier
                binary_trainer = Trainer(binary_args, train_dataset=binary_train_dataset,
                                         dev_dataset=None, test_dataset=binary_test_dataset)
                if args.do_train:
                    binary_trainer.train()

                # Load trained binary model for filtering
                logger.info("Loading trained binary model to filter training data for relation classifier")

                # Try to find the best model first
                binary_model_dir = os.path.join(binary_args.model_dir, "best")
                if not os.path.exists(binary_model_dir):
                    # Try final epoch
                    binary_model_dir = os.path.join(binary_args.model_dir, "epoch_final")
                    if not os.path.exists(binary_model_dir):
                        # Use the main directory
                        binary_model_dir = binary_args.model_dir

                # Update config for binary model
                binary_config = BertConfig.from_pretrained(
                    os.path.join(binary_model_dir, "config.json"),
                    num_labels=2
                )

                device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
                binary_model = RBERT.from_pretrained(binary_model_dir, config=binary_config, args=binary_args)
                binary_model.to(device)

                # Filter training data using binary classifier
                filtered_train_dataset = filter_dataset_with_binary_classifier(
                    args, binary_model, train_dataset, device
                )

                # Set up relation classifier training with filtered data
                relation_args = copy.deepcopy(orig_args)
                relation_args.model_dir = os.path.join(args.model_dir, "relation")
                relation_args.binary_model_dir = os.path.join(args.model_dir, "binary")

                # Train relation classifier on filtered data
                relation_trainer = Trainer(relation_args, train_dataset=filtered_train_dataset,
                                           dev_dataset=dev_dataset, test_dataset=test_dataset)
                if args.do_train:
                    relation_trainer.train()

                # Evaluate with duo-classifier approach
                if args.do_eval:
                    relation_trainer.evaluate_duo_classifier("test", prefix="duo_final", save_cm=True)
            else:
                # Standard single-classifier training
                # Initialize Trainer
                trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)

                # Train model
                if args.do_train:
                    trainer.train()

                # Evaluate model
                if args.do_eval:
                    trainer.evaluate("test", prefix="final_test", save_cm=True)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Cleanup old model files if requested
    if hasattr(args, 'cleanup_models') and args.cleanup_models:
        logger.info("Cleaning up model directories to save space...")

        # Cleanup main model directory
        cleanup_count = cleanup_models(args.model_dir,
                                       keep_best=True,
                                       keep_final=True)

        # If duo-classifier mode, also cleanup binary model directory
        if args.duo_classifier and hasattr(args, 'binary_model_dir') and args.binary_model_dir:
            binary_cleanup_count = cleanup_models(args.binary_model_dir,
                                                  keep_best=True,
                                                  keep_final=True)
            cleanup_count += binary_cleanup_count

        logger.info(f"Model cleanup complete. Removed {cleanup_count} directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic configurations
    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data directory")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation results directory")

    # Dataset files
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Training file name")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="Dev file name")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file name")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file name")

    # Model configuration
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="Model checkpoint")

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=77, help="Random seed for reproducibility")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation")
    parser.add_argument("--max_seq_len", default=128, type=int, help="Maximum input sequence length")

    parser.add_argument("--learning_rate", default=2e-05, type=float, help="Learning rate for Adam optimizer")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay for optimizer")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Maximum gradient norm")
    parser.add_argument("--max_steps", default=-1, type=int, help="Override num_train_epochs if > 0")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Number of warmup steps for scheduler")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate for layers")

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=250, help="Log progress every X steps")
    parser.add_argument("--save_steps", type=int, default=250, help="Save model every X steps")
    parser.add_argument("--save_epochs", type=int, default=1, help="Save model every X epochs")

    # Execution flags
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to evaluate on test set")
    parser.add_argument("--do_dev_eval", action="store_true", help="Whether to evaluate on validation set")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA if available")
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")

    # K-Fold cross-validation
    parser.add_argument("--k_folds", type=int, default=1, help="Number of folds for cross-validation (default: 1)")

    # Duo-classifier options
    parser.add_argument("--duo_classifier", action="store_true", help="Use duo-classifier approach (binary + relation)")
    parser.add_argument("--binary_model_dir", type=str, default="",
                        help="Path to binary classifier model (for evaluation)")
    parser.add_argument("--binary_threshold", type=float, default=0.5, help="Threshold for binary classifier")

    # Model cleanup option
    parser.add_argument("--cleanup_models", action="store_true",
                        help="Clean up intermediate model files, keeping only best and final models")

    args = parser.parse_args()
    main(args)