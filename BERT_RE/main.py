import argparse
import os
import logging
from pathlib import Path
import sys
from data_loader import load_and_cache_examples, SemEvalProcessor
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed

logger = logging.getLogger(__name__)


def main(args):
    """Main function to initialise training and evaluation."""
    init_logger()
    set_seed(args)

    try:
        tokenizer = load_tokenizer(args)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

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

            # Initialize trainer with full training dataset
            trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset)

            # Train with k-fold cross-validation (trainer will handle the fold splitting)
            if args.do_train:
                trainer.train()

            # Evaluate on test set
            if args.do_eval:
                trainer.evaluate("test", prefix="final_test", save_cm=True)
        else:
            # Standard train/dev/test procedure
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
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training")
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

    args = parser.parse_args()
    main(args)