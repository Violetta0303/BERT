import argparse
import os
import logging
from pathlib import Path
from data_loader import load_and_cache_examples, SemEvalProcessor
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed

logger = logging.getLogger(__name__)

def main(args):
    """Main function to initialise training and evaluation."""
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    processor = SemEvalProcessor(args)

    # If K-Fold cross-validation is enabled, split the dataset accordingly
    if args.k_folds > 1:
        processor.split_kfold(args.k_folds)

    # Perform K-Fold cross-validation
    for fold in range(args.k_folds):
        logger.info(f"Starting training for Fold {fold + 1}/{args.k_folds}")

        # **修正路径拼接问题，避免 data/data 错误**
        train_file = Path(args.data_dir) / f"train_k_{fold}.tsv" if args.k_folds > 1 else Path(args.data_dir) / args.train_file
        dev_file = Path(args.data_dir) / f"dev_k_{fold}.tsv" if args.k_folds > 1 else None

        # **确保文件路径格式正确**
        train_file = train_file.resolve()
        dev_file = dev_file.resolve() if dev_file else None

        # **日志记录文件路径，方便调试**
        logger.info(f"Using training file: {train_file}")
        logger.info(f"Using validation file: {dev_file if dev_file else 'None'}")

        # **加载数据集**
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train", file_override=str(train_file))
        dev_dataset = (
            load_and_cache_examples(args, tokenizer, mode="dev", file_override=str(dev_file)) if dev_file else None
        )
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

        # **初始化 Trainer**
        trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)

        # **执行训练**
        if args.do_train:
            trainer.train()

        # **交叉验证时评估模型**
        if args.do_dev_eval and dev_dataset:
            trainer.evaluate("dev")

    # **最终测试集评估**
    if args.do_eval:
        logger.info("Loading final model for evaluation on test set")
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic configurations
    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data directory")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation results directory")

    # Dataset files
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Training file name")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file name")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file name")

    # Model configuration
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="Model checkpoint")

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=77, help="Random seed for reproducibility")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation")
    parser.add_argument("--max_seq_len", default=384, type=int, help="Maximum input sequence length")

    parser.add_argument("--learning_rate", default=2e-05, type=float, help="Learning rate for Adam optimiser")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay for optimiser")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimiser")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Maximum gradient norm")
    parser.add_argument("--max_steps", default=-1, type=int, help="Override num_train_epochs if > 0")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Number of warmup steps for scheduler")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate for layers")

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=250, help="Log progress every X steps")
    parser.add_argument("--save_steps", type=int, default=250, help="Save model every X steps")

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