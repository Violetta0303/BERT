import copy
import csv
import logging
import os
import random

import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold
from utils import get_label

logger = logging.getLogger(__name__)


class InputExample:
    """
    A single training/test example for sequence classification.

    Args:
        guid: Unique identifier for the example.
        text_a: The untokenised text of the first sequence.
        label: (Optional) Integer label for classification.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialises this instance into a Python dictionary."""
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        """Serialises this instance into a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures:
    """
    A single set of tokenised input features.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialises this instance into a Python dictionary."""
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        """Serialises this instance into a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SemEvalProcessor:
    """Processor for the SemEval dataset stored in TSV format."""

    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    # def _read_tsv(self, input_file):
    #     """Reads a TSV file and returns the lines (excluding the header)."""
    #     input_file = input_file.replace("\\", "/")  # Ensure compatibility across operating systems
    #     with open(input_file, "r", encoding="utf-8") as f:
    #         reader = csv.reader(f, delimiter="\t")
    #         lines = list(reader)
    #         return lines[1:]  # Skip header row
    def _read_tsv(self, input_file):
        """Reads a TSV file and ensures correct formatting."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = [line for line in reader if line and len(line) >= 2]  # 跳过空行或列数不足的行
            return lines[1:]  # 跳过标题行

    def _create_examples(self, lines, set_type):
        """Creates InputExample instances from TSV lines."""
        examples = []
        for i, line in enumerate(lines):
            if len(line) < 2:  # Check
                logger.warning(f"Skipping malformed line {i}: {line}")
                continue

            guid = f"{set_type}-{i}"
            text_a = line[0]  # Sentence
            label = int(line[1])  # Convert relation ID to integer
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode, file_override=None):
        """
        Retrieves data examples for the given mode (train/dev/test).

        Args:
            mode (str): One of 'train', 'dev', 'test'.
            file_override (str): Optional path to override the default file.
        """
        file_map = {
            "train": self.args.train_file,
            "dev": getattr(self.args, "dev_file", None),
            "test": self.args.test_file
        }

        file_to_read = file_override if file_override else file_map.get(mode)
        if not file_to_read:
            raise ValueError(f"No file specified for mode: {mode}")

        file_path = os.path.join(self.args.data_dir, file_to_read).replace("\\", "/")
        logger.info(f"Loading {mode} dataset from {file_path}")

        return self._create_examples(self._read_tsv(file_path), mode)

    def split_kfold(self, k=5):
        """Splits the dataset into K folds for cross-validation."""
        train_path = os.path.join(self.args.data_dir, self.args.train_file).replace("\\", "/")
        lines = self._read_tsv(train_path)
        random.shuffle(lines)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for fold, (train_idx, dev_idx) in enumerate(kf.split(lines)):
            train_lines = [lines[i] for i in train_idx]
            dev_lines = [lines[i] for i in dev_idx]

            train_file = os.path.join(self.args.data_dir, f"train_k_{fold}.tsv").replace("\\", "/")
            dev_file = os.path.join(self.args.data_dir, f"dev_k_{fold}.tsv").replace("\\", "/")

            logger.info(f"Saving Fold {fold}: {len(train_lines)} train / {len(dev_lines)} dev")

            with open(train_file, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["sentence", "relation"])
                writer.writerows(train_lines)

            with open(dev_file, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["sentence", "relation"])
                writer.writerows(dev_lines)

        logger.info(f"K-Fold data split completed with {k} folds.")


processors = {"semeval": SemEvalProcessor}


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    add_sep_token=False,
    mask_padding_with_zero=True,
):
    """
    Converts InputExample instances into tokenised input features.

    Args:
        examples: List of InputExample instances.
        max_seq_len: Maximum sequence length allowed.
        tokenizer: Tokeniser instance (e.g., BERT tokenizer).
        cls_token: Special token used at the beginning of the sequence.
        cls_token_segment_id: Segment ID for the [CLS] token.
        sep_token: Special token used to separate sequences.
        pad_token: Padding token ID.
        pad_token_segment_id: Segment ID for padding tokens.
        sequence_a_segment_id: Segment ID for sentence A.
        add_sep_token: Whether to add a [SEP] token at the end.
        mask_padding_with_zero: Whether to use 1 for real tokens and 0 for padding.

    Returns:
        A list of InputFeatures instances.
    """

    features = []
    label_set = set()  # 用于检查 label 是否超出范围

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Processing example %d of %d" % (ex_index, len(examples)))

        # Tokenise input sentence
        tokens_a = tokenizer.tokenize(example.text_a)

        # Identify entity markers
        try:
            e11_p = tokens_a.index("<e1>")  # Start position of entity 1
            e12_p = tokens_a.index("</e1>")  # End position of entity 1
            e21_p = tokens_a.index("<e2>")  # Start position of entity 2
            e22_p = tokens_a.index("</e2>")  # End position of entity 2
        except ValueError:
            logger.warning(f"Skipping example {ex_index} due to missing entity markers.")
            continue

        # Replace entity markers with distinct symbols
        tokens_a[e11_p] = "$"
        tokens_a[e12_p] = "$"
        tokens_a[e21_p] = "#"
        tokens_a[e22_p] = "#"

        # Adjust entity positions to account for [CLS] token
        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        # Determine number of special tokens required
        special_tokens_count = 2 if add_sep_token else 1

        # Truncate sequence if it exceeds maximum length
        if len(tokens_a) > max_seq_len - special_tokens_count:
            tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]

        # Construct final token sequence
        tokens = tokens_a + ([sep_token] if add_sep_token else [])

        # Assign segment IDs (all 0 for single-sentence tasks)
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Prepend [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # Convert tokens to token IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Generate attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Pad sequences to maximum length
        padding_length = max_seq_len - len(input_ids)
        input_ids += [pad_token] * padding_length
        attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
        token_type_ids += [pad_token_segment_id] * padding_length

        # Create entity masks
        e1_mask = [0] * max_seq_len
        e2_mask = [0] * max_seq_len

        # Ensure entity positions are within range
        if e12_p < max_seq_len and e22_p < max_seq_len:
            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1
        else:
            logger.warning(f"Skipping example {ex_index} due to out-of-bounds entity positions.")
            continue

        # Convert label to integer
        label_id = int(example.label)
        label_set.add(label_id)  # 记录所有出现的 label

        # Log the first few examples
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        # Store the converted features
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
            )
        )

    logger.info(f"Label distribution in dataset: {sorted(label_set)}")
    return features


def load_and_cache_examples(args, tokenizer, mode, file_override=None):
    """
    Loads dataset features into a PyTorch TensorDataset.

    Args:
        args: Command-line arguments.
        tokenizer: Tokeniser instance.
        mode: Dataset mode (train/dev/test).
        file_override: Optional dataset path override.

    Returns:
        A PyTorch TensorDataset containing all processed features.
    """
    processor = processors[args.task](args)

    # Get examples from the dataset
    examples = processor.get_examples(mode, file_override)

    # Convert examples into model-ready features
    features = convert_examples_to_features(
        examples,
        args.max_seq_len,
        tokenizer,
        add_sep_token=args.add_sep_token
    )

    logger.info(f"Processed {len(features)} examples for {mode}")

    # Convert features to PyTorch tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # Ensure entity 1 mask is included
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # Ensure entity 2 mask is included

    logger.info(f"Feature shapes: input_ids={all_input_ids.shape}, "
                f"attention_mask={all_attention_mask.shape}, "
                f"token_type_ids={all_token_type_ids.shape}, "
                f"label_ids={all_label_ids.shape}, "
                f"e1_mask={all_e1_mask.shape}, "
                f"e2_mask={all_e2_mask.shape}")

    return TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_ids,
        all_e1_mask,  # Entity 1 mask
        all_e2_mask   # Entity 2 mask
    )

