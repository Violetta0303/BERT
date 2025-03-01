import logging
import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

from model import RBERT
from utils import compute_metrics, get_label, write_prediction

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        self.config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(self.label_lst)},
            label2id={label: i for i, label in enumerate(self.label_lst)},
        )
        self.model = RBERT.from_pretrained(args.model_name_or_path, config=self.config, args=args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)
        print(self.device)

    def train(self):
        """Trains the model using either standard training or K-Fold cross-validation."""

        if self.args.k_folds > 1:
            logger.info(f"Starting {self.args.k_folds}-Fold Cross-Validation")
            # Ensure the original dataset is split into K folds
            kfold_splits = self.split_kfold(self.args.k_folds)
        else:
            kfold_splits = [(self.train_dataset, None)]  # No k-fold, use entire dataset

        for fold, (train_dataset, dev_dataset) in enumerate(kfold_splits):
            logger.info(f"Training on Fold {fold + 1}/{self.args.k_folds}")

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=self.args.train_batch_size,
            )

            if self.args.max_steps > 0:
                t_total = self.args.max_steps
                self.args.num_train_epochs = (
                        self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
                )
            else:
                t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

            # Configure optimiser and scheduler
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=t_total,
            )

            # Training loop
            logger.info("***** Running training *****")
            logger.info("  Number of examples = %d", len(train_dataset))
            logger.info("  Number of Epochs = %d", self.args.num_train_epochs)
            logger.info("  Total train batch size = %d", self.args.train_batch_size)
            logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
            logger.info("  Total optimisation steps = %d", t_total)
            logger.info("  Logging steps = %d", self.args.logging_steps)
            logger.info("  Save steps = %d", self.args.save_steps)

            global_step = 0
            tr_loss = 0.0
            self.model.zero_grad()

            train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

            for epoch in train_iterator:
                epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
                for step, batch in enumerate(epoch_iterator):
                    self.model.train()
                    batch = tuple(t.to(self.device) for t in batch)  # Move batch to GPU or CPU
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3],
                        "e1_mask": batch[4],
                        "e2_mask": batch[5],
                    }
                    outputs = self.model(**inputs)
                    loss = outputs[0]

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        self.model.zero_grad()
                        global_step += 1

                        if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                            if dev_dataset:
                                self.evaluate("dev")  # Validate on dev set for cross-validation
                            else:
                                self.evaluate("test")  # Direct evaluation on the test set if no dev set exists

                        if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                            self.save_model()

                    if 0 < self.args.max_steps < global_step:
                        epoch_iterator.close()
                        break

                if 0 < self.args.max_steps < global_step:
                    train_iterator.close()
                    break

        return global_step, tr_loss / global_step

    def split_kfold(self, k=5):
        """Splits the dataset into K folds for cross-validation"""
        all_data = list(self.train_dataset)  # Convert dataset to list for indexing
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        split_data = []
        for train_idx, dev_idx in kf.split(all_data):
            train_subset = torch.utils.data.Subset(self.train_dataset, train_idx)
            dev_subset = torch.utils.data.Subset(self.train_dataset, dev_idx)
            split_data.append((train_subset, dev_subset))

        return split_data

    def evaluate(self, mode):
        """
        Evaluates the model on dev or test set.

        :param mode: "dev" or "test"
        """
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        preds = np.argmax(preds, axis=1)

        # **重新生成 answer_keys.txt**
        write_prediction(self.args, os.path.join(self.args.eval_dir, "proposed_answers.txt"), preds)
        write_prediction(self.args, os.path.join(self.args.eval_dir, "answer_keys.txt"), out_label_ids)  # ✅ 新增

        # **计算评估结果**
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        self.args = torch.load(os.path.join(self.args.model_dir, "training_args.bin"))
        self.model = RBERT.from_pretrained(self.args.model_dir, args=self.args)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")
