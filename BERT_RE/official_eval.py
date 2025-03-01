import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

EVAL_DIR = "eval"

EVAL_DIR = "eval"


def compute_metrics_from_files():
    """Reads the answer files and computes Accuracy, Precision, Recall, and F1-score"""
    proposed_answers_path = os.path.join(EVAL_DIR, "proposed_answers.txt")
    answer_keys_path = os.path.join(EVAL_DIR, "answer_keys.txt")

    if not os.path.exists(proposed_answers_path) or not os.path.exists(answer_keys_path):
        raise Exception("Error: proposed_answers.txt or answer_keys.txt is missing")

    with open(proposed_answers_path, "r", encoding="utf-8") as f:
        proposed_answers = [line.strip() for line in f.readlines()]

    with open(answer_keys_path, "r", encoding="utf-8") as f:
        answer_keys = [line.strip() for line in f.readlines()]

    if len(proposed_answers) != len(answer_keys):
        raise ValueError("Mismatch: Number of proposed answers and answer keys are not the same.")

    # Convert string labels to numerical values
    unique_labels = list(set(answer_keys + proposed_answers))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    y_true = [label_map[label] for label in answer_keys]
    y_pred = [label_map[label] for label in proposed_answers]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        # "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }

    return metrics


def official_metrics():
    """Returns all computed metrics instead of just F1"""
    return compute_metrics_from_files()


if __name__ == "__main__":
    metrics = official_metrics()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
