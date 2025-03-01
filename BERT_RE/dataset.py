import os
import pandas as pd
from datasets import load_dataset

# Ensure the 'data' directory exists
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# Download the SemEval 2010 Task 8 dataset
dataset = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8", cache_dir=data_dir)

# Define the file paths for saving the TSV files
train_tsv_path = os.path.join(data_dir, "train.tsv")
test_tsv_path = os.path.join(data_dir, "test.tsv")

# Convert the dataset into a DataFrame and save it as a TSV file
def save_as_tsv(dataset_split, file_path):
    df = pd.DataFrame(dataset_split)  # Convert to a DataFrame
    df.to_csv(file_path, sep="\t", index=False)  # Save as a TSV file
    print(f"Data has been saved to {file_path}")

# Convert and save the dataset
save_as_tsv(dataset["train"], train_tsv_path)
save_as_tsv(dataset["test"], test_tsv_path)
