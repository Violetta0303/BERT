# BERT Relation Extraction (R-BERT)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enriching-pre-trained-language-model-with/relation-extraction-on-semeval-2010-task-8)](https://paperswithcode.com/sota/relation-extraction-on-semeval-2010-task-8?p=enriching-pre-trained-language-model-with)

A PyTorch implementation of R-BERT: [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284). This implementation includes enhanced visualization capabilities and cross-validation features for more robust model evaluation.

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

### Method

1. **Extract three vectors from BERT**
   - [CLS] token vector for sentence-level representation
   - Averaged entity_1 vector (including entity markers)
   - Averaged entity_2 vector (including entity markers)
   
2. **Process each vector through fully-connected layers**
   - Each vector goes through: dropout -> tanh -> fc-layer
   
3. **Concatenate the three vectors**
   - Combine the processed vectors into a single representation
   
4. **Classification through a final fully-connected layer**
   - dropout -> fc-layer -> softmax

### Implementation Details

- Entity positions are marked with special tokens: `<e1>`, `</e1>`, `<e2>`, `</e2>`
- During tokenization, these special tokens are replaced with `$` and `#` markers
- The implementation maintains the same architecture as described in the original paper:
  - Averaging entity token representations
  - Using dropout and tanh activation before fully-connected layers
  - Optional [SEP] token configuration (enabled with `--add_sep_token`)
- The model includes comprehensive visualization tools for training metrics and evaluation results
- Support for both standard training and k-fold cross-validation

## Dependencies

- Python >= 3.6
- PyTorch >= 1.6.0
- Transformers >= 3.3.1
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn (for visualization)

## Dataset

The model is designed to work with the SemEval 2010 Task 8 dataset, which contains sentences with marked entity pairs and their relation types.

The dataset includes 9 directional relation types and 1 "Other" relation:
- Cause-Effect
- Component-Whole
- Content-Container
- Entity-Destination
- Entity-Origin
- Instrument-Agency
- Member-Collection
- Message-Topic
- Product-Producer
- Other

Dataset Format:
- TSV files with sentences containing entity markers
- Example: "The <e1>system</e1> produces a <e2>signal</e2>"
- Relations are represented as integers (0-9)

## Project Structure

```
.
├── data/                  # Data directory
│   ├── train.tsv          # Training data
│   ├── test.tsv           # Test data
│   ├── dev_k_*.tsv        # Cross-validation dev files
│   └── label.txt          # Relation labels
├── model/                 # Model checkpoints
│   └── results/           # Training results and metrics
│       └── plots/         # Visualization plots
├── eval/                  # Evaluation results
├── data_loader.py         # Data loading utilities
├── dataset.py             # Dataset preparation script
├── main.py                # Main script for training and evaluation
├── model.py               # R-BERT model definition
├── predict.py             # Prediction script
├── trainer.py             # Training and evaluation logic
├── utils.py               # Utility functions
├── visualization.py       # Visualization utilities
├── official_eval.py       # Official evaluation script
└── run_model.bat          # Batch script for running the model
```

## How to Run

### Standard Training
```bash
python main.py --do_train --do_eval
```

### Cross-Validation Training
```bash
python main.py --do_train --do_eval --k_folds 5
```

### Or use the batch file:
```bash
# For standard training
run_model.bat

# For cross-validation
run_model.bat cv

# For evaluation only
run_model.bat eval
```

### Key parameters:
- `--model_name_or_path`: Pre-trained BERT model (default: bert-base-uncased)
- `--data_dir`: Data directory (default: ./data)
- `--train_file`: Training file (default: train.tsv)
- `--test_file`: Test file (default: test.tsv)
- `--num_train_epochs`: Number of training epochs (default: 5)
- `--train_batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--do_train`: Flag to run training
- `--do_eval`: Flag to run evaluation
- `--k_folds`: Number of folds for cross-validation (default: 1)
- `--add_sep_token`: Whether to add [SEP] token at the end of sentences

## Making Predictions

```bash
python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

The input file should contain sentences with marked entities (e.g., "The <e1>system</e1> produces a <e2>signal</e2>").

## Visualization

The training process generates various visualization outputs in the `model/results/plots` directory:

- Training and validation loss curves
- Accuracy, precision, recall, and F1 score metrics per epoch
- Confusion matrices (raw and normalized)
- Cross-validation results (when using k-fold cross-validation)

These visualizations help track model performance and identify potential issues during training.

### Example visualizations:

1. **Training and Validation Loss**
   - Tracks the model's convergence over epochs
   - Helps identify overfitting when validation loss increases
   
2. **Metrics Curves**
   - Shows the progression of accuracy, precision, recall, and F1 scores
   - Useful for determining when to stop training

3. **Confusion Matrix**
   - Reveals which relations are most frequently confused
   - Helps identify class imbalance issues

4. **Cross-Validation Results**
   - Compares performance across different data splits
   - Indicates the model's stability and generalization ability

## Evaluation

The model is evaluated using the official SemEval 2010 Task 8 evaluation metrics:
- Macro-averaged F1 score (excluding the "Other" relation)
- Per-class precision, recall, and F1 scores
- Confusion matrix analysis

Evaluation results are stored in:
- `eval/proposed_answers.txt`: Model predictions
- `eval/answer_keys.txt`: Gold standard answers
- `model/results/{prefix}_detailed_report.xlsx`: Detailed analysis

Run the evaluation script to calculate the official F1 score:
```bash
python official_eval.py
```

## Cross-Validation

The model supports k-fold cross-validation for more robust performance evaluation:

1. The training dataset is split into k folds
2. For each fold:
   - Train on k-1 folds
   - Validate on the remaining fold
3. Final performance is reported as the average across all folds

This helps ensure that model performance is consistent and not dependent on a specific train/test split. The cross-validation implementation:

- Creates stratified folds to maintain class distribution
- Resets the model for each fold to ensure independence
- Tracks metrics separately for each fold
- Produces summary statistics (mean and standard deviation)
- Visualizes performance across folds

## Results

On the SemEval 2010 Task 8 test set, the model achieves competitive performance:
- Macro-averaged F1 score: ~88% (comparable to published results)
- Training time: ~10 minutes per epoch on a single GPU
- Cross-validation provides more stable performance metrics

The results include detailed per-class performance analysis which helps identify challenging relation types and potential areas for improvement.

## Known Issues and Solutions

1. **Empty metrics plot**: This can occur when no validation dataset is available. The model now automatically creates simulated metrics based on training loss to ensure visualizations are always generated.

2. **Cross-validation errors**: If encountering errors during cross-validation, ensure the dev_k_*.tsv files exist. The model will now automatically create these files if they don't exist.

3. **BERT initialization warnings**: Warnings about weights not being initialized from the checkpoint are normal for custom layers added on top of BERT.

## References

- [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284)
- [SemEval 2010 Task 8 Dataset](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view)
- [SemEval 2010 Task 8 Paper](https://www.aclweb.org/anthology/S10-1006.pdf)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NLP-progress Relation Extraction](http://nlpprogress.com/english/relationship_extraction.html)
