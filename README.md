# BERT Relation Extraction (R-BERT)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enriching-pre-trained-language-model-with/relation-extraction-on-semeval-2010-task-8)](https://paperswithcode.com/sota/relation-extraction-on-semeval-2010-task-8?p=enriching-pre-trained-language-model-with)

A PyTorch implementation of R-BERT: [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284). This implementation includes enhanced visualization capabilities, cross-validation features, and a novel duo-classifier architecture for more robust relation extraction.

## Model Architectures

This implementation supports two model architectures:

### 1. Single-Classifier Architecture (Original R-BERT)

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

### 2. Duo-Classifier Architecture (New Feature)

<p float="left" align="center">
    <img width="600" src="https://raw.githubusercontent.com/dthung1602/bert-relation-extraction/notebook/images/high_level.png" />  
</p>
<p float="left" align="center">
    <img width="600" src="https://raw.githubusercontent.com/dthung1602/bert-relation-extraction/notebook/images/high_level_bin.png" />  
</p>

## Project Structure

```
.
├── data/                  # Data directory
│   ├── train.tsv          # Training data
│   ├── test.tsv           # Test data
│   ├── dev_k_*.tsv        # Cross-validation dev files
│   ├── label.txt          # Relation labels
│   └── binary/            # Binary classification data (for duo-classifier)
│       ├── binary_train.tsv
│       ├── binary_test.tsv
│       └── binary_label.txt
├── model/                 # Model checkpoints
│   ├── standard/          # Standard single-classifier model
│   ├── cv/                # Cross-validation models
│   ├── duo/               # Duo-classifier models
│   │   └── binary/        # Binary classifier model
│   ├── duo_cv/            # CV duo-classifier models
│   │   └── binary/        # CV binary classifier model
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

## Model Implementations

### Original R-BERT Method

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

### Duo-Classifier Method

1. **Binary Classification Stage**
   - Uses the same R-BERT architecture but with only two output classes
   - Determines if any relation exists between entities
   - Training on binary labels (0: No-Relation, 1: Has-Relation)

2. **Relation Classification Stage**
   - Only processes instances that passed the binary classifier
   - Uses the standard R-BERT model to classify specific relation types
   - Focuses on distinguishing between different relation classes

## Duo-Classifier (Two-Stage) Architecture

### Design Philosophy

The duo-classifier architecture addresses class imbalance and complex classification challenges in relation extraction:

1. **Binary Classification Stage (Relation Existence)**
   - Objective: Quickly determine if a relation exists between entities
   - Output: 0 (No Relation) or 1 (Relation Exists)
   - Input: Original sentence with entity markers
   - Use case: Filter out irrelevant or noise-heavy samples

2. **Multi-class Relation Classification Stage**
   - Processes only samples identified as having a relation
   - Precisely identifies specific relation types
   - Focuses computational resources on meaningful relationship candidates

### Key Advantages

- **Reduce False Positives**: Two-stage filtering minimizes irrelevant sample misclassification
- **Address Class Imbalance**: Separate relation existence and type identification
- **Improve Classification Accuracy**: Train specialized classifiers for each stage
- **Computational Efficiency**: Reduce processing for non-relational samples
- **Flexible Thresholding**: Adjust binary classifier threshold for precision/recall trade-off

### Implementation Strategies

- **Binary Classifier**
  - Lightweight model optimized for quick relation existence detection
  - Trained to maximize early filtering efficiency
  - Uses similar R-BERT architecture with reduced complexity

- **Relation Classifier**
  - Full-capacity model for precise relation type identification
  - Only processes samples pre-filtered by binary classifier
  - Maintains high-resolution feature extraction

- - ## Visualization Capabilities
  
    The project provides comprehensive and detailed visualization tools to help understand model performance and training dynamics:
  
    ### 1. Training Loss Visualization
  
    #### Detailed Loss Curves
  
    - Plots training and validation loss across epochs
    - Features:
      - Color-coded curves for training and validation losses
      - Minimum loss point annotations
      - Handles abnormal loss values by filtering
    - Supports both single training and cross-validation modes
    - Generates high-quality PNG plots
    - Saves accompanying CSV data for further analysis
  
    #### Key Visualization Features
  
    - Dynamic y-axis scaling
    - Grid and minor grid lines for enhanced readability
    - Epoch-based x-axis
    - Informative annotations for minimum loss points
  
    ### 2. Performance Metrics Visualization
  
    #### Multi-Metric Tracking
  
    - Simultaneous visualization of key performance metrics:
      - Accuracy
      - Precision
      - Recall
      - F1 Score
    - Features:
      - Distinct colors and markers for each metric
      - Comprehensive performance tracking across training epochs
      - Supports both single training and cross-validation modes
  
    #### Visualization Characteristics
  
    - Score range from 0 to 1
    - Detailed grid system
    - Epoch-based x-axis
    - Legend with metric details
    - High-resolution PNG output
    - Accompanying CSV data export
  
    ### 3. Confusion Matrix Visualization
  
    #### Comprehensive Confusion Matrix Analysis
  
    - Two visualization modes:
      1. Raw Count Confusion Matrix
      2. Normalized Confusion Matrix
    - Features:
      - Heatmap representation using Blues color palette
      - Actual count and percentage annotations
      - Supports full and abbreviated class names
      - Handles multi-class classification scenarios
  
    #### Visualization Details
  
    - Improved readability with rotated labels
    - Tight layout and high-resolution output
    - Saves both visualization and raw data (CSV)
    - Separate plots for raw and normalized matrices
  
    ### 4. Cross-Validation Results Visualization
  
    #### Comprehensive Performance Comparison
  
    - Visualizes performance across different cross-validation folds
    - Plots all metrics with:
      - Individual fold performance
      - Average performance line
      - Standard deviation indication
  
    #### Advanced Features
  
    - Calculates and displays:
      - Mean performance
      - Performance standard deviation
    - Generates:
      - Comprehensive PNG visualization
      - Detailed CSV results
      - Textual summary report
  
    ### Output and Storage
  
    #### Visualization Outputs
  
    - PNG plots saved in `model/results/plots/`
    - CSV data files in `model/results/`
    - Supports fold-specific and aggregate visualizations
  
    #### Visualization Types
  
    - Loss curves
    - Metrics progression
    - Confusion matrices
    - Cross-validation performance
    - Model-specific performance graphics
  
    ### Customization and Extensibility
  
    - Flexible handling of different dataset sizes
    - Robust error handling
    - Support for both single and cross-validation training modes
    - Easy integration with existing training workflow

## How to Run

### Using the Batch Script (Recommended)

```bash
# Standard Training (Single-Classifier)
.\run_model.bat

# Cross-validation Training (Single-Classifier)
.\run_model.bat cv

# Evaluation Only (Single-Classifier)
.\run_model.bat eval

# Evaluation Only (CV Single-Classifier)
.\run_model.bat cv-eval

# Standard Training with Duo-Classifier
.\run_model.bat duo

# Cross-validation Training with Duo-Classifier
.\run_model.bat duo-cv

# Evaluation Only (Standard Duo-Classifier)
.\run_model.bat duo-eval

# Evaluation Only (CV Duo-Classifier)
.\run_model.bat duo-cv-eval
```

### Using Python Directly

#### Standard Single-Classifier Scenarios
```bash
# Standard Training
python main.py --do_train --do_eval --model_dir ./model/standard

# Cross-Validation Training
python main.py --do_train --do_eval --k_folds 5 --model_dir ./model/cv
```

#### Duo-Classifier Scenarios
```bash
# Standard Duo-Classifier Training
python main.py --do_train --do_eval --duo_classifier --model_dir ./model/duo

# Cross-Validation Duo-Classifier Training
python main.py --do_train --do_eval --duo_classifier --k_folds 5 --model_dir ./model/duo_cv
```

## Key Parameters

### Training Parameters
- `--model_name_or_path`: Pre-trained BERT model (default: bert-base-uncased)
- `--data_dir`: Data directory (default: ./data)
- `--train_file`: Training file (default: train.tsv)
- `--test_file`: Test file (default: test.tsv)
- `--num_train_epochs`: Number of training epochs (default: 5)
- `--train_batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)

### Duo-Classifier Specific Parameters
- `--duo_classifier`: Enable two-stage classification
- `--binary_threshold`: Relation existence threshold (default: 0.5)
  - Lower threshold: More inclusive, higher recall
  - Higher threshold: More selective, higher precision

### Execution Flags
- `--do_train`: Run training process
- `--do_eval`: Perform model evaluation
- `--k_folds`: Cross-validation fold count (default: 1)

## Cross-Validation Implementation

### K-Fold Cross-Validation Strategy

#### Core Approach
- Splits entire dataset into K stratified folds
- Ensures representative class distribution
- Provides robust performance estimation

#### Training Process
1. **Fold Preparation**
   - Randomly split data maintaining class proportions
   - Create training and validation sets for each fold

2. **Per-Fold Training**
   - Train model K times
   - Each iteration uses different train/validation split
   - Reset model parameters between folds

3. **Performance Aggregation**
   - Compute metrics for each fold
   - Calculate mean and standard deviation
   - Assess model's generalization capability

## Dataset

### SemEval 2010 Task 8 Dataset

Includes 9 directional relation types and 1 "Other" relation:
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

### Dataset Format
- TSV files with sentences containing entity markers
- Example: "The <e1>system</e1> produces a <e2>signal</e2>"
- Relations represented as integers (0-9)

## Dependencies

- Python >= 3.6
- PyTorch >= 1.6.0
- Transformers >= 3.3.1
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn (for visualization)

## Making Predictions

### Standard Model
```bash
python predict.py --input_file {INPUT_FILE} --output_file {OUTPUT_FILE} --model_dir {MODEL_PATH}
```

### Duo-Classifier Model
```bash
python predict.py \
    --input_file {INPUT_FILE} \
    --output_file {OUTPUT_FILE} \
    --model_dir {RELATION_MODEL_PATH} \
    --duo_classifier \
    --binary_model_dir {BINARY_MODEL_PATH} \
    --binary_threshold 0.5
```

## References

- [Original Paper](https://arxiv.org/abs/1905.08284)
- [SemEval 2010 Task 8 Dataset](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view)
- [Huggingface Transformers](https://github.com/huggingface/transformers)