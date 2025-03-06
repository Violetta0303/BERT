@echo on
REM Create necessary directories
mkdir .\data 2>nul
mkdir .\model 2>nul
mkdir .\eval 2>nul

echo Step 1: Checking directories...
if exist ".\data" (echo Data directory exists) else (echo Data directory missing)
if exist ".\model" (echo Model directory exists) else (echo Model directory missing)
if exist ".\eval" (echo Eval directory exists) else (echo Eval directory missing)

echo Step 2: Checking dataset files...
if exist ".\data\train.tsv" (echo Training file exists) else (echo Training file missing)
if exist ".\data\test.tsv" (echo Test file exists) else (echo Test file missing)
for /L %%i in (0,1,4) do (
    if exist ".\data\dev_k_%%i.tsv" (echo Dev file dev_k_%%i.tsv exists) else (echo Dev file dev_k_%%i.tsv missing)
)

REM Create label file if it doesn't exist
if not exist ".\data\label.txt" (
    echo Creating label file...
    (
        echo Other
        echo Cause-Effect
        echo Instrument-Agency
        echo Product-Producer
        echo Content-Container
        echo Entity-Origin
        echo Entity-Destination
        echo Component-Whole
        echo Member-Collection
        echo Message-Topic
    ) > .\data\label.txt
    echo Label file created
) else (
    echo Label file exists
)

echo Step 3: Checking required Python packages...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import numpy; import pandas; import sklearn; import matplotlib; print('All required packages installed')"

REM Optional check for openpyxl
python -c "try: import openpyxl; print('openpyxl is installed - Excel export available'); except ImportError: print('openpyxl not installed - Excel export will fall back to CSV')" 2>nul

if "%1"=="cv" (
    call :train_cv
) else if "%1"=="eval" (
    call :evaluate_model
) else if "%1"=="cv-eval" (
    call :evaluate_cv_model
) else if "%1"=="duo" (
    call :train_duo
) else if "%1"=="duo-cv" (
    call :train_duo_cv
) else if "%1"=="duo-eval" (
    call :evaluate_duo
) else if "%1"=="duo-cv-eval" (
    call :evaluate_duo_cv
) else (
    call :train_standard
)

echo Exit code: %ERRORLEVEL%
echo Process completed!
echo Check '.\model\*\results\' directory for evaluation results and metrics
echo Check '.\model\*\results\plots\' directory for visualization charts
exit /b 0

:train_standard
echo Starting standard training...
python main.py ^
    --model_name_or_path bert-base-uncased ^
    --task semeval ^
    --train_file train.tsv ^
    --test_file test.tsv ^
    --data_dir ./data ^
    --model_dir ./model/standard ^
    --eval_dir ./eval/standard ^
    --max_seq_len 128 ^
    --num_train_epochs 5 ^
    --train_batch_size 16 ^
    --learning_rate 2e-5 ^
    --save_epochs 1 ^
    --do_train ^
    --do_eval
exit /b 0

:train_cv
echo Starting 5-fold cross-validation training...
python main.py ^
    --model_name_or_path bert-base-uncased ^
    --task semeval ^
    --train_file train.tsv ^
    --test_file test.tsv ^
    --data_dir ./data ^
    --model_dir ./model/cv ^
    --eval_dir ./eval/cv ^
    --max_seq_len 128 ^
    --num_train_epochs 5 ^
    --train_batch_size 16 ^
    --learning_rate 2e-5 ^
    --save_epochs 1 ^
    --k_folds 5 ^
    --do_train ^
    --do_eval
exit /b 0

:train_duo
echo Starting duo-classifier training...
python main.py ^
    --model_name_or_path bert-base-uncased ^
    --task semeval ^
    --train_file train.tsv ^
    --test_file test.tsv ^
    --data_dir ./data ^
    --model_dir ./model/duo ^
    --eval_dir ./eval/duo ^
    --max_seq_len 128 ^
    --num_train_epochs 5 ^
    --train_batch_size 16 ^
    --learning_rate 2e-5 ^
    --save_epochs 1 ^
    --duo_classifier ^
    --binary_threshold 0.5 ^
    --do_train ^
    --do_eval
exit /b 0

:train_duo_cv
echo Starting duo-classifier training with cross-validation...
python main.py ^
    --model_name_or_path bert-base-uncased ^
    --task semeval ^
    --train_file train.tsv ^
    --test_file test.tsv ^
    --data_dir ./data ^
    --model_dir ./model/duo_cv ^
    --eval_dir ./eval/duo_cv ^
    --max_seq_len 128 ^
    --num_train_epochs 5 ^
    --train_batch_size 16 ^
    --learning_rate 2e-5 ^
    --save_epochs 1 ^
    --duo_classifier ^
    --binary_threshold 0.5 ^
    --k_folds 5 ^
    --do_train ^
    --do_eval
exit /b 0

:evaluate_model
echo Evaluating standard model...
python main.py ^
    --model_name_or_path bert-base-uncased ^
    --task semeval ^
    --test_file test.tsv ^
    --data_dir ./data ^
    --model_dir ./model/standard ^
    --eval_dir ./eval/evaluation ^
    --max_seq_len 128 ^
    --do_eval ^
    --do_dev_eval
exit /b 0

:evaluate_cv_model
echo Evaluating cross-validation model...
python main.py ^
    --model_name_or_path bert-base-uncased ^
    --task semeval ^
    --test_file test.tsv ^
    --data_dir ./data ^
    --model_dir ./model/cv ^
    --eval_dir ./eval/cv_evaluation ^
    --max_seq_len 128 ^
    --k_folds 5 ^
    --do_eval ^
    --do_dev_eval
exit /b 0

:evaluate_duo
echo Evaluating duo-classifier model...
python main.py ^
    --model_name_or_path bert-base-uncased ^
    --task semeval ^
    --test_file test.tsv ^
    --data_dir ./data ^
    --model_dir ./model/duo/relation ^
    --binary_model_dir ./model/duo/binary ^
    --eval_dir ./eval/duo_evaluation ^
    --max_seq_len 128 ^
    --duo_classifier ^
    --binary_threshold 0.5 ^
    --do_eval ^
    --do_dev_eval
exit /b 0

:evaluate_duo_cv
echo Evaluating duo-classifier cross-validation model...
python main.py ^
    --model_name_or_path bert-base-uncased ^
    --task semeval ^
    --test_file test.tsv ^
    --data_dir ./data ^
    --model_dir ./model/duo_cv/relation ^
    --binary_model_dir ./model/duo_cv/binary ^
    --eval_dir ./eval/duo_cv_evaluation ^
    --max_seq_len 128 ^
    --duo_classifier ^
    --binary_threshold 0.5 ^
    --k_folds 5 ^
    --do_eval ^
    --do_dev_eval
exit /b 0