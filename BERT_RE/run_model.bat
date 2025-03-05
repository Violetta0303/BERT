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

if "%1"=="cv" (
    call :train_cv
) else if "%1"=="eval" (
    call :evaluate_model
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
    --num_train_epochs 3 ^
    --train_batch_size 16 ^
    --learning_rate 2e-5 ^
    --save_epochs 1 ^
    --k_folds 5 ^
    --do_train ^
    --do_eval
exit /b 0

:evaluate_model
echo Evaluating model...
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