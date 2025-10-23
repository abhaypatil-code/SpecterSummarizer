@echo off
REM ============================================================================
REM T5 Summarizer Complete Pipeline Script for Windows
REM ============================================================================
REM This script runs the entire pipeline from preprocessing to evaluation
REM Usage: run_pipeline.bat [--skip-tuning] [--skip-training]
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "JUDG_PATH=data\train_judg.jsonl"
set "SUMM_PATH=data\train_ref_summ.jsonl"
set "DATA_DIR=data"
set "OUTPUT_DIR=outputs\t5_summarizer"
set "RESULTS_DIR=results"
set "HYPERPARAMS_FILE=hyperparams.json"
set "RANDOM_SEED=42"
set "N_TRIALS=20"

REM Parse command line arguments
set "SKIP_TUNING=false"
set "SKIP_TRAINING=false"

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--skip-tuning" (
    set "SKIP_TUNING=true"
    shift
    goto parse_args
)
if "%~1"=="--skip-training" (
    set "SKIP_TRAINING=true"
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse

echo.
echo ================================================================================
echo                  T5 LEGAL JUDGMENT SUMMARIZATION PIPELINE
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    exit /b 1
)

REM Check if CUDA is available
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CUDA is not available. This pipeline requires GPU.
    exit /b 1
)

echo [INFO] GPU detected
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
echo.

REM Create necessary directories
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

REM ============================================================================
REM STEP 1: Preprocessing and Data Splitting
REM ============================================================================
echo.
echo ================================================================================
echo STEP 1: PREPROCESSING AND SPLITTING DATA (80/10/10)
echo ================================================================================
echo.

if not exist "%JUDG_PATH%" (
    echo [ERROR] Judgment file not found: %JUDG_PATH%
    exit /b 1
)

if not exist "%SUMM_PATH%" (
    echo [ERROR] Summary file not found: %SUMM_PATH%
    exit /b 1
)

python preprocess.py ^
    --judg_path "%JUDG_PATH%" ^
    --summ_path "%SUMM_PATH%" ^
    --output_dir "%DATA_DIR%" ^
    --random_seed %RANDOM_SEED%

if errorlevel 1 (
    echo [ERROR] Preprocessing failed
    exit /b 1
)

echo [SUCCESS] Preprocessing completed
echo.

REM ============================================================================
REM STEP 2: Hyperparameter Tuning (Optional)
REM ============================================================================
if "%SKIP_TUNING%"=="true" (
    echo [INFO] Skipping hyperparameter tuning
    goto skip_tuning
)

echo.
echo ================================================================================
echo STEP 2: HYPERPARAMETER TUNING
echo ================================================================================
echo.

python tune.py ^
    --train_file "%DATA_DIR%\train_processed.jsonl" ^
    --val_file "%DATA_DIR%\val_processed.jsonl" ^
    --output_dir "outputs\tuning" ^
    --n_trials %N_TRIALS% ^
    --random_seed %RANDOM_SEED%

if errorlevel 1 (
    echo [ERROR] Hyperparameter tuning failed
    exit /b 1
)

echo [SUCCESS] Hyperparameter tuning completed
echo.

REM Copy best hyperparameters to root
if exist "outputs\tuning\hyperparams.json" (
    copy /Y "outputs\tuning\hyperparams.json" "%HYPERPARAMS_FILE%" >nul
    echo [INFO] Best hyperparameters saved to %HYPERPARAMS_FILE%
)

:skip_tuning

REM ============================================================================
REM STEP 3: Model Training
REM ============================================================================
if "%SKIP_TRAINING%"=="true" (
    echo [INFO] Skipping model training
    goto skip_training
)

echo.
echo ================================================================================
echo STEP 3: MODEL TRAINING
echo ================================================================================
echo.

REM Load hyperparameters if available
if exist "%HYPERPARAMS_FILE%" (
    echo [INFO] Loading hyperparameters from %HYPERPARAMS_FILE%
    
    REM Extract hyperparameters using Python
    for /f "delims=" %%i in ('python -c "import json; hp=json.load(open('%HYPERPARAMS_FILE%')); print(hp.get('learning_rate', 3e-5))"') do set "LR=%%i"
    for /f "delims=" %%i in ('python -c "import json; hp=json.load(open('%HYPERPARAMS_FILE%')); print(hp.get('batch_size', 4))"') do set "BS=%%i"
    for /f "delims=" %%i in ('python -c "import json; hp=json.load(open('%HYPERPARAMS_FILE%')); print(hp.get('num_epochs', 3))"') do set "EPOCHS=%%i"
    for /f "delims=" %%i in ('python -c "import json; hp=json.load(open('%HYPERPARAMS_FILE%')); print(hp.get('weight_decay', 0.01))"') do set "WD=%%i"
    for /f "delims=" %%i in ('python -c "import json; hp=json.load(open('%HYPERPARAMS_FILE%')); print(hp.get('warmup_steps', 500))"') do set "WARMUP=%%i"
    for /f "delims=" %%i in ('python -c "import json; hp=json.load(open('%HYPERPARAMS_FILE%')); print(hp.get('gradient_accumulation_steps', 4))"') do set "GRAD_ACC=%%i"
) else (
    echo [INFO] Using default hyperparameters
    set "LR=3e-5"
    set "BS=4"
    set "EPOCHS=3"
    set "WD=0.01"
    set "WARMUP=500"
    set "GRAD_ACC=4"
)

python train.py ^
    --train_file "%DATA_DIR%\train_processed.jsonl" ^
    --val_file "%DATA_DIR%\val_processed.jsonl" ^
    --output_dir "%OUTPUT_DIR%" ^
    --learning_rate %LR% ^
    --batch_size %BS% ^
    --num_epochs %EPOCHS% ^
    --weight_decay %WD% ^
    --warmup_steps %WARMUP% ^
    --gradient_accumulation_steps %GRAD_ACC% ^
    --fp16

if errorlevel 1 (
    echo [ERROR] Model training failed
    exit /b 1
)

echo [SUCCESS] Model training completed
echo.

:skip_training

REM ============================================================================
REM STEP 4: Validation
REM ============================================================================
echo.
echo ================================================================================
echo STEP 4: MODEL VALIDATION
echo ================================================================================
echo.

if not exist "%OUTPUT_DIR%" (
    echo [ERROR] Trained model not found at %OUTPUT_DIR%
    exit /b 1
)

python validate.py ^
    --model_path "%OUTPUT_DIR%" ^
    --validation_file "%DATA_DIR%\val_processed.jsonl" ^
    --batch_size 4 ^
    --results_file "%RESULTS_DIR%\validation_results.json"

if errorlevel 1 (
    echo [ERROR] Validation failed
    exit /b 1
)

echo [SUCCESS] Validation completed
echo.

REM ============================================================================
REM STEP 5: Test Set Evaluation
REM ============================================================================
echo.
echo ================================================================================
echo STEP 5: TEST SET EVALUATION
echo ================================================================================
echo.

python run_evaluation.py ^
    --model_path "%OUTPUT_DIR%" ^
    --input_file "%DATA_DIR%\test_processed.jsonl" ^
    --output_file "%RESULTS_DIR%\test_predictions.jsonl" ^
    --batch_size 4

if errorlevel 1 (
    echo [ERROR] Test evaluation failed
    exit /b 1
)

echo [SUCCESS] Test evaluation completed
echo.

REM ============================================================================
REM PIPELINE COMPLETE
REM ============================================================================
echo.
echo ================================================================================
echo                    PIPELINE COMPLETED SUCCESSFULLY
echo ================================================================================
echo.
echo Results saved in: %RESULTS_DIR%
echo Model saved in: %OUTPUT_DIR%
echo.
echo Next steps:
echo   1. Check validation results: %RESULTS_DIR%\validation_results.json
echo   2. Review predictions: %RESULTS_DIR%\test_predictions.jsonl
echo   3. Use the model for inference with run_evaluation.py
echo.

endlocal
exit /b 0
