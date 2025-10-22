#!/bin/bash

# ============================================================================
# T5 Summarizer Complete Pipeline Script
# ============================================================================
# This script runs the entire pipeline from preprocessing to evaluation
# Usage: bash run_pipeline.sh [--skip-tuning] [--skip-training]
# ============================================================================

set -e  # Exit on any error

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
JUDG_PATH="data/train_judg.jsonl"
SUMM_PATH="data/train_ref_summ.jsonl"
DATA_DIR="data"
OUTPUT_DIR="outputs/t5_summarizer"
RESULTS_DIR="results"
HYPERPARAMS_FILE="hyperparams.json"
RANDOM_SEED=42
N_TRIALS=20

# Parse command line arguments
SKIP_TUNING=false
SKIP_TRAINING=false
for arg in "$@"; do
    case $arg in
        --skip-tuning)
            SKIP_TUNING=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
    esac
done

# ============================================================================
# STEP 0: Environment Setup
# ============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         T5 SUMMARIZER TRAINING PIPELINE                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if required files exist
if [ ! -f "$JUDG_PATH" ]; then
    echo -e "${RED}❌ Error: Judgment file not found at $JUDG_PATH${NC}"
    exit 1
fi

if [ ! -f "$SUMM_PATH" ]; then
    echo -e "${RED}❌ Error: Summary file not found at $SUMM_PATH${NC}"
    exit 1
fi

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

echo -e "${GREEN}✓ Environment setup complete${NC}"
echo ""

# ============================================================================
# STEP 1: Data Preprocessing (80/10/10 Split)
# ============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 1: Data Preprocessing${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python preprocess.py \
    --judg_path "$JUDG_PATH" \
    --summ_path "$SUMM_PATH" \
    --output_dir "$DATA_DIR" \
    --random_seed "$RANDOM_SEED"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data preprocessing complete${NC}"
    echo -e "  - Training set: ${DATA_DIR}/train_processed.jsonl"
    echo -e "  - Validation set: ${DATA_DIR}/val_processed.jsonl"
    echo -e "  - Test set: ${DATA_DIR}/test_processed.jsonl"
else
    echo -e "${RED}❌ Preprocessing failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# STEP 2: Hyperparameter Tuning with Optuna (Optional)
# ============================================================================
if [ "$SKIP_TUNING" = false ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}STEP 2: Hyperparameter Tuning${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}Running Optuna optimization with $N_TRIALS trials...${NC}"
    echo ""

    python tune.py \
        --train_file "${DATA_DIR}/train_processed.jsonl" \
        --val_file "${DATA_DIR}/val_processed.jsonl" \
        --n_trials "$N_TRIALS" \
        --output_hyperparams "$HYPERPARAMS_FILE"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Hyperparameter tuning complete${NC}"
        echo -e "  - Best hyperparameters saved to: $HYPERPARAMS_FILE"
    else
        echo -e "${RED}❌ Hyperparameter tuning failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⊘ Skipping hyperparameter tuning (using existing hyperparams.json)${NC}"
fi
echo ""

# ============================================================================
# STEP 3: Model Training
# ============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}STEP 3: Model Training${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    python train.py \
        --train_file "${DATA_DIR}/train_processed.jsonl" \
        --val_file "${DATA_DIR}/val_processed.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --hyperparams "$HYPERPARAMS_FILE"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model training complete${NC}"
        echo -e "  - Model saved to: $OUTPUT_DIR"
    else
        echo -e "${RED}❌ Model training failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⊘ Skipping model training (using existing model)${NC}"
fi
echo ""

# ============================================================================
# STEP 4: Generate Predictions on Test Set
# ============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 4: Generate Test Set Predictions${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python run_evaluation.py \
    --model_path "$OUTPUT_DIR" \
    --input_file "${DATA_DIR}/test_processed.jsonl" \
    --output_file "${RESULTS_DIR}/test_predictions.jsonl" \
    --batch_size 8

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test predictions generated${NC}"
    echo -e "  - Predictions saved to: ${RESULTS_DIR}/test_predictions.jsonl"
else
    echo -e "${RED}❌ Prediction generation failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# STEP 5: Evaluate Model (ROUGE-2, ROUGE-L, BLEU)
# ============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 5: Evaluation (ROUGE-2, ROUGE-L, BLEU)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python validate.py \
    --model_path "$OUTPUT_DIR" \
    --validation_file "${DATA_DIR}/test_processed.jsonl" \
    --save_results \
    --results_file "${RESULTS_DIR}/evaluation_results.json"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluation complete${NC}"
    echo -e "  - Results saved to: ${RESULTS_DIR}/evaluation_results.json"
else
    echo -e "${RED}❌ Evaluation failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# STEP 6: Summary
# ============================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    PIPELINE COMPLETE                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📁 Output Files:${NC}"
echo -e "  • Processed data: ${DATA_DIR}/"
echo -e "  • Trained model: ${OUTPUT_DIR}/"
echo -e "  • Test predictions: ${RESULTS_DIR}/test_predictions.jsonl"
echo -e "  • Evaluation metrics: ${RESULTS_DIR}/evaluation_results.json"
echo ""
echo -e "${BLUE}📊 View your evaluation metrics:${NC}"
echo -e "  cat ${RESULTS_DIR}/evaluation_results.json | python -m json.tool"
echo ""
