#!/bin/bash
# Unified vLLM Pipeline for DeepSeek R1
# Usage: ./run_pipeline.sh [OPTIONS]
#
# Options:
#   --gpus GPUS       Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   --think           Enable thinking mode (default)
#   --no-think        Disable thinking mode (standard prompt)
#   --model MODEL     Model name (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   --subset SUBSET   Only run on specific subset (default: all subsets)
#   --lr LR           Learning rate for LoRA training (default: 1e-4)
#   --epochs EPOCHS   Number of epochs for LoRA training (default: 1)
#
# Examples:
#   ./run_pipeline.sh --think                          # Think mode, 8 GPUs
#   ./run_pipeline.sh --think --subset algebra         # Only algebra subset
#   ./run_pipeline.sh --no-think --gpus 1,2,3,4,5,6,7  # Standard mode, skip GPU 0

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPUS="0,1,2,3,4,5,6,7"
USE_THINK=true
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SUBSET=""
LR="1e-4"
EPOCHS="1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --think)
            USE_THINK=true
            shift
            ;;
        --no-think)
            USE_THINK=false
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Extract model name for directory
MODEL_NAME=$(echo $MODEL | sed 's|.*/||')

# Set data directory based on think mode
if [ "$USE_THINK" = true ]; then
    MODE="think"
    THINK_FLAG=""
else
    MODE="standard"
    THINK_FLAG="--no-think"
fi

DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_${MODE}_${MODEL_NAME}"

# Set subsets based on --subset argument
if [ -n "$SUBSET" ]; then
    SUBSETS=("$SUBSET")
else
    SUBSETS=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)
fi

# Token levels: -1 = mentor only, 0 = intern only, others = mentor hint + intern
TOKEN_LEVELS="-1,0,100,500,1000"

# Parse GPU count
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "DeepSeek R1 vLLM Pipeline"
echo "============================================================"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "============================================================"

# Helper function to check if data collection is complete for a subset/split
check_data_exists() {
    local subset=$1
    local split=$2
    local data_file="$DATA_DIR/$subset/$split/tokens0.json"
    if [ -f "$data_file" ]; then
        return 0  # exists
    else
        return 1  # not exists
    fi
}

echo ""
echo "========== Step 1: Collect Data (${NUM_GPUS} GPUs parallel) =========="

# Check if train data already exists
TRAIN_EXISTS=true
for subset in "${SUBSETS[@]}"; do
    if ! check_data_exists "$subset" "train"; then
        TRAIN_EXISTS=false
        break
    fi
done

if [ "$TRAIN_EXISTS" = true ]; then
    echo "Train data already exists, skipping collection..."
else
    echo "Collecting train data..."
    python collect_data_vllm_think.py --split train --parallel --gpus $GPUS "--token-levels=$TOKEN_LEVELS" $THINK_FLAG
fi

# Check if test data already exists
TEST_EXISTS=true
for subset in "${SUBSETS[@]}"; do
    if ! check_data_exists "$subset" "test"; then
        TEST_EXISTS=false
        break
    fi
done

if [ "$TEST_EXISTS" = true ]; then
    echo "Test data already exists, skipping collection..."
else
    echo "Collecting test data..."
    python collect_data_vllm_think.py --split test --parallel --gpus $GPUS "--token-levels=$TOKEN_LEVELS" $THINK_FLAG
fi

echo ""
echo "========== Step 2: Data Statistics =========="
python compute_stats.py --data-dir $DATA_DIR --split train
python compute_stats.py --data-dir $DATA_DIR --split test

echo ""
echo "========== Step 3: Train LoRA Classifiers =========="
for subset in "${SUBSETS[@]}"; do
    echo "Training: $subset (lr=$LR, epochs=$EPOCHS)"
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS train_lora_classifier.py --ddp --subset $subset --data-dir $DATA_DIR --lr $LR --epochs $EPOCHS
done

echo ""
echo "========== Step 4: Evaluate Cascade =========="
for subset in "${SUBSETS[@]}"; do
    echo "Evaluating: $subset"
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS eval_lora_cascade.py --subset $subset --data-dir $DATA_DIR
done

echo ""
echo "========== Step 5: Summarize Results =========="
python summarize_results.py --data-dir $DATA_DIR

echo ""
echo "========== Done! =========="
echo "Results saved to: $DATA_DIR"
