#!/bin/bash
# HuggingFace Transformers Pipeline (No vLLM)
#
# Uses pure HuggingFace transformers for inference.
# This may give different results than vLLM due to different sampling implementations.
#
# Usage: ./run_pipeline_hf.sh [OPTIONS]
#
# Options:
#   --gpu GPU         GPU ID (default: 0)
#   --think           Enable thinking mode (default)
#   --no-think        Disable thinking mode (standard prompt)
#   --model MODEL     Model name (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   --max-samples N   Max samples per subset (for quick testing)
#
# Examples:
#   ./run_pipeline_hf.sh --no-think --gpu 0
#   ./run_pipeline_hf.sh --think --gpu 1 --max-samples 50

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPU=0
USE_THINK=true
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_SAMPLES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
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
        --max-samples)
            MAX_SAMPLES="--max-samples $2"
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

# Set mode flag
if [ "$USE_THINK" = true ]; then
    MODE="think"
    THINK_FLAG=""
else
    MODE="standard"
    THINK_FLAG="--no-think"
fi

DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_${MODE}_hf_${MODEL_NAME}"
SUBSETS=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)

echo "============================================================"
echo "HuggingFace Transformers Pipeline (No vLLM)"
echo "============================================================"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Data dir: $DATA_DIR"
echo "GPU: $GPU"
echo "============================================================"

echo ""
echo "========== Step 1: Collect Data =========="

# Collect train data
echo "Collecting train data..."
CUDA_VISIBLE_DEVICES=$GPU python collect_data_hf.py \
    --model $MODEL \
    --split train \
    --gpu 0 \
    --output-dir $DATA_DIR \
    $THINK_FLAG \
    $MAX_SAMPLES

# Collect test data
echo "Collecting test data..."
CUDA_VISIBLE_DEVICES=$GPU python collect_data_hf.py \
    --model $MODEL \
    --split test \
    --gpu 0 \
    --output-dir $DATA_DIR \
    $THINK_FLAG \
    $MAX_SAMPLES

echo ""
echo "========== Step 2: Data Statistics =========="
python compute_stats.py --data-dir $DATA_DIR --split train
python compute_stats.py --data-dir $DATA_DIR --split test

echo ""
echo "========== Step 3: Generate Results Table =========="
python generate_results_table.py --data-dir $DATA_DIR --format table

echo ""
echo "========== Done! =========="
echo "Results saved to: $DATA_DIR"
