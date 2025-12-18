#!/bin/bash
# HuggingFace Transformers Pipeline (No vLLM)
#
# Uses pure HuggingFace transformers for inference with multi-GPU parallel support.
# Each GPU loads a separate model and processes a shard of the data.
#
# Usage: ./run_pipeline_hf.sh [OPTIONS]
#
# Options:
#   --gpus GPUS       Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   --think           Enable thinking mode (default)
#   --no-think        Disable thinking mode (standard prompt)
#   --model MODEL     Model name (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   --max-samples N   Max samples per subset (for quick testing)
#
# Examples:
#   ./run_pipeline_hf.sh --no-think --gpus 0,1,2,3,4,5,6,7
#   ./run_pipeline_hf.sh --think --gpus 0,1 --max-samples 50

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPUS="0,1,2,3,4,5,6,7"
USE_THINK=true
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_SAMPLES=""

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

# Parse GPU count
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "HuggingFace Transformers Pipeline (No vLLM)"
echo "============================================================"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS ($NUM_GPUS GPUs)"
echo "============================================================"

echo ""
echo "========== Step 1: Collect Train Data ($NUM_GPUS GPUs parallel) =========="
python collect_data_hf.py \
    --model $MODEL \
    --split train \
    --parallel \
    --gpus $GPUS \
    --output-dir $DATA_DIR \
    $THINK_FLAG \
    $MAX_SAMPLES

echo ""
echo "========== Step 2: Collect Test Data ($NUM_GPUS GPUs parallel) =========="
python collect_data_hf.py \
    --model $MODEL \
    --split test \
    --parallel \
    --gpus $GPUS \
    --output-dir $DATA_DIR \
    $THINK_FLAG \
    $MAX_SAMPLES

echo ""
echo "========== Step 3: Data Statistics =========="
python compute_stats.py --data-dir $DATA_DIR --split train
python compute_stats.py --data-dir $DATA_DIR --split test

echo ""
echo "========== Step 4: Generate Results Table =========="
python generate_results_table.py --data-dir $DATA_DIR --format table

echo ""
echo "========== Done! =========="
echo "Results saved to: $DATA_DIR"
