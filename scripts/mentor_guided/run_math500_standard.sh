#!/bin/bash
# MATH-500 Experiment Pipeline (Standard, No Think) - vLLM Version
# Usage: ./run_math500_standard.sh [GPUS] [EXP_NAME]
#   GPUS: Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   EXP_NAME: Custom experiment name (default: model name)
#
# Examples:
#   ./run_math500_standard.sh                    # Default: 8 GPUs
#   ./run_math500_standard.sh 1,2,3,4,5,6,7      # Skip GPU 0

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME=$(echo $MODEL | sed 's|.*/||')
GPUS="${1:-0,1,2,3,4,5,6,7}"
EXP_NAME="${2:-$MODEL_NAME}"
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/math500_standard_${EXP_NAME}"
VAL_RATIO=0.4

echo "============================================================"
echo "MATH-500 Experiment Pipeline (Standard, No Think) - vLLM"
echo "============================================================"
echo "Model: $MODEL"
echo "Exp name: $EXP_NAME"
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS"
echo "Val ratio: $VAL_RATIO"
echo "============================================================"

# Parse GPU count
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo ""
echo "========== Step 1: Collect Training Data (hendrycks_math_all train) =========="
python collect_data_vllm_think.py \
    --dataset hendrycks_math_all \
    --split train \
    --gpus $GPUS \
    --no-think \
    --output-dir $DATA_DIR

echo ""
echo "========== Step 2: Collect Test Data (MATH-500) =========="
python collect_data_vllm_think.py \
    --dataset math500 \
    --gpus $GPUS \
    --no-think \
    --output-dir $DATA_DIR

echo ""
echo "========== Step 3: Train LoRA Classifier (all subsets merged, 6:4 split) =========="
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS train_lora_classifier.py \
    --ddp \
    --subset all \
    --data-dir $DATA_DIR \
    --val-ratio $VAL_RATIO \
    --output-dir $DATA_DIR/all/lora_model

echo ""
echo "========== Step 4: Evaluate on MATH-500 =========="
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS eval_lora_cascade.py \
    --subset math500 \
    --data-dir $DATA_DIR \
    --model-dir $DATA_DIR/all/lora_model \
    --search-thresholds

echo ""
echo "========== Step 5: Generate Results Table =========="
python summarize_results.py --data-dir $DATA_DIR --subset math500

echo ""
echo "========== Done! =========="
echo "Results saved to: $DATA_DIR"
