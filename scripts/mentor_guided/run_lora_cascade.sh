#!/bin/bash
#
# LoRA Training and Cascade Evaluation Script
# Trains on existing data in hendrycks_math_split and evaluates cascade performance
#
# Usage:
#   ./run_lora_cascade.sh [--gpus 0,1,2,3] [--epochs 3] [--batch-size 2]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPUS="0,1,2,3,4,5,6,7"
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR="5e-5"
MAX_LENGTH=1024

SUBSETS=(
    "algebra"
    "counting_and_probability"
    "geometry"
    "intermediate_algebra"
    "number_theory"
    "prealgebra"
    "precalculus"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --subset)
            # Allow running on a single subset
            SUBSETS=("$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "LoRA Training and Cascade Evaluation"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "Model: $MODEL_PATH"
echo "GPUs: $GPUS ($NUM_GPUS GPUs)"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Subsets: ${SUBSETS[*]}"
echo "============================================================"
echo ""

# Step 1: Train LoRA on each subset
echo "========== Step 1: Train LoRA Classifiers =========="
for subset in "${SUBSETS[@]}"; do
    echo ""
    echo ">>> Training: $subset"

    # Check if already trained
    if [ -f "$DATA_DIR/$subset/lora_model/best_model.pt" ]; then
        echo "    Model already exists, skipping training..."
        continue
    fi

    if [ "$NUM_GPUS" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS \
            train_lora_classifier.py \
            --data-dir "$DATA_DIR" \
            --subset "$subset" \
            --model-path "$MODEL_PATH" \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --grad-accum $GRAD_ACCUM \
            --lr $LR \
            --max-length $MAX_LENGTH \
            --ddp
    else
        CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python train_lora_classifier.py \
            --data-dir "$DATA_DIR" \
            --subset "$subset" \
            --model-path "$MODEL_PATH" \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --grad-accum $GRAD_ACCUM \
            --lr $LR \
            --max-length $MAX_LENGTH \
            --use-4bit
    fi
done

echo ""
echo "========== Step 2: Evaluate Cascade =========="
for subset in "${SUBSETS[@]}"; do
    echo ""
    echo ">>> Evaluating: $subset"

    MODEL_DIR="$DATA_DIR/$subset/lora_model"

    if [ ! -f "$MODEL_DIR/best_model.pt" ]; then
        echo "    Model not found, skipping evaluation..."
        continue
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python eval_lora_cascade.py \
        --data-dir "$DATA_DIR" \
        --subset "$subset" \
        --model-dir "$MODEL_DIR" \
        --base-model "$MODEL_PATH" \
        --max-length $MAX_LENGTH \
        --use-4bit \
        --search-thresholds
done

echo ""
echo "========== Step 3: Generate Results Table =========="
python generate_results_table.py --data-dir "$DATA_DIR" --format all

echo ""
echo "============================================================"
echo "Done!"
echo "Results saved to: $DATA_DIR"
echo "============================================================"
