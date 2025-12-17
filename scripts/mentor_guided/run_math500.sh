#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME=$(echo $MODEL | sed 's|.*/||')
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/math500_exp_think_${MODEL_NAME}"
GPUS="${1:-0,1,2,3,4,5,6,7}"
VAL_RATIO=0.4  # 6:4 train/val split

echo "============================================================"
echo "MATH-500 Experiment Pipeline"
echo "============================================================"
echo "Model: $MODEL"
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS"
echo "Val ratio: $VAL_RATIO (train:val = $((100-$(echo "$VAL_RATIO*100" | bc | cut -d. -f1))):$(echo "$VAL_RATIO*100" | bc | cut -d. -f1))"
echo "============================================================"

# Parse GPU count
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo ""
echo "========== Step 1: Collect Training Data (hendrycks_math_all train) =========="
python collect_data_vllm_think.py \
    --dataset hendrycks_math_all \
    --split train \
    --parallel \
    --gpus $GPUS \
    --output-dir $DATA_DIR

echo ""
echo "========== Step 2: Collect Test Data (MATH-500) =========="
python collect_data_vllm_think.py \
    --dataset math500 \
    --parallel \
    --gpus $GPUS \
    --output-dir $DATA_DIR

echo ""
echo "========== Step 3: Train LoRA Classifier (all subsets merged, 6:4 split) =========="
torchrun --nproc_per_node=$NUM_GPUS train_lora_classifier.py \
    --ddp \
    --subset all \
    --data-dir $DATA_DIR \
    --val-ratio $VAL_RATIO \
    --output-dir $DATA_DIR/all/lora_model

echo ""
echo "========== Step 4: Evaluate on MATH-500 =========="
torchrun --nproc_per_node=$NUM_GPUS eval_lora_cascade.py \
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
