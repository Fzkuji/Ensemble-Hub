#!/bin/bash
# Standard (No Think) vLLM Pipeline - By Subset
# Usage: ./run_standard_vllm.sh [GPUS]
#   GPUS: Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#
# Examples:
#   ./run_standard_vllm.sh                    # Default: 8 GPUs
#   ./run_standard_vllm.sh 1,2,3,4,5,6,7      # Skip GPU 0

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPUS="${1:-0,1,2,3,4,5,6,7}"
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_standard_DeepSeek-R1-Distill-Qwen-7B"
SUBSETS=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)

# Parse GPU count
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "Standard (No Think) vLLM Pipeline - By Subset"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "============================================================"

echo ""
echo "========== Step 1: Collect Data (${NUM_GPUS} GPUs parallel) =========="
python collect_data_vllm_think.py --split train --parallel --gpus $GPUS --no-think
python collect_data_vllm_think.py --split test --parallel --gpus $GPUS --no-think

echo ""
echo "========== Step 2: Data Statistics =========="
python compute_stats.py --data-dir $DATA_DIR --split train
python compute_stats.py --data-dir $DATA_DIR --split test

echo ""
echo "========== Step 3: Train LoRA Classifiers =========="
for subset in "${SUBSETS[@]}"; do
    echo "Training: $subset"
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS train_lora_classifier.py --ddp --subset $subset --data-dir $DATA_DIR
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
