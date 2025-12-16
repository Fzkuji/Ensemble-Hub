#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B"
SUBSETS="algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus"

echo "========== Step 1: Collect Data =========="
python collect_data_vllm_think.py --split train --gpu 0
python collect_data_vllm_think.py --split test --gpu 0

echo "========== Step 2: Data Statistics =========="
python compute_stats.py --data-dir $DATA_DIR --split train
python compute_stats.py --data-dir $DATA_DIR --split test

echo "========== Step 3: Train LoRA Classifiers =========="
for subset in $SUBSETS; do
    echo "Training: $subset"
    torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset $subset --data-dir $DATA_DIR
done

echo "========== Step 4: Evaluate Cascade =========="
for subset in $SUBSETS; do
    echo "Evaluating: $subset"
    torchrun --nproc_per_node=8 eval_lora_cascade.py --subset $subset --data-dir $DATA_DIR
done

echo "========== Step 5: Summarize Results =========="
python summarize_results.py --data-dir $DATA_DIR

echo "========== Done! =========="
