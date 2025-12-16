#!/bin/bash
set -e
cd /home/fzkuji/PycharmProjects/Ensemble-Hub/scripts/mentor_guided

SUBSETS="algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus"

echo "========== Step 1: Collect Data =========="
python collect_progressive_data.py --dataset hendrycks_math --split train --parallel --gpus 0,1,2,3,4,5,6,7
python collect_progressive_data.py --dataset hendrycks_math --split test --parallel --gpus 0,1,2,3,4,5,6,7

echo "========== Step 2: Data Statistics =========="
python compute_stats.py --split train
python compute_stats.py --split test

echo "========== Step 3: Train LoRA Classifiers =========="
for subset in $SUBSETS; do
    echo "Training: $subset"
    torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset $subset
done

echo "========== Step 4: Evaluate Cascade =========="
for subset in $SUBSETS; do
    echo "Evaluating: $subset"
    torchrun --nproc_per_node=8 eval_lora_cascade.py --subset $subset
done

echo "========== Step 5: Summarize Results =========="
python summarize_results.py

echo "========== Done! =========="
