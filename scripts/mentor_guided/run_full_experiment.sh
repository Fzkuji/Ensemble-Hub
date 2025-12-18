#!/bin/bash
#
# Full ACT-E Experiment Pipeline
# 1. Train LoRA classifier on each subset (or all)
# 2. Evaluate on test set
# 3. Generate summary table
#
# Usage:
#   ./run_full_experiment.sh [--gpus 0,1,2,3] [--mode all|per-subset] [--data-dir /path/to/data]
#

set -e

# Default values
GPUS="0,1,2,3,4,5,6,7"
MODE="per-subset"  # "all" trains one model on all data, "per-subset" trains per subset
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR="2e-4"
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
        --mode)
            MODE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "ACT-E Full Experiment Pipeline"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "Model: $MODEL_PATH"
echo "GPUs: $GPUS ($NUM_GPUS GPUs)"
echo "Mode: $MODE"
echo "Epochs: $EPOCHS"
echo "============================================================"
echo ""

# Results file
RESULTS_FILE="$DATA_DIR/experiment_results.json"
echo "{}" > "$RESULTS_FILE"

train_subset() {
    local subset=$1
    echo ""
    echo "============================================================"
    echo "[TRAIN] Training LoRA classifier for: $subset"
    echo "============================================================"

    if [ "$NUM_GPUS" -gt 1 ]; then
        # Multi-GPU training with torchrun
        CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS \
            "$SCRIPT_DIR/train_lora_classifier.py" \
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
        # Single GPU training
        CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python "$SCRIPT_DIR/train_lora_classifier.py" \
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
}

eval_subset() {
    local subset=$1
    local model_dir=$2
    echo ""
    echo "============================================================"
    echo "[EVAL] Evaluating on: $subset"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python "$SCRIPT_DIR/eval_lora_cascade.py" \
        --data-dir "$DATA_DIR" \
        --subset "$subset" \
        --model-dir "$model_dir" \
        --base-model "$MODEL_PATH" \
        --max-length $MAX_LENGTH \
        --use-4bit \
        --search-thresholds
}

if [ "$MODE" == "all" ]; then
    # Train one model on all data
    echo "[MODE] Training single model on ALL subsets combined"
    train_subset "all"

    # Evaluate on each subset
    MODEL_DIR="$DATA_DIR/all/lora_model"
    for subset in "${SUBSETS[@]}"; do
        eval_subset "$subset" "$MODEL_DIR"
    done

else
    # Train per subset
    echo "[MODE] Training separate model for EACH subset"
    for subset in "${SUBSETS[@]}"; do
        train_subset "$subset"
        MODEL_DIR="$DATA_DIR/$subset/lora_model"
        eval_subset "$subset" "$MODEL_DIR"
    done
fi

echo ""
echo "============================================================"
echo "[SUMMARY] Generating final results table..."
echo "============================================================"

# Generate summary table
python3 << 'PYTHON_SCRIPT'
import json
import os
from glob import glob

DATA_DIR = os.environ.get('DATA_DIR', '/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split')

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

TOKEN_LEVELS = [0, 100, 500, 1000]

results = {}

# Collect results from each subset
for subset in SUBSETS:
    subset_dir = os.path.join(DATA_DIR, subset)
    results_file = os.path.join(subset_dir, "lora_model", "eval_results.json")

    if os.path.exists(results_file):
        with open(results_file) as f:
            results[subset] = json.load(f)
    else:
        # Try to compute from test data
        test_dir = os.path.join(subset_dir, "test")
        if os.path.exists(test_dir):
            subset_results = {'baseline': {}, 'n_samples': 0}
            for tokens in TOKEN_LEVELS:
                filepath = os.path.join(test_dir, f"tokens{tokens}.json")
                if os.path.exists(filepath):
                    with open(filepath) as f:
                        data = json.load(f)
                    n = len(data)
                    correct = sum(1 for d in data if d.get('is_correct', False))
                    subset_results['baseline'][str(tokens)] = correct / n if n > 0 else 0
                    subset_results['n_samples'] = n
            if subset_results['baseline']:
                results[subset] = subset_results

# Print table
print("\n" + "="*120)
print("EXPERIMENT RESULTS SUMMARY")
print("="*120)
print(f"{'Subset':<25} {'N':<8} {'T=0':<10} {'T=100':<10} {'T=500':<10} {'T=1000':<10} {'Oracle':<10} {'Cascade':<10}")
print("-"*120)

total_samples = 0
total_correct = {str(t): 0 for t in TOKEN_LEVELS}
total_oracle = 0
total_cascade = 0

for subset in SUBSETS:
    if subset not in results:
        print(f"{subset:<25} {'N/A':<8}")
        continue

    r = results[subset]
    n = r.get('n_samples', 0)
    total_samples += n

    baseline = r.get('baseline', {})
    oracle = r.get('oracle', 0)
    cascade = r.get('cascade_accuracy', r.get('best_accuracy', 0))

    row = f"{subset:<25} {n:<8}"
    for t in TOKEN_LEVELS:
        acc = baseline.get(str(t), 0)
        total_correct[str(t)] += int(acc * n)
        row += f"{acc:.4f}    "

    row += f"{oracle:.4f}    " if oracle else "N/A       "
    row += f"{cascade:.4f}    " if cascade else "N/A       "

    if oracle:
        total_oracle += int(oracle * n)
    if cascade:
        total_cascade += int(cascade * n)

    print(row)

print("-"*120)

# Print totals
if total_samples > 0:
    row = f"{'TOTAL':<25} {total_samples:<8}"
    for t in TOKEN_LEVELS:
        acc = total_correct[str(t)] / total_samples
        row += f"{acc:.4f}    "
    row += f"{total_oracle/total_samples:.4f}    " if total_oracle else "N/A       "
    row += f"{total_cascade/total_samples:.4f}    " if total_cascade else "N/A       "
    print(row)

print("="*120)
print("\nLegend:")
print("  T=0/100/500/1000: Baseline accuracy with N mentor tokens")
print("  Oracle: Best possible accuracy (if we knew which stage to stop)")
print("  Cascade: Accuracy using trained classifier to decide when to stop")
print("")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Experiment complete!"
echo "============================================================"
