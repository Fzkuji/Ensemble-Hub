#!/bin/bash
# Compare LoRA vs MLP classifier performance
# Usage: ./run_compare_classifier.sh [OPTIONS]
#
# Options:
#   --gpus GPUS       Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   --subset SUBSET   Specific subset to run (default: all subsets)
#   --data-dir DIR    Data directory
#   --skip-lora       Skip LoRA training (if already done)
#   --skip-mlp        Skip MLP training (if already done)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPUS="0,1,2,3,4,5,6,7"
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B"
SUBSET=""
SKIP_LORA=false
SKIP_MLP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --skip-lora)
            SKIP_LORA=true
            shift
            ;;
        --skip-mlp)
            SKIP_MLP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Parse GPU count
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Subsets to process
if [ -n "$SUBSET" ]; then
    SUBSETS=("$SUBSET")
else
    SUBSETS=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)
fi

echo "============================================================"
echo "Classifier Comparison: LoRA vs MLP (Frozen LLM)"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "Subsets: ${SUBSETS[*]}"
echo "============================================================"

# Train LoRA classifiers
if [ "$SKIP_LORA" = false ]; then
    echo ""
    echo "========== Training LoRA Classifiers =========="
    for subset in "${SUBSETS[@]}"; do
        echo ""
        echo ">>> LoRA: $subset"
        CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 train_lora_classifier.py \
            --ddp --subset $subset --data-dir $DATA_DIR --epochs 3
    done
else
    echo ""
    echo "========== Skipping LoRA (--skip-lora) =========="
fi

# Train MLP classifiers
if [ "$SKIP_MLP" = false ]; then
    echo ""
    echo "========== Training MLP Classifiers (Frozen LLM) =========="
    for subset in "${SUBSETS[@]}"; do
        echo ""
        echo ">>> MLP: $subset"
        CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 train_mlp_classifier.py \
            --ddp --subset $subset --data-dir $DATA_DIR --epochs 10
    done
else
    echo ""
    echo "========== Skipping MLP (--skip-mlp) =========="
fi

# Compare results
echo ""
echo "============================================================"
echo "                    COMPARISON RESULTS"
echo "============================================================"
printf "%-25s %12s %12s %10s\n" "Subset" "LoRA" "MLP" "Diff"
printf "%-25s %12s %12s %10s\n" "-------------------------" "------------" "------------" "----------"

for subset in "${SUBSETS[@]}"; do
    LORA_RESULT="$DATA_DIR/$subset/lora_model/results.json"
    MLP_RESULT="$DATA_DIR/$subset/mlp_model/results.json"

    LORA_ACC="-"
    MLP_ACC="-"
    DIFF="-"

    if [ -f "$LORA_RESULT" ]; then
        LORA_ACC=$(python -c "import json; print(f\"{json.load(open('$LORA_RESULT'))['best_cascade_acc']:.4f}\")" 2>/dev/null || echo "-")
    fi

    if [ -f "$MLP_RESULT" ]; then
        MLP_ACC=$(python -c "import json; print(f\"{json.load(open('$MLP_RESULT'))['best_cascade_acc']:.4f}\")" 2>/dev/null || echo "-")
    fi

    if [ "$LORA_ACC" != "-" ] && [ "$MLP_ACC" != "-" ]; then
        DIFF=$(python -c "print(f\"{float($LORA_ACC) - float($MLP_ACC):+.4f}\")" 2>/dev/null || echo "-")
    fi

    printf "%-25s %12s %12s %10s\n" "$subset" "$LORA_ACC" "$MLP_ACC" "$DIFF"
done

echo "============================================================"
echo "(Diff = LoRA - MLP, positive means LoRA is better)"
echo ""
echo "Done! Results saved in:"
echo "  LoRA: {subset}/lora_model/results.json"
echo "  MLP:  {subset}/mlp_model/results.json"
