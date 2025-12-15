#!/bin/bash
# Run transformer classifier experiment
# Usage: bash run_transformer_exp.sh [data_dir]

set -e

DATA_DIR="${1:-/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_all_DeepSeek-R1-Distill-Qwen-32B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Step 1: Collect hidden state sequences"
echo "=========================================="
echo "Data dir: $DATA_DIR"

# Collect hidden state sequences (full sequence, not pooled)
python "$SCRIPT_DIR/collect_hidden_seq.py" \
    --data-dir "$DATA_DIR" \
    --model-path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max-mentor-tokens 512 \
    --hidden-layers -1

echo ""
echo "=========================================="
echo "Step 2: Train transformer classifier"
echo "=========================================="

HIDDEN_SEQ_DIR="$DATA_DIR/hidden_seq"

python "$SCRIPT_DIR/train_transformer_classifier.py" \
    --data-dir "$HIDDEN_SEQ_DIR" \
    --d-model 256 \
    --nhead 4 \
    --num-layers 2 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-file "$HIDDEN_SEQ_DIR/results.json"

echo ""
echo "=========================================="
echo "Done! Results saved to: $HIDDEN_SEQ_DIR/results.json"
echo "=========================================="
