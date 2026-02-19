#!/bin/bash
#
# End-to-end MBPP Pipeline:
#   1. Collect MBPP train + test data with Tandem
#   2. Re-evaluate correctness via code execution
#   3. Train classifier on MBPP train
#   4. Evaluate on MBPP test (in-domain) + HumanEval (cross-domain)
#
# Usage:
#   bash run_mbpp_pipeline.sh [--gpus 0,1,2,3]
#
set -e

# ── Configuration ────────────────────────────────────────────────────────
MENTOR_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
INTERN_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # For feature extraction
TOKEN_LEVELS="0,100,500,1000"
TEMPERATURE=0.7
MAX_MODEL_LEN=4096
BATCH_SIZE=16
EXEC_TIMEOUT=15

# Paths (auto-detected from output dir naming convention)
BASE_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected"
MBPP_COLLECTED="${BASE_DIR}/mbpp_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B"
HUMANEVAL_DIR="${BASE_DIR}/humaneval_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B/humaneval"
CACHE_DIR="${BASE_DIR}/code_cascade_cache"
GPUS="0,1,2,3,4,5,6,7"

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --humaneval-dir) HUMANEVAL_DIR="$2"; shift 2 ;;
        --skip-collect) SKIP_COLLECT=1; shift ;;
        --skip-reeval) SKIP_REEVAL=1; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRST_GPU=$(echo "$GPUS" | cut -d',' -f1)

echo "============================================================"
echo "MBPP End-to-End Pipeline"
echo "============================================================"
echo "Mentor:       $MENTOR_MODEL"
echo "Intern:       $INTERN_MODEL"
echo "GPUs:         $GPUS"
echo "MBPP output:  $MBPP_COLLECTED"
echo "HumanEval:    $HUMANEVAL_DIR"
echo "Cache:        $CACHE_DIR"
echo "============================================================"
echo ""

# ── Step 1: Collect MBPP data ────────────────────────────────────────────
if [ -z "$SKIP_COLLECT" ]; then
    for SPLIT in train test; do
        echo ""
        echo "============================================================"
        echo "[COLLECT] MBPP ${SPLIT}"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES=$GPUS python "$SCRIPT_DIR/collect_data_vllm_think.py" \
            --dataset mbpp \
            --split "$SPLIT" \
            --mentor-model "$MENTOR_MODEL" \
            --intern-model "$INTERN_MODEL" \
            --token-levels="$TOKEN_LEVELS" \
            --insight-mode prompt \
            --temperature "$TEMPERATURE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$MBPP_COLLECTED"
    done
else
    echo "[SKIP] Collection (--skip-collect)"
fi

# ── Step 2: Re-evaluate correctness ─────────────────────────────────────
if [ -z "$SKIP_REEVAL" ]; then
    for SPLIT in train test; do
        echo ""
        echo "============================================================"
        echo "[REEVAL] MBPP ${SPLIT}"
        echo "============================================================"

        python "$SCRIPT_DIR/collect_mbpp_vllm.py" --reeval \
            --data-dir "$MBPP_COLLECTED/mbpp/$SPLIT" \
            --exec-timeout "$EXEC_TIMEOUT"
    done
else
    echo "[SKIP] Re-evaluation (--skip-reeval)"
fi

# ── Step 3+4: Train on MBPP train, eval on MBPP test + HumanEval ────────
echo ""
echo "============================================================"
echo "[TRAIN+EVAL] Classifier: MBPP train → MBPP test + HumanEval"
echo "============================================================"

mkdir -p "$CACHE_DIR"

CUDA_VISIBLE_DEVICES=$FIRST_GPU python "$SCRIPT_DIR/eval_code_cascade.py" \
    --mbpp-train-dir "$MBPP_COLLECTED/mbpp/train" \
    --mbpp-test-dir "$MBPP_COLLECTED/mbpp/test" \
    --humaneval-dir "$HUMANEVAL_DIR" \
    --hf-model "$HF_MODEL" \
    --device cuda:0 \
    --cache-dir "$CACHE_DIR" \
    --output "$CACHE_DIR/code_cascade_results.json"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results: $CACHE_DIR/code_cascade_results.json"
echo "============================================================"
