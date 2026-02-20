#!/bin/bash
# Latency measurement for Tandem (Reviewer yUBt W5)
#
# Usage:
#   ./run_latency.sh                         # Full (mentor + intern + tandem)
#   ./run_latency.sh --skip-mentor           # Skip 32B, only intern + tandem (use cached hints)
#   ./run_latency.sh --skip-generation       # Skip all generation, only feature extraction + cascade
#   ./run_latency.sh --mentor-gpus 0,1 --intern-gpus 2,3
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- defaults ----
MENTOR_GPUS="0,1,2,3"
INTERN_GPUS="4,5,6,7"
N_SAMPLES=100
SKIP_MENTOR=""
SKIP_GEN=""
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B"
CLASSIFIER_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B/all/ppl_model"
OUTPUT="${DATA_DIR}/latency_results.json"

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --mentor-gpus)   MENTOR_GPUS="$2"; shift 2 ;;
        --intern-gpus)   INTERN_GPUS="$2"; shift 2 ;;
        --n-samples)     N_SAMPLES="$2";   shift 2 ;;
        --skip-mentor)   SKIP_MENTOR="--skip-mentor"; shift ;;
        --skip-generation) SKIP_GEN="--skip-generation"; shift ;;
        --data-dir)      DATA_DIR="$2";    shift 2 ;;
        --classifier-dir) CLASSIFIER_DIR="$2"; shift 2 ;;
        --output)        OUTPUT="$2";      shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Latency Measurement (Reviewer yUBt W5)"
echo "============================================================"
echo "Data dir:       $DATA_DIR"
echo "Classifier:     $CLASSIFIER_DIR"
echo "Mentor GPUs:    $MENTOR_GPUS"
echo "Intern GPUs:    $INTERN_GPUS"
echo "N samples:      $N_SAMPLES"
echo "Skip mentor:    ${SKIP_MENTOR:-no}"
echo "Skip gen:       ${SKIP_GEN:-no}"
echo "Output:         $OUTPUT"
echo "============================================================"

python eval_latency.py \
    --data-dir "$DATA_DIR" \
    --classifier-dir "$CLASSIFIER_DIR" \
    --mentor-gpus "$MENTOR_GPUS" \
    --intern-gpus "$INTERN_GPUS" \
    --n-samples "$N_SAMPLES" \
    --output "$OUTPUT" \
    $SKIP_MENTOR $SKIP_GEN

echo ""
echo "Done! Results: $OUTPUT"
