#!/bin/bash
# Test different token level combinations
#
# This script automatically tests:
# - Mentor only (tokens=-1)
# - Intern only (tokens=0)
# - Different token level combinations (default: 100, 500, 1000)
#
# Usage: ./run_test_token_levels.sh [OPTIONS]
#
# Options:
#   --mentor-model MODEL     Mentor model name (large model)
#   --intern-model MODEL     Intern model name (small model)
#   --model MODEL            Model name (if mentor/intern not specified, uses same model)
#   --token-levels LEVELS    Comma-separated token levels (default: "100,500,1000")
#   --gpus GPUS              Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   --think                  Enable thinking mode (default)
#   --no-think               Disable thinking mode
#   --skip-existing          Skip token levels that already exist
#
# Examples:
#   ./run_test_token_levels.sh --mentor-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --intern-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#   ./run_test_token_levels.sh --token-levels "100,200,500,1000"
#   ./run_test_token_levels.sh --no-think --skip-existing

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPUS="0,1,2,3,4,5,6,7"
USE_THINK=true
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MENTOR_MODEL=""
INTERN_MODEL=""
TOKEN_LEVELS="100,500,1000"
SKIP_EXISTING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --think)
            USE_THINK=true
            shift
            ;;
        --no-think)
            USE_THINK=false
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --mentor-model)
            MENTOR_MODEL="$2"
            shift 2
            ;;
        --intern-model)
            INTERN_MODEL="$2"
            shift 2
            ;;
        --token-levels)
            TOKEN_LEVELS="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build model arguments for data collection (same logic as run_pipeline.sh)
MODEL_ARGS=""
if [ -n "$MENTOR_MODEL" ] && [ -n "$INTERN_MODEL" ]; then
    MODEL_ARGS="--mentor-model $MENTOR_MODEL --intern-model $INTERN_MODEL"
    echo "Using different models: Mentor=$MENTOR_MODEL, Intern=$INTERN_MODEL"
elif [ -n "$MENTOR_MODEL" ]; then
    MODEL_ARGS="--mentor-model $MENTOR_MODEL --intern-model $MODEL"
    echo "Using mentor model: $MENTOR_MODEL, intern model: $MODEL"
elif [ -n "$INTERN_MODEL" ]; then
    MODEL_ARGS="--mentor-model $MODEL --intern-model $INTERN_MODEL"
    echo "Using mentor model: $MODEL, intern model: $INTERN_MODEL"
else
    MODEL_ARGS="--model $MODEL"
    echo "Using same model for both: $MODEL"
fi

# Extract model name(s) for directory (same logic as run_pipeline.sh)
if [ -n "$MENTOR_MODEL" ] && [ -n "$INTERN_MODEL" ]; then
    MENTOR_SHORT=$(echo $MENTOR_MODEL | sed 's|.*/||')
    INTERN_SHORT=$(echo $INTERN_MODEL | sed 's|.*/||')
    MODEL_NAME="m${MENTOR_SHORT}_i${INTERN_SHORT}"
elif [ -n "$MENTOR_MODEL" ]; then
    MENTOR_SHORT=$(echo $MENTOR_MODEL | sed 's|.*/||')
    INTERN_SHORT=$(echo $MODEL | sed 's|.*/||')
    MODEL_NAME="m${MENTOR_SHORT}_i${INTERN_SHORT}"
elif [ -n "$INTERN_MODEL" ]; then
    MENTOR_SHORT=$(echo $MODEL | sed 's|.*/||')
    INTERN_SHORT=$(echo $INTERN_MODEL | sed 's|.*/||')
    MODEL_NAME="m${MENTOR_SHORT}_i${INTERN_SHORT}"
else
    MODEL_NAME=$(echo $MODEL | sed 's|.*/||')
fi

# Set data directory based on think mode (same as run_pipeline.sh)
if [ "$USE_THINK" = true ]; then
    MODE="think"
    THINK_FLAG=""
else
    MODE="standard"
    THINK_FLAG="--no-think"
fi

DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_${MODE}_${MODEL_NAME}"

# All subsets
ALL_SUBSETS=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)

# All splits
ALL_SPLITS=(train test)

# Parse token levels
IFS=',' read -ra TOKEN_ARRAY <<< "$TOKEN_LEVELS"
TOKEN_LEVELS_LIST=(-1 0)  # Mentor only and intern only
for token in "${TOKEN_ARRAY[@]}"; do
    TOKEN_LEVELS_LIST+=($(echo $token | xargs))  # Trim whitespace
done

# Parse GPU count
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "Test Token Levels Script"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "Token levels to test: ${TOKEN_LEVELS_LIST[@]}"
echo "Subsets: ${ALL_SUBSETS[@]}"
echo "Splits: ${ALL_SPLITS[@]}"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "Skip existing: $SKIP_EXISTING"
echo "============================================================"
echo ""

# Function to check if data file exists
check_data_exists() {
    local subset=$1
    local split=$2
    local token_level=$3
    local data_file="$DATA_DIR/$subset/$split/tokens${token_level}.json"
    [ -f "$data_file" ]
}

# Function to check which token levels need collection for a split
get_missing_levels() {
    local split=$1
    local missing=()
    
    for token_level in "${TOKEN_LEVELS_LIST[@]}"; do
        local all_exist=true
        for subset in "${ALL_SUBSETS[@]}"; do
            if ! check_data_exists "$subset" "$split" "$token_level"; then
                all_exist=false
                break
            fi
        done
        
        if [ "$all_exist" = false ]; then
            missing+=($token_level)
        fi
    done
    
    echo "${missing[@]}"
}

# Function to collect data for a split and token levels (all subsets at once)
collect_data_for_split() {
    local split=$1
    local token_levels_str="$2"
    
    echo "  [COLLECT] Split=$split, Token levels=$token_levels_str (all subsets)"
    
    # Run data collection for all subsets at once
    python collect_data_vllm_think.py \
        $MODEL_ARGS \
        --dataset hendrycks_math \
        --split "$split" \
        --gpus $GPUS \
        "--token-levels=$token_levels_str" \
        --output-dir "$DATA_DIR" \
        $THINK_FLAG
    
    # Verify all files were created
    local all_success=true
    for subset in "${ALL_SUBSETS[@]}"; do
        for token_level in "${TOKEN_LEVELS_LIST[@]}"; do
            if ! check_data_exists "$subset" "$split" "$token_level"; then
                echo "  [WARNING] Missing: $subset/$split tokens=$token_level"
                all_success=false
            fi
        done
    done
    
    if [ "$all_success" = true ]; then
        echo "  [SUCCESS] All data collected for split=$split"
        return 0
    else
        echo "  [PARTIAL] Some data missing for split=$split"
        return 1
    fi
}

# Main collection loop: process by split (more efficient - one model init per split)
for split in "${ALL_SPLITS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Processing Split: $split"
    echo "============================================================"
    
    # Check which token levels need collection
    if [ "$SKIP_EXISTING" = true ]; then
        MISSING_LEVELS=($(get_missing_levels "$split"))
        if [ ${#MISSING_LEVELS[@]} -eq 0 ]; then
            echo "  [SKIP] All token levels already exist for split=$split"
            continue
        fi
        TOKEN_LEVELS_STR=$(IFS=','; echo "${MISSING_LEVELS[*]}")
        echo "  Missing token levels for split=$split: ${MISSING_LEVELS[@]}"
    else
        TOKEN_LEVELS_STR=$(IFS=','; echo "${TOKEN_LEVELS_LIST[*]}")
    fi
    
    # Collect all missing token levels for all subsets at once
    if ! collect_data_for_split "$split" "$TOKEN_LEVELS_STR"; then
        echo "  [ERROR] Failed to collect some data for split=$split"
    fi
done

echo ""
echo "============================================================"
echo "Data Collection Complete!"
echo "============================================================"
echo "Data directory: $DATA_DIR"
echo ""
echo "Summary:"
for subset in "${ALL_SUBSETS[@]}"; do
    for split in "${ALL_SPLITS[@]}"; do
        echo "  $subset/$split:"
        for token_level in "${TOKEN_LEVELS_LIST[@]}"; do
            if check_data_exists "$subset" "$split" "$token_level"; then
                echo "    tokens=$token_level: ✓"
            else
                echo "    tokens=$token_level: ✗"
            fi
        done
    done
done
echo "============================================================"

