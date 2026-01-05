#!/bin/bash
# Unified vLLM Pipeline for DeepSeek R1
# Usage: ./run_pipeline.sh [OPTIONS]
#
# Options:
#   --gpus GPUS           Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   --think               Enable thinking mode (default)
#   --no-think            Disable thinking mode (standard prompt)
#   --prompt-type TYPE    Prompt type: simple (default) or structured (with insights)
#   --model MODEL         Model name (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   --mentor-model MODEL  Mentor model name (large model for guidance)
#   --intern-model MODEL  Intern model name (small model for generation)
#   --mentor-api API      API type for mentor model: "openrouter" (for GPT-4o, Claude, etc.)
#   --api-max-workers N   Max concurrent API requests (default: 8)
#   --dataset DATASET     Dataset: hendrycks_math, math500, gsm8k, aime24, aime25 (default: hendrycks_math)
#   --subset SUBSET       (Legacy) Sets both train and eval subset (supports comma-separated: 'geometry,algebra')
#   --train-subset SUBSET Which subset(s) for training. 'all' merges all, or use 'geometry,algebra' for multiple.
#   --eval-subset SUBSET  Which subset(s) for eval. 'all' tests each separately, or use 'geometry,algebra'.
#   --lr LR               Learning rate for LoRA training (default: 1e-4)
#   --epochs EPOCHS       Number of epochs for LoRA training (default: 3)
#   --batch-size BS       Batch size for LoRA training (default: 4)
#   --no-filter           Don't filter out all-correct/all-wrong samples
#   --reserve-memory GB   Pre-allocate GPU memory (released after model load)
#   --memory-lock FRAC    Lock GPU memory at this fraction (0.0-1.0)
#   --force               Force re-training even if results exist
#   --no-val              Train on entire train set, search thresholds on train, eval on test
#   --pooling MODE        Pooling strategy (default: mean_logits, only option)
#   --dropout RATE        Dropout rate for MLP classifier (default: 0.3)
#   --fixed-threshold TH  Use fixed threshold instead of searching (e.g., 0.5)
#   --unfiltered-val      Use unfiltered data for validation/threshold search
#   --skip-epoch-cascade  Skip cascade evaluation after each epoch (faster training)
#   --method METHODS      Comma-separated list of methods to run: mlp,ppl,lora,ensemble (default: all)
#   --mlp-checkpoint PATH Path to MLP checkpoint for cross-dataset evaluation (skips training)
#   --ppl-checkpoint PATH Path to PPL checkpoint for cross-dataset evaluation (skips training)
#   --lora-checkpoint PATH Path to LoRA checkpoint for cross-dataset evaluation (skips training)
#
# Examples:
#   ./run_pipeline.sh --think                                    # Think mode, 8 GPUs, all subsets
#   ./run_pipeline.sh --train-subset all --eval-subset algebra   # Train on all, test on algebra
#   ./run_pipeline.sh --train-subset all --eval-subset all       # Train on all, test each separately
#   ./run_pipeline.sh --subset algebra                           # Train & test on algebra only
#   ./run_pipeline.sh --subset geometry,algebra                  # Train & test on geometry and algebra only
#   ./run_pipeline.sh --dataset gsm8k                            # Run on GSM8K dataset
#   ./run_pipeline.sh --dataset gsm8k --method ppl               # Only train PPL classifier
#   ./run_pipeline.sh --method mlp,ppl                           # Only train MLP and PPL
#
# Cross-Dataset Evaluation Examples:
#   # Train on GSM8K, eval on AIME24
#   ./run_pipeline.sh --dataset aime24 --method mlp \
#     --mlp-checkpoint /path/to/gsm8k_model/mlp_model/best_model.pt
#
#   # Train on MATH, eval on AIME25
#   ./run_pipeline.sh --dataset aime25 --method mlp \
#     --mlp-checkpoint /path/to/math_model/algebra/mlp_model/best_model.pt
#
# API Model Examples:
#   # GPT-4o as mentor via OpenRouter
#   ./run_pipeline.sh --mentor-model openai/gpt-4o --mentor-api openrouter --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#
#   # Claude-3.5-Sonnet as mentor
#   ./run_pipeline.sh --mentor-model anthropic/claude-3.5-sonnet --mentor-api openrouter

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPUS="0,1,2,3,4,5,6,7"
USE_THINK=true
PROMPT_TYPE="simple"  # simple or structured
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MENTOR_MODEL=""
INTERN_MODEL=""
MENTOR_API=""  # API type for mentor model (e.g., "openrouter")
API_MAX_WORKERS=8
DATASET="hendrycks_math"
SUBSET=""
TRAIN_SUBSET=""
EVAL_SUBSET=""
LR="1e-4"
EPOCHS="3"
BATCH_SIZE=4
NO_FILTER=false
RESERVE_MEMORY=0
MEMORY_LOCK=0
FORCE=false
NO_VAL=false
POOLING="mean_logits"
DROPOUT=0.3
FIXED_THRESHOLD=""
UNFILTERED_VAL=false
SKIP_EPOCH_CASCADE=false
METHODS="mlp,ppl,lora,ensemble"  # Default: run all methods
MLP_CHECKPOINT=""  # Path to MLP checkpoint for cross-dataset evaluation
PPL_CHECKPOINT=""  # Path to PPL checkpoint for cross-dataset evaluation
LORA_CHECKPOINT=""  # Path to LoRA checkpoint for cross-dataset evaluation

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
        --prompt-type)
            PROMPT_TYPE="$2"
            shift 2
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
        --mentor-api)
            MENTOR_API="$2"
            shift 2
            ;;
        --api-max-workers)
            API_MAX_WORKERS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --train-subset)
            TRAIN_SUBSET="$2"
            shift 2
            ;;
        --eval-subset)
            EVAL_SUBSET="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
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
        --no-filter)
            NO_FILTER=true
            shift
            ;;
        --reserve-memory)
            RESERVE_MEMORY="$2"
            shift 2
            ;;
        --memory-lock)
            MEMORY_LOCK="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --no-val)
            NO_VAL=true
            shift
            ;;
        --pooling)
            POOLING="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --fixed-threshold)
            FIXED_THRESHOLD="$2"
            shift 2
            ;;
        --unfiltered-val)
            UNFILTERED_VAL=true
            shift
            ;;
        --skip-epoch-cascade)
            SKIP_EPOCH_CASCADE=true
            shift
            ;;
        --method)
            METHODS="$2"
            shift 2
            ;;
        --mlp-checkpoint)
            MLP_CHECKPOINT="$2"
            shift 2
            ;;
        --ppl-checkpoint)
            PPL_CHECKPOINT="$2"
            shift 2
            ;;
        --lora-checkpoint)
            LORA_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Parse methods into flags
RUN_MLP=false
RUN_PPL=false
RUN_LORA=false
RUN_ENSEMBLE=false
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
for method in "${METHOD_ARRAY[@]}"; do
    case "$method" in
        mlp) RUN_MLP=true ;;
        ppl) RUN_PPL=true ;;
        lora) RUN_LORA=true ;;
        ensemble) RUN_ENSEMBLE=true ;;
        all) RUN_MLP=true; RUN_PPL=true; RUN_LORA=true; RUN_ENSEMBLE=true ;;
        *) echo "Unknown method: $method (valid: mlp,ppl,lora,ensemble,all)"; exit 1 ;;
    esac
done

# Extract model name(s) for directory
if [ -n "$MENTOR_MODEL" ] && [ -n "$INTERN_MODEL" ]; then
    # Both models specified: use both in path
    MENTOR_SHORT=$(echo $MENTOR_MODEL | sed 's|.*/||')
    INTERN_SHORT=$(echo $INTERN_MODEL | sed 's|.*/||')
    MODEL_NAME="m${MENTOR_SHORT}_i${INTERN_SHORT}"
elif [ -n "$MENTOR_MODEL" ]; then
    # Only mentor specified: use mentor + default intern
    MENTOR_SHORT=$(echo $MENTOR_MODEL | sed 's|.*/||')
    INTERN_SHORT=$(echo $MODEL | sed 's|.*/||')
    MODEL_NAME="m${MENTOR_SHORT}_i${INTERN_SHORT}"
elif [ -n "$INTERN_MODEL" ]; then
    # Only intern specified: use default mentor + intern
    MENTOR_SHORT=$(echo $MODEL | sed 's|.*/||')
    INTERN_SHORT=$(echo $INTERN_MODEL | sed 's|.*/||')
    MODEL_NAME="m${MENTOR_SHORT}_i${INTERN_SHORT}"
else
    # Same model for both: use single model name
MODEL_NAME=$(echo $MODEL | sed 's|.*/||')
fi

# Set data directory based on think mode
if [ "$USE_THINK" = true ]; then
    MODE="think"
    THINK_FLAG=""
else
    MODE="standard"
    THINK_FLAG="--no-think"
fi

# Set prompt type flag
if [ "$PROMPT_TYPE" = "structured" ]; then
    PROMPT_TYPE_FLAG="--prompt-type structured"
else
    PROMPT_TYPE_FLAG="--prompt-type simple"
fi

# Set data directory and subsets based on dataset
case "$DATASET" in
    hendrycks_math)
        DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_${MODE}_${MODEL_NAME}"
        ALL_SUBSETS=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)
        DATASET_FLAG=""
        ;;
    math500)
        DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/math500_${MODE}_${MODEL_NAME}"
        ALL_SUBSETS=(math500)
        DATASET_FLAG="--dataset math500"
        ;;
    gsm8k)
        DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/gsm8k_${MODE}_${MODEL_NAME}"
        ALL_SUBSETS=(gsm8k)
        DATASET_FLAG="--dataset gsm8k"
        ;;
    aime24)
        DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/aime24_${MODE}_${MODEL_NAME}"
        ALL_SUBSETS=(aime24)
        DATASET_FLAG="--dataset aime24"
        ;;
    aime25)
        DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/aime25_${MODE}_${MODEL_NAME}"
        ALL_SUBSETS=(aime25)
        DATASET_FLAG="--dataset aime25"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Supported datasets: hendrycks_math, math500, gsm8k, aime24, aime25"
        exit 1
        ;;
esac

# Handle subset arguments: --subset sets both, individual args override

# If --subset is set, use it as default for both train and eval
if [ -n "$SUBSET" ]; then
    if [ -z "$TRAIN_SUBSET" ]; then
        TRAIN_SUBSET="$SUBSET"
    fi
    if [ -z "$EVAL_SUBSET" ]; then
        EVAL_SUBSET="$SUBSET"
    fi
fi

# Default: if nothing specified, run each subset individually
if [ -z "$TRAIN_SUBSET" ]; then
    TRAIN_SUBSET=""  # Will loop through all subsets individually
fi
if [ -z "$EVAL_SUBSET" ]; then
    EVAL_SUBSET="$TRAIN_SUBSET"  # Default eval to same as train
fi

# Determine which subsets need data collection (always all individual subsets)
if [ -n "$TRAIN_SUBSET" ] && [ "$TRAIN_SUBSET" != "all" ]; then
    DATA_SUBSETS=("$TRAIN_SUBSET")
else
    DATA_SUBSETS=("${ALL_SUBSETS[@]}")
fi

# Build subset flag for data collection
SUBSET_FLAG=""
if [ -n "$TRAIN_SUBSET" ] && [ "$TRAIN_SUBSET" != "all" ]; then
    SUBSET_FLAG="--subset $TRAIN_SUBSET"
fi

# Token levels: -1 = mentor only, 0 = intern only, others = mentor hint + intern
TOKEN_LEVELS="-1,0,100,500,1000"

# Parse GPU count
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Build model arguments for data collection
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

# Add API arguments if specified
if [ -n "$MENTOR_API" ]; then
    MODEL_ARGS="$MODEL_ARGS --mentor-api $MENTOR_API --api-max-workers $API_MAX_WORKERS"
    echo "Mentor API: $MENTOR_API (max_workers=$API_MAX_WORKERS)"
fi

echo "============================================================"
echo "DeepSeek R1 vLLM Pipeline"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "Model: $MODEL"
if [ -n "$MENTOR_MODEL" ]; then
    echo "Mentor model: $MENTOR_MODEL"
fi
if [ -n "$INTERN_MODEL" ]; then
    echo "Intern model: $INTERN_MODEL"
fi
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "Train subset: ${TRAIN_SUBSET:-all (individual)}"
echo "Eval subset: ${EVAL_SUBSET:-same as train}"
echo "Methods: $METHODS"
echo "============================================================"

# Helper function to check if data collection is complete for a subset/split
# Checks all token levels: -1, 0, 100, 500, 1000
check_data_exists() {
    local subset=$1
    local split=$2
    local subset_dir="$DATA_DIR/$subset/$split"
    
    # Check all required token levels
    for token_level in -1 0 100 500 1000; do
        local data_file="$subset_dir/tokens${token_level}.json"
        if [ ! -f "$data_file" ]; then
            return 1  # not exists
        fi
    done
    return 0  # all exist
}

# Helper function to check if model already trained (from run_compare_classifier.sh)
check_model_exists() {
    local subset=$1
    local model_type=$2
    # If --force, always return "not exists" to force retraining
    if [ "$FORCE" = true ]; then
        return 1
    fi
    local result_file="$DATA_DIR/$subset/${model_type}_model/results.json"
    if [ -f "$result_file" ]; then
        return 0  # exists
    else
        return 1  # not exists
    fi
}

echo ""
echo "========== Step 1: Collect Data (${NUM_GPUS} GPUs parallel) =========="

# Build force flag for data collection
FORCE_DATA_FLAG=""
if [ "$FORCE" = true ]; then
    FORCE_DATA_FLAG="--force"
    echo "FORCE mode enabled - will re-collect all data"
fi

# Check if train data already exists (show which subsets are missing)
TRAIN_MISSING=()
if [ "$FORCE" = false ]; then
    for subset in "${DATA_SUBSETS[@]}"; do
        if ! check_data_exists "$subset" "train"; then
            TRAIN_MISSING+=("$subset")
        fi
    done
fi

if [ "$FORCE" = true ]; then
    echo "Collecting train data (FORCE mode - will overwrite existing files)..."
    python collect_data_vllm_think.py $MODEL_ARGS --split train --gpus $GPUS "--token-levels=$TOKEN_LEVELS" $THINK_FLAG $PROMPT_TYPE_FLAG $DATASET_FLAG $SUBSET_FLAG $FORCE_DATA_FLAG
elif [ ${#TRAIN_MISSING[@]} -eq 0 ]; then
    echo "Train data already exists for all subsets, skipping collection..."
else
    echo "Missing train data for: ${TRAIN_MISSING[*]}"
    echo "Collecting train data (existing files will be skipped)..."
    python collect_data_vllm_think.py $MODEL_ARGS --split train --gpus $GPUS "--token-levels=$TOKEN_LEVELS" $THINK_FLAG $PROMPT_TYPE_FLAG $DATASET_FLAG $SUBSET_FLAG
fi

# Check if test data already exists (show which subsets are missing)
TEST_MISSING=()
if [ "$FORCE" = false ]; then
    for subset in "${DATA_SUBSETS[@]}"; do
        if ! check_data_exists "$subset" "test"; then
            TEST_MISSING+=("$subset")
        fi
    done
fi

if [ "$FORCE" = true ]; then
    echo "Collecting test data (FORCE mode - will overwrite existing files)..."
    python collect_data_vllm_think.py $MODEL_ARGS --split test --gpus $GPUS "--token-levels=$TOKEN_LEVELS" $THINK_FLAG $PROMPT_TYPE_FLAG $DATASET_FLAG $SUBSET_FLAG $FORCE_DATA_FLAG
elif [ ${#TEST_MISSING[@]} -eq 0 ]; then
    echo "Test data already exists for all subsets, skipping collection..."
else
    echo "Missing test data for: ${TEST_MISSING[*]}"
    echo "Collecting test data (existing files will be skipped)..."
    python collect_data_vllm_think.py $MODEL_ARGS --split test --gpus $GPUS "--token-levels=$TOKEN_LEVELS" $THINK_FLAG $PROMPT_TYPE_FLAG $DATASET_FLAG $SUBSET_FLAG
fi

echo ""
echo "========== Step 2: Data Statistics =========="
python compute_stats.py --data-dir $DATA_DIR --split train
python compute_stats.py --data-dir $DATA_DIR --split test

# Common flags for training
FILTER_FLAG=""
if [ "$NO_FILTER" = true ]; then
    FILTER_FLAG="--no-filter"
fi
NO_VAL_FLAG=""
if [ "$NO_VAL" = true ]; then
    NO_VAL_FLAG="--no-val"
fi
FIXED_THRESHOLD_FLAG=""
if [ -n "$FIXED_THRESHOLD" ]; then
    FIXED_THRESHOLD_FLAG="--fixed-threshold $FIXED_THRESHOLD"
fi
UNFILTERED_VAL_FLAG=""
if [ "$UNFILTERED_VAL" = true ]; then
    UNFILTERED_VAL_FLAG="--unfiltered-val"
fi
SKIP_EPOCH_CASCADE_FLAG=""
if [ "$SKIP_EPOCH_CASCADE" = true ]; then
    SKIP_EPOCH_CASCADE_FLAG="--skip-epoch-cascade"
fi

echo ""
echo "========== Step 3: Train MLP Classifiers =========="
if [ "$RUN_MLP" = true ]; then
    if [ -n "$MLP_CHECKPOINT" ]; then
        # Cross-dataset evaluation with loaded checkpoint
        echo ">>> MLP: Cross-dataset evaluation (eval-only mode)"
        echo "    Checkpoint: $MLP_CHECKPOINT"
        echo "    Target dataset: $DATASET"
        echo "    Data directory: $DATA_DIR"

        # Build eval flag if specified, otherwise auto-detect
        EVAL_FLAG=""
        if [ -n "$EVAL_SUBSET" ]; then
            EVAL_FLAG="--eval-subset $EVAL_SUBSET"
            echo "    Eval subset: $EVAL_SUBSET (user-specified)"
        else
            echo "    Eval subset: auto-detecting from data directory..."
        fi

        CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS train_mlp_classifier.py \
            --ddp $EVAL_FLAG --data-dir $DATA_DIR \
            --load-checkpoint "$MLP_CHECKPOINT" --eval-only \
            --pooling $POOLING
    elif [ -n "$TRAIN_SUBSET" ]; then
        # Specific train subset specified
        if [ "$TRAIN_SUBSET" = "all" ]; then
            MODEL_DIR="all"
        else
            MODEL_DIR="$TRAIN_SUBSET"
        fi

        if check_model_exists "$MODEL_DIR" "mlp"; then
            echo ">>> MLP: train=$TRAIN_SUBSET, eval=$EVAL_SUBSET [SKIP - already trained, use --force to retrain]"
        else
            echo ">>> Training MLP: train=$TRAIN_SUBSET, eval=$EVAL_SUBSET"
            echo ">>> (lr=$LR, epochs=$EPOCHS, batch_size=$BATCH_SIZE, pooling=$POOLING, dropout=$DROPOUT, no_val=$NO_VAL)"
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS train_mlp_classifier.py \
                --ddp --train-subset $TRAIN_SUBSET --eval-subset $EVAL_SUBSET --data-dir $DATA_DIR --lr $LR --epochs $EPOCHS \
                --batch-size $BATCH_SIZE --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK \
                --pooling $POOLING --dropout $DROPOUT $FILTER_FLAG $NO_VAL_FLAG $FIXED_THRESHOLD_FLAG $UNFILTERED_VAL_FLAG $SKIP_EPOCH_CASCADE_FLAG
        fi
    else
        # No train subset specified - train each subset individually
        for subset in "${ALL_SUBSETS[@]}"; do
            if check_model_exists "$subset" "mlp"; then
                echo ">>> MLP: $subset [SKIP - already trained, use --force to retrain]"
            else
                echo ">>> Training MLP: $subset (lr=$LR, epochs=$EPOCHS, batch_size=$BATCH_SIZE, pooling=$POOLING, dropout=$DROPOUT, no_val=$NO_VAL)"
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS train_mlp_classifier.py \
                    --ddp --train-subset $subset --eval-subset $subset --data-dir $DATA_DIR --lr $LR --epochs $EPOCHS \
                    --batch-size $BATCH_SIZE --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK \
                    --pooling $POOLING --dropout $DROPOUT $FILTER_FLAG $NO_VAL_FLAG $FIXED_THRESHOLD_FLAG $UNFILTERED_VAL_FLAG $SKIP_EPOCH_CASCADE_FLAG
            fi
        done
    fi
else
    echo ">>> MLP: [SKIP - not in --method]"
fi

echo ""
echo "========== Step 4: Train PPL Classifiers =========="
if [ "$RUN_PPL" = true ]; then
    if [ -n "$TRAIN_SUBSET" ]; then
        # Specific train subset specified
        if [ "$TRAIN_SUBSET" = "all" ]; then
            PPL_MODEL_DIR="all"
        else
            PPL_MODEL_DIR="$TRAIN_SUBSET"
        fi

        if check_model_exists "$PPL_MODEL_DIR" "ppl"; then
            echo ">>> PPL: train=$TRAIN_SUBSET, eval=$EVAL_SUBSET [SKIP - already trained, use --force to retrain]"
        else
            echo ">>> Training PPL: train=$TRAIN_SUBSET, eval=$EVAL_SUBSET (no_val=$NO_VAL)"
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS train_ppl_classifier.py \
                --ddp --train-subset $TRAIN_SUBSET --eval-subset $EVAL_SUBSET --data-dir $DATA_DIR $NO_VAL_FLAG
        fi
    else
        # No train subset specified - train each subset individually
        for subset in "${ALL_SUBSETS[@]}"; do
            if check_model_exists "$subset" "ppl"; then
                echo ">>> PPL: $subset [SKIP - already trained, use --force to retrain]"
            else
                echo ">>> Training PPL: $subset (no_val=$NO_VAL)"
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS train_ppl_classifier.py \
                    --ddp --train-subset $subset --eval-subset $subset --data-dir $DATA_DIR $NO_VAL_FLAG
            fi
        done
    fi
else
    echo ">>> PPL: [SKIP - not in --method]"
fi

echo ""
echo "========== Step 5: Train Ensemble (MLP + PPL) =========="
if [ "$RUN_ENSEMBLE" = true ]; then
    if [ "$TRAIN_SUBSET" = "all" ]; then
        echo ">>> Ensemble training skipped for --train-subset all (not supported)"
    else
        # Determine subsets for Ensemble training
        if [ -n "$TRAIN_SUBSET" ]; then
            ENS_SUBSETS=("$TRAIN_SUBSET")
        else
            ENS_SUBSETS=("${ALL_SUBSETS[@]}")
        fi
        for subset in "${ENS_SUBSETS[@]}"; do
            if check_model_exists "$subset" "ensemble"; then
                echo ">>> Ensemble: $subset [SKIP - already trained, use --force to retrain]"
            else
                echo ">>> Training Ensemble: $subset"
                CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python train_ensemble_classifier.py \
                    --subset $subset --data-dir $DATA_DIR --base-model $MODEL --use-mlp
            fi
        done
    fi
else
    echo ">>> Ensemble: [SKIP - not in --method]"
fi

echo ""
echo "========== Step 6: Train LoRA Classifiers =========="
if [ "$RUN_LORA" = true ]; then
    if [ -n "$TRAIN_SUBSET" ]; then
        # Specific train subset specified
        if [ "$TRAIN_SUBSET" = "all" ]; then
            LORA_MODEL_DIR="all"
            if check_model_exists "$LORA_MODEL_DIR" "lora"; then
                echo ">>> LoRA: train=all, eval=$EVAL_SUBSET [SKIP - already trained, use --force to retrain]"
            else
                echo ">>> Training LoRA: train=all, eval=$EVAL_SUBSET"
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 train_lora_classifier.py \
                    --ddp --subset all --eval-subset "$EVAL_SUBSET" --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE \
                    --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK $NO_VAL_FLAG
            fi
        else
            LORA_MODEL_DIR="$TRAIN_SUBSET"
            if check_model_exists "$LORA_MODEL_DIR" "lora"; then
                echo ">>> LoRA: $TRAIN_SUBSET [SKIP - already trained, use --force to retrain]"
            else
                echo ">>> Training LoRA: $TRAIN_SUBSET"
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 train_lora_classifier.py \
                    --ddp --subset $TRAIN_SUBSET --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE \
                    --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK $NO_VAL_FLAG
            fi
        fi
    else
        # No train subset specified - train each subset individually
        for subset in "${ALL_SUBSETS[@]}"; do
            if check_model_exists "$subset" "lora"; then
                echo ">>> LoRA: $subset [SKIP - already trained, use --force to retrain]"
            else
                echo ">>> Training LoRA: $subset"
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 train_lora_classifier.py \
                    --ddp --subset $subset --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE \
                    --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK $NO_VAL_FLAG
            fi
        done
    fi
else
    echo ">>> LoRA: [SKIP - not in --method]"
fi

echo ""
echo "========== Step 7: Summarize Results =========="
# 根据训练方式选择 model-source
if [[ "$TRAIN_SUBSET" == "all" ]]; then
    python summarize_results.py --data-dir $DATA_DIR --model-source all
else
    python summarize_results.py --data-dir $DATA_DIR --model-source individual
fi

echo ""
echo "========== Done! =========="
echo "Results saved to: $DATA_DIR"
