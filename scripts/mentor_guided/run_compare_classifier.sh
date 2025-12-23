#!/bin/bash
# Compare LoRA vs MLP vs PPL classifier performance
# Usage: ./run_compare_classifier.sh [OPTIONS]
#
# Options:
#   --gpus GPUS           Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   --subset SUBSET       (Legacy) Sets both train and eval subset
#   --train-subset SUBSET Which subset(s) for training. 'all' merges all subsets.
#   --eval-subset SUBSET  Which subset(s) for eval. 'all' tests each subset separately.
#   --data-dir DIR        Data directory
#   --methods METHODS     Comma-separated methods: lora,mlp,ppl,ensemble,all (default: lora,mlp,ppl)
#   --check               Only check file status (no training)
#   --force               Force re-training even if results exist
#   --batch-size BS       Batch size for LoRA/MLP training (default: 4)
#   --epochs EPOCHS       Number of training epochs (default: 2)
#   --reserve-memory GB   Pre-allocate GPU memory
#   --memory-lock FRAC    Lock GPU memory at this fraction (0.0-1.0)
#   --no-val              Train on entire train set, eval on test
#   --pooling MODE        Pooling strategy (default: mean_logits, only option)
#   --dropout RATE        Dropout rate for MLP classifier (default: 0.3)
#   --fixed-threshold TH  Use fixed threshold instead of searching
#   --skip-epoch-cascade  Skip cascade evaluation after each epoch
#   --classifier TYPE     PPL classifier type: gb (GradientBoosting), lr (LogisticRegression)
#
# Examples:
#   ./run_compare_classifier.sh --methods mlp                    # Only test MLP
#   ./run_compare_classifier.sh --methods mlp,ppl                # Test MLP and PPL
#   ./run_compare_classifier.sh --methods mlp,ppl,ensemble       # MLP, PPL, then ensemble (MLP+PPL)
#   ./run_compare_classifier.sh --train-subset all --eval-subset all  # Train on all, test each
#   ./run_compare_classifier.sh --subset algebra --methods lora  # LoRA on algebra only

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPUS="0,1,2,3,4,5,6,7"
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B"
SUBSET=""
TRAIN_SUBSET=""
EVAL_SUBSET=""
METHODS="lora,mlp,ppl"  # Default: all methods
CHECK_ONLY=false
FORCE=false
BATCH_SIZE=4
EPOCHS=2
RESERVE_MEMORY=0
MEMORY_LOCK=0
NO_VAL=false
POOLING="mean_logits"
DROPOUT=0.3
FIXED_THRESHOLD=""
SKIP_EPOCH_CASCADE=false
CLASSIFIER="gb"

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
        --train-subset)
            TRAIN_SUBSET="$2"
            shift 2
            ;;
        --eval-subset)
            EVAL_SUBSET="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --reserve-memory)
            RESERVE_MEMORY="$2"
            shift 2
            ;;
        --memory-lock)
            MEMORY_LOCK="$2"
            shift 2
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
        --skip-epoch-cascade)
            SKIP_EPOCH_CASCADE=true
            shift
            ;;
        --classifier)
            CLASSIFIER="$2"
            shift 2
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

# Parse methods
RUN_LORA=false
RUN_MLP=false
RUN_PPL=false
RUN_ENSEMBLE=false
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
for method in "${METHOD_ARRAY[@]}"; do
    case $method in
        lora) RUN_LORA=true ;;
        mlp) RUN_MLP=true ;;
        ppl) RUN_PPL=true ;;
        ensemble) RUN_ENSEMBLE=true ;;
        all) RUN_LORA=true; RUN_MLP=true; RUN_PPL=true; RUN_ENSEMBLE=true ;;
        *) echo "Unknown method: $method (valid: lora, mlp, ppl, ensemble, all)"; exit 1 ;;
    esac
done

# All subsets list
ALL_SUBSETS=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)

# Handle subset arguments: --subset sets both, individual args override
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

# Determine subsets to iterate (for display and individual training)
if [ -n "$TRAIN_SUBSET" ] && [ "$TRAIN_SUBSET" != "all" ]; then
    SUBSETS=("$TRAIN_SUBSET")
elif [ "$TRAIN_SUBSET" = "all" ]; then
    SUBSETS=("all")  # Single "all" entry for unified training
else
    SUBSETS=("${ALL_SUBSETS[@]}")  # Loop through each
fi

echo "============================================================"
echo "Classifier Comparison: LoRA vs MLP vs PPL"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "Train subset: ${TRAIN_SUBSET:-each individually}"
echo "Eval subset: ${EVAL_SUBSET:-same as train}"
echo "Methods: $METHODS"
echo "Batch size: $BATCH_SIZE"
echo "Memory lock: $MEMORY_LOCK"
echo "============================================================"

# Show GPU memory status
echo ""
echo "========== GPU Memory Status =========="
python3 << GPUEOF
import torch

gpus = "$GPUS".split(',')
print(f"{'GPU':<6} {'Name':<25} {'Total':>10} {'Used':>10} {'Free':>10}")
print("-" * 65)

for gpu_str in gpus:
    gpu_id = int(gpu_str)
    if gpu_id >= torch.cuda.device_count():
        print(f"{gpu_id:<6} {'Not available':<25}")
        continue

    props = torch.cuda.get_device_properties(gpu_id)
    total = props.total_memory / 1024**3
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True, text=True
        )
        used = float(result.stdout.strip()) / 1024
    except:
        used = 0
    free = total - used

    status = ""
    if free < 18:
        status = " <- May OOM for LoRA"
    elif free < 20:
        status = " <- Tight for LoRA"

    print(f"{gpu_id:<6} {props.name[:25]:<25} {total:>9.1f}G {used:>9.1f}G {free:>9.1f}G{status}")

print("-" * 65)
print("Note: 7B bf16 model needs ~14GB, LoRA training needs ~18-20GB total")

memory_lock = float("$MEMORY_LOCK") if "$MEMORY_LOCK" else 0
if memory_lock > 0:
    print(f"\nMemory lock: {memory_lock*100:.0f}%")
    for gpu_str in gpus[:1]:
        gpu_id = int(gpu_str)
        if gpu_id < torch.cuda.device_count():
            total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            locked = total * memory_lock
            print(f"  GPU {gpu_id}: Will lock {locked:.1f}GB / {total:.1f}GB")
GPUEOF
echo "============================================================"

# Helper function to check if model already trained
check_model_exists() {
    local model_dir=$1
    local model_type=$2
    if [ "$FORCE" = true ]; then
        return 1
    fi
    local result_file="$DATA_DIR/$model_dir/${model_type}_model/results.json"
    if [ -f "$result_file" ]; then
        return 0
    else
        return 1
    fi
}

# Build common flags
NO_VAL_FLAG=""
if [ "$NO_VAL" = true ]; then
    NO_VAL_FLAG="--no-val"
fi
FIXED_THRESHOLD_FLAG=""
if [ -n "$FIXED_THRESHOLD" ]; then
    FIXED_THRESHOLD_FLAG="--fixed-threshold $FIXED_THRESHOLD"
fi
SKIP_EPOCH_CASCADE_FLAG=""
if [ "$SKIP_EPOCH_CASCADE" = true ]; then
    SKIP_EPOCH_CASCADE_FLAG="--skip-epoch-cascade"
fi

# Check-only mode: show file status and results
if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "========== FILE STATUS =========="
    printf "%-25s %-10s %-10s %-10s %-10s\n" "Subset" "LoRA" "MLP" "PPL" "Ensemble"
    echo "--------------------------------------------------------------------"

    # Check for unified "all" model
    if [ -d "$DATA_DIR/all/mlp_model" ] || [ -d "$DATA_DIR/all/lora_model" ]; then
        lora_status="-"
        mlp_status="-"
        ppl_status="-"
        ens_status="-"
        [ -f "$DATA_DIR/all/lora_model/results.json" ] && lora_status="OK"
        # Check for results_all.json or results.json
        [ -f "$DATA_DIR/all/mlp_model/results_all.json" ] && mlp_status="OK"
        [ -f "$DATA_DIR/all/mlp_model/results.json" ] && mlp_status="OK"
        [ -f "$DATA_DIR/all/ppl_model/results.json" ] && ppl_status="OK"
        [ -f "$DATA_DIR/all/ensemble_model/results.json" ] && ens_status="OK"
        printf "%-25s %-10s %-10s %-10s %-10s\n" "all (unified)" "$lora_status" "$mlp_status" "$ppl_status" "$ens_status"
        echo "--------------------------------------------------------------------"
    fi

    for subset in "${ALL_SUBSETS[@]}"; do
        lora_status="-"
        mlp_status="-"
        ppl_status="-"
        ens_status="-"
        [ -f "$DATA_DIR/$subset/lora_model/results.json" ] && lora_status="OK"
        [ -f "$DATA_DIR/$subset/mlp_model/results.json" ] && mlp_status="OK"
        [ -f "$DATA_DIR/$subset/ppl_model/results.json" ] && ppl_status="OK"
        [ -f "$DATA_DIR/$subset/ensemble_model/results.json" ] && ens_status="OK"
        printf "%-25s %-10s %-10s %-10s %-10s\n" "$subset" "$lora_status" "$mlp_status" "$ppl_status" "$ens_status"
    done
    echo "======================================================================"

    # Show results comparison
    echo ""
    echo "========== CASCADE ACCURACY =========="
    python3 << PYEOF
import json
import os

data_dir = "$DATA_DIR"
all_subsets = "algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus".split()

print(f"{'Subset':<25} {'LoRA':>10} {'MLP':>10} {'PPL':>10} {'Ensemble':>10} {'Oracle':>10} {'Best':>10}")
print("-" * 90)

# Check for unified "all" model results (MLP and PPL)
all_mlp_dir = os.path.join(data_dir, "all", "mlp_model")
all_ppl_dir = os.path.join(data_dir, "all", "ppl_model")

all_mlp_results = None
for fname in ["results_all.json", "results.json"]:
    fpath = os.path.join(all_mlp_dir, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            all_mlp_results = json.load(f)
        break

all_ppl_results = None
ppl_path = os.path.join(all_ppl_dir, "results.json")
if os.path.exists(ppl_path):
    with open(ppl_path) as f:
        all_ppl_results = json.load(f)

if all_mlp_results or all_ppl_results:
    mlp_acc = all_mlp_results.get('test_best_cascade_acc', all_mlp_results.get('best_cascade_acc', 0)) if all_mlp_results else 0
    ppl_acc = all_ppl_results.get('test_best_cascade_acc', all_ppl_results.get('best_cascade_acc', 0)) if all_ppl_results else 0
    oracle = (all_ppl_results or all_mlp_results).get('test_oracle_acc', (all_ppl_results or all_mlp_results).get('oracle_acc', 0))
    mlp_str = f"{mlp_acc:.4f}" if mlp_acc else "-"
    ppl_str = f"{ppl_acc:.4f}" if ppl_acc else "-"
    best = "MLP" if mlp_acc >= ppl_acc else "PPL"
    print(f"{'all (unified)':<25} {'-':>10} {mlp_str:>10} {ppl_str:>10} {'-':>10} {oracle:>10.4f} {best:>10}")
    print("-" * 90)

for subset in all_subsets:
    row = {"lora": "-", "mlp": "-", "ppl": "-", "ensemble": "-", "oracle": "-"}
    best_val, best_name = 0, "-"

    oracles = []
    for m in ["lora", "mlp", "ppl", "ensemble"]:
        # First check unified "all" model's per-subset results
        if m == "mlp" and all_mlp_results and 'test_results_per_subset' in all_mlp_results:
            if subset in all_mlp_results['test_results_per_subset']:
                sub_r = all_mlp_results['test_results_per_subset'][subset]
                acc = sub_r.get('cascade_acc', 0)
                row[m] = f"{acc:.4f}"
                oracle_val = sub_r.get('oracle_acc')
                if oracle_val is not None:
                    oracles.append(oracle_val)
                    if row["oracle"] == "-":
                        row["oracle"] = f"{oracle_val:.4f}"
                if acc > best_val:
                    best_val, best_name = acc, "MLP"
                continue
        if m == "ppl" and all_ppl_results and 'test_results' in all_ppl_results:
            if subset in all_ppl_results['test_results']:
                sub_r = all_ppl_results['test_results'][subset]
                acc = sub_r.get('cascade_acc', 0)
                row[m] = f"{acc:.4f}"
                oracle_val = sub_r.get('oracle_acc')
                if oracle_val is not None:
                    oracles.append(oracle_val)
                    # Prefer PPL's Oracle
                    row["oracle"] = f"{oracle_val:.4f}"
                if acc > best_val:
                    best_val, best_name = acc, "PPL"
                continue
        # Fall back to per-subset results file
        path = f"{data_dir}/{subset}/{m}_model/results.json"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    r = json.load(f)
                acc = r.get('test_best_cascade_acc', r.get('best_cascade_acc', 0))
                row[m] = f"{acc:.4f}"
                oracle_val = r.get('test_oracle_acc', r.get('oracle_acc'))
                if oracle_val is not None:
                    oracles.append(oracle_val)
                    # Prefer PPL's Oracle (evaluated on raw test data)
                    if m == "ppl" or row["oracle"] == "-":
                        row["oracle"] = f"{oracle_val:.4f}"
                if acc > best_val:
                    best_val, best_name = acc, m.upper()[:3]
            except:
                pass

    # Check Oracle consistency
    if len(set(f"{o:.4f}" for o in oracles)) > 1:
        row["oracle"] = "MISMATCH"

    print(f"{subset:<25} {row['lora']:>10} {row['mlp']:>10} {row['ppl']:>10} {row['ensemble']:>10} {row['oracle']:>10} {best_name:>10}")

print("-" * 90)
PYEOF
    exit 0
fi

# Determine model directory for "all" training
if [ "$TRAIN_SUBSET" = "all" ]; then
    MODEL_DIR="all"
else
    MODEL_DIR="$TRAIN_SUBSET"
fi

# Train LoRA classifiers
if [ "$RUN_LORA" = true ]; then
    echo ""
    echo "========== Training LoRA Classifiers =========="

    if [ "$TRAIN_SUBSET" = "all" ]; then
        echo ">>> LoRA training not supported for --train-subset all"
    else
        # Determine subsets for LoRA training
        if [ -n "$TRAIN_SUBSET" ]; then
            LORA_SUBSETS=("$TRAIN_SUBSET")
        else
            LORA_SUBSETS=("${ALL_SUBSETS[@]}")
        fi

        for subset in "${LORA_SUBSETS[@]}"; do
            if check_model_exists "$subset" "lora"; then
                echo ""
                echo ">>> LoRA: $subset [SKIP - already trained]"
            else
                echo ""
                echo ">>> LoRA: $subset"
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 train_lora_classifier.py \
                    --ddp --subset $subset --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE \
                    --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK
            fi
        done
    fi
else
    echo ""
    echo "========== Skipping LoRA (not in --methods) =========="
fi

# Train MLP classifiers
if [ "$RUN_MLP" = true ]; then
    echo ""
    echo "========== Training MLP Classifiers (Frozen LLM) =========="

    if [ "$TRAIN_SUBSET" = "all" ]; then
        # Unified training on all subsets
        if check_model_exists "all" "mlp"; then
            echo ">>> MLP: train=all, eval=$EVAL_SUBSET [SKIP - already trained]"
        else
            echo ">>> MLP: train=all, eval=$EVAL_SUBSET"
            echo "    (pooling=$POOLING, dropout=$DROPOUT, no_val=$NO_VAL)"
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29506 train_mlp_classifier.py \
                --ddp --train-subset all --eval-subset "$EVAL_SUBSET" --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE \
                --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK \
                --pooling $POOLING --dropout $DROPOUT $NO_VAL_FLAG $FIXED_THRESHOLD_FLAG $SKIP_EPOCH_CASCADE_FLAG
        fi
    else
        # Individual subset training
        if [ -n "$TRAIN_SUBSET" ]; then
            MLP_SUBSETS=("$TRAIN_SUBSET")
        else
            MLP_SUBSETS=("${ALL_SUBSETS[@]}")
        fi

        for subset in "${MLP_SUBSETS[@]}"; do
            if check_model_exists "$subset" "mlp"; then
                echo ""
                echo ">>> MLP: $subset [SKIP - already trained]"
            else
                echo ""
                echo ">>> MLP: $subset"
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29506 train_mlp_classifier.py \
                    --ddp --train-subset $subset --eval-subset $subset --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE \
                    --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK \
                    --pooling $POOLING --dropout $DROPOUT $NO_VAL_FLAG $FIXED_THRESHOLD_FLAG $SKIP_EPOCH_CASCADE_FLAG
            fi
        done
    fi
else
    echo ""
    echo "========== Skipping MLP (not in --methods) =========="
fi

# Train PPL classifiers
if [ "$RUN_PPL" = true ]; then
    echo ""
    echo "========== Training PPL Classifiers (Entropy-based) =========="

    if [ "$TRAIN_SUBSET" = "all" ]; then
        # Unified training on all subsets merged
        if check_model_exists "all" "ppl"; then
            echo ">>> PPL: train=all, eval=$EVAL_SUBSET [SKIP - already trained]"
        else
            echo ">>> PPL: train=all, eval=$EVAL_SUBSET (no_val=$NO_VAL, classifier=$CLASSIFIER)"
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29507 train_ppl_classifier.py \
                --ddp --train-subset all --eval-subset "$EVAL_SUBSET" --data-dir $DATA_DIR \
                --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK --classifier $CLASSIFIER $NO_VAL_FLAG
        fi
    else
        # Individual subset training
        if [ -n "$TRAIN_SUBSET" ]; then
            PPL_SUBSETS=("$TRAIN_SUBSET")
        else
            PPL_SUBSETS=("${ALL_SUBSETS[@]}")
        fi

        for subset in "${PPL_SUBSETS[@]}"; do
            if check_model_exists "$subset" "ppl"; then
                echo ""
                echo ">>> PPL: $subset [SKIP - already trained]"
            else
                echo ""
                echo ">>> PPL: $subset (no_val=$NO_VAL, classifier=$CLASSIFIER)"
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29507 train_ppl_classifier.py \
                    --ddp --train-subset $subset --eval-subset $subset --data-dir $DATA_DIR \
                    --reserve-memory $RESERVE_MEMORY --memory-lock $MEMORY_LOCK --classifier $CLASSIFIER $NO_VAL_FLAG
            fi
        done
    fi
else
    echo ""
    echo "========== Skipping PPL (not in --methods) =========="
fi

# Train Ensemble classifiers (MLP+LoRA, MLP+PPL, or LoRA+PPL)
if [ "$RUN_ENSEMBLE" = true ]; then
    echo ""
    echo "========== Training Ensemble Classifiers =========="

    # Ensemble always trains per-subset
    if [ "$TRAIN_SUBSET" = "all" ] || [ -z "$TRAIN_SUBSET" ]; then
        ENS_SUBSETS=("${ALL_SUBSETS[@]}")
    else
        ENS_SUBSETS=("$TRAIN_SUBSET")
    fi

    # Determine which models to use for ensemble
    # Priority: MLP+LoRA > MLP > LoRA (with optional PPL)
    ENSEMBLE_FLAG=""
    if [ "$RUN_MLP" = true ] && [ "$RUN_LORA" = true ]; then
        ENSEMBLE_FLAG="--mlp-lora --no-ppl"
        echo ">>> Using MLP + LoRA for ensemble (best combination)"
    elif [ "$RUN_MLP" = true ]; then
        ENSEMBLE_FLAG="--use-mlp"
        echo ">>> Using MLP + PPL for ensemble"
    elif [ "$RUN_LORA" = true ]; then
        echo ">>> Using LoRA + PPL for ensemble"
    else
        # Check what exists - prefer MLP+LoRA if both available
        HAS_MLP=false
        HAS_LORA=false
        for subset in "${ENS_SUBSETS[@]}"; do
            if [ -f "$DATA_DIR/$subset/mlp_model/results.json" ] || [ -f "$DATA_DIR/all/mlp_model/results.json" ]; then
                HAS_MLP=true
            fi
            if [ -f "$DATA_DIR/$subset/lora_model/results.json" ] || [ -f "$DATA_DIR/all/lora_model/results.json" ]; then
                HAS_LORA=true
            fi
        done
        if [ "$HAS_MLP" = true ] && [ "$HAS_LORA" = true ]; then
            ENSEMBLE_FLAG="--mlp-lora --no-ppl"
            echo ">>> Found both MLP and LoRA models, using MLP + LoRA for ensemble"
        elif [ "$HAS_MLP" = true ]; then
            ENSEMBLE_FLAG="--use-mlp"
            echo ">>> Found MLP models, using MLP + PPL for ensemble"
        elif [ "$HAS_LORA" = true ]; then
            echo ">>> Found LoRA models, using LoRA + PPL for ensemble"
        fi
    fi

    for subset in "${ENS_SUBSETS[@]}"; do
        if check_model_exists "$subset" "ensemble"; then
            echo ""
            echo ">>> Ensemble: $subset [SKIP - already trained]"
        else
            echo ""
            echo ">>> Ensemble: $subset"
            CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python train_ensemble_classifier.py \
                --subset $subset --data-dir $DATA_DIR $ENSEMBLE_FLAG
        fi
    done
else
    echo ""
    echo "========== Skipping Ensemble (not in --methods) =========="
fi

# Compare results using Python for better formatting
echo ""
echo "============================================================"
echo "                    COMPARISON RESULTS"
echo "============================================================"

python3 << EOF
import json
import os

data_dir = "$DATA_DIR"
all_subsets = "algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus".split()
train_subset = "$TRAIN_SUBSET"

TOKEN_LEVELS = [0, 100, 500, 1000]

def get_results(subset, model_type):
    result_file = os.path.join(data_dir, subset, f"{model_type}_model", "results.json")
    if os.path.exists(result_file):
        try:
            with open(result_file) as f:
                return json.load(f)
        except:
            pass
    return None

def get_cascade_acc(r):
    if not r:
        return None
    # Check different key names used in various result formats
    return r.get('test_best_cascade_acc', r.get('best_cascade_acc', r.get('cascade_acc')))

def get_oracle(r):
    if not r:
        return None
    return r.get('test_oracle_acc', r.get('oracle_acc'))

def get_baseline(r):
    if not r:
        return None
    return r.get('test_per_stage_baseline_acc', r.get('per_stage_baseline_acc'))

print()
print("=" * 115)
print("                              CASCADE ACCURACY COMPARISON")
print("=" * 115)
print(f"{'Subset':<25} {'LoRA':>12} {'MLP':>12} {'PPL':>12} {'Ensemble':>12} {'Oracle':>12} {'Best':>12}")
print("-" * 115)

# Check for unified "all" model results (MLP and PPL)
all_mlp_dir = os.path.join(data_dir, "all", "mlp_model")
all_ppl_dir = os.path.join(data_dir, "all", "ppl_model")

all_mlp_results = None
for fname in ["results_all.json", "results.json"]:
    fpath = os.path.join(all_mlp_dir, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            all_mlp_results = json.load(f)
        break

all_ppl_results = None
ppl_path = os.path.join(all_ppl_dir, "results.json")
if os.path.exists(ppl_path):
    with open(ppl_path) as f:
        all_ppl_results = json.load(f)

if (all_mlp_results or all_ppl_results) and train_subset == "all":
    mlp_casc = get_cascade_acc(all_mlp_results) if all_mlp_results else None
    ppl_casc = get_cascade_acc(all_ppl_results) if all_ppl_results else None
    oracle_val = get_oracle(all_ppl_results) or get_oracle(all_mlp_results)
    mlp_acc = f"{mlp_casc:.4f}" if mlp_casc is not None else "-"
    ppl_acc = f"{ppl_casc:.4f}" if ppl_casc is not None else "-"
    oracle = f"{oracle_val:.4f}" if oracle_val is not None else "-"
    best = "MLP" if (mlp_casc or 0) >= (ppl_casc or 0) else "PPL"
    print(f"{'all (unified)':<25} {'-':>12} {mlp_acc:>12} {ppl_acc:>12} {'-':>12} {oracle:>12} {best:>12}")
    print("-" * 115)

all_results = {}
for subset in all_subsets:
    lora = get_results(subset, "lora")
    mlp = get_results(subset, "mlp")
    ppl = get_results(subset, "ppl")
    ensemble = get_results(subset, "ensemble")

    # Override with unified "all" model per-subset results if available
    if all_mlp_results and 'test_results_per_subset' in all_mlp_results:
        if subset in all_mlp_results['test_results_per_subset']:
            mlp = all_mlp_results['test_results_per_subset'][subset]
    if all_ppl_results and 'test_results' in all_ppl_results:
        if subset in all_ppl_results['test_results']:
            ppl = all_ppl_results['test_results'][subset]

    lora_casc = get_cascade_acc(lora)
    mlp_casc = get_cascade_acc(mlp)
    ppl_casc = get_cascade_acc(ppl)
    ensemble_casc = get_cascade_acc(ensemble)

    lora_acc = f"{lora_casc:.4f}" if lora_casc is not None else "-"
    mlp_acc = f"{mlp_casc:.4f}" if mlp_casc is not None else "-"
    ppl_acc = f"{ppl_casc:.4f}" if ppl_casc is not None else "-"
    ensemble_acc = f"{ensemble_casc:.4f}" if ensemble_casc is not None else "-"

    # Use each classifier's own Oracle for fair comparison
    # Priority: PPL > MLP > LoRA > Ensemble (PPL evaluated on raw test data)
    oracle_val = get_oracle(ppl) or get_oracle(mlp) or get_oracle(lora) or get_oracle(ensemble)
    oracle = f"{oracle_val:.4f}" if oracle_val is not None else "-"

    accs = []
    if lora_casc is not None:
        lora_oracle = get_oracle(lora)
        accs.append(('LoRA', lora_casc, lora_oracle))
    if mlp_casc is not None:
        mlp_oracle = get_oracle(mlp)
        accs.append(('MLP', mlp_casc, mlp_oracle))
    if ppl_casc is not None:
        ppl_oracle = get_oracle(ppl)
        accs.append(('PPL', ppl_casc, ppl_oracle))
    if ensemble_casc is not None:
        ens_oracle = get_oracle(ensemble)
        accs.append(('Ens', ensemble_casc, ens_oracle))

    # Check for Oracle inconsistency (bug detection)
    oracles = [o for _, _, o in accs if o is not None]
    if len(set(f"{o:.4f}" for o in oracles)) > 1:
        oracle = "MISMATCH"  # Flag Oracle inconsistency

    best = max(accs, key=lambda x: x[1])[0] if accs else "-"

    print(f"{subset:<25} {lora_acc:>12} {mlp_acc:>12} {ppl_acc:>12} {ensemble_acc:>12} {oracle:>12} {best:>12}")

    all_results[subset] = {'lora': lora, 'mlp': mlp, 'ppl': ppl, 'ensemble': ensemble}

print("-" * 115)

# Compute averages
lora_accs = [get_cascade_acc(r['lora']) for r in all_results.values() if get_cascade_acc(r['lora']) is not None]
mlp_accs = [get_cascade_acc(r['mlp']) for r in all_results.values() if get_cascade_acc(r['mlp']) is not None]
ppl_accs = [get_cascade_acc(r['ppl']) for r in all_results.values() if get_cascade_acc(r['ppl']) is not None]
ensemble_accs = [get_cascade_acc(r['ensemble']) for r in all_results.values() if get_cascade_acc(r['ensemble']) is not None]

lora_avg = f"{sum(lora_accs)/len(lora_accs):.4f}" if lora_accs else "-"
mlp_avg = f"{sum(mlp_accs)/len(mlp_accs):.4f}" if mlp_accs else "-"
ppl_avg = f"{sum(ppl_accs)/len(ppl_accs):.4f}" if ppl_accs else "-"
ensemble_avg = f"{sum(ensemble_accs)/len(ensemble_accs):.4f}" if ensemble_accs else "-"

print(f"{'AVERAGE':<25} {lora_avg:>12} {mlp_avg:>12} {ppl_avg:>12} {ensemble_avg:>12}")
print("=" * 115)

# Per-stage baseline accuracy
print()
print("=" * 100)
print("                         PER-STAGE BASELINE ACCURACY (Ground Truth)")
print("=" * 100)
print(f"{'Subset':<25} {'T0':>12} {'T100':>12} {'T500':>12} {'T1000':>12}")
print("-" * 100)

for subset in all_subsets:
    for model_type in ['lora', 'mlp', 'ppl']:
        results = all_results[subset][model_type]
        baseline = get_baseline(results)
        if baseline:
            t0 = f"{baseline.get('0', baseline.get('T0', 0)):.4f}"
            t100 = f"{baseline.get('100', baseline.get('T100', 0)):.4f}"
            t500 = f"{baseline.get('500', baseline.get('T500', 0)):.4f}"
            t1000 = f"{baseline.get('1000', baseline.get('T1000', 0)):.4f}"
            print(f"{subset:<25} {t0:>12} {t100:>12} {t500:>12} {t1000:>12}")
            break
    else:
        print(f"{subset:<25} {'-':>12} {'-':>12} {'-':>12} {'-':>12}")

print("=" * 100)

print()
print("Done! Results saved in:")
if train_subset == "all":
    print("  MLP (unified): all/mlp_model/results_*.json")
print("  LoRA:     {subset}/lora_model/results.json")
print("  MLP:      {subset}/mlp_model/results.json")
print("  PPL:      {subset}/ppl_model/results.json")
print("  Ensemble: {subset}/ensemble_model/results.json")
EOF
