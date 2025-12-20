#!/bin/bash
# Compare LoRA vs MLP vs PPL classifier performance
# Usage: ./run_compare_classifier.sh [OPTIONS]
#
# Options:
#   --gpus GPUS       Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   --subset SUBSET   Specific subset to run (default: all subsets)
#   --data-dir DIR    Data directory
#   --skip-lora       Skip LoRA training (if already done)
#   --skip-mlp        Skip MLP training (if already done)
#   --skip-ppl        Skip PPL training (if already done)
#   --check           Only check file status (no training)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
GPUS="0,1,2,3,4,5,6,7"
DATA_DIR="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B"
SUBSET=""
SKIP_LORA=false
SKIP_MLP=false
SKIP_PPL=false
CHECK_ONLY=false

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
        --skip-ppl)
            SKIP_PPL=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
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
echo "Classifier Comparison: LoRA vs MLP vs PPL"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "GPUs: $GPUS (${NUM_GPUS} GPUs)"
echo "Subsets: ${SUBSETS[*]}"
echo "============================================================"

# Helper function to check if model already trained
check_model_exists() {
    local subset=$1
    local model_type=$2
    local result_file="$DATA_DIR/$subset/${model_type}_model/results.json"
    if [ -f "$result_file" ]; then
        return 0  # exists
    else
        return 1  # not exists
    fi
}

# Check-only mode: show file status and results
if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "========== FILE STATUS =========="
    printf "%-25s %-10s %-10s %-10s\n" "Subset" "LoRA" "MLP" "PPL"
    echo "------------------------------------------------------------"
    for subset in "${SUBSETS[@]}"; do
        lora_status="-"
        mlp_status="-"
        ppl_status="-"
        [ -f "$DATA_DIR/$subset/lora_model/results.json" ] && lora_status="OK"
        [ -f "$DATA_DIR/$subset/mlp_model/results.json" ] && mlp_status="OK"
        [ -f "$DATA_DIR/$subset/ppl_model/results.json" ] && ppl_status="OK"
        printf "%-25s %-10s %-10s %-10s\n" "$subset" "$lora_status" "$mlp_status" "$ppl_status"
    done
    echo "============================================================"

    # Show results comparison
    echo ""
    echo "========== CASCADE ACCURACY =========="
    python3 << PYEOF
import json
import os

data_dir = "$DATA_DIR"
subsets = "${SUBSETS[*]}".split()

print(f"{'Subset':<25} {'LoRA':>10} {'MLP':>10} {'PPL':>10} {'Oracle':>10} {'Best':>10}")
print("-" * 75)

for subset in subsets:
    row = {"lora": "-", "mlp": "-", "ppl": "-", "oracle": "-"}
    best_val, best_name = 0, "-"

    for m in ["lora", "mlp", "ppl"]:
        path = f"{data_dir}/{subset}/{m}_model/results.json"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    r = json.load(f)
                acc = r.get('best_cascade_acc', 0)
                row[m] = f"{acc:.4f}"
                if row["oracle"] == "-" and 'oracle_acc' in r:
                    row["oracle"] = f"{r['oracle_acc']:.4f}"
                if acc > best_val:
                    best_val, best_name = acc, m.upper()
            except:
                pass

    print(f"{subset:<25} {row['lora']:>10} {row['mlp']:>10} {row['ppl']:>10} {row['oracle']:>10} {best_name:>10}")

print("-" * 75)
PYEOF
    exit 0
fi

# Train LoRA classifiers
if [ "$SKIP_LORA" = false ]; then
    echo ""
    echo "========== Training LoRA Classifiers =========="
    for subset in "${SUBSETS[@]}"; do
        if check_model_exists "$subset" "lora"; then
            echo ""
            echo ">>> LoRA: $subset [SKIP - already trained]"
        else
            echo ""
            echo ">>> LoRA: $subset"
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 train_lora_classifier.py \
                --ddp --subset $subset --data-dir $DATA_DIR --epochs 3
        fi
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
        if check_model_exists "$subset" "mlp"; then
            echo ""
            echo ">>> MLP: $subset [SKIP - already trained]"
        else
            echo ""
            echo ">>> MLP: $subset"
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29506 train_mlp_classifier.py \
                --ddp --subset $subset --data-dir $DATA_DIR --epochs 10
        fi
    done
else
    echo ""
    echo "========== Skipping MLP (--skip-mlp) =========="
fi

# Train PPL classifiers
if [ "$SKIP_PPL" = false ]; then
    echo ""
    echo "========== Training PPL Classifiers (Entropy-based) =========="
    for subset in "${SUBSETS[@]}"; do
        if check_model_exists "$subset" "ppl"; then
            echo ""
            echo ">>> PPL: $subset [SKIP - already trained]"
        else
            echo ""
            echo ">>> PPL: $subset"
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29507 train_ppl_classifier.py \
                --ddp --subset $subset --data-dir $DATA_DIR
        fi
    done
else
    echo ""
    echo "========== Skipping PPL (--skip-ppl) =========="
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
subsets = "${SUBSETS[*]}".split()

# Token levels for length calculation
TOKEN_LEVELS = [0, 100, 500, 1000]

def get_results(subset, model_type):
    """Get results for a specific subset and model type."""
    result_file = os.path.join(data_dir, subset, f"{model_type}_model", "results.json")
    if os.path.exists(result_file):
        try:
            with open(result_file) as f:
                return json.load(f)
        except:
            pass
    return None

def get_cascade_acc(r):
    """Get cascade accuracy."""
    if not r:
        return None
    return r.get('best_cascade_acc')

def get_oracle(r):
    """Get oracle accuracy."""
    if not r:
        return None
    return r.get('oracle_acc')

def get_baseline(r):
    """Get per-stage baseline."""
    if not r:
        return None
    return r.get('per_stage_baseline_acc')

print()
print("=" * 100)
print("                              CASCADE ACCURACY COMPARISON")
print("=" * 100)
print(f"{'Subset':<25} {'LoRA':>12} {'MLP':>12} {'PPL':>12} {'Oracle':>12} {'Best':>12}")
print("-" * 100)

all_results = {}
for subset in subsets:
    lora = get_results(subset, "lora")
    mlp = get_results(subset, "mlp")
    ppl = get_results(subset, "ppl")

    lora_casc = get_cascade_acc(lora)
    mlp_casc = get_cascade_acc(mlp)
    ppl_casc = get_cascade_acc(ppl)

    lora_acc = f"{lora_casc:.4f}" if lora_casc is not None else "-"
    mlp_acc = f"{mlp_casc:.4f}" if mlp_casc is not None else "-"
    ppl_acc = f"{ppl_casc:.4f}" if ppl_casc is not None else "-"

    # Get oracle from any available result
    oracle_val = get_oracle(lora) or get_oracle(mlp) or get_oracle(ppl)
    oracle = f"{oracle_val:.4f}" if oracle_val is not None else "-"

    # Find best
    accs = []
    if lora_casc is not None:
        accs.append(('LoRA', lora_casc))
    if mlp_casc is not None:
        accs.append(('MLP', mlp_casc))
    if ppl_casc is not None:
        accs.append(('PPL', ppl_casc))

    best = max(accs, key=lambda x: x[1])[0] if accs else "-"

    print(f"{subset:<25} {lora_acc:>12} {mlp_acc:>12} {ppl_acc:>12} {oracle:>12} {best:>12}")

    all_results[subset] = {'lora': lora, 'mlp': mlp, 'ppl': ppl}

print("-" * 100)

# Compute averages
lora_accs = [get_cascade_acc(r['lora']) for r in all_results.values() if get_cascade_acc(r['lora']) is not None]
mlp_accs = [get_cascade_acc(r['mlp']) for r in all_results.values() if get_cascade_acc(r['mlp']) is not None]
ppl_accs = [get_cascade_acc(r['ppl']) for r in all_results.values() if get_cascade_acc(r['ppl']) is not None]

lora_avg = f"{sum(lora_accs)/len(lora_accs):.4f}" if lora_accs else "-"
mlp_avg = f"{sum(mlp_accs)/len(mlp_accs):.4f}" if mlp_accs else "-"
ppl_avg = f"{sum(ppl_accs)/len(ppl_accs):.4f}" if ppl_accs else "-"

print(f"{'AVERAGE':<25} {lora_avg:>12} {mlp_avg:>12} {ppl_avg:>12}")
print("=" * 100)

# Per-stage AUC comparison
print()
print("=" * 100)
print("                              PER-STAGE AUC COMPARISON")
print("=" * 100)
print(f"{'Subset':<20} {'Model':<8} {'T0':>10} {'T100':>10} {'T500':>10} {'T1000':>10} {'Avg':>10}")
print("-" * 100)

for subset in subsets:
    for model_type, model_name in [('lora', 'LoRA'), ('mlp', 'MLP'), ('ppl', 'PPL')]:
        results = all_results[subset][model_type]
        if results and 'per_stage_auc' in results:
            auc = results['per_stage_auc']
            t0 = f"{auc.get('0', auc.get('T0', 0)):.4f}"
            t100 = f"{auc.get('100', auc.get('T100', 0)):.4f}"
            t500 = f"{auc.get('500', auc.get('T500', 0)):.4f}"
            t1000 = f"{auc.get('1000', auc.get('T1000', 0)):.4f}"
            avg_auc = sum([auc.get(str(t), auc.get(f'T{t}', 0)) for t in TOKEN_LEVELS]) / 4
            print(f"{subset:<20} {model_name:<8} {t0:>10} {t100:>10} {t500:>10} {t1000:>10} {avg_auc:>10.4f}")
        else:
            print(f"{subset:<20} {model_name:<8} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
    print("-" * 100)

# Best thresholds comparison
print()
print("=" * 100)
print("                              BEST THRESHOLDS")
print("=" * 100)
print(f"{'Subset':<20} {'Model':<8} {'T0':>10} {'T100':>10} {'T500':>10} {'T1000':>10}")
print("-" * 100)

for subset in subsets:
    for model_type, model_name in [('lora', 'LoRA'), ('mlp', 'MLP'), ('ppl', 'PPL')]:
        results = all_results[subset][model_type]
        if results and 'best_thresholds' in results:
            th = results['best_thresholds']
            print(f"{subset:<20} {model_name:<8} {th[0]:>10.2f} {th[1]:>10.2f} {th[2]:>10.2f} {th[3]:>10.2f}")
        else:
            print(f"{subset:<20} {model_name:<8} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
    print("-" * 100)

# Per-stage baseline accuracy (what % of samples are correct at each stage)
print()
print("=" * 100)
print("                         PER-STAGE BASELINE ACCURACY (Ground Truth)")
print("=" * 100)
print(f"{'Subset':<25} {'T0':>12} {'T100':>12} {'T500':>12} {'T1000':>12}")
print("-" * 100)

for subset in subsets:
    # Try to get from any model's results
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
print("  LoRA: {subset}/lora_model/results.json")
print("  MLP:  {subset}/mlp_model/results.json")
print("  PPL:  {subset}/ppl_model/results.json")
EOF
