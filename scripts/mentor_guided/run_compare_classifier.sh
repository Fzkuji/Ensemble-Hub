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
        CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29506 train_mlp_classifier.py \
            --ddp --subset $subset --data-dir $DATA_DIR --epochs 10
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
        echo ""
        echo ">>> PPL: $subset"
        CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS --master_port=29507 train_ppl_classifier.py \
            --ddp --subset $subset --data-dir $DATA_DIR
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

def compute_avg_tokens(results):
    """Compute average tokens used based on thresholds and stage distribution."""
    if not results or 'best_thresholds' not in results:
        return None
    # This is an approximation - actual would need per-sample data
    # Just return the cascade accuracy for now, length needs per-sample data
    return None

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

    lora_acc = f"{lora['best_cascade_acc']:.4f}" if lora and 'best_cascade_acc' in lora else "-"
    mlp_acc = f"{mlp['best_cascade_acc']:.4f}" if mlp and 'best_cascade_acc' in mlp else "-"
    ppl_acc = f"{ppl['best_cascade_acc']:.4f}" if ppl and 'best_cascade_acc' in ppl else "-"
    oracle = f"{lora['oracle_acc']:.4f}" if lora and 'oracle_acc' in lora else (
        f"{mlp['oracle_acc']:.4f}" if mlp and 'oracle_acc' in mlp else (
            f"{ppl['oracle_acc']:.4f}" if ppl and 'oracle_acc' in ppl else "-"
        )
    )

    # Find best
    accs = []
    if lora and 'best_cascade_acc' in lora:
        accs.append(('LoRA', lora['best_cascade_acc']))
    if mlp and 'best_cascade_acc' in mlp:
        accs.append(('MLP', mlp['best_cascade_acc']))
    if ppl and 'best_cascade_acc' in ppl:
        accs.append(('PPL', ppl['best_cascade_acc']))

    best = max(accs, key=lambda x: x[1])[0] if accs else "-"

    print(f"{subset:<25} {lora_acc:>12} {mlp_acc:>12} {ppl_acc:>12} {oracle:>12} {best:>12}")

    all_results[subset] = {'lora': lora, 'mlp': mlp, 'ppl': ppl}

print("-" * 100)

# Compute averages
lora_accs = [r['lora']['best_cascade_acc'] for r in all_results.values() if r['lora'] and 'best_cascade_acc' in r['lora']]
mlp_accs = [r['mlp']['best_cascade_acc'] for r in all_results.values() if r['mlp'] and 'best_cascade_acc' in r['mlp']]
ppl_accs = [r['ppl']['best_cascade_acc'] for r in all_results.values() if r['ppl'] and 'best_cascade_acc' in r['ppl']]

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
        if results and 'per_stage_baseline_acc' in results:
            acc = results['per_stage_baseline_acc']
            t0 = f"{acc.get('0', acc.get('T0', 0)):.4f}"
            t100 = f"{acc.get('100', acc.get('T100', 0)):.4f}"
            t500 = f"{acc.get('500', acc.get('T500', 0)):.4f}"
            t1000 = f"{acc.get('1000', acc.get('T1000', 0)):.4f}"
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
