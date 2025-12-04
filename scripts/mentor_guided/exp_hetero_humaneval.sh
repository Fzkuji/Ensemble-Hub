#!/bin/bash
#
# 实验脚本: 异构模型 + HumanEval
# Mentor: GPT-4o (via OpenRouter API)
# Intern: DeepSeek-R1-Distill-Qwen-7B
# Dataset: HumanEval
#

set -e

# ============================
# 配置
# ============================
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# OpenRouter API Key (请确保已设置)
if [ -z "$OPENROUTER_API_KEY" ]; then
    export OPENROUTER_API_KEY="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"
fi

API_MODEL="gpt-4o"
MENTOR_NAME="gpt-4o"
INTERN_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET="humaneval"

DATA_DIR="$PROJECT_ROOT/data/acte_experiments"
COLLECTED_DIR="$DATA_DIR/collected"
RESULTS_DIR="$DATA_DIR/results"

mkdir -p "$COLLECTED_DIR" "$RESULTS_DIR"

echo "======================================"
echo "实验: 异构模型 + HumanEval"
echo "Mentor: $API_MODEL (API)"
echo "Intern: DeepSeek-R1-Distill-Qwen-7B"
echo "======================================"

# ============================
# Step 1: 准备数据集
# ============================
echo ""
echo "[Step 1] 准备数据集..."
if [ ! -f "$DATA_DIR/$DATASET/train.json" ]; then
    python scripts/mentor_guided/prepare_datasets.py
else
    echo "数据集已存在，跳过"
fi

# ============================
# Step 2: 收集数据 (通过 API)
# ============================
echo ""
echo "[Step 2] 收集数据 (通过 API)..."

for SPLIT in train test; do
    OUTPUT_DIR="$COLLECTED_DIR/${DATASET}_${SPLIT}_${MENTOR_NAME}"

    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        echo "  $SPLIT 数据已存在，跳过"
    else
        echo "  收集 $SPLIT 数据..."
        python scripts/mentor_guided/collect_progressive_data.py \
            --dataset $DATASET \
            --split $SPLIT \
            --mentor-type api \
            --api-model "$API_MODEL" \
            --intern-model "$INTERN_MODEL" \
            --output-dir "$OUTPUT_DIR"
    fi
done

# ============================
# Step 3: 训练分类器
# ============================
echo ""
echo "[Step 3] 训练分类器..."
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset $DATASET \
    --mentor "$MENTOR_NAME" \
    --models lstm gru mlp

# ============================
# Step 4: 计算基线并输出最终结果
# ============================
echo ""
echo "[Step 4] 计算并输出最终结果..."

python << EOF
import json
import os
import numpy as np

collected_dir = "$COLLECTED_DIR"
results_dir = "$RESULTS_DIR"
dataset = "$DATASET"
mentor = "$MENTOR_NAME"

# 对于 API 模型，使用估计的参数量 (GPT-4o 约 200B)
def calculate_tflops(mentor_len, intern_len, mentor_params=200e9, intern_params=7e9):
    mentor_flops = 2 * mentor_params * mentor_len
    intern_flops = 2 * intern_params * intern_len
    return (mentor_flops + intern_flops) / 1e12

print("\n" + "="*80)
print(f"实验结果: {dataset.upper()} + {mentor}")
print("="*80)

# 加载测试集基线结果
test_dir = os.path.join(collected_dir, f"{dataset}_test_{mentor}")
baseline_results = []

print("\n### 基线结果 (固定策略)")
print("-"*70)
print(f"{'策略':<20} {'准确率':<10} {'Mentor长度':<12} {'Intern长度':<12} {'TFLOPs':<10}")
print("-"*70)

for tokens in [0, 100, 500, 1000]:
    file_path = os.path.join(test_dir, f"{dataset}_test_tokens{tokens}.json")
    if not os.path.exists(file_path):
        continue

    with open(file_path, 'r') as f:
        data = json.load(f)

    if not data:
        continue

    accuracy = np.mean([d['is_correct'] for d in data])
    avg_mentor = np.mean([d.get('mentor_length', tokens) for d in data])
    avg_intern = np.mean([d.get('intern_length', 0) for d in data])
    tflops = calculate_tflops(avg_mentor, avg_intern)

    method = f"Progressive-{tokens}" if tokens > 0 else "Intern Only"
    print(f"{method:<20} {accuracy:<10.4f} {avg_mentor:<12.1f} {avg_intern:<12.1f} {tflops:<10.2f}")

    baseline_results.append({
        'method': method,
        'tokens': tokens,
        'accuracy': accuracy,
        'mentor_len': avg_mentor,
        'intern_len': avg_intern,
        'tflops': tflops,
    })

# 加载分类器结果
classifier_file = os.path.join(results_dir, f"{dataset}_{mentor}_results.json")
classifier_results = []
if os.path.exists(classifier_file):
    with open(classifier_file, 'r') as f:
        classifier_results = json.load(f)

print("\n### ACT-E 自适应策略结果")
print("-"*70)
print(f"{'分类器':<20} {'准确率':<10} {'Mentor长度':<12} {'Intern长度':<12} {'TFLOPs':<10}")
print("-"*70)

best_result = None
for row in classifier_results:
    print(f"ACT-E ({row['model']:<5}){'':<8} {row['accuracy']:<10.4f} {row['avg_mentor_len']:<12.1f} {row['avg_intern_len']:<12.1f} {row['tflops']:<10.2f}")
    if row['model'] == 'MLP':
        best_result = row

# 输出最优解 (MLP)
if best_result:
    print("\n" + "="*80)
    print("最优解 (MLP 分类器)")
    print("="*80)
    print(f"  准确率: {best_result['accuracy']:.4f}")
    print(f"  平均 Mentor 长度: {best_result['avg_mentor_len']:.1f}")
    print(f"  平均 Intern 长度: {best_result['avg_intern_len']:.1f}")
    print(f"  计算成本 (TFLOPs): {best_result['tflops']:.2f}")

    # 对比最好的固定策略
    if baseline_results:
        best_baseline = max(baseline_results, key=lambda x: x['accuracy'])
        print(f"\n  对比最佳固定策略 ({best_baseline['method']}):")
        print(f"    准确率提升: {(best_result['accuracy'] - best_baseline['accuracy'])*100:+.2f}%")
        if best_baseline['tflops'] > 0:
            print(f"    TFLOPs 节省: {(best_baseline['tflops'] - best_result['tflops']):.2f} ({(1 - best_result['tflops']/best_baseline['tflops'])*100:.1f}%)")

# 保存完整结果
final_results = {
    'experiment': f'{dataset}_{mentor}',
    'baseline': baseline_results,
    'classifier': classifier_results,
    'best_mlp': best_result,
}
output_file = os.path.join(results_dir, f"{dataset}_{mentor}_final.json")
with open(output_file, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\n结果已保存到: {output_file}")
EOF

echo ""
echo "======================================"
echo "实验完成: 异构模型 + HumanEval"
echo "======================================"
