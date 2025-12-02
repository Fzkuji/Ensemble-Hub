#!/bin/bash
#
# ACT-E 完整实验脚本
# 运行此脚本后，将得到所有实验结果，可直接用于论文 Table
#
# 使用方法:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# 预计运行时间:
#   - 同构模型 (DS-32B + DS-7B): ~4-6 小时
#   - 异构模型 (GPT + DS-7B): ~1-2 小时 (取决于 API 速度)
#

set -e  # 遇到错误立即退出

# ============================
# 配置部分
# ============================

# 项目根目录
PROJECT_ROOT="/home/fzkuji/PycharmProjects/Ensemble-Hub"
cd "$PROJECT_ROOT"

# 模型配置
MENTOR_MODEL_LOCAL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
INTERN_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
API_MODEL="gpt-4o"  # 通过 OpenRouter

# OpenRouter API Key
export OPENROUTER_API_KEY="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"

# 数据目录
DATA_DIR="$PROJECT_ROOT/data/acte_experiments"
RESULTS_DIR="$PROJECT_ROOT/data/acte_experiments/results"
COLLECTED_DIR="$PROJECT_ROOT/data/acte_experiments/collected"

# 创建目录
mkdir -p "$RESULTS_DIR"
mkdir -p "$COLLECTED_DIR"

# 日志文件
LOG_FILE="$RESULTS_DIR/experiment_$(date +%Y%m%d_%H%M%S).log"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======================================"
log "ACT-E 实验开始"
log "======================================"

# ============================
# Step 0: 准备数据集
# ============================
log ""
log "Step 0: 准备数据集..."

if [ ! -f "$DATA_DIR/math500/train.json" ]; then
    log "准备 MATH-500 和 HumanEval 数据集..."
    python scripts/mentor_guided/prepare_datasets.py 2>&1 | tee -a "$LOG_FILE"
else
    log "数据集已存在，跳过准备步骤"
fi

# ============================
# Step 1: 同构模型实验 (DS-32B + DS-7B)
# ============================
log ""
log "======================================"
log "Step 1: 同构模型实验 (DS-32B + DS-7B)"
log "======================================"

# 1.1 MATH-500 数据收集
log ""
log "1.1 收集 MATH-500 数据 (同构模型)..."

for SPLIT in train test; do
    OUTPUT_DIR="$COLLECTED_DIR/math500_${SPLIT}_DeepSeek-R1-Distill-Qwen-32B"

    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        log "  $SPLIT 数据已存在，跳过"
    else
        log "  收集 $SPLIT 数据..."
        python scripts/mentor_guided/collect_progressive_data.py \
            --dataset math500 \
            --split $SPLIT \
            --mentor-type local \
            --mentor-model "$MENTOR_MODEL_LOCAL" \
            --intern-model "$INTERN_MODEL" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi
done

# 1.2 HumanEval 数据收集
log ""
log "1.2 收集 HumanEval 数据 (同构模型)..."

for SPLIT in train test; do
    OUTPUT_DIR="$COLLECTED_DIR/humaneval_${SPLIT}_DeepSeek-R1-Distill-Qwen-32B"

    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        log "  $SPLIT 数据已存在，跳过"
    else
        log "  收集 $SPLIT 数据..."
        python scripts/mentor_guided/collect_progressive_data.py \
            --dataset humaneval \
            --split $SPLIT \
            --mentor-type local \
            --mentor-model "$MENTOR_MODEL_LOCAL" \
            --intern-model "$INTERN_MODEL" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi
done

# ============================
# Step 2: 异构模型实验 (GPT + DS-7B)
# ============================
log ""
log "======================================"
log "Step 2: 异构模型实验 (GPT-4o + DS-7B)"
log "======================================"

# 2.1 MATH-500 数据收集 (API)
log ""
log "2.1 收集 MATH-500 数据 (异构模型)..."

for SPLIT in train test; do
    OUTPUT_DIR="$COLLECTED_DIR/math500_${SPLIT}_${API_MODEL}"

    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        log "  $SPLIT 数据已存在，跳过"
    else
        log "  收集 $SPLIT 数据 (通过 API)..."
        python scripts/mentor_guided/collect_progressive_data.py \
            --dataset math500 \
            --split $SPLIT \
            --mentor-type api \
            --api-model "$API_MODEL" \
            --intern-model "$INTERN_MODEL" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi
done

# 2.2 HumanEval 数据收集 (API)
log ""
log "2.2 收集 HumanEval 数据 (异构模型)..."

for SPLIT in train test; do
    OUTPUT_DIR="$COLLECTED_DIR/humaneval_${SPLIT}_${API_MODEL}"

    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        log "  $SPLIT 数据已存在，跳过"
    else
        log "  收集 $SPLIT 数据 (通过 API)..."
        python scripts/mentor_guided/collect_progressive_data.py \
            --dataset humaneval \
            --split $SPLIT \
            --mentor-type api \
            --api-model "$API_MODEL" \
            --intern-model "$INTERN_MODEL" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi
done

# ============================
# Step 3: 训练分类器并评估
# ============================
log ""
log "======================================"
log "Step 3: 训练分类器并评估"
log "======================================"

# 3.1 同构模型 - MATH-500
log ""
log "3.1 训练 MATH-500 分类器 (同构模型)..."
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset math500 \
    --mentor DeepSeek-R1-Distill-Qwen-32B \
    --models lstm gru mlp \
    2>&1 | tee -a "$LOG_FILE"

# 3.2 同构模型 - HumanEval
log ""
log "3.2 训练 HumanEval 分类器 (同构模型)..."
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset humaneval \
    --mentor DeepSeek-R1-Distill-Qwen-32B \
    --models lstm gru mlp \
    2>&1 | tee -a "$LOG_FILE"

# 3.3 异构模型 - MATH-500
log ""
log "3.3 训练 MATH-500 分类器 (异构模型)..."
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset math500 \
    --mentor "$API_MODEL" \
    --models lstm gru \
    2>&1 | tee -a "$LOG_FILE"

# 3.4 异构模型 - HumanEval
log ""
log "3.4 训练 HumanEval 分类器 (异构模型)..."
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset humaneval \
    --mentor "$API_MODEL" \
    --models lstm gru \
    2>&1 | tee -a "$LOG_FILE"

# ============================
# Step 4: 计算基线结果 (固定长度 Progressive)
# ============================
log ""
log "======================================"
log "Step 4: 计算基线结果"
log "======================================"

python << 'BASELINE_EOF'
import json
import os
import numpy as np
from glob import glob

collected_dir = "/home/fzkuji/PycharmProjects/Ensemble-Hub/data/acte_experiments/collected"
results_dir = "/home/fzkuji/PycharmProjects/Ensemble-Hub/data/acte_experiments/results"
os.makedirs(results_dir, exist_ok=True)

def calculate_tflops(mentor_len, intern_len, mentor_params=32e9, intern_params=7e9):
    mentor_flops = 2 * mentor_params * mentor_len
    intern_flops = 2 * intern_params * intern_len
    return (mentor_flops + intern_flops) / 1e12

baseline_results = []

# Process each collected dataset
for dataset_dir in glob(os.path.join(collected_dir, "*")):
    if not os.path.isdir(dataset_dir):
        continue

    dataset_name = os.path.basename(dataset_dir)
    print(f"\nProcessing: {dataset_name}")

    # Load data for each token level
    for tokens in [0, 100, 500, 1000]:
        # Find the file
        files = glob(os.path.join(dataset_dir, f"*_tokens{tokens}.json"))
        if not files:
            continue

        with open(files[0], 'r') as f:
            data = json.load(f)

        if not data:
            continue

        # Calculate metrics
        accuracy = np.mean([d['is_correct'] for d in data])
        avg_mentor_len = np.mean([d.get('mentor_length', tokens) for d in data])
        avg_intern_len = np.mean([d.get('intern_length', 0) for d in data])
        tflops = calculate_tflops(avg_mentor_len, avg_intern_len)

        result = {
            'dataset': dataset_name,
            'method': f'Progressive-{tokens}' if tokens > 0 else 'Intern Only',
            'mentor_tokens': tokens,
            'accuracy': accuracy,
            'avg_mentor_len': avg_mentor_len,
            'avg_intern_len': avg_intern_len,
            'tflops': tflops,
            'num_samples': len(data),
        }
        baseline_results.append(result)
        print(f"  {tokens} tokens: Acc={accuracy:.4f}, Mentor={avg_mentor_len:.1f}, Intern={avg_intern_len:.1f}")

# Save baseline results
baseline_file = os.path.join(results_dir, "baseline_results.json")
with open(baseline_file, 'w') as f:
    json.dump(baseline_results, f, indent=2)
print(f"\nBaseline results saved to {baseline_file}")
BASELINE_EOF

# ============================
# Step 5: 汇总所有结果
# ============================
log ""
log "======================================"
log "Step 5: 汇总所有结果"
log "======================================"

# 创建结果汇总脚本
python << 'EOF'
import json
import os
from glob import glob

results_dir = "/home/fzkuji/PycharmProjects/Ensemble-Hub/data/acte_experiments/results"

print("\n" + "="*80)
print("ACT-E 实验结果汇总 (可直接用于论文)")
print("="*80)

# 加载基线结果
baseline_file = os.path.join(results_dir, "baseline_results.json")
baseline_results = []
if os.path.exists(baseline_file):
    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)

# 加载分类器结果
classifier_results = {}
for f in glob(os.path.join(results_dir, "*_results.json")):
    if "baseline" in f:
        continue
    name = os.path.basename(f).replace("_results.json", "")
    with open(f, 'r') as fp:
        classifier_results[name] = json.load(fp)

# ==========================================
# Table A: 完整实验结果 (主表)
# ==========================================
print("\n" + "="*80)
print("### Table A: 完整实验结果")
print("="*80)
print("| Dataset | Mentor | Intern | Method | Mentor Len | Intern Len | Accuracy | TFLOPs |")
print("|---------|--------|--------|--------|------------|------------|----------|--------|")

# 按数据集和方法整理
table_a_data = []

# 添加基线结果
for row in baseline_results:
    dataset_name = row['dataset']
    # 解析数据集名称: math500_train_DeepSeek-R1-Distill-Qwen-32B
    parts = dataset_name.split('_')
    dataset = parts[0]  # math500 or humaneval
    split = parts[1] if len(parts) > 1 else "test"
    mentor = parts[2] if len(parts) > 2 else "DS-32B"

    if split != "test":  # 只显示测试集结果
        continue

    if "DeepSeek" in mentor:
        mentor_short = "DS-32B"
    elif "gpt" in mentor.lower():
        mentor_short = "GPT-4o"
    else:
        mentor_short = mentor[:10]

    table_a_data.append({
        'dataset': dataset.upper(),
        'mentor': mentor_short,
        'method': row['method'],
        'mentor_len': row['avg_mentor_len'],
        'intern_len': row['avg_intern_len'],
        'accuracy': row['accuracy'],
        'tflops': row['tflops'],
    })

# 添加分类器结果
for exp_name, results in classifier_results.items():
    parts = exp_name.split('_')
    dataset = parts[0]
    mentor = parts[1] if len(parts) > 1 else "DS-32B"

    if "DeepSeek" in mentor:
        mentor_short = "DS-32B"
    elif "gpt" in mentor.lower():
        mentor_short = "GPT-4o"
    else:
        mentor_short = mentor[:10]

    for row in results:
        table_a_data.append({
            'dataset': dataset.upper(),
            'mentor': mentor_short,
            'method': f"ACT-E ({row['model']})",
            'mentor_len': row['avg_mentor_len'],
            'intern_len': row['avg_intern_len'],
            'accuracy': row['accuracy'],
            'tflops': row['tflops'],
        })

# 排序并打印
table_a_data.sort(key=lambda x: (x['dataset'], x['mentor'], x['method']))
for row in table_a_data:
    print(f"| {row['dataset']:<8} | {row['mentor']:<6} | DS-7B | {row['method']:<16} | "
          f"{row['mentor_len']:>10.1f} | {row['intern_len']:>10.1f} | "
          f"{row['accuracy']:>8.4f} | {row['tflops']:>6.2f} |")

# ==========================================
# Table B: 判断模型对比 (Table 2)
# ==========================================
print("\n" + "="*80)
print("### Table B: 判断模型对比")
print("="*80)
print("| Dataset | Mentor | Classifier | Accuracy | Avg Mentor Len | TFLOPs |")
print("|---------|--------|------------|----------|----------------|--------|")

for exp_name, results in sorted(classifier_results.items()):
    parts = exp_name.split('_')
    dataset = parts[0].upper()
    mentor = parts[1] if len(parts) > 1 else "DS-32B"

    if "DeepSeek" in mentor:
        mentor_short = "DS-32B"
    elif "gpt" in mentor.lower():
        mentor_short = "GPT-4o"
    else:
        mentor_short = mentor[:10]

    for row in results:
        print(f"| {dataset:<7} | {mentor_short:<6} | {row['model']:<10} | "
              f"{row['accuracy']:>8.4f} | {row['avg_mentor_len']:>14.1f} | {row['tflops']:>6.2f} |")

# ==========================================
# 保存 LaTeX 格式
# ==========================================
latex_file = os.path.join(results_dir, "results_latex.tex")
with open(latex_file, 'w') as f:
    f.write("% Table A: 完整实验结果\n")
    f.write("\\begin{table*}[t]\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{llllrrrr}\n")
    f.write("\\toprule\n")
    f.write("Dataset & Mentor & Intern & Method & Mentor Len & Intern Len & Accuracy & TFLOPs \\\\\n")
    f.write("\\midrule\n")
    for row in table_a_data:
        f.write(f"{row['dataset']} & {row['mentor']} & DS-7B & {row['method']} & "
                f"{row['mentor_len']:.1f} & {row['intern_len']:.1f} & "
                f"{row['accuracy']:.4f} & {row['tflops']:.2f} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{Complete experimental results}\n")
    f.write("\\end{table*}\n")

print(f"\nLaTeX 表格已保存到: {latex_file}")
print("\n" + "="*80)
print("实验完成！")
print("="*80)
EOF

log ""
log "======================================"
log "实验完成！"
log "======================================"
log "结果目录: $RESULTS_DIR"
log "日志文件: $LOG_FILE"
log ""
log "收集的数据:"
ls -la "$COLLECTED_DIR" 2>/dev/null || echo "无数据"
log ""
log "结果文件:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "无结果文件"

echo ""
echo "======================================"
echo "实验完成！请查看上方的结果表格"
echo "======================================"
