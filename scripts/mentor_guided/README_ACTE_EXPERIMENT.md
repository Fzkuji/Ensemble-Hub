# ACT-E 实验指南

本目录包含运行 ACT-E (Adaptive Control of LLM Thinking Ensemble) 实验所需的所有脚本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `collect_progressive_data.py` | 收集不同 token 长度的 PPL/Entropy 数据 |
| `sequence_classifier.py` | LSTM/GRU/CNN/MLP/Attention 分类器实现 |
| `run_acte_experiment.py` | 运行交叉验证实验 |

---

## 数据集说明

| 数据集 | 样本数 | 来源 |
|--------|--------|------|
| hendrycks_math | ~12.5k | HuggingFace `EleutherAI/hendrycks_math` (7 subsets, train+test) |
| humaneval | 164 | HuggingFace `openai_humaneval` |

---

## 完整运行命令

### MATH 数据集 (hendrycks_math)

#### Step 1: 收集数据

```bash
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset hendrycks_math \
    --split all \
    --mentor-type local \
    --mentor-model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --parallel \
    --num-workers 2 \
    --mentor-gpus 0,1 \
    --intern-gpus 2,3
```

#### Step 2: 运行实验

```bash
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset hendrycks_math \
    --mentor DeepSeek-R1-Distill-Qwen-32B \
    --models mlp lstm gru attention cnn \
    --n_folds 5
```

---

### HumanEval 数据集

#### Step 1: 收集数据

```bash
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset humaneval \
    --split all \
    --mentor-type local \
    --mentor-model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --parallel \
    --num-workers 2 \
    --mentor-gpus 0,1 \
    --intern-gpus 2,3
```

#### Step 2: 运行实验

```bash
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset humaneval \
    --mentor DeepSeek-R1-Distill-Qwen-32B \
    --models mlp lstm gru attention cnn \
    --n_folds 5
```

---

## 一键运行脚本

### run_hendrycks_math.sh

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Step 1: Collecting data for hendrycks_math"
echo "=========================================="
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset hendrycks_math \
    --split all \
    --mentor-type local \
    --mentor-model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --parallel \
    --num-workers 2 \
    --mentor-gpus 0,1 \
    --intern-gpus 2,3

echo "=========================================="
echo "Step 2: Running experiments"
echo "=========================================="
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset hendrycks_math \
    --mentor DeepSeek-R1-Distill-Qwen-32B \
    --models mlp lstm gru attention cnn \
    --n_folds 5

echo "=========================================="
echo "Done! Results in data/acte_experiments/results/"
echo "=========================================="
```

### run_humaneval.sh

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Step 1: Collecting data for humaneval"
echo "=========================================="
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset humaneval \
    --split all \
    --mentor-type local \
    --mentor-model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --parallel \
    --num-workers 2 \
    --mentor-gpus 0,1 \
    --intern-gpus 2,3

echo "=========================================="
echo "Step 2: Running experiments"
echo "=========================================="
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset humaneval \
    --mentor DeepSeek-R1-Distill-Qwen-32B \
    --models mlp lstm gru attention cnn \
    --n_folds 5

echo "=========================================="
echo "Done! Results in data/acte_experiments/results/"
echo "=========================================="
```

---

## 输出文件位置

```
data/acte_experiments/
├── collected/
│   ├── hendrycks_math_all_DeepSeek-R1-Distill-Qwen-32B/
│   │   ├── hendrycks_math_all_tokens0.json
│   │   ├── hendrycks_math_all_tokens100.json
│   │   ├── hendrycks_math_all_tokens500.json
│   │   ├── hendrycks_math_all_tokens1000.json
│   │   └── hendrycks_math_all_mentor_only.json
│   └── humaneval_all_DeepSeek-R1-Distill-Qwen-32B/
│       └── ...
└── results/
    ├── hendrycks_math_DeepSeek-R1-Distill-Qwen-32B_mlp_cv_results.json
    ├── hendrycks_math_DeepSeek-R1-Distill-Qwen-32B_lstm_cv_results.json
    ├── humaneval_DeepSeek-R1-Distill-Qwen-32B_mlp_cv_results.json
    └── ...
```

---

## 实验结果示例

```
=== Baselines on All Data ===
Intern-only: Acc=0.6940, Mentor=0.0
Fixed-100:   Acc=0.7440, Mentor=100.0
Fixed-500:   Acc=0.7660, Mentor=496.1
Fixed-1000:  Acc=0.7940, Mentor=970.6
Mentor-only: Acc=0.7820, Mentor=2366.8
Oracle:      Acc=0.9000, Mentor=93.4

=== MLP Cross-Validation Results ===
Threshold  Mean Acc     Std Acc    Avg Mentor   Distribution
--------------------------------------------------------------------------------
0.30       0.7440       0.0265     100.0        100:500
0.50       0.7440       0.0258     133.0        100:470, 500:20, 1000:10
0.60       0.7940       0.0258     970.6        1000:500
```

---

## GPU 配置建议

| 配置 | Mentor GPU | Intern GPU | 说明 |
|------|------------|------------|------|
| 单卡 A100 80G | cuda:0 | cuda:0 | 顺序执行 |
| 双卡 A100 | cuda:0 | cuda:1 | 并行执行 |
| 4卡并行 | 0,1 | 2,3 | 2 workers |

---

## 常见问题

**Q: 收集数据时 OOM？**
- 减少 `--num-workers`
- 使用单卡模式（去掉 `--parallel`）

**Q: 分类器全部预测同一类？**
- 检查 label 分布是否平衡
- 调整分类阈值（0.3-0.7）

**Q: 准确率低于 baseline？**
- PPL/Entropy 特征区分度可能不够
- 尝试不同的分类器（LSTM 通常比 MLP 好）
