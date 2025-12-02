# ACT-E 实验指南

本目录包含运行 ACT-E (Adaptive Control of LLM Thinking Ensemble) 论文补充实验所需的所有脚本。

## 实验目标

回应审稿人关于：
1. 只用同一模型家族（DeepSeek-R1-Distill-Qwen）的问题
2. 只用 MATH 数据集的问题

## 文件说明

| 文件 | 说明 |
|------|------|
| `prepare_datasets.py` | 准备 MATH-500 和 HumanEval 数据集 |
| `openrouter_client.py` | OpenRouter API 客户端，用于调用 GPT-5 等模型 |
| `sequence_classifier.py` | LSTM/GRU/CNN/MLP 分类器实现 |
| `collect_progressive_data.py` | 收集不同 token 长度的 PPL/Entropy 数据 |
| `run_acte_experiment.py` | 运行完整实验流程 |

## 实验步骤

### Step 1: 准备数据集

```bash
python scripts/mentor_guided/prepare_datasets.py
```

这将创建：
- `data/acte_experiments/math500/train.json` (400 samples)
- `data/acte_experiments/math500/test.json` (100 samples)
- `data/acte_experiments/humaneval/train.json` (130 samples)
- `data/acte_experiments/humaneval/test.json` (34 samples)

### Step 2: 收集 Progressive 数据

#### 同构模型 (DeepSeek-32B + DeepSeek-7B)

```bash
# 训练集
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset math500 \
    --split train \
    --mentor-type local \
    --mentor-model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# 测试集
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset math500 \
    --split test \
    --mentor-type local \
    --mentor-model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

#### 异构模型 (GPT-4o + DeepSeek-7B)

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# 训练集
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset math500 \
    --split train \
    --mentor-type api \
    --api-model gpt-4o \
    --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# 测试集
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset math500 \
    --split test \
    --mentor-type api \
    --api-model gpt-4o \
    --intern-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

### Step 3: 训练分类器并评估

```bash
# 同构模型实验
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset math500 \
    --mentor DeepSeek-R1-Distill-Qwen-32B \
    --models lstm gru mlp

# 异构模型实验
python scripts/mentor_guided/run_acte_experiment.py \
    --dataset math500 \
    --mentor gpt-4o \
    --models lstm gru
```

## 预期输出

### Table A: 完整实验结果

| Dataset | Mentor | Intern | Method | Mentor Len | Intern Len | Accuracy | TFLOPs |
|---------|--------|--------|--------|------------|------------|----------|--------|
| MATH-500 | - | DS-7B | Intern Only | 0 | ~2000 | ~45% | 0.03 |
| MATH-500 | DS-32B | DS-7B | Progressive-100 | 100 | ~1800 | ~50% | 0.03 |
| MATH-500 | DS-32B | DS-7B | Progressive-500 | 500 | ~1500 | ~55% | 0.05 |
| MATH-500 | DS-32B | DS-7B | Progressive-1000 | 1000 | ~1200 | ~60% | 0.09 |
| MATH-500 | DS-32B | DS-7B | ACT-E (LSTM) | ~400 | ~1600 | ~58% | 0.05 |
| MATH-500 | GPT-4o | DS-7B | ACT-E (LSTM) | ~400 | ~1600 | ~62% | - |

### Table B: 判断模型对比

| Classifier | Accuracy | Avg Length | Cost |
|------------|----------|------------|------|
| Oracle | 60% | 300 | 0.04 |
| MLP (17-dim) | 55% | 450 | 0.05 |
| LSTM | 58% | 400 | 0.05 |
| GRU | 57% | 420 | 0.05 |

## API 配置

OpenRouter API 支持的模型：
- `gpt-5`: openai/gpt-5
- `gpt-4o`: openai/gpt-4o
- `claude-3.5-sonnet`: anthropic/claude-3.5-sonnet
- `deepseek-v3`: deepseek/deepseek-chat

设置 API key:
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

## 注意事项

1. **GPU 内存**: 本地模型需要足够的 GPU 内存
   - 32B 模型: ~64GB
   - 7B 模型: ~16GB
   - 建议使用 A100 或类似显卡

2. **API 成本**: GPT-4o API 调用有成本，建议先在小数据集测试

3. **时间估计**:
   - 数据收集 (本地): ~2-4 小时/数据集
   - 数据收集 (API): ~30 分钟/数据集 (取决于 rate limit)
   - 分类器训练: ~10 分钟
