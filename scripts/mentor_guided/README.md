# ACT-E: Adaptive Control of LLM Thinking Ensemble

## 完整实验流程

本文档提供从数据收集到结果对比的完整实验流程。所有命令默认使用 8 GPU 并行。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据收集](#2-数据收集)
3. [数据统计](#3-数据统计)
4. [分类器训练](#4-分类器训练)
   - [4.1 MLP 分类器](#41-mlp-分类器推荐)
   - [4.2 LoRA 分类器](#42-lora-分类器)
   - [4.3 PPL 分类器](#43-ppl-分类器)
   - [4.4 Ensemble 分类器](#44-ensemble-分类器)
5. [分类器比较](#5-分类器比较)
6. [结果汇总](#6-结果汇总)
7. [一键运行](#7-一键运行)

---

## 1. 环境准备

```bash
pip install torch transformers peft datasets scikit-learn tqdm numpy vllm
```

---

## 2. 数据收集

使用 `collect_data_vllm_think.py`，支持 Think 和 Standard 两种模式。

### 2.1 最简单用法（推荐）

脚本会自动检测所有可用 GPU 并智能分配：

```bash
cd /home/fzkuji/PycharmProjects/Ensemble-Hub/scripts/mentor_guided

# 使用不同的 Mentor 和 Intern 模型（自动检测 GPU 并分配）
python collect_data_vllm_think.py \
  --mentor-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --intern-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --split train
```

**自动行为**：
- 自动检测所有可用 GPU（通过 `CUDA_VISIBLE_DEVICES` 或 `nvidia-smi`）
- 自动将 GPU 分成两半：前半给 Mentor，后半给 Intern
- 例如 8 卡机器：Mentor 用 `[0,1,2,3]`，Intern 用 `[4,5,6,7]`
- 自动为不同 token levels 选择最优并行模式

### 2.2 智能 Token Level 处理

脚本会自动根据 token level 类型选择最优的并行策略：

| Token Level | 模式 | 说明 |
|-------------|------|------|
| `-1` (mentor only) | Tensor Parallelism | 所有 GPU 给一个 worker，大模型张量并行 |
| `0` (intern only) | Tensor Parallelism | 所有 GPU 给一个 worker |
| `>0` (both models) | Data Parallelism | 每个 worker 一对 GPU（一个 mentor + 一个 intern） |

这意味着运行 `--token-levels="-1,0,100,500,1000"` 时，会自动分三批处理，无需手动指定。

### 2.3 Think/Standard 模式

```bash
# Think 模式（默认，结构化思考）
python collect_data_vllm_think.py \
  --mentor-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --intern-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --split train

# Standard 模式（无思考）
python collect_data_vllm_think.py \
  --mentor-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --intern-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --split train --no-think
```

### 2.4 手动指定 GPU（可选）

如果需要精确控制 GPU 分配：

```bash
# 方式 1：指定总 GPU 列表（自动分半）
python collect_data_vllm_think.py \
  --mentor-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --intern-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --gpus 0,1,2,3,4,5,6,7 \
  --split train
# 结果：mentor=[0,1,2,3], intern=[4,5,6,7]

# 方式 2：分别指定（完全控制）
python collect_data_vllm_think.py \
  --mentor-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --intern-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --mentor-gpus 0,1,2,3 \
  --intern-gpus 4,5,6,7 \
  --split train
```

### 2.5 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--split` | 数据集划分 | `test` |
| `--gpus` | GPU 列表（可选，自动检测） | 自动检测所有 GPU |
| `--no-think` | 禁用思考（标准 prompt） | - |
| `--model` | 模型名称（legacy，建议使用 --mentor-model 和 --intern-model） | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| `--mentor-model` | Mentor 模型名称（大模型，如 32B） | 同 `--model` |
| `--intern-model` | Intern 模型名称（小模型，如 7B） | 同 `--model` |
| `--mentor-gpus` | Mentor 模型使用的 GPU 列表（可选） | 自动分配 |
| `--intern-gpus` | Intern 模型使用的 GPU 列表（可选） | 自动分配 |
| `--mentor-memory-util` | Mentor 模型 GPU 内存利用率 | `0.5` |
| `--intern-memory-util` | Intern 模型 GPU 内存利用率 | `0.3` |
| `--mentor-max-model-len` | Mentor 模型最大长度（自动优化） | 根据 token levels 自动计算 |
| `--intern-max-model-len` | Intern 模型最大长度 | 同 `--max-model-len` |
| `--batch-size` | 批量大小 | `16` |
| `--token-levels` | Token 级别列表 | `-1,0,100,500,1000` |
| `--max-model-len` | 最大模型长度 | `4096` |

### 2.6 自动优化功能

1. **GPU 自动检测与分配**：无需手动指定，自动检测并分半
2. **Mentor max_model_len 自动优化**：当只生成部分 token 时，自动减少 KV cache 大小
   - 例如 `--token-levels=100,500,1000` 时，mentor 只需 `max(1000) + buffer`，自动设为 2048
3. **智能并行模式选择**：根据 token level 类型自动选择 tensor/data parallelism
4. **增量数据收集**：已存在的数据文件会自动跳过，不会被覆盖

---

## 3. 数据统计

```bash
# 标准数据统计
python compute_stats.py --split test
python compute_stats.py --split train

# Think 数据统计
python compute_stats.py \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B \
    --split test
```

---

## 4. 分类器训练

本项目提供多种分类器方法，推荐使用 MLP 分类器（效果好、训练快）。

### 4.1 MLP 分类器（推荐）

MLP 分类器冻结 LLM backbone，只训练一个轻量 MLP head，训练速度快。

```bash
# 单个子集训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_mlp_classifier.py \
    --ddp --train-subset algebra --eval-subset algebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# 所有子集合并训练，分别测试
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_mlp_classifier.py \
    --ddp --train-subset all --eval-subset all \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B
```

**MLP 参数说明**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train-subset` | 训练子集，`all` 合并所有子集 | - |
| `--eval-subset` | 评估子集，`all` 分别测试每个子集 | 同 train |
| `--pooling` | 池化方式: `last`, `mean`, `mean_logits` | `last` |
| `--dropout` | Dropout 率 | `0.3` |
| `--no-val` | 不划分验证集，全量训练 | - |
| `--fixed-threshold TH` | 使用固定阈值，跳过搜索 | - |
| `--skip-epoch-cascade` | 训练时跳过每 epoch 的 cascade 评估 | - |

### 4.2 LoRA 分类器

LoRA 分类器微调 LLM，效果可能更好但训练更慢。

```bash
# 单个子集训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_lora_classifier.py \
    --ddp --subset algebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B
```

### 4.3 PPL 分类器

PPL 分类器基于 perplexity/entropy 特征，不需要训练神经网络。

```bash
# 单个子集训练
CUDA_VISIBLE_DEVICES=0 python train_ppl_classifier.py \
    --subset algebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B
```

### 4.4 Ensemble 分类器

Ensemble 分类器组合多个分类器的预测，使用 meta-classifier 学习最优组合。

```bash
# MLP + PPL 组合
python train_ensemble_classifier.py \
    --subset algebra --use-mlp \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# LoRA + PPL 组合
python train_ensemble_classifier.py \
    --subset algebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B
```

**Ensemble 参数说明**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-mlp` | 使用 MLP 模型（否则用 LoRA） | - |
| `--no-ppl` | 不使用 PPL 预测 | - |
| `--method` | Meta-classifier: `rf`, `gb`, `lr` | `rf` |

**组合原理**:
```
MLP/LoRA 预测概率 ──┐
                   ├──→ [RandomForest/GradientBoosting/LogisticRegression] ──→ 最终预测
PPL 预测概率 ──────┘
```

---

## 5. 分类器比较

使用 `run_compare_classifier.sh` 一键比较多种分类器。

### 5.1 基本用法

```bash
# 比较所有方法（LoRA、MLP、PPL）
./run_compare_classifier.sh

# 只测试 MLP
./run_compare_classifier.sh --methods mlp

# 测试 MLP 和 PPL
./run_compare_classifier.sh --methods mlp,ppl

# 在所有子集合并训练，分别测试
./run_compare_classifier.sh --train-subset all --eval-subset all --methods mlp

# 只在特定子集测试
./run_compare_classifier.sh --subset algebra --methods lora,mlp
```

### 5.2 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--methods` | 要测试的方法: `lora,mlp,ppl` | `lora,mlp,ppl` |
| `--train-subset` | 训练子集，`all` 合并所有 | 每个子集单独 |
| `--eval-subset` | 评估子集，`all` 分别测试 | 同 train |
| `--subset` | 同时设置 train 和 eval subset | - |
| `--check` | 只查看状态，不训练 | - |
| `--force` | 强制重新训练 | - |
| `--gpus` | GPU 列表 | `0,1,2,3,4,5,6,7` |

### 5.3 查看现有结果

```bash
# 只查看文件状态和已有结果
./run_compare_classifier.sh --check
```

---

## 6. 结果汇总

```bash
# 标准数据结果
python summarize_results.py

# Think 数据结果
python summarize_results.py \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B
```

**输出示例**:

```
====================================================================================================================================================================================
Subset                    N      T0                      T100                    T500                    T1000                   Oracle   Cascade  Gap
                                 acc     m_len   i_len   acc     m_len   i_len   acc     m_len   i_len   acc     m_len   i_len
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
algebra                   1187   0.7321  -       234.5   0.7489  78.8    198.3   0.7623  356.2   156.2   0.7712  712.4   112.4   0.8234   0.7956   +0.0244
counting_and_probability  474    0.6456  -       289.1   0.6624  82.3    245.6   0.6835  401.3   201.3   0.6962  758.7   158.7   0.7511   0.7234   +0.0272
geometry                  479    0.4530  -       312.8   0.4697  85.6    267.4   0.4843  423.1   223.1   0.4968  778.9   178.9   0.5678   0.5312   +0.0344
intermediate_algebra      903    0.3211  -       356.2   0.3389  89.2    301.5   0.3567  456.8   256.8   0.3689  812.1   212.1   0.4234   0.3912   +0.0223
number_theory             540    0.6012  -       267.3   0.6234  76.5    223.8   0.6423  379.5   179.5   0.6589  735.2   135.2   0.7123   0.6823   +0.0234
prealgebra                871    0.7823  -       198.6   0.7956  68.4    156.2   0.8089  312.9   112.9   0.8189  678.5   78.5    0.8567   0.8312   +0.0123
precalculus               546    0.3412  -       378.4   0.3589  92.1    323.7   0.3756  478.1   278.1   0.3889  834.5   234.5   0.4456   0.4123   +0.0234
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TOTAL (weighted)          5000   0.5538  -       278.3   0.5723  81.2    234.6   0.5889  389.2   189.2   0.6012  745.8   145.8   0.6543   0.6234   +0.0245
====================================================================================================================================================================================
```

- `acc`: 准确率
- `m_len`: mentor（大模型）平均生成长度（tokens）
- `i_len`: intern（小模型）平均生成长度（tokens）

---

## 7. 一键运行

使用统一的 `run_pipeline.sh` 脚本，通过参数控制模式：

### 7.1 Think 模式（结构化思考）

```bash
# 默认 8 GPU
./run_pipeline.sh --think

# 指定 GPU
./run_pipeline.sh --think --gpus 0,1,2,3,4,5,6,7
```

**输出目录**: `hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B/`

### 7.2 Standard 模式（无思考）

```bash
# 默认 8 GPU
./run_pipeline.sh --no-think

# 跳过 GPU 0
./run_pipeline.sh --no-think --gpus 1,2,3,4,5,6,7
```

**输出目录**: `hendrycks_math_split_standard_DeepSeek-R1-Distill-Qwen-7B/`

### 7.3 脚本参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--think` | 启用结构化思考模式 | 默认 |
| `--no-think` | 禁用思考（标准 prompt） | - |
| `--gpus` | 指定 GPU 列表 | `0,1,2,3,4,5,6,7` |
| `--model` | 指定模型 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

### 7.4 Pipeline 流程

脚本自动执行以下步骤：
1. **数据收集** - 多 GPU 并行收集 train/test 数据
2. **数据统计** - 计算 Oracle/Baseline 统计
3. **LoRA 训练** - 训练所有子集的分类器
4. **Cascade 评估** - 评估 cascade 性能
5. **结果汇总** - 生成汇总报告

---

## 8. 实验命名规范（Experiment Naming）

使用 `--exp-name` 参数指定自定义实验名称，避免不同模型组合导致文件冲突。

### 推荐命名格式

```
{MODEL_SERIES}_m{MENTOR_SIZE}_i{INTERN_SIZE}
```

### 模型缩写映射表

| 完整模型名 | 缩写 | 备注 |
|------------|------|------|
| **DeepSeek R1 Distill 系列** | | |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | R1_1.5B | Intern |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | R1_7B | Mentor/Intern |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | R1_14B | Mentor/Intern |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | R1_32B | Mentor |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | R1_L8B | Mentor/Intern |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B | R1_L70B | Mentor |
| **Qwen 系列** | | |
| Qwen/Qwen2.5-0.5B-Instruct | Q2.5_0.5B | Intern |
| Qwen/Qwen2.5-1.5B-Instruct | Q2.5_1.5B | Intern |
| Qwen/Qwen2.5-3B-Instruct | Q2.5_3B | Intern |
| Qwen/Qwen2.5-7B-Instruct | Q2.5_7B | Mentor/Intern |
| Qwen/Qwen2.5-14B-Instruct | Q2.5_14B | Mentor |
| Qwen/Qwen2.5-32B-Instruct | Q2.5_32B | Mentor |
| Qwen/Qwen2.5-72B-Instruct | Q2.5_72B | Mentor |
| Qwen/QwQ-32B-Preview | QwQ_32B | Mentor (reasoning) |
| **LLaMA 系列** | | |
| meta-llama/Llama-3.1-8B-Instruct | L3.1_8B | Mentor/Intern |
| meta-llama/Llama-3.1-70B-Instruct | L3.1_70B | Mentor |
| meta-llama/Llama-3.3-70B-Instruct | L3.3_70B | Mentor |
| **Mistral 系列** | | |
| mistralai/Mistral-7B-Instruct-v0.3 | M_7B | Mentor/Intern |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | Mx_8x7B | Mentor |

### 实验名称示例

| Mentor 模型 | Intern 模型 | Exp Name |
|-------------|-------------|----------|
| DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-Qwen-7B | `R1_m32B_i7B` |
| DeepSeek-R1-Distill-Qwen-14B | DeepSeek-R1-Distill-Qwen-1.5B | `R1_m14B_i1.5B` |
| Qwen2.5-72B-Instruct | Qwen2.5-7B-Instruct | `Q2.5_m72B_i7B` |
| Llama-3.1-70B-Instruct | Llama-3.1-8B-Instruct | `L3.1_m70B_i8B` |
| DeepSeek-R1-Distill-Qwen-32B | Qwen2.5-7B-Instruct | `R1m32B_Q2.5i7B` |

### 使用方式

```bash
# 使用 exp-name 参数
python collect_data_vllm_think.py \
    --dataset math500 \
    --exp-name R1_m32B_i7B \
    --gpus 0,1,2,3,4,5,6,7

# 输出目录将会是:
# /mnt/data/.../math500_think_R1_m32B_i7B/
```

---

## 附录

### A. 数据目录结构

```
/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/
├── hendrycks_math_split/                    # 标准数据
│   ├── algebra/
│   │   ├── train/
│   │   │   ├── tokens-1.json               # Mentor only (baseline)
│   │   │   ├── tokens0.json                # Intern only (baseline)
│   │   │   ├── tokens100.json
│   │   │   ├── tokens500.json
│   │   │   └── tokens1000.json
│   │   ├── test/
│   │   │   └── ...
│   │   └── lora_model/                      # 训练后的模型
│   │       ├── best_model.pt
│   │       ├── cascade_eval.json            # 评估结果
│   │       └── results.json
│   ├── counting_and_probability/
│   ├── geometry/
│   ├── intermediate_algebra/
│   ├── number_theory/
│   ├── prealgebra/
│   └── precalculus/
│
└── hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B/   # Think 数据
    └── ... (同上结构)
```

### B. JSON 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | str | 问题文本 |
| `ground_truth` | str | 标准答案 |
| `mentor_tokens` | int | mentor token 数量 (-1/0/100/500/1000)，-1 表示 mentor only |
| `mentor_response` | str | mentor 的推理过程 |
| `response` | str | 完整推理响应 |
| `is_correct` | bool | 答案是否正确 |
| `mentor_length` | int | mentor 生成的实际 token 数 |
| `intern_length` | int | intern 生成的实际 token 数 |

### C. 评估结果说明

| 指标 | 说明 |
|------|------|
| T-1 | Mentor only 准确率（大模型独立生成） |
| T0 | Intern only 准确率（小模型独立生成） |
| T100 | 100 mentor tokens 时的准确率 |
| T500 | 500 mentor tokens 时的准确率 |
| T1000 | 1000 mentor tokens 时的准确率 |
| Oracle | 理论最优（任一 token level 正确即正确） |
| Cascade | 分类器 cascade 准确率 |
| Gap | Cascade - Best Baseline（提升） |

### D. 文件说明

| 文件 | 说明 |
|------|------|
| `run_pipeline.sh` | 一键运行脚本（支持 --think/--no-think） |
| `run_compare_classifier.sh` | 分类器比较脚本（支持 --methods 选择方法） |
| `collect_data_vllm_think.py` | 数据收集（vLLM，支持 Think/Standard 模式） |
| `compute_stats.py` | 计算 Oracle/Baseline 统计 |
| `train_mlp_classifier.py` | MLP 分类器训练（推荐，冻结 LLM） |
| `train_lora_classifier.py` | LoRA 分类器训练（微调 LLM） |
| `train_ppl_classifier.py` | PPL 分类器训练（基于 perplexity/entropy） |
| `train_ensemble_classifier.py` | Ensemble 分类器训练（组合多个分类器） |
| `summarize_results.py` | 汇总所有子集结果 |
| `eval_model.py` | 单独评测模型性能 |

---

## 9. 单独模型评测

用于对比不同模型的性能（准确率、生成长度、推理速度）。

```bash
# 使用 vLLM 评测（推荐）
python eval_model.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --gpu 0

# 评测其他模型
python eval_model.py --model Qwen/Qwen2.5-7B-Instruct --gpu 0
python eval_model.py --model meta-llama/Llama-3.1-8B-Instruct --gpu 0

# 评测特定子集
python eval_model.py --model xxx --subset algebra

# 使用 HuggingFace 后端
python eval_model.py --model xxx --backend hf

# 快速测试（限制样本数）
python eval_model.py --model xxx --max-samples 100

# 禁用 CoT
python eval_model.py --model xxx --no-cot
```

**输出目录**: `/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/eval_results/{model_name}/`

**输出指标**:
- 准确率 (Accuracy)
- 生成长度统计 (mean/median/min/max tokens)
- 推理速度 (tokens/s)
- 按难度级别的准确率 (Level 1-5)
