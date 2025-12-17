# ACT-E: Adaptive Control of LLM Thinking Ensemble

## 完整实验流程

本文档提供从数据收集到结果对比的完整实验流程。所有命令默认使用 8 GPU 并行。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据收集](#2-数据收集)
3. [数据统计](#3-数据统计)
4. [LoRA 分类器训练](#4-lora-分类器训练)
5. [Cascade 评估](#5-cascade-评估)
6. [结果汇总](#6-结果汇总)
7. [一键运行](#7-一键运行)

---

## 1. 环境准备

```bash
pip install torch transformers peft datasets scikit-learn tqdm numpy vllm
```

---

## 2. 数据收集

### 2.1 方式一：标准推理（无结构化思考）

```bash
cd /home/fzkuji/PycharmProjects/Ensemble-Hub/scripts/mentor_guided

# 收集 train split（8 GPU 并行）
python collect_progressive_data.py \
    --dataset hendrycks_math \
    --split train \
    --parallel \
    --gpus 0,1,2,3,4,5,6,7

# 收集 test split（8 GPU 并行）
python collect_progressive_data.py \
    --dataset hendrycks_math \
    --split test \
    --parallel \
    --gpus 0,1,2,3,4,5,6,7
```

**输出目录**: `/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split/`

### 2.2 方式二：结构化思考推理（vLLM + Think Token）

```bash
# 收集 train split
python collect_data_vllm_think.py --split train --gpu 0

# 收集 test split
python collect_data_vllm_think.py --split test --gpu 0
```

**输出目录**: `/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B/`

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

## 4. LoRA 分类器训练

### 4.1 标准数据训练（所有子集）

```bash
# algebra
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset algebra

# counting_and_probability
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset counting_and_probability

# geometry
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset geometry

# intermediate_algebra
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset intermediate_algebra

# number_theory
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset number_theory

# prealgebra
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset prealgebra

# precalculus
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset precalculus

# 所有子集合并训练
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset all
```

### 4.2 Think 数据训练（所有子集）

```bash
# algebra
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset algebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# counting_and_probability
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset counting_and_probability \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# geometry
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset geometry \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# intermediate_algebra
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset intermediate_algebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# number_theory
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset number_theory \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# prealgebra
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset prealgebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# precalculus
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset precalculus \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# 所有子集合并训练
torchrun --nproc_per_node=8 train_lora_classifier.py --ddp --subset all \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B
```

---

## 5. Cascade 评估

### 5.1 标准数据评估（所有子集）

```bash
# algebra
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset algebra

# counting_and_probability
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset counting_and_probability

# geometry
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset geometry

# intermediate_algebra
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset intermediate_algebra

# number_theory
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset number_theory

# prealgebra
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset prealgebra

# precalculus
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset precalculus
```

### 5.2 Think 数据评估（所有子集）

```bash
# algebra
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset algebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# counting_and_probability
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset counting_and_probability \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# geometry
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset geometry \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# intermediate_algebra
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset intermediate_algebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# number_theory
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset number_theory \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# prealgebra
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset prealgebra \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B

# precalculus
torchrun --nproc_per_node=8 eval_lora_cascade.py --subset precalculus \
    --data-dir /mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B
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

### 7.1 标准数据完整流程

```bash
./run_standard.sh
```

### 7.2 Think 数据完整流程

```bash
./run_think.sh
```

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
    --parallel \
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
│   │   │   ├── tokens0.json
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
| `mentor_tokens` | int | mentor token 数量 (0/100/500/1000) |
| `mentor_response` | str | mentor 的推理过程 |
| `response` | str | 完整推理响应 |
| `is_correct` | bool | 答案是否正确 |

### C. 评估结果说明

| 指标 | 说明 |
|------|------|
| T0 | 无 mentor token 时的准确率 |
| T100 | 100 mentor tokens 时的准确率 |
| T500 | 500 mentor tokens 时的准确率 |
| T1000 | 1000 mentor tokens 时的准确率 |
| Oracle | 理论最优（任一 token level 正确即正确） |
| Cascade | LoRA 分类器 cascade 准确率 |
| Gap | Cascade - Best Baseline（提升） |

### D. 文件说明

| 文件 | 说明 |
|------|------|
| `collect_progressive_data.py` | 标准数据收集（8 GPU 并行） |
| `collect_data_vllm_think.py` | Think 数据收集（vLLM） |
| `compute_stats.py` | 计算 Oracle/Baseline 统计 |
| `train_lora_classifier.py` | LoRA 分类器训练（8 GPU DDP） |
| `eval_lora_cascade.py` | Cascade 评估（8 GPU DDP） |
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
