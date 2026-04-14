# Ensemble-Hub

<p align="center">
  <img src="assets/Ensemble-Hub.gif" alt="Ensemble-Hub" width="600">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/Paper-ACL%202025%20Findings-blue" alt="Paper"></a>
  <a href="https://github.com/Fzkuji/Ensemble-Hub/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>
</p>

This is the official implementation of **Tandem**, accepted at **ACL 2025 Findings**.

> **Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning**
>
> Zichuan Fu, Xian Wu, Guojing Li, Yejing Wang, Yijun Chen, Zihao Zhao, Yixuan Luo, Hanyu Yan, Yefeng Zheng, Xiangyu Zhao
>
> *Recent advancements in large language models (LLMs) have catalyzed the rise of reasoning-intensive inference paradigms, where models perform explicit step-by-step reasoning before generating final answers. While such approaches improve answer quality and interpretability, they incur substantial computational overhead due to the prolonged generation sequences. We propose Tandem, a novel collaborative framework that synergizes large and small language models (LLMs and SLMs) to achieve high-quality reasoning with significantly reduced computational cost. The LLM serves as a strategic coordinator, efficiently generating a compact set of critical reasoning insights. These insights then guide a smaller, more efficient SLM in executing the full reasoning process and delivering the final response. To balance efficiency and reliability, Tandem introduces a cost-aware termination mechanism that adaptively determines when sufficient reasoning guidance has been accumulated, enabling early stopping of the LLM's generation. Experiments on mathematical reasoning and code generation benchmarks demonstrate that Tandem reduces computational costs by approximately 40% compared to standalone LLM reasoning, while achieving superior or competitive performance.*

## Key Results

| Method | MATH Acc. (%) | Cost (TFLOPs) | Cost Reduction |
|--------|:---:|:---:|:---:|
| SLM (DeepSeek-7B) | 77.14 | 38.25 | — |
| LLM (DeepSeek-32B) | 80.90 | 168.35 | — |
| **Tandem (7B+32B)** | **83.46** | **99.72** | **40.7%** |

- **+2.56%** accuracy over the standalone 32B LLM while using only **59%** of its computational cost
- Sufficiency classifier trained on MATH transfers to HumanEval (code generation) **without retraining**
- Compatible with both open-source and API-accessible LLMs (GPT-4o-mini, etc.)

## Method Overview

Tandem establishes a **mentor-intern** collaboration between a large and a small language model:

1. **Thinking Insight Generation**: The LLM generates structured reasoning insights following the GPRA schema (Goal, Planning, Retrieval, Action), inspired by the ACT-R cognitive architecture.
2. **Cost-Aware Continual Judgment**: A lightweight classifier evaluates whether the current guidance is sufficient for the SLM, using perplexity and entropy as confidence indicators.
3. **Response Completion**: Once sufficient guidance is detected, the SLM takes over to complete the reasoning and produce the final answer.

The framework progressively generates insights across three effort levels (low, medium, high), enabling adaptive allocation of computational resources — simple problems terminate early with minimal guidance, while complex ones receive deeper support.

## Getting Started

### Installation

```bash
conda create -n tandem python=3.12
conda activate tandem

git clone https://github.com/Fzkuji/Ensemble-Hub.git
cd Ensemble-Hub
pip install -r requirements.txt
```

### Reproducing Tandem Experiments

Experiment scripts are located in `scripts/mentor_guided/`:

| Script | Mentor | Intern | Dataset |
|--------|--------|--------|---------|
| `exp_homo_math.sh` | DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-Qwen-7B | MATH-500 |
| `exp_homo_humaneval.sh` | DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-Qwen-7B | HumanEval |
| `exp_hetero_math.sh` | GPT-4o (API) | DeepSeek-R1-Distill-Qwen-7B | MATH-500 |
| `exp_hetero_humaneval.sh` | GPT-4o (API) | DeepSeek-R1-Distill-Qwen-7B | HumanEval |

```bash
# Homogeneous models on MATH-500
bash scripts/mentor_guided/exp_homo_math.sh

# Heterogeneous models (requires API key)
export OPENROUTER_API_KEY="your-api-key"
bash scripts/mentor_guided/exp_hetero_math.sh
```

**Parallel data collection** for faster processing with multiple GPUs:

```bash
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset math500 \
    --split train \
    --mentor-type local \
    --parallel \
    --num-workers 4 \
    --mentor-gpus "0,1,2,3" \
    --intern-gpus "4,5,6,7"
```

Results are saved to `data/acte_experiments/results/`.

## Ensemble-Hub Toolkit

Beyond Tandem, this repository also provides a general-purpose **LLM ensemble inference toolkit** supporting multiple ensemble strategies:

### Ensemble Methods

**Model Selection**: `zscore` | `all` | `random`

**Output Aggregation**: `reward_based` | `progressive` | `loop` | `gac` | `distribution` | `random`

### Batch Inference

```bash
python -m ensemblehub.inference \
   --config examples/all_progressive.yaml \
   --input_path data/AIME2024/aime/aime24.json \
   --output_path saves/aime24.jsonl
```

### FastAPI Server (OpenAI-compatible)

```bash
python ensemblehub/api.py examples/all_loop.yaml
```

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ensemble", "messages": [{"role": "user", "content": "Hello"}]}'
```

Compatible with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for standardized benchmarking.

### Repository Structure

```
Ensemble-Hub/
├── ensemblehub/                    # Main package
│   ├── api/                        # FastAPI server (OpenAI-compatible)
│   ├── ensemble_methods/           # Ensemble strategies
│   │   ├── ensemble.py             # Core framework
│   │   ├── model_selection/        # Pre-inference model selection
│   │   └── output_aggregation/     # Token/sentence/response-level aggregation
│   ├── generators/                 # Model backends (HF, vLLM, API)
│   ├── scorers/                    # Reward models
│   └── inference.py                # Batch inference pipeline
├── scripts/mentor_guided/          # Tandem experiment scripts
├── data/                           # Datasets (MATH, GSM8K, HumanEval, AIME)
├── examples/                       # YAML configuration examples
└── docs/                           # Documentation
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{fu2025tandem,
  title={Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning},
  author={Fu, Zichuan and Wu, Xian and Li, Guojing and Wang, Yejing and Chen, Yijun and Zhao, Zihao and Luo, Yixuan and Yan, Hanyu and Zheng, Yefeng and Zhao, Xiangyu},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025}
}
```

## License

Apache-2.0. See [LICENSE](./LICENSE) for details.

## Acknowledgements

This work was supported by City University of Hong Kong, Tencent, Renmin University of China, and Westlake University. We thank the open-source community for [DeepSeek](https://github.com/deepseek-ai), [Qwen](https://github.com/QwenLM), and [Hugging Face Transformers](https://github.com/huggingface/transformers).
