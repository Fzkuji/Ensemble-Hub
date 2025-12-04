![# LLaMA Factory](assets/Ensemble-Hub.gif)

# Ensemble-Hub

**Ensemble-Hub** is an open-source toolkit for large language model (LLM) ensemble inference. 
It is designed to support and unify multiple ensemble strategies for LLMs, including existing methods such as [LLM-Blender](https://github.com/yuchenlin/LLM-Blender), [GaC](https://github.com/yaoching0/GaC), and [UniTE](https://github.com/starrYYxuan/UniTE). 
The project is under active development.

## 🌟 Project goals

| **Why?**                                                  | **How?**                                                                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Boost answer quality** by letting several LLMs compete. | Each round, every generator writes a short segment → a reward model (Qwen 2.5-Math-PRM-7B) scores them → best segment is kept. |
| **Stay fast & memory-friendly** with model caching.       | ModelPool loads each generator/reward model once, then re-uses it for every call (CLI, notebook or API).                       |
| **Provide plug-and-play usage** for research & services.  | Python `EnsembleFramework` class **or** a production-grade FastAPI server (`ensemblehub/api.py`).                            |


## 💡 Core features

* **Unlimited generators** – mix and match multiple models (HF *and* vLLM backends supported).
* **Reward-guided selection** – uses a reward model (e.g. Qwen2.5-Math-PRM-7B) to score candidates and pick the best output each round.
* **EOS-based early stop** – if a model outputs its end-of-sequence token, the loop exits early.
* **Context accumulation** – optionally carry forward previously chosen segments into the next round (builds a running conversation context).
* **Clean prompt template** – minimal prompt format with no extraneous instructions (no stray “600 words” artifacts).
* **Singleton caches** – models load once and are reused on repeated calls (even across API requests).


## 🎯 Ensemble Methods

Ensemble-Hub supports multiple ensemble strategies that can be easily configured:

### Model Selection Methods
- **`zscore`**: Statistical selection based on perplexity and confidence scores
- **`all`**: Use all available models (no selection)
- **`random`**: Randomly select a subset of models

### Output Aggregation Methods
- **`reward_based`**: Reward-based selection using scoring models (default)
- **`progressive`**: Length or token-based model switching during generation
  - Length-based: switch models based on output length thresholds
  - Token-based: switch models when encountering special tokens
- **`random`**: Random selection from model outputs
- **`loop`**: Round-robin cycling through models
- **`gac`**: GAC token-level aggregation
- **`distribution`**: Distribution-based token aggregation

## 🗂 Repository layout

```
Ensemble-Hub/
├── ensemblehub/                         # Main package
│   ├── api/                             # FastAPI server module
│   │   ├── __main__.py                  # Command line entry point
│   │   └── app.py                       # FastAPI application
│   ├── ensemble_methods/                # Ensemble method implementations
│   │   ├── ensemble.py                  # Unified ensemble framework
│   │   ├── model_selection/             # Model selection strategies
│   │   │   ├── base.py                  # Base selector interface
│   │   │   ├── statistical.py           # Z-score, random selection
│   │   │   └── learned.py               # LLM-Blender, meta-learning
│   │   └── output_aggregation/          # Output aggregation methods
│   │       ├── token_level/             # Token-level aggregation (GAC, distribution)
│   │       ├── sentence_level/          # Sentence-level aggregation
│   │       │   ├── loop_selector.py     # Round-robin selection
│   │       │   ├── random_selector.py   # Random selection
│   │       │   ├── reward_based.py      # Reward-based selection
│   │       │   └── progressive_selector.py # Progressive selection
│   │       └── response_level/          # Response-level aggregation
│   ├── generators/                      # Model generators (HF, vLLM backends)
│   │   ├── base.py                      # Base generator interface
│   │   ├── hf.py                        # Hugging Face transformers
│   │   ├── vllm.py                      # vLLM backend
│   │   └── pool.py                      # Generator pool management
│   ├── scorers/                         # Reward models and scoring
│   │   └── base.py                      # Base scorer interface
│   ├── inference.py                     # High-level inference pipeline
│   └── utils.py                         # Utility functions
├── data/                                # Datasets (AIME, GSM8K, MATH, etc.)
├── docs/                                # Documentation
│   ├── api_usage.md                     # Complete API usage guide
│   ├── benchmark_single_model.md        # Single model benchmarking
│   └── progressive_selector_usage.md    # Progressive selector guide
├── examples/                            # Usage examples
│   └── test_single_model.py             # Single model testing
├── scripts/                             # Utility scripts
│   ├── vllm_infer.py                    # vLLM inference script
│   └── grader.py                        # Answer grading
├── requirements.txt                     # Dependencies
└── README.md                            # You're here!
```

##  Getting Started

### 🔧 Installation

```bash
conda create -n ensemble python=3.12
conda activate ensemble

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

git clone https://github.com/Fzkuji/Ensemble-Hub.git
cd Ensemble-Hub

pip install -r requirements.txt
```


### 💻 Quickstart

> [!NOTE]
> The inference script now supports both YAML configuration files and command-line arguments.

**Using YAML configuration (recommended):**
```shell
python -m ensemblehub.inference \
   --config examples/all_progressive.yaml \
   --input_path data/AIME2024/aime/aime24.json \
   --output_path saves/aime24.jsonl \
   --max_examples 10 \
   --batch_size 1
```

**Using command-line arguments only:**
```shell
python -m ensemblehub.inference \
   --input_path data/AIME2024/aime/aime24.json \
   --output_path saves/aime24.jsonl \
   --max_examples 500 \
   --batch_size 4 \
   --output_aggregation_method progressive \
   --max_tokens 2048 \
   --model_specs "Qwen/Qwen2.5-0.5B-Instruct:hf:auto" \
   --model_specs "Qwen/Qwen2.5-1.5B-Instruct:hf:auto"
```

*Under the hood: models are loaded once → the reward model scores each round → loop stops when the selected segment ends with an EOS token.*

### 🚀 Start the FastAPI

#### Using YAML Configuration (Recommended)

```bash
# Start with example configuration
python ensemblehub/api.py examples/all_loop.yaml

# Or use progressive ensemble
python ensemblehub/api.py examples/all_progressive.yaml
```

#### Using Default Configuration

```bash
# Start with default settings
python ensemblehub/api.py
```

#### Evaluate with lm-evaluation-harness

```bash
lm_eval --model hf \
   --tasks arc_challenge_chat \
   --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
   --batch_size 2 \
   --num_fewshot 5
```

or through the Ensemble-Hub API proxy:

```bash
# Start API server
python ensemblehub/api.py examples/all_loop.yaml

# Run evaluation in another terminal
export OPENAI_API_KEY=dummy_key
lm_eval --model openai-completions \
   --tasks arc_challenge_chat \
   --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
   --batch_size 2 \
   --num_fewshot 5

# For longer completions (e.g. MBPP) extend the generation budget
lm_eval --model openai-completions \
   --tasks mbpp \
   --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None,max_gen_toks=1024 \
   --batch_size 2 \
   --limit 1 \
   --confirm_run_unsafe_code
```

> **Note**: Server configuration is controlled via environment variables API_HOST and API_PORT.

#### Testing the API

```bash
# Health check
curl http://localhost:8000/status

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ensemble", "messages": [{"role": "user", "content": "Hello"}]}'

# Text completion  
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ensemble", "prompt": "Hello", "max_tokens": 50}'
```

## 🧪 ACT-E Experiments

ACT-E (Adaptive Control of LLM Thinking Ensemble) is a framework for dynamically deciding how much mentor guidance an intern model needs. The experiment scripts evaluate different model combinations and datasets.

### Experiment Scripts

Located in `scripts/mentor_guided/`:

| Script | Mentor Model | Intern Model | Dataset |
|--------|--------------|--------------|---------|
| `exp_homo_math.sh` | DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-Qwen-7B | MATH-500 |
| `exp_homo_humaneval.sh` | DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-Qwen-7B | HumanEval |
| `exp_hetero_math.sh` | GPT-4o (API) | DeepSeek-R1-Distill-Qwen-7B | MATH-500 |
| `exp_hetero_humaneval.sh` | GPT-4o (API) | DeepSeek-R1-Distill-Qwen-7B | HumanEval |

### Baseline Strategies

Each experiment evaluates the following strategies:

| Strategy | Description |
|----------|-------------|
| Mentor Only | Mentor generates complete response (no intern) |
| Intern Only | Intern generates complete response (no mentor) |
| Progressive-100 | Mentor generates 100 tokens, intern continues |
| Progressive-500 | Mentor generates 500 tokens, intern continues |
| Progressive-1000 | Mentor generates 1000 tokens, intern continues |
| ACT-E (LSTM/GRU/MLP/Attention) | Adaptive selection based on PPL/Entropy |

### Running Experiments

```bash
# Homogeneous models + MATH-500
bash scripts/mentor_guided/exp_homo_math.sh

# Homogeneous models + HumanEval
bash scripts/mentor_guided/exp_homo_humaneval.sh

# Heterogeneous models + MATH-500 (requires OPENROUTER_API_KEY)
export OPENROUTER_API_KEY="your-api-key"
bash scripts/mentor_guided/exp_hetero_math.sh

# Heterogeneous models + HumanEval
bash scripts/mentor_guided/exp_hetero_humaneval.sh
```

### Parallel Data Collection

For faster data collection with multiple GPUs, use parallel mode:

```bash
# Homogeneous models (local mentor + local intern)
# 4 workers: GPU 0-3 for Mentor (32B), GPU 4-7 for Intern (7B)
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset math500 \
    --split train \
    --mentor-type local \
    --parallel \
    --num-workers 4 \
    --mentor-gpus "0,1,2,3" \
    --intern-gpus "4,5,6,7"

# Heterogeneous models (API mentor + local intern)
# All 8 GPUs for Intern since Mentor uses API
python scripts/mentor_guided/collect_progressive_data.py \
    --dataset math500 \
    --split train \
    --mentor-type api \
    --api-model "gpt-4o" \
    --parallel \
    --num-workers 8 \
    --intern-gpus "0,1,2,3,4,5,6,7"
```

**Memory Requirements:**
- 32B Mentor model: ~64GB VRAM per instance (fp16)
- 7B Intern model: ~14GB VRAM per instance (fp16)

### Output

Results are saved to `data/acte_experiments/results/` with accuracy, average token lengths, and TFLOPs comparison.

## 📌 To-Do

- [x] Multi-model inference
- [x] HuggingFace backend
- [x] FastAPI server with OpenAI-compatible endpoints
- [x] Ray Serve integration
- [x] Command line configuration for ensemble methods
- [x] LM-evaluation-harness compatibility
- [ ] Reward model selection
- [ ] vLLM backends
- [ ] API support for closed-source models
- [ ] Streaming API interface (SSE)
- [ ] Advanced scorer aggregation methods

## 📝 Changelog

### Recent Updates

- **Enable Thinking Mode**: Refactored `enable_thinking` parameter to be configured at model initialization level instead of generation time. This allows better integration with LLaMA-Factory's template system and supports reasoning models like DeepSeek-R1.
- **Consistent Length Handling**: Updated tokenizer calls to use `cutoff_len` from DataArguments for consistent max_length handling across all generation methods.
- **API Improvements**: Added `--enable_thinking` command line flag for easy configuration of reasoning models.

## 📜 License

Apache-2.0. See the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements

Relies on **DeepSeek**, **Qwen** model weights, Hugging Face Transformers, [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the incredible open-source community.
