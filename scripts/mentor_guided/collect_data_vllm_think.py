#!/usr/bin/env python3
"""
Collect Progressive Data with vLLM and Structured Thinking Prompt

Uses vLLM for fast inference with:
- Chat template support
- <think> token for reasoning
- Structured prompt with Goal, Planning, Retrieval, Action framework

Collects data for different mentor token lengths:
- 0 (intern only)
- 100 tokens
- 500 tokens
- 1000 tokens

Usage:
    # Single GPU
    python collect_data_vllm_think.py --dataset hendrycks_math --split test

    # Specific GPU
    python collect_data_vllm_think.py --gpu 0

    # Custom output
    python collect_data_vllm_think.py --output-dir /path/to/output
"""

import argparse
import json
import logging
import os
import sys
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
import time
from tqdm import tqdm

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grader import grade_answer

# Optional: OpenRouter API support
try:
    from openrouter_inference import OpenRouterInference
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

# Single model cache for reusing -1 and 0 token level results
try:
    from single_model_cache import SingleModelCache, check_and_use_cache, save_to_cache, get_model_short_name
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    SingleModelCache = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Token levels to collect (0 = no mentor, just intern; -1 = mentor only, no intern)
TOKEN_LEVELS = [0, 100, 500, 1000]
MENTOR_ONLY_LEVEL = -1  # Special level for mentor-only baseline

# =============================================================================
# Dataset Configurations
# =============================================================================
# Each dataset config specifies:
#   - hf_path: HuggingFace dataset path
#   - hf_subset: HuggingFace subset name (None if not applicable)
#   - splits: Available splits (train, test, etc.)
#   - question_field: Field name for the question/problem
#   - answer_field: Field name for the answer/solution
#   - answer_parser: How to parse the answer (None = use as-is, 'gsm8k' = extract from ####)
#   - subsets: List of subsets (for datasets with multiple subjects)
#   - extra_fields: Additional fields to preserve (e.g., 'type', 'level')
#
DATASET_CONFIGS = {
    "hendrycks_math": {
        "hf_path": "hendrycks/competition_math",
        "hf_subset": None,  # subset is specified via 'type' field
        "splits": ["train", "test"],
        "question_field": "problem",
        "answer_field": "solution",
        "answer_parser": None,  # Use solution directly with grader
        "subsets": [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
        "extra_fields": ["type", "level"],
    },
    "math500": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "hf_subset": None,
        "splits": ["test"],  # Only test split available
        "question_field": "problem",
        "answer_field": "solution",
        "answer_parser": None,
        "subsets": ["math500"],  # Single subset
        "extra_fields": ["type", "level"],
    },
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_subset": "main",
        "splits": ["train", "test"],
        "question_field": "question",
        "answer_field": "answer",
        "answer_parser": "gsm8k",  # Extract number after ####
        "subsets": ["gsm8k"],  # Single subset
        "extra_fields": [],
    },
    # Future datasets can be added here:
    # "aime": {
    #     "hf_path": "...",
    #     "hf_subset": None,
    #     "splits": ["train", "test"],
    #     "question_field": "problem",
    #     "answer_field": "answer",
    #     "answer_parser": None,
    #     "subsets": ["aime"],
    #     "extra_fields": ["year", "problem_number"],
    # },
}

# Simple system prompt (ACT-E uses simple prompts)
SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""

# =============================================================================
# Model Family Configurations
# =============================================================================
# Different model families handle thinking mode differently:
#
# DeepSeek-R1:
#   - Chat template automatically adds <think>\n at the end
#   - For no-think mode: manually append <think>\n</think>\n\n after prompt
#   - </think> token ID: 151649
#
# Qwen3:
#   - Chat template does NOT add <think> by default (enable_thinking=True)
#   - For no-think mode: use enable_thinking=False, which adds <think>\n\n</think>\n\n
#   - Model will generate <think>...</think> on its own when thinking
#   - </think> token ID: 151668
#
# Cross-model (e.g., DeepSeek mentor + Qwen3 intern):
#   - Mentor generates with its own think format
#   - We extract thinking content and re-format for intern's expected format
#
MODEL_FAMILIES = {
    "deepseek-r1": {
        "think_start": "<think>",
        "think_end": "</think>",
        "think_end_token_id": 151649,
        "template_adds_think": True,  # Chat template adds <think>\n
        "no_think_prefill": "<think>\n</think>\n\n",  # For no-think mode
    },
    "qwen3": {
        "think_start": "<think>",
        "think_end": "</think>",
        "think_end_token_id": 151668,
        "template_adds_think": False,  # Chat template does NOT add <think>
        "enable_thinking_param": True,  # Use enable_thinking param in apply_chat_template
    },
    "gpt-oss": {
        # GPT-OSS uses <thinking>/<final_answer> tags (different from DeepSeek/Qwen's <think>)
        # Reasoning mode is controlled via system prompt: "Reasoning: high/medium/low/none"
        "think_start": "<thinking>",
        "think_end": "</thinking>",
        "final_answer_start": "<final_answer>",
        "final_answer_end": "</final_answer>",
        "template_adds_think": False,  # Need to add "Reasoning: high" to system prompt
        "reasoning_system_prompt": True,  # Use "Reasoning: high" in system prompt
        "no_think_system": "Reasoning: none",  # System prompt for no-think mode
        "think_system": "Reasoning: high",  # System prompt for think mode
    },
    "default": {
        "think_start": None,
        "think_end": None,
        "template_adds_think": False,
    },
}


def detect_model_family(model_name: str) -> str:
    """Detect model family from model name.

    Args:
        model_name: HuggingFace model name (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')

    Returns:
        Model family key ('deepseek-r1', 'qwen3', 'gpt-oss', or 'default')
    """
    model_lower = model_name.lower()

    if "deepseek-r1" in model_lower or "deepseek_r1" in model_lower:
        return "deepseek-r1"
    elif "qwen3" in model_lower or "qwen-3" in model_lower or "qwen/qwen3" in model_lower:
        return "qwen3"
    elif "gpt-oss" in model_lower or "gptoss" in model_lower or "openai/gpt-oss" in model_lower:
        return "gpt-oss"
    else:
        return "default"


def build_cross_model_prompt(
    question: str,
    mentor_output: str,
    intern_model: "VLLMInference",
    use_think: bool = True,
    mentor_family: str = None,
) -> str:
    """Build prompt for intern model to continue from mentor's output.

    When mentor and intern are different model families, we need to:
    1. Build intern's native prompt format
    2. Append mentor's thinking content in a format intern understands

    Args:
        question: The original question
        mentor_output: Mentor's generated output (thinking + partial answer)
        intern_model: The intern VLLMInference instance
        use_think: Whether thinking mode is enabled
        mentor_family: Mentor's model family (for format conversion)

    Returns:
        Formatted prompt for intern to continue generation
    """
    # Build intern's base prompt
    intern_prompt = intern_model.build_chat_prompt(question, use_think=use_think)

    # Handle cross-model format conversion
    # GPT-OSS uses <thinking>/<final_answer> while DeepSeek/Qwen use <think>
    intern_family = intern_model.model_family

    # Convert mentor output format to intern's expected format if needed
    converted_output = mentor_output
    if mentor_family == "gpt-oss" and intern_family in ("deepseek-r1", "qwen3"):
        # Convert GPT-OSS format to DeepSeek/Qwen format
        converted_output = mentor_output.replace("<thinking>", "<think>").replace("</thinking>", "</think>")
        converted_output = converted_output.replace("<final_answer>", "").replace("</final_answer>", "")
    elif mentor_family in ("deepseek-r1", "qwen3") and intern_family == "gpt-oss":
        # Convert DeepSeek/Qwen format to GPT-OSS format
        converted_output = mentor_output.replace("<think>", "<thinking>").replace("</think>", "</thinking>")

    # For cross-model scenarios, we need to handle the prompt carefully
    if intern_family == "qwen3":
        # Qwen3's prompt doesn't include <think>, so we need to add it
        if converted_output.startswith("<think>") or converted_output.strip().startswith("<think>"):
            return intern_prompt + converted_output
        else:
            return intern_prompt + "<think>\n" + converted_output
    elif intern_family == "gpt-oss":
        # GPT-OSS: the reasoning is triggered by system prompt, just append output
        if converted_output.startswith("<thinking>") or converted_output.strip().startswith("<thinking>"):
            return intern_prompt + converted_output
        else:
            return intern_prompt + "<thinking>\n" + converted_output
    else:
        # DeepSeek-R1 or default: prompt already ends with <think>\n
        return intern_prompt + converted_output


def parse_answer(raw_answer: str, parser_type: Optional[str] = None) -> str:
    """Parse answer based on dataset-specific format.

    Args:
        raw_answer: Raw answer string from dataset
        parser_type: Parser type from DATASET_CONFIGS['answer_parser']
                    None = use as-is
                    'gsm8k' = extract number after ####

    Returns:
        Parsed answer string
    """
    import re

    if parser_type is None:
        return raw_answer

    if parser_type == "gsm8k":
        # GSM8K format: "... #### 42" -> extract "42"
        match = re.search(r'####\s*(.+)$', raw_answer)
        if match:
            final_answer = match.group(1).strip()
            # Remove commas from numbers (e.g., "1,234" -> "1234")
            final_answer = final_answer.replace(',', '')
            return final_answer
        return raw_answer

    # Unknown parser type, return as-is
    logger.warning(f"Unknown answer parser type: {parser_type}")
    return raw_answer


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{}."""
    start = text.find(r'\boxed{')
    if start == -1:
        return ""
    i = start + len(r'\boxed{')
    depth = 1
    content = ""
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        if depth > 0:
            content += text[i]
        i += 1
    return content.strip()


def extract_thinking_content(text: str, model_family: str = None) -> Tuple[str, str]:
    """Extract thinking content and final answer from model output.

    Handles different model formats:
    - DeepSeek/Qwen: <think>...</think>
    - GPT-OSS: <thinking>...</thinking> and <final_answer>...</final_answer>

    Args:
        text: Model output text
        model_family: Model family for format detection (optional, will auto-detect)

    Returns:
        Tuple of (thinking_content, remaining_content)
    """
    import re

    # Auto-detect format if not specified
    if model_family is None:
        if "<thinking>" in text:
            model_family = "gpt-oss"
        elif "<think>" in text:
            model_family = "deepseek-r1"  # or qwen3, same format

    if model_family == "gpt-oss":
        # GPT-OSS format: <thinking>...</thinking> and optionally <final_answer>...</final_answer>
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
        thinking = thinking_match.group(1).strip() if thinking_match else ""

        # Get content after </thinking> or <final_answer>...</final_answer>
        remaining = text
        if thinking_match:
            remaining = text[thinking_match.end():]

        final_match = re.search(r'<final_answer>(.*?)</final_answer>', remaining, re.DOTALL)
        if final_match:
            remaining = final_match.group(1).strip()

        return thinking, remaining.strip()
    else:
        # DeepSeek/Qwen format: <think>...</think>
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""

        remaining = text
        if think_match:
            remaining = text[think_match.end():]

        return thinking, remaining.strip()


def check_math_correctness(prediction: str, ground_truth: str) -> bool:
    """Check if math answer is correct using the official grader.

    Handles two formats:
    - MATH-style: ground_truth contains \\boxed{answer}
    - GSM8K-style: ground_truth is just the answer (e.g., "72")
    """
    pred_answer = extract_boxed_answer(prediction)

    # Try to extract boxed answer from ground_truth first
    true_answer = extract_boxed_answer(ground_truth)

    # If ground_truth has no \boxed{}, use it directly (GSM8K format)
    if not true_answer:
        true_answer = ground_truth.strip()

    if not pred_answer or not true_answer:
        return False

    return grade_answer(pred_answer, true_answer)


class VLLMInference:
    """vLLM-based inference with chat template support."""

    def __init__(
        self,
        model_name: str,
        gpu_ids: List[int] = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
    ):
        """Initialize vLLM model.

        Args:
            model_name: HuggingFace model name
            gpu_ids: List of GPU IDs to use (for tensor parallelism)
            max_model_len: Maximum model context length
            gpu_memory_utilization: Fraction of GPU memory to use (default: 0.9)
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is required. Install with: pip install vllm")

        if gpu_ids is None:
            gpu_ids = [0]

        # Store model name and detect model family
        self.model_name = model_name
        self.model_family = detect_model_family(model_name)
        self.family_config = MODEL_FAMILIES.get(self.model_family, MODEL_FAMILIES["default"])

        # Set visible GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        tensor_parallel_size = len(gpu_ids)

        logger.info(f"Loading model {model_name} (family: {self.model_family}) with vLLM on GPU {gpu_ids} (tp={tensor_parallel_size}, memory_util={gpu_memory_utilization})...")

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.SamplingParams = SamplingParams
        logger.info("Model loaded successfully")

    def cleanup(self):
        """Clean up vLLM resources."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                # vLLM's LLM class doesn't have a built-in cleanup method
                # but we can delete the model to trigger garbage collection
                del self.model
                self.model = None
                import gc
                gc.collect()
                # Try to clean up CUDA memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def build_chat_prompt(
        self,
        question: str,
        use_think: bool = True,
    ) -> str:
        """Build simple chat prompt.

        Handles different model families:
        - DeepSeek-R1: template adds <think>\n, no-think mode pre-fills empty block
        - Qwen3: use enable_thinking param, no-think uses enable_thinking=False
        - GPT-OSS: use "Reasoning: high/none" in system prompt

        Args:
            question: The math problem
            use_think: Whether to allow thinking (True) or skip it (False)

        Returns:
            Formatted prompt string
        """
        # Build system prompt based on model family
        if self.model_family == "gpt-oss":
            # GPT-OSS: control reasoning via system prompt
            reasoning_directive = self.family_config.get("think_system" if use_think else "no_think_system", "")
            system_content = f"{reasoning_directive}\n\n{SYSTEM_PROMPT}" if reasoning_directive else SYSTEM_PROMPT
        else:
            system_content = SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

        # Build base prompt based on model family
        if self.model_family == "qwen3":
            # Qwen3: use enable_thinking parameter
            # enable_thinking=True: model decides whether to think (default behavior)
            # enable_thinking=False: pre-fills empty think block to skip thinking
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=use_think,
            )
        elif self.model_family == "gpt-oss":
            # GPT-OSS: standard template (reasoning controlled via system prompt)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # DeepSeek-R1 and others: standard template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # For no-think mode: pre-fill empty think block
            if not use_think and self.family_config.get("no_think_prefill"):
                prompt = prompt + self.family_config["no_think_prefill"]

        return prompt

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of formatted prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            List of generated responses
        """
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = self.model.generate(prompts, sampling_params, use_tqdm=False)

        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)

        return responses

    def generate_mentor_tokens(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate limited mentor tokens (for hint generation).

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of partial responses (hints)
        """
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )

        outputs = self.model.generate(prompts, sampling_params, use_tqdm=False)

        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)

        return responses


def load_hendrycks_math_subset(
    subset: str,
    split: str = "test",
) -> List[Dict[str, Any]]:
    """Load a specific subset of MATH dataset.

    Args:
        subset: Subset name (e.g., "algebra", "geometry")
        split: "train" or "test"

    Returns:
        List of problems
    """
    from datasets import load_dataset

    logger.info(f"Loading {subset} {split}...")
    dataset = load_dataset("EleutherAI/hendrycks_math", subset, split=split)

    data = []
    for item in dataset:
        data.append({
            'question': item['problem'],
            'ground_truth': item['solution'],
            'type': item.get('type', subset),
            'level': item.get('level', ''),
            'subset': subset,
        })

    logger.info(f"  Loaded {len(data)} problems from {subset} {split}")
    return data


def load_math500() -> List[Dict[str, Any]]:
    """Load MATH-500 dataset from HuggingFaceH4/MATH-500.

    Returns:
        List of 500 math problems
    """
    from datasets import load_dataset

    logger.info("Loading HuggingFaceH4/MATH-500...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    data = []
    for item in dataset:
        data.append({
            'question': item['problem'],
            'ground_truth': item['solution'],
            'type': item.get('type', ''),
            'level': item.get('level', ''),
            'subset': 'math500',
        })

    logger.info(f"  Loaded {len(data)} problems from MATH-500")
    return data


def load_gsm8k(split: str = "test") -> List[Dict[str, Any]]:
    """Load GSM8K dataset from openai/gsm8k.

    Args:
        split: "train" or "test"

    Returns:
        List of GSM8K problems
    """
    from datasets import load_dataset
    import re

    logger.info(f"Loading openai/gsm8k main ({split})...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    data = []
    for item in dataset:
        question = item['question']
        answer_raw = item['answer']
        # Extract final answer from "#### <number>" format
        match = re.search(r'####\s*(.+)$', answer_raw)
        if match:
            final_answer = match.group(1).strip()
            # Remove commas from numbers (e.g., "1,234" -> "1234")
            final_answer = final_answer.replace(',', '')
        else:
            final_answer = answer_raw

        data.append({
            'question': question,
            'ground_truth': final_answer,
            'full_solution': answer_raw,
            'subset': 'gsm8k',
        })

    logger.info(f"  Loaded {len(data)} problems from GSM8K ({split})")
    return data


def load_dataset_generic(dataset_name: str, split: str = "test", subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generic dataset loader using DATASET_CONFIGS.

    This function can load any dataset defined in DATASET_CONFIGS.
    For datasets with multiple subsets (like hendrycks_math), specify the subset parameter.

    Args:
        dataset_name: Name of dataset (key in DATASET_CONFIGS)
        split: Data split ("train" or "test")
        subset: Specific subset for multi-subset datasets (e.g., "algebra" for hendrycks_math)

    Returns:
        List of problem dictionaries with 'question', 'ground_truth', 'subset' fields
    """
    from datasets import load_dataset as hf_load_dataset

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]

    # Validate split
    if split not in config["splits"]:
        raise ValueError(f"Split '{split}' not available for {dataset_name}. Available: {config['splits']}")

    # Load from HuggingFace
    hf_path = config["hf_path"]
    hf_subset = config["hf_subset"]

    logger.info(f"Loading {hf_path} (subset={hf_subset}, split={split})...")
    if hf_subset:
        dataset = hf_load_dataset(hf_path, hf_subset, split=split)
    else:
        dataset = hf_load_dataset(hf_path, split=split)

    # Parse each item
    data = []
    question_field = config["question_field"]
    answer_field = config["answer_field"]
    answer_parser = config.get("answer_parser")
    extra_fields = config.get("extra_fields", [])

    for item in dataset:
        # Filter by subset if specified (for multi-subset datasets like hendrycks_math)
        if subset and "type" in item:
            item_type = item["type"].lower().replace(" ", "_")
            if item_type != subset.lower().replace(" ", "_"):
                continue

        question = item[question_field]
        raw_answer = item[answer_field]
        ground_truth = parse_answer(raw_answer, answer_parser)

        entry = {
            'question': question,
            'ground_truth': ground_truth,
            'subset': subset if subset else config["subsets"][0],
        }

        # Preserve full solution for GSM8K-like datasets
        if answer_parser:
            entry['full_solution'] = raw_answer

        # Add extra fields
        for field in extra_fields:
            if field in item:
                entry[field] = item[field]

        data.append(entry)

    logger.info(f"  Loaded {len(data)} problems from {dataset_name} ({split})")
    return data


def load_hendrycks_math_all(split: str = "train") -> List[Dict[str, Any]]:
    """Load all subsets of hendrycks_math merged together.

    Args:
        split: "train" or "test"

    Returns:
        List of all problems from all subsets
    """
    MATH_SUBSETS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    all_data = []
    for subset in MATH_SUBSETS:
        data = load_hendrycks_math_subset(subset, split)
        all_data.extend(data)

    logger.info(f"Total: {len(all_data)} problems from all subsets ({split})")
    return all_data


def collect_data_for_token_level(
    mentor_model: VLLMInference,
    intern_model: VLLMInference,
    data: List[Dict[str, Any]],
    token_level: int,
    batch_size: int = 8,
    use_think: bool = True,
) -> List[Dict[str, Any]]:
    """Collect data for a specific token level.

    ACT-E approach:
    - token_level=-1: Mentor generates full answer (mentor only baseline)
    - token_level=0: Intern generates from scratch
    - token_level>0: Mentor generates first N tokens, then Intern CONTINUES from there

    Args:
        mentor_model: VLLMInference instance for mentor (large model)
        intern_model: VLLMInference instance for intern (small model)
        data: List of problems
        token_level: -1 for mentor only, 0 for intern only, >0 for mentor tokens
        batch_size: Batch size for inference
        use_think: Whether to use think mode

    Returns:
        List of results with responses and correctness
    """
    results = []
    total_batches = (len(data) + batch_size - 1) // batch_size

    # Process in batches
    level_desc = "mentor_only" if token_level == MENTOR_ONLY_LEVEL else f"tokens={token_level}"
    for batch_start in tqdm(range(0, len(data), batch_size), desc=level_desc, total=total_batches, unit="batch", ncols=80):
        batch = data[batch_start:batch_start + batch_size]

        if token_level == MENTOR_ONLY_LEVEL:
            # Mentor only - mentor generates full answer, no intern
            prompts = [mentor_model.build_chat_prompt(item['question'], use_think=use_think) for item in batch]
            responses = mentor_model.generate(prompts)

            for item, response in zip(batch, responses):
                is_correct = check_math_correctness(response, item['ground_truth'])
                mentor_length = len(mentor_model.tokenizer.encode(response)) if response else 0
                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'mentor_tokens': -1,  # indicates mentor only
                    'mentor_response': response,
                    'response': response,
                    'is_correct': is_correct,
                    'mentor_length': mentor_length,
                    'intern_length': 0,  # no intern
                    'subset': item.get('subset', ''),
                    'level': item.get('level', ''),
                })
        elif token_level == 0:
            # No mentor - intern generates from scratch
            prompts = [intern_model.build_chat_prompt(item['question'], use_think=use_think) for item in batch]
            responses = intern_model.generate(prompts)

            for item, response in zip(batch, responses):
                is_correct = check_math_correctness(response, item['ground_truth'])
                # Calculate token length
                intern_length = len(intern_model.tokenizer.encode(response)) if response else 0
                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'mentor_tokens': 0,
                    'mentor_response': '',
                    'response': response,
                    'is_correct': is_correct,
                    'mentor_length': 0,
                    'intern_length': intern_length,
                    'subset': item.get('subset', ''),
                    'level': item.get('level', ''),
                })
        else:
            # Mentor generates first N tokens
            mentor_prompts = [mentor_model.build_chat_prompt(item['question'], use_think=use_think) for item in batch]
            mentor_outputs = mentor_model.generate_mentor_tokens(mentor_prompts, max_tokens=token_level)

            # Intern CONTINUES from mentor's output
            # Check if cross-model scenario (different model families)
            is_cross_model = mentor_model.model_family != intern_model.model_family

            if is_cross_model:
                # Cross-model: build intern-native prompts with mentor's output
                continued_prompts = [
                    build_cross_model_prompt(item['question'], mentor_output, intern_model, use_think)
                    for item, mentor_output in zip(batch, mentor_outputs)
                ]
            else:
                # Same model family: directly concatenate prompt + mentor_output
                continued_prompts = [
                    prompt + mentor_output
                    for prompt, mentor_output in zip(mentor_prompts, mentor_outputs)
                ]

            intern_continuations = intern_model.generate(continued_prompts)

            for item, mentor_output, intern_continuation in zip(batch, mentor_outputs, intern_continuations):
                # Full response = mentor_output + intern_continuation
                full_response = mentor_output + intern_continuation
                is_correct = check_math_correctness(full_response, item['ground_truth'])

                # Calculate token lengths
                mentor_length = len(mentor_model.tokenizer.encode(mentor_output)) if mentor_output else 0
                intern_length = len(intern_model.tokenizer.encode(intern_continuation)) if intern_continuation else 0

                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'mentor_tokens': token_level,
                    'mentor_response': mentor_output,
                    'response': full_response,
                    'is_correct': is_correct,
                    'mentor_length': mentor_length,
                    'intern_length': intern_length,
                    'subset': item.get('subset', ''),
                    'level': item.get('level', ''),
                })

    return results


def worker_process_all_tasks(
    rank: int,
    world_size: int,
    gpu_id: int,
    mentor_model_name: str,
    intern_model_name: str,
    max_model_len: int,
    batch_size: int,
    all_tasks: List[Tuple[str, str, List[Dict[str, Any]]]],  # [(subset, output_dir, data), ...]
    token_levels: List[int],
    use_think: bool = True,
    mentor_gpu_ids: List[int] = None,
    intern_gpu_ids: List[int] = None,
    mentor_memory_util: float = 0.5,
    intern_memory_util: float = 0.3,
    mentor_max_model_len: int = None,
    intern_max_model_len: int = None,
    force: bool = False,
    need_mentor: bool = True,
    need_intern: bool = True,
    mentor_api: str = None,
    openrouter_api_key: str = None,
    api_max_workers: int = 8,
):
    """Worker process that processes ALL subsets and token levels.

    Args:
        rank: Worker rank
        world_size: Total number of workers
        gpu_id: Default GPU ID to use (legacy, prefer mentor_gpu_ids/intern_gpu_ids)
        mentor_model_name: Mentor model name (large model)
        intern_model_name: Intern model name (small model)
        max_model_len: Max model context length
        batch_size: Batch size
        all_tasks: List of (subset_name, output_dir, data) tuples
        token_levels: List of token levels to collect
        use_think: Whether to use think prompt
        mentor_gpu_ids: List of GPU IDs for mentor model (for tensor parallelism)
        intern_gpu_ids: List of GPU IDs for intern model (for tensor parallelism)
        mentor_memory_util: GPU memory utilization for mentor model
        intern_memory_util: GPU memory utilization for intern model
        mentor_api: API type for mentor model (e.g., "openrouter")
        openrouter_api_key: OpenRouter API key
        api_max_workers: Max concurrent API requests
    """
    # Determine GPU IDs for each model
    mentor_gpus = mentor_gpu_ids if mentor_gpu_ids is not None else [gpu_id]
    intern_gpus = intern_gpu_ids if intern_gpu_ids is not None else [gpu_id]

    # Determine max_model_len for each model
    mentor_max_len = mentor_max_model_len if mentor_max_model_len is not None else max_model_len
    intern_max_len = intern_max_model_len if intern_max_model_len is not None else max_model_len

    # Determine memory utilization
    # If GPUs don't overlap, use default (0.9); otherwise use specified values
    mentor_gpu_set = set(mentor_gpus)
    intern_gpu_set = set(intern_gpus)
    gpus_overlap = bool(mentor_gpu_set & intern_gpu_set)

    if not gpus_overlap:
        mentor_mem_util = 0.9
        intern_mem_util = 0.9
        logger.info(f"[Worker {rank}] GPUs don't overlap - memory utilization set to 0.9")
    else:
        mentor_mem_util = mentor_memory_util
        intern_mem_util = intern_memory_util
        logger.info(f"[Worker {rank}] GPUs overlap - memory utilization: mentor={mentor_mem_util}, intern={intern_mem_util}")

    # Only load models that are needed
    mentor_model = None
    intern_model = None

    if need_mentor:
        if mentor_api == "openrouter":
            # Use OpenRouter API for mentor (closed-source models like GPT-4, Claude)
            if not OPENROUTER_AVAILABLE:
                raise ImportError("OpenRouter support not available. Make sure openrouter_inference.py exists.")
            logger.info(f"[Worker {rank}] Loading mentor via OpenRouter API: {mentor_model_name} (max_workers={api_max_workers})...")
            try:
                mentor_model = OpenRouterInference(
                    model_name=mentor_model_name,
                    api_key=openrouter_api_key,
                    max_workers=api_max_workers,
                )
            except Exception as e:
                logger.error(f"[Worker {rank}] Failed to initialize OpenRouter client: {e}")
                raise
        else:
            # Use vLLM for local mentor model
            logger.info(f"[Worker {rank}] Loading mentor model: {mentor_model_name} on GPU {mentor_gpus} (tp={len(mentor_gpus)}, memory_util={mentor_mem_util}, max_len={mentor_max_len})...")
            try:
                mentor_model = VLLMInference(
                    model_name=mentor_model_name,
                    gpu_ids=mentor_gpus,
                    max_model_len=mentor_max_len,
                    gpu_memory_utilization=mentor_mem_util,
                )
            except Exception as e:
                logger.error(f"[Worker {rank}] Failed to load mentor model: {e}")
                logger.error(f"[Worker {rank}] Try: 1) Reduce --mentor-max-model-len (current: {mentor_max_len})")
                logger.error(f"[Worker {rank}]     2) Lower --mentor-memory-util (current: {mentor_mem_util})")
                raise
    else:
        logger.info(f"[Worker {rank}] Skipping mentor model (not needed for token levels {token_levels})")

    if need_intern:
        logger.info(f"[Worker {rank}] Loading intern model: {intern_model_name} on GPU {intern_gpus} (tp={len(intern_gpus)}, memory_util={intern_mem_util}, max_len={intern_max_len})...")
        try:
            intern_model = VLLMInference(
                model_name=intern_model_name,
                gpu_ids=intern_gpus,
                max_model_len=intern_max_len,
                gpu_memory_utilization=intern_mem_util,
            )
        except Exception as e:
            logger.error(f"[Worker {rank}] Failed to load intern model: {e}")
            logger.error(f"[Worker {rank}] Try: 1) Reduce --intern-max-model-len (current: {intern_max_len})")
            logger.error(f"[Worker {rank}]     2) Lower --intern-memory-util (current: {intern_mem_util})")
            raise
    else:
        logger.info(f"[Worker {rank}] Skipping intern model (not needed for token levels {token_levels})")

    logger.info(f"[Worker {rank}] Models loaded, processing {len(all_tasks)} subsets × {len(token_levels)} token levels")

    # Process all tasks
    for subset_name, output_dir, data in all_tasks:
        # Shard data for this worker
        shard_data = [d for i, d in enumerate(data) if i % world_size == rank]

        if not shard_data:
            logger.info(f"[Worker {rank}] No data for subset {subset_name}, skipping")
            continue

        logger.info(f"[Worker {rank}] Processing subset {subset_name}: {len(shard_data)} samples")

        for token_level in token_levels:
            # Check if merged file already exists (skip if it does, unless force is True)
            merged_file = os.path.join(output_dir, f"tokens{token_level}.json")
            if os.path.exists(merged_file) and not force:
                logger.info(f"[Worker {rank}] {subset_name} tokens={token_level} already exists, skipping...")
                continue
            elif os.path.exists(merged_file) and force:
                logger.info(f"[Worker {rank}] {subset_name} tokens={token_level} already exists, but --force is set, will overwrite after collection...")
            
            logger.info(f"[Worker {rank}] {subset_name} tokens={token_level}...")
            try:
                results = collect_data_for_token_level(mentor_model, intern_model, shard_data, token_level, batch_size, use_think=use_think)
            except Exception as e:
                logger.error(f"[Worker {rank}] Error collecting {subset_name} tokens={token_level}: {e}", exc_info=True)
                continue

            correct = sum(1 for r in results if r['is_correct'])
            accuracy = correct / len(results) if results else 0
            logger.info(f"[Worker {rank}] {subset_name} tokens={token_level}: {accuracy:.4f} ({correct}/{len(results)})")

            # Save to temp file
            os.makedirs(output_dir, exist_ok=True)
            temp_file = os.path.join(output_dir, f"tokens{token_level}_rank{rank}.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[Worker {rank}] Saved: {temp_file}")

            # Check if all ranks finished this (subset, token_level) - if so, merge
            all_exist = all(
                os.path.exists(os.path.join(output_dir, f"tokens{token_level}_rank{r}.json"))
                for r in range(world_size)
            )
            if all_exist:
                # Use lock file to prevent race condition
                lock_file = os.path.join(output_dir, f".lock_tokens{token_level}")
                merged_file = os.path.join(output_dir, f"tokens{token_level}.json")
                try:
                    # Try to create lock file (atomic on most filesystems)
                    fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    # We got the lock - do the merge (overwrite if force=True)
                    if not os.path.exists(merged_file) or force:
                        total, correct_cnt, acc = merge_rank_files(output_dir, token_level, world_size)
                        print(f"[MERGED] {subset_name} tokens={token_level}: {total} samples, acc={acc:.4f}", flush=True)
                    os.remove(lock_file)
                except FileExistsError:
                    # Another worker is merging, skip
                    pass

    logger.info(f"[Worker {rank}] All tasks completed")

    # Clean up models before exit
    if mentor_model is not None:
        mentor_model.cleanup()
    if intern_model is not None:
        intern_model.cleanup()

    # Use os._exit to avoid hanging on multiprocessing cleanup
    os._exit(0)


def collect_parallel(
    mentor_model_name: str,
    intern_model_name: str,
    max_model_len: int,
    batch_size: int,
    data: List[Dict[str, Any]],
    token_levels: List[int],
    gpus: List[int],
    mentor_gpu_ids: List[int],
    intern_gpu_ids: List[int],
    output_dir: str,
    use_think: bool = True,
    mentor_memory_util: float = 0.5,
    intern_memory_util: float = 0.3,
    mentor_max_model_len: int = None,
    intern_max_model_len: int = None,
    force: bool = False,
    need_mentor: bool = True,
    need_intern: bool = True,
    mentor_api: str = None,
    openrouter_api_key: str = None,
    api_max_workers: int = 8,
) -> Dict[int, Dict[str, Any]]:
    """Collect data for a single dataset in parallel.

    Returns metadata (count/accuracy/file) for each token level, since the
    actual merged json files are already persisted to disk during merge.
    """
    all_tasks = [("single", output_dir, data)]
    results = collect_all_parallel(
        mentor_model_name=mentor_model_name,
        intern_model_name=intern_model_name,
        max_model_len=max_model_len,
        batch_size=batch_size,
        all_tasks=all_tasks,
        token_levels=token_levels,
        gpus=gpus,
        mentor_gpu_ids=mentor_gpu_ids,
        intern_gpu_ids=intern_gpu_ids,
        use_think=use_think,
        mentor_memory_util=mentor_memory_util,
        intern_memory_util=intern_memory_util,
        mentor_max_model_len=mentor_max_model_len,
        intern_max_model_len=intern_max_model_len,
        force=force,
        need_mentor=need_mentor,
        need_intern=need_intern,
        mentor_api=mentor_api,
        openrouter_api_key=openrouter_api_key,
        api_max_workers=api_max_workers,
    )
    return results.get("single", {})


def merge_rank_files(output_dir: str, token_level: int, world_size: int) -> Tuple[int, int, float]:
    """Merge all rank files for a single token level.

    Args:
        output_dir: Directory containing rank files
        token_level: Token level being merged
        world_size: Number of workers/ranks

    Returns: (total_samples, correct_samples, accuracy)
    """
    merged = []
    for rank in range(world_size):
        temp_file = os.path.join(output_dir, f"tokens{token_level}_rank{rank}.json")
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            merged.extend(results)
            print(f"  [MERGE] Loaded {len(results)} samples from rank {rank}", flush=True)

    if merged:
        correct = sum(1 for r in merged if r['is_correct'])
        accuracy = correct / len(merged)

        output_file = os.path.join(output_dir, f"tokens{token_level}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        # Keep rank files for recovery (they get overwritten on re-collection)

        return len(merged), correct, accuracy
    return 0, 0, 0.0


def collect_all_parallel(
    mentor_model_name: str,
    intern_model_name: str,
    max_model_len: int,
    batch_size: int,
    all_tasks: List[Tuple[str, str, List[Dict[str, Any]]]],
    token_levels: List[int],
    gpus: List[int],
    mentor_gpu_ids: List[int],
    intern_gpu_ids: List[int],
    use_think: bool = True,
    mentor_memory_util: float = 0.5,
    intern_memory_util: float = 0.3,
    mentor_max_model_len: int = None,
    intern_max_model_len: int = None,
    force: bool = False,
    need_mentor: bool = True,
    need_intern: bool = True,
    mentor_api: str = None,
    openrouter_api_key: str = None,
    api_max_workers: int = 8,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Collect data for ALL subsets in parallel.

    Two modes:
    1. Single-model tensor parallelism: When only mentor OR intern is needed,
       use a single worker with all GPUs for tensor parallelism.
    2. Data parallelism: When both models are needed, spawn multiple workers
       with each worker having one GPU for each model.

    Workers merge results immediately after each (subset, token_level) completes.

    Args:
        mentor_gpu_ids: List of GPU IDs for mentor models (can be None if not needed)
        intern_gpu_ids: List of GPU IDs for intern models (can be None if not needed)
        mentor_memory_util: GPU memory utilization for mentor model (default: 0.5)
        intern_memory_util: GPU memory utilization for intern model (default: 0.3)
        need_mentor: Whether mentor model is needed (based on token levels)
        need_intern: Whether intern model is needed (based on token levels)
    """
    # Determine mode: tensor parallelism (single GPU model) vs data parallelism (both GPU models)
    # When mentor uses API, only intern needs GPU -> use tensor parallelism for intern
    mentor_uses_api = mentor_api is not None
    mentor_needs_gpu = need_mentor and not mentor_uses_api

    # Single model mode: only one model needs GPU (tensor parallelism with all GPUs)
    # - Only mentor needs GPU (no intern, or intern not needed)
    # - Only intern needs GPU (mentor uses API, or mentor not needed)
    single_model_mode = (mentor_needs_gpu and not need_intern) or (need_intern and not mentor_needs_gpu)

    if single_model_mode:
        # Single model: use tensor parallelism with all GPUs in one worker
        world_size = 1
        if mentor_needs_gpu:
            tensor_gpus = mentor_gpu_ids if mentor_gpu_ids else gpus
            print(f"\n{'='*60}", flush=True)
            print(f"[MAIN] Single-model tensor parallelism mode (mentor only)", flush=True)
            print(f"[MAIN] Mentor model: {mentor_model_name}", flush=True)
            print(f"[MAIN] Using {len(tensor_gpus)} GPUs for tensor parallelism: {tensor_gpus}", flush=True)
        else:
            tensor_gpus = intern_gpu_ids if intern_gpu_ids else gpus
            print(f"\n{'='*60}", flush=True)
            if mentor_uses_api:
                print(f"[MAIN] Hybrid mode: mentor via API, intern via tensor parallelism", flush=True)
                print(f"[MAIN] Mentor model: {mentor_model_name} (via {mentor_api} API)", flush=True)
            else:
                print(f"[MAIN] Single-model tensor parallelism mode (intern only)", flush=True)
            print(f"[MAIN] Intern model: {intern_model_name}", flush=True)
            print(f"[MAIN] Using {len(tensor_gpus)} GPUs for tensor parallelism: {tensor_gpus}", flush=True)
        print(f"[MAIN] Workers: {world_size}", flush=True)
        print(f"[MAIN] Subsets: {len(all_tasks)}", flush=True)
        print(f"[MAIN] Token levels: {token_levels}", flush=True)
        print(f"{'='*60}\n", flush=True)
    else:
        # Both models need GPU: use data parallelism
        world_size = len(gpus)

        # Validate GPU list lengths (only for models that need GPU)
        if mentor_needs_gpu and mentor_gpu_ids and len(mentor_gpu_ids) != world_size:
            raise ValueError(f"mentor_gpu_ids length ({len(mentor_gpu_ids)}) must match gpus length ({world_size})")
        if need_intern and intern_gpu_ids and len(intern_gpu_ids) != world_size:
            raise ValueError(f"intern_gpu_ids length ({len(intern_gpu_ids)}) must match gpus length ({world_size})")

        print(f"\n{'='*60}", flush=True)
        print(f"[MAIN] Data parallelism mode (both models on GPU)", flush=True)
        print(f"[MAIN] Mentor model: {mentor_model_name} (GPU: {mentor_gpu_ids}, memory_util={mentor_memory_util})", flush=True)
        print(f"[MAIN] Intern model: {intern_model_name} (GPU: {intern_gpu_ids}, memory_util={intern_memory_util})", flush=True)
        print(f"[MAIN] Workers: {world_size}", flush=True)
        print(f"[MAIN] Subsets: {len(all_tasks)}", flush=True)
        print(f"[MAIN] Token levels: {token_levels}", flush=True)
        print(f"{'='*60}\n", flush=True)

    # Set spawn method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Only clean up lock files, keep all data files for recovery
    for subset_name, output_dir, _ in all_tasks:
        os.makedirs(output_dir, exist_ok=True)
        for token_level in token_levels:
            # Only remove lock files to prevent deadlock
            # Keep rank files (tokens{level}_rank{rank}.json) for recovery
            # Keep merged files (tokens{level}.json) - they will be skipped if they exist
            lock_file = os.path.join(output_dir, f".lock_tokens{token_level}")
            if os.path.exists(lock_file):
                os.remove(lock_file)

    # Start workers
    processes = []

    if single_model_mode:
        # Single worker with all GPUs for tensor parallelism
        rank = 0
        gpu_id = tensor_gpus[0]  # First GPU as default
        # Pass full GPU list for tensor parallelism
        mentor_gpu_list = tensor_gpus if need_mentor else None
        intern_gpu_list = tensor_gpus if need_intern else None
        p = mp.Process(
            target=worker_process_all_tasks,
            args=(rank, world_size, gpu_id, mentor_model_name, intern_model_name, max_model_len, batch_size, all_tasks, token_levels, use_think, mentor_gpu_list, intern_gpu_list, mentor_memory_util, intern_memory_util, mentor_max_model_len, intern_max_model_len, force, need_mentor, need_intern, mentor_api, openrouter_api_key, api_max_workers)
        )
        p.start()
        processes.append(p)
        if need_mentor:
            if mentor_api:
                print(f"[MAIN] Started worker 0 (mentor via {mentor_api} API, PID: {p.pid})", flush=True)
            else:
                print(f"[MAIN] Started worker 0 (mentor on GPUs {tensor_gpus}, tp={len(tensor_gpus)}, PID: {p.pid})", flush=True)
        else:
            print(f"[MAIN] Started worker 0 (intern on GPUs {tensor_gpus}, tp={len(tensor_gpus)}, PID: {p.pid})", flush=True)
    else:
        # Multiple workers, each with one GPU per model
        for rank, gpu_id in enumerate(gpus):
            mentor_gpu = [mentor_gpu_ids[rank]] if mentor_gpu_ids else [gpu_id]
            intern_gpu = [intern_gpu_ids[rank]] if intern_gpu_ids else [gpu_id]
            p = mp.Process(
                target=worker_process_all_tasks,
                args=(rank, world_size, gpu_id, mentor_model_name, intern_model_name, max_model_len, batch_size, all_tasks, token_levels, use_think, mentor_gpu, intern_gpu, mentor_memory_util, intern_memory_util, mentor_max_model_len, intern_max_model_len, force, need_mentor, need_intern, mentor_api, openrouter_api_key, api_max_workers)
            )
            p.start()
            processes.append(p)
            gpu_info = []
            if need_mentor:
                gpu_info.append(f"mentor GPU {mentor_gpu}")
            if need_intern:
                gpu_info.append(f"intern GPU {intern_gpu}")
            print(f"[MAIN] Started worker {rank} ({', '.join(gpu_info)}, PID: {p.pid})", flush=True)

    print(f"\n[MAIN] All {world_size} workers started. Waiting...\n", flush=True)

    # Wait for all workers
    for p in processes:
        p.join()

    print(f"\n{'='*60}", flush=True)
    print(f"[MAIN] All workers finished.", flush=True)
    print(f"{'='*60}\n", flush=True)

    return {}


def main():
    parser = argparse.ArgumentParser(description="Collect data with vLLM and thinking prompt")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Model name (legacy, use --mentor-model and --intern-model instead)")
    parser.add_argument("--mentor-model", type=str, default=None,
                        help="Mentor model name (large model, e.g., 32B). If not set, uses --model")
    parser.add_argument("--intern-model", type=str, default=None,
                        help="Intern model name (small model, e.g., 7B). If not set, uses --model")
    parser.add_argument("--dataset", type=str, default="hendrycks_math",
                        choices=["hendrycks_math", "math500", "hendrycks_math_all", "gsm8k"],
                        help="Dataset: hendrycks_math (by subset), math500 (MATH-500), hendrycks_math_all (all subsets merged), gsm8k (GSM8K)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset for hendrycks_math (e.g., algebra). If None, process all subsets")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Split for hendrycks_math/hendrycks_math_all (ignored for math500)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name for output directory (e.g., R1_m32B_i7B). If not set, uses model name.")
    parser.add_argument("--token-levels", type=str, default="-1,0,100,500,1000",
                        help="Comma-separated token levels to collect")
    # Parallel mode arguments
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated list of worker GPUs. Auto-inferred from --mentor-gpus/--intern-gpus if not specified.")
    parser.add_argument("--mentor-gpus", type=str, default=None,
                        help="Comma-separated list of GPUs for mentor model (e.g., '0,1,2,3,4,5,6,7').")
    parser.add_argument("--intern-gpus", type=str, default=None,
                        help="Comma-separated list of GPUs for intern model (e.g., '0,1,2,3,4,5,6,7').")
    parser.add_argument("--mentor-memory-util", type=float, default=0.5,
                        help="GPU memory utilization for mentor model (default: 0.5, recommended: 0.4-0.6 for 32B models)")
    parser.add_argument("--intern-memory-util", type=float, default=0.3,
                        help="GPU memory utilization for intern model (default: 0.3, recommended: 0.2-0.4 for 7B models)")
    parser.add_argument("--mentor-max-model-len", type=int, default=None,
                        help="Max model length for mentor model (if None, uses --max-model-len)")
    parser.add_argument("--intern-max-model-len", type=int, default=None,
                        help="Max model length for intern model (if None, uses --max-model-len)")
    # Think mode control
    parser.add_argument("--no-think", action="store_true",
                        help="Disable structured thinking prompt (use standard prompt)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-collection even if data files already exist")
    # OpenRouter API support for closed-source mentor models
    parser.add_argument("--mentor-api", type=str, default=None,
                        choices=["openrouter"],
                        help="Use API for mentor model (e.g., 'openrouter' for GPT-4, Claude, etc.)")
    parser.add_argument("--openrouter-api-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--api-max-workers", type=int, default=8,
                        help="Max concurrent API requests for OpenRouter (default: 8)")

    args = parser.parse_args()

    # Determine mentor and intern models
    mentor_model = args.mentor_model if args.mentor_model else args.model
    intern_model = args.intern_model if args.intern_model else args.model
    
    if mentor_model != intern_model:
        logger.info(f"Using different models: Mentor={mentor_model}, Intern={intern_model}")
    else:
        logger.info(f"Using same model for both: {mentor_model}")

    # Determine if using think mode
    use_think = not args.no_think

    # Parse token levels
    token_levels = [int(x) for x in args.token_levels.split(",")]

    # Determine which models are needed based on token levels
    need_mentor = any(t == -1 or t > 0 for t in token_levels)  # -1 = mentor only, >0 = mentor + intern
    need_intern = any(t == 0 or t > 0 for t in token_levels)   # 0 = intern only, >0 = mentor + intern

    logger.info(f"Token levels {token_levels}: need_mentor={need_mentor}, need_intern={need_intern}")

    # Auto-calculate mentor max_model_len if not specified
    # When mentor only generates partial tokens (not full answer), it needs much less context
    if args.mentor_max_model_len is None and need_mentor:
        mentor_only_mode = -1 in token_levels  # -1 means mentor generates full answer
        if mentor_only_mode:
            # Mentor generates full answer, use default max_model_len
            args.mentor_max_model_len = args.max_model_len
            logger.info(f"Mentor in full-generation mode, using max_model_len={args.mentor_max_model_len}")
        else:
            # Mentor only generates partial tokens, calculate optimal length
            # max_tokens needed = max(token_levels) for generation
            # plus ~1024 buffer for prompt (question + system prompt + chat template)
            max_mentor_tokens = max(t for t in token_levels if t > 0)
            # Buffer for prompt: ~512 tokens for question, ~256 for system/template
            prompt_buffer = 1024
            optimal_mentor_len = max_mentor_tokens + prompt_buffer
            # Round up to nearest power of 2 for efficiency, min 2048
            optimal_mentor_len = max(2048, 2 ** (optimal_mentor_len - 1).bit_length())
            args.mentor_max_model_len = optimal_mentor_len
            logger.info(f"Mentor in partial-generation mode (max {max_mentor_tokens} tokens), auto-set max_model_len={args.mentor_max_model_len}")

    # Parse GPU lists
    # Parse --gpus if specified
    gpus_from_arg = None
    if args.gpus is not None and args.gpus.strip():
        gpus_from_arg = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    # Parse mentor/intern GPU lists if specified
    mentor_gpu_ids = None
    intern_gpu_ids = None

    if args.mentor_gpus is not None and args.mentor_gpus.strip():
        mentor_gpu_ids = [int(g.strip()) for g in args.mentor_gpus.split(",") if g.strip()]
    if args.intern_gpus is not None and args.intern_gpus.strip():
        intern_gpu_ids = [int(g.strip()) for g in args.intern_gpus.split(",") if g.strip()]

    # Auto-detect available GPUs if none specified
    def get_available_gpus():
        """Detect available GPUs from CUDA_VISIBLE_DEVICES or nvidia-smi."""
        import subprocess
        # First check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            return [int(g.strip()) for g in cuda_visible.split(",") if g.strip()]
        # Otherwise use nvidia-smi to count GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
        except Exception:
            pass
        # Default to GPU 0
        return [0]

    # Check if mentor uses API (doesn't need GPU)
    mentor_uses_api = args.mentor_api is not None
    mentor_needs_gpu = need_mentor and not mentor_uses_api

    # Determine worker GPUs based on what models NEED GPU
    if mentor_needs_gpu and not need_intern:
        # Only mentor needs GPU (no intern)
        if mentor_gpu_ids:
            gpus = mentor_gpu_ids
            logger.info(f"Only mentor needs GPU, using --mentor-gpus: {gpus}")
        elif gpus_from_arg:
            gpus = gpus_from_arg
            mentor_gpu_ids = gpus
            logger.info(f"Only mentor needs GPU, using --gpus: {gpus}")
        else:
            # Auto-detect GPUs
            gpus = get_available_gpus()
            mentor_gpu_ids = gpus
            logger.info(f"Only mentor needs GPU, auto-detected GPUs: {gpus}")
    elif need_intern and not mentor_needs_gpu:
        # Only intern needs GPU (mentor uses API or not needed)
        if intern_gpu_ids:
            gpus = intern_gpu_ids
            logger.info(f"Only intern needs GPU, using --intern-gpus: {gpus}")
        elif gpus_from_arg:
            gpus = gpus_from_arg
            intern_gpu_ids = gpus
            if mentor_uses_api:
                logger.info(f"Mentor via API, intern uses all GPUs: {gpus}")
            else:
                logger.info(f"Only intern needs GPU, using --gpus: {gpus}")
        else:
            # Auto-detect GPUs
            gpus = get_available_gpus()
            intern_gpu_ids = gpus
            if mentor_uses_api:
                logger.info(f"Mentor via API, intern uses all auto-detected GPUs: {gpus}")
            else:
                logger.info(f"Only intern needs GPU, auto-detected GPUs: {gpus}")
    elif mentor_needs_gpu and need_intern:
        # Both models need GPU - split GPUs
        if mentor_gpu_ids and intern_gpu_ids:
            # Both explicitly specified
            gpus = list(range(len(mentor_gpu_ids)))
            logger.info(f"Using explicit GPU assignment: mentor={mentor_gpu_ids}, intern={intern_gpu_ids}")
        elif gpus_from_arg:
            # Auto-split GPUs in half for mentor and intern
            gpus = gpus_from_arg
            num_gpus = len(gpus)
            if num_gpus >= 2:
                half = num_gpus // 2
                mentor_gpu_ids = gpus[:half]
                intern_gpu_ids = gpus[half:]
                logger.info(f"Auto-split {num_gpus} GPUs: mentor={mentor_gpu_ids}, intern={intern_gpu_ids}")
            else:
                # Single GPU: both models share it
                mentor_gpu_ids = gpus
                intern_gpu_ids = gpus
                logger.info(f"Single GPU mode: both models on GPU {gpus}")
        else:
            # Auto-detect and split
            available_gpus = get_available_gpus()
            num_gpus = len(available_gpus)
            if num_gpus >= 2:
                half = num_gpus // 2
                mentor_gpu_ids = available_gpus[:half]
                intern_gpu_ids = available_gpus[half:]
                gpus = list(range(half))  # Worker count = half (each worker has one mentor GPU + one intern GPU)
                logger.info(f"Auto-detected {num_gpus} GPUs, split: mentor={mentor_gpu_ids}, intern={intern_gpu_ids}")
            else:
                # Single GPU: both models share it
                gpus = available_gpus
                mentor_gpu_ids = available_gpus
                intern_gpu_ids = available_gpus
                logger.info(f"Auto-detected single GPU: both models on GPU {gpus}")

        # Validate GPU list lengths match for data parallelism
        if len(mentor_gpu_ids) != len(intern_gpu_ids):
            raise ValueError(f"mentor_gpu_ids length ({len(mentor_gpu_ids)}) must match intern_gpu_ids length ({len(intern_gpu_ids)})")
        gpus = list(range(len(mentor_gpu_ids)))
    else:
        # Neither model needs GPU (shouldn't happen, but handle gracefully)
        gpus = gpus_from_arg if gpus_from_arg else get_available_gpus()
        logger.warning(f"Neither model needs GPU? Using: {gpus}")

    # Set output directory (default: server path)
    # Build experiment name from models
    if args.exp_name:
        exp_name = args.exp_name
    elif mentor_model != intern_model:
        # Different models: include both in path
        mentor_short = mentor_model.split('/')[-1]
        intern_short = intern_model.split('/')[-1]
        exp_name = f"m{mentor_short}_i{intern_short}"
    else:
        # Same model: use single model name
        exp_name = args.model.split('/')[-1]
    
    mode_suffix = "think" if use_think else "standard"
    if args.output_dir is None:
        base_dir = "/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected"
        if args.dataset == "math500":
            args.output_dir = f"{base_dir}/math500_{mode_suffix}_{exp_name}"
        elif args.dataset == "hendrycks_math_all":
            args.output_dir = f"{base_dir}/hendrycks_math_all_{mode_suffix}_{exp_name}"
        elif args.dataset == "gsm8k":
            args.output_dir = f"{base_dir}/gsm8k_{mode_suffix}_{exp_name}"
        else:
            # Default: hendrycks_math
            args.output_dir = f"{base_dir}/hendrycks_math_split_{mode_suffix}_{exp_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Initialize single model cache for reusing -1 and 0 token level results
    single_model_cache = None
    if CACHE_AVAILABLE:
        cache_base_dir = "/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments"
        single_model_cache = SingleModelCache(cache_base_dir)
        logger.info(f"Single model cache enabled: {single_model_cache.cache_dir}")

    # Define subsets
    MATH_SUBSETS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    def collect_and_save(data: List[Dict[str, Any]], output_subdir: str, subset_name: str = None, split: str = None):
        """Helper to collect data and save results.

        Intelligently batches token levels by their GPU requirements:
        - Mentor-only (-1): tensor parallelism with all GPUs
        - Intern-only (0): tensor parallelism with all GPUs
        - Both models (>0): data parallelism with each GPU running both models

        Uses single model cache for -1 and 0 token levels to avoid redundant computation.
        """
        os.makedirs(output_subdir, exist_ok=True)

        all_stats = {}

        # Split token levels by mode
        mentor_only_levels = [t for t in token_levels if t == -1]
        intern_only_levels = [t for t in token_levels if t == 0]
        both_models_levels = [t for t in token_levels if t > 0]

        # Check cache for single-model levels
        cached_mentor_levels = []
        cached_intern_levels = []

        if single_model_cache and subset_name and split:
            # Check mentor-only cache (-1)
            for t in mentor_only_levels:
                cache_stats = check_and_use_cache(
                    single_model_cache, args.dataset, mode_suffix, mentor_model,
                    subset_name, split, output_subdir, t, args.force
                )
                if cache_stats:
                    logger.info(f"  [CACHE HIT] tokens={t} (mentor): {cache_stats['accuracy']:.4f} ({cache_stats['correct']}/{cache_stats['total']})")
                    all_stats[t] = cache_stats
                    cached_mentor_levels.append(t)

            # Check intern-only cache (0)
            for t in intern_only_levels:
                cache_stats = check_and_use_cache(
                    single_model_cache, args.dataset, mode_suffix, intern_model,
                    subset_name, split, output_subdir, t, args.force
                )
                if cache_stats:
                    logger.info(f"  [CACHE HIT] tokens={t} (intern): {cache_stats['accuracy']:.4f} ({cache_stats['correct']}/{cache_stats['total']})")
                    all_stats[t] = cache_stats
                    cached_intern_levels.append(t)

        # Remove cached levels from processing
        mentor_only_levels = [t for t in mentor_only_levels if t not in cached_mentor_levels]
        intern_only_levels = [t for t in intern_only_levels if t not in cached_intern_levels]

        # Process mentor-only levels (tensor parallelism)
        if mentor_only_levels:
            logger.info(f"Processing mentor-only levels {mentor_only_levels} with tensor parallelism...")
            stats = collect_parallel(
                mentor_model_name=mentor_model,
                intern_model_name=intern_model,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                data=data,
                token_levels=mentor_only_levels,
                gpus=gpus,
                mentor_gpu_ids=mentor_gpu_ids,
                intern_gpu_ids=intern_gpu_ids,
                output_dir=output_subdir,
                use_think=use_think,
                mentor_memory_util=args.mentor_memory_util,
                intern_memory_util=args.intern_memory_util,
                mentor_max_model_len=args.mentor_max_model_len,
                intern_max_model_len=args.intern_max_model_len,
                force=args.force,
                need_mentor=True,
                need_intern=False,
                mentor_api=args.mentor_api,
                openrouter_api_key=args.openrouter_api_key,
                api_max_workers=args.api_max_workers,
            )
            all_stats.update(stats)

            # Save to cache
            if single_model_cache and subset_name and split:
                for t in mentor_only_levels:
                    output_file = os.path.join(output_subdir, f"tokens{t}.json")
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            results = json.load(f)
                        save_to_cache(single_model_cache, results, args.dataset, mode_suffix, mentor_model, subset_name, split)
                        logger.info(f"  [CACHE SAVE] tokens={t} (mentor) -> {get_model_short_name(mentor_model)}/{subset_name}/{split}")

        # Process intern-only levels (tensor parallelism)
        if intern_only_levels:
            logger.info(f"Processing intern-only levels {intern_only_levels} with tensor parallelism...")
            stats = collect_parallel(
                mentor_model_name=mentor_model,
                intern_model_name=intern_model,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                data=data,
                token_levels=intern_only_levels,
                gpus=gpus,
                mentor_gpu_ids=mentor_gpu_ids,
                intern_gpu_ids=intern_gpu_ids,
                output_dir=output_subdir,
                use_think=use_think,
                mentor_memory_util=args.mentor_memory_util,
                intern_memory_util=args.intern_memory_util,
                mentor_max_model_len=args.mentor_max_model_len,
                intern_max_model_len=args.intern_max_model_len,
                force=args.force,
                need_mentor=False,
                need_intern=True,
                mentor_api=args.mentor_api,
                openrouter_api_key=args.openrouter_api_key,
                api_max_workers=args.api_max_workers,
            )
            all_stats.update(stats)

            # Save to cache
            if single_model_cache and subset_name and split:
                for t in intern_only_levels:
                    output_file = os.path.join(output_subdir, f"tokens{t}.json")
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            results = json.load(f)
                        save_to_cache(single_model_cache, results, args.dataset, mode_suffix, intern_model, subset_name, split)
                        logger.info(f"  [CACHE SAVE] tokens={t} (intern) -> {get_model_short_name(intern_model)}/{subset_name}/{split}")

        # Process both-models levels (data parallelism)
        if both_models_levels:
            logger.info(f"Processing both-models levels {both_models_levels} with data parallelism...")
            stats = collect_parallel(
                mentor_model_name=mentor_model,
                intern_model_name=intern_model,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                data=data,
                token_levels=both_models_levels,
                gpus=gpus,
                mentor_gpu_ids=mentor_gpu_ids,
                intern_gpu_ids=intern_gpu_ids,
                output_dir=output_subdir,
                use_think=use_think,
                mentor_memory_util=args.mentor_memory_util,
                intern_memory_util=args.intern_memory_util,
                mentor_max_model_len=args.mentor_max_model_len,
                intern_max_model_len=args.intern_max_model_len,
                force=args.force,
                need_mentor=True,
                need_intern=True,
                mentor_api=args.mentor_api,
                openrouter_api_key=args.openrouter_api_key,
                api_max_workers=args.api_max_workers,
            )
            all_stats.update(stats)

        for token_level in token_levels:
            token_stats = all_stats.get(token_level)
            if token_stats:
                logger.info(
                    "  tokens=%s: %.4f (%d/%d) saved to %s",
                    token_level,
                    token_stats['accuracy'],
                    token_stats['correct'],
                    token_stats['total'],
                    token_stats['output_file'],
                )
            else:
                logger.warning("  tokens=%s: no merged results found", token_level)

    logger.info(f"Parallel mode with {len(gpus)} GPUs: {gpus}")
    logger.info(f"Prompt mode: {'THINK (structured)' if use_think else 'STANDARD (no think)'}")

    if args.dataset == "math500":
        # MATH-500 dataset
        logger.info(f"\n{'='*60}")
        logger.info("Processing MATH-500")
        logger.info(f"{'='*60}")

        data = load_math500()
        output_subdir = os.path.join(args.output_dir, "math500", "test")
        collect_and_save(data, output_subdir, subset_name="math500", split="test")

    elif args.dataset == "hendrycks_math_all":
        # All hendrycks_math subsets merged
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing hendrycks_math_all ({args.split})")
        logger.info(f"{'='*60}")

        data = load_hendrycks_math_all(args.split)
        output_subdir = os.path.join(args.output_dir, "all", args.split)
        collect_and_save(data, output_subdir, subset_name="all", split=args.split)

    elif args.dataset == "gsm8k":
        # GSM8K dataset
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing GSM8K ({args.split})")
        logger.info(f"{'='*60}")

        data = load_gsm8k(args.split)
        output_subdir = os.path.join(args.output_dir, "gsm8k", args.split)
        collect_and_save(data, output_subdir, subset_name="gsm8k", split=args.split)

    else:
        # hendrycks_math by subset
        subsets = [args.subset] if args.subset else MATH_SUBSETS

        # First, check and use cache for single-model levels (-1 and 0)
        cache_hits = {"mentor": [], "intern": []}
        if single_model_cache and not args.force:
            logger.info(f"\n{'='*60}")
            logger.info("Checking single model cache...")
            logger.info(f"{'='*60}")

            for subset in subsets:
                output_subdir = os.path.join(args.output_dir, subset, args.split)
                os.makedirs(output_subdir, exist_ok=True)

                # Check mentor cache (-1)
                if -1 in token_levels:
                    cache_stats = check_and_use_cache(
                        single_model_cache, args.dataset, mode_suffix, mentor_model,
                        subset, args.split, output_subdir, -1, args.force
                    )
                    if cache_stats:
                        logger.info(f"  [CACHE HIT] {subset}/tokens-1 (mentor): {cache_stats['accuracy']:.4f}")
                        cache_hits["mentor"].append(subset)

                # Check intern cache (0)
                if 0 in token_levels:
                    cache_stats = check_and_use_cache(
                        single_model_cache, args.dataset, mode_suffix, intern_model,
                        subset, args.split, output_subdir, 0, args.force
                    )
                    if cache_stats:
                        logger.info(f"  [CACHE HIT] {subset}/tokens0 (intern): {cache_stats['accuracy']:.4f}")
                        cache_hits["intern"].append(subset)

            logger.info(f"Cache hits: mentor={len(cache_hits['mentor'])}/{len(subsets)}, intern={len(cache_hits['intern'])}/{len(subsets)}")

        # Check if all data files already exist (after cache check)
        if not args.force:
            all_exist = True
            missing_files = []
            for subset in subsets:
                output_subdir = os.path.join(args.output_dir, subset, args.split)
                for token_level in token_levels:
                    merged_file = os.path.join(output_subdir, f"tokens{token_level}.json")
                    if not os.path.exists(merged_file):
                        all_exist = False
                        missing_files.append(f"{subset}/{args.split}/tokens{token_level}.json")

            if all_exist:
                logger.info(f"\n{'='*60}")
                logger.info(f"All data files already exist for split={args.split}")
                logger.info(f"Subsets: {subsets}")
                logger.info(f"Token levels: {token_levels}")
                logger.info(f"Skipping data collection. Use --force to re-collect.")
                logger.info(f"{'='*60}\n")
                return
            else:
                logger.info(f"\n{'='*60}")
                logger.info(f"Some data files are missing for split={args.split}")
                logger.info(f"Missing files ({len(missing_files)}): {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}")
                logger.info(f"Proceeding with data collection...")
                logger.info(f"{'='*60}\n")

        # Determine which token levels still need computation
        # Exclude levels that are fully cached for all subsets
        remaining_token_levels = []
        for t in token_levels:
            if t == -1:
                if len(cache_hits["mentor"]) < len(subsets):
                    remaining_token_levels.append(t)
            elif t == 0:
                if len(cache_hits["intern"]) < len(subsets):
                    remaining_token_levels.append(t)
            else:
                remaining_token_levels.append(t)

        if not remaining_token_levels:
            logger.info("All token levels fully served from cache!")
            return

        logger.info(f"Token levels to compute: {remaining_token_levels}")

        # Always use parallel mode: load all subsets and process together (ONE model init per GPU)
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading all {len(subsets)} subsets for parallel processing...")
        logger.info(f"{'='*60}")

        all_tasks = []
        for subset in subsets:
            data = load_hendrycks_math_subset(subset, args.split)
            output_subdir = os.path.join(args.output_dir, subset, args.split)
            all_tasks.append((subset, output_subdir, data))

        logger.info(f"Total samples across all subsets: {sum(len(t[2]) for t in all_tasks)}")

        # Recalculate need_mentor and need_intern based on remaining levels
        remaining_need_mentor = any(t == -1 or t > 0 for t in remaining_token_levels)
        remaining_need_intern = any(t == 0 or t > 0 for t in remaining_token_levels)

        collect_all_parallel(
            mentor_model_name=mentor_model,
            intern_model_name=intern_model,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            all_tasks=all_tasks,
            token_levels=remaining_token_levels,
            gpus=gpus,
            mentor_gpu_ids=mentor_gpu_ids,
            intern_gpu_ids=intern_gpu_ids,
            use_think=use_think,
            mentor_memory_util=args.mentor_memory_util,
            intern_memory_util=args.intern_memory_util,
            mentor_max_model_len=args.mentor_max_model_len,
            intern_max_model_len=args.intern_max_model_len,
            force=args.force,
            need_mentor=remaining_need_mentor,
            need_intern=remaining_need_intern,
            mentor_api=args.mentor_api,
            openrouter_api_key=args.openrouter_api_key,
            api_max_workers=args.api_max_workers,
        )

        # Save newly computed single-model results to cache
        if single_model_cache:
            logger.info(f"\n{'='*60}")
            logger.info("Saving new results to single model cache...")
            logger.info(f"{'='*60}")

            for subset in subsets:
                output_subdir = os.path.join(args.output_dir, subset, args.split)

                # Save mentor results (-1) if newly computed
                if -1 in remaining_token_levels and subset not in cache_hits["mentor"]:
                    output_file = os.path.join(output_subdir, "tokens-1.json")
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            results = json.load(f)
                        save_to_cache(single_model_cache, results, args.dataset, mode_suffix, mentor_model, subset, args.split)
                        logger.info(f"  [CACHE SAVE] {subset}/tokens-1 -> {get_model_short_name(mentor_model)}")

                # Save intern results (0) if newly computed
                if 0 in remaining_token_levels and subset not in cache_hits["intern"]:
                    output_file = os.path.join(output_subdir, "tokens0.json")
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            results = json.load(f)
                        save_to_cache(single_model_cache, results, args.dataset, mode_suffix, intern_model, subset, args.split)
                        logger.info(f"  [CACHE SAVE] {subset}/tokens0 -> {get_model_short_name(intern_model)}")

    logger.info("\nData collection complete!")

    # Force exit to avoid hanging on cleanup
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()
