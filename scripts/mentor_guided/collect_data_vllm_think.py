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
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Token levels to collect (0 = no mentor, just intern)
TOKEN_LEVELS = [0, 100, 500, 1000]

# Structured thinking prompt with Goal, Planning, Retrieval, Action framework
THINKING_SYSTEM_PROMPT = """You are a mathematical reasoning expert. When solving problems, structure your thinking using the following framework:

<think>
**Goal (I₁)**: Define the ultimate objective or question to be solved. Clarify what you aim to achieve.

**Planning (I₂)**: Outline the high-level reasoning strategy. Decompose subproblems and select solution paths.

**Retrieval (I₃)**: Recall relevant knowledge, facts, formulas, or contextual information necessary for solving.

**Action (I₄)**: Execute concrete reasoning steps, calculations, or logical operations leading to the answer.
</think>

After your reasoning, provide the final answer in \\boxed{}.
"""

THINKING_SYSTEM_PROMPT_WITH_HINT = """You are a mathematical reasoning expert. When solving problems, structure your thinking using the following framework:

<think>
**Goal (I₁)**: Define the ultimate objective or question to be solved. Clarify what you aim to achieve.

**Planning (I₂)**: Outline the high-level reasoning strategy. Decompose subproblems and select solution paths.

**Retrieval (I₃)**: Recall relevant knowledge, facts, formulas, or contextual information necessary for solving.

**Action (I₄)**: Execute concrete reasoning steps, calculations, or logical operations leading to the answer.
</think>

You are also provided with a hint from a mentor model. Use the hint to guide your reasoning, but still follow the structured thinking framework above.

After your reasoning, provide the final answer in \\boxed{}.
"""


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


def check_math_correctness(prediction: str, ground_truth: str) -> bool:
    """Check if math answer is correct using the official grader."""
    pred_answer = extract_boxed_answer(prediction)
    true_answer = extract_boxed_answer(ground_truth)

    if not pred_answer or not true_answer:
        return False

    return grade_answer(pred_answer, true_answer)


class VLLMInference:
    """vLLM-based inference with chat template support."""

    def __init__(
        self,
        model_name: str,
        gpu_id: int = 0,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
    ):
        """Initialize vLLM model.

        Args:
            model_name: HuggingFace model name
            gpu_id: GPU ID to use
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum model context length
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is required. Install with: pip install vllm")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logger.info(f"Loading model {model_name} with vLLM on GPU {gpu_id}...")

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = self.model.get_tokenizer()
        self.SamplingParams = SamplingParams
        logger.info("Model loaded successfully")

    def build_chat_prompt(
        self,
        question: str,
        hint: Optional[str] = None,
    ) -> str:
        """Build chat prompt with template.

        Args:
            question: The math problem
            hint: Optional mentor hint/reasoning

        Returns:
            Formatted prompt string
        """
        if hint:
            system_prompt = THINKING_SYSTEM_PROMPT_WITH_HINT
            user_content = f"Problem: {question}\n\nHint from mentor:\n{hint}"
        else:
            system_prompt = THINKING_SYSTEM_PROMPT
            user_content = f"Problem: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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

        outputs = self.model.generate(prompts, sampling_params)

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

        outputs = self.model.generate(prompts, sampling_params)

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


def collect_data_for_token_level(
    model: VLLMInference,
    data: List[Dict[str, Any]],
    token_level: int,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Collect data for a specific token level.

    Args:
        model: VLLMInference instance
        data: List of problems
        token_level: 0 for intern only, >0 for mentor tokens
        batch_size: Batch size for inference

    Returns:
        List of results with responses and correctness
    """
    results = []

    # Process in batches
    for batch_start in tqdm(range(0, len(data), batch_size), desc=f"tokens={token_level}"):
        batch = data[batch_start:batch_start + batch_size]

        if token_level == 0:
            # No mentor, just generate with structured thinking
            prompts = [model.build_chat_prompt(item['question']) for item in batch]
            responses = model.generate(prompts)

            for item, response in zip(batch, responses):
                is_correct = check_math_correctness(response, item['ground_truth'])
                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'mentor_tokens': 0,
                    'mentor_response': '',
                    'response': response,
                    'is_correct': is_correct,
                    'subset': item.get('subset', ''),
                    'level': item.get('level', ''),
                })
        else:
            # First generate mentor hints
            mentor_prompts = [model.build_chat_prompt(item['question']) for item in batch]
            mentor_hints = model.generate_mentor_tokens(mentor_prompts, max_tokens=token_level)

            # Then generate with hints
            prompts_with_hints = [
                model.build_chat_prompt(item['question'], hint=hint)
                for item, hint in zip(batch, mentor_hints)
            ]
            responses = model.generate(prompts_with_hints)

            for item, hint, response in zip(batch, mentor_hints, responses):
                full_response = hint + response
                is_correct = check_math_correctness(full_response, item['ground_truth'])
                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'mentor_tokens': token_level,
                    'mentor_response': hint,
                    'response': response,
                    'is_correct': is_correct,
                    'subset': item.get('subset', ''),
                    'level': item.get('level', ''),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect data with vLLM and thinking prompt")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Model name")
    parser.add_argument("--dataset", type=str, default="hendrycks_math",
                        choices=["hendrycks_math"])
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset (e.g., algebra). If None, process all subsets")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model context length")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--token-levels", type=str, default="0,100,500,1000",
                        help="Comma-separated token levels to collect")

    args = parser.parse_args()

    # Parse token levels
    token_levels = [int(x) for x in args.token_levels.split(",")]

    # Set output directory (default: server path)
    if args.output_dir is None:
        model_name = args.model.split('/')[-1]
        args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_{model_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Initialize model
    model = VLLMInference(
        model_name=args.model,
        gpu_id=args.gpu,
        max_model_len=args.max_model_len,
    )

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

    subsets = [args.subset] if args.subset else MATH_SUBSETS

    # Process each subset
    for subset in subsets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing subset: {subset}")
        logger.info(f"{'='*60}")

        # Load data
        data = load_hendrycks_math_subset(subset, args.split)

        # Create output directory for this subset
        subset_dir = os.path.join(args.output_dir, subset, args.split)
        os.makedirs(subset_dir, exist_ok=True)

        # Collect data for each token level
        for token_level in token_levels:
            logger.info(f"\nCollecting data for tokens={token_level}...")

            results = collect_data_for_token_level(
                model, data, token_level, args.batch_size
            )

            # Calculate accuracy
            correct = sum(1 for r in results if r['is_correct'])
            accuracy = correct / len(results) if results else 0
            logger.info(f"  Accuracy: {accuracy:.4f} ({correct}/{len(results)})")

            # Save results
            output_file = os.path.join(subset_dir, f"tokens{token_level}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved to {output_file}")

        # Summary for this subset
        logger.info(f"\nSubset {subset} summary:")
        for token_level in token_levels:
            output_file = os.path.join(subset_dir, f"tokens{token_level}.json")
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    results = json.load(f)
                correct = sum(1 for r in results if r['is_correct'])
                accuracy = correct / len(results) if results else 0
                logger.info(f"  tokens={token_level}: {accuracy:.4f}")

    logger.info("\nData collection complete!")


if __name__ == "__main__":
    main()
