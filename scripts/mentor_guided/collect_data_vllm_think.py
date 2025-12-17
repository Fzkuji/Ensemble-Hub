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


def worker_process(
    rank: int,
    world_size: int,
    gpu_id: int,
    model_name: str,
    max_model_len: int,
    batch_size: int,
    data: List[Dict[str, Any]],
    token_levels: List[int],
    result_queue: mp.Queue,
):
    """Worker process for parallel data collection.

    Each worker loads its own vLLM model and processes a shard of the data.
    """
    # Shard data for this worker
    shard_data = [d for i, d in enumerate(data) if i % world_size == rank]

    if not shard_data:
        result_queue.put((rank, {}))
        return

    logger.info(f"[Worker {rank}] GPU {gpu_id}: Processing {len(shard_data)} samples")

    # Initialize model on this GPU
    model = VLLMInference(
        model_name=model_name,
        gpu_id=gpu_id,
        max_model_len=max_model_len,
    )

    # Collect data for each token level
    results_by_level = {}
    for token_level in token_levels:
        logger.info(f"[Worker {rank}] Collecting tokens={token_level}...")
        results = collect_data_for_token_level(model, shard_data, token_level, batch_size)
        results_by_level[token_level] = results

        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / len(results) if results else 0
        logger.info(f"[Worker {rank}] tokens={token_level}: {accuracy:.4f} ({correct}/{len(results)})")

    result_queue.put((rank, results_by_level))
    logger.info(f"[Worker {rank}] Done")


def collect_parallel(
    model_name: str,
    max_model_len: int,
    batch_size: int,
    data: List[Dict[str, Any]],
    token_levels: List[int],
    gpus: List[int],
) -> Dict[int, List[Dict[str, Any]]]:
    """Collect data in parallel across multiple GPUs.

    Args:
        model_name: HuggingFace model name
        max_model_len: Maximum model context length
        batch_size: Batch size for inference
        data: List of problems to process
        token_levels: List of token levels to collect
        gpus: List of GPU IDs to use

    Returns:
        Dictionary mapping token_level -> list of results
    """
    world_size = len(gpus)
    logger.info(f"Starting parallel collection with {world_size} workers on GPUs {gpus}")

    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    # Start workers
    processes = []
    for rank, gpu_id in enumerate(gpus):
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, gpu_id, model_name, max_model_len, batch_size, data, token_levels, result_queue)
        )
        p.start()
        processes.append(p)

    # Collect results
    all_results = {}
    for _ in range(world_size):
        rank, results_by_level = result_queue.get()
        for token_level, results in results_by_level.items():
            if token_level not in all_results:
                all_results[token_level] = []
            all_results[token_level].extend(results)

    # Wait for all workers
    for p in processes:
        p.join()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Collect data with vLLM and thinking prompt")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Model name")
    parser.add_argument("--dataset", type=str, default="hendrycks_math",
                        choices=["hendrycks_math", "math500", "hendrycks_math_all"],
                        help="Dataset: hendrycks_math (by subset), math500 (MATH-500), hendrycks_math_all (all subsets merged)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset for hendrycks_math (e.g., algebra). If None, process all subsets")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Split for hendrycks_math/hendrycks_math_all (ignored for math500)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (single GPU mode)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model context length")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--token-levels", type=str, default="0,100,500,1000",
                        help="Comma-separated token levels to collect")
    # Parallel mode arguments
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel data collection with multiple GPUs")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated list of GPUs for parallel mode")

    args = parser.parse_args()

    # Parse token levels
    token_levels = [int(x) for x in args.token_levels.split(",")]

    # Parse GPUs for parallel mode
    gpus = [int(g.strip()) for g in args.gpus.split(",")]

    # Set output directory (default: server path)
    model_name = args.model.split('/')[-1]
    if args.output_dir is None:
        if args.dataset == "math500":
            args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/math500_think_{model_name}"
        elif args.dataset == "hendrycks_math_all":
            args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_all_think_{model_name}"
        else:
            args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_{model_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

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

    def collect_and_save(data: List[Dict[str, Any]], output_subdir: str):
        """Helper to collect data and save results."""
        os.makedirs(output_subdir, exist_ok=True)

        if args.parallel:
            all_results = collect_parallel(
                model_name=args.model,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                data=data,
                token_levels=token_levels,
                gpus=gpus,
            )
        else:
            model = VLLMInference(
                model_name=args.model,
                gpu_id=args.gpu,
                max_model_len=args.max_model_len,
            )
            all_results = {}
            for token_level in token_levels:
                logger.info(f"\nCollecting data for tokens={token_level}...")
                results = collect_data_for_token_level(model, data, token_level, args.batch_size)
                all_results[token_level] = results

        # Save results
        for token_level in token_levels:
            results = all_results.get(token_level, [])
            correct = sum(1 for r in results if r['is_correct'])
            accuracy = correct / len(results) if results else 0
            logger.info(f"  tokens={token_level}: {accuracy:.4f} ({correct}/{len(results)})")

            output_file = os.path.join(output_subdir, f"tokens{token_level}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved to {output_file}")

    if args.parallel:
        logger.info(f"Parallel mode with {len(gpus)} GPUs: {gpus}")
    else:
        logger.info(f"Single GPU mode on GPU {args.gpu}")

    if args.dataset == "math500":
        # MATH-500 dataset
        logger.info(f"\n{'='*60}")
        logger.info("Processing MATH-500")
        logger.info(f"{'='*60}")

        data = load_math500()
        output_subdir = os.path.join(args.output_dir, "math500", "test")
        collect_and_save(data, output_subdir)

    elif args.dataset == "hendrycks_math_all":
        # All hendrycks_math subsets merged
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing hendrycks_math_all ({args.split})")
        logger.info(f"{'='*60}")

        data = load_hendrycks_math_all(args.split)
        output_subdir = os.path.join(args.output_dir, "all", args.split)
        collect_and_save(data, output_subdir)

    else:
        # hendrycks_math by subset
        subsets = [args.subset] if args.subset else MATH_SUBSETS

        for subset in subsets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing subset: {subset}")
            logger.info(f"{'='*60}")

            data = load_hendrycks_math_subset(subset, args.split)
            output_subdir = os.path.join(args.output_dir, subset, args.split)
            collect_and_save(data, output_subdir)

    logger.info("\nData collection complete!")


if __name__ == "__main__":
    main()
