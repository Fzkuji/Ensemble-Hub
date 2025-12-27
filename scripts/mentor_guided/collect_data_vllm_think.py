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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Token levels to collect (0 = no mentor, just intern; -1 = mentor only, no intern)
TOKEN_LEVELS = [0, 100, 500, 1000]
MENTOR_ONLY_LEVEL = -1  # Special level for mentor-only baseline

# Simple system prompt (ACT-E uses simple prompts)
SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""


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
        gpu_memory_utilization: float = 0.9,
    ):
        """Initialize vLLM model.

        Args:
            model_name: HuggingFace model name
            gpu_id: GPU ID to use
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum model context length
            gpu_memory_utilization: Fraction of GPU memory to use (default: 0.9)
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is required. Install with: pip install vllm")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logger.info(f"Loading model {model_name} with vLLM on GPU {gpu_id} (memory_util={gpu_memory_utilization})...")

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

    def build_chat_prompt(
        self,
        question: str,
        use_think: bool = True,
    ) -> str:
        """Build simple chat prompt.

        ACT-E uses simple prompts without complex frameworks.
        For DeepSeek R1 no-think mode, we pre-fill empty think block.

        Args:
            question: The math problem
            use_think: Whether to allow thinking (True) or skip it (False)

        Returns:
            Formatted prompt string
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # For DeepSeek R1 no-think mode: pre-fill empty think block
        if not use_think:
            prompt = prompt + "<think>\n</think>\n\n"

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
            prompts = [mentor_model.build_chat_prompt(item['question'], use_think=use_think) for item in batch]
            mentor_outputs = mentor_model.generate_mentor_tokens(prompts, max_tokens=token_level)

            # Intern CONTINUES from mentor's output (not starting over)
            # Concatenate prompt + mentor_output, then continue generating
            continued_prompts = [
                prompt + mentor_output
                for prompt, mentor_output in zip(prompts, mentor_outputs)
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
    mentor_gpu_id: int = None,
    intern_gpu_id: int = None,
    mentor_memory_util: float = 0.5,
    intern_memory_util: float = 0.3,
    mentor_max_model_len: int = None,
    intern_max_model_len: int = None,
    force: bool = False,
):
    """Worker process that processes ALL subsets and token levels with TWO model inits.

    Args:
        rank: Worker rank
        world_size: Total number of workers
        gpu_id: Default GPU ID to use
        mentor_model_name: Mentor model name (large model)
        intern_model_name: Intern model name (small model)
        max_model_len: Max model context length
        batch_size: Batch size
        all_tasks: List of (subset_name, output_dir, data) tuples
        token_levels: List of token levels to collect
        use_think: Whether to use think prompt
        mentor_gpu_id: GPU ID for mentor model (if None, uses gpu_id)
        intern_gpu_id: GPU ID for intern model (if None, uses gpu_id)
        mentor_memory_util: GPU memory utilization for mentor model (default: 0.6)
        intern_memory_util: GPU memory utilization for intern model (default: 0.3)
    """
    # Determine GPU IDs for each model
    mentor_gpu = mentor_gpu_id if mentor_gpu_id is not None else gpu_id
    intern_gpu = intern_gpu_id if intern_gpu_id is not None else gpu_id
    
    # Determine max_model_len for each model
    mentor_max_len = mentor_max_model_len if mentor_max_model_len is not None else max_model_len
    intern_max_len = intern_max_model_len if intern_max_model_len is not None else max_model_len
    
    # If using different GPUs, use default memory utilization (0.9) since they don't share memory
    # If using same GPU, use the specified memory utilization values
    using_different_gpus = (mentor_gpu != intern_gpu)
    if using_different_gpus:
        mentor_mem_util = 0.9  # Default when using separate GPU
        intern_mem_util = 0.9  # Default when using separate GPU
        logger.info(f"[Worker {rank}] Using different GPUs - memory utilization set to default (0.9) for both models")
    else:
        mentor_mem_util = mentor_memory_util
        intern_mem_util = intern_memory_util
        logger.info(f"[Worker {rank}] Using same GPU - memory utilization: mentor={mentor_mem_util}, intern={intern_mem_util}")
    
    logger.info(f"[Worker {rank}] Initializing models (mentor on GPU {mentor_gpu}, intern on GPU {intern_gpu})...")

    # Initialize mentor model (large model)
    logger.info(f"[Worker {rank}] Loading mentor model: {mentor_model_name} on GPU {mentor_gpu} (memory_util={mentor_mem_util}, max_len={mentor_max_len})...")
    try:
        mentor_model = VLLMInference(
            model_name=mentor_model_name,
            gpu_id=mentor_gpu,
            max_model_len=mentor_max_len,
            gpu_memory_utilization=mentor_mem_util,
        )
    except Exception as e:
        logger.error(f"[Worker {rank}] Failed to load mentor model: {e}")
        if using_different_gpus:
            logger.error(f"[Worker {rank}] Try: 1) Reduce --mentor-max-model-len (current: {mentor_max_len})")
            logger.error(f"[Worker {rank}]     2) Check if GPU {mentor_gpu} has enough free memory")
        else:
            logger.error(f"[Worker {rank}] Try: 1) Lower --mentor-memory-util (current: {mentor_mem_util})")
            logger.error(f"[Worker {rank}]     2) Use separate GPU with --mentor-gpus")
            logger.error(f"[Worker {rank}]     3) Reduce --mentor-max-model-len (current: {mentor_max_len})")
        raise

    # Initialize intern model (small model)
    logger.info(f"[Worker {rank}] Loading intern model: {intern_model_name} on GPU {intern_gpu} (memory_util={intern_mem_util}, max_len={intern_max_len})...")
    try:
        intern_model = VLLMInference(
            model_name=intern_model_name,
            gpu_id=intern_gpu,
            max_model_len=intern_max_len,
            gpu_memory_utilization=intern_mem_util,
        )
    except Exception as e:
        logger.error(f"[Worker {rank}] Failed to load intern model: {e}")
        if using_different_gpus:
            logger.error(f"[Worker {rank}] Try: 1) Reduce --intern-max-model-len (current: {intern_max_len})")
            logger.error(f"[Worker {rank}]     2) Check if GPU {intern_gpu} has enough free memory")
        else:
            logger.error(f"[Worker {rank}] Try: 1) Lower --intern-memory-util (current: {intern_mem_util})")
            logger.error(f"[Worker {rank}]     2) Use separate GPU with --intern-gpus")
            logger.error(f"[Worker {rank}]     3) Reduce --intern-max-model-len (current: {intern_max_len})")
        raise

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
                logger.info(f"[Worker {rank}] {subset_name} tokens={token_level} already exists, but --force is set, re-collecting...")
                # Remove existing file to force re-collection
                os.remove(merged_file)
            
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
                    # We got the lock - do the merge
                    if not os.path.exists(merged_file):
                        total, correct_cnt, acc = merge_rank_files(output_dir, token_level, world_size)
                        print(f"[MERGED] {subset_name} tokens={token_level}: {total} samples, acc={acc:.4f}", flush=True)
                    os.remove(lock_file)
                except FileExistsError:
                    # Another worker is merging, skip
                    pass

    logger.info(f"[Worker {rank}] All tasks completed")
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
    )
    return results.get("single", {})


def merge_rank_files(output_dir: str, token_level: int, world_size: int) -> Tuple[int, int, float]:
    """Merge all rank files for a single token level.

    Returns: (total_samples, correct_samples, accuracy)
    """
    merged = []
    for rank in range(world_size):
        temp_file = os.path.join(output_dir, f"tokens{token_level}_rank{rank}.json")
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            merged.extend(results)
            os.remove(temp_file)
            print(f"  [MERGE] Loaded {len(results)} samples from rank {rank}", flush=True)

    if merged:
        correct = sum(1 for r in merged if r['is_correct'])
        accuracy = correct / len(merged)

        output_file = os.path.join(output_dir, f"tokens{token_level}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

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
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Collect data for ALL subsets in parallel with TWO model inits per GPU.

    Workers merge results immediately after each (subset, token_level) completes.
    
    Args:
        mentor_gpu_ids: List of GPU IDs for mentor models (if None, uses gpus)
        intern_gpu_ids: List of GPU IDs for intern models (if None, uses gpus)
        mentor_memory_util: GPU memory utilization for mentor model (default: 0.6)
        intern_memory_util: GPU memory utilization for intern model (default: 0.3)
    """
    world_size = len(gpus)
    
    # Validate GPU list lengths (should already be validated in main(), but double-check)
    if len(mentor_gpu_ids) != world_size:
        raise ValueError(f"mentor_gpu_ids length ({len(mentor_gpu_ids)}) must match gpus length ({world_size})")
    if len(intern_gpu_ids) != world_size:
        raise ValueError(f"intern_gpu_ids length ({len(intern_gpu_ids)}) must match gpus length ({world_size})")

    print(f"\n{'='*60}", flush=True)
    print(f"[MAIN] Starting parallel collection", flush=True)
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

    # Clean up old files
    for subset_name, output_dir, _ in all_tasks:
        os.makedirs(output_dir, exist_ok=True)
        for token_level in token_levels:
            for rank in range(world_size):
                temp_file = os.path.join(output_dir, f"tokens{token_level}_rank{rank}.json")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            # Also clean lock files and merged files
            lock_file = os.path.join(output_dir, f".lock_tokens{token_level}")
            merged_file = os.path.join(output_dir, f"tokens{token_level}.json")
            if os.path.exists(lock_file):
                os.remove(lock_file)
            if os.path.exists(merged_file):
                os.remove(merged_file)

    # Start all workers
    processes = []
    for rank, gpu_id in enumerate(gpus):
        mentor_gpu = mentor_gpu_ids[rank]
        intern_gpu = intern_gpu_ids[rank]
        p = mp.Process(
            target=worker_process_all_tasks,
            args=(rank, world_size, gpu_id, mentor_model_name, intern_model_name, max_model_len, batch_size, all_tasks, token_levels, use_think, mentor_gpu, intern_gpu, mentor_memory_util, intern_memory_util, mentor_max_model_len, intern_max_model_len, force)
        )
        p.start()
        processes.append(p)
        print(f"[MAIN] Started worker {rank} (mentor GPU {mentor_gpu}, intern GPU {intern_gpu}, PID: {p.pid})", flush=True)

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
                        choices=["hendrycks_math", "math500", "hendrycks_math_all"],
                        help="Dataset: hendrycks_math (by subset), math500 (MATH-500), hendrycks_math_all (all subsets merged)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset for hendrycks_math (e.g., algebra). If None, process all subsets")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Split for hendrycks_math/hendrycks_math_all (ignored for math500)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model context length")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name for output directory (e.g., R1_m32B_i7B). If not set, uses model name.")
    parser.add_argument("--token-levels", type=str, default="0,100,500,1000",
                        help="Comma-separated token levels to collect")
    # Parallel mode arguments (always enabled)
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated list of worker GPUs (e.g., '0,1,2,3'). Each worker processes a shard of data.")
    parser.add_argument("--mentor-gpus", type=str, default=None,
                        help="Comma-separated list of GPUs for mentor models (e.g., '0,1,2,3'). If not specified, uses --gpus.")
    parser.add_argument("--intern-gpus", type=str, default=None,
                        help="Comma-separated list of GPUs for intern models (e.g., '4,5,6,7'). If not specified, uses --gpus.")
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

    # Parse GPUs for parallel mode
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    
    # Use --gpus as default if mentor/intern GPUs not specified
    if args.mentor_gpus is None:
        mentor_gpu_ids = gpus
    else:
        mentor_gpu_ids = [int(g.strip()) for g in args.mentor_gpus.split(",")]
    
    if args.intern_gpus is None:
        intern_gpu_ids = gpus
    else:
        intern_gpu_ids = [int(g.strip()) for g in args.intern_gpus.split(",")]
    
    # Validate GPU list lengths match
    if len(mentor_gpu_ids) != len(gpus):
        raise ValueError(f"--mentor-gpus length ({len(mentor_gpu_ids)}) must match --gpus length ({len(gpus)})")
    if len(intern_gpu_ids) != len(gpus):
        raise ValueError(f"--intern-gpus length ({len(intern_gpu_ids)}) must match --gpus length ({len(gpus)})")

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
        if args.dataset == "math500":
            args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/math500_{mode_suffix}_{exp_name}"
        elif args.dataset == "hendrycks_math_all":
            args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_all_{mode_suffix}_{exp_name}"
        else:
            args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_{mode_suffix}_{exp_name}"

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

        stats = collect_parallel(
            mentor_model_name=mentor_model,
            intern_model_name=intern_model,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            data=data,
            token_levels=token_levels,
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
        )
        for token_level in token_levels:
            token_stats = stats.get(token_level)
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

        # Check if all data files already exist (before loading data and initializing models)
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

        collect_all_parallel(
            mentor_model_name=mentor_model,
            intern_model_name=intern_model,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            all_tasks=all_tasks,
            token_levels=token_levels,
            gpus=gpus,
            mentor_gpu_ids=mentor_gpu_ids,
            intern_gpu_ids=intern_gpu_ids,
            use_think=use_think,
            mentor_memory_util=args.mentor_memory_util,
            intern_memory_util=args.intern_memory_util,
            mentor_max_model_len=args.mentor_max_model_len,
            intern_max_model_len=args.intern_max_model_len,
            force=args.force,
        )

    logger.info("\nData collection complete!")


if __name__ == "__main__":
    main()
