#!/usr/bin/env python3
"""
Collect Progressive Data with HuggingFace Transformers (No vLLM)

Supports multi-GPU parallel inference where each GPU loads a separate model
and processes a shard of the data.

Usage:
    # Single GPU
    python collect_data_hf.py --split test --gpu 0

    # Multi-GPU parallel
    python collect_data_hf.py --split test --parallel --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import logging
import os
import sys
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Token levels to collect
TOKEN_LEVELS = [0, 100, 500, 1000]

# System prompts
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

STANDARD_SYSTEM_PROMPT = """You are a mathematical reasoning expert. Solve the problem step by step, showing your work clearly.

After your reasoning, provide the final answer in \\boxed{}.
"""

STANDARD_SYSTEM_PROMPT_WITH_HINT = """You are a mathematical reasoning expert. Solve the problem step by step, showing your work clearly.

You are also provided with a hint from a mentor model. Use the hint to guide your reasoning.

After your reasoning, provide the final answer in \\boxed{}.
"""


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{}."""
    start = text.find(r'\boxed{')
    if start == -1:
        return ""

    depth = 0
    end = start + 7
    while end < len(text):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            if depth == 0:
                return text[start + 7:end]
            depth -= 1
        end += 1
    return ""


def check_math_correctness(response: str, ground_truth: str) -> bool:
    """Check if the response is mathematically correct."""
    predicted = extract_boxed_answer(response)
    return grade_answer(predicted, ground_truth)


def load_hendrycks_math_subset(subset: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load hendrycks/competition_math subset."""
    from datasets import load_dataset

    ds = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)
    data = []
    for item in ds:
        if item.get('type', '').lower().replace(' ', '_') == subset.lower().replace(' ', '_'):
            data.append({
                'question': item['problem'],
                'ground_truth': item['solution'],
                'subset': subset,
                'level': item.get('level', ''),
            })
    return data


class HFInference:
    """HuggingFace Transformers inference wrapper."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
    ):
        self.device = device
        self.model_name = model_name

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model: {model_name} on {device}")
        dtype = getattr(torch, torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        logger.info("Model loaded successfully")

    def build_chat_prompt(
        self,
        question: str,
        hint: Optional[str] = None,
        use_think: bool = True,
    ) -> str:
        """Build chat prompt with template."""
        if hint:
            if use_think:
                system_prompt = THINKING_SYSTEM_PROMPT_WITH_HINT
            else:
                system_prompt = STANDARD_SYSTEM_PROMPT_WITH_HINT
            user_content = f"Problem: {question}\n\nHint from mentor:\n{hint}"
        else:
            if use_think:
                system_prompt = THINKING_SYSTEM_PROMPT
            else:
                system_prompt = STANDARD_SYSTEM_PROMPT
            user_content = f"Problem: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # For DeepSeek R1: skip thinking by pre-filling empty think block
        if not use_think:
            prompt = prompt + "<think>\n</think>\n\n"

        return prompt

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """Generate response for a single prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(self.device)

        input_len = inputs['input_ids'].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        return response

    @torch.no_grad()
    def generate_mentor_tokens(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """Generate exactly max_tokens from mentor."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(self.device)

        input_len = inputs['input_ids'].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        hint = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        return hint


def collect_data_for_token_level(
    model: HFInference,
    data: List[Dict[str, Any]],
    token_level: int,
    use_think: bool = True,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Collect data for a specific token level."""
    results = []

    iterator = tqdm(data, desc=f"tokens={token_level}") if show_progress else data

    for item in iterator:
        if token_level == 0:
            # No mentor hint - direct generation
            prompt = model.build_chat_prompt(item['question'], use_think=use_think)
            response = model.generate(prompt)
            hint = ""
        else:
            # First generate mentor hint
            mentor_prompt = model.build_chat_prompt(item['question'], use_think=use_think)
            hint = model.generate_mentor_tokens(mentor_prompt, max_tokens=token_level)

            # Then generate with hint
            prompt_with_hint = model.build_chat_prompt(item['question'], hint=hint, use_think=use_think)
            response = model.generate(prompt_with_hint)

        full_response = hint + response if hint else response
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


def merge_rank_files(output_dir: str, token_level: int, world_size: int) -> Tuple[int, int, float]:
    """Merge all rank files for a single token level."""
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


def worker_process(
    rank: int,
    world_size: int,
    gpu_id: int,
    model_name: str,
    all_tasks: List[Tuple[str, str, List[Dict[str, Any]]]],
    token_levels: List[int],
    use_think: bool = True,
):
    """Worker process that processes data shard on a single GPU."""
    device = f"cuda:{gpu_id}"
    logger.info(f"[Worker {rank}] GPU {gpu_id}: Initializing model...")

    # Initialize model
    model = HFInference(
        model_name=model_name,
        device=device,
    )

    logger.info(f"[Worker {rank}] Model loaded, processing {len(all_tasks)} subsets × {len(token_levels)} token levels")

    # Process all tasks
    for subset_name, output_dir, data in all_tasks:
        # Shard data for this worker
        shard_data = [d for i, d in enumerate(data) if i % world_size == rank]

        if not shard_data:
            logger.info(f"[Worker {rank}] No data for subset {subset_name}, skipping")
            continue

        logger.info(f"[Worker {rank}] Processing subset {subset_name}: {len(shard_data)} samples")

        for token_level in token_levels:
            logger.info(f"[Worker {rank}] {subset_name} tokens={token_level}...")
            results = collect_data_for_token_level(
                model, shard_data, token_level, use_think=use_think, show_progress=False
            )

            correct = sum(1 for r in results if r['is_correct'])
            accuracy = correct / len(results) if results else 0
            logger.info(f"[Worker {rank}] {subset_name} tokens={token_level}: {accuracy:.4f} ({correct}/{len(results)})")

            # Save to temp file
            os.makedirs(output_dir, exist_ok=True)
            temp_file = os.path.join(output_dir, f"tokens{token_level}_rank{rank}.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[Worker {rank}] Saved: {temp_file}")

            # Check if all ranks finished - if so, merge
            all_exist = all(
                os.path.exists(os.path.join(output_dir, f"tokens{token_level}_rank{r}.json"))
                for r in range(world_size)
            )
            if all_exist:
                lock_file = os.path.join(output_dir, f".lock_tokens{token_level}")
                merged_file = os.path.join(output_dir, f"tokens{token_level}.json")
                try:
                    fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    if not os.path.exists(merged_file):
                        total, correct_cnt, acc = merge_rank_files(output_dir, token_level, world_size)
                        print(f"[MERGED] {subset_name} tokens={token_level}: {total} samples, acc={acc:.4f}", flush=True)
                    os.remove(lock_file)
                except FileExistsError:
                    pass

    logger.info(f"[Worker {rank}] All tasks completed")
    os._exit(0)


def collect_all_parallel(
    model_name: str,
    all_tasks: List[Tuple[str, str, List[Dict[str, Any]]]],
    token_levels: List[int],
    gpus: List[int],
    use_think: bool = True,
):
    """Collect data in parallel across multiple GPUs."""
    world_size = len(gpus)

    print(f"\n{'='*60}", flush=True)
    print(f"[MAIN] Starting parallel collection (HuggingFace)", flush=True)
    print(f"[MAIN] GPUs: {gpus} ({world_size} workers)", flush=True)
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
            lock_file = os.path.join(output_dir, f".lock_tokens{token_level}")
            if os.path.exists(lock_file):
                os.remove(lock_file)

    # Start all workers
    processes = []
    for rank, gpu_id in enumerate(gpus):
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, gpu_id, model_name, all_tasks, token_levels, use_think)
        )
        p.start()
        processes.append(p)
        print(f"[MAIN] Started worker {rank} on GPU {gpu_id} (PID: {p.pid})", flush=True)

    print(f"\n[MAIN] All {world_size} workers started. Waiting...\n", flush=True)

    # Wait for all workers
    for p in processes:
        p.join()

    print(f"\n{'='*60}", flush=True)
    print(f"[MAIN] All workers finished.", flush=True)
    print(f"{'='*60}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Collect data with HuggingFace Transformers")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Model name")
    parser.add_argument("--dataset", type=str, default="hendrycks_math",
                        choices=["hendrycks_math"],
                        help="Dataset")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset (e.g., algebra). If None, process all subsets")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Split")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID (single GPU mode)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--token-levels", type=str, default="0,100,500,1000",
                        help="Comma-separated token levels")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per subset (for testing)")
    # Parallel mode
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel data collection with multiple GPUs")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated list of GPUs for parallel mode")

    args = parser.parse_args()

    use_think = not args.no_think
    token_levels = [int(x) for x in args.token_levels.split(",")]
    gpus = [int(g.strip()) for g in args.gpus.split(",")]

    # Set output directory
    model_name = args.model.split('/')[-1]
    mode_suffix = "think" if use_think else "standard"
    if args.output_dir is None:
        args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_{mode_suffix}_hf_{model_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Mode: {'THINK' if use_think else 'STANDARD (no-think)'}")

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

    if args.parallel:
        # Parallel mode: load all data and process together
        logger.info(f"Parallel mode with {len(gpus)} GPUs: {gpus}")

        all_tasks = []
        for subset in subsets:
            data = load_hendrycks_math_subset(subset, args.split)
            if args.max_samples:
                data = data[:args.max_samples]
            output_subdir = os.path.join(args.output_dir, subset, args.split)
            all_tasks.append((subset, output_subdir, data))
            logger.info(f"Loaded {subset}: {len(data)} samples")

        collect_all_parallel(
            model_name=args.model,
            all_tasks=all_tasks,
            token_levels=token_levels,
            gpus=gpus,
            use_think=use_think,
        )

    else:
        # Single GPU mode
        device = f"cuda:{args.gpu}"
        logger.info(f"Single GPU mode on {device}")

        model = HFInference(
            model_name=args.model,
            device=device,
        )

        for subset in subsets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing subset: {subset}")
            logger.info(f"{'='*60}")

            data = load_hendrycks_math_subset(subset, args.split)
            if args.max_samples:
                data = data[:args.max_samples]
            logger.info(f"Loaded {len(data)} samples")

            output_subdir = os.path.join(args.output_dir, subset, args.split)
            os.makedirs(output_subdir, exist_ok=True)

            for token_level in token_levels:
                output_file = os.path.join(output_subdir, f"tokens{token_level}.json")

                if os.path.exists(output_file):
                    logger.info(f"Skipping tokens={token_level} (already exists)")
                    continue

                logger.info(f"\nCollecting tokens={token_level}...")
                results = collect_data_for_token_level(model, data, token_level, use_think=use_think)

                correct = sum(1 for r in results if r['is_correct'])
                accuracy = correct / len(results) if results else 0
                logger.info(f"tokens={token_level}: {accuracy:.4f} ({correct}/{len(results)})")

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved to {output_file}")

    logger.info("\nData collection complete!")


if __name__ == "__main__":
    main()
