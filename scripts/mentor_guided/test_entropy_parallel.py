#!/usr/bin/env python3
"""
Multi-GPU parallel version: each GPU loads both models, processes different samples.

每个GPU加载两个模型，并行处理不同的问题。
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import torch.multiprocessing as mp

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_boxed_content(text: str) -> str:
    """Extract content inside \\boxed{} command."""
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


@dataclass
class SampleTask:
    """A sample to process."""
    idx: int
    problem: str
    ground_truth: str
    level: Any


@dataclass
class SampleResult:
    """Result for one sample."""
    idx: int
    ground_truth: str
    level: Any
    length_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    mentor_model_name: str,
    student_model_name: str,
    mentor_lengths: List[int],
    student_max_tokens: int,
):
    """Worker: load both models on one GPU, process samples from queue."""

    # Set GPU
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    logger.info(f"[GPU {gpu_id}] Loading mentor model...")
    mentor_tokenizer = AutoTokenizer.from_pretrained(mentor_model_name, trust_remote_code=True)
    mentor_model = AutoModelForCausalLM.from_pretrained(
        mentor_model_name,
        torch_dtype=torch.float16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    mentor_model.eval()

    logger.info(f"[GPU {gpu_id}] Loading student model...")
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    student_model.eval()

    for tok in [mentor_tokenizer, student_tokenizer]:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    logger.info(f"[GPU {gpu_id}] Models loaded. Starting processing...")

    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            logger.info(f"[GPU {gpu_id}] Received stop signal, exiting.")
            break

        try:
            # Build prompt
            prompt = f"""Solve the following math problem. Put your final answer in \\boxed{{}}.

Problem: {task.problem}

Solution:"""

            result = SampleResult(
                idx=task.idx,
                ground_truth=task.ground_truth,
                level=task.level,
            )

            for length in mentor_lengths:
                # Generate mentor tokens
                if length == 0:
                    mentor_text = ""
                else:
                    inputs = mentor_tokenizer(prompt, return_tensors="pt", truncation=True)
                    input_ids = inputs["input_ids"].to(device)
                    with torch.no_grad():
                        outputs = mentor_model.generate(
                            input_ids,
                            max_new_tokens=length,
                            do_sample=False,
                            pad_token_id=mentor_tokenizer.pad_token_id,
                        )
                    new_ids = outputs[0, input_ids.shape[1]:].tolist()
                    mentor_text = mentor_tokenizer.decode(new_ids, skip_special_tokens=True)

                # Generate student answer
                full_prompt = prompt + mentor_text
                inputs = student_tokenizer(full_prompt, return_tensors="pt", truncation=True)
                input_ids = inputs["input_ids"].to(device)
                with torch.no_grad():
                    outputs = student_model.generate(
                        input_ids,
                        max_new_tokens=student_max_tokens,
                        do_sample=False,
                        pad_token_id=student_tokenizer.pad_token_id,
                    )
                student_ids = outputs[0, input_ids.shape[1]:]
                student_answer = student_tokenizer.decode(student_ids, skip_special_tokens=True)

                full_answer = mentor_text + student_answer
                predicted_boxed = extract_boxed_content(full_answer)
                is_correct = grade_answer(predicted_boxed, task.ground_truth)

                result.length_results[length] = {
                    "mentor_text": mentor_text[:200],
                    "student_answer": student_answer[:200],
                    "predicted_boxed": predicted_boxed,
                    "is_correct": is_correct,
                }

            result_queue.put(result)
            logger.info(f"[GPU {gpu_id}] Done sample {task.idx}: {[r['is_correct'] for r in result.length_results.values()]}")

        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Error on sample {task.idx}: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put(None)


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Parallel Test (each GPU loads both models)')

    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-32B-Instruct')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', default='HuggingFaceH4/MATH-500')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--difficulty', type=str, default='4',
                       help='Comma-separated difficulty levels (1-5)')
    parser.add_argument('--lengths', type=str, default='0,50,100,200')
    parser.add_argument('--output-file', default='parallel_results.json')
    parser.add_argument('--student-max-tokens', type=int, default=2048)
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs (default: all available)')

    args = parser.parse_args()

    mentor_lengths = [int(x) for x in args.lengths.split(',')]
    difficulty_levels = [int(d.strip()) for d in args.difficulty.split(',')]

    num_gpus = args.num_gpus or torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPUs (each loads both models)")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset, split="test")
    except:
        dataset = load_dataset(args.dataset, split="train")

    if 'level' in dataset.column_names:
        dataset = dataset.filter(lambda x: x['level'] in difficulty_levels)
        logger.info(f"Filtered to {len(dataset)} samples with difficulty {difficulty_levels}")

    num_samples = min(args.num_samples, len(dataset))

    # Create tasks
    tasks = []
    for i in range(num_samples):
        sample = dataset[i]

        if "problem" in sample:
            problem = sample["problem"]
        elif "question" in sample:
            problem = sample["question"]
        else:
            problem = str(sample)

        if "solution" in sample:
            ground_truth = extract_boxed_content(sample["solution"])
        elif "answer" in sample:
            ground_truth = str(sample["answer"])
        else:
            ground_truth = ""

        tasks.append(SampleTask(
            idx=i,
            problem=problem,
            ground_truth=ground_truth,
            level=sample.get("level", "unknown"),
        ))

    logger.info(f"Created {len(tasks)} tasks for {num_gpus} GPUs")

    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start workers
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                args.mentor_model,
                args.student_model,
                mentor_lengths,
                args.student_max_tokens,
            )
        )
        p.start()
        workers.append(p)

    # Submit tasks
    for task in tasks:
        task_queue.put(task)

    # Send poison pills
    for _ in range(num_gpus):
        task_queue.put(None)

    # Collect results
    results = []
    pbar = tqdm(total=len(tasks), desc="Processing samples")
    while len(results) < len(tasks):
        result = result_queue.get()
        if result is not None:
            results.append(result)
        pbar.update(1)
    pbar.close()

    # Wait for workers
    for p in workers:
        p.join()

    # Sort by index
    results.sort(key=lambda x: x.idx)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ACCURACY BY MENTOR LENGTH")
    logger.info("="*60)
    for length in mentor_lengths:
        correct = sum(1 for r in results if r.length_results[length]["is_correct"])
        acc = correct / len(results) * 100
        logger.info(f"  Length {length}: {acc:.1f}% ({correct}/{len(results)})")

    # Save results
    output_path = os.path.join(script_dir, args.output_file)
    save_data = {
        "config": {
            "mentor_model": args.mentor_model,
            "student_model": args.student_model,
            "dataset": args.dataset,
            "difficulty": difficulty_levels,
            "num_samples": len(results),
            "mentor_lengths": mentor_lengths,
            "num_gpus": num_gpus,
        },
        "results": [
            {
                "idx": r.idx,
                "ground_truth": r.ground_truth,
                "level": r.level,
                "length_results": {
                    str(k): v for k, v in r.length_results.items()
                }
            }
            for r in results
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
