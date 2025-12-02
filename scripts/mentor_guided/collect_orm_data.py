#!/usr/bin/env python3
"""
Collect training data for ORM (Outcome Reward Model).

For each problem:
1. Get student-alone answer (baseline)
2. Get mentor outputs at various lengths
3. Get student answer with each mentor output
4. Label: helpful (student wrong → mentored right) or harmful (student right → mentored wrong)

收集ORM训练数据：判断mentor输出是否对student有帮助
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import torch.multiprocessing as mp

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_boxed_content(text: str) -> str:
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
class ORMSample:
    """One training sample for ORM."""
    idx: int
    problem: str
    ground_truth: str
    mentor_tokens: int
    mentor_text: str
    student_alone_correct: bool
    mentored_correct: bool
    label: int  # 1=helpful, 0=neutral, -1=harmful


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    mentor_model_name: str,
    student_model_name: str,
    mentor_lengths: List[int],
    student_max_tokens: int,
):
    """Worker process for one GPU."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    logger.info(f"[GPU {gpu_id}] Loading models...")

    mentor_tokenizer = AutoTokenizer.from_pretrained(mentor_model_name, trust_remote_code=True)
    mentor_model = AutoModelForCausalLM.from_pretrained(
        mentor_model_name,
        torch_dtype=torch.float16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    mentor_model.eval()

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

    logger.info(f"[GPU {gpu_id}] Ready.")

    while True:
        task = task_queue.get()
        if task is None:
            break

        idx, problem, ground_truth = task
        samples = []

        try:
            prompt = f"""Solve the following math problem. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

            # Student alone (baseline)
            inputs = student_tokenizer(prompt, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                outputs = student_model.generate(
                    input_ids,
                    max_new_tokens=student_max_tokens,
                    do_sample=False,
                    pad_token_id=student_tokenizer.pad_token_id,
                )
            student_alone = student_tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
            student_alone_boxed = extract_boxed_content(student_alone)
            student_alone_correct = grade_answer(student_alone_boxed, ground_truth)

            # Test each mentor length
            for length in mentor_lengths:
                if length == 0:
                    mentor_text = ""
                    mentored_correct = student_alone_correct
                else:
                    # Generate mentor tokens
                    inputs = mentor_tokenizer(prompt, return_tensors="pt", truncation=True)
                    input_ids = inputs["input_ids"].to(device)
                    with torch.no_grad():
                        outputs = mentor_model.generate(
                            input_ids,
                            max_new_tokens=length,
                            do_sample=False,
                            pad_token_id=mentor_tokenizer.pad_token_id,
                        )
                    mentor_text = mentor_tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

                    # Student continues from mentor
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
                    student_answer = student_tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
                    full_answer = mentor_text + student_answer
                    mentored_boxed = extract_boxed_content(full_answer)
                    mentored_correct = grade_answer(mentored_boxed, ground_truth)

                # Determine label
                if not student_alone_correct and mentored_correct:
                    label = 1  # Helpful
                elif student_alone_correct and not mentored_correct:
                    label = -1  # Harmful
                else:
                    label = 0  # Neutral

                samples.append(ORMSample(
                    idx=idx,
                    problem=problem,
                    ground_truth=ground_truth,
                    mentor_tokens=length,
                    mentor_text=mentor_text,
                    student_alone_correct=student_alone_correct,
                    mentored_correct=mentored_correct,
                    label=label,
                ))

            result_queue.put(samples)
            logger.info(f"[GPU {gpu_id}] Done {idx}: baseline={'✓' if student_alone_correct else '✗'}, labels={[s.label for s in samples]}")

        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Error on {idx}: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put([])


def main():
    parser = argparse.ArgumentParser(description='Collect ORM Training Data')
    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-32B-Instruct')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', default='HuggingFaceH4/MATH-500')
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--difficulty', type=str, default='5')
    parser.add_argument('--lengths', type=str, default='0,20,50,100,150,200')
    parser.add_argument('--output-file', default='orm_training_data.json')
    parser.add_argument('--student-max-tokens', type=int, default=2048)
    parser.add_argument('--num-gpus', type=int, default=None)

    args = parser.parse_args()

    mentor_lengths = [int(x) for x in args.lengths.split(',')]
    difficulty_levels = [int(d.strip()) for d in args.difficulty.split(',')]
    num_gpus = args.num_gpus or torch.cuda.device_count()

    logger.info(f"Collecting ORM data with {num_gpus} GPUs")
    logger.info(f"Mentor lengths: {mentor_lengths}")

    # Load dataset
    try:
        dataset = load_dataset(args.dataset, split="test")
    except:
        dataset = load_dataset(args.dataset, split="train")

    if 'level' in dataset.column_names:
        dataset = dataset.filter(lambda x: x['level'] in difficulty_levels)
        logger.info(f"Filtered to {len(dataset)} samples")

    num_samples = min(args.num_samples, len(dataset))

    # Create tasks
    tasks = []
    for i in range(num_samples):
        sample = dataset[i]
        problem = sample.get("problem", sample.get("question", ""))
        if "solution" in sample:
            ground_truth = extract_boxed_content(sample["solution"])
        else:
            ground_truth = str(sample.get("answer", ""))
        tasks.append((i, problem, ground_truth))

    # Start workers
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, args.mentor_model, args.student_model,
                  mentor_lengths, args.student_max_tokens)
        )
        p.start()
        workers.append(p)

    for task in tasks:
        task_queue.put(task)
    for _ in range(num_gpus):
        task_queue.put(None)

    # Collect results
    all_samples = []
    pbar = tqdm(total=len(tasks), desc="Collecting data")
    collected = 0
    while collected < len(tasks):
        samples = result_queue.get()
        all_samples.extend(samples)
        collected += 1
        pbar.update(1)
    pbar.close()

    for p in workers:
        p.join()

    # Statistics
    helpful = sum(1 for s in all_samples if s.label == 1)
    harmful = sum(1 for s in all_samples if s.label == -1)
    neutral = sum(1 for s in all_samples if s.label == 0)

    logger.info(f"\n{'='*60}")
    logger.info("DATA STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Helpful (label=1): {helpful} ({100*helpful/len(all_samples):.1f}%)")
    logger.info(f"Harmful (label=-1): {harmful} ({100*harmful/len(all_samples):.1f}%)")
    logger.info(f"Neutral (label=0): {neutral} ({100*neutral/len(all_samples):.1f}%)")

    # Save
    output_path = os.path.join(script_dir, args.output_file)
    save_data = {
        "config": vars(args),
        "statistics": {"helpful": helpful, "harmful": harmful, "neutral": neutral},
        "samples": [asdict(s) for s in all_samples]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
