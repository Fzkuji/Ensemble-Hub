#!/usr/bin/env python3
"""
Batch-parallel version of entropy indicator testing.

使用batch推理加速，模型自动分布到多卡。
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

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
class SampleResult:
    """Result for one sample."""
    idx: int
    ground_truth: str
    level: Any
    length_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class BatchTester:
    """Batch-parallel tester using device_map=auto for multi-GPU."""

    def __init__(self, mentor_model_name: str, student_model_name: str):
        logger.info(f"Loading mentor: {mentor_model_name}")
        self.mentor_tokenizer = AutoTokenizer.from_pretrained(mentor_model_name, trust_remote_code=True)
        self.mentor_model = AutoModelForCausalLM.from_pretrained(
            mentor_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.mentor_model.eval()

        logger.info(f"Loading student: {student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.student_model.eval()

        for tok in [self.mentor_tokenizer, self.student_tokenizer]:
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"

    def generate_mentor_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """Batch generate mentor tokens."""
        if max_tokens == 0:
            return [""] * len(prompts)

        inputs = self.mentor_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.mentor_model.device)

        with torch.no_grad():
            outputs = self.mentor_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.mentor_tokenizer.pad_token_id,
            )

        results = []
        for i, output in enumerate(outputs):
            input_len = inputs["attention_mask"][i].sum().item()
            new_ids = output[input_len:].tolist()
            text = self.mentor_tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text)

        return results

    def generate_student_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """Batch generate student answers."""
        inputs = self.student_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.student_model.device)

        with torch.no_grad():
            outputs = self.student_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.student_tokenizer.pad_token_id,
            )

        results = []
        for i, output in enumerate(outputs):
            input_len = inputs["attention_mask"][i].sum().item()
            new_ids = output[input_len:].tolist()
            text = self.student_tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text)

        return results

    def process_samples_batch(
        self,
        problems: List[str],
        ground_truths: List[str],
        levels: List[Any],
        mentor_lengths: List[int],
        student_max_tokens: int,
        batch_size: int = 4,
    ) -> List[SampleResult]:
        """Process multiple samples with batching."""

        results = [
            SampleResult(idx=i, ground_truth=gt, level=lvl)
            for i, (gt, lvl) in enumerate(zip(ground_truths, levels))
        ]

        # Build base prompts
        base_prompts = [
            f"""Solve the following math problem. Put your final answer in \\boxed{{}}.

Problem: {prob}

Solution:"""
            for prob in problems
        ]

        for length in tqdm(mentor_lengths, desc="Mentor lengths"):
            # Generate mentor outputs for all samples
            logger.info(f"Generating mentor tokens (length={length})...")

            mentor_texts = []
            for i in range(0, len(base_prompts), batch_size):
                batch = base_prompts[i:i+batch_size]
                batch_results = self.generate_mentor_batch(batch, length)
                mentor_texts.extend(batch_results)

            # Generate student outputs for all samples
            logger.info(f"Generating student answers...")
            full_prompts = [p + m for p, m in zip(base_prompts, mentor_texts)]

            student_texts = []
            for i in range(0, len(full_prompts), batch_size):
                batch = full_prompts[i:i+batch_size]
                batch_results = self.generate_student_batch(batch, student_max_tokens)
                student_texts.extend(batch_results)

            # Evaluate all
            for i, (mentor_text, student_text) in enumerate(zip(mentor_texts, student_texts)):
                full_answer = mentor_text + student_text
                predicted_boxed = extract_boxed_content(full_answer)
                is_correct = grade_answer(predicted_boxed, ground_truths[i])

                results[i].length_results[length] = {
                    "mentor_text": mentor_text[:200],
                    "student_answer": student_text[:200],
                    "predicted_boxed": predicted_boxed,
                    "is_correct": is_correct,
                }

                status = "✓" if is_correct else "✗"
                logger.info(f"  Sample {i} [{length}]: {status} pred={predicted_boxed}, gt={ground_truths[i]}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Batch-Parallel Entropy Indicator Test')

    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-32B-Instruct')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', default='HuggingFaceH4/MATH-500')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--difficulty', type=str, default='4',
                       help='Comma-separated difficulty levels (1-5)')
    parser.add_argument('--lengths', type=str, default='0,50,100,200')
    parser.add_argument('--output-file', default='parallel_results.json')
    parser.add_argument('--student-max-tokens', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for generation')

    args = parser.parse_args()

    mentor_lengths = [int(x) for x in args.lengths.split(',')]
    difficulty_levels = [int(d.strip()) for d in args.difficulty.split(',')]

    logger.info(f"GPUs available: {torch.cuda.device_count()}")

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

    # Extract data
    problems = []
    ground_truths = []
    levels = []

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

        problems.append(problem)
        ground_truths.append(ground_truth)
        levels.append(sample.get("level", "unknown"))

    logger.info(f"Processing {len(problems)} samples with batch_size={args.batch_size}")

    # Initialize tester
    tester = BatchTester(args.mentor_model, args.student_model)

    # Run batch processing
    results = tester.process_samples_batch(
        problems=problems,
        ground_truths=ground_truths,
        levels=levels,
        mentor_lengths=mentor_lengths,
        student_max_tokens=args.student_max_tokens,
        batch_size=args.batch_size,
    )

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
            "batch_size": args.batch_size,
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
    main()
