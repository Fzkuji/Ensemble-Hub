#!/usr/bin/env python3
"""
Truncated CoT Baseline for Efficiency Comparison (AizX W3).

Compares two approaches at the same thinking token budget N:
- Truncated CoT: 32B thinks N tokens → 32B answers (all compute on 32B)
- Tandem:        32B thinks N tokens → 7B answers  (cheaper 7B for answer)

Also collects 32B full reasoning (no truncation) as the upper-bound baseline.

Usage:
    # Run on MATH test (all subsets)
    python collect_truncated_cot.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --gpus 0 --split test

    # Custom token levels
    python collect_truncated_cot.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --token-levels 100,500,1000 --gpus 0
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm

# Add scripts directory to path
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MATH_SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


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


def check_correctness(prediction: str, ground_truth: str) -> bool:
    """Check if math answer is correct."""
    pred_answer = extract_boxed_answer(prediction)
    true_answer = extract_boxed_answer(ground_truth)
    if not true_answer:
        true_answer = ground_truth.strip()
    if not pred_answer or not true_answer:
        return False
    return grade_answer(pred_answer, true_answer)


def main():
    parser = argparse.ArgumentParser(description="Truncated CoT Baseline")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--dataset", type=str, default="hendrycks_math",
                        choices=["hendrycks_math", "gsm8k"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--token-levels", type=str, default="100,500,1000",
                        help="Comma-separated thinking token budgets")
    parser.add_argument("--subset", type=str, default=None,
                        help="Run on a single MATH subset only (e.g. algebra). "
                             "Useful for quick sanity checks.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7, matches paper pipeline)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling (default: 0.95, matches paper pipeline)")
    parser.add_argument("--skip-full", action="store_true",
                        help="Skip full reasoning (only run truncated)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-generation even if output files exist")

    args = parser.parse_args()

    truncation_levels = [int(x) for x in args.token_levels.split(",")]
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    # Output directory
    if args.output_dir is None:
        model_short = args.model.split('/')[-1]
        base = "/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected"
        args.output_dir = os.path.join(base, f"truncated_cot_{args.dataset}_{model_short}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Temperature: {args.temperature}, Top-p: {args.top_p}")

    # Import from existing codebase
    from collect_data_vllm_think import (
        VLLMInference,
        load_hendrycks_math_all,
        load_gsm8k,
    )

    # Load model
    logger.info(f"Loading model {args.model}...")
    model = VLLMInference(
        args.model,
        gpu_ids=gpu_ids,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Load data
    if args.dataset == "hendrycks_math":
        data = load_hendrycks_math_all(args.split)
    elif args.dataset == "gsm8k":
        data = load_gsm8k(args.split)
    logger.info(f"Loaded {len(data)} problems")
    if args.subset:
        data = [d for d in data if d.get('subset', '') == args.subset]
        logger.info(f"Filtered to subset '{args.subset}': {len(data)} problems")

    # ========================================
    # Phase 1: Full Reasoning (no truncation)
    # ========================================
    full_path = os.path.join(args.output_dir, f"full_reasoning_{args.split}.json")
    full_cache = {}   # question -> full_response
    prompt_cache = {}  # question -> prompt

    if not args.skip_full and (args.force or not os.path.exists(full_path)):
        logger.info("Phase 1: Generating full reasoning...")
        full_results = []

        for batch_start in tqdm(range(0, len(data), args.batch_size),
                                desc="Full reasoning", unit="batch"):
            batch = data[batch_start:batch_start + args.batch_size]
            prompts = [model.build_chat_prompt(item['question'], use_think=True)
                       for item in batch]
            responses = model.generate(prompts, max_tokens=4096, temperature=args.temperature, top_p=args.top_p)

            for item, prompt, response in zip(batch, prompts, responses):
                full_cache[item['question']] = response
                prompt_cache[item['question']] = prompt

                is_correct = check_correctness(response, item['ground_truth'])
                total_tokens = len(model.tokenizer.encode(response)) if response else 0

                full_results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'response': response,
                    'is_correct': is_correct,
                    'total_tokens': total_tokens,
                    'subset': item.get('subset', ''),
                    'level': item.get('level', ''),
                })

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        full_correct = sum(1 for r in full_results if r['is_correct'])
        full_acc = full_correct / len(full_results)
        avg_tokens = sum(r['total_tokens'] for r in full_results) / len(full_results)
        logger.info(f"Full reasoning: {full_acc:.4f} ({full_correct}/{len(full_results)}), "
                     f"avg tokens: {avg_tokens:.0f}")
    elif os.path.exists(full_path):
        logger.info(f"Loading cached full reasoning from {full_path}")
        with open(full_path, 'r') as f:
            full_results = json.load(f)
        # Rebuild caches
        for r in full_results:
            full_cache[r['question']] = r['response']
        # Rebuild prompt cache
        for item in data:
            prompt_cache[item['question']] = model.build_chat_prompt(
                item['question'], use_think=True
            )
    else:
        logger.info("Skipping full reasoning (--skip-full)")
        full_results = None
        # Still need prompts for truncated CoT
        for item in data:
            prompt_cache[item['question']] = model.build_chat_prompt(
                item['question'], use_think=True
            )

    # If we need full_cache but don't have it, generate thinking only
    if not full_cache:
        logger.info("Generating full thinking for truncation cache...")
        for batch_start in tqdm(range(0, len(data), args.batch_size),
                                desc="Thinking cache", unit="batch"):
            batch = data[batch_start:batch_start + args.batch_size]
            prompts = [model.build_chat_prompt(item['question'], use_think=True)
                       for item in batch]
            responses = model.generate(prompts, max_tokens=4096, temperature=args.temperature, top_p=args.top_p)
            for item, response in zip(batch, responses):
                full_cache[item['question']] = response

    # ========================================
    # Phase 2: Truncated CoT
    # ========================================
    for N in truncation_levels:
        out_path = os.path.join(args.output_dir, f"truncated_cot_{N}_{args.split}.json")
        if not args.force and os.path.exists(out_path):
            logger.info(f"T{N}: already exists, skipping ({out_path})")
            continue

        logger.info(f"\nPhase 2: Truncated CoT with {N} thinking tokens...")

        # Prepare continuation prompts
        cont_data = []
        for item in data:
            full_resp = full_cache[item['question']]
            prompt = prompt_cache[item['question']]

            # Extract thinking part (before </think>)
            think_end = full_resp.find("</think>")
            thinking = full_resp[:think_end] if think_end >= 0 else full_resp

            # Truncate thinking to N tokens
            tokens = model.tokenizer.encode(thinking)
            if len(tokens) > N:
                truncated = model.tokenizer.decode(tokens[:N])
            else:
                truncated = thinking

            # Build continuation: prompt + truncated thinking + close think
            cont_prompt = prompt + truncated + "</think>\n\n"

            cont_data.append({
                'item': item,
                'cont_prompt': cont_prompt,
                'truncated_thinking': truncated,
                'thinking_tokens': min(len(tokens), N),
            })

        # Generate answers in batches
        all_answers = []
        for batch_start in tqdm(range(0, len(cont_data), args.batch_size),
                                desc=f"T{N} answers", unit="batch"):
            batch = cont_data[batch_start:batch_start + args.batch_size]
            batch_prompts = [d['cont_prompt'] for d in batch]
            answers = model.generate(batch_prompts, max_tokens=2048, temperature=args.temperature, top_p=args.top_p)
            all_answers.extend(answers)

        # Grade and save
        results = []
        for d, answer in zip(cont_data, all_answers):
            item = d['item']
            full_response = d['truncated_thinking'] + "</think>\n\n" + answer
            is_correct = check_correctness(full_response, item['ground_truth'])
            answer_tokens = len(model.tokenizer.encode(answer)) if answer else 0

            results.append({
                'question': item['question'],
                'ground_truth': item['ground_truth'],
                'thinking': d['truncated_thinking'],
                'answer': answer,
                'response': full_response,
                'is_correct': is_correct,
                'thinking_tokens': d['thinking_tokens'],
                'answer_tokens': answer_tokens,
                'total_tokens': d['thinking_tokens'] + answer_tokens,
                'subset': item.get('subset', ''),
                'level': item.get('level', ''),
            })

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / len(results)
        avg_total = sum(r['total_tokens'] for r in results) / len(results)
        logger.info(f"T{N}: {accuracy:.4f} ({correct}/{len(results)}), avg tokens: {avg_total:.0f}")

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'=' * 70}")
    print(f"Truncated CoT Baseline Results")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({args.split}, {len(data)} samples)")
    print()

    # Overall results
    print(f"{'Method':<35} {'Accuracy':>10} {'Avg Tokens':>12}")
    print(f"{'-' * 57}")

    if full_results is not None:
        fc = sum(1 for r in full_results if r['is_correct'])
        fa = fc / len(full_results)
        ft = sum(r['total_tokens'] for r in full_results) / len(full_results)
        print(f"{'32B Full Reasoning':<35} {fa:>10.4f} {ft:>12.0f}")

    for N in truncation_levels:
        out_path = os.path.join(args.output_dir, f"truncated_cot_{N}_{args.split}.json")
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                results = json.load(f)
            c = sum(1 for r in results if r['is_correct'])
            a = c / len(results)
            t = sum(r['total_tokens'] for r in results) / len(results)
            print(f"{'32B Truncated CoT T' + str(N):<35} {a:>10.4f} {t:>12.0f}")

    # Per-subset breakdown
    if args.dataset == "hendrycks_math":
        print(f"\nPer-Subset Breakdown:")
        header = f"{'Subset':<25}"
        if full_results is not None:
            header += f" {'Full':>8}"
        for N in truncation_levels:
            header += f" {'T'+str(N):>8}"
        print(header)
        print(f"{'-' * len(header)}")

        for subset in MATH_SUBSETS:
            row = f"{subset:<25}"

            if full_results is not None:
                sr = [r for r in full_results if r['subset'] == subset]
                if sr:
                    acc = sum(1 for r in sr if r['is_correct']) / len(sr)
                    row += f" {acc:>8.4f}"
                else:
                    row += f" {'N/A':>8}"

            for N in truncation_levels:
                out_path = os.path.join(args.output_dir, f"truncated_cot_{N}_{args.split}.json")
                if os.path.exists(out_path):
                    with open(out_path, 'r') as f:
                        results = json.load(f)
                    sr = [r for r in results if r['subset'] == subset]
                    if sr:
                        acc = sum(1 for r in sr if r['is_correct']) / len(sr)
                        row += f" {acc:>8.4f}"
                    else:
                        row += f" {'N/A':>8}"
                else:
                    row += f" {'N/A':>8}"

            print(row)

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
