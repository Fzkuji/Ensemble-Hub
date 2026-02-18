#!/usr/bin/env python3
"""
Insight Type Ablation Study (4FYq W2).

Tests the contribution of each Thinking Insight type by removing it
from the structured prompt. Runs mentor+intern pipeline at a specified
token level (default T500) on MATH test.

Variants:
  full         - All 4 types: Goal + Planning + Retrieval + Action
  no-goal      - Remove Goal
  no-planning  - Remove Planning
  no-retrieval - Remove Retrieval
  no-action    - Remove Action
  unstructured - Simple CoT prompt (no insight structure)

Usage:
    # Run a single variant
    python collect_insight_ablation.py \
        --variant no-goal \
        --mentor-gpus 2 --intern-gpus 3

    # Run all variants in parallel (3 pairs of GPUs)
    python collect_insight_ablation.py --variant full         --mentor-gpus 2 --intern-gpus 3 &
    python collect_insight_ablation.py --variant no-goal      --mentor-gpus 4 --intern-gpus 5 &
    python collect_insight_ablation.py --variant no-planning  --mentor-gpus 6 --intern-gpus 7 &
    wait
    # Then run remaining variants on freed GPUs
    python collect_insight_ablation.py --variant no-retrieval --mentor-gpus 2 --intern-gpus 3 &
    python collect_insight_ablation.py --variant no-action    --mentor-gpus 4 --intern-gpus 5 &
    wait
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Ablation Prompt Variants
# ============================================================
ABLATION_PROMPTS = {
    "full": (
        "You are a reasoning assistant. Analyze the given problem and "
        "follow the structured insights below.\n\n"
        "INSIGHTS\n"
        "[Thinking Insights]\n"
        "1. Goal: <objective, constraints, and required output form>\n"
        "2. Planning: <high-level strategy; subproblem decomposition; edge cases>\n"
        "3. Retrieval: <relevant facts, formulas, or definitions; N/A if none>\n"
        "4. Action: <concrete steps and intermediate calculations>\n\n"
        "CONSTRAINTS\n"
        "- Keep each component concise.\n"
        "- Maintain notational consistency with the original problem.\n"
        "- Put your final answer within \\boxed{}."
    ),
    "no-goal": (
        "You are a reasoning assistant. Analyze the given problem and "
        "follow the structured insights below.\n\n"
        "INSIGHTS\n"
        "[Thinking Insights]\n"
        "1. Planning: <high-level strategy; subproblem decomposition; edge cases>\n"
        "2. Retrieval: <relevant facts, formulas, or definitions; N/A if none>\n"
        "3. Action: <concrete steps and intermediate calculations>\n\n"
        "CONSTRAINTS\n"
        "- Keep each component concise.\n"
        "- Maintain notational consistency with the original problem.\n"
        "- Put your final answer within \\boxed{}."
    ),
    "no-planning": (
        "You are a reasoning assistant. Analyze the given problem and "
        "follow the structured insights below.\n\n"
        "INSIGHTS\n"
        "[Thinking Insights]\n"
        "1. Goal: <objective, constraints, and required output form>\n"
        "2. Retrieval: <relevant facts, formulas, or definitions; N/A if none>\n"
        "3. Action: <concrete steps and intermediate calculations>\n\n"
        "CONSTRAINTS\n"
        "- Keep each component concise.\n"
        "- Maintain notational consistency with the original problem.\n"
        "- Put your final answer within \\boxed{}."
    ),
    "no-retrieval": (
        "You are a reasoning assistant. Analyze the given problem and "
        "follow the structured insights below.\n\n"
        "INSIGHTS\n"
        "[Thinking Insights]\n"
        "1. Goal: <objective, constraints, and required output form>\n"
        "2. Planning: <high-level strategy; subproblem decomposition; edge cases>\n"
        "3. Action: <concrete steps and intermediate calculations>\n\n"
        "CONSTRAINTS\n"
        "- Keep each component concise.\n"
        "- Maintain notational consistency with the original problem.\n"
        "- Put your final answer within \\boxed{}."
    ),
    "no-action": (
        "You are a reasoning assistant. Analyze the given problem and "
        "follow the structured insights below.\n\n"
        "INSIGHTS\n"
        "[Thinking Insights]\n"
        "1. Goal: <objective, constraints, and required output form>\n"
        "2. Planning: <high-level strategy; subproblem decomposition; edge cases>\n"
        "3. Retrieval: <relevant facts, formulas, or definitions; N/A if none>\n\n"
        "CONSTRAINTS\n"
        "- Keep each component concise.\n"
        "- Maintain notational consistency with the original problem.\n"
        "- Put your final answer within \\boxed{}."
    ),
    "unstructured": (
        "Please reason step by step, and put your final answer within \\boxed{}."
    ),
}

MATH_SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def main():
    parser = argparse.ArgumentParser(description="Insight Type Ablation Study")
    parser.add_argument("--variant", type=str, required=True,
                        choices=list(ABLATION_PROMPTS.keys()),
                        help="Which ablation variant to run")
    parser.add_argument("--mentor-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--intern-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--mentor-gpus", type=str, required=True,
                        help="GPU(s) for mentor model (e.g., '2' or '2,3')")
    parser.add_argument("--intern-gpus", type=str, required=True,
                        help="GPU(s) for intern model (e.g., '3' or '4,5')")
    parser.add_argument("--token-level", type=int, default=500,
                        help="Mentor token budget (default: 500)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--mentor-memory-util", type=float, default=0.9)
    parser.add_argument("--intern-memory-util", type=float, default=0.9)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output exists")

    args = parser.parse_args()

    # Set the system prompt BEFORE importing collection functions
    # (they reference the module-level SYSTEM_PROMPT)
    import collect_data_vllm_think as pipeline
    pipeline.SYSTEM_PROMPT = ABLATION_PROMPTS[args.variant]
    logger.info(f"Ablation variant: {args.variant}")
    logger.info(f"System prompt: {pipeline.SYSTEM_PROMPT[:80]}...")

    # Output directory
    if args.output_dir is None:
        base = "/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected"
        mentor_short = args.mentor_model.split('/')[-1]
        intern_short = args.intern_model.split('/')[-1]
        args.output_dir = os.path.join(
            base,
            f"insight_ablation_{args.variant}_m{mentor_short}_i{intern_short}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if output already exists
    out_path = os.path.join(args.output_dir, f"tokens{args.token_level}_{args.split}.json")
    if not args.force and os.path.exists(out_path):
        logger.info(f"Output already exists: {out_path}")
        with open(out_path, 'r') as f:
            results = json.load(f)
        _print_results(args.variant, args.token_level, results)
        return

    # Parse GPUs
    mentor_gpu_ids = [int(g) for g in args.mentor_gpus.split(",")]
    intern_gpu_ids = [int(g) for g in args.intern_gpus.split(",")]

    # Load models
    logger.info(f"Loading mentor {args.mentor_model} on GPU {mentor_gpu_ids}...")
    mentor = pipeline.VLLMInference(
        args.mentor_model,
        gpu_ids=mentor_gpu_ids,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.mentor_memory_util,
    )

    logger.info(f"Loading intern {args.intern_model} on GPU {intern_gpu_ids}...")
    intern = pipeline.VLLMInference(
        args.intern_model,
        gpu_ids=intern_gpu_ids,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.intern_memory_util,
    )

    # Load MATH test data (all subsets)
    data = pipeline.load_hendrycks_math_all(args.split)
    logger.info(f"Loaded {len(data)} problems")

    # Collect data at the specified token level
    logger.info(f"Collecting T{args.token_level} with variant={args.variant}...")
    results = pipeline.collect_data_for_token_level(
        mentor_model=mentor,
        intern_model=intern,
        data=data,
        token_level=args.token_level,
        batch_size=args.batch_size,
        use_think=True,
    )

    # Save results
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} results to {out_path}")

    _print_results(args.variant, args.token_level, results)

    # Cleanup
    mentor.cleanup()
    intern.cleanup()


def _print_results(variant: str, token_level: int, results: list):
    """Print summary of ablation results."""
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / len(results) if results else 0

    print(f"\n{'=' * 60}")
    print(f"Insight Ablation: {variant}")
    print(f"{'=' * 60}")
    print(f"Token level: T{token_level}")
    print(f"Total: {accuracy:.4f} ({correct}/{len(results)})")
    print()

    print(f"Per-subset:")
    for subset in MATH_SUBSETS:
        sr = [r for r in results if r.get('subset', '') == subset]
        if sr:
            sc = sum(1 for r in sr if r['is_correct'])
            sa = sc / len(sr)
            print(f"  {subset:<25} {sa:.4f} ({sc}/{len(sr)})")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
