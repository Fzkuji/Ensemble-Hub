#!/usr/bin/env python3
"""
Test script for Mentor-Guided Adaptive Inference.

This script tests the mentor-guided inference method on sample problems
and compares the entropy reduction achieved.
"""

import argparse
import json
import logging
import os
import sys

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from mentor_guided_inference import MentorGuidedInference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Test problems of varying difficulty
TEST_PROBLEMS = [
    {
        "id": "simple_algebra",
        "prompt": """Solve the following math problem step by step:

Problem: Solve for x: 2x + 3 = 7

Solution:""",
        "expected_answer": "x = 2"
    },
    {
        "id": "quadratic",
        "prompt": """Solve the following math problem step by step:

Problem: Find all real numbers x such that x^2 - 5x + 6 = 0.

Solution:""",
        "expected_answer": "x = 2 or x = 3"
    },
    {
        "id": "word_problem",
        "prompt": """Solve the following math problem step by step:

Problem: A train travels at 60 km/h for the first hour and 80 km/h for the second hour. What is the average speed?

Solution:""",
        "expected_answer": "70 km/h"
    },
    {
        "id": "geometry",
        "prompt": """Solve the following math problem step by step:

Problem: A rectangle has a perimeter of 24 cm. If its length is twice its width, find the dimensions.

Solution:""",
        "expected_answer": "width = 4 cm, length = 8 cm"
    },
    {
        "id": "harder_algebra",
        "prompt": """Solve the following math problem step by step:

Problem: If f(x) = x^2 + 2x + 1, find f(f(1)).

Solution:""",
        "expected_answer": "f(1) = 4, f(f(1)) = f(4) = 25"
    },
]


def run_comparison_test(inference: MentorGuidedInference, problem: dict, args) -> dict:
    """Run a comparison test between mentor-guided and pure student inference."""
    prompt = problem["prompt"]

    # Test 1: Pure student inference (baseline)
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing problem: {problem['id']}")
    logger.info(f"{'='*60}")

    # Get baseline entropy
    baseline_entropy, baseline_prob, _ = inference.get_student_entropy_for_next_token(prompt)
    logger.info(f"Baseline student entropy: {baseline_entropy:.4f}, top1_prob: {baseline_prob:.4f}")

    # Test 2: Mentor-guided inference
    result = inference.run_adaptive_inference(
        prompt=prompt,
        max_total_tokens=args.max_total_tokens,
        temperature=args.temperature,
        verbose=True,
    )

    return {
        "problem_id": problem["id"],
        "expected_answer": problem["expected_answer"],
        "baseline_entropy": baseline_entropy,
        "baseline_top1_prob": baseline_prob,
        "mentor_tokens_used": result["mentor_tokens_used"],
        "final_entropy": result["final_entropy"],
        "entropy_reduction": result["entropy_reduction"],
        "switch_reason": result["switch_reason"],
        "generated_text": result["generated_text"],
        "mentor_text": result["mentor_text"],
        "student_text": result["student_text"],
    }


def main():
    parser = argparse.ArgumentParser(description='Test Mentor-Guided Adaptive Inference')

    # Use smaller models for testing
    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-1.5B-Instruct',
                       help='Mentor (large) model name')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Student (small) model name')
    parser.add_argument('--entropy-threshold', type=float, default=2.0,
                       help='Entropy threshold for switching to student')
    parser.add_argument('--reduction-threshold', type=float, default=0.3,
                       help='Minimum entropy reduction to consider helpful')
    parser.add_argument('--max-mentor-tokens', type=int, default=50,
                       help='Maximum tokens from mentor')
    parser.add_argument('--max-total-tokens', type=int, default=256,
                       help='Maximum total tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--output-file', default='mentor_guided_test_results.json',
                       help='Output file for results')
    parser.add_argument('--test-single', type=int, default=None,
                       help='Test only a single problem by index (0-4)')

    args = parser.parse_args()

    # Initialize inference system
    logger.info("Initializing Mentor-Guided Inference system...")
    inference = MentorGuidedInference(
        mentor_model_name=args.mentor_model,
        student_model_name=args.student_model,
        entropy_threshold=args.entropy_threshold,
        entropy_reduction_threshold=args.reduction_threshold,
        max_mentor_tokens=args.max_mentor_tokens,
    )

    # Select test problems
    if args.test_single is not None:
        problems = [TEST_PROBLEMS[args.test_single]]
    else:
        problems = TEST_PROBLEMS

    # Run tests
    results = []
    for problem in problems:
        result = run_comparison_test(inference, problem, args)
        results.append(result)

        # Print summary
        logger.info(f"\n--- Summary for {problem['id']} ---")
        logger.info(f"Baseline entropy: {result['baseline_entropy']:.4f}")
        logger.info(f"Final entropy: {result['final_entropy']:.4f}" if result['final_entropy'] else "N/A")
        logger.info(f"Entropy reduction: {result['entropy_reduction']:.2%}" if result['entropy_reduction'] else "N/A")
        logger.info(f"Mentor tokens used: {result['mentor_tokens_used']}")
        logger.info(f"Switch reason: {result['switch_reason']}")

    # Overall statistics
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL STATISTICS")
    logger.info(f"{'='*60}")

    avg_baseline = sum(r['baseline_entropy'] for r in results) / len(results)
    valid_finals = [r['final_entropy'] for r in results if r['final_entropy'] is not None]
    avg_final = sum(valid_finals) / len(valid_finals) if valid_finals else 0

    valid_reductions = [r['entropy_reduction'] for r in results if r['entropy_reduction'] is not None]
    avg_reduction = sum(valid_reductions) / len(valid_reductions) if valid_reductions else 0

    avg_mentor_tokens = sum(r['mentor_tokens_used'] for r in results) / len(results)

    logger.info(f"Average baseline entropy: {avg_baseline:.4f}")
    logger.info(f"Average final entropy: {avg_final:.4f}")
    logger.info(f"Average entropy reduction: {avg_reduction:.2%}")
    logger.info(f"Average mentor tokens used: {avg_mentor_tokens:.1f}")

    # Save results
    output_path = os.path.join(script_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump({
            "config": {
                "mentor_model": args.mentor_model,
                "student_model": args.student_model,
                "entropy_threshold": args.entropy_threshold,
                "reduction_threshold": args.reduction_threshold,
                "max_mentor_tokens": args.max_mentor_tokens,
            },
            "statistics": {
                "avg_baseline_entropy": avg_baseline,
                "avg_final_entropy": avg_final,
                "avg_entropy_reduction": avg_reduction,
                "avg_mentor_tokens": avg_mentor_tokens,
            },
            "results": results
        }, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
