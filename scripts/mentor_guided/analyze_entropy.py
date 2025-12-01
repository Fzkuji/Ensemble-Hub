#!/usr/bin/env python3
"""
Entropy Analysis for Mentor-Guided Adaptive Inference.

This script provides detailed analysis and visualization of how
the student model's entropy changes as the mentor provides more tokens.
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from mentor_guided_inference import MentorGuidedInference, InferenceState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_entropy_trajectory(
    inference: MentorGuidedInference,
    prompt: str,
    num_mentor_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Collect detailed entropy trajectory as mentor generates tokens.

    This function doesn't stop early - it collects all data points
    so we can analyze the full trajectory.
    """
    # Get baseline entropy
    baseline_entropy, baseline_prob, _ = inference.get_student_entropy_for_next_token(prompt)

    entropies = [baseline_entropy]
    top1_probs = [baseline_prob]
    tokens = ["<baseline>"]
    mentor_text = ""

    logger.info(f"Collecting entropy trajectory for {num_mentor_tokens} mentor tokens...")
    logger.info(f"Baseline entropy: {baseline_entropy:.4f}")

    for i in range(num_mentor_tokens):
        # Generate one token from mentor
        token_id, token_text = inference.generate_mentor_token(prompt, mentor_text, temperature)

        # Check for EOS
        if token_id == inference.mentor_tokenizer.eos_token_id:
            logger.info(f"Mentor reached EOS at token {i}")
            break

        mentor_text += token_text

        # Measure student's entropy with new context
        entropy, top1_prob, _ = inference.get_student_entropy_for_next_token(prompt, mentor_text)

        entropies.append(entropy)
        top1_probs.append(top1_prob)
        tokens.append(token_text)

        if (i + 1) % 10 == 0:
            reduction = (baseline_entropy - entropy) / baseline_entropy if baseline_entropy > 0 else 0
            logger.info(f"Token {i+1}: entropy={entropy:.4f}, reduction={reduction:.2%}")

    return {
        "prompt": prompt,
        "mentor_text": mentor_text,
        "baseline_entropy": baseline_entropy,
        "entropies": entropies,
        "top1_probs": top1_probs,
        "tokens": tokens,
        "num_tokens": len(entropies) - 1,  # Exclude baseline
    }


def plot_entropy_trajectory(
    data: Dict[str, Any],
    output_path: str,
    entropy_threshold: float = 2.0,
    reduction_threshold: float = 0.3,
):
    """Plot the entropy trajectory with threshold lines."""
    entropies = data["entropies"]
    baseline = data["baseline_entropy"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Raw entropy trajectory
    ax1 = axes[0, 0]
    ax1.plot(range(len(entropies)), entropies, 'b-', linewidth=2, label='Student entropy')
    ax1.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.2f})', alpha=0.7)
    ax1.axhline(y=entropy_threshold, color='g', linestyle=':', label=f'Threshold ({entropy_threshold})', alpha=0.7)
    ax1.set_xlabel('Mentor tokens')
    ax1.set_ylabel('Entropy (bits)')
    ax1.set_title('Student Entropy vs Mentor Tokens')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Entropy reduction
    ax2 = axes[0, 1]
    reductions = [(baseline - e) / baseline * 100 if baseline > 0 else 0 for e in entropies]
    ax2.plot(range(len(reductions)), reductions, 'g-', linewidth=2)
    ax2.axhline(y=reduction_threshold * 100, color='r', linestyle='--',
                label=f'Threshold ({reduction_threshold*100}%)', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Mentor tokens')
    ax2.set_ylabel('Entropy reduction (%)')
    ax2.set_title('Entropy Reduction from Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Top-1 probability
    ax3 = axes[1, 0]
    top1_probs = data["top1_probs"]
    ax3.plot(range(len(top1_probs)), top1_probs, 'purple', linewidth=2)
    ax3.set_xlabel('Mentor tokens')
    ax3.set_ylabel('Top-1 probability')
    ax3.set_title('Student Model Confidence (Top-1 Prob)')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sliding window entropy
    ax4 = axes[1, 1]
    window_size = 5
    if len(entropies) >= window_size:
        sliding_avg = []
        for i in range(len(entropies)):
            start = max(0, i - window_size + 1)
            sliding_avg.append(np.mean(entropies[start:i+1]))
        ax4.plot(range(len(sliding_avg)), sliding_avg, 'orange', linewidth=2, label='Sliding avg')
        ax4.plot(range(len(entropies)), entropies, 'b-', alpha=0.3, label='Raw entropy')
        ax4.axhline(y=entropy_threshold, color='g', linestyle=':', label=f'Threshold', alpha=0.7)
        ax4.set_xlabel('Mentor tokens')
        ax4.set_ylabel('Entropy (bits)')
        ax4.set_title(f'Sliding Window Average (window={window_size})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")
    plt.close()


def find_optimal_switch_point(
    entropies: List[float],
    baseline: float,
    entropy_threshold: float = 2.0,
    reduction_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Find the optimal point to switch from mentor to student."""
    switch_points = {
        "by_absolute_threshold": None,
        "by_reduction_threshold": None,
        "by_minimum_entropy": None,
    }

    for i, entropy in enumerate(entropies):
        # Check absolute threshold
        if switch_points["by_absolute_threshold"] is None and entropy < entropy_threshold:
            switch_points["by_absolute_threshold"] = {
                "token_index": i,
                "entropy": entropy,
                "reduction": (baseline - entropy) / baseline if baseline > 0 else 0
            }

        # Check reduction threshold
        reduction = (baseline - entropy) / baseline if baseline > 0 else 0
        if switch_points["by_reduction_threshold"] is None and reduction > reduction_threshold:
            switch_points["by_reduction_threshold"] = {
                "token_index": i,
                "entropy": entropy,
                "reduction": reduction
            }

    # Find minimum entropy point
    min_idx = np.argmin(entropies)
    min_entropy = entropies[min_idx]
    switch_points["by_minimum_entropy"] = {
        "token_index": min_idx,
        "entropy": min_entropy,
        "reduction": (baseline - min_entropy) / baseline if baseline > 0 else 0
    }

    return switch_points


def main():
    parser = argparse.ArgumentParser(description='Analyze Entropy Trajectory')

    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-1.5B-Instruct',
                       help='Mentor model name')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Student model name')
    parser.add_argument('--num-tokens', type=int, default=100,
                       help='Number of mentor tokens to collect')
    parser.add_argument('--entropy-threshold', type=float, default=2.0,
                       help='Entropy threshold')
    parser.add_argument('--reduction-threshold', type=float, default=0.3,
                       help='Reduction threshold')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for plots and data')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt to analyze')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = script_dir

    os.makedirs(args.output_dir, exist_ok=True)

    # Default prompt
    if args.prompt is None:
        args.prompt = """Solve the following math problem step by step:

Problem: A triangle has sides of length 3, 4, and 5. What is its area?

Solution:"""

    # Initialize inference system
    logger.info("Initializing models...")
    inference = MentorGuidedInference(
        mentor_model_name=args.mentor_model,
        student_model_name=args.student_model,
        entropy_threshold=args.entropy_threshold,
        entropy_reduction_threshold=args.reduction_threshold,
        max_mentor_tokens=args.num_tokens,
    )

    # Collect entropy trajectory
    logger.info("Collecting entropy trajectory...")
    data = collect_entropy_trajectory(
        inference,
        args.prompt,
        num_mentor_tokens=args.num_tokens,
        temperature=args.temperature,
    )

    # Find optimal switch points
    switch_points = find_optimal_switch_point(
        data["entropies"],
        data["baseline_entropy"],
        args.entropy_threshold,
        args.reduction_threshold,
    )

    logger.info("\n" + "="*60)
    logger.info("OPTIMAL SWITCH POINTS")
    logger.info("="*60)

    for method, point in switch_points.items():
        if point:
            logger.info(f"{method}:")
            logger.info(f"  Token index: {point['token_index']}")
            logger.info(f"  Entropy: {point['entropy']:.4f}")
            logger.info(f"  Reduction: {point['reduction']:.2%}")
        else:
            logger.info(f"{method}: Not reached")

    # Plot trajectory
    plot_path = os.path.join(args.output_dir, "entropy_trajectory.png")
    plot_entropy_trajectory(
        data,
        plot_path,
        args.entropy_threshold,
        args.reduction_threshold,
    )

    # Save data
    data_path = os.path.join(args.output_dir, "entropy_data.json")
    save_data = {
        "config": {
            "mentor_model": args.mentor_model,
            "student_model": args.student_model,
            "num_tokens": args.num_tokens,
            "entropy_threshold": args.entropy_threshold,
            "reduction_threshold": args.reduction_threshold,
        },
        "prompt": args.prompt,
        "baseline_entropy": data["baseline_entropy"],
        "num_mentor_tokens": data["num_tokens"],
        "mentor_text": data["mentor_text"],
        "entropies": data["entropies"],
        "top1_probs": data["top1_probs"],
        "switch_points": switch_points,
        "statistics": {
            "min_entropy": float(min(data["entropies"])),
            "max_entropy": float(max(data["entropies"])),
            "mean_entropy": float(np.mean(data["entropies"])),
            "std_entropy": float(np.std(data["entropies"])),
            "max_reduction": float(max((data["baseline_entropy"] - e) / data["baseline_entropy"]
                                       for e in data["entropies"]) if data["baseline_entropy"] > 0 else 0),
        }
    }

    with open(data_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Data saved to: {data_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Baseline entropy: {data['baseline_entropy']:.4f}")
    logger.info(f"Min entropy: {save_data['statistics']['min_entropy']:.4f}")
    logger.info(f"Max reduction: {save_data['statistics']['max_reduction']:.2%}")
    logger.info(f"Mentor tokens collected: {data['num_tokens']}")
    logger.info(f"Mentor text preview: {data['mentor_text'][:200]}...")


if __name__ == "__main__":
    main()
