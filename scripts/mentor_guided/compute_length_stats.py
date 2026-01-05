#!/usr/bin/env python3
"""
Compute generation length statistics for all token levels.
"""

import argparse
import json
import os
import numpy as np
from typing import Dict, List


def compute_length_stats(data_dir: str, subset: str, split: str = "test"):
    """Compute length statistics for a subset."""
    subset_dir = os.path.join(data_dir, subset, split)

    if not os.path.exists(subset_dir):
        print(f"Directory not found: {subset_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Length Statistics: {subset} ({split})")
    print(f"{'='*80}")

    token_levels = [-1, 0, 100, 500, 1000]

    for tokens in token_levels:
        filepath = os.path.join(subset_dir, f"tokens{tokens}.json")
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        if not data:
            continue

        # Extract lengths
        lengths = []
        correct_lengths = []
        wrong_lengths = []

        for item in data:
            # Try different possible fields for generation length
            length = None
            if 'generated_text' in item:
                length = len(item['generated_text'])
            elif 'generation' in item:
                length = len(item['generation'])
            elif 'output' in item:
                length = len(item['output'])

            if length is not None:
                lengths.append(length)
                if item.get('is_correct', False):
                    correct_lengths.append(length)
                else:
                    wrong_lengths.append(length)

        if lengths:
            print(f"\nToken Level {tokens}:")
            print(f"  Samples: {len(lengths)}")
            print(f"  Mean length: {np.mean(lengths):.1f} chars")
            print(f"  Median length: {np.median(lengths):.1f} chars")
            print(f"  Min/Max: {min(lengths)} / {max(lengths)} chars")
            print(f"  Std: {np.std(lengths):.1f}")

            if correct_lengths:
                print(f"  Correct answers - Mean: {np.mean(correct_lengths):.1f}, Median: {np.median(correct_lengths):.1f}")
            if wrong_lengths:
                print(f"  Wrong answers - Mean: {np.mean(wrong_lengths):.1f}, Median: {np.median(wrong_lengths):.1f}")

            # Accuracy
            correct = sum(1 for item in data if item.get('is_correct', False))
            print(f"  Accuracy: {correct}/{len(data)} ({correct/len(data)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compute length statistics")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with split data")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset (if None, auto-detect)")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to analyze (train/test)")

    args = parser.parse_args()

    # Auto-detect subset if not specified
    if args.subset:
        subsets = [args.subset]
    else:
        subsets = []
        for name in os.listdir(args.data_dir):
            subset_dir = os.path.join(args.data_dir, name, args.split)
            if os.path.isdir(subset_dir):
                token_file = os.path.join(subset_dir, "tokens0.json")
                if os.path.exists(token_file):
                    subsets.append(name)
        subsets = sorted(subsets)

    for subset in subsets:
        compute_length_stats(args.data_dir, subset, args.split)


if __name__ == "__main__":
    main()
