#!/usr/bin/env python3
"""
Fix is_correct labels for GSM8K data.

The original code incorrectly tried to extract \boxed{} from GSM8K ground_truth,
but GSM8K ground_truth is just a plain number (e.g., "72"), not boxed.

This script re-evaluates all samples and fixes the is_correct field.

Usage:
    python fix_gsm8k_labels.py --data-dir /path/to/gsm8k_data
"""

import argparse
import json
import os
import sys

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grader import grade_answer


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
    """Check if answer is correct, handling both MATH and GSM8K formats."""
    pred_answer = extract_boxed_answer(prediction)

    # Try to extract boxed answer from ground_truth first
    true_answer = extract_boxed_answer(ground_truth)

    # If ground_truth has no \boxed{}, use it directly (GSM8K format)
    if not true_answer:
        true_answer = ground_truth.strip()

    if not pred_answer or not true_answer:
        return False

    return grade_answer(pred_answer, true_answer)


def fix_data_file(filepath: str) -> dict:
    """Fix is_correct labels in a single data file.

    Returns:
        dict with 'total', 'correct', 'fixed' counts
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    total = len(data)
    fixed = 0

    for item in data:
        old_correct = item.get('is_correct', False)
        new_correct = check_correctness(item['response'], item['ground_truth'])

        if old_correct != new_correct:
            fixed += 1
            item['is_correct'] = new_correct

    # Save fixed data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    correct = sum(1 for d in data if d['is_correct'])

    return {
        'total': total,
        'correct': correct,
        'fixed': fixed,
    }


def main():
    parser = argparse.ArgumentParser(description="Fix is_correct labels for GSM8K data")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory (e.g., .../gsm8k_think_xxx)")

    args = parser.parse_args()

    print("=" * 60)
    print("Fixing is_correct labels for GSM8K data")
    print(f"Data dir: {args.data_dir}")
    print("=" * 60)

    # Find all subset directories
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory not found: {args.data_dir}")
        return

    # Token levels to fix
    token_levels = ['-1', '0', '100', '500', '1000']
    splits = ['train', 'test']

    # Find subsets (directories that contain train/test subdirs)
    subsets = []
    for name in os.listdir(args.data_dir):
        subset_path = os.path.join(args.data_dir, name)
        if os.path.isdir(subset_path):
            # Check if it has train or test subdirectory
            if os.path.isdir(os.path.join(subset_path, 'train')) or \
               os.path.isdir(os.path.join(subset_path, 'test')):
                subsets.append(name)

    if not subsets:
        print("No subsets found!")
        return

    print(f"Found subsets: {subsets}")
    print()

    total_fixed = 0

    for subset in sorted(subsets):
        print(f"\n{subset}:")

        for split in splits:
            split_dir = os.path.join(args.data_dir, subset, split)
            if not os.path.isdir(split_dir):
                continue

            for tokens in token_levels:
                filepath = os.path.join(split_dir, f"tokens{tokens}.json")
                result = fix_data_file(filepath)

                if result:
                    acc = result['correct'] / result['total'] * 100 if result['total'] > 0 else 0
                    print(f"  {split}/tokens{tokens}: {result['correct']}/{result['total']} correct ({acc:.1f}%), {result['fixed']} fixed")
                    total_fixed += result['fixed']

    print()
    print("=" * 60)
    print(f"Total fixed: {total_fixed} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
