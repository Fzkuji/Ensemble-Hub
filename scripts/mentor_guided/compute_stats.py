#!/usr/bin/env python3
"""
Compute Oracle accuracy and baseline statistics for all subsets.

Usage:
    python compute_stats.py
    python compute_stats.py --data-dir /path/to/data
"""

import argparse
import json
import os
from typing import Dict, List

TOKEN_LEVELS = [0, 100, 500, 1000]

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_json_data(filepath: str) -> List[Dict]:
    """Load JSON data from file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


def compute_stats(data_dir: str, subset: str, split: str = "test") -> Dict:
    """Compute statistics for a subset."""
    subset_dir = os.path.join(data_dir, subset, split)

    if not os.path.exists(subset_dir):
        return None

    # Load all token levels
    all_data = {}
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(subset_dir, f"tokens{tokens}.json")
        data = load_json_data(filepath)
        if data:
            all_data[tokens] = data

    if not all_data or TOKEN_LEVELS[0] not in all_data:
        return None

    n_samples = len(all_data[TOKEN_LEVELS[0]])

    # Baseline accuracy at each token level
    baseline_acc = {}
    for tokens in TOKEN_LEVELS:
        if tokens in all_data:
            correct = sum(1 for item in all_data[tokens] if item.get('is_correct', False))
            baseline_acc[tokens] = correct / n_samples

    # Oracle accuracy: correct if ANY token level is correct
    oracle_correct = 0
    for i in range(n_samples):
        for tokens in TOKEN_LEVELS:
            if tokens in all_data and all_data[tokens][i].get('is_correct', False):
                oracle_correct += 1
                break
    oracle_acc = oracle_correct / n_samples

    # Count samples that are always wrong (wrong at all levels)
    always_wrong = 0
    for i in range(n_samples):
        all_wrong = True
        for tokens in TOKEN_LEVELS:
            if tokens in all_data and all_data[tokens][i].get('is_correct', False):
                all_wrong = False
                break
        if all_wrong:
            always_wrong += 1

    return {
        'n_samples': n_samples,
        'baseline': baseline_acc,
        'oracle': oracle_acc,
        'always_wrong': always_wrong,
        'always_wrong_pct': always_wrong / n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Oracle and baseline stats")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Directory with split data")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to analyze (train/test)")

    args = parser.parse_args()

    print("=" * 80)
    print(f"Statistics for {args.split} split")
    print(f"Data dir: {args.data_dir}")
    print("=" * 80)

    results = {}

    for subset in SUBSETS:
        stats = compute_stats(args.data_dir, subset, args.split)
        if stats:
            results[subset] = stats
            print(f"\n{subset}:")
            print(f"  N samples: {stats['n_samples']}")
            print(f"  Baseline accuracy:")
            for tokens, acc in stats['baseline'].items():
                print(f"    tokens={tokens}: {acc:.4f} ({acc*100:.1f}%)")
            print(f"  Oracle: {stats['oracle']:.4f} ({stats['oracle']*100:.1f}%)")
            print(f"  Always wrong: {stats['always_wrong']} ({stats['always_wrong_pct']*100:.1f}%)")

    # Summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Subset':<25} {'N':<8} {'T0':<8} {'T100':<8} {'T500':<8} {'T1000':<8} {'Oracle':<8}")
    print("-" * 80)

    total_n = 0
    total_oracle_correct = 0

    for subset in SUBSETS:
        if subset in results:
            s = results[subset]
            total_n += s['n_samples']
            total_oracle_correct += int(s['oracle'] * s['n_samples'])

            row = f"{subset:<25} {s['n_samples']:<8}"
            for tokens in TOKEN_LEVELS:
                acc = s['baseline'].get(tokens, 0)
                row += f" {acc:.4f}  "
            row += f" {s['oracle']:.4f}"
            print(row)

    if total_n > 0:
        print("-" * 80)
        print(f"{'Total':<25} {total_n:<8} {'':<36} {total_oracle_correct/total_n:.4f}")

    # Save results
    output_file = os.path.join(args.data_dir, f"stats_{args.split}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
