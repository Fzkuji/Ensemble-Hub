#!/usr/bin/env python3
"""
Plot Entropy Trend Curves

For sufficient and insufficient samples, plot average entropy
vs token position at different token levels (T=100, 500, 1000).

Usage:
    python plot_entropy_trend.py --data-dir /path/to/data
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def load_data(data_dir: str, subset: str, token_level: int) -> List[Dict]:
    """Load PPL analysis results."""
    ppl_file = os.path.join(data_dir, "ppl_analysis", f"{subset}_tokens{token_level}_ppl.json")
    if not os.path.exists(ppl_file):
        return []
    with open(ppl_file, 'r') as f:
        return json.load(f)


def compute_average_curve(samples: List[Dict], field: str, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average curve up to max_len tokens.

    Returns:
        x: token positions (1 to max_len)
        y: average values at each position
    """
    # Filter samples that have the field
    valid = [s for s in samples if field in s and s[field] and len(s[field]) > 0]
    if not valid:
        return np.array([]), np.array([])

    # Collect values at each position (only up to max_len)
    position_values = [[] for _ in range(max_len)]

    for s in valid:
        values = s[field]
        # Only use values up to max_len
        for i, v in enumerate(values[:max_len]):
            if not np.isnan(v) and not np.isinf(v):
                position_values[i].append(v)

    # Compute mean at each position (only where we have data)
    x_list = []
    y_list = []
    for i, vals in enumerate(position_values):
        if vals:  # Only include positions with data
            x_list.append(i + 1)  # 1-indexed
            y_list.append(np.mean(vals))

    return np.array(x_list), np.array(y_list)


def plot_entropy_by_category(
    sufficient: Dict[int, List[Dict]],
    insufficient: Dict[int, List[Dict]],
    token_levels: List[int],
    output_path: str,
):
    """
    Plot entropy trends: Sufficient vs Insufficient side by side.
    Each curve ends at its corresponding token level.
    """
    colors = {100: '#1f77b4', 500: '#ff7f0e', 1000: '#2ca02c'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Sufficient
    ax = axes[0]
    for tl in token_levels:
        samples = sufficient.get(tl, [])
        if not samples:
            continue
        x, y = compute_average_curve(samples, 'per_token_entropy', max_len=tl)
        if len(x) > 0:
            ax.plot(x, y, color=colors[tl], label=f'T={tl} (n={len(samples)})', linewidth=1.5)

    ax.set_title('Sufficient', fontsize=14)
    ax.set_xlabel('Token Position (log scale)', fontsize=12)
    ax.set_ylabel('Average Entropy', fontsize=12)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right plot: Insufficient
    ax = axes[1]
    for tl in token_levels:
        samples = insufficient.get(tl, [])
        if not samples:
            continue
        x, y = compute_average_curve(samples, 'per_token_entropy', max_len=tl)
        if len(x) > 0:
            ax.plot(x, y, color=colors[tl], label=f'T={tl} (n={len(samples)})', linewidth=1.5)

    ax.set_title('Insufficient', fontsize=14)
    ax.set_xlabel('Token Position (log scale)', fontsize=12)
    ax.set_ylabel('Average Entropy', fontsize=12)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_entropy_comparison(
    sufficient: Dict[int, List[Dict]],
    insufficient: Dict[int, List[Dict]],
    token_levels: List[int],
    output_path: str,
):
    """
    Plot entropy comparison: Sufficient vs Insufficient for each token level.
    """
    fig, axes = plt.subplots(1, len(token_levels), figsize=(5 * len(token_levels), 4))
    if len(token_levels) == 1:
        axes = [axes]

    for idx, tl in enumerate(token_levels):
        ax = axes[idx]

        # Sufficient
        suff = sufficient.get(tl, [])
        if suff:
            x, y = compute_average_curve(suff, 'per_token_entropy', max_len=tl)
            if len(x) > 0:
                ax.plot(x, y, color='green', label=f'Sufficient (n={len(suff)})', linewidth=1.5)

        # Insufficient
        insuff = insufficient.get(tl, [])
        if insuff:
            x, y = compute_average_curve(insuff, 'per_token_entropy', max_len=tl)
            if len(x) > 0:
                ax.plot(x, y, color='red', label=f'Insufficient (n={len(insuff)})', linewidth=1.5)

        ax.set_title(f'T = {tl}', fontsize=14)
        ax.set_xlabel('Token Position (log scale)', fontsize=12)
        ax.set_ylabel('Average Entropy', fontsize=12)
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot entropy trend curves")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data-dir/ppl_analysis)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset (default: all)")
    parser.add_argument("--token-levels", type=str, default="100,500,1000",
                        help="Comma-separated token levels")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "ppl_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    token_levels = [int(x) for x in args.token_levels.split(',')]

    # Find all subsets
    if args.subset:
        subsets = [args.subset]
    else:
        subsets = set()
        ppl_dir = os.path.join(args.data_dir, "ppl_analysis")
        if os.path.exists(ppl_dir):
            for name in os.listdir(ppl_dir):
                if name.endswith('_ppl.json'):
                    parts = name.replace('_ppl.json', '').rsplit('_tokens', 1)
                    if len(parts) == 2:
                        subsets.add(parts[0])
        subsets = sorted(subsets)

    print(f"Subsets: {subsets}")
    print(f"Token levels: {token_levels}")

    # Collect data
    sufficient = {tl: [] for tl in token_levels}
    insufficient = {tl: [] for tl in token_levels}

    for subset in subsets:
        for tl in token_levels:
            data = load_data(args.data_dir, subset, tl)
            for item in data:
                if item.get('is_correct', False):
                    sufficient[tl].append(item)
                else:
                    insufficient[tl].append(item)

    # Print stats
    print("\nData statistics:")
    for tl in token_levels:
        print(f"  T={tl}: Sufficient={len(sufficient[tl])}, Insufficient={len(insufficient[tl])}")

    # Check if we have per-token entropy data
    has_data = any(
        any('per_token_entropy' in s for s in sufficient[tl] + insufficient[tl])
        for tl in token_levels
    )

    if not has_data:
        print("No per_token_entropy data found. Run compute_ppl_entropy.py with --save-per-token first.")
        return

    # Plot
    plot_entropy_by_category(
        sufficient, insufficient, token_levels,
        os.path.join(args.output_dir, 'entropy_trend_by_category.png')
    )

    plot_entropy_comparison(
        sufficient, insufficient, token_levels,
        os.path.join(args.output_dir, 'entropy_trend_comparison.png')
    )


if __name__ == "__main__":
    main()
