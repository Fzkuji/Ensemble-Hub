#!/usr/bin/env python3
"""
Plot PPL and Entropy Distribution Comparison

Compare the distribution of perplexity and entropy between
sufficient (correct) and insufficient (incorrect) samples.

Usage:
    python plot_ppl_distribution.py --data-dir /path/to/data
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Set Times New Roman font (with fallback)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def load_data(data_dir: str, subset: str, token_level: int) -> List[Dict]:
    """Load PPL analysis results."""
    ppl_file = os.path.join(data_dir, "ppl_analysis", f"{subset}_tokens{token_level}_ppl.json")
    if not os.path.exists(ppl_file):
        return []
    with open(ppl_file, 'r') as f:
        return json.load(f)


def plot_distribution(
    sufficient: Dict[int, List[Dict]],
    insufficient: Dict[int, List[Dict]],
    token_levels: List[int],
    output_path: str,
    entropy_only: bool = False,
):
    """Plot PPL and Entropy distribution comparison."""
    if entropy_only:
        # Single row for entropy only
        fig, axes = plt.subplots(1, len(token_levels), figsize=(4 * len(token_levels), 3.2))
        if len(token_levels) == 1:
            axes = [axes]

        for idx, tl in enumerate(token_levels):
            ax = axes[idx]
            suff_ent = [s['avg_entropy'] for s in sufficient[tl] if s['avg_entropy'] < 1]
            insuff_ent = [s['avg_entropy'] for s in insufficient[tl] if s['avg_entropy'] < 1]

            if suff_ent and insuff_ent:
                ax.hist(suff_ent, bins=25, alpha=0.7, color='#2ca02c', label='Sufficient', density=True, edgecolor='white', linewidth=0.5)
                ax.hist(insuff_ent, bins=25, alpha=0.7, color='#d62728', label='Insufficient', density=True, edgecolor='white', linewidth=0.5)
                ax.axvline(np.mean(suff_ent), color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8)
                ax.axvline(np.mean(insuff_ent), color='#d62728', linestyle='--', linewidth=2, alpha=0.8)

            if idx == 0:
                ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'Guidance = {tl} tokens', fontsize=13, fontweight='bold')
            ax.set_xlabel('Average Token Entropy', fontsize=12)
            if idx == len(token_levels) - 1:
                ax.legend(loc='upper right', fontsize=10)
            ax.tick_params(labelsize=10)
    else:
        # Two rows: PPL and Entropy
        fig, axes = plt.subplots(2, len(token_levels), figsize=(4 * len(token_levels), 6))
        if len(token_levels) == 1:
            axes = axes.reshape(2, 1)

        for idx, tl in enumerate(token_levels):
            # Top row: PPL
            ax = axes[0, idx]
            suff_ppl = [s['ppl'] for s in sufficient[tl] if 1 < s['ppl'] < 2]
            insuff_ppl = [s['ppl'] for s in insufficient[tl] if 1 < s['ppl'] < 2]

            if suff_ppl and insuff_ppl:
                ax.hist(suff_ppl, bins=25, alpha=0.7, color='#2ca02c', label='Sufficient', density=True, edgecolor='white', linewidth=0.5)
                ax.hist(insuff_ppl, bins=25, alpha=0.7, color='#d62728', label='Insufficient', density=True, edgecolor='white', linewidth=0.5)
                ax.axvline(np.mean(suff_ppl), color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8)
                ax.axvline(np.mean(insuff_ppl), color='#d62728', linestyle='--', linewidth=2, alpha=0.8)

            if idx == 0:
                ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Guidance = {tl} tokens', fontsize=12, fontweight='bold')
            ax.set_xlabel('Perplexity', fontsize=11)
            if idx == len(token_levels) - 1:
                ax.legend(loc='upper right', fontsize=9)
            ax.tick_params(labelsize=9)

            # Bottom row: Entropy
            ax = axes[1, idx]
            suff_ent = [s['avg_entropy'] for s in sufficient[tl] if s['avg_entropy'] < 1]
            insuff_ent = [s['avg_entropy'] for s in insufficient[tl] if s['avg_entropy'] < 1]

            if suff_ent and insuff_ent:
                ax.hist(suff_ent, bins=25, alpha=0.7, color='#2ca02c', label='Sufficient', density=True, edgecolor='white', linewidth=0.5)
                ax.hist(insuff_ent, bins=25, alpha=0.7, color='#d62728', label='Insufficient', density=True, edgecolor='white', linewidth=0.5)
                ax.axvline(np.mean(suff_ent), color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8)
                ax.axvline(np.mean(insuff_ent), color='#d62728', linestyle='--', linewidth=2, alpha=0.8)

            if idx == 0:
                ax.set_ylabel('Density', fontsize=11)
            ax.set_xlabel('Average Entropy', fontsize=11)
            if idx == len(token_levels) - 1:
                ax.legend(loc='upper right', fontsize=9)
            ax.tick_params(labelsize=9)

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PPL and Entropy distribution")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data-dir/ppl_analysis)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset (default: all)")
    parser.add_argument("--token-levels", type=str, default="100,500,1000",
                        help="Comma-separated token levels")
    parser.add_argument("--entropy-only", action="store_true",
                        help="Only plot entropy distribution (single row)")
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
        suff_ppl = [s['ppl'] for s in sufficient[tl] if s['ppl'] < 10]
        insuff_ppl = [s['ppl'] for s in insufficient[tl] if s['ppl'] < 10]
        suff_ent = [s['avg_entropy'] for s in sufficient[tl] if s['avg_entropy'] < 10]
        insuff_ent = [s['avg_entropy'] for s in insufficient[tl] if s['avg_entropy'] < 10]

        print(f"  T={tl}: Sufficient={len(sufficient[tl])}, Insufficient={len(insufficient[tl])}")
        print(f"    PPL: Suff={np.mean(suff_ppl):.4f}, Insuff={np.mean(insuff_ppl):.4f}, diff={np.mean(insuff_ppl)-np.mean(suff_ppl):.4f}")
        print(f"    Entropy: Suff={np.mean(suff_ent):.4f}, Insuff={np.mean(insuff_ent):.4f}, diff={np.mean(insuff_ent)-np.mean(suff_ent):.4f}")

    # Plot
    output_name = 'entropy_distribution.png' if args.entropy_only else 'ppl_entropy_distribution.png'
    plot_distribution(
        sufficient, insufficient, token_levels,
        os.path.join(args.output_dir, output_name),
        entropy_only=args.entropy_only
    )


if __name__ == "__main__":
    main()
