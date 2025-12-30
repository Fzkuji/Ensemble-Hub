#!/usr/bin/env python3
"""
绘制 Entropy 趋势曲线

对于 sufficient 和 non-sufficient 样本，分别绘制不同 token level 下的
平均 entropy 随 token 位置变化的曲线。

Usage:
    python plot_entropy_trend.py --data-dir /path/to/data
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def load_ppl_data(data_dir: str, subset: str, token_level: int) -> List[Dict]:
    """加载 PPL 分析结果"""
    ppl_file = os.path.join(data_dir, "ppl_analysis", f"{subset}_tokens{token_level}_ppl.json")
    if not os.path.exists(ppl_file):
        print(f"File not found: {ppl_file}")
        return []

    with open(ppl_file, 'r') as f:
        return json.load(f)


def compute_average_entropy_curve(
    samples: List[Dict],
    max_length: int = None,
    normalize_length: bool = False,
    num_bins: int = 100,
) -> np.ndarray:
    """
    计算平均 entropy 曲线

    Args:
        samples: 样本列表，每个样本需要有 per_token_entropy 字段
        max_length: 最大长度（截断）
        normalize_length: 是否归一化长度到 [0, 1]
        num_bins: 归一化时的 bin 数量

    Returns:
        average_entropy: 平均 entropy 曲线
    """
    # 过滤出有 per_token_entropy 的样本
    valid_samples = [s for s in samples if 'per_token_entropy' in s and s['per_token_entropy']]

    if not valid_samples:
        return np.array([])

    if normalize_length:
        # 归一化到相同长度
        all_curves = []
        for s in valid_samples:
            entropy = np.array(s['per_token_entropy'])
            # 插值到 num_bins 个点
            x_old = np.linspace(0, 1, len(entropy))
            x_new = np.linspace(0, 1, num_bins)
            interpolated = np.interp(x_new, x_old, entropy)
            all_curves.append(interpolated)

        return np.mean(all_curves, axis=0)
    else:
        # 按实际位置计算平均
        # 找到最短长度
        lengths = [len(s['per_token_entropy']) for s in valid_samples]
        if max_length:
            target_length = min(max_length, max(lengths))
        else:
            target_length = max(lengths)

        # 对齐并计算平均
        all_curves = []
        for s in valid_samples:
            entropy = s['per_token_entropy'][:target_length]
            # 补齐到 target_length (用 nan)
            if len(entropy) < target_length:
                entropy = entropy + [np.nan] * (target_length - len(entropy))
            all_curves.append(entropy)

        all_curves = np.array(all_curves)
        # 对每个位置计算平均（忽略 nan）
        return np.nanmean(all_curves, axis=0)


def plot_entropy_trends(
    data_dir: str,
    subsets: List[str],
    token_levels: List[int],
    output_dir: str,
    normalize_length: bool = True,
    num_bins: int = 100,
    max_length: int = 2000,
):
    """绘制 entropy 趋势曲线"""
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有子集的数据
    all_sufficient = {tl: [] for tl in token_levels}
    all_non_sufficient = {tl: [] for tl in token_levels}

    for subset in subsets:
        for token_level in token_levels:
            data = load_ppl_data(data_dir, subset, token_level)
            if not data:
                continue

            for item in data:
                if 'per_token_entropy' not in item:
                    continue
                if item.get('is_correct', False):
                    all_sufficient[token_level].append(item)
                else:
                    all_non_sufficient[token_level].append(item)

    # 检查是否有数据
    has_per_token = any(
        any('per_token_entropy' in s for s in all_sufficient[tl] + all_non_sufficient[tl])
        for tl in token_levels
    )

    if not has_per_token:
        print("No per-token entropy data found. Run compute_ppl_entropy.py with --save-per-token first.")
        return

    # 绘制两个图：Sufficient 和 Non-sufficient
    colors = {100: '#1f77b4', 500: '#ff7f0e', 1000: '#2ca02c'}
    labels = {100: 'T=100', 500: 'T=500', 1000: 'T=1000'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Sufficient samples
    ax = axes[0]
    for token_level in token_levels:
        samples = all_sufficient[token_level]
        if not samples:
            continue

        curve = compute_average_entropy_curve(
            samples,
            normalize_length=normalize_length,
            num_bins=num_bins,
            max_length=max_length,
        )

        if len(curve) > 0:
            if normalize_length:
                x = np.linspace(0, 100, len(curve))
                ax.plot(x, curve, color=colors[token_level], label=f'{labels[token_level]} (n={len(samples)})', linewidth=2)
            else:
                ax.plot(curve, color=colors[token_level], label=f'{labels[token_level]} (n={len(samples)})', linewidth=2)

    ax.set_title('Sufficient (SLM Correct with Guidance)', fontsize=12)
    ax.set_xlabel('Position (%)' if normalize_length else 'Token Position', fontsize=11)
    ax.set_ylabel('Average Entropy', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Non-sufficient samples
    ax = axes[1]
    for token_level in token_levels:
        samples = all_non_sufficient[token_level]
        if not samples:
            continue

        curve = compute_average_entropy_curve(
            samples,
            normalize_length=normalize_length,
            num_bins=num_bins,
            max_length=max_length,
        )

        if len(curve) > 0:
            if normalize_length:
                x = np.linspace(0, 100, len(curve))
                ax.plot(x, curve, color=colors[token_level], label=f'{labels[token_level]} (n={len(samples)})', linewidth=2)
            else:
                ax.plot(curve, color=colors[token_level], label=f'{labels[token_level]} (n={len(samples)})', linewidth=2)

    ax.set_title('Non-sufficient (SLM Wrong with Guidance)', fontsize=12)
    ax.set_xlabel('Position (%)' if normalize_length else 'Token Position', fontsize=11)
    ax.set_ylabel('Average Entropy', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'entropy_trend_by_category.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    # 绘制对比图：同一 token level，比较 sufficient vs non-sufficient
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, token_level in enumerate(token_levels):
        ax = axes[idx]

        # Sufficient
        suff_samples = all_sufficient[token_level]
        if suff_samples:
            suff_curve = compute_average_entropy_curve(
                suff_samples,
                normalize_length=normalize_length,
                num_bins=num_bins,
                max_length=max_length,
            )
            if len(suff_curve) > 0:
                if normalize_length:
                    x = np.linspace(0, 100, len(suff_curve))
                    ax.plot(x, suff_curve, color='green', label=f'Sufficient (n={len(suff_samples)})', linewidth=2)
                else:
                    ax.plot(suff_curve, color='green', label=f'Sufficient (n={len(suff_samples)})', linewidth=2)

        # Non-sufficient
        non_suff_samples = all_non_sufficient[token_level]
        if non_suff_samples:
            non_suff_curve = compute_average_entropy_curve(
                non_suff_samples,
                normalize_length=normalize_length,
                num_bins=num_bins,
                max_length=max_length,
            )
            if len(non_suff_curve) > 0:
                if normalize_length:
                    x = np.linspace(0, 100, len(non_suff_curve))
                    ax.plot(x, non_suff_curve, color='red', label=f'Non-sufficient (n={len(non_suff_samples)})', linewidth=2)
                else:
                    ax.plot(non_suff_curve, color='red', label=f'Non-sufficient (n={len(non_suff_samples)})', linewidth=2)

        ax.set_title(f'Token Level = {token_level}', fontsize=12)
        ax.set_xlabel('Position (%)' if normalize_length else 'Token Position', fontsize=11)
        ax.set_ylabel('Average Entropy', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'entropy_trend_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("Statistics Summary")
    print("=" * 60)
    for token_level in token_levels:
        print(f"\nToken Level {token_level}:")
        suff = all_sufficient[token_level]
        non_suff = all_non_sufficient[token_level]
        print(f"  Sufficient samples: {len(suff)}")
        print(f"  Non-sufficient samples: {len(non_suff)}")

        if suff:
            avg_entropies = [s.get('avg_entropy', 0) for s in suff if s.get('avg_entropy') and s['avg_entropy'] < float('inf')]
            if avg_entropies:
                print(f"  Sufficient avg entropy: {np.mean(avg_entropies):.4f} +/- {np.std(avg_entropies):.4f}")

        if non_suff:
            avg_entropies = [s.get('avg_entropy', 0) for s in non_suff if s.get('avg_entropy') and s['avg_entropy'] < float('inf')]
            if avg_entropies:
                print(f"  Non-sufficient avg entropy: {np.mean(avg_entropies):.4f} +/- {np.std(avg_entropies):.4f}")


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
    parser.add_argument("--no-normalize", action="store_true",
                        help="Don't normalize length (use absolute positions)")
    parser.add_argument("--num-bins", type=int, default=100,
                        help="Number of bins for normalization")
    parser.add_argument("--max-length", type=int, default=2000,
                        help="Maximum token length (for non-normalized mode)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "ppl_analysis")

    token_levels = [int(x) for x in args.token_levels.split(',')]

    # 确定子集
    if args.subset:
        subsets = [args.subset]
    else:
        subsets = []
        ppl_dir = os.path.join(args.data_dir, "ppl_analysis")
        if os.path.exists(ppl_dir):
            for name in os.listdir(ppl_dir):
                if name.endswith('_ppl.json'):
                    # 提取 subset 名称
                    parts = name.replace('_ppl.json', '').rsplit('_tokens', 1)
                    if len(parts) == 2:
                        subset = parts[0]
                        if subset not in subsets:
                            subsets.append(subset)
        subsets = sorted(subsets)

    print(f"Processing subsets: {subsets}")
    print(f"Token levels: {token_levels}")

    plot_entropy_trends(
        args.data_dir,
        subsets,
        token_levels,
        args.output_dir,
        normalize_length=not args.no_normalize,
        num_bins=args.num_bins,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
