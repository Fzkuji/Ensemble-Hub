#!/usr/bin/env python3
"""
可视化分析：Sufficient vs Non-sufficient 情况下 SLM 对 guidance 的分析

Sufficient: SLM 在 guidance 下答对
Non-sufficient: SLM 在 guidance 下仍然答错

分析内容：
1. 长度分布差异 (mentor_length, intern_length)
2. 可选：PPL 和 Entropy（需要加载模型计算）
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
from typing import Dict, List, Optional


def load_data(data_dir: str, subset: str, split: str = "test", token_level: int = 100):
    """加载指定 token level 的数据"""
    data_file = os.path.join(data_dir, subset, split, f"tokens{token_level}.json")
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return None

    with open(data_file, 'r') as f:
        data = json.load(f)

    # 尝试加载 PPL 分析结果
    ppl_file = os.path.join(data_dir, "ppl_analysis", f"{subset}_tokens{token_level}_ppl.json")
    if os.path.exists(ppl_file):
        print(f"Loading PPL data from: {ppl_file}")
        with open(ppl_file, 'r') as f:
            ppl_data = json.load(f)
        # 合并 PPL 数据到原始数据
        if len(ppl_data) == len(data):
            for i, item in enumerate(data):
                item['ppl'] = ppl_data[i].get('ppl')
                item['avg_entropy'] = ppl_data[i].get('avg_entropy')
                item['max_entropy'] = ppl_data[i].get('max_entropy')
        else:
            print(f"Warning: PPL data length mismatch ({len(ppl_data)} vs {len(data)})")

    return data


def compute_stats(values, name=""):
    """计算统计量"""
    if not values:
        return {}
    arr = np.array(values)
    return {
        'name': name,
        'count': len(arr),
        'mean': np.mean(arr),
        'std': np.std(arr),
        'median': np.median(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
    }


def analyze_subset(data_dir: str, subset: str, split: str = "test", token_levels: list = None):
    """分析单个子集的数据"""
    if token_levels is None:
        token_levels = [100, 500, 1000]

    results = {}

    for token_level in token_levels:
        data = load_data(data_dir, subset, split, token_level)
        if data is None:
            continue

        # 分离 sufficient 和 non-sufficient
        sufficient = []  # is_correct = True
        non_sufficient = []  # is_correct = False

        for item in data:
            is_correct = item.get('is_correct', False)

            # 收集可用的指标
            metrics = {
                'intern_length': item.get('intern_length', 0),
                'mentor_length': item.get('mentor_length', 0),
                'mentor_tokens': item.get('mentor_tokens', token_level),
                'level': item.get('level', ''),
            }

            # 计算长度比例
            if metrics['mentor_length'] > 0:
                metrics['length_ratio'] = metrics['intern_length'] / metrics['mentor_length']
            else:
                metrics['length_ratio'] = 0

            # PPL/Entropy 如果有的话
            if 'ppl' in item and item['ppl'] is not None and item['ppl'] != float('inf'):
                metrics['ppl'] = item['ppl']
            if 'avg_entropy' in item and item['avg_entropy'] is not None and item['avg_entropy'] != float('inf'):
                metrics['avg_entropy'] = item['avg_entropy']
            if 'max_entropy' in item and item['max_entropy'] is not None and item['max_entropy'] != float('inf'):
                metrics['max_entropy'] = item['max_entropy']
            if 'entropy' in item and item['entropy'] is not None:
                metrics['entropy'] = item['entropy']

            # 分类
            if is_correct:
                sufficient.append(metrics)
            else:
                non_sufficient.append(metrics)

        results[token_level] = {
            'sufficient': sufficient,
            'non_sufficient': non_sufficient,
            'total': len(data),
            'n_sufficient': len(sufficient),
            'n_non_sufficient': len(non_sufficient),
        }

    return results


def plot_comparison(results: dict, metric: str, subset: str, output_dir: str, title_suffix: str = ""):
    """绘制 sufficient vs non-sufficient 的对比图"""
    token_levels = sorted(results.keys())

    fig, axes = plt.subplots(1, len(token_levels), figsize=(5 * len(token_levels), 5))
    if len(token_levels) == 1:
        axes = [axes]

    for ax, token_level in zip(axes, token_levels):
        data = results[token_level]

        suff_values = [item.get(metric, None) for item in data['sufficient'] if item.get(metric) is not None]
        non_suff_values = [item.get(metric, None) for item in data['non_sufficient'] if item.get(metric) is not None]

        if not suff_values and not non_suff_values:
            ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'T{token_level}')
            continue

        # Box plot
        box_data = []
        labels = []
        colors_used = []
        if suff_values:
            box_data.append(suff_values)
            labels.append(f'Sufficient\n(n={len(suff_values)})')
            colors_used.append('#90EE90')  # 绿色
        if non_suff_values:
            box_data.append(non_suff_values)
            labels.append(f'Non-sufficient\n(n={len(non_suff_values)})')
            colors_used.append('#FFB6C1')  # 粉色

        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_used):
            patch.set_facecolor(color)

        ax.set_title(f'T{token_level}')
        ax.set_ylabel(metric.replace('_', ' ').title())

        # 添加统计检验结果
        if len(suff_values) > 1 and len(non_suff_values) > 1:
            try:
                stat, p_value = stats.mannwhitneyu(suff_values, non_suff_values, alternative='two-sided')
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                ax.text(0.5, 0.02, f'p={p_value:.4f} ({significance})', ha='center', transform=ax.transAxes, fontsize=9)
            except:
                pass

        # 添加均值
        if suff_values:
            ax.axhline(np.mean(suff_values), color='green', linestyle='--', alpha=0.5)
        if non_suff_values:
            ax.axhline(np.mean(non_suff_values), color='red', linestyle='--', alpha=0.5)

    plt.suptitle(f'{subset}: {metric.replace("_", " ").title()} - Sufficient vs Non-sufficient{title_suffix}', fontsize=12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{subset}_{metric}_boxplot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_histogram(results: dict, metric: str, subset: str, output_dir: str):
    """绘制直方图对比"""
    token_levels = sorted(results.keys())

    fig, axes = plt.subplots(len(token_levels), 1, figsize=(10, 4 * len(token_levels)))
    if len(token_levels) == 1:
        axes = [axes]

    for ax, token_level in zip(axes, token_levels):
        data = results[token_level]

        suff_values = [item.get(metric, None) for item in data['sufficient'] if item.get(metric) is not None]
        non_suff_values = [item.get(metric, None) for item in data['non_sufficient'] if item.get(metric) is not None]

        if not suff_values and not non_suff_values:
            ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'T{token_level}')
            continue

        # 确定 bins
        all_values = suff_values + non_suff_values
        bins = np.linspace(min(all_values), np.percentile(all_values, 95), 50)  # 截断 95%

        if suff_values:
            ax.hist(suff_values, bins=bins, alpha=0.6, label=f'Sufficient (n={len(suff_values)}, μ={np.mean(suff_values):.1f})', color='green')
        if non_suff_values:
            ax.hist(non_suff_values, bins=bins, alpha=0.6, label=f'Non-sufficient (n={len(non_suff_values)}, μ={np.mean(non_suff_values):.1f})', color='red')

        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Count')
        ax.set_title(f'T{token_level}: {metric.replace("_", " ").title()} Distribution')
        ax.legend()

    plt.suptitle(f'{subset}: {metric.replace("_", " ").title()} Histogram', fontsize=12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{subset}_{metric}_histogram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_subsets_summary(all_results: Dict[str, dict], metric: str, output_dir: str, token_level: int = 100):
    """绘制所有子集的汇总对比图"""
    subsets = sorted(all_results.keys())

    suff_means = []
    non_suff_means = []
    suff_stds = []
    non_suff_stds = []
    valid_subsets = []

    for subset in subsets:
        if token_level not in all_results[subset]:
            continue
        data = all_results[subset][token_level]

        suff_values = [item.get(metric) for item in data['sufficient'] if item.get(metric) is not None]
        non_suff_values = [item.get(metric) for item in data['non_sufficient'] if item.get(metric) is not None]

        if suff_values and non_suff_values:
            suff_means.append(np.mean(suff_values))
            suff_stds.append(np.std(suff_values))
            non_suff_means.append(np.mean(non_suff_values))
            non_suff_stds.append(np.std(non_suff_values))
            valid_subsets.append(subset)

    if not valid_subsets:
        return

    x = np.arange(len(valid_subsets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, suff_means, width, yerr=suff_stds, label='Sufficient', color='green', alpha=0.7, capsize=3)
    bars2 = ax.bar(x + width/2, non_suff_means, width, yerr=non_suff_stds, label='Non-sufficient', color='red', alpha=0.7, capsize=3)

    ax.set_xlabel('Subset')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'T{token_level}: {metric.replace("_", " ").title()} by Subset')
    ax.set_xticks(x)
    ax.set_xticklabels([s[:15] for s in valid_subsets], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'all_subsets_{metric}_T{token_level}_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(results: dict, subset: str):
    """打印统计摘要"""
    print(f"\n{'='*70}")
    print(f"Subset: {subset}")
    print(f"{'='*70}")

    for token_level in sorted(results.keys()):
        data = results[token_level]
        print(f"\n--- Token Level: {token_level} ---")
        print(f"Total: {data['total']}, Sufficient: {data['n_sufficient']}, Non-sufficient: {data['n_non_sufficient']}")
        if data['total'] > 0:
            print(f"Accuracy: {data['n_sufficient']/data['total']*100:.2f}%")

        # 统计各指标
        for metric in ['intern_length', 'mentor_length', 'length_ratio', 'ppl', 'avg_entropy', 'max_entropy']:
            suff_values = [item.get(metric) for item in data['sufficient'] if item.get(metric) is not None]
            non_suff_values = [item.get(metric) for item in data['non_sufficient'] if item.get(metric) is not None]

            if suff_values or non_suff_values:
                print(f"\n  {metric.upper()}:")
                if suff_values:
                    s = compute_stats(suff_values, 'Sufficient')
                    print(f"    Sufficient:     mean={s['mean']:.2f}, std={s['std']:.2f}, median={s['median']:.2f}")
                if non_suff_values:
                    s = compute_stats(non_suff_values, 'Non-sufficient')
                    print(f"    Non-sufficient: mean={s['mean']:.2f}, std={s['std']:.2f}, median={s['median']:.2f}")

                # 统计检验
                if len(suff_values) > 1 and len(non_suff_values) > 1:
                    try:
                        stat, p_value = stats.mannwhitneyu(suff_values, non_suff_values, alternative='two-sided')
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        diff = np.mean(suff_values) - np.mean(non_suff_values)
                        print(f"    Difference: {diff:+.2f}, Mann-Whitney U: p={p_value:.6f} ({significance})")
                    except Exception as e:
                        print(f"    Statistical test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize guidance analysis: Sufficient vs Non-sufficient")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: data-dir/visualizations)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset to analyze (default: all)")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split: train or test")
    parser.add_argument("--token-levels", type=str, default="100,500,1000",
                        help="Comma-separated token levels to analyze")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting, only print statistics")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "visualizations")

    token_levels = [int(x) for x in args.token_levels.split(',')]

    # 确定要分析的子集
    if args.subset:
        subsets = [args.subset]
    else:
        # 自动检测子集
        subsets = []
        for name in os.listdir(args.data_dir):
            subset_dir = os.path.join(args.data_dir, name, args.split)
            if os.path.isdir(subset_dir):
                token_file = os.path.join(subset_dir, f"tokens{token_levels[0]}.json")
                if os.path.exists(token_file):
                    subsets.append(name)
        subsets = sorted(subsets)

    print(f"Analyzing subsets: {subsets}")
    print(f"Token levels: {token_levels}")
    print(f"Split: {args.split}")
    print(f"Output dir: {args.output_dir}")

    all_results = {}

    for subset in subsets:
        results = analyze_subset(args.data_dir, subset, args.split, token_levels)
        if not results:
            print(f"No data for subset: {subset}")
            continue

        all_results[subset] = results

        # 打印统计摘要
        print_summary(results, subset)

        # 绘图
        if not args.no_plot:
            for metric in ['intern_length', 'mentor_length', 'length_ratio', 'ppl', 'avg_entropy']:
                # 检查是否有该指标的数据
                has_data = False
                for token_level in token_levels:
                    if token_level not in results:
                        continue
                    data = results[token_level]
                    if any(item.get(metric) is not None for item in data['sufficient'] + data['non_sufficient']):
                        has_data = True
                        break
                if has_data:
                    plot_comparison(results, metric, subset, args.output_dir)
                    plot_histogram(results, metric, subset, args.output_dir)

    # 汇总所有子集
    if len(all_results) > 1 and not args.no_plot:
        for metric in ['intern_length', 'mentor_length', 'length_ratio', 'ppl', 'avg_entropy']:
            for token_level in token_levels:
                plot_all_subsets_summary(all_results, metric, args.output_dir, token_level)

    # 打印总体汇总
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")

        for token_level in token_levels:
            total_suff = 0
            total_non_suff = 0
            all_suff_intern_len = []
            all_non_suff_intern_len = []

            for subset, results in all_results.items():
                if token_level in results:
                    total_suff += results[token_level]['n_sufficient']
                    total_non_suff += results[token_level]['n_non_sufficient']
                    all_suff_intern_len.extend([item['intern_length'] for item in results[token_level]['sufficient']])
                    all_non_suff_intern_len.extend([item['intern_length'] for item in results[token_level]['non_sufficient']])

            total = total_suff + total_non_suff
            if total > 0:
                print(f"\nT{token_level}:")
                print(f"  Total={total}, Sufficient={total_suff} ({total_suff/total*100:.2f}%), Non-sufficient={total_non_suff} ({total_non_suff/total*100:.2f}%)")
                if all_suff_intern_len and all_non_suff_intern_len:
                    print(f"  Intern Length: Sufficient={np.mean(all_suff_intern_len):.1f}, Non-sufficient={np.mean(all_non_suff_intern_len):.1f}")
                    stat, p_value = stats.mannwhitneyu(all_suff_intern_len, all_non_suff_intern_len, alternative='two-sided')
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  Mann-Whitney U: p={p_value:.6f} ({significance})")

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
