#!/usr/bin/env python3
"""
汇总 baseline 结果（不需要训练分类器）
只统计各个 token level 的准确率和生成长度
"""

import argparse
import json
import os
import re
import numpy as np
from tabulate import tabulate


def detect_token_levels(data_dir: str, split: str = "test"):
    """Auto-detect all available token levels from data directory."""
    token_levels = set()

    if not os.path.exists(data_dir):
        return []

    for name in os.listdir(data_dir):
        subset_dir = os.path.join(data_dir, name, split)
        if os.path.isdir(subset_dir):
            # Find all tokens*.json files
            for filename in os.listdir(subset_dir):
                match = re.match(r'tokens(-?\d+)\.json', filename)
                if match:
                    token_levels.add(int(match.group(1)))

    return sorted(token_levels)


def detect_subsets(data_dir: str, split: str = "test"):
    """Auto-detect subsets from data directory."""
    subsets = []
    if not os.path.exists(data_dir):
        return []

    for name in os.listdir(data_dir):
        subset_dir = os.path.join(data_dir, name, split)
        if os.path.isdir(subset_dir):
            token_file = os.path.join(subset_dir, "tokens0.json")
            if os.path.exists(token_file):
                subsets.append(name)

    return sorted(subsets)


def compute_stats(data_dir: str, subset: str, split: str = "test", token_levels=None):
    """计算单个子集的统计信息"""
    subset_dir = os.path.join(data_dir, subset, split)

    if not os.path.exists(subset_dir):
        return None

    results = {}

    for tokens in token_levels:
        filepath = os.path.join(subset_dir, f"tokens{tokens}.json")
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        if not data:
            continue

        # 准确率
        correct = sum(1 for item in data if item.get('is_correct', False))
        accuracy = correct / len(data)

        # 长度统计
        mentor_lengths = []
        intern_lengths = []

        for item in data:
            # Mentor length - use mentor_tokens field
            if 'mentor_tokens' in item and item['mentor_tokens'] > 0:
                mentor_lengths.append(item['mentor_tokens'])
            elif 'mentor_length' in item and item['mentor_length'] > 0:
                mentor_lengths.append(item['mentor_length'])
            elif 'mentor_response' in item and item['mentor_response']:
                mentor_lengths.append(len(item['mentor_response']) // 4)

            # Intern length
            # Special case: tokens=-1 (mentor-only) stores length in intern_length field
            if tokens == -1 and 'intern_length' in item and item['intern_length'] > 0:
                # For mentor-only mode, the actual length is in intern_length field
                mentor_lengths.append(item['intern_length'])
            elif 'intern_length' in item:
                intern_lengths.append(item['intern_length'])
            elif 'num_tokens' in item:
                intern_lengths.append(item['num_tokens'])
            elif 'response' in item:
                intern_lengths.append(len(item['response']) // 4)

        mentor_len_mean = np.mean(mentor_lengths) if mentor_lengths else 0
        intern_len_mean = np.mean(intern_lengths) if intern_lengths else 0
        total_len_mean = mentor_len_mean + intern_len_mean

        results[tokens] = {
            'n_samples': len(data),
            'accuracy': accuracy,
            'n_correct': correct,
            'mentor_length': mentor_len_mean,
            'intern_length': intern_len_mean,
            'total_length': total_len_mean,
        }

    # Oracle: 至少一个 token level 正确
    if 0 in results:
        n_samples = results[0]['n_samples']
        oracle_correct = 0

        # 加载所有数据
        all_data = {}
        for tokens in token_levels:
            filepath = os.path.join(subset_dir, f"tokens{tokens}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    all_data[tokens] = json.load(f)

        for i in range(n_samples):
            for tokens in token_levels:
                if tokens in all_data and all_data[tokens][i].get('is_correct', False):
                    oracle_correct += 1
                    break

        results['oracle'] = {
            'accuracy': oracle_correct / n_samples,
            'n_correct': oracle_correct,
        }

    return results


def summarize(data_dir: str, split: str = "test"):
    """汇总所有子集的结果"""
    subsets = detect_subsets(data_dir, split)
    token_levels = detect_token_levels(data_dir, split)

    if not subsets:
        print(f"No subsets found in {data_dir}")
        return

    if not token_levels:
        print(f"No token level data found in {data_dir}")
        return

    print(f"\n**Baseline Results Summary** ({split} split)")
    print(f"Data: {data_dir}")
    print(f"Token levels: {token_levels}\n")

    # 准备表格数据
    table_data = []

    # 统计变量
    total_stats = {tokens: {'n': 0, 'correct': 0} for tokens in token_levels}
    total_stats['oracle'] = {'n': 0, 'correct': 0}
    total_lengths = {tokens: {'mentor': [], 'intern': [], 'total': []} for tokens in token_levels}

    for subset in subsets:
        stats = compute_stats(data_dir, subset, split, token_levels)

        if not stats:
            # Calculate number of columns needed
            num_cols = 2 + sum(4 if t > 0 else 3 for t in token_levels) + 1
            table_data.append([subset] + ["-"] * num_cols)
            continue

        # Get sample count
        n = next((stats[t]['n_samples'] for t in token_levels if t in stats), 0)
        row = [subset, n]

        # Add stats for each token level
        for tokens in token_levels:
            if tokens in stats:
                s = stats[tokens]
                if tokens == -1:
                    # Mentor-only: show accuracy, mentor length, total
                    row.extend([
                        f"{s['accuracy']:.4f}",
                        f"{s['mentor_length']:.1f}",
                        f"{s['total_length']:.1f}"
                    ])
                elif tokens == 0:
                    # Intern-only: show accuracy, intern length, total
                    row.extend([
                        f"{s['accuracy']:.4f}",
                        f"{s['intern_length']:.1f}",
                        f"{s['total_length']:.1f}"
                    ])
                else:
                    # Collaboration: show accuracy, mentor length, intern length, total
                    row.extend([
                        f"{s['accuracy']:.4f}",
                        f"{s['mentor_length']:.1f}" if s['mentor_length'] > 0 else "-",
                        f"{s['intern_length']:.1f}" if s['intern_length'] > 0 else "-",
                        f"{s['total_length']:.1f}" if s['total_length'] > 0 else "-"
                    ])

                # Update totals
                total_stats[tokens]['n'] += n
                total_stats[tokens]['correct'] += s['n_correct']
                if s['mentor_length'] > 0:
                    total_lengths[tokens]['mentor'].append(s['mentor_length'])
                if s['intern_length'] > 0:
                    total_lengths[tokens]['intern'].append(s['intern_length'])
                if s['total_length'] > 0:
                    total_lengths[tokens]['total'].append(s['total_length'])
            else:
                # Missing data
                if tokens == -1 or tokens == 0:
                    row.extend(["-", "-", "-"])
                else:
                    row.extend(["-", "-", "-", "-"])

        # Oracle
        if 'oracle' in stats:
            s = stats['oracle']
            row.append(f"{s['accuracy']:.4f}")
            total_stats['oracle']['n'] += n
            total_stats['oracle']['correct'] += s['n_correct']
        else:
            row.append("-")

        table_data.append(row)

    # TOTAL row
    total_row = ["TOTAL"]
    total_n = next((total_stats[t]['n'] for t in token_levels if total_stats[t]['n'] > 0), 0)
    total_row.append(total_n)

    for tokens in token_levels:
        if total_stats[tokens]['n'] > 0:
            acc = total_stats[tokens]['correct'] / total_stats[tokens]['n']
            m_len = np.mean(total_lengths[tokens]['mentor']) if total_lengths[tokens]['mentor'] else 0
            i_len = np.mean(total_lengths[tokens]['intern']) if total_lengths[tokens]['intern'] else 0
            t_len = np.mean(total_lengths[tokens]['total']) if total_lengths[tokens]['total'] else 0

            if tokens == -1:
                total_row.extend([
                    f"{acc:.4f}",
                    f"{m_len:.1f}" if m_len > 0 else "-",
                    f"{t_len:.1f}" if t_len > 0 else "-"
                ])
            elif tokens == 0:
                total_row.extend([
                    f"{acc:.4f}",
                    f"{i_len:.1f}" if i_len > 0 else "-",
                    f"{t_len:.1f}" if t_len > 0 else "-"
                ])
            else:
                total_row.extend([
                    f"{acc:.4f}",
                    f"{m_len:.1f}" if m_len > 0 else "-",
                    f"{i_len:.1f}" if i_len > 0 else "-",
                    f"{t_len:.1f}" if t_len > 0 else "-"
                ])
        else:
            if tokens == -1 or tokens == 0:
                total_row.extend(["-", "-", "-"])
            else:
                total_row.extend(["-", "-", "-", "-"])

    # Oracle
    if total_stats['oracle']['n'] > 0:
        acc = total_stats['oracle']['correct'] / total_stats['oracle']['n']
        total_row.append(f"{acc:.4f}")
    else:
        total_row.append("-")

    table_data.append(total_row)

    # Generate headers dynamically
    headers = ["Subset", "N"]

    for tokens in token_levels:
        if tokens == -1:
            headers.extend(["T-1(M)", "M_len", "M_total"])
        elif tokens == 0:
            headers.extend(["T0(I)", "I_len", "I_total"])
        else:
            headers.extend([f"T{tokens}", f"T{tokens}_m", f"T{tokens}_i", f"T{tokens}_total"])

    headers.append("Oracle")

    print(tabulate(table_data, headers=headers, tablefmt="pipe", numalign="right", stralign="left"))
    print("\nLegend:")
    print("  T-1 = Mentor only, T0 = Intern only")
    print("  T<N> = Mentor generates N tokens, then Intern continues")
    print("  Oracle = Best possible (if we knew which level to use)")
    print("  M_len/I_len = Average generation length (tokens)")
    print("  *_total = Total length (mentor + intern, shows overall computational cost)")


def main():
    parser = argparse.ArgumentParser(description="Summarize baseline results")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to analyze (train/test)")

    args = parser.parse_args()
    summarize(args.data_dir, args.split)


if __name__ == "__main__":
    main()
