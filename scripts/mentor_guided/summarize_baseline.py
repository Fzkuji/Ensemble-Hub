#!/usr/bin/env python3
"""
汇总 baseline 结果（不需要训练分类器）
只统计各个 token level 的准确率和生成长度
"""

import argparse
import json
import os
import numpy as np
from tabulate import tabulate

TOKEN_LEVELS = [-1, 0, 100, 500, 1000]


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


def compute_stats(data_dir: str, subset: str, split: str = "test"):
    """计算单个子集的统计信息"""
    subset_dir = os.path.join(data_dir, subset, split)

    if not os.path.exists(subset_dir):
        return None

    results = {}

    for tokens in TOKEN_LEVELS:
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
            # Mentor length
            if 'mentor_length' in item:
                mentor_lengths.append(item['mentor_length'])
            elif 'mentor_response' in item and item['mentor_response']:
                mentor_lengths.append(len(item['mentor_response']) // 4)

            # Intern length
            if 'intern_length' in item:
                intern_lengths.append(item['intern_length'])
            elif 'num_tokens' in item:
                intern_lengths.append(item['num_tokens'])
            elif 'response' in item:
                intern_lengths.append(len(item['response']) // 4)

        results[tokens] = {
            'n_samples': len(data),
            'accuracy': accuracy,
            'n_correct': correct,
            'mentor_length': np.mean(mentor_lengths) if mentor_lengths else 0,
            'intern_length': np.mean(intern_lengths) if intern_lengths else 0,
        }

    # Oracle: 至少一个 token level 正确
    if 0 in results:
        n_samples = results[0]['n_samples']
        oracle_correct = 0

        # 加载所有数据
        all_data = {}
        for tokens in TOKEN_LEVELS:
            filepath = os.path.join(subset_dir, f"tokens{tokens}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    all_data[tokens] = json.load(f)

        for i in range(n_samples):
            for tokens in TOKEN_LEVELS:
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

    if not subsets:
        print(f"No subsets found in {data_dir}")
        return

    print(f"\n**Baseline Results Summary** ({split} split)")
    print(f"Data: {data_dir}\n")

    # 准备表格数据
    table_data = []

    # 统计变量
    total_stats = {tokens: {'n': 0, 'correct': 0} for tokens in TOKEN_LEVELS}
    total_stats['oracle'] = {'n': 0, 'correct': 0}
    total_lengths = {tokens: {'mentor': [], 'intern': []} for tokens in TOKEN_LEVELS}

    for subset in subsets:
        stats = compute_stats(data_dir, subset, split)

        if not stats:
            table_data.append([subset] + ["-"] * 15)
            continue

        n = stats[0]['n_samples'] if 0 in stats else 0

        row = [subset, n]

        # T=-1 (Mentor only)
        if -1 in stats:
            s = stats[-1]
            row.extend([f"{s['accuracy']:.4f}", f"{s['mentor_length']:.1f}"])
            total_stats[-1]['n'] += n
            total_stats[-1]['correct'] += s['n_correct']
            if s['mentor_length'] > 0:
                total_lengths[-1]['mentor'].append(s['mentor_length'])
        else:
            row.extend(["-", "-"])

        # T=0 (Intern only)
        if 0 in stats:
            s = stats[0]
            row.extend([f"{s['accuracy']:.4f}", f"{s['intern_length']:.1f}"])
            total_stats[0]['n'] += n
            total_stats[0]['correct'] += s['n_correct']
            if s['intern_length'] > 0:
                total_lengths[0]['intern'].append(s['intern_length'])
        else:
            row.extend(["-", "-"])

        # T=100, 500, 1000
        for tokens in [100, 500, 1000]:
            if tokens in stats:
                s = stats[tokens]
                row.extend([f"{s['accuracy']:.4f}",
                           f"{s['mentor_length']:.1f}" if s['mentor_length'] > 0 else "-",
                           f"{s['intern_length']:.1f}" if s['intern_length'] > 0 else "-"])
                total_stats[tokens]['n'] += n
                total_stats[tokens]['correct'] += s['n_correct']
                if s['mentor_length'] > 0:
                    total_lengths[tokens]['mentor'].append(s['mentor_length'])
                if s['intern_length'] > 0:
                    total_lengths[tokens]['intern'].append(s['intern_length'])
            else:
                row.extend(["-", "-", "-"])

        # Oracle
        if 'oracle' in stats:
            s = stats['oracle']
            row.append(f"{s['accuracy']:.4f}")
            total_stats['oracle']['n'] += n
            total_stats['oracle']['correct'] += s['n_correct']
        else:
            row.append("-")

        table_data.append(row)

    # TOTAL 行
    total_row = ["TOTAL"]
    total_n = total_stats[0]['n'] if 0 in total_stats and total_stats[0]['n'] > 0 else sum(s['n'] for s in total_stats.values() if isinstance(s, dict) and 'n' in s) // len(TOKEN_LEVELS)
    total_row.append(total_n)

    # T=-1
    if total_stats[-1]['n'] > 0:
        acc = total_stats[-1]['correct'] / total_stats[-1]['n']
        m_len = np.mean(total_lengths[-1]['mentor']) if total_lengths[-1]['mentor'] else 0
        total_row.extend([f"{acc:.4f}", f"{m_len:.1f}" if m_len > 0 else "-"])
    else:
        total_row.extend(["-", "-"])

    # T=0
    if total_stats[0]['n'] > 0:
        acc = total_stats[0]['correct'] / total_stats[0]['n']
        i_len = np.mean(total_lengths[0]['intern']) if total_lengths[0]['intern'] else 0
        total_row.extend([f"{acc:.4f}", f"{i_len:.1f}" if i_len > 0 else "-"])
    else:
        total_row.extend(["-", "-"])

    # T=100, 500, 1000
    for tokens in [100, 500, 1000]:
        if total_stats[tokens]['n'] > 0:
            acc = total_stats[tokens]['correct'] / total_stats[tokens]['n']
            m_len = np.mean(total_lengths[tokens]['mentor']) if total_lengths[tokens]['mentor'] else 0
            i_len = np.mean(total_lengths[tokens]['intern']) if total_lengths[tokens]['intern'] else 0
            total_row.extend([f"{acc:.4f}",
                             f"{m_len:.1f}" if m_len > 0 else "-",
                             f"{i_len:.1f}" if i_len > 0 else "-"])
        else:
            total_row.extend(["-", "-", "-"])

    # Oracle
    if total_stats['oracle']['n'] > 0:
        acc = total_stats['oracle']['correct'] / total_stats['oracle']['n']
        total_row.append(f"{acc:.4f}")
    else:
        total_row.append("-")

    table_data.append(total_row)

    # 表头
    headers = [
        "Subset", "N",
        "T-1(M)", "M_len",
        "T0(I)", "I_len",
        "T100", "T100_m", "T100_i",
        "T500", "T500_m", "T500_i",
        "T1000", "T1000_m", "T1000_i",
        "Oracle"
    ]

    print(tabulate(table_data, headers=headers, tablefmt="pipe", numalign="right", stralign="left"))
    print("\nLegend:")
    print("  T-1 = Mentor only, T0 = Intern only")
    print("  T100/500/1000 = Mentor generates N tokens, then Intern continues")
    print("  Oracle = Best possible (if we knew which level to use)")
    print("  M_len/I_len = Average generation length (tokens)")


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
