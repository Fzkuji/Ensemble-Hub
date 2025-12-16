#!/usr/bin/env python3
"""
汇总所有子集的评估结果

Usage:
    python summarize_results.py
    python summarize_results.py --data-dir /path/to/data
"""

import argparse
import json
import os
import numpy as np

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

TOKEN_LEVELS = [0, 100, 500, 1000]

# 列宽定义
W_SUBSET = 25
W_N = 6
W_ACC = 8
W_LEN = 8
W_TOKEN_GROUP = W_ACC + W_LEN + W_LEN  # acc + m_len + i_len = 24
W_ORACLE = 8
W_CASCADE = 8
W_GAP = 8
LINE_WIDTH = W_SUBSET + W_N + W_TOKEN_GROUP * 4 + W_ORACLE + W_CASCADE + W_GAP + 10  # 约180


def compute_length_stats(data_dir: str, subset: str, split: str = "test"):
    """计算每个 token level 的平均生成长度（mentor 和 intern 分别统计）"""
    length_stats = {}

    for tokens in TOKEN_LEVELS:
        data_file = os.path.join(data_dir, subset, split, f"tokens{tokens}.json")
        if not os.path.exists(data_file):
            length_stats[tokens] = None
            continue

        with open(data_file, 'r') as f:
            data = json.load(f)

        # 统计 mentor 和 intern 的生成长度
        mentor_lengths = []
        intern_lengths = []

        for item in data:
            # Mentor length
            if 'mentor_length' in item:
                mentor_lengths.append(item['mentor_length'])
            elif 'mentor_response' in item and item['mentor_response']:
                # 估算：按字符数/4估算token数
                mentor_lengths.append(len(item['mentor_response']) // 4)

            # Intern length
            if 'intern_length' in item:
                intern_lengths.append(item['intern_length'])
            elif 'num_tokens' in item:
                intern_lengths.append(item['num_tokens'])
            elif 'response' in item:
                intern_lengths.append(len(item['response']) // 4)

        stats = {}
        if mentor_lengths:
            stats['mentor'] = {
                'mean': np.mean(mentor_lengths),
                'std': np.std(mentor_lengths),
            }
        if intern_lengths:
            stats['intern'] = {
                'mean': np.mean(intern_lengths),
                'std': np.std(intern_lengths),
            }

        length_stats[tokens] = stats if stats else None

    return length_stats


def summarize(data_dir: str, show_length: bool = True):
    """汇总所有子集的结果"""
    # 计算实际行宽
    if show_length:
        line_width = W_SUBSET + W_N + W_TOKEN_GROUP * 4 + W_ORACLE + W_CASCADE + W_GAP + 8
    else:
        line_width = 110

    print("=" * line_width)
    print(f"Results Summary: {data_dir}")
    print("=" * line_width)

    if show_length:
        # 表头第一行：Subset, N, T0, T100, T500, T1000, Oracle, Cascade, Gap
        print(f"{'Subset':<{W_SUBSET}} {'N':<{W_N}} "
              f"{'T0':<{W_TOKEN_GROUP}} {'T100':<{W_TOKEN_GROUP}} "
              f"{'T500':<{W_TOKEN_GROUP}} {'T1000':<{W_TOKEN_GROUP}} "
              f"{'Oracle':<{W_ORACLE}} {'Cascade':<{W_CASCADE}} {'Gap':<{W_GAP}}")
        # 表头第二行：acc, m_len, i_len
        print(f"{'':<{W_SUBSET}} {'':<{W_N}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}}")
    else:
        print(f"{'Subset':<{W_SUBSET}} {'N':<{W_N}} {'T0':<{W_ACC}} {'T100':<{W_ACC}} "
              f"{'T500':<{W_ACC}} {'T1000':<{W_ACC}} {'Oracle':<{W_ORACLE}} "
              f"{'Cascade':<{W_CASCADE}} {'Gap':<{W_GAP}}")

    print("-" * line_width)

    total_n = 0
    total_oracle = 0
    total_cascade = 0
    total_best_baseline = 0
    total_t0 = 0
    total_t100 = 0
    total_t500 = 0
    total_t1000 = 0
    total_mentor_len = {0: 0, 100: 0, 500: 0, 1000: 0}
    total_intern_len = {0: 0, 100: 0, 500: 0, 1000: 0}
    total_len_count = {0: 0, 100: 0, 500: 0, 1000: 0}

    for subset in SUBSETS:
        result_file = os.path.join(data_dir, subset, "lora_model", "cascade_eval.json")
        if not os.path.exists(result_file):
            print(f"{subset:<{W_SUBSET}} (not evaluated)")
            continue

        with open(result_file, 'r') as f:
            r = json.load(f)

        n = r['n_test']
        b = r['baseline']
        oracle = r['oracle']
        cascade = r['cascade_accuracy']

        # Handle both string and int keys
        t0 = b.get('0', b.get(0, 0))
        t100 = b.get('100', b.get(100, 0))
        t500 = b.get('500', b.get(500, 0))
        t1000 = b.get('1000', b.get(1000, 0))

        best_baseline = max(b.values())
        gap = cascade - best_baseline

        if show_length:
            # 计算长度统计
            length_stats = compute_length_stats(data_dir, subset)

            # 格式化输出：acc + m_len + i_len
            def fmt_token_group(acc, l_stat):
                """格式化一个 token level 的 acc + m_len + i_len"""
                m_len = l_stat.get('mentor', {}).get('mean', 0) if l_stat else 0
                i_len = l_stat.get('intern', {}).get('mean', 0) if l_stat else 0
                m_str = f"{m_len:<{W_LEN}.1f}" if m_len else f"{'-':<{W_LEN}}"
                i_str = f"{i_len:<{W_LEN}.1f}" if i_len else f"{'-':<{W_LEN}}"
                return f"{acc:<{W_ACC}.4f}{m_str}{i_str}"

            l0 = length_stats.get(0)
            l100 = length_stats.get(100)
            l500 = length_stats.get(500)
            l1000 = length_stats.get(1000)

            print(f"{subset:<{W_SUBSET}} {n:<{W_N}} "
                  f"{fmt_token_group(t0, l0)} {fmt_token_group(t100, l100)} "
                  f"{fmt_token_group(t500, l500)} {fmt_token_group(t1000, l1000)} "
                  f"{oracle:<{W_ORACLE}.4f} {cascade:<{W_CASCADE}.4f} {gap:+.4f}")

            # 累计长度统计
            for tokens, l_stat in [(0, l0), (100, l100), (500, l500), (1000, l1000)]:
                if l_stat:
                    if l_stat.get('mentor', {}).get('mean'):
                        total_mentor_len[tokens] += l_stat['mentor']['mean'] * n
                    if l_stat.get('intern', {}).get('mean'):
                        total_intern_len[tokens] += l_stat['intern']['mean'] * n
                    total_len_count[tokens] += n
        else:
            print(f"{subset:<{W_SUBSET}} {n:<{W_N}} {t0:<{W_ACC}.4f} {t100:<{W_ACC}.4f} "
                  f"{t500:<{W_ACC}.4f} {t1000:<{W_ACC}.4f} {oracle:<{W_ORACLE}.4f} "
                  f"{cascade:<{W_CASCADE}.4f} {gap:+.4f}")

        total_n += n
        total_oracle += oracle * n
        total_cascade += cascade * n
        total_best_baseline += best_baseline * n
        total_t0 += t0 * n
        total_t100 += t100 * n
        total_t500 += t500 * n
        total_t1000 += t1000 * n

    print("-" * line_width)
    if total_n > 0:
        avg_t0 = total_t0 / total_n
        avg_t100 = total_t100 / total_n
        avg_t500 = total_t500 / total_n
        avg_t1000 = total_t1000 / total_n
        avg_oracle = total_oracle / total_n
        avg_cascade = total_cascade / total_n
        avg_gap = (total_cascade - total_best_baseline) / total_n

        if show_length:
            def fmt_total_token_group(acc, tokens):
                """格式化 TOTAL 行的 acc + m_len + i_len"""
                if total_len_count[tokens] > 0:
                    m_len = total_mentor_len[tokens] / total_len_count[tokens]
                    i_len = total_intern_len[tokens] / total_len_count[tokens]
                else:
                    m_len = 0
                    i_len = 0
                m_str = f"{m_len:<{W_LEN}.1f}" if m_len else f"{'-':<{W_LEN}}"
                i_str = f"{i_len:<{W_LEN}.1f}" if i_len else f"{'-':<{W_LEN}}"
                return f"{acc:<{W_ACC}.4f}{m_str}{i_str}"

            print(f"{'TOTAL (weighted)':<{W_SUBSET}} {total_n:<{W_N}} "
                  f"{fmt_total_token_group(avg_t0, 0)} {fmt_total_token_group(avg_t100, 100)} "
                  f"{fmt_total_token_group(avg_t500, 500)} {fmt_total_token_group(avg_t1000, 1000)} "
                  f"{avg_oracle:<{W_ORACLE}.4f} {avg_cascade:<{W_CASCADE}.4f} {avg_gap:+.4f}")
        else:
            print(f"{'TOTAL (weighted)':<{W_SUBSET}} {total_n:<{W_N}} {avg_t0:<{W_ACC}.4f} "
                  f"{avg_t100:<{W_ACC}.4f} {avg_t500:<{W_ACC}.4f} {avg_t1000:<{W_ACC}.4f} "
                  f"{avg_oracle:<{W_ORACLE}.4f} {avg_cascade:<{W_CASCADE}.4f} {avg_gap:+.4f}")

    print("=" * line_width)

    # Additional analysis
    print("\n" + "=" * 60)
    print("Gap Analysis (Cascade - Best Baseline)")
    print("=" * 60)

    gaps = []
    for subset in SUBSETS:
        result_file = os.path.join(data_dir, subset, "lora_model", "cascade_eval.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                r = json.load(f)
            gap = r['cascade_accuracy'] - max(r['baseline'].values())
            gaps.append((subset, gap, r['n_test']))

    if gaps:
        gaps.sort(key=lambda x: x[1], reverse=True)
        for subset, gap, n in gaps:
            status = "+" if gap > 0 else ""
            print(f"  {subset:<{W_SUBSET}}: {status}{gap:.4f} ({status}{gap*100:.2f}%)")

    # Oracle gap analysis
    print("\n" + "=" * 60)
    print("Oracle Gap Analysis (Oracle - Cascade)")
    print("=" * 60)

    oracle_gaps = []
    for subset in SUBSETS:
        result_file = os.path.join(data_dir, subset, "lora_model", "cascade_eval.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                r = json.load(f)
            oracle_gap = r['oracle'] - r['cascade_accuracy']
            oracle_gaps.append((subset, oracle_gap, r['n_test']))

    if oracle_gaps:
        oracle_gaps.sort(key=lambda x: x[1], reverse=True)
        for subset, gap, n in oracle_gaps:
            print(f"  {subset:<{W_SUBSET}}: {gap:.4f} ({gap*100:.2f}%)")

    # 长度统计汇总
    if show_length:
        print("\n" + "=" * 90)
        print("Generation Length Statistics (avg tokens)")
        print("=" * 90)
        print(f"{'Subset':<{W_SUBSET}} {'T0':<16} {'T100':<16} {'T500':<16} {'T1000':<16}")
        print(f"{'':<{W_SUBSET}} {'m':<8}{'i':<8} {'m':<8}{'i':<8} {'m':<8}{'i':<8} {'m':<8}{'i':<8}")
        print("-" * 90)

        for subset in SUBSETS:
            length_stats = compute_length_stats(data_dir, subset)
            if any(length_stats.get(t) for t in TOKEN_LEVELS):
                row = f"{subset:<{W_SUBSET}}"
                for tokens in TOKEN_LEVELS:
                    l_stat = length_stats.get(tokens)
                    if l_stat:
                        m_len = l_stat.get('mentor', {}).get('mean', 0)
                        i_len = l_stat.get('intern', {}).get('mean', 0)
                        m_str = f"{m_len:<8.1f}" if m_len else f"{'-':<8}"
                        i_str = f"{i_len:<8.1f}" if i_len else f"{'-':<8}"
                        row += f" {m_str}{i_str}"
                    else:
                        row += f" {'-':<8}{'-':<8}"
                print(row)

        # Total average
        print("-" * 90)
        row = f"{'TOTAL (weighted)':<{W_SUBSET}}"
        for tokens in TOKEN_LEVELS:
            if total_len_count[tokens] > 0:
                m_len = total_mentor_len[tokens] / total_len_count[tokens]
                i_len = total_intern_len[tokens] / total_len_count[tokens]
                m_str = f"{m_len:<8.1f}" if m_len else f"{'-':<8}"
                i_str = f"{i_len:<8.1f}" if i_len else f"{'-':<8}"
                row += f" {m_str}{i_str}"
            else:
                row += f" {'-':<8}{'-':<8}"
        print(row)

        print("\n  m = mentor (大模型), i = intern (小模型)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Data directory")
    parser.add_argument("--no-length", action="store_true",
                        help="Don't show generation length statistics")

    args = parser.parse_args()
    summarize(args.data_dir, show_length=not args.no_length)


if __name__ == "__main__":
    main()
