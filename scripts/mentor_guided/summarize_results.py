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
from typing import List
import numpy as np

# Default subsets for hendrycks_math (used as fallback)
DEFAULT_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def detect_subsets(data_dir: str, split: str = "test") -> List[str]:
    """Auto-detect subsets from data directory."""
    subsets = []
    if not os.path.exists(data_dir):
        return DEFAULT_SUBSETS

    for name in os.listdir(data_dir):
        subset_dir = os.path.join(data_dir, name, split)
        # Check if it's a valid subset directory (has tokens*.json files)
        if os.path.isdir(subset_dir):
            token_file = os.path.join(subset_dir, "tokens0.json")
            if os.path.exists(token_file):
                subsets.append(name)

    return sorted(subsets) if subsets else DEFAULT_SUBSETS

# Token levels: -1 = mentor only, 0 = intern only, others = mentor hint + intern
TOKEN_LEVELS = [0, 100, 500, 1000]
MENTOR_ONLY_LEVEL = -1

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


def compute_mentor_only_stats(data_dir: str, subset: str, split: str = "test"):
    """计算 mentor only 的准确率和长度统计"""
    data_file = os.path.join(data_dir, subset, split, "tokens-1.json")
    if not os.path.exists(data_file):
        return None

    with open(data_file, 'r') as f:
        data = json.load(f)

    if not data:
        return None

    correct = sum(1 for item in data if item.get('is_correct', False))
    accuracy = correct / len(data) if data else 0

    # 统计长度
    mentor_lengths = []
    for item in data:
        if 'mentor_length' in item:
            mentor_lengths.append(item['mentor_length'])
        elif 'mentor_response' in item and item['mentor_response']:
            mentor_lengths.append(len(item['mentor_response']) // 4)

    return {
        'accuracy': accuracy,
        'n_samples': len(data),
        'n_correct': correct,
        'mentor_length_mean': np.mean(mentor_lengths) if mentor_lengths else 0,
        'mentor_length_std': np.std(mentor_lengths) if mentor_lengths else 0,
    }


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


def summarize(data_dir: str, show_length: bool = True, model_source: str = "individual"):
    """汇总所有子集的结果

    Args:
        data_dir: 数据目录
        show_length: 是否显示长度统计
        model_source: 模型来源 - "individual" (各子集单独训练), "all" (合并训练)
    """
    # Auto-detect subsets from data directory
    SUBSETS = detect_subsets(data_dir, "test")
    print(f"Detected subsets: {SUBSETS}")

    # Mentor 列宽度 (acc + len)
    W_MENTOR = W_ACC + W_LEN
    # 计算实际行宽 (T0-T1000 各 24, Mentor 16, Oracle/Cascade 各 24)
    if show_length:
        line_width = W_SUBSET + W_N + W_TOKEN_GROUP * 4 + W_MENTOR + W_TOKEN_GROUP * 2 + W_GAP + 10
    else:
        line_width = 110

    source_label = "[individual]" if model_source == "individual" else "[all]"
    print("=" * line_width)
    print(f"Results Summary: {data_dir} {source_label}")
    print("=" * line_width)

    if show_length:
        # Mentor 列宽度 (acc + len，没有 i_len)
        W_MENTOR = W_ACC + W_LEN
        # 表头第一行：Subset, N, T0, T100, T500, T1000, Mentor, Oracle, Cascade, Gap
        print(f"{'Subset':<{W_SUBSET}} {'N':<{W_N}} "
              f"{'T0':<{W_TOKEN_GROUP}} {'T100':<{W_TOKEN_GROUP}} "
              f"{'T500':<{W_TOKEN_GROUP}} {'T1000':<{W_TOKEN_GROUP}} "
              f"{'Mentor':<{W_MENTOR}} "
              f"{'Oracle':<{W_TOKEN_GROUP}} {'Cascade':<{W_TOKEN_GROUP}} {'Gap':<{W_GAP}}")
        # 表头第二行：acc, m_len, i_len
        print(f"{'':<{W_SUBSET}} {'':<{W_N}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}} "
              f"{'acc':<{W_ACC}}{'m_len':<{W_LEN}}{'i_len':<{W_LEN}}")
    else:
        print(f"{'Subset':<{W_SUBSET}} {'N':<{W_N}} {'T0':<{W_ACC}} {'T100':<{W_ACC}} "
              f"{'T500':<{W_ACC}} {'T1000':<{W_ACC}} {'Mentor':<{W_ACC}} {'Oracle':<{W_ORACLE}} "
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
    # Mentor-only 累计
    total_mentor_acc = 0
    total_mentor_only_len = 0
    # Oracle/Cascade 长度累计
    total_oracle_m_len = 0
    total_oracle_i_len = 0
    total_cascade_m_len = 0
    total_cascade_i_len = 0

    # Collect gap data for later analysis
    collected_gaps = []  # [(subset, gap, n, oracle, cascade)]

    # 根据 model_source 决定加载路径
    all_mlp_results = None
    all_mlp_dir = os.path.join(data_dir, "all", "mlp_model")

    for subset in SUBSETS:
        r = None

        if model_source == "individual":
            # 优先使用各子集单独训练的结果
            # Priority: 1) subset/mlp_model, 2) subset/lora_model
            
            # 1. 首先检查各子集单独训练的 MLP 结果
            mlp_file = os.path.join(data_dir, subset, "mlp_model", "results.json")
            if os.path.exists(mlp_file):
                with open(mlp_file, 'r') as f:
                    mlp_r = json.load(f)
                r = {
                    'n_test': mlp_r.get('n_val', 0),
                    'baseline': mlp_r.get('test_per_stage_baseline_acc', mlp_r.get('per_stage_baseline_acc', {})),
                    'oracle': mlp_r.get('test_oracle_acc', mlp_r.get('oracle_acc', 0)),
                    'cascade_accuracy': mlp_r.get('test_best_cascade_acc', mlp_r.get('best_cascade_acc', 0)),
                    'oracle_length': mlp_r.get('test_oracle_length', mlp_r.get('oracle_length', {})),
                    'cascade_length': mlp_r.get('test_cascade_length', mlp_r.get('cascade_length', {})),
                }

            # 2. 然后检查各子集的 LoRA 结果
            if r is None:
                result_file = os.path.join(data_dir, subset, "lora_model", "cascade_eval.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        r = json.load(f)

        elif model_source == "all":
            # 从 all/ 目录加载合并训练的结果
            # 延迟加载 all_mlp_results
            if all_mlp_results is None:
                for fname in ["results_all.json", "results.json"]:
                    fpath = os.path.join(all_mlp_dir, fname)
                    if os.path.exists(fpath):
                        with open(fpath, 'r') as f:
                            all_mlp_results = json.load(f)
                        break
            
            # 检查 all/mlp_model 的 per-subset 结果
            if all_mlp_results and 'test_results_per_subset' in all_mlp_results:
                subset_results = all_mlp_results['test_results_per_subset'].get(subset)
                if subset_results:
                    r = {
                        'n_test': subset_results.get('n_test', subset_results.get('n_samples', 0)),
                        'baseline': subset_results.get('per_stage_baseline_acc', {}),
                        'oracle': subset_results.get('oracle_acc', 0),
                        'cascade_accuracy': subset_results.get('cascade_acc', 0),
                        'oracle_length': subset_results.get('oracle_length', {}),
                        'cascade_length': subset_results.get('cascade_length', {}),
                    }
            
            # 检查 all/mlp_model 的 per-subset 结果文件
            if r is None:
                subset_result_file = os.path.join(all_mlp_dir, f"results_{subset}.json")
                if os.path.exists(subset_result_file):
                    with open(subset_result_file, 'r') as f:
                        subset_mlp = json.load(f)
                    if 'test_results_per_subset' in subset_mlp and subset in subset_mlp['test_results_per_subset']:
                        subset_results = subset_mlp['test_results_per_subset'][subset]
                        r = {
                            'n_test': subset_results.get('n_test', subset_results.get('n_samples', 0)),
                            'baseline': subset_results.get('per_stage_baseline_acc', {}),
                            'oracle': subset_results.get('oracle_acc', 0),
                            'cascade_accuracy': subset_results.get('cascade_acc', 0),
                            'oracle_length': subset_results.get('oracle_length', {}),
                            'cascade_length': subset_results.get('cascade_length', {}),
                        }
                    else:
                        r = {
                            'n_test': subset_mlp.get('n_val', 0),
                            'baseline': subset_mlp.get('test_per_stage_baseline_acc', subset_mlp.get('per_stage_baseline_acc', {})),
                            'oracle': subset_mlp.get('test_oracle_acc', subset_mlp.get('oracle_acc', 0)),
                            'cascade_accuracy': subset_mlp.get('test_best_cascade_acc', subset_mlp.get('best_cascade_acc', 0)),
                            'oracle_length': subset_mlp.get('oracle_length', {}),
                            'cascade_length': subset_mlp.get('cascade_length', {}),
                        }

        if r is None:
            print(f"{subset:<{W_SUBSET}} (not evaluated)")
            continue

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
            
            # 获取 Mentor-only 结果
            mentor_stats = compute_mentor_only_stats(data_dir, subset)
            mentor_acc = mentor_stats['accuracy'] if mentor_stats else 0
            mentor_len = mentor_stats['mentor_length_mean'] if mentor_stats else 0
            
            # 获取 Oracle/Cascade 长度（从结果文件中读取）
            oracle_len = r.get('oracle_length', {})
            cascade_len = r.get('cascade_length', {})
            oracle_m = oracle_len.get('mentor_mean', 0)
            oracle_i = oracle_len.get('intern_mean', 0)
            cascade_m = cascade_len.get('mentor_mean', 0)
            cascade_i = cascade_len.get('intern_mean', 0)
            
            # Mentor 列格式化
            mentor_acc_str = f"{mentor_acc:<{W_ACC}.4f}" if mentor_acc else f"{'-':<{W_ACC}}"
            mentor_len_str = f"{mentor_len:<{W_LEN}.1f}" if mentor_len else f"{'-':<{W_LEN}}"
            
            oracle_m_str = f"{oracle_m:<{W_LEN}.1f}" if oracle_m else f"{'-':<{W_LEN}}"
            oracle_i_str = f"{oracle_i:<{W_LEN}.1f}" if oracle_i else f"{'-':<{W_LEN}}"
            cascade_m_str = f"{cascade_m:<{W_LEN}.1f}" if cascade_m else f"{'-':<{W_LEN}}"
            cascade_i_str = f"{cascade_i:<{W_LEN}.1f}" if cascade_i else f"{'-':<{W_LEN}}"

            print(f"{subset:<{W_SUBSET}} {n:<{W_N}} "
                  f"{fmt_token_group(t0, l0)} {fmt_token_group(t100, l100)} "
                  f"{fmt_token_group(t500, l500)} {fmt_token_group(t1000, l1000)} "
                  f"{mentor_acc_str}{mentor_len_str} "
                  f"{oracle:<{W_ACC}.4f}{oracle_m_str}{oracle_i_str} "
                  f"{cascade:<{W_ACC}.4f}{cascade_m_str}{cascade_i_str} {gap:+.4f}")

            # 累计长度统计
            for tokens, l_stat in [(0, l0), (100, l100), (500, l500), (1000, l1000)]:
                if l_stat:
                    if l_stat.get('mentor', {}).get('mean'):
                        total_mentor_len[tokens] += l_stat['mentor']['mean'] * n
                    if l_stat.get('intern', {}).get('mean'):
                        total_intern_len[tokens] += l_stat['intern']['mean'] * n
                    total_len_count[tokens] += n
            
            # 累计 Mentor 统计
            if mentor_stats:
                total_mentor_acc += mentor_acc * n
                total_mentor_only_len += mentor_len * n
            
            # 累计 Oracle/Cascade 长度
            if oracle_m:
                total_oracle_m_len += oracle_m * n
            if oracle_i:
                total_oracle_i_len += oracle_i * n
            if cascade_m:
                total_cascade_m_len += cascade_m * n
            if cascade_i:
                total_cascade_i_len += cascade_i * n
        else:
            # 获取 Mentor-only 结果（不显示长度时也需要）
            mentor_stats = compute_mentor_only_stats(data_dir, subset)
            mentor_acc = mentor_stats['accuracy'] if mentor_stats else 0
            mentor_acc_str = f"{mentor_acc:<{W_ACC}.4f}" if mentor_acc else "-"
            print(f"{subset:<{W_SUBSET}} {n:<{W_N}} {t0:<{W_ACC}.4f} {t100:<{W_ACC}.4f} "
                  f"{t500:<{W_ACC}.4f} {t1000:<{W_ACC}.4f} {mentor_acc_str:<{W_ACC}} {oracle:<{W_ORACLE}.4f} "
                  f"{cascade:<{W_CASCADE}.4f} {gap:+.4f}")
            # 累计 Mentor（不显示长度时）
            if mentor_stats:
                total_mentor_acc += mentor_acc * n

        total_n += n
        total_oracle += oracle * n
        total_cascade += cascade * n
        total_best_baseline += best_baseline * n
        total_t0 += t0 * n
        total_t100 += t100 * n
        total_t500 += t500 * n
        total_t1000 += t1000 * n

        # Collect for gap analysis
        collected_gaps.append((subset, gap, n, oracle, cascade))

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

            # Mentor 平均值
            avg_mentor_acc = total_mentor_acc / total_n if total_n > 0 else 0
            avg_mentor_len = total_mentor_only_len / total_n if total_n > 0 else 0
            
            # Oracle/Cascade 平均长度
            avg_oracle_m = total_oracle_m_len / total_n if total_n > 0 else 0
            avg_oracle_i = total_oracle_i_len / total_n if total_n > 0 else 0
            avg_cascade_m = total_cascade_m_len / total_n if total_n > 0 else 0
            avg_cascade_i = total_cascade_i_len / total_n if total_n > 0 else 0
            
            mentor_acc_str = f"{avg_mentor_acc:<{W_ACC}.4f}" if avg_mentor_acc else f"{'-':<{W_ACC}}"
            mentor_len_str = f"{avg_mentor_len:<{W_LEN}.1f}" if avg_mentor_len else f"{'-':<{W_LEN}}"
            oracle_m_str = f"{avg_oracle_m:<{W_LEN}.1f}" if avg_oracle_m else f"{'-':<{W_LEN}}"
            oracle_i_str = f"{avg_oracle_i:<{W_LEN}.1f}" if avg_oracle_i else f"{'-':<{W_LEN}}"
            cascade_m_str = f"{avg_cascade_m:<{W_LEN}.1f}" if avg_cascade_m else f"{'-':<{W_LEN}}"
            cascade_i_str = f"{avg_cascade_i:<{W_LEN}.1f}" if avg_cascade_i else f"{'-':<{W_LEN}}"
            
            print(f"{'TOTAL (weighted)':<{W_SUBSET}} {total_n:<{W_N}} "
                  f"{fmt_total_token_group(avg_t0, 0)} {fmt_total_token_group(avg_t100, 100)} "
                  f"{fmt_total_token_group(avg_t500, 500)} {fmt_total_token_group(avg_t1000, 1000)} "
                  f"{mentor_acc_str}{mentor_len_str} "
                  f"{avg_oracle:<{W_ACC}.4f}{oracle_m_str}{oracle_i_str} "
                  f"{avg_cascade:<{W_ACC}.4f}{cascade_m_str}{cascade_i_str} {avg_gap:+.4f}")
        else:
            avg_mentor_acc = total_mentor_acc / total_n if total_n > 0 else 0
            mentor_acc_str = f"{avg_mentor_acc:<{W_ACC}.4f}" if avg_mentor_acc else "-"
            print(f"{'TOTAL (weighted)':<{W_SUBSET}} {total_n:<{W_N}} {avg_t0:<{W_ACC}.4f} "
                  f"{avg_t100:<{W_ACC}.4f} {avg_t500:<{W_ACC}.4f} {avg_t1000:<{W_ACC}.4f} "
                  f"{mentor_acc_str:<{W_ACC}} {avg_oracle:<{W_ORACLE}.4f} {avg_cascade:<{W_CASCADE}.4f} {avg_gap:+.4f}")

    print("=" * line_width)

    # Additional analysis (using collected data from main loop)
    print("\n" + "=" * 60)
    print("Gap Analysis (Cascade - Best Baseline)")
    print("=" * 60)

    if collected_gaps:
        # Sort by gap descending
        sorted_gaps = sorted(collected_gaps, key=lambda x: x[1], reverse=True)
        for subset, gap, n, oracle, cascade in sorted_gaps:
            status = "+" if gap > 0 else ""
            print(f"  {subset:<{W_SUBSET}}: {status}{gap:.4f} ({status}{gap*100:.2f}%)")

    # Oracle gap analysis
    print("\n" + "=" * 60)
    print("Oracle Gap Analysis (Oracle - Cascade)")
    print("=" * 60)

    if collected_gaps:
        # Sort by oracle gap descending
        oracle_gaps = [(s, o - c, n) for s, g, n, o, c in collected_gaps]
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

    # Mentor Only 结果
    print("\n" + "=" * 70)
    print("Mentor Only Results (baseline: mentor model without intern)")
    print("=" * 70)
    print(f"{'Subset':<{W_SUBSET}} {'N':<{W_N}} {'Acc':<10} {'Len (avg)':<12}")
    print("-" * 70)

    mentor_only_total_n = 0
    mentor_only_total_correct = 0
    mentor_only_total_len = 0
    mentor_only_count = 0

    for subset in SUBSETS:
        mentor_stats = compute_mentor_only_stats(data_dir, subset)
        if mentor_stats:
            acc = mentor_stats['accuracy']
            n = mentor_stats['n_samples']
            m_len = mentor_stats['mentor_length_mean']
            print(f"{subset:<{W_SUBSET}} {n:<{W_N}} {acc:<10.4f} {m_len:<12.1f}")
            mentor_only_total_n += n
            mentor_only_total_correct += mentor_stats['n_correct']
            mentor_only_total_len += m_len * n
            mentor_only_count += n
        else:
            print(f"{subset:<{W_SUBSET}} (no mentor-only data)")

    if mentor_only_total_n > 0:
        print("-" * 70)
        avg_acc = mentor_only_total_correct / mentor_only_total_n
        avg_len = mentor_only_total_len / mentor_only_count if mentor_only_count > 0 else 0
        print(f"{'TOTAL (weighted)':<{W_SUBSET}} {mentor_only_total_n:<{W_N}} {avg_acc:<10.4f} {avg_len:<12.1f}")

        # 比较 mentor only vs cascade
        if total_n > 0:
            print("\n" + "-" * 70)
            avg_cascade = total_cascade / total_n
            cascade_vs_mentor = avg_cascade - avg_acc
            print(f"Cascade vs Mentor-Only: {cascade_vs_mentor:+.4f} ({cascade_vs_mentor*100:+.2f}%)")

    # 分类器对比表格
    print_classifier_comparison(data_dir, model_source, SUBSETS)

    print()


def get_classifier_results(data_dir: str, subset: str, model_type: str, model_source: str = "individual"):
    """获取指定分类器的结果

    Args:
        data_dir: 数据目录
        subset: 子集名称
        model_type: 模型类型 (mlp, ppl, lora, ensemble)
        model_source: 模型来源 - "individual" (各子集单独训练), "all" (合并训练)
    """
    if model_source == "individual":
        # 从 subset 目录加载各子集单独训练的结果
        result_file = os.path.join(data_dir, subset, f"{model_type}_model", "results.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                r = json.load(f)
            # 获取长度信息
            cascade_len = r.get('test_cascade_length', r.get('cascade_length', {}))
            if isinstance(cascade_len, dict):
                avg_len = cascade_len.get('mean', cascade_len.get('avg', 0))
            else:
                avg_len = cascade_len if cascade_len else 0
            return {
                'cascade_acc': r.get('test_best_cascade_acc', r.get('best_cascade_acc', r.get('cascade_acc', 0))),
                'oracle_acc': r.get('test_oracle_acc', r.get('oracle_acc', 0)),
                'cascade_length': avg_len,
                'source': 'individual',
            }
        
        # 对于 LoRA，还检查 cascade_eval.json
        if model_type == "lora":
            lora_eval_file = os.path.join(data_dir, subset, "lora_model", "cascade_eval.json")
            if os.path.exists(lora_eval_file):
                with open(lora_eval_file, 'r') as f:
                    r = json.load(f)
                cascade_len = r.get('cascade_length', r.get('avg_length', {}))
                if isinstance(cascade_len, dict):
                    avg_len = cascade_len.get('mean', cascade_len.get('avg', 0))
                else:
                    avg_len = cascade_len if cascade_len else 0
                return {
                    'cascade_acc': r.get('cascade_accuracy', 0),
                    'oracle_acc': r.get('oracle', 0),
                    'cascade_length': avg_len,
                    'source': 'individual',
                }

    elif model_source == "all":
        # 从 all 目录加载合并训练的结果
        all_model_dir = os.path.join(data_dir, "all", f"{model_type}_model")

        # 对于 MLP，检查 per-subset 结果
        if model_type == "mlp":
            for fname in [f"results_{subset}.json", "results_all.json", "results.json"]:
                fpath = os.path.join(all_model_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'r') as f:
                        r = json.load(f)
                    # 检查是否有 per-subset 结果
                    if 'test_results_per_subset' in r and subset in r['test_results_per_subset']:
                        sub_r = r['test_results_per_subset'][subset]
                        cascade_len = sub_r.get('cascade_length', {})
                        if isinstance(cascade_len, dict):
                            avg_len = cascade_len.get('mean', cascade_len.get('avg', 0))
                        else:
                            avg_len = cascade_len if cascade_len else 0
                        return {
                            'cascade_acc': sub_r.get('cascade_acc', 0),
                            'oracle_acc': sub_r.get('oracle_acc', 0),
                            'cascade_length': avg_len,
                            'source': 'all',
                        }

        # 对于 PPL，检查 test_results
        if model_type == "ppl":
            ppl_path = os.path.join(all_model_dir, "results.json")
            if os.path.exists(ppl_path):
                with open(ppl_path, 'r') as f:
                    r = json.load(f)
                if 'test_results' in r and subset in r['test_results']:
                    sub_r = r['test_results'][subset]
                    cascade_len = sub_r.get('cascade_length', {})
                    if isinstance(cascade_len, dict):
                        avg_len = cascade_len.get('mean', cascade_len.get('avg', 0))
                    else:
                        avg_len = cascade_len if cascade_len else 0
                    return {
                        'cascade_acc': sub_r.get('cascade_acc', 0),
                        'oracle_acc': sub_r.get('oracle_acc', 0),
                        'cascade_length': avg_len,
                        'source': 'all',
                    }

        # 对于 LoRA，检查 all/lora_model 的结果
        if model_type == "lora":
            # 检查 results.json
            lora_result = os.path.join(all_model_dir, "results.json")
            if os.path.exists(lora_result):
                with open(lora_result, 'r') as f:
                    r = json.load(f)
                # 检查是否有 per-subset 结果
                if 'test_results_per_subset' in r and subset in r['test_results_per_subset']:
                    sub_r = r['test_results_per_subset'][subset]
                    cascade_len = sub_r.get('cascade_length', {})
                    if isinstance(cascade_len, dict):
                        avg_len = cascade_len.get('mean', cascade_len.get('avg', 0))
                    else:
                        avg_len = cascade_len if cascade_len else 0
                    return {
                        'cascade_acc': sub_r.get('cascade_acc', sub_r.get('cascade_accuracy', 0)),
                        'oracle_acc': sub_r.get('oracle_acc', sub_r.get('oracle', 0)),
                        'cascade_length': avg_len,
                        'source': 'all',
                    }
            # 检查 cascade_eval.json
            lora_eval = os.path.join(all_model_dir, "cascade_eval.json")
            if os.path.exists(lora_eval):
                with open(lora_eval, 'r') as f:
                    r = json.load(f)
                # 检查是否有 per-subset 结果
                if 'per_subset' in r and subset in r['per_subset']:
                    sub_r = r['per_subset'][subset]
                    cascade_len = sub_r.get('cascade_length', sub_r.get('avg_length', {}))
                    if isinstance(cascade_len, dict):
                        avg_len = cascade_len.get('mean', cascade_len.get('avg', 0))
                    else:
                        avg_len = cascade_len if cascade_len else 0
                    return {
                        'cascade_acc': sub_r.get('cascade_accuracy', sub_r.get('cascade_acc', 0)),
                        'oracle_acc': sub_r.get('oracle', sub_r.get('oracle_acc', 0)),
                        'cascade_length': avg_len,
                        'source': 'all',
                    }

    return None


def get_subset_n_test(data_dir: str, subset: str) -> int:
    """获取子集的测试样本数"""
    # 从 tokens0.json 获取样本数
    test_file = os.path.join(data_dir, subset, "test", "tokens0.json")
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            data = json.load(f)
        return len(data)
    return 0


def print_classifier_comparison(data_dir: str, model_source: str = "individual", subsets: list = None):
    """打印分类器对比表格（包含准确率和长度）"""
    if subsets is None:
        subsets = detect_subsets(data_dir, "test")

    source_label = "[individual]" if model_source == "individual" else "[all]"
    print("\n" + "=" * 160)
    print(f"                                          CLASSIFIER COMPARISON {source_label}")
    print("=" * 160)
    print(f"{'Subset':<20} {'N':>5} {'LoRA Acc':>10} {'LoRA Len':>9} {'MLP Acc':>10} {'MLP Len':>9} {'PPL Acc':>10} {'PPL Len':>9} {'Ens Acc':>10} {'Oracle':>10} {'Best':>8}")
    print("-" * 160)

    # 加权平均统计
    totals = {'lora': 0, 'mlp': 0, 'ppl': 0, 'ensemble': 0}
    counts = {'lora': 0, 'mlp': 0, 'ppl': 0, 'ensemble': 0}
    len_totals = {'lora': 0, 'mlp': 0, 'ppl': 0, 'ensemble': 0}
    len_counts = {'lora': 0, 'mlp': 0, 'ppl': 0, 'ensemble': 0}
    total_n = 0

    for subset in subsets:
        n = get_subset_n_test(data_dir, subset)
        total_n += n

        lora_r = get_classifier_results(data_dir, subset, "lora", model_source)
        mlp_r = get_classifier_results(data_dir, subset, "mlp", model_source)
        ppl_r = get_classifier_results(data_dir, subset, "ppl", model_source)
        ens_r = get_classifier_results(data_dir, subset, "ensemble", model_source)

        lora_acc = lora_r['cascade_acc'] if lora_r else None
        mlp_acc = mlp_r['cascade_acc'] if mlp_r else None
        ppl_acc = ppl_r['cascade_acc'] if ppl_r else None
        ens_acc = ens_r['cascade_acc'] if ens_r else None

        lora_len = lora_r.get('cascade_length', 0) if lora_r else 0
        mlp_len = mlp_r.get('cascade_length', 0) if mlp_r else 0
        ppl_len = ppl_r.get('cascade_length', 0) if ppl_r else 0
        ens_len = ens_r.get('cascade_length', 0) if ens_r else 0

        # Oracle (优先使用 PPL 的，因为它在原始测试数据上评估)
        oracle = None
        if ppl_r and ppl_r.get('oracle_acc'):
            oracle = ppl_r['oracle_acc']
        elif mlp_r and mlp_r.get('oracle_acc'):
            oracle = mlp_r['oracle_acc']
        elif lora_r and lora_r.get('oracle_acc'):
            oracle = lora_r['oracle_acc']
        elif ens_r and ens_r.get('oracle_acc'):
            oracle = ens_r['oracle_acc']

        # 格式化输出
        lora_acc_str = f"{lora_acc:.4f}" if lora_acc is not None else "-"
        mlp_acc_str = f"{mlp_acc:.4f}" if mlp_acc is not None else "-"
        ppl_acc_str = f"{ppl_acc:.4f}" if ppl_acc is not None else "-"
        ens_acc_str = f"{ens_acc:.4f}" if ens_acc is not None else "-"
        oracle_str = f"{oracle:.4f}" if oracle is not None else "-"

        lora_len_str = f"{lora_len:.0f}" if lora_len else "-"
        mlp_len_str = f"{mlp_len:.0f}" if mlp_len else "-"
        ppl_len_str = f"{ppl_len:.0f}" if ppl_len else "-"
        ens_len_str = f"{ens_len:.0f}" if ens_len else "-"

        # 找最佳并累计加权统计
        accs = []
        if lora_acc is not None and lora_acc > 0:
            accs.append(('LoRA', lora_acc))
            totals['lora'] += lora_acc * n
            counts['lora'] += n
            if lora_len:
                len_totals['lora'] += lora_len * n
                len_counts['lora'] += n
        if mlp_acc is not None and mlp_acc > 0:
            accs.append(('MLP', mlp_acc))
            totals['mlp'] += mlp_acc * n
            counts['mlp'] += n
            if mlp_len:
                len_totals['mlp'] += mlp_len * n
                len_counts['mlp'] += n
        if ppl_acc is not None and ppl_acc > 0:
            accs.append(('PPL', ppl_acc))
            totals['ppl'] += ppl_acc * n
            counts['ppl'] += n
            if ppl_len:
                len_totals['ppl'] += ppl_len * n
                len_counts['ppl'] += n
        if ens_acc is not None and ens_acc > 0:
            accs.append(('Ens', ens_acc))
            totals['ensemble'] += ens_acc * n
            counts['ensemble'] += n
            if ens_len:
                len_totals['ensemble'] += ens_len * n
                len_counts['ensemble'] += n

        best = max(accs, key=lambda x: x[1])[0] if accs else "-"

        print(f"{subset:<20} {n:>5} {lora_acc_str:>10} {lora_len_str:>9} {mlp_acc_str:>10} {mlp_len_str:>9} {ppl_acc_str:>10} {ppl_len_str:>9} {ens_acc_str:>10} {oracle_str:>10} {best:>8}")

    print("-" * 160)

    # 计算加权平均
    lora_avg = f"{totals['lora'] / counts['lora']:.4f}" if counts['lora'] > 0 else "-"
    mlp_avg = f"{totals['mlp'] / counts['mlp']:.4f}" if counts['mlp'] > 0 else "-"
    ppl_avg = f"{totals['ppl'] / counts['ppl']:.4f}" if counts['ppl'] > 0 else "-"
    ens_avg = f"{totals['ensemble'] / counts['ensemble']:.4f}" if counts['ensemble'] > 0 else "-"

    lora_len_avg = f"{len_totals['lora'] / len_counts['lora']:.0f}" if len_counts['lora'] > 0 else "-"
    mlp_len_avg = f"{len_totals['mlp'] / len_counts['mlp']:.0f}" if len_counts['mlp'] > 0 else "-"
    ppl_len_avg = f"{len_totals['ppl'] / len_counts['ppl']:.0f}" if len_counts['ppl'] > 0 else "-"
    ens_len_avg = f"{len_totals['ensemble'] / len_counts['ensemble']:.0f}" if len_counts['ensemble'] > 0 else "-"

    print(f"{'TOTAL (weighted)':<20} {total_n:>5} {lora_avg:>10} {lora_len_avg:>9} {mlp_avg:>10} {mlp_len_avg:>9} {ppl_avg:>10} {ppl_len_avg:>9} {ens_avg:>10}")
    print("=" * 160)


def summarize_single(data_dir: str, subset: str, model_dir: str = None, show_length: bool = True):
    """Summarize results for a single subset (e.g., math500, all)."""
    if model_dir is None:
        model_dir = os.path.join(data_dir, "all", "lora_model")

    result_file = os.path.join(model_dir, "cascade_eval.json")
    if not os.path.exists(result_file):
        print(f"Results file not found: {result_file}")
        return

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

    print("\n" + "=" * 100)
    print(f"Results for: {subset} (N={n})")
    print("=" * 100)

    if show_length:
        # 计算长度统计
        length_stats = compute_length_stats(data_dir, subset)

        # Header with length columns
        print(f"\n{'Strategy':<15} {'Acc':<10} {'M_Len':<10} {'I_Len':<10}")
        print("-" * 45)

        for tokens in TOKEN_LEVELS:
            acc = b.get(str(tokens), b.get(tokens, 0))
            l_stat = length_stats.get(tokens)
            m_len = l_stat.get('mentor', {}).get('mean', 0) if l_stat else 0
            i_len = l_stat.get('intern', {}).get('mean', 0) if l_stat else 0

            m_str = f"{m_len:.1f}" if m_len else "-"
            i_str = f"{i_len:.1f}" if i_len else "-"
            print(f"T{tokens:<14} {acc:<10.4f} {m_str:<10} {i_str:<10}")

        print("-" * 45)
        # Oracle with length
        oracle_len = r.get('oracle_length', {})
        o_m = oracle_len.get('mentor_mean', 0)
        o_i = oracle_len.get('intern_mean', 0)
        o_m_str = f"{o_m:.1f}" if o_m else "-"
        o_i_str = f"{o_i:.1f}" if o_i else "-"
        print(f"{'Oracle':<15} {oracle:<10.4f} {o_m_str:<10} {o_i_str:<10}")

        # Cascade with length
        cascade_len = r.get('cascade_length', {})
        c_m = cascade_len.get('mentor_mean', 0)
        c_i = cascade_len.get('intern_mean', 0)
        c_m_str = f"{c_m:.1f}" if c_m else "-"
        c_i_str = f"{c_i:.1f}" if c_i else "-"
        print(f"{'Cascade':<15} {cascade:<10.4f} {c_m_str:<10} {c_i_str:<10}")
        print(f"{'Gap':<15} {gap:+.4f}")
    else:
        print(f"\n{'Strategy':<15} {'Accuracy':<10}")
        print("-" * 25)
        for tokens in TOKEN_LEVELS:
            acc = b.get(str(tokens), b.get(tokens, 0))
            print(f"T{tokens:<14} {acc:<10.4f}")
        print("-" * 25)
        print(f"{'Oracle':<15} {oracle:<10.4f}")
        print(f"{'Cascade':<15} {cascade:<10.4f}")
        print(f"{'Gap':<15} {gap:+.4f}")

    print("\n" + "=" * 100)

    # Thresholds used
    if 'thresholds' in r:
        print("\nThresholds used:")
        for i, thresh in enumerate(r['thresholds']):
            print(f"  Stage {i} (T{TOKEN_LEVELS[i]} -> T{TOKEN_LEVELS[i+1]}): {thresh:.4f}")

    # Mentor Only 结果
    mentor_stats = compute_mentor_only_stats(data_dir, subset)
    if mentor_stats:
        print("\n" + "-" * 50)
        print("Mentor Only Baseline:")
        print(f"  Accuracy: {mentor_stats['accuracy']:.4f}")
        print(f"  Avg Length: {mentor_stats['mentor_length_mean']:.1f} tokens")
        cascade_vs_mentor = cascade - mentor_stats['accuracy']
        print(f"  Cascade vs Mentor-Only: {cascade_vs_mentor:+.4f} ({cascade_vs_mentor*100:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Data directory")
    parser.add_argument("--subset", type=str, default=None,
                        help="Single subset to summarize (e.g., math500, all). If not specified, summarize all MATH subsets")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model directory for single subset mode")
    parser.add_argument("--no-length", action="store_true",
                        help="Don't show generation length statistics")
    parser.add_argument("--model-source", type=str, default="individual",
                        choices=["individual", "all"],
                        help="Model source: 'individual' (per-subset trained) or 'all' (unified trained)")

    args = parser.parse_args()

    if args.subset:
        summarize_single(args.data_dir, args.subset, args.model_dir, show_length=not args.no_length)
    else:
        summarize(args.data_dir, show_length=not args.no_length, model_source=args.model_source)


if __name__ == "__main__":
    main()
