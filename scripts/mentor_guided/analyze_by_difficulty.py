#!/usr/bin/env python3
"""Analyze Tandem cascade results by MATH difficulty level.

This script maps difficulty levels from the original MATH dataset back to
collected inference results, then computes per-level accuracy and stage
distribution.

Usage:
    python analyze_by_difficulty.py --data-dir /path/to/hendrycks_math_split

The data-dir should contain subdirectories like algebra/test/tokens0.json, etc.
"""

import json
import os
import re
import argparse
from collections import defaultdict

# MATH subsets
SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

TOKEN_LEVELS = [0, 100, 500, 1000]


def load_math_with_levels():
    """Load original MATH dataset from HuggingFace to get difficulty levels.

    Returns:
        - level_map: dict of question_text -> difficulty_level (int)
        - index_map: dict of (subset, index) -> difficulty_level (int)
    """
    from datasets import load_dataset

    HF_NAMES = [
        "EleutherAI/hendrycks_math",
        "hendrycks/competition_math",
    ]

    level_map = {}       # question_text -> level
    index_map = {}       # (subset, index) -> level

    for hf_name in HF_NAMES:
        try:
            print(f"Trying to load from '{hf_name}'...")
            for subset in SUBSETS:
                dataset = load_dataset(hf_name, subset, split="test")
                for idx, item in enumerate(dataset):
                    q = item["problem"].strip()
                    level_str = item.get("level", "")
                    m = re.search(r"(\d)", str(level_str))
                    level = int(m.group(1)) if m else None
                    level_map[q] = level
                    index_map[(subset, idx)] = level
                print(f"  {subset}: {len(dataset)} problems")
            print(f"Loaded {len(level_map)} problems with difficulty levels from '{hf_name}'")
            return level_map, index_map
        except Exception as e:
            print(f"  Failed: {e}")
            level_map = {}
            index_map = {}
            continue

    raise RuntimeError("Could not load MATH dataset from any known source.")


def load_collected_data(data_dir):
    """Load collected inference results for all subsets and token levels."""
    all_data = {}  # (subset, token_level) -> list of items
    for subset in SUBSETS:
        for tl in TOKEN_LEVELS:
            fname = f"tokens{tl}.json"
            fpath = os.path.join(data_dir, subset, "test", fname)
            if not os.path.exists(fpath):
                print(f"  [WARN] Missing: {fpath}")
                continue
            with open(fpath) as f:
                items = json.load(f)
            all_data[(subset, tl)] = items
            if tl == 0:
                print(f"  Loaded {subset}: {len(items)} items, "
                      f"keys={list(items[0].keys()) if items else 'empty'}")
    return all_data


def get_level_from_item(item):
    """Try to extract difficulty level directly from a collected data item."""
    level_str = item.get("level", "")
    if level_str:
        m = re.search(r"(\d)", str(level_str))
        if m:
            return int(m.group(1))
    return None


def analyze_by_difficulty(data_dir, level_map, index_map):
    """Compute accuracy by difficulty level and stage distribution."""
    all_data = load_collected_data(data_dir)

    # Build per-question results
    # Key: (subset, index) for reliable matching
    questions = {}  # (subset, idx) -> {level, correct_at: {tl: bool}}

    # Track matching method stats
    match_direct = 0
    match_text = 0
    match_index = 0
    match_none = 0

    for (subset, tl), items in all_data.items():
        for idx, item in enumerate(items):
            key = (subset, idx)
            if key not in questions:
                # Strategy 1: Direct level field from collected data
                level = get_level_from_item(item)
                method = "direct"

                if level is None:
                    # Strategy 2: Exact text match
                    q = item["question"].strip()
                    level = level_map.get(q)
                    method = "text"

                if level is None:
                    # Strategy 3: Index-based match (same subset, same position)
                    level = index_map.get((subset, idx))
                    method = "index"

                if level is not None:
                    if method == "direct":
                        match_direct += 1
                    elif method == "text":
                        match_text += 1
                    else:
                        match_index += 1
                else:
                    match_none += 1

                questions[key] = {"level": level, "subset": subset, "correct_at": {}}
            questions[key]["correct_at"][tl] = item.get("is_correct", False)

    print(f"\nMatching stats:")
    print(f"  Direct (level field):  {match_direct}")
    print(f"  Text match:            {match_text}")
    print(f"  Index match:           {match_index}")
    print(f"  Unmatched:             {match_none}")
    print(f"  Total unique questions: {len(questions)}")

    # Filter to questions with all token levels and known difficulty
    complete = {k: info for k, info in questions.items()
                if info["level"] is not None and len(info["correct_at"]) == len(TOKEN_LEVELS)}

    print(f"\nTotal questions with complete data and difficulty level: {len(complete)}")

    if len(complete) == 0:
        print("\nERROR: No questions matched. Debug info:")
        # Show sample data
        sample_key = list(questions.keys())[0] if questions else None
        if sample_key:
            sample = questions[sample_key]
            print(f"  Sample key: {sample_key}")
            print(f"  Sample level: {sample['level']}")
            print(f"  Sample correct_at keys: {list(sample['correct_at'].keys())}")
            print(f"  Expected TOKEN_LEVELS: {TOKEN_LEVELS}")
        return

    # ===== 1. Accuracy by difficulty level and token level =====
    print("\n" + "=" * 70)
    print("ACCURACY BY DIFFICULTY LEVEL AND TOKEN LEVEL")
    print("=" * 70)

    acc_by_level = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for k, info in complete.items():
        level = info["level"]
        for tl in TOKEN_LEVELS:
            if info["correct_at"].get(tl, False):
                acc_by_level[level][tl]["correct"] += 1
            acc_by_level[level][tl]["total"] += 1

    # Print table
    header = f"{'Level':<10}" + "".join(f"{'T' + str(tl):>12}" for tl in TOKEN_LEVELS)
    print(header)
    print("-" * len(header))

    for level in sorted(acc_by_level.keys()):
        row = f"Level {level:<5}"
        for tl in TOKEN_LEVELS:
            d = acc_by_level[level][tl]
            acc = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
            row += f"{acc:>10.2f}% "
        n = acc_by_level[level][TOKEN_LEVELS[0]]["total"]
        row += f"  (n={n})"
        print(row)

    # Overall
    row = f"{'Overall':<10}"
    for tl in TOKEN_LEVELS:
        total_c = sum(acc_by_level[l][tl]["correct"] for l in acc_by_level)
        total_n = sum(acc_by_level[l][tl]["total"] for l in acc_by_level)
        acc = total_c / total_n * 100 if total_n > 0 else 0
        row += f"{acc:>10.2f}% "
    row += f"  (n={sum(acc_by_level[l][TOKEN_LEVELS[0]]['total'] for l in acc_by_level)})"
    print(row)

    # ===== 2. Stage distribution by difficulty level =====
    print("\n" + "=" * 70)
    print("EARLIEST CORRECT STAGE BY DIFFICULTY LEVEL")
    print("(Fraction of problems first solved at each token level)")
    print("=" * 70)

    stage_by_level = defaultdict(lambda: defaultdict(int))
    unsolvable_by_level = defaultdict(int)
    total_by_level = defaultdict(int)

    for k, info in complete.items():
        level = info["level"]
        total_by_level[level] += 1
        found = False
        for tl in TOKEN_LEVELS:
            if info["correct_at"].get(tl, False):
                stage_by_level[level][tl] += 1
                found = True
                break
        if not found:
            unsolvable_by_level[level] += 1

    header = f"{'Level':<10}" + "".join(f"{'T' + str(tl):>12}" for tl in TOKEN_LEVELS) + f"{'Unsolvable':>12}"
    print(header)
    print("-" * len(header))

    for level in sorted(stage_by_level.keys()):
        row = f"Level {level:<5}"
        n = total_by_level[level]
        for tl in TOKEN_LEVELS:
            cnt = stage_by_level[level][tl]
            pct = cnt / n * 100 if n > 0 else 0
            row += f"{pct:>10.2f}% "
        unsolvable_pct = unsolvable_by_level[level] / n * 100 if n > 0 else 0
        row += f"{unsolvable_pct:>10.2f}% "
        row += f"  (n={n})"
        print(row)

    # Overall
    row = f"{'Overall':<10}"
    total_n = sum(total_by_level.values())
    for tl in TOKEN_LEVELS:
        cnt = sum(stage_by_level[l][tl] for l in stage_by_level)
        pct = cnt / total_n * 100 if total_n > 0 else 0
        row += f"{pct:>10.2f}% "
    unsolvable_total = sum(unsolvable_by_level.values())
    pct = unsolvable_total / total_n * 100 if total_n > 0 else 0
    row += f"{pct:>10.2f}% "
    row += f"  (n={total_n})"
    print(row)

    # ===== 3. Oracle accuracy by difficulty =====
    print("\n" + "=" * 70)
    print("ORACLE ACCURACY BY DIFFICULTY LEVEL")
    print("(Solvable at ANY token level)")
    print("=" * 70)

    for level in sorted(total_by_level.keys()):
        n = total_by_level[level]
        solvable = n - unsolvable_by_level[level]
        oracle_acc = solvable / n * 100 if n > 0 else 0
        print(f"Level {level}: {oracle_acc:.2f}% ({solvable}/{n})")

    total_solvable = total_n - unsolvable_total
    oracle_pct = total_solvable / total_n * 100 if total_n > 0 else 0
    print(f"Overall:  {oracle_pct:.2f}% ({total_solvable}/{total_n})")

    # ===== Output for rebuttal =====
    print("\n" + "=" * 70)
    print("MARKDOWN TABLE FOR REBUTTAL (Accuracy by difficulty level)")
    print("=" * 70)
    print()
    print("| Difficulty | SLM (7B) | 7B+32B (low) | 7B+32B (medium) | 7B+32B (high) | n |")
    print("| ---------- | -------- | ------------ | --------------- | ------------- | - |")
    for level in sorted(acc_by_level.keys()):
        n = acc_by_level[level][TOKEN_LEVELS[0]]["total"]
        cols = []
        for tl in TOKEN_LEVELS:
            d = acc_by_level[level][tl]
            acc = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
            cols.append(f"{acc:.2f}%")
        print(f"| Level {level}    | {cols[0]:>8} | {cols[1]:>12} | {cols[2]:>15} | {cols[3]:>13} | {n} |")
    # Overall
    cols = []
    for tl in TOKEN_LEVELS:
        total_c = sum(acc_by_level[l][tl]["correct"] for l in acc_by_level)
        total_n_tl = sum(acc_by_level[l][tl]["total"] for l in acc_by_level)
        acc = total_c / total_n_tl * 100 if total_n_tl > 0 else 0
        cols.append(f"{acc:.2f}%")
    total_n_all = sum(acc_by_level[l][TOKEN_LEVELS[0]]["total"] for l in acc_by_level)
    print(f"| **Overall**| **{cols[0]}** | **{cols[1]}** | **{cols[2]}** | **{cols[3]}** | {total_n_all} |")


def main():
    parser = argparse.ArgumentParser(description="Analyze Tandem results by MATH difficulty level")
    parser.add_argument("--data-dir", required=True,
                        help="Path to collected data (e.g., .../hendrycks_math_split)")
    args = parser.parse_args()

    print("Loading MATH difficulty levels from HuggingFace...")
    level_map, index_map = load_math_with_levels()

    print(f"\nLoading collected data from {args.data_dir}...")
    analyze_by_difficulty(args.data_dir, level_map, index_map)


if __name__ == "__main__":
    main()
