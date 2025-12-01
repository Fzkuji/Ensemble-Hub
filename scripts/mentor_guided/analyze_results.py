#!/usr/bin/env python3
"""
Analyze mentor-guided results to find patterns.

分析哪些题被mentor"救回来"，哪些被"害了"。
"""

import json
import argparse
from collections import defaultdict


def analyze_results(results_file: str):
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data['results']
    lengths = data['config']['mentor_lengths']

    # Categories
    rescued = []      # baseline wrong, some mentor length correct
    hurt = []         # baseline correct, some mentor length wrong
    always_correct = []
    always_wrong = []

    for r in results:
        baseline_correct = r['length_results']['0']['is_correct']
        mentor_results = {int(k): v['is_correct'] for k, v in r['length_results'].items() if k != '0'}

        any_mentor_correct = any(mentor_results.values())
        any_mentor_wrong = not all(mentor_results.values())

        if baseline_correct:
            if any_mentor_wrong:
                hurt.append(r)
            else:
                always_correct.append(r)
        else:
            if any_mentor_correct:
                rescued.append(r)
            else:
                always_wrong.append(r)

    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(results)}")
    print(f"Always correct (mentor doesn't matter): {len(always_correct)}")
    print(f"Always wrong (mentor can't help): {len(always_wrong)}")
    print(f"RESCUED (baseline wrong -> mentor helps): {len(rescued)}")
    print(f"HURT (baseline correct -> mentor hurts): {len(hurt)}")

    print("\n" + "=" * 70)
    print("RESCUED CASES (mentor helped)")
    print("=" * 70)
    for r in rescued:
        print(f"\n--- Sample {r['idx']} (GT: {r['ground_truth']}) ---")
        for length, lr in r['length_results'].items():
            status = "✓" if lr['is_correct'] else "✗"
            print(f"  Length {length}: {status} pred={lr['predicted_boxed']}")
        # Show which length helped
        helpful_lengths = [int(k) for k, v in r['length_results'].items() if v['is_correct'] and k != '0']
        print(f"  Helpful lengths: {helpful_lengths}")

    print("\n" + "=" * 70)
    print("HURT CASES (mentor made it worse)")
    print("=" * 70)
    for r in hurt:
        print(f"\n--- Sample {r['idx']} (GT: {r['ground_truth']}) ---")
        for length, lr in r['length_results'].items():
            status = "✓" if lr['is_correct'] else "✗"
            print(f"  Length {length}: {status} pred={lr['predicted_boxed']}")
        # Show which length hurt
        harmful_lengths = [int(k) for k, v in r['length_results'].items() if not v['is_correct'] and k != '0']
        print(f"  Harmful lengths: {harmful_lengths}")

    # Detailed stats by length
    print("\n" + "=" * 70)
    print("TRANSITION ANALYSIS BY LENGTH")
    print("=" * 70)

    for length in lengths:
        if length == 0:
            continue

        wrong_to_right = 0  # rescued
        right_to_wrong = 0  # hurt
        stayed_right = 0
        stayed_wrong = 0

        for r in results:
            baseline = r['length_results']['0']['is_correct']
            with_mentor = r['length_results'][str(length)]['is_correct']

            if not baseline and with_mentor:
                wrong_to_right += 1
            elif baseline and not with_mentor:
                right_to_wrong += 1
            elif baseline and with_mentor:
                stayed_right += 1
            else:
                stayed_wrong += 1

        net_gain = wrong_to_right - right_to_wrong
        print(f"\nLength {length}:")
        print(f"  Rescued (wrong->right): {wrong_to_right}")
        print(f"  Hurt (right->wrong): {right_to_wrong}")
        print(f"  Stayed right: {stayed_right}")
        print(f"  Stayed wrong: {stayed_wrong}")
        print(f"  NET GAIN: {net_gain:+d}")

    return rescued, hurt, always_correct, always_wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-file', default='parallel_results.json')
    args = parser.parse_args()

    analyze_results(args.results_file)


if __name__ == "__main__":
    main()
