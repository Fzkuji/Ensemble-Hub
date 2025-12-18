#!/usr/bin/env python3
"""
Generate comprehensive results table from experiment data.

Usage:
    python generate_results_table.py --data-dir /path/to/data [--format table|latex|csv]
"""

import argparse
import json
import os
from typing import Dict, List, Optional
import sys

TOKEN_LEVELS = [0, 100, 500, 1000]

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

SUBSET_SHORT = {
    "algebra": "Algebra",
    "counting_and_probability": "Count&Prob",
    "geometry": "Geometry",
    "intermediate_algebra": "Inter.Alg",
    "number_theory": "NumTheory",
    "prealgebra": "PreAlg",
    "precalculus": "PreCalc",
}


def load_subset_results(data_dir: str, subset: str, split: str = "test") -> Dict:
    """Load results for a subset."""
    if subset == "all":
        subset_dir = os.path.join(data_dir, "all")
    else:
        subset_dir = os.path.join(data_dir, subset)

    split_dir = os.path.join(subset_dir, split)

    results = {
        'subset': subset,
        'split': split,
        'n_samples': 0,
        'baseline': {},
        'mentor_only': None,
        'oracle': None,
        'cascade': None,
    }

    # Load baseline accuracy for each token level
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(split_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            n = len(data)
            correct = sum(1 for d in data if d.get('is_correct', False))
            results['baseline'][tokens] = {
                'accuracy': correct / n if n > 0 else 0,
                'correct': correct,
                'total': n,
            }
            results['n_samples'] = n

    # Load mentor_only if exists
    mentor_file = os.path.join(split_dir, "tokensmentor_only.json")
    if os.path.exists(mentor_file):
        with open(mentor_file, 'r') as f:
            data = json.load(f)
        n = len(data)
        correct = sum(1 for d in data if d.get('is_correct', False))
        results['mentor_only'] = correct / n if n > 0 else 0

    # Compute oracle (best across all stages)
    if results['baseline']:
        oracle_correct = 0
        n = results['n_samples']
        # Load all data to compute oracle
        all_data = {}
        for tokens in TOKEN_LEVELS:
            filepath = os.path.join(split_dir, f"tokens{tokens}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    all_data[tokens] = json.load(f)

        if all_data:
            for i in range(n):
                # Oracle: correct if ANY stage is correct
                for tokens in TOKEN_LEVELS:
                    if tokens in all_data and i < len(all_data[tokens]):
                        if all_data[tokens][i].get('is_correct', False):
                            oracle_correct += 1
                            break
            results['oracle'] = oracle_correct / n if n > 0 else 0

    # Load cascade results if available (try both filenames)
    cascade_file = os.path.join(subset_dir, "lora_model", "cascade_eval.json")
    if not os.path.exists(cascade_file):
        cascade_file = os.path.join(subset_dir, "lora_model", "eval_results.json")
    if os.path.exists(cascade_file):
        with open(cascade_file, 'r') as f:
            cascade_data = json.load(f)
        results['cascade'] = cascade_data.get('cascade_accuracy', cascade_data.get('best_accuracy'))
        results['thresholds'] = cascade_data.get('best_thresholds')
        results['auc'] = cascade_data.get('auc')

    return results


def compute_totals(all_results: List[Dict]) -> Dict:
    """Compute totals across all subsets."""
    totals = {
        'subset': 'TOTAL',
        'n_samples': 0,
        'baseline': {t: {'correct': 0, 'total': 0} for t in TOKEN_LEVELS},
        'mentor_only_correct': 0,
        'mentor_only_total': 0,
        'oracle_correct': 0,
        'cascade_correct': 0,
    }

    for r in all_results:
        n = r.get('n_samples', 0)
        totals['n_samples'] += n

        for tokens in TOKEN_LEVELS:
            if tokens in r.get('baseline', {}):
                totals['baseline'][tokens]['correct'] += r['baseline'][tokens]['correct']
                totals['baseline'][tokens]['total'] += r['baseline'][tokens]['total']

        if r.get('mentor_only') is not None:
            totals['mentor_only_correct'] += int(r['mentor_only'] * n)
            totals['mentor_only_total'] += n

        if r.get('oracle') is not None:
            totals['oracle_correct'] += int(r['oracle'] * n)

        if r.get('cascade') is not None:
            totals['cascade_correct'] += int(r['cascade'] * n)

    # Compute final accuracies
    totals['baseline_acc'] = {}
    for tokens in TOKEN_LEVELS:
        t = totals['baseline'][tokens]['total']
        c = totals['baseline'][tokens]['correct']
        totals['baseline_acc'][tokens] = c / t if t > 0 else 0

    if totals['mentor_only_total'] > 0:
        totals['mentor_only'] = totals['mentor_only_correct'] / totals['mentor_only_total']
    else:
        totals['mentor_only'] = None

    if totals['n_samples'] > 0:
        totals['oracle'] = totals['oracle_correct'] / totals['n_samples']
        totals['cascade'] = totals['cascade_correct'] / totals['n_samples'] if totals['cascade_correct'] > 0 else None
    else:
        totals['oracle'] = None
        totals['cascade'] = None

    return totals


def print_table(all_results: List[Dict], totals: Dict, split: str):
    """Print results as ASCII table."""
    print(f"\n{'='*130}")
    print(f"RESULTS SUMMARY ({split.upper()} SET)")
    print(f"{'='*130}")

    header = f"{'Subset':<20} {'N':<7}"
    for t in TOKEN_LEVELS:
        header += f"{'T='+str(t):<10}"
    header += f"{'Mentor':<10} {'Oracle':<10} {'Cascade':<10}"
    print(header)
    print("-"*130)

    for r in all_results:
        row = f"{SUBSET_SHORT.get(r['subset'], r['subset']):<20} {r['n_samples']:<7}"
        for t in TOKEN_LEVELS:
            if t in r.get('baseline', {}):
                row += f"{r['baseline'][t]['accuracy']:.4f}    "
            else:
                row += f"{'N/A':<10}"

        if r.get('mentor_only') is not None:
            row += f"{r['mentor_only']:.4f}    "
        else:
            row += f"{'N/A':<10}"

        if r.get('oracle') is not None:
            row += f"{r['oracle']:.4f}    "
        else:
            row += f"{'N/A':<10}"

        if r.get('cascade') is not None:
            row += f"{r['cascade']:.4f}    "
        else:
            row += f"{'N/A':<10}"

        print(row)

    print("-"*130)

    # Totals row
    row = f"{'TOTAL':<20} {totals['n_samples']:<7}"
    for t in TOKEN_LEVELS:
        row += f"{totals['baseline_acc'][t]:.4f}    "

    if totals.get('mentor_only') is not None:
        row += f"{totals['mentor_only']:.4f}    "
    else:
        row += f"{'N/A':<10}"

    if totals.get('oracle') is not None:
        row += f"{totals['oracle']:.4f}    "
    else:
        row += f"{'N/A':<10}"

    if totals.get('cascade') is not None:
        row += f"{totals['cascade']:.4f}    "
    else:
        row += f"{'N/A':<10}"

    print(row)
    print("="*130)


def print_latex(all_results: List[Dict], totals: Dict, split: str):
    """Print results as LaTeX table."""
    print(f"\n% Results for {split} set")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{ACT-E Results on MATH Dataset}")
    print("\\label{tab:results}")
    print("\\begin{tabular}{l|r|cccc|c|c|c}")
    print("\\toprule")
    print("Subset & N & T=0 & T=100 & T=500 & T=1000 & Mentor & Oracle & Cascade \\\\")
    print("\\midrule")

    for r in all_results:
        name = SUBSET_SHORT.get(r['subset'], r['subset'])
        row = f"{name} & {r['n_samples']}"

        for t in TOKEN_LEVELS:
            if t in r.get('baseline', {}):
                acc = r['baseline'][t]['accuracy']
                row += f" & {acc:.2%}"
            else:
                row += " & --"

        if r.get('mentor_only') is not None:
            row += f" & {r['mentor_only']:.2%}"
        else:
            row += " & --"

        if r.get('oracle') is not None:
            row += f" & {r['oracle']:.2%}"
        else:
            row += " & --"

        if r.get('cascade') is not None:
            row += f" & {r['cascade']:.2%}"
        else:
            row += " & --"

        row += " \\\\"
        print(row)

    print("\\midrule")

    # Totals
    row = f"\\textbf{{Total}} & {totals['n_samples']}"
    for t in TOKEN_LEVELS:
        row += f" & {totals['baseline_acc'][t]:.2%}"

    if totals.get('mentor_only') is not None:
        row += f" & {totals['mentor_only']:.2%}"
    else:
        row += " & --"

    if totals.get('oracle') is not None:
        row += f" & {totals['oracle']:.2%}"
    else:
        row += " & --"

    if totals.get('cascade') is not None:
        row += f" & \\textbf{{{totals['cascade']:.2%}}}"
    else:
        row += " & --"

    row += " \\\\"
    print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_csv(all_results: List[Dict], totals: Dict, split: str):
    """Print results as CSV."""
    header = "subset,n_samples"
    for t in TOKEN_LEVELS:
        header += f",t{t}"
    header += ",mentor_only,oracle,cascade"
    print(header)

    for r in all_results:
        row = f"{r['subset']},{r['n_samples']}"
        for t in TOKEN_LEVELS:
            if t in r.get('baseline', {}):
                row += f",{r['baseline'][t]['accuracy']:.6f}"
            else:
                row += ","

        row += f",{r.get('mentor_only', '')}"
        row += f",{r.get('oracle', '')}"
        row += f",{r.get('cascade', '')}"
        print(row)

    # Totals
    row = f"TOTAL,{totals['n_samples']}"
    for t in TOKEN_LEVELS:
        row += f",{totals['baseline_acc'][t]:.6f}"
    row += f",{totals.get('mentor_only', '')}"
    row += f",{totals.get('oracle', '')}"
    row += f",{totals.get('cascade', '')}"
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Generate results table")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Base directory with subset folders")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Which split to summarize")
    parser.add_argument("--format", type=str, default="table",
                        choices=["table", "latex", "csv", "all"],
                        help="Output format")
    parser.add_argument("--subsets", type=str, default=None,
                        help="Comma-separated list of subsets (default: all)")

    args = parser.parse_args()

    # Determine subsets
    if args.subsets:
        subsets = [s.strip() for s in args.subsets.split(",")]
    else:
        subsets = SUBSETS

    # Load results
    all_results = []
    for subset in subsets:
        results = load_subset_results(args.data_dir, subset, args.split)
        if results['n_samples'] > 0:
            all_results.append(results)

    if not all_results:
        print("No results found!")
        sys.exit(1)

    # Compute totals
    totals = compute_totals(all_results)

    # Print results
    if args.format == "table" or args.format == "all":
        print_table(all_results, totals, args.split)

    if args.format == "latex" or args.format == "all":
        print_latex(all_results, totals, args.split)

    if args.format == "csv" or args.format == "all":
        print_csv(all_results, totals, args.split)


if __name__ == "__main__":
    main()
