#!/usr/bin/env python3
"""
Process GPT-4o geometry results to create:
1. Standalone evaluation results
2. Mentor-only data (tokens-1.json) for cascade
3. Training data for MLP/PPL classifiers

Usage:
    python process_gpt4o_geometry.py --split train
    python process_gpt4o_geometry.py --split test
"""

import json
import os
import re
from typing import Dict, List, Any
from pathlib import Path
import argparse

# Import grading function
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grader import grade_answer


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} in the text."""
    # Find all \boxed{...} patterns
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return last boxed answer
    return ""


def load_math_dataset(split: str = "train") -> Dict[str, Any]:
    """Load MATH geometry dataset ground truth from existing processed data."""
    # Try to load from existing processed data first
    existing_data_path = (
        "/home/fzkuji/PycharmProjects/Ensemble-Hub/data/acte_experiments/collected/"
        "hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B/geometry/"
        f"{split}/tokens0.json"
    )

    if os.path.exists(existing_data_path):
        print(f"Loading ground truth from existing data: {existing_data_path}")
        with open(existing_data_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        geometry_data = {}
        for item in existing_data:
            problem_text = item["question"].strip()
            # Extract answer from \boxed{} in the solution
            gt_answer = extract_boxed_answer(item["ground_truth"])
            geometry_data[problem_text] = {
                "problem": problem_text,
                "ground_truth": gt_answer,  # Extracted answer only
                "full_solution": item["ground_truth"],  # Keep full solution for reference
                "level": item.get("level", ""),
                "type": "Geometry",
            }

        print(f"Loaded {len(geometry_data)} geometry problems from existing data ({split} split)")
        return geometry_data

    # Fallback: try to load from HuggingFace
    try:
        from datasets import load_dataset

        dataset = load_dataset("hendrycks/competition_math", split=split)

        geometry_data = {}
        for item in dataset:
            if item.get("type") == "Geometry":
                problem_text = item["problem"].strip()
                geometry_data[problem_text] = {
                    "problem": problem_text,
                    "ground_truth": extract_boxed_answer(item["solution"]),
                    "level": item.get("level", ""),
                    "type": item.get("type", "Geometry"),
                }

        print(f"Loaded {len(geometry_data)} geometry problems from HuggingFace ({split} split)")
        return geometry_data

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("WARNING: No ground truth available - evaluation will be limited")
        return {}


def load_gpt4o_results(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load GPT-4o results from JSONL file."""
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results.append(data)

    print(f"Loaded {len(results)} GPT-4o results from {jsonl_path}")
    return results


def create_tokens_format(
    gpt4o_results: List[Dict[str, Any]],
    math_dataset: Dict[str, Any],
    output_dir: str,
    split: str = "train"
) -> Dict[str, Any]:
    """
    Create tokens-1.json format (mentor-only results) from GPT-4o data.

    Returns statistics about matching and correctness.
    """
    processed_results = []
    stats = {
        "total": len(gpt4o_results),
        "matched": 0,
        "unmatched": 0,
        "correct": 0,
        "incorrect": 0,
    }

    for item in gpt4o_results:
        problem = item["user"].strip()
        gpt4o_output = item["gpt-4o"]

        # Extract GPT-4o answer from \boxed{}
        predicted_answer = extract_boxed_answer(gpt4o_output)

        # Try to find ground truth
        ground_truth = None
        level = ""

        if problem in math_dataset:
            stats["matched"] += 1
            gt_item = math_dataset[problem]
            ground_truth = gt_item["ground_truth"]
            level = gt_item.get("level", "")
        else:
            stats["unmatched"] += 1
            # Try fuzzy matching (first 100 chars)
            problem_prefix = problem[:100]
            for key, value in math_dataset.items():
                if key.startswith(problem_prefix):
                    stats["matched"] += 1
                    stats["unmatched"] -= 1
                    ground_truth = value["ground_truth"]
                    level = value.get("level", "")
                    break

        # Grade the answer
        is_correct = False
        if ground_truth:
            is_correct = grade_answer(predicted_answer, ground_truth)
            if is_correct:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1

        # Create result entry in tokens-1.json format
        result_entry = {
            "question": problem,
            "answer": ground_truth if ground_truth else "",
            "generated": gpt4o_output,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "level": level,
            "type": "Geometry",
        }

        processed_results.append(result_entry)

    # Save to tokens-1.json (mentor-only format)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tokens-1.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Processing Statistics ===")
    print(f"Total samples: {stats['total']}")
    print(f"Matched with ground truth: {stats['matched']}")
    print(f"Unmatched: {stats['unmatched']}")
    print(f"Correct: {stats['correct']}")
    print(f"Incorrect: {stats['incorrect']}")
    if stats['matched'] > 0:
        accuracy = stats['correct'] / stats['matched'] * 100
        print(f"Accuracy: {accuracy:.2f}%")

    print(f"\nSaved to: {output_path}")

    return stats


def create_evaluation_report(
    stats: Dict[str, Any],
    output_dir: str,
    model_name: str = "gpt-4o"
):
    """Create standalone evaluation report."""
    report = {
        "model": model_name,
        "dataset": "MATH Geometry",
        "total_samples": stats["total"],
        "matched_samples": stats["matched"],
        "correct": stats["correct"],
        "incorrect": stats["incorrect"],
        "accuracy": stats["correct"] / stats["matched"] if stats["matched"] > 0 else 0,
    }

    report_path = os.path.join(output_dir, "gpt4o_evaluation.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Evaluation Report ===")
    print(f"Model: {report['model']}")
    print(f"Dataset: {report['dataset']}")
    print(f"Accuracy: {report['accuracy']*100:.2f}%")
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Process GPT-4o geometry results")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/fzkuji/PycharmProjects/Ensemble-Hub/data/acte_experiments/collected",
        help="Directory containing GPT-4o JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: hendrycks_math_think_mGPT-4o_iNone/geometry/{split})"
    )

    args = parser.parse_args()

    # Load GPT-4o results
    jsonl_path = os.path.join(args.input_dir, f"geometry_{args.split}_result.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found")
        return

    gpt4o_results = load_gpt4o_results(jsonl_path)

    # Load MATH dataset
    math_dataset = load_math_dataset(split=args.split)

    # Determine output directory
    if args.output_dir is None:
        # Match the pipeline's expected structure
        # Use local path for development, will be uploaded to server later
        base_dir = "/home/fzkuji/PycharmProjects/Ensemble-Hub/data/acte_experiments/collected"
        args.output_dir = os.path.join(
            base_dir,
            "hendrycks_math_think_mGPT-4o_iNone",
            "geometry",
            args.split
        )

    # Create tokens-1.json and get statistics
    stats = create_tokens_format(
        gpt4o_results,
        math_dataset,
        args.output_dir,
        args.split
    )

    # Create evaluation report
    create_evaluation_report(stats, args.output_dir)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. To evaluate GPT-4o alone: Check {os.path.join(args.output_dir, 'gpt4o_evaluation.json')}")
    print(f"2. To use in cascade: Add intern model results (tokens0.json, tokens100.json, etc.)")
    print(f"3. To train classifier: Run train_mlp_classifier.py with this data directory")


if __name__ == "__main__":
    main()
