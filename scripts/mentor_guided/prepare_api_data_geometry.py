#!/usr/bin/env python3
"""
Prepare MATH Geometry dataset for API-based thinking insight experiments.

This script processes the MATH Geometry subset and converts it to JSONL format
suitable for training/evaluating models with the 4 thinking insights framework.

Input: Hendrycks MATH Geometry dataset (train/test JSON files)
Output: JSONL files with system prompt (introducing 4 insights) and user questions

Usage:
    python prepare_api_data_geometry.py --output_dir /path/to/output
    python prepare_api_data_geometry.py  # Uses default output directory

Output format (each line is a JSON object):
    {
        "system": "<4 insights introduction prompt>",
        "user": "<geometry question>"
    }
"""

import argparse
import json
import os
from pathlib import Path

# Default paths (relative to Ensemble-Hub root)
SCRIPT_DIR = Path(__file__).parent.absolute()
ENSEMBLE_HUB_ROOT = SCRIPT_DIR.parent.parent

# Source data paths
TRAIN_PATH = ENSEMBLE_HUB_ROOT / "data/math/hendrycks_math/hendrycks_math/train/geometry.json"
TEST_PATH = ENSEMBLE_HUB_ROOT / "data/math/hendrycks_math/hendrycks_math/test/geometry.json"

# Default output directory
DEFAULT_OUTPUT_DIR = ENSEMBLE_HUB_ROOT / "data/acte_experiments/api_data/geometry"

# System prompt with 4 thinking insights introduction
# Based on the ACT-E framework from "2025-ACL-Zichuan-Fu-LLM-Ensemble"
SYSTEM_PROMPT = """You are a helpful assistant that solves geometry problems step by step. When reasoning, consider the following structured thinking insights to guide your problem-solving process:

1. **Goal**: Clearly define the ultimate objective or question to be solved. Clarify what you aim to achieve through reasoning.

2. **Planning**: Outline your high-level reasoning strategy. Consider how to decompose the problem into subproblems and select appropriate solution paths.

3. **Retrieval**: Recall or gather relevant knowledge, facts, theorems, or contextual information necessary for solving the problem. This includes geometric formulas, properties, and relationships.

4. **Action**: Execute concrete reasoning steps, calculations, or logical operations that directly lead to the final answer.

Use these insights to structure your thinking and provide clear, step-by-step solutions to geometry problems."""


def load_json(filepath: Path) -> list:
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_data(data: list) -> list:
    """
    Convert MATH dataset format to API-ready JSONL format.

    Input format (each item):
        {"instruction": "", "input": "<question>", "output": "<solution>"}

    Output format (each item):
        {"system": "<insights prompt>", "user": "<question>"}
    """
    processed = []
    for item in data:
        processed.append({
            "system": SYSTEM_PROMPT,
            "user": item["input"]  # The question content
        })
    return processed


def save_jsonl(data: list, filepath: Path) -> None:
    """Save data as JSONL file (one JSON object per line)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MATH Geometry dataset for API-based experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default output directory
    python prepare_api_data_geometry.py

    # Specify custom output directory
    python prepare_api_data_geometry.py --output_dir ./my_data

    # Show sample output only
    python prepare_api_data_geometry.py --dry_run
        """
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for JSONL files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show sample output, don't write files"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check source files exist
    if not TRAIN_PATH.exists():
        print(f"Error: Train file not found at {TRAIN_PATH}")
        return
    if not TEST_PATH.exists():
        print(f"Error: Test file not found at {TEST_PATH}")
        return

    # Process train data
    print(f"Loading train data from {TRAIN_PATH}")
    train_data = load_json(TRAIN_PATH)
    print(f"  -> Loaded {len(train_data)} train samples")

    train_processed = process_data(train_data)

    # Process test data
    print(f"\nLoading test data from {TEST_PATH}")
    test_data = load_json(TEST_PATH)
    print(f"  -> Loaded {len(test_data)} test samples")

    test_processed = process_data(test_data)

    # Print sample
    print("\n" + "=" * 70)
    print("Sample output (first item from train):")
    print("=" * 70)
    sample = train_processed[0]
    print(f"system: {sample['system'][:200]}...")
    print(f"\nuser: {sample['user'][:300]}...")
    print("=" * 70)

    if args.dry_run:
        print("\n[Dry run] No files written.")
        return

    # Save files
    train_output_path = output_dir / "geometry_train.jsonl"
    save_jsonl(train_processed, train_output_path)
    print(f"\nSaved train data ({len(train_processed)} samples) to:")
    print(f"  {train_output_path}")

    test_output_path = output_dir / "geometry_test.jsonl"
    save_jsonl(test_processed, test_output_path)
    print(f"\nSaved test data ({len(test_processed)} samples) to:")
    print(f"  {test_output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
