#!/usr/bin/env python3
"""
Prepare datasets for ACT-E experiments.

Creates:
1. MATH-500: 400 train + 100 test from MATH test set (legacy)
2. MATH-Full: Complete MATH dataset from HuggingFace EleutherAI/hendrycks_math
3. HumanEval: 130 train + 34 test
"""

import json
import os
import random
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MATH_DIR = os.path.join(DATA_DIR, "math", "hendrycks_math", "hendrycks_math")
OUTPUT_DIR = os.path.join(DATA_DIR, "acte_experiments")

# MATH subsets from EleutherAI/hendrycks_math
MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Enhanced instruction with structured thinking guidance
MATH_INSTRUCTION = """Solve the following math problem step by step. Structure your reasoning using the following framework:

1. **Goal**: Define the ultimate objective or question to be solved.
2. **Planning**: Outline the high-level reasoning strategy, including decomposition of subproblems.
3. **Retrieval**: Recall relevant knowledge, facts, or formulas necessary for problem solving.
4. **Action**: Execute concrete reasoning steps, calculations, or logical operations.

Write your reasoning clearly using LaTeX. Box the final answer using \\boxed{}."""


def load_math_data_from_huggingface(split: str = "test") -> List[Dict[str, Any]]:
    """Load MATH data from HuggingFace EleutherAI/hendrycks_math.

    Args:
        split: "train" or "test"

    Returns:
        List of all problems from all subsets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    all_data = []

    for subset in MATH_SUBSETS:
        logger.info(f"Loading {subset} {split}...")
        dataset = load_dataset("EleutherAI/hendrycks_math", subset, split=split)

        for item in dataset:
            all_data.append({
                'input': item['problem'],
                'output': item['solution'],
                'type': item.get('type', subset),
                'level': item.get('level', ''),
                'subset': subset,
            })

        logger.info(f"  Loaded {len(dataset)} problems from {subset}")

    logger.info(f"Total {split}: {len(all_data)} problems")
    return all_data


def load_math_data() -> List[Dict[str, Any]]:
    """Load MATH test data (legacy function for backward compatibility)."""
    # Try HuggingFace first
    try:
        return load_math_data_from_huggingface("test")
    except Exception as e:
        logger.warning(f"Failed to load from HuggingFace: {e}")
        logger.info("Falling back to local file...")

    # Fallback to local file
    test_path = os.path.join(MATH_DIR, "test.json")
    with open(test_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_math_500(seed: int = 42) -> None:
    """
    Prepare MATH-500 dataset (legacy).

    - Sample 500 problems from MATH test set
    - Split into 400 train + 100 test
    """
    logger.info("Preparing MATH-500 dataset...")

    data = load_math_data()
    logger.info(f"Loaded {len(data)} MATH problems")

    # Set seed for reproducibility
    random.seed(seed)

    # Sample 500 problems
    sampled = random.sample(data, 500)
    logger.info(f"Sampled 500 problems")

    # Shuffle and split
    random.shuffle(sampled)
    train_data = sampled[:400]
    test_data = sampled[400:]

    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, "math500")
    os.makedirs(output_dir, exist_ok=True)

    # Save train and test sets
    train_path = os.path.join(output_dir, "train.json")
    test_path = os.path.join(output_dir, "test.json")

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(train_data)} train samples to {train_path}")

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(test_data)} test samples to {test_path}")

    # Also save in JSONL format for easier processing
    train_jsonl = os.path.join(output_dir, "train.jsonl")
    test_jsonl = os.path.join(output_dir, "test.jsonl")

    with open(train_jsonl, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(test_jsonl, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info("MATH-500 dataset prepared successfully!")


def prepare_math_full(seed: int = 42, use_enhanced_prompt: bool = True) -> None:
    """
    Prepare full MATH dataset with enhanced prompt.

    - Loads from HuggingFace EleutherAI/hendrycks_math (all 7 subsets)
    - Combines train + test splits (~12.5k problems total)
    - NO pre-split - cross-validation will be done during experiment
    - Updates instruction with structured thinking guidance

    Args:
        seed: Random seed for reproducibility
        use_enhanced_prompt: Whether to use enhanced instruction with thinking structure
    """
    logger.info("Preparing MATH-Full dataset from HuggingFace...")

    # Load both train and test from HuggingFace
    train_data = load_math_data_from_huggingface("train")
    test_data = load_math_data_from_huggingface("test")

    # Combine all data
    data = train_data + test_data
    logger.info(f"Combined {len(train_data)} train + {len(test_data)} test = {len(data)} total problems")

    # Update instruction if using enhanced prompt
    if use_enhanced_prompt:
        logger.info("Applying enhanced instruction with structured thinking guidance...")
        for item in data:
            item['instruction'] = MATH_INSTRUCTION

    # Set seed for reproducibility
    random.seed(seed)

    # Shuffle data (but no split - CV will be done later)
    random.shuffle(data)

    logger.info(f"Total samples: {len(data)} (no train/test split, CV will be done during experiment)")

    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, "math_full")
    os.makedirs(output_dir, exist_ok=True)

    # Save all data as a single file
    all_path = os.path.join(output_dir, "all.json")
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} samples to {all_path}")

    # Also save in JSONL format for easier processing
    all_jsonl = os.path.join(output_dir, "all.jsonl")
    with open(all_jsonl, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info("MATH-Full dataset prepared successfully!")
    logger.info(f"Enhanced prompt: {use_enhanced_prompt}")


def download_humaneval() -> List[Dict[str, Any]]:
    """Download HumanEval dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai_humaneval", split="test")
        return list(dataset)
    except ImportError:
        logger.warning("datasets library not installed. Trying manual download...")
        import requests

        url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
        import gzip
        import io

        response = requests.get(url)
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            data = [json.loads(line) for line in f.read().decode('utf-8').strip().split('\n')]
        return data


def prepare_humaneval(seed: int = 42) -> None:
    """
    Prepare HumanEval dataset.

    - 164 total problems
    - Split into 130 train + 34 test
    """
    logger.info("Preparing HumanEval dataset...")

    data = download_humaneval()
    logger.info(f"Loaded {len(data)} HumanEval problems")

    # Set seed for reproducibility
    random.seed(seed)

    # Shuffle and split
    indices = list(range(len(data)))
    random.shuffle(indices)

    train_indices = indices[:130]
    test_indices = indices[130:]

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, "humaneval")
    os.makedirs(output_dir, exist_ok=True)

    # Save train and test sets
    train_path = os.path.join(output_dir, "train.json")
    test_path = os.path.join(output_dir, "test.json")

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(train_data)} train samples to {train_path}")

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(test_data)} test samples to {test_path}")

    # Also save in JSONL format
    train_jsonl = os.path.join(output_dir, "train.jsonl")
    test_jsonl = os.path.join(output_dir, "test.jsonl")

    with open(train_jsonl, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(test_jsonl, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info("HumanEval dataset prepared successfully!")


def main():
    """Prepare all datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare datasets for ACT-E experiments")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "math500", "math_full", "humaneval"],
                        help="Which dataset to prepare")
    parser.add_argument("--no_enhanced_prompt", action="store_true",
                        help="Disable enhanced prompt for math_full")
    args = parser.parse_args()

    logger.info(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.dataset in ["all", "math500"]:
        prepare_math_500()

    if args.dataset in ["all", "math_full"]:
        prepare_math_full(
            use_enhanced_prompt=not args.no_enhanced_prompt
        )

    if args.dataset in ["all", "humaneval"]:
        try:
            prepare_humaneval()
        except Exception as e:
            logger.error(f"Failed to prepare HumanEval: {e}")
            logger.info("You may need to install: pip install datasets")

    logger.info("\n=== Summary ===")
    for dataset_name in ["math500", "math_full", "humaneval"]:
        dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
        if os.path.exists(dataset_dir):
            # Check for new single-file format (all.json)
            all_path = os.path.join(dataset_dir, "all.json")
            if os.path.exists(all_path):
                with open(all_path) as f:
                    all_count = len(json.load(f))
                logger.info(f"{dataset_name}: {all_count} samples (single file, no pre-split)")
            else:
                # Legacy train/test format
                train_path = os.path.join(dataset_dir, "train.json")
                test_path = os.path.join(dataset_dir, "test.json")
                if os.path.exists(train_path):
                    with open(train_path) as f:
                        train_count = len(json.load(f))
                else:
                    train_count = 0
                if os.path.exists(test_path):
                    with open(test_path) as f:
                        test_count = len(json.load(f))
                else:
                    test_count = 0
                logger.info(f"{dataset_name}: {train_count} train, {test_count} test")


if __name__ == "__main__":
    main()
