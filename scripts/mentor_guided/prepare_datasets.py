#!/usr/bin/env python3
"""
Prepare datasets for ACT-E experiments.

Creates:
1. MATH-500: 400 train + 100 test from MATH test set
2. HumanEval: 130 train + 34 test
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


def load_math_data() -> List[Dict[str, Any]]:
    """Load MATH test data."""
    test_path = os.path.join(MATH_DIR, "test.json")
    with open(test_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_math_500(seed: int = 42) -> None:
    """
    Prepare MATH-500 dataset.

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
    logger.info(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare MATH-500
    prepare_math_500()

    # Prepare HumanEval
    try:
        prepare_humaneval()
    except Exception as e:
        logger.error(f"Failed to prepare HumanEval: {e}")
        logger.info("You may need to install: pip install datasets")

    logger.info("\n=== Summary ===")
    for dataset_name in ["math500", "humaneval"]:
        dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
        if os.path.exists(dataset_dir):
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
