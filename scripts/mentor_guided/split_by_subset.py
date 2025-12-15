#!/usr/bin/env python3
"""
Split collected hidden states data by hendrycks_math subsets and train/test.

The hendrycks_math dataset has 7 subsets:
- algebra
- counting_and_probability
- geometry
- intermediate_algebra
- number_theory
- prealgebra
- precalculus

Each subset has train and test splits.
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List
import torch
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def load_hendrycks_math_metadata() -> Dict[str, Dict]:
    """Load hendrycks_math dataset to get subset and split info for each problem."""
    logger.info("Loading hendrycks_math dataset metadata...")

    metadata = {}

    for subset in SUBSETS:
        logger.info(f"  Loading subset: {subset}")
        ds = load_dataset("EleutherAI/hendrycks_math", subset)

        for split in ["train", "test"]:
            if split in ds:
                for idx, item in enumerate(ds[split]):
                    # Use solution text as key (matches ground_truth in collected data)
                    solution = item["solution"].strip()
                    metadata[solution] = {
                        "subset": subset,
                        "split": split,
                        "original_idx": idx,
                        "problem": item["problem"].strip(),
                    }

    logger.info(f"Loaded metadata for {len(metadata)} problems")
    return metadata


def load_collected_data(data_dir: str) -> Dict[int, Dict]:
    """Load collected hidden states and json data."""
    data = {}

    # Try to load hidden states from various locations
    hidden_dirs = [
        os.path.join(data_dir, "hidden_states"),
        data_dir,
    ]

    for hidden_dir in hidden_dirs:
        if not os.path.exists(hidden_dir):
            continue
        for tokens in TOKEN_LEVELS:
            filepath = os.path.join(hidden_dir, f"tokens{tokens}.pt")
            if os.path.exists(filepath) and tokens not in data:
                loaded = torch.load(filepath)
                data[tokens] = {
                    "hidden_states": loaded["hidden_states"],
                    "labels": loaded["labels"],
                }
                logger.info(f"Loaded tokens={tokens}: {len(loaded['labels'])} samples")

    # Load all json files
    json_data_by_tokens = {}
    for f in os.listdir(data_dir):
        if f.endswith(".json"):
            filepath = os.path.join(data_dir, f)
            # Extract token level from filename using regex for exact match
            # Match patterns like tokens0, tokens100, tokens500, tokens1000
            match = re.search(r'tokens(\d+)', f)
            if match:
                token_num = int(match.group(1))
                if token_num in TOKEN_LEVELS:
                    with open(filepath, 'r') as file:
                        json_data_by_tokens[str(token_num)] = json.load(file)
                    logger.info(f"Loaded JSON: {f} ({len(json_data_by_tokens[str(token_num)])} samples)")
            elif "mentor_only" in f:
                with open(filepath, 'r') as file:
                    json_data_by_tokens["mentor_only"] = json.load(file)
                logger.info(f"Loaded JSON: {f}")

    # Get problems from tokens0 or first available
    problems = []
    reference_data = None
    for key in ["0", "tokens0", "mentor_only"]:
        if key in json_data_by_tokens:
            reference_data = json_data_by_tokens[key]
            break

    if reference_data is None and json_data_by_tokens:
        reference_data = list(json_data_by_tokens.values())[0]

    # Extract ground_truth for matching (not question, which is formatted prompt)
    ground_truths = []
    if reference_data:
        for item in reference_data:
            # Use ground_truth field which matches original dataset's solution
            ground_truths.append(item.get("ground_truth", "").strip())
        logger.info(f"Loaded {len(ground_truths)} samples")

    return data, ground_truths, json_data_by_tokens


def split_data_by_subset(
    data: Dict[int, Dict],
    ground_truths: List[str],
    metadata: Dict[str, Dict],
    output_dir: str,
    json_data_by_tokens: Dict[str, List] = None,
):
    """Split data by subset and train/test."""

    # Create index mapping using ground_truth (solution) for matching
    subset_split_indices = defaultdict(lambda: defaultdict(list))
    unmatched = []

    for idx, solution in enumerate(ground_truths):
        if solution in metadata:
            info = metadata[solution]
            subset_split_indices[info["subset"]][info["split"]].append(idx)
        else:
            # Try partial match (first 200 chars) for solutions
            matched = False
            for key in metadata:
                if len(solution) > 100 and len(key) > 100 and solution[:200] == key[:200]:
                    info = metadata[key]
                    subset_split_indices[info["subset"]][info["split"]].append(idx)
                    matched = True
                    break
            if not matched:
                # Try shorter match (first 100 chars)
                for key in metadata:
                    if len(solution) > 50 and len(key) > 50 and solution[:100] == key[:100]:
                        info = metadata[key]
                        subset_split_indices[info["subset"]][info["split"]].append(idx)
                        matched = True
                        break
            if not matched:
                unmatched.append(idx)

    logger.info(f"Unmatched problems: {len(unmatched)}")

    # Print statistics
    logger.info("\nData distribution:")
    total_train = 0
    total_test = 0
    for subset in SUBSETS:
        train_count = len(subset_split_indices[subset]["train"])
        test_count = len(subset_split_indices[subset]["test"])
        total_train += train_count
        total_test += test_count
        logger.info(f"  {subset}: train={train_count}, test={test_count}")
    logger.info(f"  Total: train={total_train}, test={total_test}")

    # Save split data
    os.makedirs(output_dir, exist_ok=True)

    has_hidden_states = len(data) > 0

    if has_hidden_states:
        # Save by subset
        for subset in SUBSETS:
            subset_dir = os.path.join(output_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)

            for split in ["train", "test"]:
                indices = subset_split_indices[subset][split]
                if not indices:
                    continue

                split_dir = os.path.join(subset_dir, split)
                os.makedirs(split_dir, exist_ok=True)

                for tokens in TOKEN_LEVELS:
                    if tokens not in data:
                        continue

                    split_hidden = data[tokens]["hidden_states"][indices]
                    split_labels = data[tokens]["labels"][indices]

                    save_path = os.path.join(split_dir, f"tokens{tokens}.pt")
                    torch.save({
                        "hidden_states": split_hidden,
                        "labels": split_labels,
                        "indices": indices,
                    }, save_path)

                logger.info(f"Saved {subset}/{split}: {len(indices)} samples")

        # Save combined train/test
        for split in ["train", "test"]:
            all_indices = []
            for subset in SUBSETS:
                all_indices.extend(subset_split_indices[subset][split])

            if not all_indices:
                continue

            split_dir = os.path.join(output_dir, f"all_{split}")
            os.makedirs(split_dir, exist_ok=True)

            for tokens in TOKEN_LEVELS:
                if tokens not in data:
                    continue

                split_hidden = data[tokens]["hidden_states"][all_indices]
                split_labels = data[tokens]["labels"][all_indices]

                save_path = os.path.join(split_dir, f"tokens{tokens}.pt")
                torch.save({
                    "hidden_states": split_hidden,
                    "labels": split_labels,
                    "indices": all_indices,
                }, save_path)

            logger.info(f"Saved all_{split}: {len(all_indices)} samples")
    else:
        logger.info("No hidden states found, saving only index mapping")

    # Save JSON data if available
    if json_data_by_tokens:
        logger.info("\nSaving split JSON files...")

        # Save by subset
        for subset in SUBSETS:
            for split in ["train", "test"]:
                indices = subset_split_indices[subset][split]
                if not indices:
                    continue

                subset_split_dir = os.path.join(output_dir, subset, split)
                os.makedirs(subset_split_dir, exist_ok=True)

                for token_key, json_data in json_data_by_tokens.items():
                    split_json = [json_data[i] for i in indices]
                    save_path = os.path.join(subset_split_dir, f"tokens{token_key}.json")
                    with open(save_path, 'w') as f:
                        json.dump(split_json, f, indent=2, ensure_ascii=False)

                logger.info(f"Saved JSON for {subset}/{split}: {len(indices)} samples")

        # Save combined train/test
        for split in ["train", "test"]:
            all_indices = []
            for subset in SUBSETS:
                all_indices.extend(subset_split_indices[subset][split])

            if not all_indices:
                continue

            split_dir = os.path.join(output_dir, f"all_{split}")
            os.makedirs(split_dir, exist_ok=True)

            for token_key, json_data in json_data_by_tokens.items():
                split_json = [json_data[i] for i in all_indices]
                save_path = os.path.join(split_dir, f"tokens{token_key}.json")
                with open(save_path, 'w') as f:
                    json.dump(split_json, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved JSON for all_{split}: {len(all_indices)} samples")

    # Save index mapping for reference
    index_map = {
        "subsets": {
            subset: {
                split: subset_split_indices[subset][split]
                for split in ["train", "test"]
            }
            for subset in SUBSETS
        },
        "unmatched": unmatched,
    }

    with open(os.path.join(output_dir, "index_mapping.json"), 'w') as f:
        json.dump(index_map, f, indent=2)

    logger.info(f"\nSaved index mapping to {output_dir}/index_mapping.json")


def main():
    parser = argparse.ArgumentParser(description="Split data by hendrycks_math subsets")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with collected data (containing hidden_states/ and json files)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for split data")

    args = parser.parse_args()

    # Load metadata from original dataset
    metadata = load_hendrycks_math_metadata()

    # Load collected data
    data, ground_truths, json_data_by_tokens = load_collected_data(args.data_dir)

    if not ground_truths:
        logger.error("Could not load ground truth data. Please check data directory.")
        return

    # Split and save
    split_data_by_subset(data, ground_truths, metadata, args.output_dir, json_data_by_tokens)

    logger.info("Done!")


if __name__ == "__main__":
    main()
