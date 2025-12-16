#!/usr/bin/env python3
"""
Evaluate cascaded classifier on each hendrycks_math subset.
Focus on finding optimal accuracy only.
"""

import argparse
import json
import logging
import os
from typing import Dict, List
import numpy as np
import torch
from itertools import product

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def load_subset_data(data_dir: str, subset: str, split: str = "test") -> Dict[int, Dict]:
    """Load hidden states or JSON data for a specific subset."""
    data = {}
    subset_dir = os.path.join(data_dir, subset, split)

    if not os.path.exists(subset_dir):
        logger.warning(f"Directory not found: {subset_dir}")
        return data

    # Try to load .pt files first (with hidden states)
    has_pt = False
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(subset_dir, f"tokens{tokens}.pt")
        if os.path.exists(filepath):
            loaded = torch.load(filepath)
            data[tokens] = {
                'hidden_states': loaded['hidden_states'],
                'labels': loaded['labels'],
            }
            has_pt = True

    # If no .pt files, load labels from JSON
    if not has_pt:
        for tokens in TOKEN_LEVELS:
            filepath = os.path.join(subset_dir, f"tokens{tokens}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                labels = torch.tensor([1 if item.get('is_correct', False) else 0 for item in json_data])
                data[tokens] = {
                    'hidden_states': None,
                    'labels': labels,
                }

    return data


def compute_oracle_accuracy(data: Dict[int, Dict]) -> float:
    """Compute oracle accuracy (best possible if we pick optimal tokens per sample)."""
    if not data:
        return 0.0

    n_samples = len(data[TOKEN_LEVELS[0]]['labels'])
    oracle_correct = 0

    for i in range(n_samples):
        # Check if any token level gives correct answer
        for tokens in TOKEN_LEVELS:
            if data[tokens]['labels'][i] == 1:
                oracle_correct += 1
                break

    return oracle_correct / n_samples


def train_and_eval_xgboost(
    train_data: Dict[int, Dict],
    test_data: Dict[int, Dict],
) -> Dict:
    """Train XGBoost on train set, evaluate on test set."""

    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed")

    n_train = len(train_data[TOKEN_LEVELS[0]]['labels'])
    n_test = len(test_data[TOKEN_LEVELS[0]]['labels'])

    # Train per-stage models
    stage_models = {}

    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        X_train = train_data[tokens]['hidden_states'].numpy()
        y_train = train_data[tokens]['labels'].numpy()

        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)

        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train, verbose=False)
        stage_models[stage_idx] = model

    # Evaluate cascade on test set
    gt = {tokens: test_data[tokens]['labels'].numpy() for tokens in TOKEN_LEVELS}

    # Search for best thresholds
    threshold_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_acc = 0
    best_thresholds = None

    for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
        thresholds = list(combo)

        final_correct = np.zeros(n_test, dtype=bool)

        for i in range(n_test):
            # Track probabilities for fallback selection
            stage_probs = []
            decided = False

            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                hidden = test_data[tokens]['hidden_states'][i:i+1].numpy()
                if hidden.ndim == 3:
                    hidden = hidden.reshape(1, -1)

                proba = stage_models[stage_idx].predict_proba(hidden)
                prob_sufficient = proba[0, 1]
                stage_probs.append((tokens, prob_sufficient))

                if prob_sufficient >= thresholds[stage_idx]:
                    final_correct[i] = gt[tokens][i] == 1
                    decided = True
                    break

            if not decided:
                # No stage passed threshold, select the one with highest confidence
                best_tokens, _ = max(stage_probs, key=lambda x: x[1])
                final_correct[i] = gt[best_tokens][i] == 1

        acc = final_correct.mean()
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    return {
        'best_accuracy': float(best_acc),
        'best_thresholds': best_thresholds,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on each subset")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with split data (containing subset folders)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    results = {}

    # Baseline accuracies for each token level
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating each subset...")
    logger.info("=" * 60)

    for subset in SUBSETS:
        logger.info(f"\n--- {subset} ---")

        # Load train and test data
        train_data = load_subset_data(args.data_dir, subset, "train")
        test_data = load_subset_data(args.data_dir, subset, "test")

        if not test_data:
            logger.warning(f"No test data for {subset}, skipping")
            continue

        n_test = len(test_data[TOKEN_LEVELS[0]]['labels'])
        logger.info(f"Test samples: {n_test}")

        # Baseline: accuracy at each token level
        baseline_acc = {}
        for tokens in TOKEN_LEVELS:
            if tokens in test_data:
                acc = test_data[tokens]['labels'].float().mean().item()
                baseline_acc[tokens] = acc
                logger.info(f"  Tokens {tokens}: {acc:.4f}")

        # Oracle accuracy
        oracle_acc = compute_oracle_accuracy(test_data)
        logger.info(f"  Oracle: {oracle_acc:.4f}")

        # Train and evaluate XGBoost (only if hidden states available)
        has_hidden = train_data and train_data.get(TOKEN_LEVELS[0], {}).get('hidden_states') is not None
        if has_hidden and HAS_XGBOOST:
            n_train = len(train_data[TOKEN_LEVELS[0]]['labels'])
            logger.info(f"Training XGBoost on {n_train} samples...")

            xgb_result = train_and_eval_xgboost(train_data, test_data)
            logger.info(f"  XGBoost Best: {xgb_result['best_accuracy']:.4f} (thresholds: {xgb_result['best_thresholds']})")

            results[subset] = {
                'n_test': n_test,
                'baseline': baseline_acc,
                'oracle': oracle_acc,
                'xgboost': xgb_result,
            }
        else:
            if not has_hidden:
                logger.info("  (No hidden states, skipping XGBoost)")
            results[subset] = {
                'n_test': n_test,
                'baseline': baseline_acc,
                'oracle': oracle_acc,
            }

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    header = f"{'Subset':<25} {'T0':<8} {'T100':<8} {'T500':<8} {'T1000':<8} {'Oracle':<8} {'XGB':<8}"
    logger.info(header)
    logger.info("-" * 80)

    for subset in SUBSETS:
        if subset not in results:
            continue
        r = results[subset]
        baseline = r['baseline']
        t0 = baseline.get(0, 0)
        t100 = baseline.get(100, 0)
        t500 = baseline.get(500, 0)
        t1000 = baseline.get(1000, 0)
        oracle = r['oracle']
        xgb_acc = r.get('xgboost', {}).get('best_accuracy', 0)

        logger.info(f"{subset:<25} {t0:<8.4f} {t100:<8.4f} {t500:<8.4f} {t1000:<8.4f} {oracle:<8.4f} {xgb_acc:<8.4f}")

    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
