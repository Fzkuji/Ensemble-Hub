#!/usr/bin/env python3
"""
Run ACT-E Experiments

Main script to run the complete ACT-E experiment pipeline:
1. Collect data with different mentor token lengths
2. Train classifiers (LSTM, GRU, MLP)
3. Evaluate on test set
4. Generate results for paper tables
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequence_classifier import ClassifierTrainer, extract_statistical_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_collected_data(data_dir: str, dataset: str, split: str) -> Dict[int, List[Dict]]:
    """Load collected data for all token lengths."""
    data = {}
    for tokens in [0, 100, 500, 1000]:
        file_path = os.path.join(data_dir, f"{dataset}_{split}_tokens{tokens}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data[tokens] = json.load(f)
            logger.info(f"Loaded {len(data[tokens])} samples for {tokens} tokens")
    return data


def prepare_classifier_data(
    data_by_tokens: Dict[int, List[Dict]],
) -> List[Dict]:
    """
    Prepare data for classifier training.

    For each sample at each token level, determine whether that level is sufficient:
    - Use PPL/Entropy from that token level
    - Label is the optimal strategy (smallest sufficient token count)

    Labels:
    - 0: Intern only (no mentor help needed)
    - 1: 100 tokens sufficient
    - 2: 500 tokens sufficient
    - 3: 1000 tokens needed (or nothing works)
    """
    classifier_data = []

    # Build question -> results mapping
    question_results = {}

    for tokens, items in data_by_tokens.items():
        for item in items:
            question = item['question']
            if question not in question_results:
                question_results[question] = {}
            question_results[question][tokens] = {
                'is_correct': item['is_correct'],
                'ppl': item.get('ppl', []),
                'entropy': item.get('entropy', []),
            }

    # For each question, determine optimal strategy and create training samples
    for question, token_data in question_results.items():
        # Determine optimal strategy
        results = {t: d['is_correct'] for t, d in token_data.items()}

        optimal_label = 3  # Default: need 1000 tokens
        if results.get(0, False):
            optimal_label = 0  # Intern alone is sufficient
        elif results.get(100, False):
            optimal_label = 1  # 100 tokens sufficient
        elif results.get(500, False):
            optimal_label = 2  # 500 tokens sufficient
        elif results.get(1000, False):
            optimal_label = 3  # 1000 tokens needed
        # else: none work, keep default 3

        # Create training samples from each token level that has PPL/Entropy data
        # We use the PPL/Entropy to predict whether MORE help is needed
        for tokens in [100, 500, 1000]:
            if tokens in token_data:
                ppl = token_data[tokens].get('ppl', [])
                entropy = token_data[tokens].get('entropy', [])

                if ppl and entropy and len(ppl) >= 10:
                    # Binary label: is this token level sufficient?
                    # For multi-class: use the optimal label
                    classifier_data.append({
                        'ppl': ppl,
                        'entropy': entropy,
                        'label': optimal_label,
                        'question': question,
                        'token_level': tokens,
                    })

    logger.info(f"Prepared {len(classifier_data)} samples for classifier training")

    # Label distribution
    label_counts = {}
    for item in classifier_data:
        label_counts[item['label']] = label_counts.get(item['label'], 0) + 1
    logger.info(f"Label distribution: {label_counts}")

    return classifier_data


def evaluate_adaptive_strategy(
    classifier: ClassifierTrainer,
    test_data_by_tokens: Dict[int, List[Dict]],
) -> Dict[str, Any]:
    """
    Evaluate the adaptive strategy using trained classifier.

    Strategy: Use PPL/Entropy from 100-token level to predict optimal strategy.
    Then execute with predicted token level.

    Returns:
        Results including accuracy, average tokens, and cost.
    """
    token_map = {0: 0, 1: 100, 2: 500, 3: 1000}
    results = []

    # Build question -> all token data mapping
    question_data = {}
    for tokens, items in test_data_by_tokens.items():
        for item in items:
            question = item['question']
            if question not in question_data:
                question_data[question] = {}
            question_data[question][tokens] = item

    for question, token_items in question_data.items():
        # Use 100-token PPL/Entropy for prediction (first checkpoint)
        if 100 not in token_items:
            continue

        item_100 = token_items[100]
        ppl = item_100.get('ppl', [])
        entropy = item_100.get('entropy', [])

        if not ppl or not entropy or len(ppl) < 10:
            continue

        # Predict optimal strategy
        predicted_label = classifier.predict(ppl, entropy)
        predicted_tokens = token_map[predicted_label]

        # Get actual result for predicted token count
        actual_correct = False
        actual_mentor_len = 0
        actual_intern_len = 0

        if predicted_tokens == 0:
            # Use intern only - check 0 token result
            if 0 in token_items:
                actual_correct = token_items[0]['is_correct']
                actual_mentor_len = 0
                actual_intern_len = token_items[0].get('intern_length', 0)
        elif predicted_tokens in token_items:
            item = token_items[predicted_tokens]
            actual_correct = item['is_correct']
            actual_mentor_len = item.get('mentor_length', predicted_tokens)
            actual_intern_len = item.get('intern_length', 0)

        results.append({
            'predicted_tokens': predicted_tokens,
            'is_correct': actual_correct,
            'mentor_length': actual_mentor_len,
            'intern_length': actual_intern_len,
        })

    if not results:
        return {
            'accuracy': 0,
            'avg_mentor_length': 0,
            'avg_intern_length': 0,
            'num_samples': 0,
            'results': [],
        }

    # Calculate metrics
    accuracy = np.mean([r['is_correct'] for r in results])
    avg_mentor_len = np.mean([r['mentor_length'] for r in results])
    avg_intern_len = np.mean([r['intern_length'] for r in results])

    return {
        'accuracy': accuracy,
        'avg_mentor_length': avg_mentor_len,
        'avg_intern_length': avg_intern_len,
        'num_samples': len(results),
        'results': results,
    }


def calculate_tflops(mentor_len: float, intern_len: float, mentor_params: float = 32e9, intern_params: float = 7e9) -> float:
    """
    Calculate TFLOPs cost.

    Approximate: 2 * params * tokens for forward pass
    """
    mentor_flops = 2 * mentor_params * mentor_len
    intern_flops = 2 * intern_params * intern_len
    total_tflops = (mentor_flops + intern_flops) / 1e12
    return total_tflops


def run_experiment(args):
    """Run the complete experiment."""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "acte_experiments")

    # Load collected training data
    train_data_dir = os.path.join(data_dir, "collected", f"{args.dataset}_train_{args.mentor}")
    test_data_dir = os.path.join(data_dir, "collected", f"{args.dataset}_test_{args.mentor}")

    if not os.path.exists(train_data_dir):
        logger.error(f"Training data not found: {train_data_dir}")
        logger.info("Please run collect_progressive_data.py first")
        return

    train_data = load_collected_data(train_data_dir, args.dataset, "train")
    test_data = load_collected_data(test_data_dir, args.dataset, "test")

    # Prepare classifier data
    classifier_train_data = prepare_classifier_data(train_data)

    if len(classifier_train_data) < 50:
        logger.error("Not enough training data")
        return

    # Split into train/val
    np.random.seed(42)
    np.random.shuffle(classifier_train_data)
    split_idx = int(0.8 * len(classifier_train_data))
    train_split = classifier_train_data[:split_idx]
    val_split = classifier_train_data[split_idx:]

    # Train classifiers
    results_table = []

    for model_type in args.models:
        logger.info(f"\n=== Training {model_type.upper()} Classifier ===")

        trainer = ClassifierTrainer(
            model_type=model_type,
            hidden_dim=64,
            num_classes=4,
            dropout=0.2,
        )

        history = trainer.train(
            train_split, val_split,
            epochs=100,
            batch_size=32,
            early_stopping_patience=15,
        )

        # Evaluate on test set
        if test_data:
            test_results = evaluate_adaptive_strategy(trainer, test_data)

            # Calculate cost
            tflops = calculate_tflops(
                test_results['avg_mentor_length'],
                test_results['avg_intern_length'],
            )

            results_table.append({
                'model': model_type.upper(),
                'accuracy': test_results['accuracy'],
                'avg_mentor_len': test_results['avg_mentor_length'],
                'avg_intern_len': test_results['avg_intern_length'],
                'tflops': tflops,
            })

            logger.info(f"{model_type.upper()} Test Results:")
            logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
            logger.info(f"  Avg Mentor Length: {test_results['avg_mentor_length']:.1f}")
            logger.info(f"  Avg Intern Length: {test_results['avg_intern_length']:.1f}")
            logger.info(f"  TFLOPs: {tflops:.2f}")

        # Save model
        model_save_path = os.path.join(data_dir, "models", f"{args.dataset}_{args.mentor}_{model_type}.pt")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        trainer.save(model_save_path)

    # Print results table
    logger.info("\n=== Results Table ===")
    logger.info(f"{'Model':<10} {'Accuracy':<10} {'Mentor Len':<12} {'Intern Len':<12} {'TFLOPs':<10}")
    logger.info("-" * 54)
    for row in results_table:
        logger.info(f"{row['model']:<10} {row['accuracy']:.4f}     {row['avg_mentor_len']:<12.1f} {row['avg_intern_len']:<12.1f} {row['tflops']:.2f}")

    # Save results
    results_path = os.path.join(data_dir, "results", f"{args.dataset}_{args.mentor}_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_table, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ACT-E experiment")
    parser.add_argument("--dataset", type=str, default="math500", choices=["math500", "humaneval"])
    parser.add_argument("--mentor", type=str, default="DeepSeek-R1-Distill-Qwen-32B",
                        help="Mentor model name (for path)")
    parser.add_argument("--models", nargs="+", default=["lstm", "gru", "mlp"],
                        help="Classifier models to train")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
