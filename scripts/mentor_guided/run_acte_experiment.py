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
from sequence_classifier import ClassifierTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_collected_data(data_dir: str, dataset: str, split: str) -> Dict[int, List[Dict]]:
    """Load collected data for all token lengths.

    Token lengths:
        -1: Mentor only (mentor generates complete response)
         0: Intern only (no mentor)
        100, 500, 1000: Progressive (mentor prefix + intern continuation)
    """
    data = {}

    # Load mentor only (-1)
    mentor_only_path = os.path.join(data_dir, f"{dataset}_{split}_mentor_only.json")
    if os.path.exists(mentor_only_path):
        with open(mentor_only_path, 'r', encoding='utf-8') as f:
            data[-1] = json.load(f)
        logger.info(f"Loaded {len(data[-1])} samples for mentor only")

    # Load other token lengths
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
    Prepare data for binary classifier training.

    For each sample at each token level, determine whether that level is sufficient.
    This is a BINARY classification: 1 = current level is sufficient, 0 = need more/fallback.

    The classifier will be used in a cascade manner:
    - At 100 tokens: predict if 100 is enough
    - At 500 tokens: predict if 500 is enough
    - At 1000 tokens: predict if 1000 is enough
    - If all predict "not enough", fallback to intern-only result
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

    # For each question, create binary classification samples for each token level
    for question, token_data in question_results.items():
        results = {t: d['is_correct'] for t, d in token_data.items()}

        # Create training samples for each token level
        # Binary label: 1 = this token level gives correct answer, 0 = incorrect
        for tokens in [100, 500, 1000]:
            if tokens in token_data:
                ppl = token_data[tokens].get('ppl', [])
                entropy = token_data[tokens].get('entropy', [])

                if ppl and entropy and len(ppl) >= 10:
                    # Binary label: is this token level sufficient (correct)?
                    is_sufficient = 1 if results.get(tokens, False) else 0

                    classifier_data.append({
                        'ppl': ppl,
                        'entropy': entropy,
                        'label': is_sufficient,
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
    Evaluate the adaptive strategy using trained binary classifier with CASCADE logic.

    Cascade Strategy:
    1. At 100 tokens: predict if sufficient → if yes, use 100 tokens result
    2. At 500 tokens: predict if sufficient → if yes, use 500 tokens result
    3. At 1000 tokens: predict if sufficient → if yes, use 1000 tokens result
    4. If all predict "not sufficient", fallback to intern-only (0 tokens) result

    Returns:
        Results including accuracy, average tokens, and cost.
    """
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
        # Cascade decision: check 100 → 500 → 1000 → fallback to 0
        selected_tokens = None

        for tokens in [100, 500, 1000]:
            if tokens not in token_items:
                continue

            item = token_items[tokens]
            ppl = item.get('ppl', [])
            entropy = item.get('entropy', [])

            if not ppl or not entropy or len(ppl) < 10:
                continue

            # Binary prediction: 1 = sufficient, 0 = need more
            predicted_sufficient = classifier.predict(ppl, entropy)

            if predicted_sufficient == 1:
                selected_tokens = tokens
                break

        # If no level was predicted as sufficient, fallback to intern-only
        if selected_tokens is None:
            selected_tokens = 0

        # Get actual result for selected token count
        actual_correct = False
        actual_mentor_len = 0
        actual_intern_len = 0

        if selected_tokens in token_items:
            item = token_items[selected_tokens]
            actual_correct = item['is_correct']
            actual_mentor_len = item.get('mentor_length', selected_tokens if selected_tokens > 0 else 0)
            actual_intern_len = item.get('intern_length', 0)

        results.append({
            'selected_tokens': selected_tokens,
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
            'token_distribution': {},
            'results': [],
        }

    # Calculate metrics
    accuracy = np.mean([r['is_correct'] for r in results])
    avg_mentor_len = np.mean([r['mentor_length'] for r in results])
    avg_intern_len = np.mean([r['intern_length'] for r in results])

    # Token distribution statistics
    token_distribution = {}
    for r in results:
        t = r['selected_tokens']
        token_distribution[t] = token_distribution.get(t, 0) + 1

    return {
        'accuracy': accuracy,
        'avg_mentor_length': avg_mentor_len,
        'avg_intern_length': avg_intern_len,
        'num_samples': len(results),
        'token_distribution': token_distribution,
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


def evaluate_baselines(test_data_by_tokens: Dict[int, List[Dict]]) -> List[Dict]:
    """
    Evaluate baseline strategies for comparison.

    Baselines:
    - Intern-only (0 tokens): Always use intern alone
    - Fixed-100: Always use 100 mentor tokens
    - Fixed-500: Always use 500 mentor tokens
    - Fixed-1000: Always use 1000 mentor tokens
    - Mentor-only (-1): Always use mentor alone
    - Oracle: Always choose the best option for each question
    """
    baselines = []

    # Build question -> all token data mapping
    question_data = {}
    for tokens, items in test_data_by_tokens.items():
        for item in items:
            question = item['question']
            if question not in question_data:
                question_data[question] = {}
            question_data[question][tokens] = item

    # Fixed baselines
    for fixed_tokens in [0, 100, 500, 1000, -1]:
        correct = 0
        total = 0
        total_mentor_len = 0
        total_intern_len = 0

        for question, token_items in question_data.items():
            if fixed_tokens in token_items:
                item = token_items[fixed_tokens]
                if item['is_correct']:
                    correct += 1
                total += 1
                total_mentor_len += item.get('mentor_length', 0)
                total_intern_len += item.get('intern_length', 0)

        if total > 0:
            name = "Mentor-only" if fixed_tokens == -1 else f"Intern-only" if fixed_tokens == 0 else f"Fixed-{fixed_tokens}"
            baselines.append({
                'model': name,
                'accuracy': correct / total,
                'avg_mentor_len': total_mentor_len / total,
                'avg_intern_len': total_intern_len / total,
                'tflops': calculate_tflops(total_mentor_len / total, total_intern_len / total),
            })

    # Oracle: choose best result for each question
    oracle_correct = 0
    oracle_total = 0
    oracle_mentor_len = 0
    oracle_intern_len = 0

    for question, token_items in question_data.items():
        # Priority: smallest token count that gives correct answer
        best_item = None
        for tokens in [0, 100, 500, 1000, -1]:
            if tokens in token_items and token_items[tokens]['is_correct']:
                best_item = token_items[tokens]
                break

        if best_item is None and token_items:
            # None correct, use intern-only as fallback
            best_item = token_items.get(0, list(token_items.values())[0])

        if best_item:
            if best_item['is_correct']:
                oracle_correct += 1
            oracle_total += 1
            oracle_mentor_len += best_item.get('mentor_length', 0)
            oracle_intern_len += best_item.get('intern_length', 0)

    if oracle_total > 0:
        baselines.append({
            'model': 'Oracle',
            'accuracy': oracle_correct / oracle_total,
            'avg_mentor_len': oracle_mentor_len / oracle_total,
            'avg_intern_len': oracle_intern_len / oracle_total,
            'tflops': calculate_tflops(oracle_mentor_len / oracle_total, oracle_intern_len / oracle_total),
        })

    return baselines


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

    # Evaluate baselines on train data (always available)
    logger.info("\n=== Baselines on Train Data ===")
    train_baselines = evaluate_baselines(train_data)
    for baseline in train_baselines:
        logger.info(f"{baseline['model']}: Acc={baseline['accuracy']:.4f}, "
                   f"Mentor={baseline['avg_mentor_len']:.1f}, Intern={baseline['avg_intern_len']:.1f}")

    # Evaluate baselines on test data (if exists)
    results_table = []
    if test_data:
        logger.info("\n=== Baselines on Test Data ===")
        baselines = evaluate_baselines(test_data)
        for baseline in baselines:
            logger.info(f"{baseline['model']}: Acc={baseline['accuracy']:.4f}, "
                       f"Mentor={baseline['avg_mentor_len']:.1f}, Intern={baseline['avg_intern_len']:.1f}")
        results_table.extend(baselines)

    # Train classifiers
    for model_type in args.models:
        logger.info(f"\n=== Training {model_type.upper()} Classifier ===")

        trainer = ClassifierTrainer(
            model_type=model_type,
            hidden_dim=64,
            num_classes=2,  # Binary: sufficient (1) or not (0)
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
            logger.info(f"  Token Distribution: {test_results.get('token_distribution', {})}")

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
    parser.add_argument("--models", nargs="+", default=["lstm", "gru", "mlp", "attention"],
                        help="Classifier models to train (lstm, gru, mlp, attention, cnn)")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
