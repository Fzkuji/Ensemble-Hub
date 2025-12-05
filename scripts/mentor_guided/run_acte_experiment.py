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


def evaluate_adaptive_strategy_with_threshold(
    classifier: ClassifierTrainer,
    test_data_by_tokens: Dict[int, List[Dict]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate the adaptive strategy using trained binary classifier with CASCADE logic and adjustable threshold.

    Cascade Strategy:
    1. At 100 tokens: predict if sufficient → if yes, use 100 tokens result
    2. At 500 tokens: predict if sufficient → if yes, use 500 tokens result
    3. At 1000 tokens: always use 1000 tokens (final fallback)

    Args:
        classifier: Trained binary classifier
        test_data_by_tokens: Data organized by token levels
        threshold: Classification threshold (higher = stricter, fewer "sufficient" predictions)

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
        # Cascade decision: check 100 → 500 → fallback to 1000
        selected_tokens = 1000  # Default fallback

        for tokens in [100, 500]:
            if tokens not in token_items:
                continue

            item = token_items[tokens]
            ppl = item.get('ppl', [])
            entropy = item.get('entropy', [])

            if not ppl or not entropy or len(ppl) < 10:
                continue

            # Binary prediction with threshold: 1 = sufficient, 0 = need more
            predicted_sufficient = classifier.predict(ppl, entropy, threshold=threshold)

            if predicted_sufficient == 1:
                selected_tokens = tokens
                break

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
            'question': question,
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


def run_cross_validation(
    all_data_by_tokens: Dict[int, List[Dict]],
    model_type: str = "mlp",
    n_folds: int = 5,
    thresholds: List[float] = None,
) -> Dict[str, Any]:
    """
    Run K-fold cross-validation with multiple thresholds.

    Args:
        all_data_by_tokens: All collected data organized by token levels
        model_type: Type of classifier to use
        n_folds: Number of CV folds
        thresholds: List of thresholds to evaluate

    Returns:
        Cross-validation results for each threshold
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Build question list (questions that have data for all required token levels)
    all_questions = set()
    for tokens, items in all_data_by_tokens.items():
        for item in items:
            all_questions.add(item['question'])
    all_questions = list(all_questions)

    # Shuffle questions for random split
    np.random.seed(42)
    np.random.shuffle(all_questions)

    # Create folds
    fold_size = len(all_questions) // n_folds
    folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        if i == n_folds - 1:
            # Last fold gets remaining questions
            end_idx = len(all_questions)
        else:
            end_idx = start_idx + fold_size
        folds.append(set(all_questions[start_idx:end_idx]))

    logger.info(f"Cross-validation: {n_folds} folds, {len(all_questions)} questions total")
    logger.info(f"Fold sizes: {[len(f) for f in folds]}")

    # Results for each threshold
    threshold_results = {t: [] for t in thresholds}

    for fold_idx in range(n_folds):
        logger.info(f"\n=== Fold {fold_idx + 1}/{n_folds} ===")

        # Split data: current fold = test, rest = train
        test_questions = folds[fold_idx]
        train_questions = set()
        for i in range(n_folds):
            if i != fold_idx:
                train_questions.update(folds[i])

        # Create train/test data by tokens
        train_data_by_tokens = {}
        test_data_by_tokens = {}

        for tokens, items in all_data_by_tokens.items():
            train_data_by_tokens[tokens] = [item for item in items if item['question'] in train_questions]
            test_data_by_tokens[tokens] = [item for item in items if item['question'] in test_questions]

        # Prepare classifier training data
        classifier_train_data = prepare_classifier_data(train_data_by_tokens)

        if len(classifier_train_data) < 30:
            logger.warning(f"Fold {fold_idx + 1}: Not enough training data, skipping")
            continue

        # Split training data for internal validation (80/20)
        np.random.shuffle(classifier_train_data)
        split_idx = int(0.8 * len(classifier_train_data))
        train_split = classifier_train_data[:split_idx]
        val_split = classifier_train_data[split_idx:]

        # Train classifier
        trainer = ClassifierTrainer(
            model_type=model_type,
            hidden_dim=64,
            num_classes=2,
            dropout=0.2,
        )

        history = trainer.train(
            train_split, val_split,
            epochs=100,
            batch_size=32,
            early_stopping_patience=15,
        )

        # Evaluate with different thresholds (fallback to 1000 tokens)
        for threshold in thresholds:
            results = evaluate_adaptive_strategy_with_threshold(
                trainer, test_data_by_tokens, threshold=threshold
            )
            threshold_results[threshold].append({
                'fold': fold_idx + 1,
                'accuracy': results['accuracy'],
                'avg_mentor_len': results['avg_mentor_length'],
                'token_distribution': results['token_distribution'],
                'num_samples': results['num_samples'],
            })

    # Aggregate results across folds
    summary = {}
    for threshold in thresholds:
        fold_results = threshold_results[threshold]
        if not fold_results:
            continue

        accuracies = [r['accuracy'] for r in fold_results]
        mentor_lens = [r['avg_mentor_len'] for r in fold_results]

        # Aggregate token distribution
        total_dist = {}
        for r in fold_results:
            for t, count in r['token_distribution'].items():
                total_dist[t] = total_dist.get(t, 0) + count

        summary[threshold] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_mentor_len': np.mean(mentor_lens),
            'std_mentor_len': np.std(mentor_lens),
            'token_distribution': total_dist,
            'fold_accuracies': accuracies,
        }

    return summary


def run_experiment(args):
    """Run the complete experiment."""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "acte_experiments")

    # Try to load from "all" directory first (new format without pre-split)
    all_data_dir = os.path.join(data_dir, "collected", f"{args.dataset}_all_{args.mentor}")

    if os.path.exists(all_data_dir):
        # New format: single "all" directory
        logger.info(f"Loading data from: {all_data_dir}")
        all_data = load_collected_data(all_data_dir, args.dataset, "all")
    else:
        # Legacy format: combine train and test directories
        train_data_dir = os.path.join(data_dir, "collected", f"{args.dataset}_train_{args.mentor}")
        test_data_dir = os.path.join(data_dir, "collected", f"{args.dataset}_test_{args.mentor}")

        if not os.path.exists(train_data_dir):
            logger.error(f"Data not found. Tried:")
            logger.error(f"  - {all_data_dir}")
            logger.error(f"  - {train_data_dir}")
            logger.info("Please run collect_progressive_data.py first")
            return

        logger.info(f"Loading legacy format (train + test)...")
        train_data = load_collected_data(train_data_dir, args.dataset, "train")
        test_data = load_collected_data(test_data_dir, args.dataset, "test")

        # Combine all data for cross-validation
        all_data = {}
        for tokens in [-1, 0, 100, 500, 1000]:
            all_data[tokens] = []
            if tokens in train_data:
                all_data[tokens].extend(train_data[tokens])
            if tokens in test_data:
                all_data[tokens].extend(test_data[tokens])

    # Count total questions
    all_questions = set()
    for tokens, items in all_data.items():
        for item in items:
            all_questions.add(item['question'])
    logger.info(f"\n=== Total questions: {len(all_questions)} ===")

    # Evaluate baselines on ALL data
    logger.info("\n=== Baselines on All Data ===")
    baselines = evaluate_baselines(all_data)
    for baseline in baselines:
        logger.info(f"{baseline['model']}: Acc={baseline['accuracy']:.4f}, "
                   f"Mentor={baseline['avg_mentor_len']:.1f}, Intern={baseline['avg_intern_len']:.1f}")

    # Run cross-validation with multiple thresholds
    logger.info(f"\n=== Running {args.n_folds}-Fold Cross-Validation ===")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for model_type in args.models:
        logger.info(f"\n=== {model_type.upper()} Classifier ===")

        cv_results = run_cross_validation(
            all_data,
            model_type=model_type,
            n_folds=args.n_folds,
            thresholds=thresholds,
        )

        # Print results table
        logger.info(f"\n{model_type.upper()} Cross-Validation Results:")
        logger.info(f"{'Threshold':<10} {'Mean Acc':<12} {'Std Acc':<10} {'Avg Mentor':<12} {'Distribution'}")
        logger.info("-" * 80)

        for threshold in thresholds:
            if threshold not in cv_results:
                continue
            r = cv_results[threshold]
            dist_str = ", ".join([f"{k}:{v}" for k, v in sorted(r['token_distribution'].items())])
            logger.info(
                f"{threshold:<10.2f} {r['mean_accuracy']:<12.4f} {r['std_accuracy']:<10.4f} "
                f"{r['mean_mentor_len']:<12.1f} {dist_str}"
            )

        # Save detailed results
        results_path = os.path.join(
            data_dir, "results",
            f"{args.dataset}_{args.mentor}_{model_type}_cv_results.json"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        # Convert numpy types for JSON serialization
        cv_results_serializable = {}
        for t, r in cv_results.items():
            cv_results_serializable[str(t)] = {
                'mean_accuracy': float(r['mean_accuracy']),
                'std_accuracy': float(r['std_accuracy']),
                'mean_mentor_len': float(r['mean_mentor_len']),
                'std_mentor_len': float(r['std_mentor_len']),
                'token_distribution': {str(k): int(v) for k, v in r['token_distribution'].items()},
                'fold_accuracies': [float(x) for x in r['fold_accuracies']],
            }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_type,
                'n_folds': args.n_folds,
                'baselines': baselines,
                'cv_results': cv_results_serializable,
            }, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")

    # Print comparison with baselines
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Comparing Adaptive Methods with Baselines")
    logger.info("=" * 80)
    logger.info(f"\n{'Method':<20} {'Accuracy':<12} {'Avg Mentor Tokens'}")
    logger.info("-" * 50)
    for baseline in baselines:
        logger.info(f"{baseline['model']:<20} {baseline['accuracy']:.4f}       {baseline['avg_mentor_len']:.1f}")

    logger.info("\n(See detailed threshold results above for adaptive methods)")


def main():
    parser = argparse.ArgumentParser(description="Run ACT-E experiment")
    parser.add_argument("--dataset", type=str, default="hendrycks_math",
                        choices=["hendrycks_math", "math500", "humaneval"])
    parser.add_argument("--mentor", type=str, default="DeepSeek-R1-Distill-Qwen-32B",
                        help="Mentor model name (for path)")
    parser.add_argument("--models", nargs="+", default=["mlp"],
                        help="Classifier models to train (lstm, gru, mlp, attention, cnn)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
