#!/usr/bin/env python3
"""
Routing baseline evaluation for Tandem rebuttal.

Compares Tandem's cascade approach against simple routing baselines:

1. Binary Routing (T0 vs T1000): Classifier decides per-problem whether SLM
   alone (T0) is sufficient or needs maximum guidance (T1000). Skips intermediate
   stages. Shows the value of Tandem's multi-stage cascade.

2. LLM Routing (7B vs 32B): Classifier decides whether to use SLM (7B standalone)
   or LLM (32B standalone) per problem. No mentor-intern collaboration.
   Shows the value of Tandem's structured knowledge distillation.

Usage:
    # Binary routing (uses existing Tandem data, needs GPU for feature extraction)
    python eval_routing_baseline.py --data-dir /path/to/hendrycks_math_split \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

    # LLM routing (also needs 32B standalone results from lm-eval)
    python eval_routing_baseline.py --data-dir /path/to/hendrycks_math_split \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --llm-results-dir /path/to/results/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B

    # Use saved features (skip GPU feature extraction)
    python eval_routing_baseline.py --data-dir /path/to/hendrycks_math_split \
        --load-features /path/to/saved_features.npz
"""

import argparse
import json
import logging
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]

DEFAULT_SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

# Paper Table 1 cost data (TFLOPs)
COST_TABLE = {
    0: 38.25,       # SLM (7B) standalone
    100: 44.76,     # 7B+32B (low)
    500: 71.96,     # 7B+32B (medium)
    1000: 104.62,   # 7B+32B (high)
    "32B": 168.35,  # 32B standalone
}


def detect_subsets(data_dir: str, split: str = "test") -> List[str]:
    subsets = []
    for name in os.listdir(data_dir):
        subset_dir = os.path.join(data_dir, name, split)
        if os.path.isdir(subset_dir):
            token_file = os.path.join(subset_dir, "tokens0.json")
            if os.path.exists(token_file):
                subsets.append(name)
    return sorted(subsets) if subsets else DEFAULT_SUBSETS


def load_json_data(data_dir: str, split: str = "test") -> Dict[int, List[Dict]]:
    data = {}
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        split_dir = data_dir
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(split_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
            logger.info(f"Loaded {len(data[tokens])} samples from {filepath}")
    return data


def compute_trend_stats(values: np.ndarray) -> Dict[str, float]:
    n = len(values)
    if n < 2:
        return {
            'slope': 0.0, 'increase_ratio': 0.5, 'decrease_ratio': 0.5,
            'last_quarter_mean': float(values[0]) if n > 0 else 0.0,
            'first_quarter_mean': float(values[0]) if n > 0 else 0.0,
            'trend_change': 0.0,
        }
    x = np.arange(n)
    slope = np.polyfit(x, values, 1)[0]
    diffs = np.diff(values)
    increase_ratio = np.mean(diffs > 0)
    decrease_ratio = np.mean(diffs < 0)
    quarter = max(1, n // 4)
    first_quarter_mean = np.mean(values[:quarter])
    last_quarter_mean = np.mean(values[-quarter:])
    trend_change = last_quarter_mean - first_quarter_mean
    return {
        'slope': float(slope), 'increase_ratio': float(increase_ratio),
        'decrease_ratio': float(decrease_ratio),
        'last_quarter_mean': float(last_quarter_mean),
        'first_quarter_mean': float(first_quarter_mean),
        'trend_change': float(trend_change),
    }


def compute_stats(token_logprobs: List[float], token_entropies: List[float]) -> Dict[str, float]:
    if not token_logprobs or len(token_logprobs) == 0:
        return {k: 0.0 for k in [
            'ppl', 'log_ppl', 'entropy_mean', 'entropy_std', 'entropy_max', 'entropy_min',
            'entropy_slope', 'entropy_increase_ratio', 'entropy_decrease_ratio',
            'entropy_first_quarter', 'entropy_last_quarter', 'entropy_trend_change',
            'log_prob_mean', 'log_prob_std', 'log_prob_max', 'log_prob_min',
            'log_prob_slope', 'log_prob_increase_ratio', 'log_prob_decrease_ratio',
            'log_prob_first_quarter', 'log_prob_last_quarter', 'log_prob_trend_change',
            'seq_len',
        ]}
    logprobs = np.array(token_logprobs)
    entropies = np.array(token_entropies)
    mean_logprob = np.mean(logprobs)
    ppl = np.exp(-mean_logprob)
    logprob_trend = compute_trend_stats(logprobs)
    entropy_trend = compute_trend_stats(entropies)
    return {
        'ppl': float(ppl), 'log_ppl': float(np.log(ppl + 1e-10)),
        'entropy_mean': float(np.mean(entropies)), 'entropy_std': float(np.std(entropies)),
        'entropy_max': float(np.max(entropies)), 'entropy_min': float(np.min(entropies)),
        'entropy_slope': entropy_trend['slope'],
        'entropy_increase_ratio': entropy_trend['increase_ratio'],
        'entropy_decrease_ratio': entropy_trend['decrease_ratio'],
        'entropy_first_quarter': entropy_trend['first_quarter_mean'],
        'entropy_last_quarter': entropy_trend['last_quarter_mean'],
        'entropy_trend_change': entropy_trend['trend_change'],
        'log_prob_mean': float(np.mean(logprobs)), 'log_prob_std': float(np.std(logprobs)),
        'log_prob_max': float(np.max(logprobs)), 'log_prob_min': float(np.min(logprobs)),
        'log_prob_slope': logprob_trend['slope'],
        'log_prob_increase_ratio': logprob_trend['increase_ratio'],
        'log_prob_decrease_ratio': logprob_trend['decrease_ratio'],
        'log_prob_first_quarter': logprob_trend['first_quarter_mean'],
        'log_prob_last_quarter': logprob_trend['last_quarter_mean'],
        'log_prob_trend_change': logprob_trend['trend_change'],
        'seq_len': len(logprobs),
    }


FEATURE_KEYS = [
    "ppl", "log_ppl",
    "entropy_mean", "entropy_std", "entropy_max", "entropy_min",
    "entropy_slope", "entropy_increase_ratio", "entropy_decrease_ratio",
    "entropy_first_quarter", "entropy_last_quarter", "entropy_trend_change",
    "log_prob_mean", "log_prob_std", "log_prob_max", "log_prob_min",
    "log_prob_slope", "log_prob_increase_ratio", "log_prob_decrease_ratio",
    "log_prob_first_quarter", "log_prob_last_quarter", "log_prob_trend_change",
    "seq_len",
]


def extract_features_gpu(
    data: Dict[int, List[Dict]],
    model_path: str,
    device: str = "cuda:0",
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract T0 features using GPU (SLM model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    t0_data = data[0]
    all_features = []

    for i in tqdm(range(len(t0_data)), desc="Extracting T0 features"):
        item = t0_data[i]
        question = item['question']
        text = f"Question: {question}\n\nAnswer:"

        encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            logits = outputs.logits
            shifted_logits = logits[:, :-1, :]
            shifted_input_ids = input_ids[:, 1:]
            log_probs = torch.log_softmax(shifted_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=shifted_input_ids.unsqueeze(-1)
            ).squeeze(-1)
            token_logprobs = token_log_probs[0].float().cpu().numpy().tolist()
            probs = torch.softmax(shifted_logits, dim=-1)
            log_probs_clamped = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs_clamped, dim=-1)
            token_entropies = entropy[0].float().cpu().numpy().tolist()

        stats = compute_stats(token_logprobs, token_entropies)
        features = [stats[k] for k in FEATURE_KEYS]
        all_features.append(features)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # Collect labels for all stages
    n_samples = len(t0_data)
    labels = np.zeros((n_samples, len(TOKEN_LEVELS)), dtype=int)
    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        if tokens in data:
            for i in range(n_samples):
                labels[i, stage_idx] = 1 if data[tokens][i].get('is_correct', False) else 0

    return np.array(all_features), labels


def load_llm_results(results_dir: str, subsets: List[str]) -> Dict[str, List[bool]]:
    """Load 32B standalone per-problem results from lm-eval output."""
    llm_correct = {}

    # Map subset names to lm-eval task names
    subset_to_task = {
        "algebra": "hendrycks_math_algebra",
        "counting_and_probability": "hendrycks_math_counting_and_prob",
        "geometry": "hendrycks_math_geometry",
        "intermediate_algebra": "hendrycks_math_intermediate_algebra",
        "number_theory": "hendrycks_math_num_theory",
        "prealgebra": "hendrycks_math_prealgebra",
        "precalculus": "hendrycks_math_precalc",
    }

    for subset in subsets:
        task_name = subset_to_task.get(subset, f"hendrycks_math_{subset}")
        # Find the samples file
        pattern = f"samples_{task_name}_"
        found = False
        for fname in sorted(os.listdir(results_dir), reverse=True):
            if fname.startswith(pattern) and fname.endswith(".jsonl"):
                filepath = os.path.join(results_dir, fname)
                results = []
                with open(filepath, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        # lm-eval stores exact_match metric
                        # We need to check if the model got it right
                        resp = entry.get('filtered_resps', [[""]])[0]
                        target = entry.get('target', '')
                        # Extract boxed answer from response
                        correct = check_math_answer(resp, target)
                        results.append(correct)
                llm_correct[subset] = results
                logger.info(f"Loaded {len(results)} 32B results for {subset} "
                           f"({sum(results)}/{len(results)} correct = {sum(results)/len(results)*100:.1f}%)")
                found = True
                break
        if not found:
            logger.warning(f"No 32B results found for {subset}")

    return llm_correct


def check_math_answer(response: str, target: str) -> bool:
    """Check if the model response contains the correct answer."""
    # Extract answer from \\boxed{...}
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, response)
    if matches:
        extracted = matches[-1].strip()  # Use last boxed answer
        return extracted == target.strip()
    return False


def eval_binary_routing(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    low_stage: int = 0,
    high_stage: int = 3,
) -> Dict:
    """
    Train a binary routing classifier: route to low_stage or high_stage.

    Label = 1 if problem is correct at low_stage (can stay at cheap stage).
    If classifier says 0, route to high_stage.
    """
    # Labels: can the problem be solved at the low stage?
    y_train = labels[train_idx, low_stage]
    y_test = labels[test_idx, low_stage]

    X_train = features[train_idx]
    X_test = features[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    clf.fit(X_train_s, y_train)

    probs = clf.predict_proba(X_test_s)[:, 1]

    # Search for best threshold
    best_acc = 0
    best_thresh = 0.5
    best_details = {}

    for thresh in np.arange(0.1, 0.95, 0.05):
        route_low = probs >= thresh
        route_high = ~route_low

        correct = 0
        for i, idx in enumerate(test_idx):
            if route_low[i]:
                correct += labels[idx, low_stage]
            else:
                correct += labels[idx, high_stage]

        acc = correct / len(test_idx)
        pct_low = route_low.sum() / len(test_idx) * 100
        pct_high = route_high.sum() / len(test_idx) * 100

        # Compute cost
        cost_low = COST_TABLE[TOKEN_LEVELS[low_stage]]
        cost_high = COST_TABLE[TOKEN_LEVELS[high_stage]]
        avg_cost = (route_low.sum() * cost_low + route_high.sum() * cost_high) / len(test_idx)

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_details = {
                'accuracy': acc * 100,
                'threshold': thresh,
                'pct_low_stage': pct_low,
                'pct_high_stage': pct_high,
                'avg_cost_tflops': avg_cost,
            }

    return best_details


def eval_llm_routing(
    features: np.ndarray,
    labels_t0: np.ndarray,
    llm_correct: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict:
    """
    Train a routing classifier: SLM (7B) or LLM (32B standalone).

    Label = 1 if SLM can solve it (route to SLM).
    If classifier says 0, route to LLM (32B standalone).
    """
    y_train = labels_t0[train_idx]
    y_test = labels_t0[test_idx]

    X_train = features[train_idx]
    X_test = features[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    clf.fit(X_train_s, y_train)

    probs = clf.predict_proba(X_test_s)[:, 1]

    best_acc = 0
    best_details = {}

    for thresh in np.arange(0.1, 0.95, 0.05):
        route_slm = probs >= thresh
        route_llm = ~route_slm

        correct = 0
        for i, idx in enumerate(test_idx):
            if route_slm[i]:
                correct += labels_t0[idx]
            else:
                correct += llm_correct[idx]

        acc = correct / len(test_idx)
        pct_slm = route_slm.sum() / len(test_idx) * 100
        pct_llm = route_llm.sum() / len(test_idx) * 100
        avg_cost = (route_slm.sum() * COST_TABLE[0] + route_llm.sum() * COST_TABLE["32B"]) / len(test_idx)

        if acc > best_acc:
            best_acc = acc
            best_details = {
                'accuracy': acc * 100,
                'threshold': thresh,
                'pct_slm': pct_slm,
                'pct_llm': pct_llm,
                'avg_cost_tflops': avg_cost,
            }

    return best_details


def eval_tandem_cascade(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict:
    """
    Tandem cascade: multi-stage with threshold search (same as train_ppl_classifier.py).
    """
    from itertools import product

    n_stages = len(TOKEN_LEVELS)

    # Train per-stage classifiers
    X_train = features[train_idx]
    X_test = features[test_idx]

    # For cascade, we need features at each stage. Since we only have T0 features,
    # we add stage_idx as an extra feature.
    all_X_train = []
    all_y_train = []
    for stage_idx in range(n_stages):
        stage_features = np.column_stack([X_train, np.full(len(train_idx), stage_idx)])
        all_X_train.append(stage_features)
        all_y_train.append(labels[train_idx, stage_idx])

    all_X_train = np.vstack(all_X_train)
    all_y_train = np.concatenate(all_y_train)

    scaler = StandardScaler()
    all_X_train_s = scaler.fit_transform(all_X_train)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    clf.fit(all_X_train_s, all_y_train)

    # Predict for test set at each stage
    probs = np.zeros((len(test_idx), n_stages))
    for stage_idx in range(n_stages):
        stage_features = np.column_stack([X_test, np.full(len(test_idx), stage_idx)])
        stage_features_s = scaler.transform(stage_features)
        probs[:, stage_idx] = clf.predict_proba(stage_features_s)[:, 1]

    gt = labels[test_idx]

    # Grid search for thresholds
    threshold_candidates = [round(i * 0.1, 1) for i in range(11)]
    best_acc = 0
    best_thresholds = None

    for combo in product(threshold_candidates, repeat=n_stages):
        thresholds = list(combo)
        correct = 0
        stage_counts = [0] * n_stages
        for i in range(len(test_idx)):
            decided = False
            for stage_idx in range(n_stages):
                if probs[i, stage_idx] >= thresholds[stage_idx]:
                    correct += gt[i, stage_idx]
                    stage_counts[stage_idx] += 1
                    decided = True
                    break
            if not decided:
                correct += gt[i, -1]
                stage_counts[-1] += 1

        acc = correct / len(test_idx)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds
            best_stage_counts = stage_counts[:]

    # Compute cost for best cascade
    n_test = len(test_idx)
    avg_cost = sum(
        count * COST_TABLE[TOKEN_LEVELS[s]] for s, count in enumerate(best_stage_counts)
    ) / n_test

    stage_pcts = [count / n_test * 100 for count in best_stage_counts]

    return {
        'accuracy': best_acc * 100,
        'thresholds': best_thresholds,
        'stage_distribution': {
            f"T{TOKEN_LEVELS[s]}": f"{pct:.1f}%" for s, pct in enumerate(stage_pcts)
        },
        'avg_cost_tflops': avg_cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Routing baseline evaluation")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split")
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--llm-results-dir", type=str, default=None,
                        help="Directory with 32B lm-eval results (samples_*.jsonl)")
    parser.add_argument("--load-features", type=str, default=None,
                        help="Load pre-computed features from .npz file")
    parser.add_argument("--save-features", type=str, default=None,
                        help="Save computed features to .npz file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    args = parser.parse_args()

    subsets = detect_subsets(args.data_dir, args.test_split)
    logger.info(f"Found subsets: {subsets}")

    # ── Load or compute features ──────────────────────────────────────
    if args.load_features and os.path.exists(args.load_features):
        logger.info(f"Loading features from {args.load_features}")
        npz = np.load(args.load_features, allow_pickle=True)
        all_train_features = npz['train_features']
        all_train_labels = npz['train_labels']
        all_test_features = npz['test_features']
        all_test_labels = npz['test_labels']
    else:
        all_train_features = []
        all_train_labels = []
        all_test_features = []
        all_test_labels = []

        for subset in subsets:
            subset_dir = os.path.join(args.data_dir, subset)
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing subset: {subset}")
            logger.info(f"{'='*60}")

            # Load train data
            train_data = load_json_data(subset_dir, args.train_split)
            test_data = load_json_data(subset_dir, args.test_split)

            if not train_data or not test_data:
                logger.warning(f"Missing data for {subset}, skipping")
                continue

            # Extract features
            train_feat, train_lab = extract_features_gpu(
                train_data, args.model_path, args.device, args.max_length
            )
            test_feat, test_lab = extract_features_gpu(
                test_data, args.model_path, args.device, args.max_length
            )

            all_train_features.append(train_feat)
            all_train_labels.append(train_lab)
            all_test_features.append(test_feat)
            all_test_labels.append(test_lab)

        all_train_features = np.vstack(all_train_features)
        all_train_labels = np.vstack(all_train_labels)
        all_test_features = np.vstack(all_test_features)
        all_test_labels = np.vstack(all_test_labels)

        if args.save_features:
            np.savez(args.save_features,
                     train_features=all_train_features,
                     train_labels=all_train_labels,
                     test_features=all_test_features,
                     test_labels=all_test_labels)
            logger.info(f"Saved features to {args.save_features}")

    n_train = len(all_train_features)
    n_test = len(all_test_features)
    logger.info(f"\nTotal: {n_train} train, {n_test} test samples")

    # Merge train and test for index-based splitting
    all_features = np.vstack([all_train_features, all_test_features])
    all_labels = np.vstack([all_train_labels, all_test_labels])
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_train + n_test)

    # ── Baseline results ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON: Routing vs. Tandem Cascade")
    print("=" * 70)

    # 1. Individual stage baselines
    print("\n── Individual Stage Baselines ──")
    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        acc = all_labels[test_idx, stage_idx].mean() * 100
        cost = COST_TABLE[tokens]
        print(f"  SLM (7B) {'alone' if tokens == 0 else f'+ 32B ({tokens} tokens)'}: "
              f"{acc:.2f}%  |  Cost: {cost:.2f} TFLOPs")

    # 2. Binary Routing (T0 vs T1000)
    print("\n── Binary Routing: SLM (7B) vs 7B+32B (high) ──")
    binary_results = eval_binary_routing(
        all_features, all_labels, train_idx, test_idx,
        low_stage=0, high_stage=3,
    )
    print(f"  Accuracy: {binary_results['accuracy']:.2f}%")
    print(f"  Route to SLM (7B): {binary_results['pct_low_stage']:.1f}%")
    print(f"  Route to 7B+32B (high): {binary_results['pct_high_stage']:.1f}%")
    print(f"  Avg Cost: {binary_results['avg_cost_tflops']:.2f} TFLOPs")

    # 3. Tandem Cascade (re-trained from T0 features only for fair comparison)
    print("\n── Tandem Cascade (4-stage) ──")
    cascade_results = eval_tandem_cascade(
        all_features, all_labels, train_idx, test_idx,
    )
    print(f"  Accuracy: {cascade_results['accuracy']:.2f}%")
    print(f"  Stage distribution: {cascade_results['stage_distribution']}")
    print(f"  Avg Cost: {cascade_results['avg_cost_tflops']:.2f} TFLOPs")

    # 4. LLM Routing (7B vs 32B standalone) if data available
    if args.llm_results_dir and os.path.exists(args.llm_results_dir):
        print("\n── LLM Routing: SLM (7B) vs LLM (32B standalone) ──")
        llm_results = load_llm_results(args.llm_results_dir, subsets)

        if llm_results:
            # Build per-problem 32B correctness array aligned with test data
            llm_correct_all = []
            offset = 0
            for subset in subsets:
                if subset in llm_results:
                    llm_correct_all.extend(llm_results[subset])
                else:
                    # Assume 32B average accuracy for missing subsets
                    n_sub = all_test_labels[offset:].shape[0]  # rough estimate
                    llm_correct_all.extend([False] * n_sub)
            llm_correct_arr = np.array(llm_correct_all[:n_test], dtype=int)

            llm_routing_results = eval_llm_routing(
                all_features[test_idx], all_labels[test_idx, 0],
                llm_correct_arr,
                np.arange(int(n_test * 0.7)),
                np.arange(int(n_test * 0.7), n_test),
            )
            print(f"  Accuracy: {llm_routing_results['accuracy']:.2f}%")
            print(f"  Route to SLM (7B): {llm_routing_results['pct_slm']:.1f}%")
            print(f"  Route to LLM (32B): {llm_routing_results['pct_llm']:.1f}%")
            print(f"  Avg Cost: {llm_routing_results['avg_cost_tflops']:.2f} TFLOPs")

    # ── Summary Table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary (Markdown Table)")
    print("=" * 70)
    print()
    print("| Method | MATH Acc | Cost (TFLOPs) |")
    print("| ------ | -------- | ------------- |")
    print(f"| SLM (7B) standalone | {all_labels[test_idx, 0].mean()*100:.2f}% | {COST_TABLE[0]:.2f} |")
    print(f"| LLM (32B) standalone | -- | {COST_TABLE['32B']:.2f} |")
    print(f"| Binary Routing (T0/T1000) | {binary_results['accuracy']:.2f}% | {binary_results['avg_cost_tflops']:.2f} |")
    print(f"| Tandem Cascade | {cascade_results['accuracy']:.2f}% | {cascade_results['avg_cost_tflops']:.2f} |")
    print(f"| 7B+32B (high) single-stage | {all_labels[test_idx, 3].mean()*100:.2f}% | {COST_TABLE[1000]:.2f} |")

    # Save results
    output = {
        'individual_stages': {
            f"T{tokens}": {
                'accuracy': float(all_labels[test_idx, si].mean() * 100),
                'cost_tflops': COST_TABLE[tokens],
            } for si, tokens in enumerate(TOKEN_LEVELS)
        },
        'binary_routing': binary_results,
        'tandem_cascade': cascade_results,
    }

    output_path = os.path.join(args.data_dir, "routing_baseline_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
