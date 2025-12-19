#!/usr/bin/env python3
"""
PPL/Entropy-based classifier for mentor sufficiency prediction.

Method:
1. Feed mentor hint + question to intern model
2. Extract PPL and entropy statistics from model output
3. Train a simple regression model (LogisticRegression/XGBoost) on these features

Usage:
    python train_ppl_classifier.py --subset algebra --data-dir /path/to/data
"""

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split as sk_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def load_json_data(data_dir: str, split: str = "train") -> Dict[int, List[Dict]]:
    """Load JSON data for all token levels."""
    data = {}
    split_dir = os.path.join(data_dir, split)

    if not os.path.exists(split_dir):
        split_dir = data_dir
        logger.warning(f"Split dir not found, using: {split_dir}")

    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(split_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
            logger.info(f"Loaded {len(data[tokens])} samples from {filepath}")
        else:
            logger.warning(f"File not found: {filepath}")

    return data


def compute_trend_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute trend statistics for a sequence of values."""
    n = len(values)
    if n < 2:
        return {
            'slope': 0.0,
            'increase_ratio': 0.5,
            'decrease_ratio': 0.5,
            'last_quarter_mean': float(values[0]) if n > 0 else 0.0,
            'first_quarter_mean': float(values[0]) if n > 0 else 0.0,
            'trend_change': 0.0,
        }

    # Linear regression slope (normalized by sequence length)
    x = np.arange(n)
    slope = np.polyfit(x, values, 1)[0]

    # Ratio of increasing/decreasing transitions
    diffs = np.diff(values)
    increase_ratio = np.mean(diffs > 0)
    decrease_ratio = np.mean(diffs < 0)

    # Compare first and last quarters
    quarter = max(1, n // 4)
    first_quarter_mean = np.mean(values[:quarter])
    last_quarter_mean = np.mean(values[-quarter:])
    trend_change = last_quarter_mean - first_quarter_mean

    return {
        'slope': float(slope),
        'increase_ratio': float(increase_ratio),
        'decrease_ratio': float(decrease_ratio),
        'last_quarter_mean': float(last_quarter_mean),
        'first_quarter_mean': float(first_quarter_mean),
        'trend_change': float(trend_change),
    }


def compute_ppl_entropy(
    model,
    tokenizer,
    text: str,
    max_length: int = 1024,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute PPL and entropy statistics for given text.

    Returns dict with:
    - ppl: perplexity
    - entropy stats: mean, std, max, min, trend
    - log_prob stats: mean, std, trend
    """
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        # Get logits for entropy calculation
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Compute token-level log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]

        # Get log prob of actual tokens (shifted by 1)
        # logits[t] predicts token[t+1]
        shifted_input_ids = input_ids[:, 1:]  # [1, seq_len-1]
        shifted_log_probs = log_probs[:, :-1, :]  # [1, seq_len-1, vocab_size]

        # Gather log probs of actual tokens
        token_log_probs = shifted_log_probs.gather(
            dim=-1,
            index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)  # [1, seq_len-1]

        # Compute entropy at each position
        probs = torch.softmax(logits[:, :-1, :], dim=-1)  # [1, seq_len-1, vocab_size]
        entropy = -torch.sum(probs * log_probs[:, :-1, :], dim=-1)  # [1, seq_len-1]

        # Only consider non-padding tokens
        valid_mask = attention_mask[:, 1:].bool()  # [1, seq_len-1]

        valid_log_probs = token_log_probs[valid_mask].cpu().numpy()
        valid_entropy = entropy[valid_mask].cpu().numpy()

        # Compute PPL
        ppl = torch.exp(outputs.loss).item()

    # Compute trend statistics
    entropy_trend = compute_trend_stats(valid_entropy)
    logprob_trend = compute_trend_stats(valid_log_probs)

    return {
        'ppl': ppl,
        'log_ppl': np.log(ppl + 1e-10),
        # Entropy basic stats
        'entropy_mean': float(np.mean(valid_entropy)),
        'entropy_std': float(np.std(valid_entropy)),
        'entropy_max': float(np.max(valid_entropy)),
        'entropy_min': float(np.min(valid_entropy)),
        # Entropy trend stats
        'entropy_slope': entropy_trend['slope'],
        'entropy_increase_ratio': entropy_trend['increase_ratio'],
        'entropy_decrease_ratio': entropy_trend['decrease_ratio'],
        'entropy_first_quarter': entropy_trend['first_quarter_mean'],
        'entropy_last_quarter': entropy_trend['last_quarter_mean'],
        'entropy_trend_change': entropy_trend['trend_change'],
        # Log prob basic stats
        'log_prob_mean': float(np.mean(valid_log_probs)),
        'log_prob_std': float(np.std(valid_log_probs)),
        'log_prob_max': float(np.max(valid_log_probs)),
        'log_prob_min': float(np.min(valid_log_probs)),
        # Log prob trend stats
        'log_prob_slope': logprob_trend['slope'],
        'log_prob_increase_ratio': logprob_trend['increase_ratio'],
        'log_prob_decrease_ratio': logprob_trend['decrease_ratio'],
        'log_prob_first_quarter': logprob_trend['first_quarter_mean'],
        'log_prob_last_quarter': logprob_trend['last_quarter_mean'],
        'log_prob_trend_change': logprob_trend['trend_change'],
        # Sequence info
        'seq_len': len(valid_log_probs),
    }


def extract_features(
    model,
    tokenizer,
    data: Dict[int, List[Dict]],
    max_length: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract PPL/entropy features for all samples.

    Returns:
        features: [n_samples * n_stages, n_features]
        labels: [n_samples * n_stages]
        stages: [n_samples * n_stages]
    """
    all_features = []
    all_labels = []
    all_stages = []

    n_samples = len(data[TOKEN_LEVELS[0]])

    for i in tqdm(range(n_samples), desc="Extracting features"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            # Build prompt
            if mentor_response:
                text = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
            else:
                text = f"Question: {question}\n\nAnswer:"

            # Compute features
            stats = compute_ppl_entropy(model, tokenizer, text, max_length, device)

            # Feature vector (all stats from compute_ppl_entropy + stage info)
            features = [
                # PPL
                stats['ppl'],
                stats['log_ppl'],
                # Entropy basic
                stats['entropy_mean'],
                stats['entropy_std'],
                stats['entropy_max'],
                stats['entropy_min'],
                # Entropy trend
                stats['entropy_slope'],
                stats['entropy_increase_ratio'],
                stats['entropy_decrease_ratio'],
                stats['entropy_first_quarter'],
                stats['entropy_last_quarter'],
                stats['entropy_trend_change'],
                # Log prob basic
                stats['log_prob_mean'],
                stats['log_prob_std'],
                stats['log_prob_max'],
                stats['log_prob_min'],
                # Log prob trend
                stats['log_prob_slope'],
                stats['log_prob_increase_ratio'],
                stats['log_prob_decrease_ratio'],
                stats['log_prob_first_quarter'],
                stats['log_prob_last_quarter'],
                stats['log_prob_trend_change'],
                # Sequence info
                stats['seq_len'],
                stage_idx,  # stage as feature
                tokens,  # token level as feature
            ]

            all_features.append(features)
            all_labels.append(1 if item.get('is_correct', False) else 0)
            all_stages.append(stage_idx)

        # Clear cache periodically
        if i % 50 == 0:
            torch.cuda.empty_cache()

    return np.array(all_features), np.array(all_labels), np.array(all_stages)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "gb",
) -> Tuple[object, StandardScaler, Dict]:
    """
    Train a classifier on PPL/entropy features.

    Args:
        model_type: "lr" for LogisticRegression, "gb" for GradientBoosting
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if model_type == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

    clf.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = clf.predict(X_train_scaled)
    val_pred = clf.predict(X_val_scaled)

    train_proba = clf.predict_proba(X_train_scaled)[:, 1]
    val_proba = clf.predict_proba(X_val_scaled)[:, 1]

    results = {
        'train_acc': accuracy_score(y_train, train_pred),
        'val_acc': accuracy_score(y_val, val_pred),
        'train_auc': roc_auc_score(y_train, train_proba),
        'val_auc': roc_auc_score(y_val, val_proba),
    }

    return clf, scaler, results


def eval_cascade(
    clf,
    scaler: StandardScaler,
    data: Dict[int, List[Dict]],
    features: np.ndarray,
    labels: np.ndarray,
    stages: np.ndarray,
) -> Tuple[float, List[float], Dict]:
    """
    Evaluate cascade accuracy with threshold search.
    """
    n_samples = len(data[TOKEN_LEVELS[0]])
    n_stages = len(TOKEN_LEVELS)

    # Reshape features and get probabilities
    features_scaled = scaler.transform(features)
    probs = clf.predict_proba(features_scaled)[:, 1]

    # Reshape to [n_samples, n_stages]
    probs = probs.reshape(n_samples, n_stages)
    gt = labels.reshape(n_samples, n_stages)

    def compute_cascade_acc(thresholds):
        correct = 0
        for i in range(n_samples):
            decided = False
            stage_probs = []
            for stage_idx in range(n_stages):
                prob = probs[i, stage_idx]
                stage_probs.append((stage_idx, prob))
                if prob >= thresholds[stage_idx]:
                    correct += gt[i, stage_idx]
                    decided = True
                    break
            if not decided:
                best_stage, _ = max(stage_probs, key=lambda x: x[1])
                correct += gt[i, best_stage]
        return correct / n_samples

    # Threshold search
    from itertools import product
    threshold_candidates = [round(i * 0.05, 2) for i in range(21)]
    best_acc = 0
    best_thresholds = None

    for combo in product(threshold_candidates, repeat=n_stages):
        thresholds = list(combo)
        acc = compute_cascade_acc(thresholds)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    # Oracle accuracy
    oracle_correct = 0
    for i in range(n_samples):
        if any(gt[i, :] == 1):
            oracle_correct += 1
    oracle_acc = oracle_correct / n_samples

    # Per-stage accuracy
    stage_acc = {}
    stage_auc = {}
    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        stage_labels = gt[:, stage_idx]
        stage_probs_flat = probs[:, stage_idx]
        stage_acc[tokens] = np.mean(stage_labels)
        try:
            stage_auc[tokens] = roc_auc_score(stage_labels, stage_probs_flat)
        except ValueError:
            stage_auc[tokens] = 0.5

    detailed = {
        'oracle': oracle_acc,
        'baseline': stage_acc,
        'auc': stage_auc,
    }

    return best_acc, best_thresholds, detailed


def main():
    parser = argparse.ArgumentParser(description="PPL/Entropy-based classifier")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Base directory with subset folders")
    parser.add_argument("--subset", type=str, default="algebra",
                        choices=SUBSETS + ["all"],
                        help="Which subset to train on")
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--classifier", type=str, default="gb",
                        choices=["lr", "gb"],
                        help="Classifier type: lr=LogisticRegression, gb=GradientBoosting")
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--no-filter", action="store_true",
                        help="Don't filter out all-correct/all-wrong samples")

    args = parser.parse_args()

    # Determine subset directory
    subset_dir = os.path.join(args.data_dir, args.subset)
    if not os.path.exists(subset_dir):
        logger.error(f"Data directory not found: {subset_dir}")
        return

    if args.output_dir is None:
        args.output_dir = os.path.join(subset_dir, "ppl_model")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Subset: {args.subset}")
    logger.info(f"Data dir: {subset_dir}")
    logger.info(f"Classifier: {args.classifier}")

    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    # Load data
    logger.info("Loading training data...")
    train_data = load_json_data(subset_dir, split="train")
    if not train_data:
        logger.error("No training data found!")
        return

    # Split train data into train/val
    n_samples = len(train_data[TOKEN_LEVELS[0]])
    train_idx, val_idx = sk_split(
        np.arange(n_samples), test_size=args.val_ratio, random_state=42
    )

    val_data = {}
    actual_train_data = {}
    for tokens in TOKEN_LEVELS:
        if tokens in train_data:
            val_data[tokens] = [train_data[tokens][i] for i in val_idx]
            actual_train_data[tokens] = [train_data[tokens][i] for i in train_idx]
    train_data = actual_train_data

    logger.info(f"Train: {len(train_data[TOKEN_LEVELS[0]])} samples")
    logger.info(f"Val: {len(val_data[TOKEN_LEVELS[0]])} samples")

    # Filter uniform samples if needed
    if not args.no_filter:
        def filter_varied(data):
            n = len(data[TOKEN_LEVELS[0]])
            varied_indices = []
            for i in range(n):
                labels = [1 if data[tokens][i].get('is_correct', False) else 0
                          for tokens in TOKEN_LEVELS]
                if not (all(l == 1 for l in labels) or all(l == 0 for l in labels)):
                    varied_indices.append(i)
            filtered = {}
            for tokens in TOKEN_LEVELS:
                filtered[tokens] = [data[tokens][i] for i in varied_indices]
            return filtered

        train_data = filter_varied(train_data)
        val_data = filter_varied(val_data)
        logger.info(f"After filtering: Train={len(train_data[TOKEN_LEVELS[0]])}, Val={len(val_data[TOKEN_LEVELS[0]])}")

    # Extract features
    logger.info("Extracting features from training data...")
    X_train, y_train, stages_train = extract_features(
        model, tokenizer, train_data, args.max_length, args.device
    )

    logger.info("Extracting features from validation data...")
    X_val, y_val, stages_val = extract_features(
        model, tokenizer, val_data, args.max_length, args.device
    )

    logger.info(f"Train features shape: {X_train.shape}")
    logger.info(f"Val features shape: {X_val.shape}")

    # Train classifier
    logger.info(f"Training {args.classifier} classifier...")
    clf, scaler, train_results = train_classifier(
        X_train, y_train, X_val, y_val, args.classifier
    )

    logger.info(f"Train Acc: {train_results['train_acc']:.4f}, Train AUC: {train_results['train_auc']:.4f}")
    logger.info(f"Val Acc: {train_results['val_acc']:.4f}, Val AUC: {train_results['val_auc']:.4f}")

    # Cascade evaluation on combined train+val
    logger.info("Running cascade evaluation...")
    combined_data = {}
    for tokens in TOKEN_LEVELS:
        combined_data[tokens] = train_data[tokens] + val_data[tokens]

    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    stages_combined = np.concatenate([stages_train, stages_val])

    cascade_acc, thresholds, detailed = eval_cascade(
        clf, scaler, combined_data, X_combined, y_combined, stages_combined
    )

    logger.info(f"Cascade Accuracy: {cascade_acc:.4f} (Oracle: {detailed['oracle']:.4f})")
    logger.info(f"Thresholds: {thresholds}")

    auc_str = ", ".join([f"T{t}={detailed['auc'][t]:.4f}" for t in TOKEN_LEVELS])
    logger.info(f"Per-stage AUC: {auc_str}")

    baseline_str = ", ".join([f"T{t}={detailed['baseline'][t]:.4f}" for t in TOKEN_LEVELS])
    logger.info(f"Per-stage baseline acc: {baseline_str}")

    # Save model
    model_path = os.path.join(args.output_dir, "classifier.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'classifier': clf, 'scaler': scaler, 'thresholds': thresholds}, f)
    logger.info(f"Model saved to {model_path}")

    # Save results
    results = {
        'subset': args.subset,
        'classifier': args.classifier,
        'train_acc': train_results['train_acc'],
        'val_acc': train_results['val_acc'],
        'train_auc': train_results['train_auc'],
        'val_auc': train_results['val_auc'],
        'cascade_acc': cascade_acc,
        'best_thresholds': thresholds,
        'oracle': detailed['oracle'],
        'per_stage_auc': detailed['auc'],
        'per_stage_baseline': detailed['baseline'],
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
