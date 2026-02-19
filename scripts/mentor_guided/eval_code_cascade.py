#!/usr/bin/env python3
"""
Code Cascade Evaluation: Train on MBPP, Eval on MBPP + HumanEval.

1. Extract PPL/entropy features from MBPP train data
2. Train classifier on MBPP train
3. Evaluate cascade on MBPP test (in-domain effectiveness)
4. Evaluate cascade on HumanEval (cross-domain generalization)

Usage:
    python eval_code_cascade.py \
        --mbpp-train-dir /path/to/mbpp_collected/mbpp/train \
        --mbpp-test-dir  /path/to/mbpp_collected/mbpp/test \
        --humaneval-dir  /path/to/humaneval_collected/humaneval \
        --hf-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --device cuda:0
"""

import argparse
import json
import logging
import os
import sys
import pickle
import numpy as np
import torch
from itertools import product
from typing import Dict, List, Tuple

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]


# ── feature extraction ───────────────────────────────────────────────────

def compute_stats(token_logprobs, token_entropies):
    if not token_logprobs or len(token_logprobs) == 0:
        return {k: 0.0 for k in [
            'ppl', 'log_ppl',
            'entropy_mean', 'entropy_std', 'entropy_max', 'entropy_min',
            'entropy_slope', 'entropy_increase_ratio', 'entropy_decrease_ratio',
            'entropy_first_quarter', 'entropy_last_quarter', 'entropy_trend_change',
            'log_prob_mean', 'log_prob_std', 'log_prob_max', 'log_prob_min',
            'log_prob_slope', 'log_prob_increase_ratio', 'log_prob_decrease_ratio',
            'log_prob_first_quarter', 'log_prob_last_quarter', 'log_prob_trend_change',
            'seq_len',
        ]}

    lp = np.array(token_logprobs)
    ent = np.array(token_entropies)
    n = len(lp)

    def slope(a):
        return np.polyfit(np.arange(len(a)), a, 1)[0] if len(a) >= 2 else 0.0

    def inc_ratio(a):
        return float(np.mean(np.diff(a) > 0)) if len(a) >= 2 else 0.5

    def dec_ratio(a):
        return float(np.mean(np.diff(a) < 0)) if len(a) >= 2 else 0.5

    def trend_change(a):
        if len(a) < 3:
            return 0.0
        q = max(1, len(a) // 4)
        return float(np.mean(a[-q:]) - np.mean(a[:q]))

    q = max(1, n // 4)
    mean_lp = np.mean(lp)

    return {
        'ppl': float(np.exp(-mean_lp)),
        'log_ppl': float(np.log(np.exp(-mean_lp) + 1e-10)),
        'entropy_mean': float(np.mean(ent)),
        'entropy_std': float(np.std(ent)),
        'entropy_max': float(np.max(ent)),
        'entropy_min': float(np.min(ent)),
        'entropy_slope': float(slope(ent)),
        'entropy_increase_ratio': inc_ratio(ent),
        'entropy_decrease_ratio': dec_ratio(ent),
        'entropy_first_quarter': float(np.mean(ent[:q])),
        'entropy_last_quarter': float(np.mean(ent[-q:])),
        'entropy_trend_change': trend_change(ent),
        'log_prob_mean': float(np.mean(lp)),
        'log_prob_std': float(np.std(lp)),
        'log_prob_max': float(np.max(lp)),
        'log_prob_min': float(np.min(lp)),
        'log_prob_slope': float(slope(lp)),
        'log_prob_increase_ratio': inc_ratio(lp),
        'log_prob_decrease_ratio': dec_ratio(lp),
        'log_prob_first_quarter': float(np.mean(lp[:q])),
        'log_prob_last_quarter': float(np.mean(lp[-q:])),
        'log_prob_trend_change': trend_change(lp),
        'seq_len': n,
    }


def extract_features(model, tokenizer, data, device, max_length=1024):
    """Extract features for all samples at all token levels.

    Returns:
        X: (n_samples, n_stages, n_features)
        y: (n_samples, n_stages)
    """
    from tqdm import tqdm

    n_samples = len(data[TOKEN_LEVELS[0]])
    n_stages = len(TOKEN_LEVELS)

    all_X = []
    all_y = []

    for i in tqdm(range(n_samples), desc="Extracting features"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = data[tokens][i]
            question = item.get('question', item.get('prompt', ''))
            mentor_response = item.get('mentor_response', '')

            if mentor_response:
                text = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
            else:
                text = f"Question: {question}\n\nAnswer:"

            encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = encoded['input_ids'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                logits = outputs.logits
                shifted_logits = logits[:, :-1, :]
                shifted_ids = input_ids[:, 1:]

                log_probs = torch.log_softmax(shifted_logits, dim=-1)
                token_lp = log_probs.gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
                token_logprobs = token_lp[0].float().cpu().numpy().tolist()

                probs = torch.softmax(shifted_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                token_entropies = entropy[0].float().cpu().numpy().tolist()

            stats = compute_stats(token_logprobs, token_entropies)
            feat = [
                stats['ppl'], stats['log_ppl'],
                stats['entropy_mean'], stats['entropy_std'],
                stats['entropy_max'], stats['entropy_min'],
                stats['entropy_slope'], stats['entropy_increase_ratio'],
                stats['entropy_decrease_ratio'], stats['entropy_first_quarter'],
                stats['entropy_last_quarter'], stats['entropy_trend_change'],
                stats['log_prob_mean'], stats['log_prob_std'],
                stats['log_prob_max'], stats['log_prob_min'],
                stats['log_prob_slope'], stats['log_prob_increase_ratio'],
                stats['log_prob_decrease_ratio'], stats['log_prob_first_quarter'],
                stats['log_prob_last_quarter'], stats['log_prob_trend_change'],
                stats['seq_len'],
                stage_idx,
                tokens,
            ]
            all_X.append(feat)
            all_y.append(1 if item.get('is_correct', False) else 0)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    X = np.array(all_X).reshape(n_samples, n_stages, -1)
    y = np.array(all_y).reshape(n_samples, n_stages)
    return X, y


# ── cascade evaluation ───────────────────────────────────────────────────

def _cascade_correct(probs, y, thresholds):
    """Count correct predictions under given thresholds."""
    n_samples, n_stages = probs.shape
    correct = 0
    for i in range(n_samples):
        decided = False
        for s in range(n_stages):
            if probs[i, s] >= thresholds[s]:
                correct += y[i, s]
                decided = True
                break
        if not decided:
            correct += y[i, -1]
    return correct


def search_thresholds(clf, scaler, X, y):
    """Grid-search best cascade thresholds."""
    n_samples, n_stages, n_feat = X.shape
    X_flat = X.reshape(-1, n_feat)
    X_scaled = scaler.transform(X_flat)
    probs = clf.predict_proba(X_scaled)[:, 1].reshape(n_samples, n_stages)

    candidates = [round(i * 0.05, 2) for i in range(21)]
    best_acc = 0
    best_thresholds = [0.5] * n_stages

    for combo in product(candidates, repeat=n_stages):
        th = list(combo)
        correct = _cascade_correct(probs, y, th)
        acc = correct / n_samples
        if acc > best_acc:
            best_acc = acc
            best_thresholds = th

    return best_thresholds, best_acc


def eval_cascade(clf, scaler, X, y, thresholds):
    """Evaluate cascade with given thresholds. Returns detailed results."""
    n_samples, n_stages, n_feat = X.shape
    X_flat = X.reshape(-1, n_feat)
    X_scaled = scaler.transform(X_flat)
    probs = clf.predict_proba(X_scaled)[:, 1].reshape(n_samples, n_stages)

    n_correct = _cascade_correct(probs, y, thresholds)
    cascade_acc = n_correct / n_samples

    # Baselines
    baselines = {TOKEN_LEVELS[s]: float(y[:, s].mean()) for s in range(n_stages)}
    oracle = float(np.any(y, axis=1).mean())

    # Stage distribution
    stage_counts = [0] * n_stages
    for i in range(n_samples):
        decided = False
        for s in range(n_stages):
            if probs[i, s] >= thresholds[s]:
                stage_counts[s] += 1
                decided = True
                break
        if not decided:
            stage_counts[-1] += 1

    # Per-stage AUC
    stage_auc = {}
    for s in range(n_stages):
        try:
            stage_auc[TOKEN_LEVELS[s]] = roc_auc_score(y[:, s], probs[:, s])
        except ValueError:
            stage_auc[TOKEN_LEVELS[s]] = 0.5

    return {
        'n_samples': n_samples,
        'n_correct': int(n_correct),
        'cascade_acc': cascade_acc,
        'oracle': oracle,
        'baselines': baselines,
        'thresholds': thresholds,
        'stage_counts': {TOKEN_LEVELS[s]: stage_counts[s] for s in range(n_stages)},
        'stage_auc': stage_auc,
    }


def load_data(data_dir):
    """Load tokens*.json files from a directory."""
    data = {}
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
            logger.info(f"Loaded {len(data[tokens])} samples from {filepath}")
        else:
            logger.error(f"Missing: {filepath}")
            return None
    return data


def print_results(name, results):
    """Pretty-print cascade results."""
    print(f"\n  [{name}]")
    print(f"  Samples:          {results['n_samples']}")
    print(f"  Cascade Accuracy: {results['cascade_acc']:.4f} ({results['n_correct']}/{results['n_samples']})")
    print(f"  Oracle Accuracy:  {results['oracle']:.4f}")
    print(f"  Baselines:")
    for t in TOKEN_LEVELS:
        print(f"    T{t}: {results['baselines'][t]:.4f}")
    best_bl = max(results['baselines'].values())
    delta = results['cascade_acc'] - best_bl
    print(f"  Best baseline:    {best_bl:.4f}")
    print(f"  Improvement:      {delta:+.4f}")
    print(f"  Thresholds:       {results['thresholds']}")
    print(f"  Stage distribution:")
    for t in TOKEN_LEVELS:
        cnt = results['stage_counts'][t]
        pct = cnt / results['n_samples'] * 100
        print(f"    T{t}: {cnt} ({pct:.1f}%)")
    print(f"  Per-stage AUC:")
    for t in TOKEN_LEVELS:
        print(f"    T{t}: {results['stage_auc'][t]:.4f}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train classifier on MBPP, eval on MBPP + HumanEval")
    parser.add_argument("--mbpp-train-dir", type=str, required=True,
                        help="MBPP train data (tokens{0,100,500,1000}.json)")
    parser.add_argument("--mbpp-test-dir", type=str, default=None,
                        help="MBPP test data (optional, for in-domain eval)")
    parser.add_argument("--humaneval-dir", type=str, default=None,
                        help="HumanEval data (optional, for cross-domain eval)")
    parser.add_argument("--hf-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--classifier", type=str, default="gb",
                        choices=["lr", "gb"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to cache extracted features (npz files)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Helper to load or extract features ───────────────────────────
    def get_features(data_dir, cache_name):
        cache_path = None
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            cache_path = os.path.join(args.cache_dir, f"{cache_name}_features.npz")
            if os.path.exists(cache_path):
                logger.info(f"Loading cached features from {cache_path}")
                cached = np.load(cache_path)
                return cached['X'], cached['y']

        data = load_data(data_dir)
        if data is None:
            return None, None

        # Lazy-load model
        if not hasattr(get_features, '_model'):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading {args.hf_model} on {device}...")
            get_features._tokenizer = AutoTokenizer.from_pretrained(
                args.hf_model, trust_remote_code=True)
            get_features._model = AutoModelForCausalLM.from_pretrained(
                args.hf_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ).to(device).eval()

        X, y = extract_features(
            get_features._model, get_features._tokenizer,
            data, device, args.max_length)

        if cache_path:
            np.savez(cache_path, X=X, y=y)
            logger.info(f"Cached features to {cache_path}")

        return X, y

    # ── Extract features ─────────────────────────────────────────────
    logger.info("="*60)
    logger.info("Step 1: Extract MBPP train features")
    logger.info("="*60)
    X_train, y_train = get_features(args.mbpp_train_dir, "mbpp_train")
    if X_train is None:
        logger.error("Failed to load MBPP train data")
        sys.exit(1)
    logger.info(f"MBPP train: X={X_train.shape}, y={y_train.shape}")

    X_mbpp_test, y_mbpp_test = None, None
    if args.mbpp_test_dir:
        logger.info("="*60)
        logger.info("Step 2a: Extract MBPP test features")
        logger.info("="*60)
        X_mbpp_test, y_mbpp_test = get_features(args.mbpp_test_dir, "mbpp_test")

    X_he, y_he = None, None
    if args.humaneval_dir:
        logger.info("="*60)
        logger.info("Step 2b: Extract HumanEval features")
        logger.info("="*60)
        X_he, y_he = get_features(args.humaneval_dir, "humaneval")

    # Free GPU
    if hasattr(get_features, '_model'):
        del get_features._model
        torch.cuda.empty_cache()

    # ── Train classifier ─────────────────────────────────────────────
    logger.info("="*60)
    logger.info("Step 3: Train classifier on MBPP train")
    logger.info("="*60)

    n_train, n_stages, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)
    y_train_flat = y_train.reshape(-1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)

    if args.classifier == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    clf.fit(X_train_scaled, y_train_flat)
    logger.info(f"Trained {type(clf).__name__} on {n_train} samples x {n_stages} stages")

    # Search thresholds on train
    thresholds, train_cascade_acc = search_thresholds(clf, scaler, X_train, y_train)
    logger.info(f"Best thresholds (from train): {thresholds}")
    logger.info(f"Train cascade accuracy: {train_cascade_acc:.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Code Cascade Results")
    print(f"{'='*60}")
    print(f"Classifier: {args.classifier}")
    print(f"Trained on: MBPP train ({n_train} samples)")
    print(f"Thresholds: {thresholds}")

    all_results = {}

    # MBPP train (sanity check)
    train_results = eval_cascade(clf, scaler, X_train, y_train, thresholds)
    print_results("MBPP Train (sanity check)", train_results)
    all_results['mbpp_train'] = train_results

    # MBPP test (in-domain)
    if X_mbpp_test is not None:
        test_results = eval_cascade(clf, scaler, X_mbpp_test, y_mbpp_test, thresholds)
        print_results("MBPP Test (in-domain)", test_results)
        all_results['mbpp_test'] = test_results

    # HumanEval (cross-domain)
    if X_he is not None:
        he_results = eval_cascade(clf, scaler, X_he, y_he, thresholds)
        print_results("HumanEval (cross-domain: MBPP → HumanEval)", he_results)
        all_results['humaneval'] = he_results

    print(f"\n{'='*60}")

    # Summary table
    print(f"\n  Summary:")
    print(f"  {'Dataset':<35} {'Cascade':<10} {'Best BL':<10} {'Delta':<10} {'Oracle':<10}")
    print(f"  {'-'*75}")
    for name, r in all_results.items():
        best_bl = max(r['baselines'].values())
        delta = r['cascade_acc'] - best_bl
        print(f"  {name:<35} {r['cascade_acc']:<10.4f} {best_bl:<10.4f} {delta:<+10.4f} {r['oracle']:<10.4f}")
    print(f"{'='*60}")

    # Save
    if args.output:
        # Convert numpy types for JSON
        def to_json(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        out = {k: {kk: to_json(vv) for kk, vv in v.items()} for k, v in all_results.items()}
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2, default=to_json)
        logger.info(f"Saved to {args.output}")

    # Save classifier for later use
    if args.cache_dir:
        clf_path = os.path.join(args.cache_dir, "classifier.pkl")
        with open(clf_path, 'wb') as f:
            pickle.dump({
                'classifier': clf,
                'scaler': scaler,
                'thresholds': thresholds,
                'token_levels': TOKEN_LEVELS,
            }, f)
        logger.info(f"Saved classifier to {clf_path}")


if __name__ == "__main__":
    main()
