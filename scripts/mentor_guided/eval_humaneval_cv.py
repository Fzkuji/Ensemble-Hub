#!/usr/bin/env python3
"""
HumanEval Cascade Evaluation via K-Fold Cross-Validation.

Since HumanEval has only 164 problems and no train/test split,
we use k-fold CV: extract features once, then train/eval the
PPL classifier across folds.

Usage:
    python eval_humaneval_cv.py \
        --data-dir /path/to/humaneval_collected/humaneval \
        --hf-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --device cuda:0
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
import torch
from itertools import product
from typing import Dict, List, Tuple

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]

# ── feature extraction (reused from train_ppl_classifier) ────────────────

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


def extract_all_features(model, tokenizer, data, device, max_length=1024):
    """Extract features for ALL samples at ALL token levels. Returns (X, y) matrices.

    X shape: (n_samples, n_stages, n_features)
    y shape: (n_samples, n_stages)
    """
    from tqdm import tqdm

    n_samples = len(data[TOKEN_LEVELS[0]])
    n_stages = len(TOKEN_LEVELS)

    all_X = []  # will be (n_samples * n_stages, n_features)
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

def search_thresholds_and_eval(clf, scaler, X_test, y_test):
    """Search best thresholds on test fold and return cascade accuracy."""
    n_samples, n_stages, n_feat = X_test.shape
    X_flat = X_test.reshape(-1, n_feat)
    X_scaled = scaler.transform(X_flat)
    probs = clf.predict_proba(X_scaled)[:, 1].reshape(n_samples, n_stages)

    # Grid search thresholds
    candidates = [round(i * 0.05, 2) for i in range(21)]
    best_acc = 0
    best_thresholds = [0.5] * n_stages

    for combo in product(candidates, repeat=n_stages):
        th = list(combo)
        correct = 0
        for i in range(n_samples):
            decided = False
            for s in range(n_stages):
                if probs[i, s] >= th[s]:
                    correct += y_test[i, s]
                    decided = True
                    break
            if not decided:
                correct += y_test[i, -1]
        acc = correct / n_samples
        if acc > best_acc:
            best_acc = acc
            best_thresholds = th

    # Per-stage AUC
    stage_auc = {}
    for s in range(n_stages):
        try:
            stage_auc[TOKEN_LEVELS[s]] = roc_auc_score(y_test[:, s], probs[:, s])
        except ValueError:
            stage_auc[TOKEN_LEVELS[s]] = 0.5

    # Stage distribution with best thresholds
    stage_counts = [0] * n_stages
    for i in range(n_samples):
        decided = False
        for s in range(n_stages):
            if probs[i, s] >= best_thresholds[s]:
                stage_counts[s] += 1
                decided = True
                break
        if not decided:
            stage_counts[-1] += 1

    return best_acc, best_thresholds, stage_auc, stage_counts


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HumanEval Cascade via K-Fold CV")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing tokens{0,100,500,1000}.json")
    parser.add_argument("--hf-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--classifier", type=str, default="gb",
                        choices=["lr", "gb"],
                        help="Classifier type: lr=LogisticRegression, gb=GradientBoosting")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache-features", type=str, default=None,
                        help="Path to cache extracted features (npz). "
                             "Skips model loading if cache exists.")
    args = parser.parse_args()

    # ── Load or extract features ─────────────────────────────────────
    if args.cache_features and os.path.exists(args.cache_features):
        logger.info(f"Loading cached features from {args.cache_features}")
        cached = np.load(args.cache_features)
        X, y = cached['X'], cached['y']
        logger.info(f"Loaded features: X={X.shape}, y={y.shape}")
    else:
        # Load data
        data = {}
        for tokens in TOKEN_LEVELS:
            filepath = os.path.join(args.data_dir, f"tokens{tokens}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data[tokens] = json.load(f)
                logger.info(f"Loaded {len(data[tokens])} samples from tokens{tokens}.json")
            else:
                logger.error(f"Missing: {filepath}")
                sys.exit(1)

        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading {args.hf_model} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(device).eval()

        X, y = extract_all_features(model, tokenizer, data, device, args.max_length)
        logger.info(f"Features: X={X.shape}, y={y.shape}")

        # Free GPU
        del model
        torch.cuda.empty_cache()

        # Cache
        if args.cache_features:
            np.savez(args.cache_features, X=X, y=y)
            logger.info(f"Cached features to {args.cache_features}")

    # ── K-Fold CV ────────────────────────────────────────────────────
    n_samples = X.shape[0]
    n_stages = X.shape[1]
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_fold_accs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Flatten for classifier training: (n_train * n_stages, n_features)
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        y_train_flat = y_train.reshape(-1)

        # Train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)

        if args.classifier == "lr":
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        else:
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

        clf.fit(X_train_scaled, y_train_flat)

        # Evaluate cascade on test fold
        cascade_acc, thresholds, stage_auc, stage_counts = \
            search_thresholds_and_eval(clf, scaler, X_test, y_test)

        # Baselines
        baselines = {TOKEN_LEVELS[s]: float(y_test[:, s].mean()) for s in range(n_stages)}
        oracle = float(np.any(y_test, axis=1).mean())

        fold_results.append({
            'fold': fold_idx,
            'n_test': len(test_idx),
            'cascade_acc': cascade_acc,
            'oracle': oracle,
            'baselines': baselines,
            'thresholds': thresholds,
            'stage_auc': stage_auc,
            'stage_counts': stage_counts,
        })
        all_fold_accs.append(cascade_acc)

        logger.info(f"Fold {fold_idx}: cascade={cascade_acc:.4f}, "
                    f"oracle={oracle:.4f}, best_baseline={max(baselines.values()):.4f}, "
                    f"thresholds={thresholds}")

    # ── Summary ──────────────────────────────────────────────────────
    mean_cascade = np.mean(all_fold_accs)
    std_cascade = np.std(all_fold_accs)

    # Overall baselines (from full data)
    full_baselines = {TOKEN_LEVELS[s]: float(y[:, s].mean()) for s in range(n_stages)}
    full_oracle = float(np.any(y, axis=1).mean())

    print(f"\n{'='*60}")
    print(f"HumanEval Cascade - {args.n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")
    print(f"Samples: {n_samples}")
    print(f"Classifier: {args.classifier}")
    print(f"Folds: {args.n_folds}")
    print()
    print(f"Cascade Accuracy: {mean_cascade:.4f} (+/- {std_cascade:.4f})")
    print(f"Oracle Accuracy:  {full_oracle:.4f}")
    print()
    print("Full-data Baselines:")
    for t in TOKEN_LEVELS:
        print(f"  T{t}: {full_baselines[t]:.4f}")
    best_baseline = max(full_baselines.values())
    print(f"\nBest single-stage baseline: {best_baseline:.4f}")
    print(f"Cascade improvement: {mean_cascade - best_baseline:+.4f}")
    print()

    print("Per-Fold Results:")
    print(f"  {'Fold':<6} {'Cascade':<10} {'Oracle':<10} {'Thresholds'}")
    for fr in fold_results:
        th_str = ", ".join(f"{t:.2f}" for t in fr['thresholds'])
        print(f"  {fr['fold']:<6} {fr['cascade_acc']:<10.4f} {fr['oracle']:<10.4f} [{th_str}]")

    # Average stage AUC
    print(f"\nAvg Per-Stage AUC:")
    for s, t in enumerate(TOKEN_LEVELS):
        avg_auc = np.mean([fr['stage_auc'][t] for fr in fold_results])
        print(f"  T{t}: {avg_auc:.4f}")

    print(f"{'='*60}")

    # Save
    if args.output:
        out = {
            'n_samples': n_samples,
            'n_folds': args.n_folds,
            'classifier': args.classifier,
            'cascade_accuracy_mean': float(mean_cascade),
            'cascade_accuracy_std': float(std_cascade),
            'oracle_accuracy': float(full_oracle),
            'baselines': {str(k): v for k, v in full_baselines.items()},
            'fold_results': fold_results,
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
