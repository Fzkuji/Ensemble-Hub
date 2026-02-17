#!/usr/bin/env python3
"""
Evaluate MATH-trained PPL Classifier on HumanEval (Cross-Domain Transfer)

DESIGN NOTE: HumanEval has NO train/test split in our pipeline.
The classifier is trained on MATH data and evaluated on the full HumanEval set
(164 problems) for cross-domain transfer. No HumanEval splitting is needed.

Loads the PPL classifier trained on MATH data, extracts distributional features
from HumanEval collected data using a HuggingFace model, and evaluates whether
the classifier generalizes across domains.

Usage:
    python eval_humaneval_transfer.py \
        --math-model-dir /path/to/hendrycks_math_split/all/ppl_model \
        --humaneval-data-dir /path/to/humaneval_collected/humaneval \
        --hf-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""

import argparse
import json
import logging
import os
import sys
import pickle
import numpy as np
import torch
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add scripts directory to path
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Reuse from train_ppl_classifier
from train_ppl_classifier import (
    TOKEN_LEVELS,
    compute_stats,
    extract_features,
    load_json_data,
)


def load_classifier(model_dir: str):
    """Load trained classifier, scaler, and thresholds."""
    pkl_path = os.path.join(model_dir, "classifier.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Classifier not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)

    clf = saved['classifier']
    scaler = saved['scaler']
    thresholds = saved.get('thresholds', [0.5] * len(TOKEN_LEVELS))
    feature_names = saved.get('feature_names', None)

    logger.info(f"Loaded classifier from {pkl_path}")
    logger.info(f"  Type: {type(clf).__name__}")
    logger.info(f"  Thresholds: {thresholds}")

    return clf, scaler, thresholds, feature_names


def cascade_evaluate(clf, scaler, thresholds, X, y):
    """Run cascade evaluation with the classifier.

    At each stage, if classifier predicts "correct" (prob > threshold),
    stop early. Otherwise, proceed to next stage.
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    n_total = len(X)
    n_stages = len(TOKEN_LEVELS)
    n_samples = n_total // n_stages

    # Scale and predict
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]
    probs = probs.reshape(n_samples, n_stages)
    y_mat = y.reshape(n_samples, n_stages)

    # Cascade decision
    final_correct = 0
    stage_counts = [0] * n_stages
    decisions = []

    for i in range(n_samples):
        stopped = False
        for s in range(n_stages):
            if probs[i, s] >= thresholds[s]:
                final_correct += y_mat[i, s]
                stage_counts[s] += 1
                decisions.append({
                    'sample_idx': i,
                    'stop_stage': s,
                    'stop_token_level': TOKEN_LEVELS[s],
                    'predicted_correct': True,
                    'actually_correct': bool(y_mat[i, s]),
                    'prob': float(probs[i, s]),
                })
                stopped = True
                break

        if not stopped:
            final_correct += y_mat[i, -1]
            stage_counts[-1] += 1
            decisions.append({
                'sample_idx': i,
                'stop_stage': n_stages - 1,
                'stop_token_level': TOKEN_LEVELS[-1],
                'predicted_correct': False,
                'actually_correct': bool(y_mat[i, -1]),
                'prob': float(probs[i, -1]),
            })

    cascade_accuracy = final_correct / n_samples if n_samples > 0 else 0

    # Per-stage metrics
    stage_metrics = []
    for s in range(n_stages):
        y_true = y_mat[:, s]
        y_prob = probs[:, s]
        y_pred = (y_prob >= thresholds[s]).astype(int)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0

        stage_metrics.append({
            'token_level': TOKEN_LEVELS[s],
            'auc_roc': auc,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'threshold': thresholds[s],
            'n_stopped': stage_counts[s],
        })

    # Baselines
    baselines = {}
    for s in range(n_stages):
        baselines[f"T{TOKEN_LEVELS[s]}"] = float(y_mat[:, s].mean())

    oracle_correct = sum(1 for i in range(n_samples) if any(y_mat[i, s] for s in range(n_stages)))
    oracle_accuracy = oracle_correct / n_samples if n_samples > 0 else 0

    return {
        'n_samples': n_samples,
        'cascade_accuracy': cascade_accuracy,
        'oracle_accuracy': oracle_accuracy,
        'baselines': baselines,
        'stage_counts': {TOKEN_LEVELS[s]: stage_counts[s] for s in range(n_stages)},
        'stage_metrics': stage_metrics,
        'decisions': decisions,
        'y_mat': y_mat,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MATH-trained classifier on HumanEval (cross-domain transfer)")
    parser.add_argument("--math-model-dir", type=str, required=True,
                        help="Directory containing MATH-trained classifier.pkl")
    parser.add_argument("--humaneval-data-dir", type=str, required=True,
                        help="Directory containing HumanEval tokens{0,100,500,1000}.json")
    parser.add_argument("--hf-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="HuggingFace model for feature extraction (same model used during training)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for model (auto-detect if not specified)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length for feature extraction")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results JSON")

    args = parser.parse_args()

    # Load classifier
    clf, scaler, thresholds, feature_names = load_classifier(args.math_model_dir)

    # Load HumanEval collected data
    data = {}
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(args.humaneval_data_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
            logger.info(f"Loaded {len(data[tokens])} samples from {filepath}")
        else:
            logger.warning(f"File not found: {filepath}")

    if not data:
        logger.error("No data loaded. Check --humaneval-data-dir")
        sys.exit(1)

    available_levels = sorted(data.keys())
    logger.info(f"Available token levels: {available_levels}")

    # Load HuggingFace model for feature extraction
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Loading model {args.hf_model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    logger.info("Model loaded")

    # Extract features (reuse same function as training)
    logger.info("Extracting PPL/entropy features from HumanEval data...")
    X, y, stages = extract_features(model, tokenizer, data, device, max_length=args.max_length)
    logger.info(f"Features: {X.shape}, Labels: {y.shape}")

    # Evaluate with MATH-trained classifier
    results = cascade_evaluate(clf, scaler, thresholds, X, y)

    # Print results
    print(f"\n{'='*60}")
    print(f"Cross-Domain Transfer: MATH -> HumanEval")
    print(f"{'='*60}")
    print(f"Classifier: {args.math_model_dir}")
    print(f"HumanEval data: {args.humaneval_data_dir}")
    print(f"Feature model: {args.hf_model}")
    print(f"Samples: {results['n_samples']}")
    print()
    print(f"Cascade Accuracy: {results['cascade_accuracy']:.4f}")
    print(f"Oracle Accuracy:  {results['oracle_accuracy']:.4f}")
    print()

    print("Baselines (single-stage accuracy):")
    for name, acc in results['baselines'].items():
        print(f"  {name}: {acc:.4f}")
    print()

    best_baseline = max(results['baselines'].values())
    print(f"Best single-stage baseline: {best_baseline:.4f}")
    delta = results['cascade_accuracy'] - best_baseline
    print(f"Cascade improvement over best baseline: {delta:+.4f}")
    print()

    print("Stage Distribution:")
    for level, count in results['stage_counts'].items():
        pct = count / results['n_samples'] * 100
        print(f"  T{level}: {count} samples ({pct:.1f}%)")
    print()

    print("Per-Stage Classifier Metrics:")
    print(f"  {'Stage':<8} {'AUC-ROC':<10} {'Accuracy':<10} {'F1':<10} {'Threshold':<10}")
    for sm in results['stage_metrics']:
        print(f"  T{sm['token_level']:<7} {sm['auc_roc']:<10.4f} {sm['accuracy']:<10.4f} {sm['f1']:<10.4f} {sm['threshold']:<10.4f}")

    print()
    print("Failure Analysis:")
    y_mat = results['y_mat']
    decisions = results['decisions']
    n = results['n_samples']

    false_early = sum(1 for d in decisions
                      if d['predicted_correct'] and not d['actually_correct']
                      and d['stop_stage'] < len(TOKEN_LEVELS) - 1)
    false_cont = sum(1 for d in decisions
                     if not d['predicted_correct'] and d['actually_correct'])
    correct_route = sum(1 for d in decisions
                        if d['predicted_correct'] and d['actually_correct']
                        and d['stop_stage'] < len(TOKEN_LEVELS) - 1)
    unavoidable = sum(1 for d in decisions
                      if not any(y_mat[d['sample_idx'], s] for s in range(len(TOKEN_LEVELS))))

    print(f"  False early stop:    {false_early:>4} ({false_early/n*100:.1f}%)")
    print(f"  False continuation:  {false_cont:>4} ({false_cont/n*100:.1f}%)")
    print(f"  Correct routing:     {correct_route:>4} ({correct_route/n*100:.1f}%)")
    print(f"  Unavoidable wrong:   {unavoidable:>4} ({unavoidable/n*100:.1f}%)")
    print(f"{'='*60}")

    # Save results
    if args.output:
        output_data = {
            'classifier_source': args.math_model_dir,
            'humaneval_data': args.humaneval_data_dir,
            'feature_model': args.hf_model,
            'n_samples': results['n_samples'],
            'cascade_accuracy': results['cascade_accuracy'],
            'oracle_accuracy': results['oracle_accuracy'],
            'baselines': results['baselines'],
            'stage_counts': {str(k): v for k, v in results['stage_counts'].items()},
            'stage_metrics': results['stage_metrics'],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
