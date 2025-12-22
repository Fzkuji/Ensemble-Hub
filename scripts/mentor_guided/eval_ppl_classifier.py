#!/usr/bin/env python3
"""
Evaluate a trained PPL classifier on all subsets.

Usage:
    # Evaluate model trained on "all" subsets
    python eval_ppl_classifier.py --model-dir /path/to/all/ppl_model

    # Evaluate on specific subsets only
    python eval_ppl_classifier.py --model-dir /path/to/ppl_model --subsets algebra,geometry

    # Use different LLM for feature extraction
    python eval_ppl_classifier.py --model-dir /path/to/ppl_model --llm-path Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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


def load_json_data(data_dir: str, split: str = "test") -> Dict[int, List[Dict]]:
    """Load JSON data for all token levels."""
    data = {}
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, split, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
        else:
            logger.warning(f"File not found: {filepath}")
    return data if data else None


def compute_stats(token_logprobs: List[float], token_entropies: List[float]) -> Dict[str, float]:
    """Compute statistics from token-level log probs and entropies."""
    if not token_logprobs or not token_entropies:
        return {
            'ppl': 1000.0, 'log_ppl': 10.0,
            'entropy_mean': 10.0, 'entropy_std': 0.0, 'entropy_max': 10.0, 'entropy_min': 0.0,
            'entropy_slope': 0.0, 'entropy_increase_ratio': 0.0, 'entropy_decrease_ratio': 0.0,
            'entropy_first_quarter': 10.0, 'entropy_last_quarter': 10.0, 'entropy_trend_change': 0.0,
            'log_prob_mean': -10.0, 'log_prob_std': 0.0, 'log_prob_max': 0.0, 'log_prob_min': -10.0,
            'log_prob_slope': 0.0, 'log_prob_increase_ratio': 0.0, 'log_prob_decrease_ratio': 0.0,
            'log_prob_first_quarter': -10.0, 'log_prob_last_quarter': -10.0, 'log_prob_trend_change': 0.0,
            'seq_len': 0,
        }

    n = len(token_logprobs)
    avg_log_prob = np.mean(token_logprobs)
    ppl = np.exp(-avg_log_prob)
    log_ppl = -avg_log_prob

    ent_arr = np.array(token_entropies)
    lp_arr = np.array(token_logprobs)

    def calc_slope(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]

    def calc_ratio(arr, increase=True):
        if len(arr) < 2:
            return 0.0
        diffs = np.diff(arr)
        if increase:
            return np.sum(diffs > 0) / len(diffs)
        return np.sum(diffs < 0) / len(diffs)

    def calc_trend_changes(arr):
        if len(arr) < 3:
            return 0
        diffs = np.diff(arr)
        signs = np.sign(diffs)
        return np.sum(np.abs(np.diff(signs)) > 0)

    q1 = max(1, n // 4)

    return {
        'ppl': float(ppl),
        'log_ppl': float(log_ppl),
        'entropy_mean': float(np.mean(ent_arr)),
        'entropy_std': float(np.std(ent_arr)),
        'entropy_max': float(np.max(ent_arr)),
        'entropy_min': float(np.min(ent_arr)),
        'entropy_slope': float(calc_slope(ent_arr)),
        'entropy_increase_ratio': float(calc_ratio(ent_arr, True)),
        'entropy_decrease_ratio': float(calc_ratio(ent_arr, False)),
        'entropy_first_quarter': float(np.mean(ent_arr[:q1])),
        'entropy_last_quarter': float(np.mean(ent_arr[-q1:])),
        'entropy_trend_change': float(calc_trend_changes(ent_arr)),
        'log_prob_mean': float(np.mean(lp_arr)),
        'log_prob_std': float(np.std(lp_arr)),
        'log_prob_max': float(np.max(lp_arr)),
        'log_prob_min': float(np.min(lp_arr)),
        'log_prob_slope': float(calc_slope(lp_arr)),
        'log_prob_increase_ratio': float(calc_ratio(lp_arr, True)),
        'log_prob_decrease_ratio': float(calc_ratio(lp_arr, False)),
        'log_prob_first_quarter': float(np.mean(lp_arr[:q1])),
        'log_prob_last_quarter': float(np.mean(lp_arr[-q1:])),
        'log_prob_trend_change': float(calc_trend_changes(lp_arr)),
        'seq_len': n,
    }


def extract_features(
    model,
    tokenizer,
    data: Dict[int, List[Dict]],
    device: str,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract PPL/entropy features using HuggingFace model."""
    all_features = []
    all_labels = []
    all_stages = []

    n_samples = len(data[TOKEN_LEVELS[0]])

    for i in tqdm(range(n_samples), desc="Extracting features"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            if mentor_response:
                text = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
            else:
                text = f"Question: {question}\n\nAnswer:"

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded['input_ids'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                logits = outputs.logits

                shifted_logits = logits[:, :-1, :]
                shifted_input_ids = input_ids[:, 1:]

                log_probs = torch.log_softmax(shifted_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1,
                    index=shifted_input_ids.unsqueeze(-1)
                ).squeeze(-1)
                token_logprobs = token_log_probs[0].float().cpu().numpy().tolist()

                probs = torch.softmax(shifted_logits, dim=-1)
                log_probs_clamped = torch.log(probs + 1e-10)
                entropy = -torch.sum(probs * log_probs_clamped, dim=-1)
                token_entropies = entropy[0].float().cpu().numpy().tolist()

            stats = compute_stats(token_logprobs, token_entropies)

            features = [
                stats['ppl'], stats['log_ppl'],
                stats['entropy_mean'], stats['entropy_std'], stats['entropy_max'], stats['entropy_min'],
                stats['entropy_slope'], stats['entropy_increase_ratio'], stats['entropy_decrease_ratio'],
                stats['entropy_first_quarter'], stats['entropy_last_quarter'], stats['entropy_trend_change'],
                stats['log_prob_mean'], stats['log_prob_std'], stats['log_prob_max'], stats['log_prob_min'],
                stats['log_prob_slope'], stats['log_prob_increase_ratio'], stats['log_prob_decrease_ratio'],
                stats['log_prob_first_quarter'], stats['log_prob_last_quarter'], stats['log_prob_trend_change'],
                stats['seq_len'], stage_idx, tokens,
            ]

            all_features.append(features)
            all_labels.append(1 if item.get('is_correct', False) else 0)
            all_stages.append(stage_idx)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    return np.array(all_features), np.array(all_labels), np.array(all_stages)


def eval_cascade(clf, scaler, features: np.ndarray, labels: np.ndarray, thresholds: List[float] = None):
    """Evaluate cascade accuracy."""
    from itertools import product

    n_stages = len(TOKEN_LEVELS)
    n_samples = len(features) // n_stages

    features_scaled = scaler.transform(features)
    probs = clf.predict_proba(features_scaled)[:, 1]
    probs = probs.reshape(n_samples, n_stages)
    gt = labels.reshape(n_samples, n_stages)

    def compute_cascade_acc(ths):
        correct = 0
        for i in range(n_samples):
            decided = False
            stage_probs = []
            for stage_idx in range(n_stages):
                prob = probs[i, stage_idx]
                stage_probs.append((stage_idx, prob))
                if prob >= ths[stage_idx]:
                    correct += gt[i, stage_idx]
                    decided = True
                    break
            if not decided:
                best_stage, _ = max(stage_probs, key=lambda x: x[1])
                correct += gt[i, best_stage]
        return correct / n_samples

    if thresholds is None:
        # Search for best thresholds
        threshold_candidates = [round(i * 0.05, 2) for i in range(21)]
        best_acc = 0
        best_thresholds = None
        for combo in product(threshold_candidates, repeat=n_stages):
            ths = list(combo)
            acc = compute_cascade_acc(ths)
            if acc > best_acc:
                best_acc = acc
                best_thresholds = ths
    else:
        best_thresholds = thresholds
        best_acc = compute_cascade_acc(thresholds)

    oracle_correct = sum(1 for i in range(n_samples) if any(gt[i, :] == 1))
    oracle_acc = oracle_correct / n_samples

    stage_acc = {}
    stage_auc = {}
    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        stage_labels = gt[:, stage_idx]
        stage_probs_flat = probs[:, stage_idx]
        stage_acc[tokens] = float(np.mean(stage_labels))
        try:
            stage_auc[tokens] = float(roc_auc_score(stage_labels, stage_probs_flat))
        except ValueError:
            stage_auc[tokens] = 0.5

    return best_acc, best_thresholds, {
        'oracle': oracle_acc,
        'baseline': stage_acc,
        'auc': stage_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPL classifier on all subsets")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing classifier.pkl")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_DeepSeek-R1-Distill-Qwen-7B",
                        help="Base directory with subset folders")
    parser.add_argument("--subsets", type=str, default=None,
                        help="Comma-separated list of subsets to evaluate (default: all)")
    parser.add_argument("--llm-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Path to LLM for feature extraction")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-saved-thresholds", action="store_true",
                        help="Use thresholds from training (otherwise re-search on test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (default: model-dir/eval_results.json)")

    args = parser.parse_args()

    # Load classifier
    model_path = os.path.join(args.model_dir, "classifier.pkl")
    if not os.path.exists(model_path):
        logger.error(f"Classifier not found: {model_path}")
        return

    logger.info(f"Loading classifier from {model_path}")
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    clf = saved['classifier']
    scaler = saved['scaler']
    saved_thresholds = saved.get('thresholds', None)

    logger.info(f"Classifier type: {type(clf).__name__}")
    if saved_thresholds:
        logger.info(f"Saved thresholds: {saved_thresholds}")

    # Determine subsets to evaluate
    if args.subsets:
        eval_subsets = [s.strip() for s in args.subsets.split(',')]
    else:
        eval_subsets = SUBSETS

    logger.info(f"Will evaluate on: {eval_subsets}")

    # Load LLM
    logger.info(f"Loading LLM from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    logger.info("LLM loaded")

    # Evaluate on each subset
    results = {}
    print()
    print("=" * 80)
    print(f"{'Subset':<25} {'Cascade':>10} {'Oracle':>10} {'T0':>8} {'T100':>8} {'T500':>8} {'T1000':>8}")
    print("-" * 80)

    for subset in eval_subsets:
        subset_dir = os.path.join(args.data_dir, subset)
        if not os.path.exists(subset_dir):
            logger.warning(f"Subset directory not found: {subset_dir}")
            continue

        test_data = load_json_data(subset_dir, split="test")
        if not test_data or TOKEN_LEVELS[0] not in test_data:
            logger.warning(f"No test data for {subset}")
            continue

        n_samples = len(test_data[TOKEN_LEVELS[0]])
        logger.info(f"\nEvaluating {subset} ({n_samples} samples)...")

        # Extract features
        X_test, y_test, _ = extract_features(model, tokenizer, test_data, args.device, args.max_length)

        # Evaluate
        thresholds = saved_thresholds if args.use_saved_thresholds else None
        cascade_acc, best_thresholds, detailed = eval_cascade(clf, scaler, X_test, y_test, thresholds)

        results[subset] = {
            'n_samples': n_samples,
            'cascade_acc': cascade_acc,
            'oracle_acc': detailed['oracle'],
            'thresholds': best_thresholds,
            'per_stage_baseline': detailed['baseline'],
            'per_stage_auc': detailed['auc'],
        }

        baseline = detailed['baseline']
        print(f"{subset:<25} {cascade_acc:>10.4f} {detailed['oracle']:>10.4f} "
              f"{baseline[0]:>8.4f} {baseline[100]:>8.4f} {baseline[500]:>8.4f} {baseline[1000]:>8.4f}")

    print("-" * 80)

    # Compute average
    if results:
        avg_cascade = np.mean([r['cascade_acc'] for r in results.values()])
        avg_oracle = np.mean([r['oracle_acc'] for r in results.values()])
        print(f"{'Average':<25} {avg_cascade:>10.4f} {avg_oracle:>10.4f}")
        results['_average'] = {'cascade_acc': avg_cascade, 'oracle_acc': avg_oracle}

    print("=" * 80)

    # Save results
    output_path = args.output or os.path.join(args.model_dir, "eval_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
