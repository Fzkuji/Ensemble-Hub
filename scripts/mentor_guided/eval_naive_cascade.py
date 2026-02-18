#!/usr/bin/env python3
"""
Naive LLM Cascade Baseline (FrugalGPT-style).

Routing logic:
  1. Run 7B intern without any hints (T0)
  2. Use PPL/entropy classifier at T0 to decide "sufficient"
  3. If sufficient  → return 7B answer  (low cost)
  4. If insufficient → fallback to 32B full reasoning (high cost)

Compare against Tandem which instead gives hints from 32B to 7B.

Usage:
    python eval_naive_cascade.py \
        --model-dir /path/to/classifier/all \
        --data-dir  /path/to/tandem/collected/hendrycks_math_... \
        --full-reasoning /path/to/truncated_cot_hendrycks_math_.../full_reasoning_test.json \
        --llm-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --device cuda:0
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Params in billions (for TFLOPs cost: 2 * B * tokens / 1000)
PARAMS_7B  = 7.0
PARAMS_32B = 32.0


# ---------------------------------------------------------------------------
# Feature extraction (same as eval_ppl_classifier.py)
# ---------------------------------------------------------------------------

def compute_stats(token_logprobs, token_entropies):
    if not token_logprobs or not token_entropies:
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

    n = len(token_logprobs)
    ent_arr = np.array(token_entropies)
    lp_arr  = np.array(token_logprobs)

    def slope(a): return np.polyfit(np.arange(len(a)), a, 1)[0] if len(a) >= 2 else 0.0
    def ratio(a, inc): return (np.sum(np.diff(a) > 0) if inc else np.sum(np.diff(a) < 0)) / max(len(a) - 1, 1)
    def trend_changes(a): return int(np.sum(np.abs(np.diff(np.sign(np.diff(a)))) > 0)) if len(a) >= 3 else 0
    q1 = max(1, n // 4)

    avg_lp = np.mean(lp_arr)
    return {
        'ppl':                    float(np.exp(-avg_lp)),
        'log_ppl':                float(-avg_lp),
        'entropy_mean':           float(np.mean(ent_arr)),
        'entropy_std':            float(np.std(ent_arr)),
        'entropy_max':            float(np.max(ent_arr)),
        'entropy_min':            float(np.min(ent_arr)),
        'entropy_slope':          float(slope(ent_arr)),
        'entropy_increase_ratio': float(ratio(ent_arr, True)),
        'entropy_decrease_ratio': float(ratio(ent_arr, False)),
        'entropy_first_quarter':  float(np.mean(ent_arr[:q1])),
        'entropy_last_quarter':   float(np.mean(ent_arr[-q1:])),
        'entropy_trend_change':   float(trend_changes(ent_arr)),
        'log_prob_mean':          float(np.mean(lp_arr)),
        'log_prob_std':           float(np.std(lp_arr)),
        'log_prob_max':           float(np.max(lp_arr)),
        'log_prob_min':           float(np.min(lp_arr)),
        'log_prob_slope':         float(slope(lp_arr)),
        'log_prob_increase_ratio':float(ratio(lp_arr, True)),
        'log_prob_decrease_ratio':float(ratio(lp_arr, False)),
        'log_prob_first_quarter': float(np.mean(lp_arr[:q1])),
        'log_prob_last_quarter':  float(np.mean(lp_arr[-q1:])),
        'log_prob_trend_change':  float(trend_changes(lp_arr)),
        'seq_len':                n,
    }


def extract_t0_features(model, tokenizer, items, device, max_length=1024):
    """Extract PPL/entropy features at T0 (no hints) for each sample."""
    features = []
    for item in tqdm(items, desc="Extracting T0 features", ncols=80):
        question = item['question']
        text = f"Question: {question}\n\nAnswer:"
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            logits  = outputs.logits
            shifted_logits = logits[:, :-1, :]
            shifted_ids    = input_ids[:, 1:]
            log_probs      = torch.log_softmax(shifted_logits, dim=-1)
            token_lp       = log_probs.gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
            token_logprobs = token_lp[0].float().cpu().numpy().tolist()
            probs          = torch.softmax(shifted_logits, dim=-1)
            entropy        = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            token_entropies = entropy[0].float().cpu().numpy().tolist()

        stats = compute_stats(token_logprobs, token_entropies)
        # stage_idx=0, tokens=0 (same convention as training features)
        feat = [
            stats['ppl'], stats['log_ppl'],
            stats['entropy_mean'], stats['entropy_std'], stats['entropy_max'], stats['entropy_min'],
            stats['entropy_slope'], stats['entropy_increase_ratio'], stats['entropy_decrease_ratio'],
            stats['entropy_first_quarter'], stats['entropy_last_quarter'], stats['entropy_trend_change'],
            stats['log_prob_mean'], stats['log_prob_std'], stats['log_prob_max'], stats['log_prob_min'],
            stats['log_prob_slope'], stats['log_prob_increase_ratio'], stats['log_prob_decrease_ratio'],
            stats['log_prob_first_quarter'], stats['log_prob_last_quarter'], stats['log_prob_trend_change'],
            stats['seq_len'],
            0,   # stage_idx = 0
            0,   # tokens = 0
        ]
        features.append(feat)
    return np.array(features)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Naive Cascade Baseline (FrugalGPT-style)")
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing classifier.pkl (same as Tandem's classifier)")
    parser.add_argument("--data-dir", required=True,
                        help="Tandem collected data dir (contains per-subset folders with tokens0.json)")
    parser.add_argument("--full-reasoning", required=True,
                        help="Path to truncated CoT full_reasoning_test.json (32B full results)")
    parser.add_argument("--llm-path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="7B model path for feature extraction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--split", default="test")
    parser.add_argument("--subsets", default=None,
                        help="Comma-separated subsets (default: all 7 MATH subsets)")
    parser.add_argument("--t0-threshold", type=float, default=None,
                        help="Classifier threshold at T0 to decide escalation "
                             "(default: use saved threshold from classifier.pkl)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load classifier
    # ------------------------------------------------------------------
    clf_path = os.path.join(args.model_dir, "classifier.pkl")
    logger.info(f"Loading classifier from {clf_path}")
    with open(clf_path, 'rb') as f:
        saved = pickle.load(f)
    clf     = saved['classifier']
    scaler  = saved['scaler']
    saved_thresholds = saved.get('thresholds', None)

    # T0 threshold: use saved[0] if available, else 0.5
    if args.t0_threshold is not None:
        t0_thresh = args.t0_threshold
    elif saved_thresholds is not None:
        t0_thresh = saved_thresholds[0]   # threshold for stage 0 (T0)
        logger.info(f"Using saved T0 threshold: {t0_thresh:.2f} (all thresholds: {saved_thresholds})")
    else:
        t0_thresh = 0.5
        logger.info("No saved thresholds found, using T0 threshold = 0.5")

    # ------------------------------------------------------------------
    # Load 32B full reasoning data (build question → is_correct + tokens lookup)
    # ------------------------------------------------------------------
    logger.info(f"Loading 32B full reasoning from {args.full_reasoning}")
    with open(args.full_reasoning, 'r') as f:
        full_data = json.load(f)

    full_lookup = {}  # question_str -> {is_correct, total_tokens}
    for item in full_data:
        q = item['question'].strip()
        full_lookup[q] = {
            'is_correct':   item.get('is_correct', False),
            'total_tokens': item.get('total_tokens', 2872),
        }
    logger.info(f"Loaded {len(full_lookup)} samples from 32B full reasoning data")

    # ------------------------------------------------------------------
    # Load 7B model for feature extraction
    # ------------------------------------------------------------------
    logger.info(f"Loading 7B model from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_path, torch_dtype=torch.float16, device_map=args.device
    )
    model.eval()

    # ------------------------------------------------------------------
    # Evaluate per subset
    # ------------------------------------------------------------------
    eval_subsets = [s.strip() for s in args.subsets.split(',')] if args.subsets else SUBSETS

    all_results = []
    subset_stats = {}

    for subset in eval_subsets:
        t0_path = os.path.join(args.data_dir, subset, args.split, "tokens0.json")
        if not os.path.exists(t0_path):
            logger.warning(f"T0 data not found: {t0_path}, skipping")
            continue

        with open(t0_path, 'r') as f:
            t0_items = json.load(f)
        logger.info(f"[{subset}] Loaded {len(t0_items)} T0 samples")

        # Extract T0 features using 7B
        feats = extract_t0_features(model, tokenizer, t0_items, args.device, args.max_length)
        feats_scaled = scaler.transform(feats)
        probs = clf.predict_proba(feats_scaled)[:, 1]  # P(correct)

        # Cascade decisions
        n_stop   = 0  # stopped at T0 (7B)
        n_escal  = 0  # escalated to 32B full
        n_miss   = 0  # 32B data not found (fallback: use 32B as wrong)
        correct  = 0

        cost_7b_total  = 0.0   # TFLOPs from 7B
        cost_32b_total = 0.0   # TFLOPs from 32B (escalated only)

        for i, item in enumerate(t0_items):
            q = item['question'].strip()
            t0_correct     = item.get('is_correct', False)
            t0_intern_len  = item.get('intern_length', 500)   # 7B output tokens at T0
            t0_cost        = 2 * PARAMS_7B * t0_intern_len / 1000.0   # TFLOPs

            confident = probs[i] >= t0_thresh

            if confident:
                # Stop at T0: use 7B answer
                n_stop += 1
                correct += int(t0_correct)
                cost_7b_total += t0_cost
            else:
                # Escalate: pay 7B cost (already run for features) + 32B full cost
                n_escal += 1
                cost_7b_total += t0_cost

                full_info = full_lookup.get(q)
                if full_info is None:
                    n_miss += 1
                    # Fallback: treat as wrong
                    full_tokens = 2872
                else:
                    correct += int(full_info['is_correct'])
                    full_tokens = full_info.get('total_tokens', 2872)

                cost_32b_total += 2 * PARAMS_32B * full_tokens / 1000.0

            all_results.append({
                'question': q,
                'subset':   subset,
                'confident': confident,
                't0_correct': t0_correct,
                'escalated': not confident,
            })

        n = len(t0_items)
        acc         = correct / n if n > 0 else 0.0
        total_cost  = (cost_7b_total + cost_32b_total) / n if n > 0 else 0.0
        stop_rate   = n_stop / n if n > 0 else 0.0

        subset_stats[subset] = {
            'n':            n,
            'accuracy':     acc,
            'stop_rate':    stop_rate,
            'escalation_rate': n_escal / n if n > 0 else 0.0,
            'avg_cost':     total_cost,
            'n_miss':       n_miss,
        }
        logger.info(f"[{subset}] Acc={acc:.4f}, Stop@T0={stop_rate:.1%}, "
                    f"Escalated={n_escal/n:.1%}, AvgCost={total_cost:.2f} TFLOPs, Miss={n_miss}")

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    total_n       = sum(s['n'] for s in subset_stats.values())
    total_correct = sum(s['accuracy'] * s['n'] for s in subset_stats.values())
    total_cost_w  = sum(s['avg_cost'] * s['n'] for s in subset_stats.values())
    total_stop    = sum(s['stop_rate'] * s['n'] for s in subset_stats.values())
    total_escal   = sum(s['escalation_rate'] * s['n'] for s in subset_stats.values())

    overall_acc   = total_correct / total_n if total_n else 0
    overall_cost  = total_cost_w  / total_n if total_n else 0
    overall_stop  = total_stop    / total_n if total_n else 0
    overall_escal = total_escal   / total_n if total_n else 0

    print("\n" + "=" * 70)
    print("Naive LLM Cascade Baseline Results (FrugalGPT-style)")
    print("=" * 70)
    print(f"T0 threshold: {t0_thresh:.2f}")
    print(f"Overall Accuracy:       {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Stop at T0 (7B):        {overall_stop:.1%}")
    print(f"Escalated to 32B Full:  {overall_escal:.1%}")
    print(f"Avg Cost per sample:    {overall_cost:.2f} TFLOPs")
    print()
    print(f"{'Subset':<28} {'Acc':>7} {'Stop@T0':>9} {'Escalated':>10} {'Cost(TF)':>9}")
    print("-" * 70)
    for subset in eval_subsets:
        if subset not in subset_stats:
            continue
        s = subset_stats[subset]
        print(f"  {subset:<26} {s['accuracy']:>7.4f} {s['stop_rate']:>9.1%} "
              f"{s['escalation_rate']:>10.1%} {s['avg_cost']:>9.2f}")
    print(f"  {'Overall':<26} {overall_acc:>7.4f} {overall_stop:>9.1%} "
          f"{overall_escal:>10.1%} {overall_cost:>9.2f}")
    print("=" * 70)

    print("\nComparison (from paper / previous experiments):")
    print(f"  SLM Standard CoT (7B):      77.14%,  38.25 TFLOPs")
    print(f"  Budget Forcing T1000 (32B): 82.42%,  105.22 TFLOPs")
    print(f"  Naive Cascade (ours):       {overall_acc*100:.2f}%,  {overall_cost:.2f} TFLOPs")
    print(f"  Tandem (ours, cascade):     83.46%,  99.72 TFLOPs")
    print(f"  LLM Full (32B):             80.90%,  168.35 TFLOPs")

    # Save results
    out_path = os.path.join(args.model_dir, "naive_cascade_results.json")
    with open(out_path, 'w') as f:
        json.dump({
            'overall': {
                'accuracy':        overall_acc,
                'avg_cost':        overall_cost,
                'stop_rate':       overall_stop,
                'escalation_rate': overall_escal,
                't0_threshold':    t0_thresh,
            },
            'per_subset': subset_stats,
        }, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
