#!/usr/bin/env python3
"""
Comprehensive classifier evaluation for rebuttal / paper reporting.

Produces paper-ready metrics for the cost-aware sufficiency classifier:
  - Per-stage binary classification: AUC-ROC, Accuracy, Precision, Recall, F1
  - Cascade accuracy with optimal thresholds
  - Stage selection distribution (by subject and difficulty)
  - Failure case analysis (false early stop / false continuation)
  - Feature importance (PPL classifier only)
  - LaTeX-formatted tables

Usage:
    # Evaluate PPL classifier on all subsets (requires GPU for feature extraction)
    python eval_classifier_report.py \
        --data-dir /path/to/hendrycks_math_split \
        --classifier ppl \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

    # Evaluate MLP classifier on all subsets
    python eval_classifier_report.py \
        --data-dir /path/to/hendrycks_math_split \
        --classifier mlp \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

    # Evaluate on a single subset
    python eval_classifier_report.py \
        --data-dir /path/to/hendrycks_math_split \
        --classifier ppl --subset algebra

    # Use pre-saved results (no GPU needed) — reads results.json
    python eval_classifier_report.py \
        --data-dir /path/to/hendrycks_math_split \
        --from-saved
"""

import argparse
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]
STAGE_NAMES = ["T0 (No Hint)", "T100 (Low)", "T500 (Med)", "T1000 (High)"]

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# MATH difficulty levels (parsed from problem filenames or metadata)
DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_json_data(data_dir: str, split: str = "test") -> Optional[Dict[int, List[Dict]]]:
    """Load JSON data for all token levels."""
    data = {}
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, split, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data[tokens] = json.load(f)
        else:
            logger.warning(f"File not found: {filepath}")
    return data if data and TOKEN_LEVELS[0] in data else None


def parse_difficulty(item: dict) -> Optional[int]:
    """Try to extract difficulty level from a data item."""
    # Directly available
    if "difficulty" in item:
        return int(item["difficulty"])
    if "level" in item:
        m = re.search(r"(\d)", str(item["level"]))
        if m:
            return int(m.group(1))
    # From filename / id
    for key in ("id", "filename", "problem_id", "source"):
        if key in item:
            m = re.search(r"level(\d)", str(item[key]), re.IGNORECASE)
            if m:
                return int(m.group(1))
    return None


# ──────────────────────────────────────────────────────────────────────
# PPL feature extraction (mirrors train_ppl_classifier.py)
# ──────────────────────────────────────────────────────────────────────

def compute_trend_stats(values: np.ndarray) -> Dict[str, float]:
    n = len(values)
    if n < 2:
        return {
            "slope": 0.0, "increase_ratio": 0.5, "decrease_ratio": 0.5,
            "first_quarter": float(values[0]) if n > 0 else 0.0,
            "last_quarter": float(values[0]) if n > 0 else 0.0,
            "trend_change": 0.0,
        }
    x = np.arange(n)
    slope = np.polyfit(x, values, 1)[0]
    diffs = np.diff(values)
    quarter = max(1, n // 4)
    return {
        "slope": float(slope),
        "increase_ratio": float(np.mean(diffs > 0)),
        "decrease_ratio": float(np.mean(diffs < 0)),
        "first_quarter": float(np.mean(values[:quarter])),
        "last_quarter": float(np.mean(values[-quarter:])),
        "trend_change": float(np.mean(values[-quarter:]) - np.mean(values[:quarter])),
    }


def compute_ppl_stats(token_logprobs: List[float], token_entropies: List[float]) -> Dict[str, float]:
    if not token_logprobs or len(token_logprobs) == 0:
        return {k: 0.0 for k in [
            "ppl", "log_ppl",
            "entropy_mean", "entropy_std", "entropy_max", "entropy_min",
            "entropy_slope", "entropy_increase_ratio", "entropy_decrease_ratio",
            "entropy_first_quarter", "entropy_last_quarter", "entropy_trend_change",
            "log_prob_mean", "log_prob_std", "log_prob_max", "log_prob_min",
            "log_prob_slope", "log_prob_increase_ratio", "log_prob_decrease_ratio",
            "log_prob_first_quarter", "log_prob_last_quarter", "log_prob_trend_change",
            "seq_len",
        ]}
    logprobs = np.array(token_logprobs)
    entropies = np.array(token_entropies)
    mean_lp = np.mean(logprobs)
    ppl = np.exp(-mean_lp)
    lp_trend = compute_trend_stats(logprobs)
    ent_trend = compute_trend_stats(entropies)
    return {
        "ppl": float(ppl), "log_ppl": float(np.log(ppl + 1e-10)),
        "entropy_mean": float(np.mean(entropies)), "entropy_std": float(np.std(entropies)),
        "entropy_max": float(np.max(entropies)), "entropy_min": float(np.min(entropies)),
        "entropy_slope": ent_trend["slope"],
        "entropy_increase_ratio": ent_trend["increase_ratio"],
        "entropy_decrease_ratio": ent_trend["decrease_ratio"],
        "entropy_first_quarter": ent_trend["first_quarter"],
        "entropy_last_quarter": ent_trend["last_quarter"],
        "entropy_trend_change": ent_trend["trend_change"],
        "log_prob_mean": float(np.mean(logprobs)), "log_prob_std": float(np.std(logprobs)),
        "log_prob_max": float(np.max(logprobs)), "log_prob_min": float(np.min(logprobs)),
        "log_prob_slope": lp_trend["slope"],
        "log_prob_increase_ratio": lp_trend["increase_ratio"],
        "log_prob_decrease_ratio": lp_trend["decrease_ratio"],
        "log_prob_first_quarter": lp_trend["first_quarter"],
        "log_prob_last_quarter": lp_trend["last_quarter"],
        "log_prob_trend_change": lp_trend["trend_change"],
        "seq_len": len(logprobs),
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


# ──────────────────────────────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────────────────────────────

def get_predictions_ppl(
    data: Dict[int, List[Dict]],
    model_dir: str,
    llm_path: str,
    device: str,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probs, gt) each of shape [n_samples, n_stages]."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load classifier
    with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
        saved = pickle.load(f)
    clf = saved["classifier"]
    scaler = saved["scaler"]

    # Load LLM
    logger.info(f"Loading LLM from {llm_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        llm_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    n_samples = len(data[TOKEN_LEVELS[0]])
    all_features, all_labels = [], []

    for i in tqdm(range(n_samples), desc="Extracting PPL features"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = data[tokens][i]
            question = item["question"]
            mentor_response = item.get("mentor_response", "")
            text = (f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
                    if mentor_response else f"Question: {question}\n\nAnswer:")

            encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                logits = outputs.logits
                shifted_logits = logits[:, :-1, :]
                shifted_ids = input_ids[:, 1:]

                log_probs = torch.log_softmax(shifted_logits, dim=-1)
                token_lp = log_probs.gather(dim=-1, index=shifted_ids.unsqueeze(-1)).squeeze(-1)
                token_logprobs = token_lp[0].float().cpu().numpy().tolist()

                probs_dist = torch.softmax(shifted_logits, dim=-1)
                entropy = -torch.sum(probs_dist * torch.log(probs_dist + 1e-10), dim=-1)
                token_entropies = entropy[0].float().cpu().numpy().tolist()

            stats = compute_ppl_stats(token_logprobs, token_entropies)
            features = [stats[k] for k in FEATURE_KEYS] + [stage_idx, tokens]
            all_features.append(features)
            all_labels.append(1 if item.get("is_correct", False) else 0)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    X = np.array(all_features)
    y = np.array(all_labels)
    X_scaled = scaler.transform(X)
    prob_pos = clf.predict_proba(X_scaled)[:, 1]

    n_stages = len(TOKEN_LEVELS)
    return prob_pos.reshape(n_samples, n_stages), y.reshape(n_samples, n_stages)


def get_predictions_mlp(
    data: Dict[int, List[Dict]],
    model_dir: str,
    llm_path: str,
    device: str,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probs, gt) each of shape [n_samples, n_stages]."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class MentorClassifierHead(torch.nn.Module):
        def __init__(self, hidden_size, num_stages=4, dropout=0.1):
            super().__init__()
            self.stage_embedding = torch.nn.Embedding(num_stages, 64)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size + 64, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(256, 2),
            )

        def forward(self, hidden_state, stage):
            stage_embed = self.stage_embedding(stage)
            x = torch.cat([hidden_state, stage_embed], dim=-1)
            return self.classifier(x)

    logger.info(f"Loading LLM from {llm_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        llm_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    base_model.eval()

    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    hidden_size = base_model.config.hidden_size
    classifier_head = MentorClassifierHead(hidden_size, num_stages=4, dropout=0.1).to(device)
    classifier_head.load_state_dict(ckpt["classifier"])
    classifier_head.eval()

    n_samples = len(data[TOKEN_LEVELS[0]])
    all_probs, all_labels = [], []

    for i in tqdm(range(n_samples), desc="MLP inference"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = data[tokens][i]
            question = item["question"]
            mentor_response = item.get("mentor_response", "")
            text = (f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
                    if mentor_response else f"Question: {question}\n\nAnswer:")

            encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1][:, -1, :]
                stage_tensor = torch.tensor([stage_idx], device=device)
                logits = classifier_head(hidden, stage_tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].item()

            all_probs.append(prob)
            all_labels.append(1 if item.get("is_correct", False) else 0)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    n_stages = len(TOKEN_LEVELS)
    return (np.array(all_probs).reshape(n_samples, n_stages),
            np.array(all_labels).reshape(n_samples, n_stages))


# ──────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Compute binary classification metrics at a given threshold."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = 0.5
    return metrics


def search_best_thresholds(probs: np.ndarray, gt: np.ndarray):
    """Grid-search thresholds to maximize cascade accuracy.

    Returns: (best_acc, best_thresholds, cascade_decisions)
        cascade_decisions: np.ndarray of shape [n_samples] — selected stage index per sample
    """
    n_samples, n_stages = probs.shape
    threshold_candidates = [round(i * 0.05, 2) for i in range(21)]

    def cascade_eval(thresholds):
        correct = 0
        decisions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            decided = False
            best_stage, best_prob = 0, -1
            for s in range(n_stages):
                if probs[i, s] > best_prob:
                    best_prob = probs[i, s]
                    best_stage = s
                if probs[i, s] >= thresholds[s]:
                    correct += gt[i, s]
                    decisions[i] = s
                    decided = True
                    break
            if not decided:
                correct += gt[i, best_stage]
                decisions[i] = best_stage
        return correct / n_samples, decisions

    best_acc, best_thresholds, best_decisions = 0, None, None
    for combo in product(threshold_candidates, repeat=n_stages):
        ths = list(combo)
        acc, decisions = cascade_eval(ths)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = ths
            best_decisions = decisions

    return best_acc, best_thresholds, best_decisions


def compute_stage_distribution(decisions: np.ndarray, n_stages: int) -> Dict[int, float]:
    """Fraction of samples assigned to each stage."""
    counts = np.bincount(decisions, minlength=n_stages)
    total = len(decisions)
    return {TOKEN_LEVELS[s]: float(counts[s] / total) for s in range(n_stages)}


def compute_failure_analysis(decisions: np.ndarray, gt: np.ndarray):
    """Analyze routing errors.

    - false_early_stop: stopped early but wrong, and a later stage was correct
    - false_continuation: went to a later stage, but an earlier stage was already correct
    - correct_routing: stopped and answered correctly
    - unavoidable_wrong: wrong at all stages anyway
    """
    n_samples, n_stages = gt.shape
    false_early = 0
    false_cont = 0
    correct_routing = 0
    unavoidable_wrong = 0

    for i in range(n_samples):
        chosen = decisions[i]
        is_correct = gt[i, chosen] == 1
        any_correct = np.any(gt[i] == 1)

        if is_correct:
            # Check if we could have stopped earlier
            earlier_correct = any(gt[i, s] == 1 for s in range(chosen))
            if earlier_correct:
                false_cont += 1  # wasted cost — could have stopped earlier
            else:
                correct_routing += 1
        else:
            if any_correct:
                false_early += 1  # stopped at wrong stage, a better one existed
            else:
                unavoidable_wrong += 1  # no stage could solve this

    total = n_samples
    return {
        "false_early_stop": {"count": false_early, "pct": false_early / total},
        "false_continuation": {"count": false_cont, "pct": false_cont / total},
        "correct_routing": {"count": correct_routing, "pct": correct_routing / total},
        "unavoidable_wrong": {"count": unavoidable_wrong, "pct": unavoidable_wrong / total},
    }


def compute_feature_importance(model_dir: str) -> Optional[Dict[str, float]]:
    """Extract feature importance from a saved PPL (sklearn) classifier."""
    pkl_path = os.path.join(model_dir, "classifier.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    clf = saved["classifier"]

    # Feature names (23 PPL stats + stage_idx + tokens)
    feature_names = FEATURE_KEYS + ["stage_idx", "token_level"]

    importances = None
    clf_type = type(clf).__name__

    if hasattr(clf, "feature_importances_"):
        # GradientBoosting, RandomForest
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # LogisticRegression
        importances = np.abs(clf.coef_[0])
    else:
        return None

    if importances is not None and len(importances) == len(feature_names):
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return {name: float(imp) for name, imp in ranked}
    return None


# ──────────────────────────────────────────────────────────────────────
# Output formatters
# ──────────────────────────────────────────────────────────────────────

def print_section(title: str, width: int = 80):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def format_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_per_stage_metrics(all_metrics: Dict[str, Dict]):
    """Pretty-print per-stage classification metrics."""
    print_section("Per-Stage Binary Classification Metrics")
    header = f"{'Stage':<22} {'AUC-ROC':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Pos%':>8}"
    print(header)
    print("-" * len(header))
    for stage_name, m in all_metrics.items():
        print(f"{stage_name:<22} {m['auc_roc']:>8.4f} {m['accuracy']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m.get('positive_rate', 0):>8.4f}")


def print_latex_per_stage(all_metrics: Dict[str, Dict]):
    """Output LaTeX table for per-stage metrics."""
    print_section("LaTeX Table: Per-Stage Classifier Metrics")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Per-stage binary classification performance of the sufficiency classifier.}")
    print(r"\label{tab:classifier_metrics}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Stage & AUC-ROC & Accuracy & Precision & Recall & F1 & Positive\% \\")
    print(r"\midrule")
    for stage_name, m in all_metrics.items():
        pos_rate = m.get("positive_rate", 0)
        print(f"{stage_name} & {m['auc_roc']:.4f} & {m['accuracy']:.4f} & "
              f"{m['precision']:.4f} & {m['recall']:.4f} & {m['f1']:.4f} & "
              f"{pos_rate:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_stage_distribution(dist: Dict[int, float], label: str = ""):
    """Pretty-print stage selection distribution."""
    print(f"  {label}")
    for tokens, frac in dist.items():
        bar = "█" * int(frac * 40)
        print(f"    T{tokens:<5} {format_pct(frac):>8}  {bar}")


def print_latex_stage_dist_by_subject(all_dists: Dict[str, Dict[int, float]]):
    """Output LaTeX table for stage distribution by subject."""
    print_section("LaTeX Table: Stage Selection Distribution by Subject")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Fraction of test samples assigned to each guidance stage by the cascade classifier.}")
    print(r"\label{tab:stage_distribution}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Subject & T0 & T100 & T500 & T1000 \\")
    print(r"\midrule")
    for subset, dist in all_dists.items():
        row = f"{subset}"
        for tokens in TOKEN_LEVELS:
            row += f" & {dist.get(tokens, 0):.2%}"
        print(row + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_failure_analysis(fa: Dict):
    print_section("Failure Case Analysis (Cascade Routing)")
    for key, val in fa.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<25} {val['count']:>6} samples  ({format_pct(val['pct']):>7})")


def print_feature_importance(fi: Dict[str, float], top_n: int = 15):
    print_section(f"Feature Importance (Top {top_n})")
    for i, (name, imp) in enumerate(fi.items()):
        if i >= top_n:
            break
        bar = "█" * int(imp / max(fi.values()) * 30)
        print(f"  {i+1:>2}. {name:<30} {imp:.6f}  {bar}")


# ──────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ──────────────────────────────────────────────────────────────────────

def evaluate_subset(
    subset: str,
    data: Dict[int, List[Dict]],
    probs: np.ndarray,
    gt: np.ndarray,
) -> Dict:
    """Run full evaluation on one subset. Returns results dict."""
    n_samples, n_stages = gt.shape

    # 1) Per-stage binary metrics
    per_stage = {}
    for s, (tokens, name) in enumerate(zip(TOKEN_LEVELS, STAGE_NAMES)):
        m = compute_binary_metrics(gt[:, s], probs[:, s], threshold=0.5)
        m["positive_rate"] = float(np.mean(gt[:, s]))
        per_stage[name] = m

    # Average across stages
    avg_m = {}
    for key in ["auc_roc", "accuracy", "precision", "recall", "f1"]:
        avg_m[key] = float(np.mean([per_stage[n][key] for n in STAGE_NAMES]))
    avg_m["positive_rate"] = float(np.mean(gt))
    per_stage["Average"] = avg_m

    # 2) Cascade with threshold search
    cascade_acc, best_thresholds, decisions = search_best_thresholds(probs, gt)

    # Oracle
    oracle_acc = float(np.mean(np.any(gt == 1, axis=1)))

    # Best single-stage baseline
    baselines = {TOKEN_LEVELS[s]: float(np.mean(gt[:, s])) for s in range(n_stages)}
    best_baseline = max(baselines.values())

    # 3) Stage selection distribution
    stage_dist = compute_stage_distribution(decisions, n_stages)

    # 4) Failure analysis
    failure = compute_failure_analysis(decisions, gt)

    # 5) Stage distribution by difficulty (if available)
    diff_dist = {}
    difficulties = []
    for i in range(n_samples):
        d = parse_difficulty(data[TOKEN_LEVELS[0]][i])
        difficulties.append(d)

    if any(d is not None for d in difficulties):
        for level in DIFFICULTY_LEVELS:
            indices = [i for i, d in enumerate(difficulties) if d == level]
            if indices:
                sub_decisions = decisions[indices]
                sub_gt = gt[indices]
                sub_correct = float(np.mean([sub_gt[j, sub_decisions[j]] for j in range(len(indices))]))
                diff_dist[level] = {
                    "n_samples": len(indices),
                    "stage_distribution": compute_stage_distribution(sub_decisions, n_stages),
                    "cascade_accuracy": sub_correct,
                }

    return {
        "subset": subset,
        "n_samples": n_samples,
        "per_stage_metrics": per_stage,
        "cascade_accuracy": cascade_acc,
        "oracle_accuracy": oracle_acc,
        "best_baseline": best_baseline,
        "gap": cascade_acc - best_baseline,
        "baselines": baselines,
        "thresholds": best_thresholds,
        "stage_distribution": stage_dist,
        "failure_analysis": failure,
        "difficulty_distribution": diff_dist,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive classifier evaluation for paper / rebuttal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Base directory with subset folders (e.g., hendrycks_math_split)")
    parser.add_argument("--classifier", type=str, default="ppl", choices=["ppl", "mlp"],
                        help="Classifier type to evaluate")
    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Path to base LLM (for feature extraction)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Evaluate single subset (default: all)")
    parser.add_argument("--model-dir-name", type=str, default=None,
                        help="Model subdirectory name (default: {classifier}_model)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: data_dir/classifier_report.json)")
    parser.add_argument("--no-latex", action="store_true", help="Skip LaTeX output")

    args = parser.parse_args()

    eval_subsets = [args.subset] if args.subset else SUBSETS
    model_dir_name = args.model_dir_name or f"{args.classifier}_model"

    # Collect all results
    all_results = {}
    all_stage_dists = {}

    # Aggregated probs and gt for overall metrics
    agg_probs_list = []
    agg_gt_list = []
    agg_decisions_list = []
    agg_data_list = []

    for subset in eval_subsets:
        subset_dir = os.path.join(args.data_dir, subset)
        model_dir = os.path.join(subset_dir, model_dir_name)

        # Also check for unified "all" model
        if not os.path.exists(model_dir):
            alt_model_dir = os.path.join(args.data_dir, "all", model_dir_name)
            if os.path.exists(alt_model_dir):
                model_dir = alt_model_dir
            else:
                logger.warning(
                    f"No model found for {subset}, checked:\n"
                    f"  1) {os.path.join(subset_dir, model_dir_name)}\n"
                    f"  2) {alt_model_dir}\n"
                    f"  Skipping."
                )
                continue

        data = load_json_data(subset_dir, split=args.split)
        if data is None:
            logger.warning(f"No {args.split} data for {subset}, skipping.")
            continue

        n_samples = len(data[TOKEN_LEVELS[0]])
        logger.info(f"Evaluating {subset} ({n_samples} samples, classifier={args.classifier}) ...")

        # Get predictions
        if args.classifier == "ppl":
            probs, gt = get_predictions_ppl(data, model_dir, args.model_path, args.device, args.max_length)
        elif args.classifier == "mlp":
            probs, gt = get_predictions_mlp(data, model_dir, args.model_path, args.device, args.max_length)
        else:
            raise ValueError(f"Unsupported classifier: {args.classifier}")

        result = evaluate_subset(subset, data, probs, gt)
        all_results[subset] = result
        all_stage_dists[subset] = result["stage_distribution"]

        agg_probs_list.append(probs)
        agg_gt_list.append(gt)
        agg_data_list.extend([data[TOKEN_LEVELS[0]][i] for i in range(n_samples)])

        # Per-subset output
        print_section(f"Results: {subset} (N={n_samples})")
        print_per_stage_metrics(result["per_stage_metrics"])
        print(f"\n  Cascade Accuracy:  {result['cascade_accuracy']:.4f}")
        print(f"  Oracle Accuracy:   {result['oracle_accuracy']:.4f}")
        print(f"  Best Baseline:     {result['best_baseline']:.4f}")
        print(f"  Gap (Cascade-BL):  {result['gap']:+.4f}")
        print(f"  Thresholds:        {result['thresholds']}")

        print_section(f"Stage Distribution: {subset}")
        print_stage_distribution(result["stage_distribution"])

        if result["difficulty_distribution"]:
            print_section(f"Stage Distribution by Difficulty: {subset}")
            for level, dd in sorted(result["difficulty_distribution"].items()):
                print(f"\n  Level {level} (N={dd['n_samples']}, Cascade Acc={dd['cascade_accuracy']:.4f}):")
                print_stage_distribution(dd["stage_distribution"], label=f"Level {level}")

        print_failure_analysis(result["failure_analysis"])

    # ── Overall aggregated results ──
    if len(all_results) > 1:
        agg_probs = np.concatenate(agg_probs_list, axis=0)
        agg_gt = np.concatenate(agg_gt_list, axis=0)
        n_stages = len(TOKEN_LEVELS)

        # Overall per-stage metrics
        overall_per_stage = {}
        for s, (tokens, name) in enumerate(zip(TOKEN_LEVELS, STAGE_NAMES)):
            m = compute_binary_metrics(agg_gt[:, s], agg_probs[:, s], threshold=0.5)
            m["positive_rate"] = float(np.mean(agg_gt[:, s]))
            overall_per_stage[name] = m

        avg_m = {}
        for key in ["auc_roc", "accuracy", "precision", "recall", "f1"]:
            avg_m[key] = float(np.mean([overall_per_stage[n][key] for n in STAGE_NAMES]))
        avg_m["positive_rate"] = float(np.mean(agg_gt))
        overall_per_stage["Average"] = avg_m

        # Overall cascade
        overall_cascade_acc, overall_thresholds, overall_decisions = search_best_thresholds(agg_probs, agg_gt)
        overall_oracle = float(np.mean(np.any(agg_gt == 1, axis=1)))
        overall_baselines = {TOKEN_LEVELS[s]: float(np.mean(agg_gt[:, s])) for s in range(n_stages)}
        overall_best_bl = max(overall_baselines.values())
        overall_stage_dist = compute_stage_distribution(overall_decisions, n_stages)
        overall_failure = compute_failure_analysis(overall_decisions, agg_gt)

        print_section(f"OVERALL RESULTS (N={len(agg_probs)})", width=90)
        print_per_stage_metrics(overall_per_stage)
        print(f"\n  Cascade Accuracy:  {overall_cascade_acc:.4f}")
        print(f"  Oracle Accuracy:   {overall_oracle:.4f}")
        print(f"  Best Baseline:     {overall_best_bl:.4f}")
        print(f"  Gap (Cascade-BL):  {overall_cascade_acc - overall_best_bl:+.4f}")
        print(f"  Thresholds:        {overall_thresholds}")

        print_section("Overall Stage Distribution")
        print_stage_distribution(overall_stage_dist)

        print_failure_analysis(overall_failure)

        all_results["_overall"] = {
            "n_samples": len(agg_probs),
            "per_stage_metrics": overall_per_stage,
            "cascade_accuracy": overall_cascade_acc,
            "oracle_accuracy": overall_oracle,
            "best_baseline": overall_best_bl,
            "gap": overall_cascade_acc - overall_best_bl,
            "thresholds": overall_thresholds,
            "stage_distribution": overall_stage_dist,
            "failure_analysis": overall_failure,
        }

        # LaTeX tables
        if not args.no_latex:
            print_latex_per_stage(overall_per_stage)
            print_latex_stage_dist_by_subject(all_stage_dists)

            # LaTeX: per-subset cascade summary
            print_section("LaTeX Table: Per-Subset Cascade Summary")
            print(r"\begin{table}[h]")
            print(r"\centering")
            print(r"\caption{Cascade accuracy and stage distribution across MATH subsets.}")
            print(r"\label{tab:cascade_summary}")
            print(r"\begin{tabular}{lccccccc}")
            print(r"\toprule")
            print(r"Subject & N & T0 & T100 & T500 & T1000 & Oracle & Cascade \\")
            print(r"\midrule")
            for subset in eval_subsets:
                if subset not in all_results:
                    continue
                r = all_results[subset]
                bl = r["baselines"]
                print(f"{subset} & {r['n_samples']} "
                      f"& {bl[0]:.4f} & {bl[100]:.4f} & {bl[500]:.4f} & {bl[1000]:.4f} "
                      f"& {r['oracle_accuracy']:.4f} & {r['cascade_accuracy']:.4f} \\\\")
            print(r"\midrule")
            ov = all_results["_overall"]
            ovbl = {TOKEN_LEVELS[s]: float(np.mean(agg_gt[:, s])) for s in range(n_stages)}
            print(f"Overall & {ov['n_samples']} "
                  f"& {ovbl[0]:.4f} & {ovbl[100]:.4f} & {ovbl[500]:.4f} & {ovbl[1000]:.4f} "
                  f"& {ov['oracle_accuracy']:.4f} & {ov['cascade_accuracy']:.4f} \\\\")
            print(r"\bottomrule")
            print(r"\end{tabular}")
            print(r"\end{table}")

            # LaTeX: difficulty-level distribution (aggregated)
            agg_diff = defaultdict(lambda: {"decisions": [], "gt": []})
            offset = 0
            for subset in eval_subsets:
                if subset not in all_results:
                    continue
                r = all_results[subset]
                n = r["n_samples"]
                subset_dir = os.path.join(args.data_dir, subset)
                sub_data = load_json_data(subset_dir, split=args.split)
                if sub_data is None:
                    offset += n
                    continue
                sub_decisions = overall_decisions[offset:offset + n]
                sub_gt = agg_gt[offset:offset + n]
                for i in range(n):
                    d = parse_difficulty(sub_data[TOKEN_LEVELS[0]][i])
                    if d is not None:
                        agg_diff[d]["decisions"].append(sub_decisions[i])
                        agg_diff[d]["gt"].append(sub_gt[i])
                offset += n

            if agg_diff:
                print_section("LaTeX Table: Stage Distribution by Difficulty Level")
                print(r"\begin{table}[h]")
                print(r"\centering")
                print(r"\caption{Stage selection distribution and cascade accuracy by problem difficulty.}")
                print(r"\label{tab:difficulty_dist}")
                print(r"\begin{tabular}{lccccc}")
                print(r"\toprule")
                print(r"Difficulty & T0 & T100 & T500 & T1000 & Cascade Acc \\")
                print(r"\midrule")
                for level in DIFFICULTY_LEVELS:
                    if level not in agg_diff:
                        continue
                    dd = agg_diff[level]
                    d_arr = np.array(dd["decisions"])
                    g_arr = np.array(dd["gt"])
                    dist = compute_stage_distribution(d_arr, n_stages)
                    casc_acc = float(np.mean([g_arr[j, d_arr[j]] for j in range(len(d_arr))]))
                    print(f"Level {level} (N={len(d_arr)}) "
                          f"& {dist[0]:.2%} & {dist[100]:.2%} & {dist[500]:.2%} & {dist[1000]:.2%} "
                          f"& {casc_acc:.4f} \\\\")
                print(r"\bottomrule")
                print(r"\end{tabular}")
                print(r"\end{table}")

    # Feature importance (PPL only)
    if args.classifier == "ppl":
        # Try per-subset or unified model
        for subset in eval_subsets:
            model_dir = os.path.join(args.data_dir, subset, model_dir_name)
            if not os.path.exists(model_dir):
                model_dir = os.path.join(args.data_dir, "all", model_dir_name)
            fi = compute_feature_importance(model_dir)
            if fi:
                print_feature_importance(fi)
                all_results["_feature_importance"] = fi

                if not args.no_latex:
                    print_section("LaTeX Table: Top-10 Feature Importance")
                    print(r"\begin{table}[h]")
                    print(r"\centering")
                    print(r"\caption{Top-10 most important features for the sufficiency classifier.}")
                    print(r"\label{tab:feature_importance}")
                    print(r"\begin{tabular}{clc}")
                    print(r"\toprule")
                    print(r"Rank & Feature & Importance \\")
                    print(r"\midrule")
                    for i, (name, imp) in enumerate(fi.items()):
                        if i >= 10:
                            break
                        name_escaped = name.replace("_", r"\_")
                        print(f"{i+1} & {name_escaped} & {imp:.6f} \\\\")
                    print(r"\bottomrule")
                    print(r"\end{tabular}")
                    print(r"\end{table}")
                break  # Only need one model for feature importance

    # Save JSON report
    if not all_results:
        logger.error(
            "No subsets were evaluated. Check that:\n"
            f"  1) --data-dir points to the correct directory (got: {args.data_dir})\n"
            f"  2) Trained {args.classifier} classifiers exist in {{subset}}/{model_dir_name}/ or all/{model_dir_name}/\n"
            f"  3) Test data exists in {{subset}}/{args.split}/tokens*.json"
        )
        return

    output_path = args.output or os.path.join(args.data_dir, f"{args.classifier}_classifier_report.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    logger.info(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
