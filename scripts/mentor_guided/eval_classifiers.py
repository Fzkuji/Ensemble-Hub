#!/usr/bin/env python3
"""
Re-evaluate trained classifiers on val data (unfiltered).
Usage: python eval_classifiers.py --data-dir DATA_DIR [--subset SUBSET]
"""

import argparse
import json
import os
import pickle
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKEN_LEVELS = [0, 100, 500, 1000]
SUBSETS = ["algebra", "counting_and_probability", "geometry",
           "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def load_json_data(data_dir: str, split: str = "train") -> Dict[int, List[Dict]]:
    """Load data from JSON files."""
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        split_dir = data_dir

    data = {}
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(split_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath) as f:
                data[tokens] = json.load(f)
    return data


def get_val_data(data_dir: str, val_ratio: float = 0.3) -> Dict[int, List[Dict]]:
    """Get unfiltered val data using the same split as training."""
    from sklearn.model_selection import train_test_split as sk_split

    train_data = load_json_data(data_dir, split="train")
    if not train_data:
        return {}

    n_samples = len(train_data[TOKEN_LEVELS[0]])
    _, val_idx = sk_split(
        np.arange(n_samples), test_size=val_ratio, random_state=42
    )

    val_data = {}
    for tokens in TOKEN_LEVELS:
        if tokens in train_data:
            val_data[tokens] = [train_data[tokens][i] for i in val_idx]

    return val_data


def eval_cascade_lora(
    model_dir: str,
    val_data: Dict[int, List[Dict]],
    device: str = "cuda",
) -> Tuple[float, List[float], Dict]:
    """Evaluate LoRA classifier on val data."""
    from peft import PeftModel

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load classifier head
    checkpoint = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)

    # Simple classifier head
    hidden_size = base_model.config.hidden_size
    classifier_head = torch.nn.Sequential(
        torch.nn.Linear(hidden_size + 4, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 2),
    ).to(device)
    classifier_head.load_state_dict(checkpoint['classifier'])
    classifier_head.eval()

    # Collect predictions
    n_samples = len(val_data[TOKEN_LEVELS[0]])
    all_probs = {tokens: [] for tokens in TOKEN_LEVELS}
    gt = {tokens: [] for tokens in TOKEN_LEVELS}

    base_model.eval()
    for i in tqdm(range(n_samples), desc="LoRA eval"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = val_data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            if mentor_response:
                text = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
            else:
                text = f"Question: {question}\n\nAnswer:"

            encoded = tokenizer(
                text, truncation=True, max_length=2048,
                padding=False, return_tensors="pt",
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1][:, -1, :]

                stage_onehot = torch.zeros(1, 4, device=device)
                stage_onehot[0, stage_idx] = 1.0
                combined = torch.cat([hidden, stage_onehot], dim=-1)

                logits = classifier_head(combined)
                prob = torch.softmax(logits, dim=1)[0, 1].item()

            all_probs[tokens].append(prob)
            gt[tokens].append(1 if item.get('is_correct', False) else 0)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    return search_thresholds(all_probs, gt, n_samples)


def eval_cascade_mlp(
    model_dir: str,
    val_data: Dict[int, List[Dict]],
    base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    device: str = "cuda",
) -> Tuple[float, List[float], Dict]:
    """Evaluate MLP classifier on val data."""

    # Load base model (frozen)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model.eval()

    # Load classifier head
    checkpoint = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)

    hidden_size = base_model.config.hidden_size
    classifier_head = torch.nn.Sequential(
        torch.nn.Linear(hidden_size + 4, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 2),
    ).to(device)
    classifier_head.load_state_dict(checkpoint['classifier'])
    classifier_head.eval()

    # Collect predictions
    n_samples = len(val_data[TOKEN_LEVELS[0]])
    all_probs = {tokens: [] for tokens in TOKEN_LEVELS}
    gt = {tokens: [] for tokens in TOKEN_LEVELS}

    for i in tqdm(range(n_samples), desc="MLP eval"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = val_data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            if mentor_response:
                text = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
            else:
                text = f"Question: {question}\n\nAnswer:"

            encoded = tokenizer(
                text, truncation=True, max_length=2048,
                padding=False, return_tensors="pt",
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1][:, -1, :]

                stage_onehot = torch.zeros(1, 4, device=device)
                stage_onehot[0, stage_idx] = 1.0
                combined = torch.cat([hidden, stage_onehot], dim=-1)

                logits = classifier_head(combined)
                prob = torch.softmax(logits, dim=1)[0, 1].item()

            all_probs[tokens].append(prob)
            gt[tokens].append(1 if item.get('is_correct', False) else 0)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    return search_thresholds(all_probs, gt, n_samples)


def eval_cascade_ppl(
    model_dir: str,
    val_data: Dict[int, List[Dict]],
    base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    device: str = "cuda",
) -> Tuple[float, List[float], Dict]:
    """Evaluate PPL classifier on val data."""
    from sklearn.preprocessing import StandardScaler

    # Load classifier
    with open(os.path.join(model_dir, "classifier.pkl"), 'rb') as f:
        saved = pickle.load(f)
    clf = saved['classifier']
    scaler = saved['scaler']

    # Load LLM for feature extraction
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.eval()

    # Extract features
    all_features = []
    all_labels = []
    n_samples = len(val_data[TOKEN_LEVELS[0]])

    for i in tqdm(range(n_samples), desc="PPL feature extraction"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = val_data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            if mentor_response:
                text = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
            else:
                text = f"Question: {question}\n\nAnswer:"

            encoded = tokenizer(
                text, truncation=True, max_length=2048, return_tensors="pt",
            )
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

            features = compute_ppl_features(token_logprobs, token_entropies, stage_idx, tokens)
            all_features.append(features)
            all_labels.append(1 if item.get('is_correct', False) else 0)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    X = np.array(all_features)
    y = np.array(all_labels)

    # Predict
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]

    # Reshape for cascade
    probs = probs.reshape(n_samples, len(TOKEN_LEVELS))
    gt_arr = y.reshape(n_samples, len(TOKEN_LEVELS))

    return search_thresholds_array(probs, gt_arr, n_samples)


def compute_ppl_features(token_logprobs, token_entropies, stage_idx, tokens):
    """Compute PPL/entropy features for a sample."""
    if not token_logprobs:
        return [1.0, 0.0] + [0.0] * 20 + [0, stage_idx, tokens]

    logprobs = np.array(token_logprobs)
    entropies = np.array(token_entropies)

    ppl = np.exp(-np.mean(logprobs))

    def trend_stats(arr):
        n = len(arr)
        if n < 2:
            return {'slope': 0, 'increase_ratio': 0.5, 'decrease_ratio': 0.5,
                    'first_quarter': float(arr[0]) if n > 0 else 0,
                    'last_quarter': float(arr[-1]) if n > 0 else 0, 'trend_change': 0}
        x = np.arange(n)
        slope = np.polyfit(x, arr, 1)[0]
        diffs = np.diff(arr)
        inc = np.sum(diffs > 0) / len(diffs)
        dec = np.sum(diffs < 0) / len(diffs)
        q = max(1, n // 4)
        return {
            'slope': float(slope), 'increase_ratio': float(inc), 'decrease_ratio': float(dec),
            'first_quarter': float(np.mean(arr[:q])), 'last_quarter': float(np.mean(arr[-q:])),
            'trend_change': float(np.sum(np.diff(np.sign(diffs)) != 0)),
        }

    lp_trend = trend_stats(logprobs)
    ent_trend = trend_stats(entropies)

    return [
        float(ppl), float(np.log(ppl + 1e-10)),
        float(np.mean(entropies)), float(np.std(entropies)),
        float(np.max(entropies)), float(np.min(entropies)),
        ent_trend['slope'], ent_trend['increase_ratio'], ent_trend['decrease_ratio'],
        ent_trend['first_quarter'], ent_trend['last_quarter'], ent_trend['trend_change'],
        float(np.mean(logprobs)), float(np.std(logprobs)),
        float(np.max(logprobs)), float(np.min(logprobs)),
        lp_trend['slope'], lp_trend['increase_ratio'], lp_trend['decrease_ratio'],
        lp_trend['first_quarter'], lp_trend['last_quarter'], lp_trend['trend_change'],
        len(logprobs), stage_idx, tokens,
    ]


def search_thresholds(all_probs, gt, n_samples):
    """Search for best thresholds."""
    def compute_cascade_acc(thresholds):
        correct = 0
        for i in range(n_samples):
            decided = False
            stage_probs = []
            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                prob = all_probs[tokens][i]
                stage_probs.append((tokens, prob))
                if prob >= thresholds[stage_idx]:
                    correct += gt[tokens][i]
                    decided = True
                    break
            if not decided:
                best_tokens, _ = max(stage_probs, key=lambda x: x[1])
                correct += gt[best_tokens][i]
        return correct / n_samples

    threshold_candidates = [round(i * 0.05, 2) for i in range(21)]
    best_acc, best_thresholds = 0, None

    for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
        thresholds = list(combo)
        acc = compute_cascade_acc(thresholds)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    # Oracle and per-stage stats
    oracle_correct = sum(1 for i in range(n_samples)
                         if any(gt[tokens][i] == 1 for tokens in TOKEN_LEVELS))
    oracle_acc = oracle_correct / n_samples

    stage_auc = {}
    stage_baseline = {}
    for tokens in TOKEN_LEVELS:
        stage_baseline[tokens] = sum(gt[tokens]) / n_samples
        try:
            stage_auc[tokens] = roc_auc_score(gt[tokens], all_probs[tokens])
        except:
            stage_auc[tokens] = 0.5

    return best_acc, best_thresholds, {'oracle': oracle_acc, 'auc': stage_auc, 'baseline': stage_baseline}


def search_thresholds_array(probs, gt, n_samples):
    """Search for best thresholds (array version for PPL)."""
    def compute_cascade_acc(thresholds):
        correct = 0
        for i in range(n_samples):
            decided = False
            stage_probs = []
            for stage_idx in range(len(TOKEN_LEVELS)):
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

    threshold_candidates = [round(i * 0.05, 2) for i in range(21)]
    best_acc, best_thresholds = 0, None

    for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
        thresholds = list(combo)
        acc = compute_cascade_acc(thresholds)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    oracle_correct = sum(1 for i in range(n_samples) if any(gt[i, :] == 1))
    oracle_acc = oracle_correct / n_samples

    stage_auc = {}
    stage_baseline = {}
    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        stage_baseline[tokens] = float(np.mean(gt[:, stage_idx]))
        try:
            stage_auc[tokens] = roc_auc_score(gt[:, stage_idx], probs[:, stage_idx])
        except:
            stage_auc[tokens] = 0.5

    return best_acc, best_thresholds, {'oracle': oracle_acc, 'auc': stage_auc, 'baseline': stage_baseline}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--subset", type=str, default="all", choices=SUBSETS + ["all"])
    parser.add_argument("--classifier", type=str, default="all",
                        choices=["lora", "mlp", "ppl", "all"])
    parser.add_argument("--base-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    args = parser.parse_args()

    subsets = SUBSETS if args.subset == "all" else [args.subset]
    classifiers = ["lora", "mlp", "ppl"] if args.classifier == "all" else [args.classifier]

    for subset in subsets:
        subset_dir = os.path.join(args.data_dir, subset)
        print(f"\n{'='*60}")
        print(f"Evaluating: {subset}")
        print(f"{'='*60}")

        # Load val data (unfiltered)
        val_data = get_val_data(subset_dir)
        if not val_data:
            print(f"No data found for {subset}, skipping...")
            continue

        print(f"Val samples: {len(val_data[TOKEN_LEVELS[0]])}")

        for clf_type in classifiers:
            model_dir = os.path.join(subset_dir, f"{clf_type}_model")
            if not os.path.exists(model_dir):
                print(f"  {clf_type.upper()}: No model found, skipping...")
                continue

            print(f"\n  Evaluating {clf_type.upper()}...")

            try:
                if clf_type == "lora":
                    cascade_acc, thresholds, detailed = eval_cascade_lora(
                        model_dir, val_data
                    )
                elif clf_type == "mlp":
                    cascade_acc, thresholds, detailed = eval_cascade_mlp(
                        model_dir, val_data, args.base_model
                    )
                else:  # ppl
                    cascade_acc, thresholds, detailed = eval_cascade_ppl(
                        model_dir, val_data, args.base_model
                    )

                print(f"  Cascade Acc: {cascade_acc:.4f} (Oracle: {detailed['oracle']:.4f})")
                print(f"  Thresholds: {thresholds}")

                # Update results.json
                results_path = os.path.join(model_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        results = json.load(f)
                else:
                    results = {}

                results['best_cascade_acc'] = cascade_acc
                results['best_thresholds'] = thresholds
                results['oracle_acc'] = detailed['oracle']
                results['per_stage_auc'] = detailed['auc']
                results['per_stage_baseline_acc'] = detailed['baseline']

                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  Results updated: {results_path}")

            except Exception as e:
                print(f"  Error evaluating {clf_type}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
