#!/usr/bin/env python3
"""
Train ensemble classifier combining LoRA and PPL predictions.
Uses Random Forest to combine the signals from both classifiers.

Usage:
    python train_ensemble_classifier.py --data-dir DATA_DIR --subset algebra
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from itertools import product
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn as nn

TOKEN_LEVELS = [0, 100, 500, 1000]
SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]


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


def load_ppl_classifier(model_dir: str) -> Tuple:
    """Load PPL classifier and scaler."""
    clf_file = os.path.join(model_dir, "classifier.pkl")
    if os.path.exists(clf_file):
        with open(clf_file, 'rb') as f:
            saved = pickle.load(f)
        return saved['classifier'], saved.get('scaler'), saved.get('thresholds')
    return None, None, None


class StageAwareClassifier(nn.Module):
    """Classifier head for LoRA model."""
    def __init__(self, hidden_size: int, num_stages: int = 4):
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, 64)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, hidden_states, stage_ids):
        stage_emb = self.stage_embedding(stage_ids)
        combined = torch.cat([hidden_states, stage_emb], dim=-1)
        return self.classifier(combined)


class LoRAClassifierModel(nn.Module):
    """Combined model with LoRA base + classifier head."""
    def __init__(self, base_model, classifier_head, pooling_mode="last"):
        super().__init__()
        self.base_model = base_model
        self.classifier_head = classifier_head
        self.pooling_mode = pooling_mode

    def forward(self, input_ids, attention_mask, stage_ids):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        if self.pooling_mode == "mean":
            # Mean pooling over all valid tokens
            mask = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask).sum(dim=1)
            seq_lens = mask.sum(dim=1)
            pooled = sum_hidden / seq_lens
        else:
            # Last token pooling (default)
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.size(0)
            pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]

        return self.classifier_head(pooled, stage_ids)


def load_lora_model(model_dir: str, base_model_name: str, device: str):
    """Load LoRA model and classifier head."""
    lora_path = os.path.join(model_dir, "lora_adapter")
    head_path = os.path.join(model_dir, "classifier_head.pt")
    results_path = os.path.join(model_dir, "results.json")

    if not os.path.exists(lora_path) or not os.path.exists(head_path):
        return None, None

    # Load pooling mode from saved results
    pooling_mode = "last"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        pooling_mode = results.get('args', {}).get('pooling', 'last')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model = PeftModel.from_pretrained(base_model, lora_path)
    base_model.eval()

    # Load classifier head
    hidden_size = base_model.config.hidden_size
    classifier_head = StageAwareClassifier(hidden_size, num_stages=len(TOKEN_LEVELS))
    classifier_head.load_state_dict(torch.load(head_path, map_location=device))
    classifier_head = classifier_head.to(device)
    classifier_head.eval()

    model = LoRAClassifierModel(base_model, classifier_head, pooling_mode=pooling_mode)
    print(f"  Loaded LoRA model with pooling mode: {pooling_mode}")
    return model, tokenizer


def get_lora_predictions(
    model, tokenizer, data: Dict[int, List[Dict]], device: str, max_length: int = 512, batch_size: int = 8
) -> Dict[int, List[float]]:
    """Get LoRA model predictions for all samples."""
    if model is None:
        return None

    model.eval()
    predictions = {tokens: [] for tokens in TOKEN_LEVELS}
    n_samples = len(data[TOKEN_LEVELS[0]])

    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        stage_preds = []

        for batch_start in tqdm(range(0, n_samples, batch_size), desc=f"LoRA T{tokens}"):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_items = [data[tokens][i] for i in range(batch_start, batch_end)]

            # Prepare batch
            texts = []
            for item in batch_items:
                q = item.get('question', '')
                m = item.get('mentor_response', '')
                i_resp = item.get('intern_response', '')
                if m:
                    text = f"Question: {q}\nMentor: {m}\nIntern: {i_resp}"
                else:
                    text = f"Question: {q}\nAnswer: {i_resp}"
                texts.append(text)

            encodings = tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            stages = torch.full((len(batch_items),), stage_idx, dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask, stages)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

            stage_preds.extend(probs)

        predictions[tokens] = stage_preds
        torch.cuda.empty_cache()

    return predictions


# ============= MLP Model Loading =============

class MLPClassifierHead(nn.Module):
    """MLP classifier head (same as train_mlp_classifier.py)."""
    def __init__(self, hidden_size: int, num_stages: int = 4, dropout: float = 0.1):
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, 64)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, hidden_state, stage):
        stage_emb = self.stage_embedding(stage)
        combined = torch.cat([hidden_state, stage_emb], dim=-1)
        return self.classifier(combined)


class MLPClassifierModel(nn.Module):
    """Frozen LLM + MLP classifier (no LoRA)."""
    def __init__(self, base_model, classifier_head, pooling_mode="last"):
        super().__init__()
        self.base_model = base_model
        self.classifier_head = classifier_head
        self.pooling_mode = pooling_mode

    def forward(self, input_ids, attention_mask, stage_ids):
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]

        if self.pooling_mode == "mean_logits":
            # Per-token classification, then average logits (no for loop)
            batch_size, seq_len, hidden_size = hidden_states.shape

            # Expand stage embedding to all tokens
            stages_expanded = stage_ids.unsqueeze(1).expand(-1, seq_len)
            stage_embed = self.classifier_head.stage_embedding(stages_expanded)

            # Concatenate and classify all tokens at once
            combined = torch.cat([hidden_states, stage_embed], dim=-1)
            combined_flat = combined.view(batch_size * seq_len, -1)
            logits_flat = self.classifier_head.classifier(combined_flat)
            logits_all = logits_flat.view(batch_size, seq_len, -1)

            # Masked mean over sequence
            mask = attention_mask.unsqueeze(-1).float()
            logits_sum = (logits_all * mask).sum(dim=1)
            seq_lens = mask.sum(dim=1)
            return logits_sum / seq_lens

        elif self.pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask).sum(dim=1)
            seq_lens = mask.sum(dim=1)
            pooled = sum_hidden / seq_lens
        else:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.size(0)
            pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]

        return self.classifier_head(pooled, stage_ids)


def load_mlp_model(model_dir: str, base_model_name: str, device: str):
    """Load MLP model (frozen LLM + MLP head)."""
    head_path = os.path.join(model_dir, "classifier_head.pt")
    results_path = os.path.join(model_dir, "results.json")

    if not os.path.exists(head_path):
        return None, None

    # Load pooling mode and dropout from saved results
    pooling_mode = "last"
    dropout = 0.3
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        pooling_mode = results.get('args', {}).get('pooling', 'last')
        dropout = results.get('args', {}).get('dropout', 0.3)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (no LoRA)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()

    # Load classifier head
    hidden_size = base_model.config.hidden_size
    classifier_head = MLPClassifierHead(hidden_size, num_stages=len(TOKEN_LEVELS), dropout=dropout)
    classifier_head.load_state_dict(torch.load(head_path, map_location=device))
    classifier_head = classifier_head.to(device)
    classifier_head.eval()

    model = MLPClassifierModel(base_model, classifier_head, pooling_mode=pooling_mode)
    print(f"  Loaded MLP model with pooling mode: {pooling_mode}, dropout: {dropout}")
    return model, tokenizer


def get_mlp_predictions(
    model, tokenizer, data: Dict[int, List[Dict]], device: str, max_length: int = 512, batch_size: int = 8
) -> Dict[int, List[float]]:
    """Get MLP model predictions for all samples."""
    if model is None:
        return None

    model.eval()
    predictions = {tokens: [] for tokens in TOKEN_LEVELS}
    n_samples = len(data[TOKEN_LEVELS[0]])

    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        stage_preds = []

        for batch_start in tqdm(range(0, n_samples, batch_size), desc=f"MLP T{tokens}"):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_items = [data[tokens][i] for i in range(batch_start, batch_end)]

            texts = []
            for item in batch_items:
                q = item.get('question', '')
                m = item.get('mentor_response', '')
                i_resp = item.get('intern_response', '')
                if m:
                    text = f"Question: {q}\nMentor: {m}\nIntern: {i_resp}"
                else:
                    text = f"Question: {q}\nAnswer: {i_resp}"
                texts.append(text)

            encodings = tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            stages = torch.full((len(batch_items),), stage_idx, dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask, stages)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

            stage_preds.extend(probs)

        predictions[tokens] = stage_preds
        torch.cuda.empty_cache()

    return predictions


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


def get_ppl_predictions(
    clf, scaler, data: Dict[int, List[Dict]]
) -> Dict[int, List[float]]:
    """Get PPL classifier predictions for all samples."""
    if clf is None:
        return None

    predictions = {tokens: [] for tokens in TOKEN_LEVELS}
    n_samples = len(data[TOKEN_LEVELS[0]])

    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        features = []
        for i in range(n_samples):
            item = data[tokens][i]
            logprobs = item.get('token_logprobs', [])
            entropies = item.get('token_entropies', [])
            feat = compute_ppl_features(logprobs, entropies, stage_idx, tokens)
            features.append(feat)

        X = np.array(features)
        if scaler:
            X = scaler.transform(X)

        # Get probabilities
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X)[:, 1].tolist()
        else:
            probs = clf.decision_function(X).tolist()

        predictions[tokens] = probs

    return predictions


def eval_cascade(
    probs: np.ndarray,  # [n_samples, n_stages]
    gt: np.ndarray,     # [n_samples, n_stages]
    thresholds: List[float]
) -> float:
    """Evaluate cascade accuracy with given thresholds."""
    n_samples = probs.shape[0]
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
            best_stage = max(stage_probs, key=lambda x: x[1])[0]
            correct += gt[i, best_stage]

    return correct / n_samples


def search_thresholds(probs: np.ndarray, gt: np.ndarray) -> Tuple[float, List[float]]:
    """Search for best thresholds."""
    threshold_candidates = [round(i * 0.05, 2) for i in range(21)]
    best_acc = 0
    best_thresholds = None

    for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
        thresholds = list(combo)
        acc = eval_cascade(probs, gt, thresholds)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    return best_acc, best_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--subset", type=str, default="algebra", choices=SUBSETS + ["all"])
    parser.add_argument("--base-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Base model name for loading LoRA/MLP")
    parser.add_argument("--method", type=str, default="rf", choices=["rf", "gb", "lr"],
                        help="Ensemble method: rf=RandomForest, gb=GradientBoosting, lr=LogisticRegression")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model inference")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-lora", action="store_true", help="Skip LoRA predictions (use MLP or PPL only)")
    parser.add_argument("--use-mlp", action="store_true", help="Use MLP model instead of LoRA")
    parser.add_argument("--no-ppl", action="store_true", help="Skip PPL predictions")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subsets = SUBSETS if args.subset == "all" else [args.subset]

    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"Training ensemble for: {subset}")
        print(f"{'='*60}")

        subset_dir = os.path.join(args.data_dir, subset)
        unified_dir = os.path.join(args.data_dir, "all")  # Unified model directory

        # Check for per-subset models first, fallback to unified models
        lora_dir = os.path.join(subset_dir, "lora_model")
        if not os.path.exists(os.path.join(lora_dir, "classifier_head.pt")):
            unified_lora = os.path.join(unified_dir, "lora_model")
            if os.path.exists(os.path.join(unified_lora, "classifier_head.pt")):
                lora_dir = unified_lora

        mlp_dir = os.path.join(subset_dir, "mlp_model")
        if not os.path.exists(os.path.join(mlp_dir, "classifier_head.pt")):
            unified_mlp = os.path.join(unified_dir, "mlp_model")
            if os.path.exists(os.path.join(unified_mlp, "classifier_head.pt")):
                mlp_dir = unified_mlp

        ppl_dir = os.path.join(subset_dir, "ppl_model")
        if not os.path.exists(os.path.join(ppl_dir, "classifier.pkl")):
            unified_ppl = os.path.join(unified_dir, "ppl_model")
            if os.path.exists(os.path.join(unified_ppl, "classifier.pkl")):
                ppl_dir = unified_ppl

        ensemble_dir = os.path.join(subset_dir, "ensemble_model")
        os.makedirs(ensemble_dir, exist_ok=True)

        # Log model directories being used
        if "all" in mlp_dir:
            print(f"Using unified MLP model from: {mlp_dir}")
        if "all" in lora_dir:
            print(f"Using unified LoRA model from: {lora_dir}")
        if "all" in ppl_dir:
            print(f"Using unified PPL model from: {ppl_dir}")

        # Load data
        train_data = load_json_data(subset_dir, split="train")
        test_data = load_json_data(subset_dir, split="test")

        if not train_data:
            print(f"No training data found for {subset}, skipping...")
            continue

        n_train = len(train_data[TOKEN_LEVELS[0]])
        n_test = len(test_data[TOKEN_LEVELS[0]]) if test_data else 0
        print(f"Train: {n_train}, Test: {n_test}")

        # Load neural network model (LoRA or MLP)
        nn_model = None
        tokenizer = None
        nn_model_type = None

        if args.use_mlp:
            # Use MLP model
            print("Loading MLP model...")
            nn_model, tokenizer = load_mlp_model(mlp_dir, args.base_model, device)
            if nn_model is not None:
                nn_model_type = "mlp"
            else:
                print("  MLP model not found")
        elif not args.no_lora:
            # Use LoRA model
            print("Loading LoRA model...")
            nn_model, tokenizer = load_lora_model(lora_dir, args.base_model, device)
            if nn_model is not None:
                nn_model_type = "lora"
            else:
                print("  LoRA model not found, trying MLP...")
                nn_model, tokenizer = load_mlp_model(mlp_dir, args.base_model, device)
                if nn_model is not None:
                    nn_model_type = "mlp"

        # Load PPL classifier
        ppl_clf = None
        ppl_scaler = None
        if not args.no_ppl:
            print("Loading PPL classifier...")
            ppl_clf, ppl_scaler, _ = load_ppl_classifier(ppl_dir)
            if ppl_clf is None:
                print("  PPL classifier not found, skipping PPL predictions")

        if nn_model is None and ppl_clf is None:
            print("No models available for ensemble, skipping...")
            continue

        # Get predictions on train data
        print("\nGenerating predictions on train data...")
        if nn_model_type == "mlp":
            train_nn_preds = get_mlp_predictions(
                nn_model, tokenizer, train_data, device, args.max_length, args.batch_size
            )
        elif nn_model_type == "lora":
            train_nn_preds = get_lora_predictions(
                nn_model, tokenizer, train_data, device, args.max_length, args.batch_size
            )
        else:
            train_nn_preds = None

        train_ppl_preds = get_ppl_predictions(ppl_clf, ppl_scaler, train_data) if ppl_clf else None

        # Extract labels
        train_labels = np.zeros((n_train, len(TOKEN_LEVELS)))
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            for i in range(n_train):
                train_labels[i, stage_idx] = 1 if train_data[tokens][i].get('is_correct', False) else 0

        # Build features for ensemble
        print("\nBuilding ensemble features...")
        train_features = []
        for i in range(n_train):
            sample_feat = []
            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                feat = [stage_idx, tokens / 1000.0]

                if train_nn_preds:
                    feat.append(train_nn_preds[tokens][i])
                if train_ppl_preds:
                    feat.append(train_ppl_preds[tokens][i])

                sample_feat.extend(feat)
            train_features.append(sample_feat)

        X_train = np.array(train_features)
        print(f"Train features shape: {X_train.shape}")

        # Train per-stage classifiers
        stage_classifiers = []
        stage_aucs = []
        train_probs = np.zeros((n_train, len(TOKEN_LEVELS)))

        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            print(f"\nTraining classifier for stage T{tokens}...")

            y_stage = train_labels[:, stage_idx]

            if args.method == "rf":
                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
            elif args.method == "gb":
                clf = GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
                )
            else:
                clf = LogisticRegression(random_state=42, max_iter=1000)

            clf.fit(X_train, y_stage)
            stage_classifiers.append(clf)

            # Get train predictions
            train_probs[:, stage_idx] = clf.predict_proba(X_train)[:, 1]

            # Calculate AUC
            try:
                auc = roc_auc_score(y_stage, train_probs[:, stage_idx])
                stage_aucs.append(auc)
                print(f"  T{tokens} Train AUC: {auc:.4f}")
            except ValueError:
                stage_aucs.append(0.5)
                print(f"  T{tokens} Train AUC: N/A")

        # Search thresholds on train
        print("\nSearching thresholds on train...")
        train_cascade_acc, best_thresholds = search_thresholds(train_probs, train_labels)
        train_oracle = np.mean(np.any(train_labels == 1, axis=1))
        print(f"Train Cascade Acc: {train_cascade_acc:.4f}")
        print(f"Train Oracle: {train_oracle:.4f}")
        print(f"Best Thresholds: {best_thresholds}")

        # Evaluate on test
        if test_data:
            print("\nGenerating predictions on test data...")
            if nn_model_type == "mlp":
                test_nn_preds = get_mlp_predictions(
                    nn_model, tokenizer, test_data, device, args.max_length, args.batch_size
                )
            elif nn_model_type == "lora":
                test_nn_preds = get_lora_predictions(
                    nn_model, tokenizer, test_data, device, args.max_length, args.batch_size
                )
            else:
                test_nn_preds = None

            test_ppl_preds = get_ppl_predictions(ppl_clf, ppl_scaler, test_data) if ppl_clf else None

            # Build test features
            test_features = []
            test_labels = np.zeros((n_test, len(TOKEN_LEVELS)))
            for i in range(n_test):
                sample_feat = []
                for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                    test_labels[i, stage_idx] = 1 if test_data[tokens][i].get('is_correct', False) else 0

                    feat = [stage_idx, tokens / 1000.0]
                    if test_nn_preds:
                        feat.append(test_nn_preds[tokens][i])
                    if test_ppl_preds:
                        feat.append(test_ppl_preds[tokens][i])
                    sample_feat.extend(feat)
                test_features.append(sample_feat)

            X_test = np.array(test_features)

            # Get test predictions
            test_probs = np.zeros((n_test, len(TOKEN_LEVELS)))
            test_aucs = []
            for stage_idx, clf in enumerate(stage_classifiers):
                test_probs[:, stage_idx] = clf.predict_proba(X_test)[:, 1]
                try:
                    auc = roc_auc_score(test_labels[:, stage_idx], test_probs[:, stage_idx])
                    test_aucs.append(auc)
                except ValueError:
                    test_aucs.append(0.5)

            print(f"\nTest per-stage AUC: {dict(zip(TOKEN_LEVELS, [f'{a:.4f}' for a in test_aucs]))}")

            # Apply thresholds from train
            test_cascade_acc = eval_cascade(test_probs, test_labels, best_thresholds)
            test_oracle = np.mean(np.any(test_labels == 1, axis=1))

            print(f"Test Cascade Acc: {test_cascade_acc:.4f}")
            print(f"Test Oracle: {test_oracle:.4f}")

            # Compare with baselines
            test_baselines = {tokens: np.mean(test_labels[:, i]) for i, tokens in enumerate(TOKEN_LEVELS)}
            print(f"Test baselines (T1000): {test_baselines[1000]:.4f}")
            print(f"Gap vs T1000: {test_cascade_acc - test_baselines[1000]:+.4f}")

        # Save model and results
        model_data = {
            'classifiers': stage_classifiers,
            'thresholds': best_thresholds,
            'method': args.method,
            'nn_model_type': nn_model_type,
            'has_ppl': ppl_clf is not None,
        }
        with open(os.path.join(ensemble_dir, "model.pkl"), 'wb') as f:
            pickle.dump(model_data, f)

        results = {
            'subset': subset,
            'method': args.method,
            'nn_model_type': nn_model_type,
            'has_ppl': ppl_clf is not None,
            'train_cascade_acc': float(train_cascade_acc),
            'train_oracle': float(train_oracle),
            'best_thresholds': best_thresholds,
            'train_stage_aucs': dict(zip([str(t) for t in TOKEN_LEVELS], [float(a) for a in stage_aucs])),
        }

        if test_data:
            results['test_cascade_acc'] = float(test_cascade_acc)
            results['test_oracle'] = float(test_oracle)
            results['test_stage_aucs'] = dict(zip([str(t) for t in TOKEN_LEVELS], [float(a) for a in test_aucs]))
            results['test_baselines'] = {str(k): float(v) for k, v in test_baselines.items()}

        with open(os.path.join(ensemble_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nModel saved to: {ensemble_dir}")

        # Clean up GPU memory
        if nn_model:
            del nn_model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
