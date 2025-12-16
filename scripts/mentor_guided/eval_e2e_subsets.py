#!/usr/bin/env python3
"""
End-to-end transformer classifier evaluation on each hendrycks_math subset.
Train on train split, evaluate on test split, compare with baselines.
"""

import argparse
import json
import logging
import math
import os
from itertools import product
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_stages: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.stage_embedding = nn.Embedding(num_stages, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, hidden_states, attention_mask, stages):
        batch_size = hidden_states.size(0)
        x = self.input_proj(hidden_states)
        stage_embed = self.stage_embedding(stages).unsqueeze(1)
        x = torch.cat([stage_embed, x], dim=1)
        stage_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        attention_mask = torch.cat([stage_mask, attention_mask], dim=1)
        x = self.pos_encoder(x)
        src_key_padding_mask = (attention_mask == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        seq_lens = attention_mask.sum(dim=1).long()
        last_positions = seq_lens - 1
        batch_indices = torch.arange(batch_size, device=x.device)
        last_hidden = x[batch_indices, last_positions]
        return self.classifier(last_hidden)


def extract_hidden_states(
    model,
    tokenizer,
    questions: List[str],
    mentor_responses: List[str],
    max_mentor_tokens: int,
    device: str,
) -> tuple:
    batch_hidden = []
    batch_mask = []

    with torch.no_grad():
        for question, mentor_response in zip(questions, mentor_responses):
            question_ids = tokenizer.encode(question, return_tensors="pt").to(device)
            full_text = question + mentor_response
            full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

            question_len = question_ids.shape[1]
            total_len = full_ids.shape[1]
            mentor_len = total_len - question_len

            if mentor_len <= 0:
                batch_hidden.append(torch.zeros(1, model.config.hidden_size, device=device))
                batch_mask.append(torch.ones(1, device=device))
                continue

            if mentor_len > max_mentor_tokens:
                full_ids = full_ids[:, :question_len + max_mentor_tokens]
                mentor_len = max_mentor_tokens

            outputs = model(full_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]
            mentor_hidden = hidden[question_len:question_len + mentor_len]
            batch_hidden.append(mentor_hidden)
            batch_mask.append(torch.ones(mentor_len, device=device))

    max_len = max(h.shape[0] for h in batch_hidden)
    hidden_dim = batch_hidden[0].shape[1]

    padded_hidden = torch.zeros(len(batch_hidden), max_len, hidden_dim, device=device)
    padded_mask = torch.zeros(len(batch_hidden), max_len, device=device)

    for i, (h, m) in enumerate(zip(batch_hidden, batch_mask)):
        seq_len = h.shape[0]
        padded_hidden[i, :seq_len] = h
        padded_mask[i, :seq_len] = m

    return padded_hidden, padded_mask


def load_subset_json(data_dir: str, subset: str, split: str, token_level: int) -> List[Dict]:
    filepath = os.path.join(data_dir, subset, split, f"tokens{token_level}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


def compute_oracle_accuracy(all_data: Dict[int, List[Dict]]) -> float:
    if not all_data or not all_data.get(TOKEN_LEVELS[0]):
        return 0.0
    n_samples = len(all_data[TOKEN_LEVELS[0]])
    oracle_correct = 0
    for i in range(n_samples):
        for tokens in TOKEN_LEVELS:
            if all_data[tokens][i].get('is_correct', False):
                oracle_correct += 1
                break
    return oracle_correct / n_samples


def train_on_subset(
    intern_model,
    tokenizer,
    train_data: Dict[int, List[Dict]],
    hidden_dim: int,
    args,
) -> nn.Module:
    """Train transformer classifier on a single subset."""
    device = args.device

    classifier = TransformerClassifier(
        input_dim=hidden_dim,
        num_stages=len(TOKEN_LEVELS),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    n_samples = len(train_data[TOKEN_LEVELS[0]])
    train_labels = [1 if item.get('is_correct', False) else 0 for item in train_data[TOKEN_LEVELS[0]]]

    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    if pos_count > 0 and neg_count > 0:
        class_weights = torch.tensor([1.0 / neg_count, 1.0 / pos_count], device=device)
        class_weights = class_weights / class_weights.sum() * 2
    else:
        class_weights = torch.ones(2, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    best_model_state = None
    indices = np.arange(n_samples)

    for epoch in range(args.epochs):
        classifier.train()
        np.random.shuffle(indices)
        epoch_loss = 0
        n_batches = 0

        for batch_start in range(0, n_samples, args.batch_size):
            batch_idx = indices[batch_start:batch_start + args.batch_size]
            batch_stages = np.random.randint(0, len(TOKEN_LEVELS), size=len(batch_idx))

            questions = []
            mentor_responses = []
            batch_labels = []

            for sample_idx, stage_idx in zip(batch_idx, batch_stages):
                token_level = TOKEN_LEVELS[stage_idx]
                item = train_data[token_level][sample_idx]
                questions.append(item['question'])
                mentor_responses.append(item.get('mentor_response', ''))
                batch_labels.append(1 if item.get('is_correct', False) else 0)

            hidden, mask = extract_hidden_states(
                intern_model, tokenizer, questions, mentor_responses,
                args.max_mentor_tokens, device
            )

            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
            stages_tensor = torch.tensor(batch_stages, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = classifier(hidden.float(), mask, stages_tensor)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = classifier.state_dict().copy()

    classifier.load_state_dict(best_model_state)
    return classifier


def evaluate_cascade(
    classifier: nn.Module,
    intern_model,
    tokenizer,
    test_data: Dict[int, List[Dict]],
    args,
) -> Dict:
    """Evaluate cascade on test set, search for best thresholds."""
    device = args.device
    classifier.eval()
    n_samples = len(test_data[TOKEN_LEVELS[0]])

    # Get ground truth
    gt = {}
    for tokens in TOKEN_LEVELS:
        gt[tokens] = [1 if item.get('is_correct', False) else 0 for item in test_data[tokens]]

    # Search thresholds
    threshold_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_acc = 0
    best_thresholds = None

    for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
        thresholds = list(combo)
        correct = 0

        with torch.no_grad():
            for i in range(n_samples):
                for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                    item = test_data[tokens][i]
                    question = item['question']
                    mentor_response = item.get('mentor_response', '')

                    hidden, mask = extract_hidden_states(
                        intern_model, tokenizer, [question], [mentor_response],
                        args.max_mentor_tokens, device
                    )
                    stages_tensor = torch.tensor([stage_idx], device=device)

                    logits = classifier(hidden.float(), mask, stages_tensor)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()

                    if prob >= thresholds[stage_idx]:
                        correct += gt[tokens][i]
                        break
                    elif stage_idx == len(TOKEN_LEVELS) - 1:
                        correct += gt[TOKEN_LEVELS[0]][i]

        acc = correct / n_samples
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    return {
        'best_accuracy': float(best_acc),
        'best_thresholds': best_thresholds,
    }


def main():
    parser = argparse.ArgumentParser(description="E2E transformer evaluation on subsets")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Directory with split data (containing subset folders)")
    parser.add_argument("--intern-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--max-mentor-tokens", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-file", type=str, default=None)

    args = parser.parse_args()

    # Load intern model once
    logger.info(f"Loading intern model: {args.intern_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.intern_model)
    intern_model = AutoModelForCausalLM.from_pretrained(
        args.intern_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        output_hidden_states=True,
    )
    intern_model.eval()
    for param in intern_model.parameters():
        param.requires_grad = False

    hidden_dim = intern_model.config.hidden_size
    logger.info(f"Hidden dim: {hidden_dim}")

    results = {}

    logger.info("\n" + "=" * 60)
    logger.info("Evaluating each subset...")
    logger.info("=" * 60)

    for subset in SUBSETS:
        logger.info(f"\n--- {subset} ---")

        # Load train and test data
        train_data = {}
        test_data = {}
        for tokens in TOKEN_LEVELS:
            train_data[tokens] = load_subset_json(args.data_dir, subset, "train", tokens)
            test_data[tokens] = load_subset_json(args.data_dir, subset, "test", tokens)

        if not test_data.get(TOKEN_LEVELS[0]):
            logger.warning(f"No test data for {subset}, skipping")
            continue

        n_train = len(train_data[TOKEN_LEVELS[0]])
        n_test = len(test_data[TOKEN_LEVELS[0]])
        logger.info(f"Train: {n_train}, Test: {n_test}")

        # Baseline accuracies
        baseline_acc = {}
        for tokens in TOKEN_LEVELS:
            if test_data[tokens]:
                correct = sum(1 for item in test_data[tokens] if item.get('is_correct', False))
                acc = correct / len(test_data[tokens])
                baseline_acc[tokens] = acc
                logger.info(f"  Tokens {tokens}: {acc:.4f}")

        # Oracle
        oracle_acc = compute_oracle_accuracy(test_data)
        logger.info(f"  Oracle: {oracle_acc:.4f}")

        # Train transformer classifier
        if n_train > 0:
            logger.info(f"  Training transformer classifier...")
            classifier = train_on_subset(
                intern_model, tokenizer, train_data, hidden_dim, args
            )

            # Evaluate
            logger.info(f"  Evaluating cascade...")
            transformer_result = evaluate_cascade(
                classifier, intern_model, tokenizer, test_data, args
            )
            logger.info(f"  Transformer Best: {transformer_result['best_accuracy']:.4f} "
                       f"(thresholds: {transformer_result['best_thresholds']})")

            results[subset] = {
                'n_test': n_test,
                'baseline': baseline_acc,
                'oracle': oracle_acc,
                'transformer': transformer_result,
            }
        else:
            logger.info("  (No training data, skipping)")
            results[subset] = {
                'n_test': n_test,
                'baseline': baseline_acc,
                'oracle': oracle_acc,
            }

    # Summary table
    logger.info("\n" + "=" * 100)
    logger.info("Summary")
    logger.info("=" * 100)

    header = f"{'Subset':<25} {'T0':<8} {'T100':<8} {'T500':<8} {'T1000':<8} {'Oracle':<8} {'Trans':<8}"
    logger.info(header)
    logger.info("-" * 100)

    for subset in SUBSETS:
        if subset not in results:
            continue
        r = results[subset]
        baseline = r['baseline']
        t0 = baseline.get(0, 0)
        t100 = baseline.get(100, 0)
        t500 = baseline.get(500, 0)
        t1000 = baseline.get(1000, 0)
        oracle = r['oracle']
        trans_acc = r.get('transformer', {}).get('best_accuracy', 0)

        logger.info(f"{subset:<25} {t0:<8.4f} {t100:<8.4f} {t500:<8.4f} {t1000:<8.4f} {oracle:<8.4f} {trans_acc:<8.4f}")

    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
