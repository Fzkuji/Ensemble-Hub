#!/usr/bin/env python3
"""
Train a Causal ORM (decoder-based) for streaming mentor evaluation.

Input: [problem] [SEP] [mentor_token_1] [mentor_token_2] ...
Output at each position: P(student correct | stop here)

Training: for each (problem, mentor_prefix, student_correct) pair,
          train the model to predict student_correct at that position.

用 decoder 模型做因果 ORM，每个位置预测 student 能否答对
"""

import argparse
import json
import logging
import os
from typing import List, Dict, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    GPT2LMHeadModel, GPT2Config,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))


class CausalORMDataset(Dataset):
    """
    Dataset for causal ORM training.

    Each sample: (problem, mentor_text, label) where label = student_correct
    We create training examples at multiple positions within mentor_text.
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 512,
        num_positions_per_sample: int = 5,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for sample in samples:
            problem = sample["problem"][:400]
            mentor_text = sample["mentor_text"]
            label = 1.0 if sample["mentored_correct"] else 0.0

            # Skip baseline (no mentor)
            if sample["mentor_tokens"] == 0:
                continue

            # Create examples at different positions
            mentor_tokens = self.tokenizer.encode(mentor_text, add_special_tokens=False)

            if len(mentor_tokens) < 5:
                positions = list(range(len(mentor_tokens)))
            else:
                # Sample positions: start, some middle, end
                positions = [0, len(mentor_tokens) // 4, len(mentor_tokens) // 2,
                            3 * len(mentor_tokens) // 4, len(mentor_tokens) - 1]
                positions = sorted(set(positions))

            for pos in positions:
                prefix_tokens = mentor_tokens[:pos + 1]
                prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)

                self.examples.append({
                    "problem": problem,
                    "mentor_prefix": prefix_text,
                    "full_mentor_text": mentor_text,
                    "position": pos + 1,  # 1-indexed
                    "total_length": len(mentor_tokens),
                    "label": label,
                })

        logger.info(f"Created {len(self.examples)} training examples from {len(samples)} samples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format: [problem] [SEP] [mentor_prefix]
        text = f"{ex['problem']} [SEP] {ex['mentor_prefix']}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.float),
            "position": ex["position"],
            "total_length": ex["total_length"],
        }


class CausalORM(nn.Module):
    """
    Causal ORM based on a small GPT-2.

    Takes input sequence and outputs a score at each position.
    For training, we only use the score at the last position.
    For inference, we can get scores at any position.
    """

    def __init__(self, model_name: str = "gpt2", hidden_size: int = 128):
        super().__init__()

        # Use a small GPT-2 as base
        self.config = GPT2Config.from_pretrained(model_name)
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name).transformer

        # Score head: predict P(student correct)
        self.score_head = nn.Sequential(
            nn.Linear(self.config.n_embd, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None, return_all_scores=False):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            return_all_scores: if True, return scores at all positions

        Returns:
            if return_all_scores: [B, L] scores at each position
            else: [B] score at the last non-padded position
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state  # [B, L, D]
        all_scores = self.score_head(hidden_states).squeeze(-1)  # [B, L]

        if return_all_scores:
            return all_scores

        # Get score at last position (where attention_mask is 1)
        if attention_mask is not None:
            # Find last non-padded position
            seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            last_scores = all_scores[batch_indices, seq_lengths]
        else:
            last_scores = all_scores[:, -1]

        return last_scores


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion = nn.BCELoss()

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Get score at last position
        scores = model(input_ids, attention_mask)
        loss = criterion(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device) -> Tuple[float, float, float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            scores = model(input_ids, attention_mask)
            preds = (scores.cpu() > 0.5).float()

            all_scores.extend(scores.cpu().tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    accuracy = (all_preds == all_labels).mean()

    # AUC-like metric: average score for positive vs negative
    pos_scores = all_scores[all_labels == 1].mean() if (all_labels == 1).sum() > 0 else 0
    neg_scores = all_scores[all_labels == 0].mean() if (all_labels == 0).sum() > 0 else 0
    score_gap = pos_scores - neg_scores

    # Precision and recall for predicting "helpful"
    true_pos = ((all_preds == 1) & (all_labels == 1)).sum()
    pred_pos = (all_preds == 1).sum()
    actual_pos = (all_labels == 1).sum()

    precision = true_pos / pred_pos if pred_pos > 0 else 0
    recall = true_pos / actual_pos if actual_pos > 0 else 0

    return accuracy, precision, recall, score_gap


def main():
    parser = argparse.ArgumentParser(description='Train Causal ORM')
    parser.add_argument('--data-file', default='orm_training_data.json')
    parser.add_argument('--base-model', default='gpt2',
                       help='Base model (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--output-dir', default='causal_orm_model')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    data_path = os.path.join(script_dir, args.data_file)
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data["samples"]
    logger.info(f"Loaded {len(samples)} raw samples")

    # Filter to only mentor samples (not baseline)
    samples = [s for s in samples if s["mentor_tokens"] > 0]
    logger.info(f"After filtering baseline: {len(samples)}")

    # Split
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_dataset = CausalORMDataset(train_samples, tokenizer, args.max_length)
    val_dataset = CausalORMDataset(val_samples, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    model = CausalORM(args.base_model, args.hidden_size)
    model.to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training
    best_acc = 0
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Train Loss: {train_loss:.4f}")

        acc, prec, rec, gap = evaluate(model, val_loader, device)
        logger.info(f"Val Accuracy: {acc:.4f}")
        logger.info(f"Val Precision: {prec:.4f}, Recall: {rec:.4f}")
        logger.info(f"Score Gap (pos-neg): {gap:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'base_model': args.base_model,
                'hidden_size': args.hidden_size,
                'config': vars(args),
            }, os.path.join(output_dir, 'best_model.pt'))
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved best model (acc={acc:.4f})")

    logger.info(f"\nBest accuracy: {best_acc:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
