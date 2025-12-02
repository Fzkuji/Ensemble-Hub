#!/usr/bin/env python3
"""
Train a Process Reward Model (PRM) for mentor guidance.

Input: [problem] [SEP] [mentor_token_1] [mentor_token_2] ...
Output at each position: score indicating helpfulness
    > 1: very helpful, can stop here
    0~1: somewhat helpful, might need more
    < 0: harmful, shouldn't use these tokens

Training approaches:
1. Regression: predict P(student correct | stop here)
2. Preference: compare different stopping points
3. Mixed: combine both signals

用 PRM 训练每个位置的帮助程度分数
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
    AutoTokenizer, GPT2LMHeadModel, GPT2Config,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))


class PRMDataset(Dataset):
    """
    Dataset for PRM training.

    Each sample has:
    - problem: the math problem
    - mentor_text: full mentor output
    - positions: list of positions with their labels
    - label: continuous score based on student correctness

    Label design:
    - If student correct at this position: label = 1.0 + bonus (based on efficiency)
    - If student wrong: label = -1.0 (harmful) or 0.0 (neutral)
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 512,
        num_positions_per_sample: int = 5,
        baseline_correct_rate: float = 0.5,  # student-alone correct rate for this problem
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.pairs = []  # For preference learning

        # Group samples by problem
        problem_samples = {}
        for sample in samples:
            key = sample["problem"][:200]
            if key not in problem_samples:
                problem_samples[key] = []
            problem_samples[key].append(sample)

        for problem_key, problem_group in problem_samples.items():
            # Get baseline (no mentor)
            baseline = [s for s in problem_group if s["mentor_tokens"] == 0]
            baseline_correct = baseline[0]["mentored_correct"] if baseline else False

            # Get mentor samples
            mentor_samples = [s for s in problem_group if s["mentor_tokens"] > 0]
            mentor_samples.sort(key=lambda x: x["mentor_tokens"])

            for sample in mentor_samples:
                problem = sample["problem"][:400]
                mentor_text = sample["mentor_text"]
                mentored_correct = sample["mentored_correct"]

                # Calculate label based on outcome
                if not baseline_correct and mentored_correct:
                    # Rescued: positive score, bonus for fewer tokens
                    efficiency_bonus = max(0, 0.5 - sample["mentor_tokens"] / 400)
                    label = 1.0 + efficiency_bonus
                elif baseline_correct and not mentored_correct:
                    # Hurt: negative score
                    label = -1.0
                elif baseline_correct and mentored_correct:
                    # Both correct: small positive (didn't hurt)
                    label = 0.3
                else:
                    # Both wrong: neutral to slightly negative
                    label = -0.3

                mentor_tokens = self.tokenizer.encode(mentor_text, add_special_tokens=False)

                if len(mentor_tokens) < 3:
                    continue

                # Create examples at different positions
                positions = self._get_positions(len(mentor_tokens), num_positions_per_sample)

                for pos in positions:
                    prefix_tokens = mentor_tokens[:pos + 1]
                    prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)

                    # Score should increase towards the end if helpful
                    position_ratio = (pos + 1) / len(mentor_tokens)
                    position_adjusted_label = label * position_ratio if label > 0 else label

                    self.examples.append({
                        "problem": problem,
                        "mentor_prefix": prefix_text,
                        "position": pos + 1,
                        "total_length": len(mentor_tokens),
                        "label": position_adjusted_label,
                        "final_correct": mentored_correct,
                        "baseline_correct": baseline_correct,
                    })

            # Create preference pairs for this problem
            self._create_preference_pairs(mentor_samples, problem_key)

        logger.info(f"Created {len(self.examples)} examples, {len(self.pairs)} preference pairs")

    def _get_positions(self, length: int, num_positions: int) -> List[int]:
        if length <= num_positions:
            return list(range(length))
        # Sample: start, middle points, end
        positions = [0, length - 1]
        step = length // (num_positions - 1)
        for i in range(1, num_positions - 1):
            positions.append(min(i * step, length - 1))
        return sorted(set(positions))

    def _create_preference_pairs(self, samples: List[Dict], problem_key: str):
        """Create pairs where one outcome is better than another."""
        for i, s1 in enumerate(samples):
            for s2 in samples[i + 1:]:
                # Compare outcomes
                score1 = self._get_preference_score(s1)
                score2 = self._get_preference_score(s2)

                if abs(score1 - score2) > 0.5:  # Significant difference
                    if score1 > score2:
                        self.pairs.append((s1, s2, problem_key))
                    else:
                        self.pairs.append((s2, s1, problem_key))

    def _get_preference_score(self, sample: Dict) -> float:
        """Score for preference comparison."""
        if sample["mentored_correct"]:
            # Correct: prefer fewer tokens
            return 1.0 - sample["mentor_tokens"] / 500
        else:
            return -1.0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

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

    def get_preference_batch(self, batch_size: int) -> List[Tuple]:
        """Get a batch of preference pairs."""
        if len(self.pairs) < batch_size:
            return self.pairs
        return random.sample(self.pairs, batch_size)


class ProcessRewardModel(nn.Module):
    """
    PRM: outputs a continuous score at each position.

    Score interpretation:
    > 1.0: Very helpful, confident can stop
    0.5~1.0: Helpful, might need more
    0~0.5: Slightly helpful
    < 0: Harmful
    """

    def __init__(self, model_name: str = "gpt2", hidden_size: int = 256):
        super().__init__()

        self.config = GPT2Config.from_pretrained(model_name)
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name).transformer

        # Score head: outputs unbounded score
        self.score_head = nn.Sequential(
            nn.Linear(self.config.n_embd, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            # No sigmoid - output unbounded score
        )

    def forward(self, input_ids, attention_mask=None, return_all_scores=False):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state  # [B, L, D]
        all_scores = self.score_head(hidden_states).squeeze(-1)  # [B, L]

        if return_all_scores:
            return all_scores

        # Get score at last non-padded position
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            last_scores = all_scores[batch_indices, seq_lengths]
        else:
            last_scores = all_scores[:, -1]

        return last_scores


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    dataset,  # For preference pairs
    preference_weight: float = 0.3,
):
    model.train()
    total_loss = 0
    total_reg_loss = 0
    total_pref_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Regression loss
        scores = model(input_ids, attention_mask)
        reg_loss = F.mse_loss(scores, labels)

        # Preference loss (sample pairs from dataset)
        pref_loss = torch.tensor(0.0, device=device)
        pairs = dataset.get_preference_batch(len(input_ids) // 2)

        if pairs and preference_weight > 0:
            better_texts = []
            worse_texts = []

            for better, worse, problem_key in pairs:
                better_text = f"{better['problem'][:400]} [SEP] {better['mentor_text'][:200]}"
                worse_text = f"{worse['problem'][:400]} [SEP] {worse['mentor_text'][:200]}"
                better_texts.append(better_text)
                worse_texts.append(worse_text)

            if better_texts:
                better_enc = dataset.tokenizer(
                    better_texts, max_length=512, padding=True,
                    truncation=True, return_tensors="pt"
                )
                worse_enc = dataset.tokenizer(
                    worse_texts, max_length=512, padding=True,
                    truncation=True, return_tensors="pt"
                )

                better_scores = model(
                    better_enc["input_ids"].to(device),
                    better_enc["attention_mask"].to(device)
                )
                worse_scores = model(
                    worse_enc["input_ids"].to(device),
                    worse_enc["attention_mask"].to(device)
                )

                # Bradley-Terry loss: better should have higher score
                pref_loss = -F.logsigmoid(better_scores - worse_scores).mean()

        # Combined loss
        loss = reg_loss + preference_weight * pref_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_pref_loss += pref_loss.item()

    n = len(dataloader)
    return total_loss / n, total_reg_loss / n, total_pref_loss / n


def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_scores = []
    all_labels = []
    all_correct = []
    all_baseline = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            scores = model(input_ids, attention_mask)

            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Metrics
    mse = ((all_scores - all_labels) ** 2).mean()
    correlation = np.corrcoef(all_scores, all_labels)[0, 1] if len(all_scores) > 1 else 0

    # Threshold-based metrics
    # If score > 0.5, predict "helpful"
    pred_helpful = all_scores > 0.5
    actual_helpful = all_labels > 0.5

    accuracy = (pred_helpful == actual_helpful).mean()

    # Score statistics by label
    positive_mask = all_labels > 0.5
    negative_mask = all_labels < -0.5

    pos_score_mean = all_scores[positive_mask].mean() if positive_mask.sum() > 0 else 0
    neg_score_mean = all_scores[negative_mask].mean() if negative_mask.sum() > 0 else 0

    return {
        "mse": mse,
        "correlation": correlation,
        "accuracy": accuracy,
        "pos_score_mean": pos_score_mean,
        "neg_score_mean": neg_score_mean,
        "score_gap": pos_score_mean - neg_score_mean,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Process Reward Model')
    parser.add_argument('--data-file', default='orm_training_data.json')
    parser.add_argument('--base-model', default='gpt2')
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--preference-weight', type=float, default=0.3,
                       help='Weight for preference loss')
    parser.add_argument('--output-dir', default='prm_model')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    data_path = os.path.join(script_dir, args.data_file)
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data["samples"]
    logger.info(f"Loaded {len(samples)} raw samples")

    # Split by problem (not by sample)
    problems = list(set(s["problem"][:200] for s in samples))
    train_problems, val_problems = train_test_split(problems, test_size=0.2, random_state=42)

    train_samples = [s for s in samples if s["problem"][:200] in train_problems]
    val_samples = [s for s in samples if s["problem"][:200] in val_problems]

    logger.info(f"Train problems: {len(train_problems)}, Val problems: {len(val_problems)}")
    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_dataset = PRMDataset(train_samples, tokenizer, args.max_length)
    val_dataset = PRMDataset(val_samples, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    model = ProcessRewardModel(args.base_model, args.hidden_size)
    model.to(device)

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
    best_corr = -1
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        total_loss, reg_loss, pref_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            train_dataset, args.preference_weight
        )
        logger.info(f"Train Loss: {total_loss:.4f} (reg={reg_loss:.4f}, pref={pref_loss:.4f})")

        metrics = evaluate(model, val_loader, device)
        logger.info(f"Val MSE: {metrics['mse']:.4f}")
        logger.info(f"Val Correlation: {metrics['correlation']:.4f}")
        logger.info(f"Val Accuracy (threshold 0.5): {metrics['accuracy']:.4f}")
        logger.info(f"Score Gap (pos - neg): {metrics['score_gap']:.4f}")
        logger.info(f"  Positive samples mean score: {metrics['pos_score_mean']:.4f}")
        logger.info(f"  Negative samples mean score: {metrics['neg_score_mean']:.4f}")

        if metrics['correlation'] > best_corr:
            best_corr = metrics['correlation']
            torch.save({
                'model_state_dict': model.state_dict(),
                'base_model': args.base_model,
                'hidden_size': args.hidden_size,
                'config': vars(args),
            }, os.path.join(output_dir, 'best_model.pt'))
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved best model (corr={best_corr:.4f})")

    logger.info(f"\nBest correlation: {best_corr:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
