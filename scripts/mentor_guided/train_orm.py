#!/usr/bin/env python3
"""
Train a small ORM (Outcome Reward Model) to predict if mentor output helps student.

Uses a small encoder (e.g., sentence-transformer) + classification head.
Input: [problem] [SEP] [mentor_output]
Output: probability that this mentor output is helpful

训练小型ORM判断mentor输出是否对student有帮助
"""

import argparse
import json
import logging
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))


class ORMDataset(Dataset):
    """Dataset for ORM training."""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format: [problem] [SEP] [mentor_output]
        problem = sample["problem"][:500]  # Truncate problem if too long
        mentor_text = sample["mentor_text"][:300]

        text = f"{problem} [SEP] {mentor_text}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Binary label: 1 if helpful (label=1), 0 otherwise
        # Could also use 3-class: harmful(-1), neutral(0), helpful(1)
        label = 1 if sample["label"] == 1 else 0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }


class ORMModel(nn.Module):
    """Small ORM model: encoder + classification head."""

    def __init__(self, encoder_name: str, hidden_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(encoder_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token or mean pooling
        # Mean pooling usually works better
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

        score = self.classifier(pooled)
        return score.squeeze(-1)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion = nn.BCELoss()

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        scores = model(input_ids, attention_mask)
        loss = criterion(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device) -> Tuple[float, float, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            scores = model(input_ids, attention_mask)
            preds = (scores.cpu() > 0.5).float()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    # Precision, Recall for helpful class
    true_pos = ((all_preds == 1) & (all_labels == 1)).sum()
    pred_pos = (all_preds == 1).sum()
    actual_pos = (all_labels == 1).sum()

    precision = true_pos / pred_pos if pred_pos > 0 else 0
    recall = true_pos / actual_pos if actual_pos > 0 else 0

    return accuracy, precision, recall


def main():
    parser = argparse.ArgumentParser(description='Train ORM Model')
    parser.add_argument('--data-file', default='orm_training_data.json')
    parser.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Encoder model (small and fast)')
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--output-dir', default='orm_model')
    parser.add_argument('--filter-neutral', action='store_true',
                       help='Only use helpful/harmful samples for training')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    data_path = os.path.join(script_dir, args.data_file)
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data["samples"]
    logger.info(f"Loaded {len(samples)} samples")

    # Filter out length=0 (baseline) samples
    samples = [s for s in samples if s["mentor_tokens"] > 0]
    logger.info(f"After filtering baseline: {len(samples)} samples")

    if args.filter_neutral:
        # Only keep helpful (1) and harmful (-1) samples
        samples = [s for s in samples if s["label"] != 0]
        logger.info(f"After filtering neutral: {len(samples)} samples")

    # Split train/val
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Load tokenizer and create datasets
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    train_dataset = ORMDataset(train_samples, tokenizer, args.max_length)
    val_dataset = ORMDataset(val_samples, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = ORMModel(args.encoder, args.hidden_size)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Train Loss: {train_loss:.4f}")

        acc, prec, rec = evaluate(model, val_loader, device)
        logger.info(f"Val Accuracy: {acc:.4f}")
        logger.info(f"Val Precision (helpful): {prec:.4f}")
        logger.info(f"Val Recall (helpful): {rec:.4f}")

        if acc > best_acc:
            best_acc = acc
            # Save best model
            output_dir = os.path.join(script_dir, args.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_name': args.encoder,
                'hidden_size': args.hidden_size,
                'config': vars(args),
            }, os.path.join(output_dir, 'best_model.pt'))
            logger.info(f"Saved best model (acc={acc:.4f})")

    logger.info(f"\nBest validation accuracy: {best_acc:.4f}")
    logger.info(f"Model saved to: {os.path.join(script_dir, args.output_dir)}")


if __name__ == "__main__":
    main()
