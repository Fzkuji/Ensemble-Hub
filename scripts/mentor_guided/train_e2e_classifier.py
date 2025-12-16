#!/usr/bin/env python3
"""
End-to-end training: extract hidden states on-the-fly and train classifier.
No need to save intermediate hidden states.

Architecture:
1. Frozen intern model extracts hidden states
2. Transformer classifier processes the sequence
3. Last token output → MLP → binary classification
"""

import argparse
import json
import logging
import math
import os
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]


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
    """Transformer classifier on top of hidden states."""

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
        """
        Args:
            hidden_states: [batch, seq_len, input_dim]
            attention_mask: [batch, seq_len]
            stages: [batch]
        """
        batch_size = hidden_states.size(0)

        # Project
        x = self.input_proj(hidden_states)

        # Prepend stage embedding
        stage_embed = self.stage_embedding(stages).unsqueeze(1)
        x = torch.cat([stage_embed, x], dim=1)

        # Update mask
        stage_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        attention_mask = torch.cat([stage_mask, attention_mask], dim=1)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        src_key_padding_mask = (attention_mask == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Get last valid token
        seq_lens = attention_mask.sum(dim=1).long()
        last_positions = seq_lens - 1
        batch_indices = torch.arange(batch_size, device=x.device)
        last_hidden = x[batch_indices, last_positions]

        return self.classifier(last_hidden)


class E2EDataset(Dataset):
    """Dataset that stores raw text, labels are extracted on-the-fly."""

    def __init__(self, data: List[Dict], tokenizer, max_mentor_tokens: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_mentor_tokens = max_mentor_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'question': item['question'],
            'mentor_response': item.get('mentor_response', ''),
            'label': 1 if item.get('is_correct', False) else 0,
        }


def collate_fn(batch, tokenizer, max_mentor_tokens, device):
    """Custom collate that tokenizes on-the-fly."""
    questions = [item['question'] for item in batch]
    mentor_responses = [item['mentor_response'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long, device=device)

    # We'll return raw data and process in training loop with model
    return {
        'questions': questions,
        'mentor_responses': mentor_responses,
        'labels': labels,
    }


def extract_hidden_states(
    model,
    tokenizer,
    questions: List[str],
    mentor_responses: List[str],
    max_mentor_tokens: int,
    device: str,
) -> tuple:
    """Extract hidden states from intern model."""
    batch_hidden = []
    batch_mask = []

    with torch.no_grad():
        for question, mentor_response in zip(questions, mentor_responses):
            # Tokenize
            question_ids = tokenizer.encode(question, return_tensors="pt").to(device)
            full_text = question + mentor_response
            full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

            question_len = question_ids.shape[1]
            total_len = full_ids.shape[1]
            mentor_len = total_len - question_len

            if mentor_len <= 0:
                # Empty mentor response
                batch_hidden.append(torch.zeros(1, model.config.hidden_size, device=device))
                batch_mask.append(torch.ones(1, device=device))
                continue

            # Truncate if needed
            if mentor_len > max_mentor_tokens:
                full_ids = full_ids[:, :question_len + max_mentor_tokens]
                mentor_len = max_mentor_tokens

            # Forward pass
            outputs = model(full_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]  # Last layer, [seq, hidden]

            # Extract mentor portion
            mentor_hidden = hidden[question_len:question_len + mentor_len]  # [mentor_len, hidden]
            batch_hidden.append(mentor_hidden)
            batch_mask.append(torch.ones(mentor_len, device=device))

    # Pad to same length
    max_len = max(h.shape[0] for h in batch_hidden)
    hidden_dim = batch_hidden[0].shape[1]

    padded_hidden = torch.zeros(len(batch_hidden), max_len, hidden_dim, device=device)
    padded_mask = torch.zeros(len(batch_hidden), max_len, device=device)

    for i, (h, m) in enumerate(zip(batch_hidden, batch_mask)):
        seq_len = h.shape[0]
        padded_hidden[i, :seq_len] = h
        padded_mask[i, :seq_len] = m

    return padded_hidden, padded_mask


def load_json_data(data_dir: str, token_level: int) -> List[Dict]:
    """Load collected JSON data."""
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and f'tokens{token_level}' in filename:
            if 'mentor_only' in filename:
                continue
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description="End-to-end classifier training")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_all_DeepSeek-R1-Distill-Qwen-32B",
                        help="Directory with collected JSON files")
    parser.add_argument("--intern-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--token-levels", type=int, nargs='+',
                        default=[0, 100, 500, 1000])
    parser.add_argument("--max-mentor-tokens", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-file", type=str, default=None)

    args = parser.parse_args()
    device = args.device

    # Load intern model (frozen)
    logger.info(f"Loading intern model: {args.intern_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.intern_model)
    intern_model = AutoModelForCausalLM.from_pretrained(
        args.intern_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        output_hidden_states=True,
    )
    intern_model.eval()
    for param in intern_model.parameters():
        param.requires_grad = False

    hidden_dim = intern_model.config.hidden_size
    logger.info(f"Hidden dim: {hidden_dim}")

    # Load data for all token levels
    all_data = {}
    for token_level in args.token_levels:
        data = load_json_data(args.data_dir, token_level)
        if data:
            all_data[token_level] = data
            logger.info(f"Loaded tokens{token_level}: {len(data)} samples")

    if not all_data:
        logger.error("No data found!")
        return

    # Split data (use same split across all token levels)
    n_samples = len(all_data[args.token_levels[0]])
    labels = [1 if item.get('is_correct', False) else 0 for item in all_data[args.token_levels[0]]]

    train_idx, val_idx = train_test_split(
        np.arange(n_samples),
        test_size=args.val_ratio,
        stratify=labels,
        random_state=42,
    )
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Create classifier
    classifier = TransformerClassifier(
        input_dim=hidden_dim,
        num_stages=len(args.token_levels),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    # Class weights
    train_labels = [labels[i] for i in train_idx]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    class_weights = torch.tensor([1.0 / neg_count, 1.0 / pos_count], device=device)
    class_weights = class_weights / class_weights.sum() * 2

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(args.epochs):
        # Training
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        # Shuffle training samples
        np.random.shuffle(train_idx)

        pbar = tqdm(range(0, len(train_idx), args.batch_size), desc=f"Epoch {epoch+1}")
        for batch_start in pbar:
            batch_idx = train_idx[batch_start:batch_start + args.batch_size]

            # Sample a random stage for each sample in batch
            batch_stages = np.random.randint(0, len(args.token_levels), size=len(batch_idx))

            questions = []
            mentor_responses = []
            batch_labels = []

            for i, (sample_idx, stage_idx) in enumerate(zip(batch_idx, batch_stages)):
                token_level = args.token_levels[stage_idx]
                item = all_data[token_level][sample_idx]
                questions.append(item['question'])
                mentor_responses.append(item.get('mentor_response', ''))
                batch_labels.append(1 if item.get('is_correct', False) else 0)

            # Extract hidden states
            hidden, mask = extract_hidden_states(
                intern_model, tokenizer, questions, mentor_responses,
                args.max_mentor_tokens, device
            )

            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
            stages_tensor = torch.tensor(batch_stages, dtype=torch.long, device=device)

            # Forward
            optimizer.zero_grad()
            logits = classifier(hidden.float(), mask, stages_tensor)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_idx)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels_tensor).sum().item()
            train_total += len(batch_idx)

            pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})

        scheduler.step()

        # Validation
        classifier.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_start in range(0, len(val_idx), args.batch_size):
                batch_idx = val_idx[batch_start:batch_start + args.batch_size]

                # Evaluate on all stages
                for stage_idx, token_level in enumerate(args.token_levels):
                    questions = []
                    mentor_responses = []
                    batch_labels = []

                    for sample_idx in batch_idx:
                        item = all_data[token_level][sample_idx]
                        questions.append(item['question'])
                        mentor_responses.append(item.get('mentor_response', ''))
                        batch_labels.append(1 if item.get('is_correct', False) else 0)

                    hidden, mask = extract_hidden_states(
                        intern_model, tokenizer, questions, mentor_responses,
                        args.max_mentor_tokens, device
                    )

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
                    stages_tensor = torch.full((len(batch_idx),), stage_idx, dtype=torch.long, device=device)

                    logits = classifier(hidden.float(), mask, stages_tensor)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels_tensor).sum().item()
                    val_total += len(batch_idx)

        val_acc = val_correct / val_total
        train_acc = train_correct / train_total

        logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = classifier.state_dict().copy()

    # Load best model
    classifier.load_state_dict(best_model_state)
    logger.info(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Cascade evaluation
    logger.info("\nCascade Evaluation:")
    classifier.eval()

    for threshold in [0.3, 0.5, 0.7, 0.9]:
        correct = 0
        total = 0
        token_usage = []

        with torch.no_grad():
            for sample_idx in val_idx:
                for stage_idx, token_level in enumerate(args.token_levels):
                    item = all_data[token_level][sample_idx]
                    question = item['question']
                    mentor_response = item.get('mentor_response', '')
                    label = 1 if item.get('is_correct', False) else 0

                    hidden, mask = extract_hidden_states(
                        intern_model, tokenizer, [question], [mentor_response],
                        args.max_mentor_tokens, device
                    )
                    stages_tensor = torch.tensor([stage_idx], device=device)

                    logits = classifier(hidden.float(), mask, stages_tensor)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()

                    if prob >= threshold:
                        correct += (label == 1)
                        token_usage.append(token_level)
                        break
                    elif stage_idx == len(args.token_levels) - 1:
                        # Fallback to stage 0
                        item0 = all_data[args.token_levels[0]][sample_idx]
                        label0 = 1 if item0.get('is_correct', False) else 0
                        correct += (label0 == 1)
                        token_usage.append(0)

                total += 1

        acc = correct / total
        avg_tokens = np.mean(token_usage)
        logger.info(f"  Threshold {threshold}: Acc={acc:.4f}, Avg Tokens={avg_tokens:.1f}")

    # Save results
    if args.output_file:
        results = {'best_val_acc': float(best_val_acc)}
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
