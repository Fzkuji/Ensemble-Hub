#!/usr/bin/env python3
"""
Train a transformer-based classifier on hidden state sequences.

Architecture:
1. Input: sequence of hidden states [batch, seq_len, hidden_dim]
2. Add learnable positional encoding
3. Pass through transformer encoder layers
4. Take the last valid token's output (using attention mask)
5. Pass through MLP for binary classification
"""

import argparse
import json
import logging
import math
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for hidden state sequences.

    Takes a sequence of hidden states, processes with transformer,
    and uses the last token's representation for classification.
    """

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

        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stage embedding (prepended as first token)
        self.stage_embedding = nn.Embedding(num_stages, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
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
            attention_mask: [batch, seq_len] - 1 for valid, 0 for padding
            stages: [batch] - stage indices

        Returns:
            logits: [batch, 2]
        """
        batch_size = hidden_states.size(0)

        # Project to d_model
        x = self.input_proj(hidden_states)  # [batch, seq_len, d_model]

        # Prepend stage embedding as first token
        stage_embed = self.stage_embedding(stages).unsqueeze(1)  # [batch, 1, d_model]
        x = torch.cat([stage_embed, x], dim=1)  # [batch, seq_len+1, d_model]

        # Update attention mask for stage token
        stage_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        attention_mask = torch.cat([stage_mask, attention_mask], dim=1)  # [batch, seq_len+1]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for transformer (True = ignore)
        src_key_padding_mask = (attention_mask == 0)

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Get last valid token's output for each sample
        # Find the last valid position (where mask is 1)
        seq_lens = attention_mask.sum(dim=1).long()  # [batch]
        last_positions = seq_lens - 1  # [batch]

        # Gather last valid positions
        batch_indices = torch.arange(batch_size, device=x.device)
        last_hidden = x[batch_indices, last_positions]  # [batch, d_model]

        # Classify
        logits = self.classifier(last_hidden)
        return logits


class HiddenSeqDataset(Dataset):
    """Dataset for hidden state sequences."""

    def __init__(self, hidden_states, attention_mask, labels, stages):
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self.labels = labels
        self.stages = stages

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'hidden_states': self.hidden_states[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'stages': self.stages[idx],
        }


def load_seq_data(data_dir: str) -> Dict[int, Dict]:
    """Load hidden state sequences from .pt files."""
    data = {}
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, f"tokens{tokens}.pt")
        if os.path.exists(filepath):
            loaded = torch.load(filepath)
            data[tokens] = loaded
            logger.info(f"Loaded tokens{tokens}: {loaded['hidden_states'].shape}")
    return data


def prepare_data(data: Dict[int, Dict]) -> Tuple[torch.Tensor, ...]:
    """Prepare data for training - combine all stages."""
    all_hidden = []
    all_mask = []
    all_labels = []
    all_stages = []

    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        if tokens not in data:
            continue

        stage_data = data[tokens]
        n = len(stage_data['labels'])

        all_hidden.append(stage_data['hidden_states'])
        all_mask.append(stage_data['attention_mask'])
        all_labels.append(stage_data['labels'])
        all_stages.append(torch.full((n,), stage_idx, dtype=torch.long))

    return (
        torch.cat(all_hidden, dim=0),
        torch.cat(all_mask, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_stages, dim=0),
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        hidden = batch['hidden_states'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        stages = batch['stages'].to(device)

        optimizer.zero_grad()
        logits = model(hidden, mask, stages)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_stages = []

    with torch.no_grad():
        for batch in dataloader:
            hidden = batch['hidden_states'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            stages = batch['stages'].to(device)

            logits = model(hidden, mask, stages)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_stages.extend(stages.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels, all_stages


def simulate_cascade(
    model: nn.Module,
    data: Dict[int, Dict],
    device: str,
    threshold: float = 0.5,
) -> Dict:
    """Simulate cascaded inference."""
    model.eval()
    n_samples = len(data[TOKEN_LEVELS[0]]['labels'])

    gt = {tokens: data[tokens]['labels'].numpy() for tokens in TOKEN_LEVELS}

    final_tokens = np.zeros(n_samples, dtype=int)
    final_correct = np.zeros(n_samples, dtype=bool)

    with torch.no_grad():
        for i in range(n_samples):
            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                hidden = data[tokens]['hidden_states'][i:i+1].to(device)
                mask = data[tokens]['attention_mask'][i:i+1].to(device)
                stage_tensor = torch.tensor([stage_idx], device=device)

                logits = model(hidden, mask, stage_tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].item()

                if prob >= threshold:
                    final_tokens[i] = tokens
                    final_correct[i] = gt[tokens][i] == 1
                    break
                elif stage_idx == len(TOKEN_LEVELS) - 1:
                    final_tokens[i] = 0
                    final_correct[i] = gt[0][i] == 1

    accuracy = final_correct.mean()
    avg_tokens = final_tokens.mean()
    token_dist = {tokens: int((final_tokens == tokens).sum()) for tokens in TOKEN_LEVELS}

    return {
        'accuracy': float(accuracy),
        'avg_tokens': float(avg_tokens),
        'token_distribution': token_dist,
    }


def main():
    parser = argparse.ArgumentParser(description="Train transformer classifier")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with hidden_seq .pt files")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-file", type=str, default=None)

    args = parser.parse_args()
    device = args.device

    # Load data
    data = load_seq_data(args.data_dir)
    if not data:
        logger.error("No data found!")
        return

    # Prepare combined data
    all_hidden, all_mask, all_labels, all_stages = prepare_data(data)
    input_dim = all_hidden.shape[2]
    logger.info(f"Total samples: {len(all_labels)}, Input dim: {input_dim}")

    # Train/val split (stratified by stage 0 labels)
    n_per_stage = len(data[TOKEN_LEVELS[0]]['labels'])
    stage0_labels = all_labels[:n_per_stage].numpy()

    train_idx, val_idx = train_test_split(
        np.arange(n_per_stage),
        test_size=args.val_ratio,
        stratify=stage0_labels,
        random_state=42,
    )

    # Expand indices for all stages
    all_train_idx = []
    all_val_idx = []
    for stage_idx in range(len(TOKEN_LEVELS)):
        offset = stage_idx * n_per_stage
        all_train_idx.extend(train_idx + offset)
        all_val_idx.extend(val_idx + offset)

    # Create datasets
    train_dataset = HiddenSeqDataset(
        all_hidden[all_train_idx],
        all_mask[all_train_idx],
        all_labels[all_train_idx],
        all_stages[all_train_idx],
    )
    val_dataset = HiddenSeqDataset(
        all_hidden[all_val_idx],
        all_mask[all_val_idx],
        all_labels[all_val_idx],
        all_stages[all_val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = TransformerClassifier(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    # Class weights for imbalanced data
    train_labels = all_labels[all_train_idx]
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                   f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                   f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    logger.info(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Prepare validation data for cascade simulation
    val_data = {}
    for tokens in TOKEN_LEVELS:
        if tokens in data:
            val_data[tokens] = {
                'hidden_states': data[tokens]['hidden_states'][val_idx],
                'attention_mask': data[tokens]['attention_mask'][val_idx],
                'labels': data[tokens]['labels'][val_idx],
            }

    # Evaluate cascade at different thresholds
    logger.info("\nCascade Results:")
    cascade_results = {}
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        result = simulate_cascade(model, val_data, device, threshold)
        cascade_results[threshold] = result
        logger.info(f"  Threshold {threshold}: Acc={result['accuracy']:.4f}, "
                   f"Avg Tokens={result['avg_tokens']:.1f}, Dist={result['token_distribution']}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Binary Classification Accuracy: {best_val_acc:.4f}")
    logger.info(f"\n{'Threshold':<10} {'Accuracy':<15} {'Avg Tokens':<15}")
    logger.info("-" * 50)
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        r = cascade_results[threshold]
        logger.info(f"{threshold:<10} {r['accuracy']:.4f}          {r['avg_tokens']:<15.1f}")

    # Save results
    if args.output_file:
        results = {
            'best_val_acc': float(best_val_acc),
            'cascade_results': {str(k): v for k, v in cascade_results.items()},
        }
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
