#!/usr/bin/env python3
"""
Train classifier on hidden states to predict if more mentor tokens are needed.

Uses hidden states extracted by collect_hidden_states.py to train a lightweight
classifier (MLP) for binary classification: need more tokens (1) or sufficient (0).
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HiddenStateDataset(Dataset):
    """Dataset for hidden states classification."""

    def __init__(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            hidden_states: [num_samples, num_layers, hidden_dim] or [num_samples, hidden_dim]
            labels: [num_samples]
        """
        # Flatten if multi-layer
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(hidden_states.size(0), -1)
        self.hidden_states = hidden_states.float()
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.labels[idx]


class HiddenClassifier(nn.Module):
    """Simple MLP classifier for hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_hidden_data(data_dir: str, token_levels: List[int] = None) -> Dict[int, Dict]:
    """Load hidden states from .pt files."""
    data = {}
    token_levels = token_levels or [0, 100, 500, 1000]

    for tokens in token_levels:
        filepath = os.path.join(data_dir, f"tokens{tokens}.pt")
        if os.path.exists(filepath):
            data[tokens] = torch.load(filepath)
            logger.info(f"Loaded tokens={tokens}: shape={data[tokens]['hidden_states'].shape}")
        else:
            logger.warning(f"File not found: {filepath}")

    return data


def prepare_binary_data(
    data: Dict[int, Dict],
    low_tokens: int,
    high_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Prepare binary classification data.

    Label = 0: low_tokens is sufficient (correct with low tokens)
    Label = 1: need more tokens (wrong with low, correct with high)

    Args:
        data: Dict mapping token counts to hidden states and labels
        low_tokens: Lower token count (e.g., 100)
        high_tokens: Higher token count (e.g., 500)

    Returns:
        hidden_states, labels, sample_indices
    """
    low_data = data[low_tokens]
    high_data = data[high_tokens]

    low_hidden = low_data['hidden_states']
    low_labels = low_data['labels']
    high_labels = high_data['labels']

    n_samples = len(low_labels)
    assert len(high_labels) == n_samples, "Sample count mismatch"

    # Binary classification:
    # 0: correct at low tokens (no need for more)
    # 1: wrong at low but correct at high (need more tokens)
    binary_labels = []
    valid_indices = []

    for i in range(n_samples):
        if low_labels[i] == 1:
            # Already correct at low tokens
            binary_labels.append(0)
            valid_indices.append(i)
        elif low_labels[i] == 0 and high_labels[i] == 1:
            # Need more tokens
            binary_labels.append(1)
            valid_indices.append(i)
        # Skip samples that are wrong at both levels

    valid_hidden = low_hidden[valid_indices]
    binary_labels = torch.tensor(binary_labels)

    logger.info(f"Binary data: {len(valid_indices)} samples, "
                f"label=0 (sufficient): {(binary_labels == 0).sum()}, "
                f"label=1 (need more): {(binary_labels == 1).sum()}")

    return valid_hidden, binary_labels, valid_indices


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for hidden, labels in dataloader:
        hidden = hidden.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(hidden)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []

    with torch.no_grad():
        for hidden, labels in dataloader:
            hidden = hidden.to(device)
            labels = labels.to(device)

            outputs = model(hidden)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    return total_loss / total, correct / total, all_probs


def run_cross_validation(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Dict:
    """Run k-fold cross-validation."""

    # Flatten hidden states if needed
    if hidden_states.dim() == 3:
        hidden_states = hidden_states.view(hidden_states.size(0), -1)

    input_dim = hidden_states.size(1)
    logger.info(f"Input dim: {input_dim}")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(hidden_states)):
        logger.info(f"\n=== Fold {fold + 1}/{n_folds} ===")

        train_hidden = hidden_states[train_idx]
        train_labels = labels[train_idx]
        val_hidden = hidden_states[val_idx]
        val_labels = labels[val_idx]

        train_dataset = HiddenStateDataset(train_hidden, train_labels)
        val_dataset = HiddenStateDataset(val_hidden, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Class weights for imbalanced data
        class_counts = torch.bincount(train_labels)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * 2
        class_weights = class_weights.to(device)

        model = HiddenClassifier(input_dim).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                           f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
        })
        logger.info(f"Fold {fold + 1} best val acc: {best_val_acc:.4f}")

    # Summary
    mean_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_acc = np.std([r['best_val_acc'] for r in fold_results])
    logger.info(f"\n=== Cross-Validation Results ===")
    logger.info(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    return {
        'fold_results': fold_results,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train hidden state classifier")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with hidden states .pt files")
    parser.add_argument("--low-tokens", type=int, default=100,
                        help="Lower token level for comparison")
    parser.add_argument("--high-tokens", type=int, default=500,
                        help="Higher token level for comparison")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-file", type=str, default=None)

    args = parser.parse_args()

    # Load hidden states
    data = load_hidden_data(args.data_dir, [args.low_tokens, args.high_tokens])
    if len(data) < 2:
        logger.error("Need both low and high token data")
        return

    # Prepare binary classification data
    hidden_states, labels, _ = prepare_binary_data(data, args.low_tokens, args.high_tokens)

    # Run cross-validation
    results = run_cross_validation(
        hidden_states,
        labels,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
