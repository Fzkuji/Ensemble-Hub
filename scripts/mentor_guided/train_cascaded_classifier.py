#!/usr/bin/env python3
"""
Train cascaded binary classifier for adaptive mentor token allocation.

Each stage predicts: is current information sufficient to answer correctly?
- Stage 0 (0 tokens):   sufficient? → Yes: output, No: → Stage 1
- Stage 1 (100 tokens): sufficient? → Yes: output, No: → Stage 2
- Stage 2 (500 tokens): sufficient? → Yes: output, No: → Stage 3
- Stage 3 (1000 tokens): sufficient? → Yes: output, No: → back to Stage 0

Uses hidden states from each stage to train a shared binary classifier.
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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Token levels for each stage
TOKEN_LEVELS = [0, 100, 500, 1000]


class CascadedDataset(Dataset):
    """Dataset for cascaded classifier training."""

    def __init__(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        stages: torch.Tensor,
    ):
        self.hidden_states = hidden_states.float()
        self.labels = labels.long()
        self.stages = stages.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.labels[idx], self.stages[idx]


class CascadedClassifier(nn.Module):
    """Shared classifier for all stages with stage embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_stages: int = 4,
        stage_embed_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, stage_embed_dim)
        total_input_dim = input_dim + stage_embed_dim

        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary: sufficient (1) or not (0)
        )

    def forward(self, hidden_states, stages):
        stage_embed = self.stage_embedding(stages)
        x = torch.cat([hidden_states, stage_embed], dim=-1)
        return self.net(x)


def load_all_hidden_data(data_dir: str) -> Dict[int, Dict]:
    """Load hidden states from all token level .pt files."""
    data = {}
    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, f"tokens{tokens}.pt")
        if os.path.exists(filepath):
            loaded = torch.load(filepath)
            data[tokens] = loaded
            logger.info(f"Loaded tokens={tokens}: shape={loaded['hidden_states'].shape}, "
                       f"correct={loaded['labels'].sum()}/{len(loaded['labels'])}")
        else:
            logger.warning(f"File not found: {filepath}")
    return data


def prepare_cascaded_data(
    data: Dict[int, Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare training data for cascaded classifier.

    Each sample at each stage becomes a training example:
    - hidden_states from that stage
    - label = 1 if correct at that stage, 0 if wrong
    - stage index
    """
    all_hidden = []
    all_labels = []
    all_stages = []

    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        if tokens not in data:
            continue

        stage_data = data[tokens]
        hidden = stage_data['hidden_states']
        labels = stage_data['labels']

        # Flatten hidden states if multi-layer
        if hidden.dim() == 3:
            hidden = hidden.view(hidden.size(0), -1)

        n_samples = len(labels)
        all_hidden.append(hidden)
        all_labels.append(labels)
        all_stages.append(torch.full((n_samples,), stage_idx, dtype=torch.long))

        logger.info(f"Stage {stage_idx} (tokens={tokens}): {n_samples} samples, "
                   f"sufficient={labels.sum()}, insufficient={n_samples - labels.sum()}")

    all_hidden = torch.cat(all_hidden, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_stages = torch.cat(all_stages, dim=0)

    logger.info(f"\nTotal: {len(all_labels)} samples")
    logger.info(f"Label distribution: sufficient={all_labels.sum()}, "
               f"insufficient={len(all_labels) - all_labels.sum()}")

    return all_hidden, all_labels, all_stages


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

    for hidden, labels, stages in dataloader:
        hidden = hidden.to(device)
        labels = labels.to(device)
        stages = stages.to(device)

        optimizer.zero_grad()
        outputs = model(hidden, stages)
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
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_stages = []

    with torch.no_grad():
        for hidden, labels, stages in dataloader:
            hidden = hidden.to(device)
            labels = labels.to(device)
            stages_device = stages.to(device)

            outputs = model(hidden, stages_device)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_stages.append(stages)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_stages = torch.cat(all_stages).numpy()

    accuracy = (all_preds == all_labels).mean()

    return total_loss / len(all_labels), accuracy, all_preds, all_labels, all_stages


def simulate_cascade(
    model: nn.Module,
    data: Dict[int, Dict],
    device: str,
    threshold: float = 0.5,
) -> Dict:
    """
    Simulate the cascaded inference process.
    """
    model.eval()
    n_samples = len(data[TOKEN_LEVELS[0]]['labels'])

    # Ground truth at each stage
    gt = {tokens: data[tokens]['labels'].numpy() for tokens in TOKEN_LEVELS}

    # Track decisions
    final_tokens = np.zeros(n_samples, dtype=int)
    final_correct = np.zeros(n_samples, dtype=bool)

    with torch.no_grad():
        for i in range(n_samples):
            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                hidden = data[tokens]['hidden_states'][i:i+1]
                if hidden.dim() == 3:
                    hidden = hidden.view(1, -1)
                hidden = hidden.float().to(device)
                stage_tensor = torch.tensor([stage_idx], device=device)

                output = model(hidden, stage_tensor)
                prob_sufficient = torch.softmax(output, dim=1)[0, 1].item()

                if prob_sufficient >= threshold:
                    # Predict sufficient at this stage
                    final_tokens[i] = tokens
                    final_correct[i] = gt[tokens][i] == 1
                    break
                elif stage_idx == len(TOKEN_LEVELS) - 1:
                    # Last stage, fall back to stage 0
                    final_tokens[i] = 0
                    final_correct[i] = gt[0][i] == 1

    # Compute statistics
    accuracy = final_correct.mean()
    avg_tokens = final_tokens.mean()
    token_dist = {tokens: int((final_tokens == tokens).sum()) for tokens in TOKEN_LEVELS}

    return {
        'accuracy': float(accuracy),
        'avg_tokens': float(avg_tokens),
        'token_distribution': token_dist,
    }


def run_cross_validation(
    all_hidden: torch.Tensor,
    all_labels: torch.Tensor,
    all_stages: torch.Tensor,
    data: Dict[int, Dict],
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Dict:
    """Run stratified k-fold cross-validation."""

    input_dim = all_hidden.size(1)
    logger.info(f"Input dim: {input_dim}")

    # Use sample indices from stage 0 for stratification
    n_samples_per_stage = len(data[TOKEN_LEVELS[0]]['labels'])
    stage0_labels = all_labels[:n_samples_per_stage].numpy()

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_sample_idx, val_sample_idx) in enumerate(kfold.split(
        np.zeros(n_samples_per_stage), stage0_labels
    )):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{n_folds}")
        logger.info(f"{'='*50}")

        # Expand indices to all stages
        train_idx = []
        val_idx = []
        for stage_idx in range(len(TOKEN_LEVELS)):
            offset = stage_idx * n_samples_per_stage
            train_idx.extend(train_sample_idx + offset)
            val_idx.extend(val_sample_idx + offset)

        train_hidden = all_hidden[train_idx]
        train_labels = all_labels[train_idx]
        train_stages = all_stages[train_idx]
        val_hidden = all_hidden[val_idx]
        val_labels = all_labels[val_idx]
        val_stages = all_stages[val_idx]

        train_dataset = CascadedDataset(train_hidden, train_labels, train_stages)
        val_dataset = CascadedDataset(val_hidden, val_labels, val_stages)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Class weights for imbalanced data
        class_counts = torch.bincount(train_labels)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * 2
        class_weights = class_weights.to(device)

        model = CascadedClassifier(input_dim).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                           f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        _, _, val_preds, val_true, val_stage_arr = evaluate(model, val_loader, criterion, device)

        # Per-stage accuracy
        stage_accs = {}
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            mask = val_stage_arr == stage_idx
            if mask.sum() > 0:
                stage_acc = (val_preds[mask] == val_true[mask]).mean()
                stage_accs[tokens] = float(stage_acc)
                logger.info(f"  Stage {stage_idx} (tokens={tokens}): Acc={stage_acc:.4f}")

        # Simulate cascade on validation set
        val_data = {}
        for tokens in TOKEN_LEVELS:
            val_data[tokens] = {
                'hidden_states': data[tokens]['hidden_states'][val_sample_idx],
                'labels': data[tokens]['labels'][val_sample_idx],
            }

        cascade_results = {}
        for threshold in [0.3, 0.5, 0.7]:
            result = simulate_cascade(model, val_data, device, threshold)
            cascade_results[threshold] = result
            logger.info(f"  Threshold {threshold}: Acc={result['accuracy']:.4f}, "
                       f"Avg Tokens={result['avg_tokens']:.1f}, "
                       f"Dist={result['token_distribution']}")

        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': float(best_val_acc),
            'stage_accs': stage_accs,
            'cascade_results': cascade_results,
        })

    # Summary
    mean_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_acc = np.std([r['best_val_acc'] for r in fold_results])

    logger.info(f"\n{'='*50}")
    logger.info(f"Cross-Validation Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Mean Binary Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Average cascade results
    for threshold in [0.3, 0.5, 0.7]:
        accs = [r['cascade_results'][threshold]['accuracy'] for r in fold_results]
        tokens = [r['cascade_results'][threshold]['avg_tokens'] for r in fold_results]
        logger.info(f"Cascade (threshold={threshold}): "
                   f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
                   f"Avg Tokens={np.mean(tokens):.1f}")

    return {
        'fold_results': fold_results,
        'mean_acc': float(mean_acc),
        'std_acc': float(std_acc),
    }


def main():
    parser = argparse.ArgumentParser(description="Train cascaded hidden state classifier")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with hidden states .pt files")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-file", type=str, default=None)

    args = parser.parse_args()

    # Load all hidden states
    data = load_all_hidden_data(args.data_dir)
    if len(data) < len(TOKEN_LEVELS):
        logger.warning(f"Only found {len(data)} token levels, expected {len(TOKEN_LEVELS)}")

    # Prepare cascaded training data
    all_hidden, all_labels, all_stages = prepare_cascaded_data(data)

    # Run cross-validation
    results = run_cross_validation(
        all_hidden,
        all_labels,
        all_stages,
        data,
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
