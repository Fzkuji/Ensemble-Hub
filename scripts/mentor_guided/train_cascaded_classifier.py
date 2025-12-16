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
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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


class AttentionClassifier(nn.Module):
    """
    Attention-based classifier that treats hidden states as a sequence of chunks.
    Uses self-attention to learn which parts of the hidden states are most important.
    """

    def __init__(
        self,
        input_dim: int,
        num_stages: int = 4,
        num_heads: int = 4,
        chunk_size: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_chunks = input_dim // chunk_size
        assert input_dim % chunk_size == 0, f"input_dim {input_dim} must be divisible by chunk_size {chunk_size}"

        # Stage embedding
        self.stage_embedding = nn.Embedding(num_stages, chunk_size)

        # Project chunks to hidden_dim
        self.input_proj = nn.Linear(chunk_size, hidden_dim)

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Cross-attention for pooling
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, hidden_states, stages):
        batch_size = hidden_states.size(0)

        # Reshape to chunks: [batch, num_chunks, chunk_size]
        x = hidden_states.view(batch_size, self.num_chunks, self.chunk_size)

        # Add stage embedding as an extra token
        stage_embed = self.stage_embedding(stages).unsqueeze(1)  # [batch, 1, chunk_size]
        x = torch.cat([stage_embed, x], dim=1)  # [batch, num_chunks+1, chunk_size]

        # Project to hidden dim
        x = self.input_proj(x)  # [batch, num_chunks+1, hidden_dim]

        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.attn_norm(x + attn_out)

        # Cross-attention pooling with learnable query
        query = self.pool_query.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        pooled, attn_weights = self.cross_attn(query, x, x)
        pooled = self.cross_norm(pooled).squeeze(1)  # [batch, hidden_dim]

        # Classify
        return self.classifier(pooled)


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
    per_stage_thresholds: List[float] = None,
) -> Dict:
    """
    Simulate the cascaded inference process.

    Args:
        model: Trained classifier
        data: Hidden states and labels for each token level
        device: Device to run on
        threshold: Global threshold (used if per_stage_thresholds is None)
        per_stage_thresholds: List of thresholds for each stage [t0, t1, t2, t3]
    """
    model.eval()
    n_samples = len(data[TOKEN_LEVELS[0]]['labels'])

    # Use per-stage thresholds if provided, otherwise use global threshold
    if per_stage_thresholds is not None:
        thresholds = per_stage_thresholds
    else:
        thresholds = [threshold] * len(TOKEN_LEVELS)

    # Ground truth at each stage
    gt = {tokens: data[tokens]['labels'].numpy() for tokens in TOKEN_LEVELS}

    # Track decisions
    final_tokens = np.zeros(n_samples, dtype=int)
    final_correct = np.zeros(n_samples, dtype=bool)

    with torch.no_grad():
        for i in range(n_samples):
            # Track probabilities for fallback selection
            stage_probs = []
            decided = False

            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                hidden = data[tokens]['hidden_states'][i:i+1]
                if hidden.dim() == 3:
                    hidden = hidden.view(1, -1)
                hidden = hidden.float().to(device)
                stage_tensor = torch.tensor([stage_idx], device=device)

                output = model(hidden, stage_tensor)
                prob_sufficient = torch.softmax(output, dim=1)[0, 1].item()
                stage_probs.append((stage_idx, tokens, prob_sufficient))

                stage_threshold = thresholds[stage_idx]
                if prob_sufficient >= stage_threshold:
                    # Predict sufficient at this stage
                    final_tokens[i] = tokens
                    final_correct[i] = gt[tokens][i] == 1
                    decided = True
                    break

            if not decided:
                # No stage passed threshold, select the one with highest confidence
                best_stage_idx, best_tokens, _ = max(stage_probs, key=lambda x: x[2])
                final_tokens[i] = best_tokens
                final_correct[i] = gt[best_tokens][i] == 1

    # Compute statistics
    accuracy = final_correct.mean()
    avg_tokens = final_tokens.mean()
    token_dist = {tokens: int((final_tokens == tokens).sum()) for tokens in TOKEN_LEVELS}

    return {
        'accuracy': float(accuracy),
        'avg_tokens': float(avg_tokens),
        'token_distribution': token_dist,
        'thresholds': thresholds,
    }


def search_optimal_thresholds(
    model: nn.Module,
    data: Dict[int, Dict],
    device: str,
    threshold_candidates: List[float] = None,
    optimize_for: str = "accuracy",  # "accuracy" or "efficiency" or "balanced"
) -> Dict:
    """
    Search for optimal per-stage thresholds using grid search.

    Args:
        model: Trained classifier
        data: Hidden states and labels
        device: Device
        threshold_candidates: List of threshold values to try (default: [0.3, 0.5, 0.7, 0.9])
        optimize_for: Optimization target
            - "accuracy": maximize accuracy
            - "efficiency": minimize tokens while maintaining accuracy
            - "balanced": balance accuracy and token efficiency

    Returns:
        Dict with best thresholds and results
    """
    if threshold_candidates is None:
        threshold_candidates = [0.3, 0.5, 0.7, 0.9]

    n_stages = len(TOKEN_LEVELS)
    best_result = None
    best_score = -float('inf')
    all_results = []

    # Grid search over all combinations
    from itertools import product
    total_combos = len(threshold_candidates) ** n_stages
    logger.info(f"Searching {total_combos} threshold combinations...")

    for combo in product(threshold_candidates, repeat=n_stages):
        thresholds = list(combo)
        result = simulate_cascade(model, data, device, per_stage_thresholds=thresholds)

        # Calculate score based on optimization target
        if optimize_for == "accuracy":
            score = result['accuracy']
        elif optimize_for == "efficiency":
            # Maximize accuracy - penalty for more tokens
            score = result['accuracy'] - result['avg_tokens'] / 1000.0 * 0.1
        elif optimize_for == "balanced":
            # Weighted combination
            score = result['accuracy'] * 0.7 + (1 - result['avg_tokens'] / 1000.0) * 0.3
        else:
            score = result['accuracy']

        result['score'] = score
        all_results.append(result)

        if score > best_score:
            best_score = score
            best_result = result

    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)

    return {
        'best': best_result,
        'top_5': all_results[:5],
        'optimize_for': optimize_for,
    }


class XGBoostCascadeClassifier:
    """Wrapper for per-stage XGBoost classifiers."""

    def __init__(self, models: Dict[int, 'xgb.XGBClassifier']):
        self.models = models  # {stage_idx: xgb_model}

    def predict_proba(self, hidden_states: np.ndarray, stage_idx: int) -> np.ndarray:
        """Predict probability of sufficient for given stage."""
        return self.models[stage_idx].predict_proba(hidden_states)


def simulate_cascade_xgb(
    model: XGBoostCascadeClassifier,
    data: Dict[int, Dict],
    threshold: float = 0.5,
    per_stage_thresholds: List[float] = None,
) -> Dict:
    """Simulate cascade for XGBoost models."""
    n_samples = len(data[TOKEN_LEVELS[0]]['labels'])

    if per_stage_thresholds is not None:
        thresholds = per_stage_thresholds
    else:
        thresholds = [threshold] * len(TOKEN_LEVELS)

    gt = {tokens: data[tokens]['labels'].numpy() for tokens in TOKEN_LEVELS}

    final_tokens = np.zeros(n_samples, dtype=int)
    final_correct = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        # Track probabilities for fallback selection
        stage_probs = []
        decided = False

        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            hidden = data[tokens]['hidden_states'][i:i+1].numpy()
            if hidden.ndim == 3:
                hidden = hidden.reshape(1, -1)

            proba = model.predict_proba(hidden, stage_idx)
            prob_sufficient = proba[0, 1]
            stage_probs.append((stage_idx, tokens, prob_sufficient))

            if prob_sufficient >= thresholds[stage_idx]:
                final_tokens[i] = tokens
                final_correct[i] = gt[tokens][i] == 1
                decided = True
                break

        if not decided:
            # No stage passed threshold, select the one with highest confidence
            best_stage_idx, best_tokens, _ = max(stage_probs, key=lambda x: x[2])
            final_tokens[i] = best_tokens
            final_correct[i] = gt[best_tokens][i] == 1

    accuracy = final_correct.mean()
    avg_tokens = final_tokens.mean()
    token_dist = {tokens: int((final_tokens == tokens).sum()) for tokens in TOKEN_LEVELS}

    return {
        'accuracy': float(accuracy),
        'avg_tokens': float(avg_tokens),
        'token_distribution': token_dist,
        'thresholds': thresholds,
    }


def run_xgboost_single_split(
    all_hidden: torch.Tensor,
    all_labels: torch.Tensor,
    all_stages: torch.Tensor,
    data: Dict[int, Dict],
    val_ratio: float = 0.2,
) -> Dict:
    """Train per-stage XGBoost classifiers with 80:20 split."""

    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    input_dim = all_hidden.size(1)
    logger.info(f"Input dim: {input_dim}")

    n_samples_per_stage = len(data[TOKEN_LEVELS[0]]['labels'])
    stage0_labels = all_labels[:n_samples_per_stage].numpy()

    train_sample_idx, val_sample_idx = train_test_split(
        np.arange(n_samples_per_stage),
        test_size=val_ratio,
        stratify=stage0_labels,
        random_state=42
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"XGBoost Train/Val Split: {len(train_sample_idx)}/{len(val_sample_idx)} samples per stage")
    logger.info(f"{'='*50}")

    # Train per-stage XGBoost models
    stage_models = {}
    stage_accs = {}

    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        logger.info(f"\nTraining XGBoost for Stage {stage_idx} (tokens={tokens})...")

        # Get data for this stage
        offset = stage_idx * n_samples_per_stage
        stage_train_idx = train_sample_idx + offset
        stage_val_idx = val_sample_idx + offset

        X_train = all_hidden[stage_train_idx].numpy()
        y_train = all_labels[stage_train_idx].numpy()
        X_val = all_hidden[stage_val_idx].numpy()
        y_val = all_labels[stage_val_idx].numpy()

        # Compute scale_pos_weight for imbalanced data
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            early_stopping_rounds=10,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        val_pred = model.predict(X_val)
        stage_acc = (val_pred == y_val).mean()
        stage_accs[tokens] = float(stage_acc)
        stage_models[stage_idx] = model

        logger.info(f"  Stage {stage_idx} (tokens={tokens}): Val Acc={stage_acc:.4f}")

    # Create cascade classifier
    cascade_model = XGBoostCascadeClassifier(stage_models)

    # Prepare validation data
    val_data = {}
    for tokens in TOKEN_LEVELS:
        val_data[tokens] = {
            'hidden_states': data[tokens]['hidden_states'][val_sample_idx],
            'labels': data[tokens]['labels'][val_sample_idx],
        }

    # Evaluate cascade with global thresholds
    logger.info(f"\nCascade Results (Global Threshold):")
    cascade_results = {}
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        result = simulate_cascade_xgb(cascade_model, val_data, threshold)
        cascade_results[threshold] = result
        logger.info(f"  Threshold {threshold}: Acc={result['accuracy']:.4f}, "
                   f"Avg Tokens={result['avg_tokens']:.1f}, "
                   f"Dist={result['token_distribution']}")

    # Search for optimal per-stage thresholds
    logger.info(f"\nSearching Optimal Per-Stage Thresholds (XGBoost):")
    per_stage_results = {}

    threshold_candidates = [0.3, 0.5, 0.7, 0.9]
    from itertools import product

    for opt_target in ["accuracy", "efficiency", "balanced"]:
        best_result = None
        best_score = -float('inf')

        for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
            thresholds = list(combo)
            result = simulate_cascade_xgb(cascade_model, val_data, per_stage_thresholds=thresholds)

            if opt_target == "accuracy":
                score = result['accuracy']
            elif opt_target == "efficiency":
                score = result['accuracy'] - result['avg_tokens'] / 1000.0 * 0.1
            elif opt_target == "balanced":
                score = result['accuracy'] * 0.7 + (1 - result['avg_tokens'] / 1000.0) * 0.3
            else:
                score = result['accuracy']

            if score > best_score:
                best_score = score
                best_result = result

        per_stage_results[opt_target] = {'best': best_result}
        best = best_result
        logger.info(f"  [{opt_target}] Best thresholds: {best['thresholds']}")
        logger.info(f"           Acc={best['accuracy']:.4f}, Avg Tokens={best['avg_tokens']:.1f}, "
                   f"Dist={best['token_distribution']}")

    # Summary
    mean_stage_acc = np.mean(list(stage_accs.values()))
    logger.info(f"\n{'='*50}")
    logger.info(f"XGBoost Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Mean Per-Stage Binary Accuracy: {mean_stage_acc:.4f}")

    logger.info(f"\n{'Threshold':<10} {'Accuracy':<15} {'Avg Tokens':<15}")
    logger.info("-" * 50)
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        result = cascade_results[threshold]
        logger.info(f"{threshold:<10} {result['accuracy']:.4f}          {result['avg_tokens']:<15.1f}")

    return {
        'mean_stage_acc': float(mean_stage_acc),
        'stage_accs': stage_accs,
        'cascade_results': cascade_results,
        'per_stage_thresholds': per_stage_results,
    }


def run_single_split(
    all_hidden: torch.Tensor,
    all_labels: torch.Tensor,
    all_stages: torch.Tensor,
    data: Dict[int, Dict],
    val_ratio: float = 0.2,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda",
    model_type: str = "mlp",  # "mlp" or "attention"
) -> Dict:
    """Run training with simple train/val split (80:20)."""

    input_dim = all_hidden.size(1)
    logger.info(f"Input dim: {input_dim}")

    # Use sample indices from stage 0 for stratification
    n_samples_per_stage = len(data[TOKEN_LEVELS[0]]['labels'])
    stage0_labels = all_labels[:n_samples_per_stage].numpy()

    # Stratified 80:20 split
    train_sample_idx, val_sample_idx = train_test_split(
        np.arange(n_samples_per_stage),
        test_size=val_ratio,
        stratify=stage0_labels,
        random_state=42
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Train/Val Split: {len(train_sample_idx)}/{len(val_sample_idx)} samples per stage")
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

    if model_type == "attention":
        model = AttentionClassifier(input_dim).to(device)
        logger.info("Using AttentionClassifier")
    else:
        model = CascadedClassifier(input_dim).to(device)
        logger.info("Using CascadedClassifier (MLP)")
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
    logger.info(f"\nPer-stage Binary Accuracy:")
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

    logger.info(f"\nCascade Results (Global Threshold):")
    cascade_results = {}
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        result = simulate_cascade(model, val_data, device, threshold)
        cascade_results[threshold] = result
        logger.info(f"  Threshold {threshold}: Acc={result['accuracy']:.4f}, "
                   f"Avg Tokens={result['avg_tokens']:.1f}, "
                   f"Dist={result['token_distribution']}")

    # Search for optimal per-stage thresholds
    logger.info(f"\nSearching Optimal Per-Stage Thresholds:")
    per_stage_results = {}
    for opt_target in ["accuracy", "efficiency", "balanced"]:
        opt_result = search_optimal_thresholds(model, val_data, device, optimize_for=opt_target)
        per_stage_results[opt_target] = opt_result
        best = opt_result['best']
        logger.info(f"  [{opt_target}] Best thresholds: {best['thresholds']}")
        logger.info(f"           Acc={best['accuracy']:.4f}, Avg Tokens={best['avg_tokens']:.1f}, "
                   f"Dist={best['token_distribution']}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Binary Accuracy: {best_val_acc:.4f}")

    logger.info(f"\n{'Threshold':<10} {'Accuracy':<15} {'Avg Tokens':<15}")
    logger.info("-" * 50)
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        result = cascade_results[threshold]
        logger.info(f"{threshold:<10} {result['accuracy']:.4f}          {result['avg_tokens']:<15.1f}")

    return {
        'best_val_acc': float(best_val_acc),
        'stage_accs': stage_accs,
        'cascade_results': cascade_results,
        'per_stage_thresholds': per_stage_results,
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
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
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
    logger.info(f"\n{'Threshold':<10} {'Accuracy':<20} {'Avg Tokens':<15}")
    logger.info("-" * 50)
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        accs = [r['cascade_results'][threshold]['accuracy'] for r in fold_results]
        tokens = [r['cascade_results'][threshold]['avg_tokens'] for r in fold_results]
        logger.info(f"{threshold:<10} {np.mean(accs):.4f}±{np.std(accs):.4f}     {np.mean(tokens):<15.1f}")

    return {
        'fold_results': fold_results,
        'mean_acc': float(mean_acc),
        'std_acc': float(std_acc),
    }


def main():
    parser = argparse.ArgumentParser(description="Train cascaded hidden state classifier")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with hidden states .pt files")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "attention", "xgboost"],
                        help="Model type: mlp (default), attention, or xgboost")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true",
                        help="Use simple 80:20 train/val split instead of cross-validation")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation ratio when using --no-cv (default: 0.2)")
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

    # Run training
    if args.model == "xgboost":
        logger.info("Using XGBoost (per-stage classifiers)")
        results = run_xgboost_single_split(
            all_hidden,
            all_labels,
            all_stages,
            data,
            val_ratio=args.val_ratio,
        )
    elif args.model in ["mlp", "attention"] and args.no_cv:
        results = run_single_split(
            all_hidden,
            all_labels,
            all_stages,
            data,
            val_ratio=args.val_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            model_type=args.model,
        )
    elif args.model in ["mlp", "attention"]:
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
