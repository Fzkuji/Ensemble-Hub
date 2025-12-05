#!/usr/bin/env python3
"""
Sequence Classifier for ACT-E Framework

All classifiers directly process variable-length PPL/Entropy sequences (no manual feature engineering).

Supports:
- LSTM: Recurrent model with pack_padded_sequence for variable length
- GRU: Similar to LSTM but with gated recurrent units
- 1D-CNN: Convolutional model with global max pooling
- MLP: Uses global pooling (mean/max/min) to handle variable length
- Attention: Small Transformer with positional encoding and mean pooling

Input format for all models: (batch, seq_len, 2) where 2 = [PPL, Entropy]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Dataset for PPL/Entropy sequences with variable length."""

    def __init__(self, sequences: List[Dict], num_classes: int = 4):
        """
        Args:
            sequences: List of dicts with 'ppl', 'entropy', 'label' keys
                       ppl/entropy are lists of floats (variable length)
                       label is int (0=solo, 1=100t, 2=500t, 3=1000t)
            num_classes: Number of output classes
        """
        self.samples = []
        self.num_classes = num_classes

        for seq in sequences:
            ppl = np.array(seq['ppl'], dtype=np.float32)
            entropy = np.array(seq['entropy'], dtype=np.float32)
            label = seq['label']

            # Stack PPL and entropy as 2-channel input
            # Shape: (seq_len, 2)
            features = np.stack([ppl, entropy], axis=1)
            self.samples.append({
                'features': features,
                'label': label,
                'length': len(ppl)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'features': torch.tensor(sample['features'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'length': sample['length']
        }


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    features = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])

    # Pad sequences to max length in batch
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        'features': features_padded,
        'labels': labels,
        'lengths': lengths
    }


class LSTMClassifier(nn.Module):
    """LSTM-based sequence classifier."""

    def __init__(
        self,
        input_dim: int = 2,  # PPL + Entropy
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_seq_len, input_dim)
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # Sort by length for pack_padded_sequence
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        x_sorted = x[sort_idx]

        # Pack padded sequence
        packed = pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )

        # LSTM forward
        packed_out, (h_n, c_n) = self.lstm(packed)

        # Get last hidden state
        if self.bidirectional:
            # Concatenate forward and backward last hidden states
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]

        # Unsort to original order
        _, unsort_idx = sort_idx.sort()
        h_last = h_last[unsort_idx]

        # Classification
        out = self.dropout(h_last)
        logits = self.fc(out)

        return logits


class GRUClassifier(nn.Module):
    """GRU-based sequence classifier."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        lengths_sorted, sort_idx = lengths.sort(descending=True)
        x_sorted = x[sort_idx]

        packed = pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )

        packed_out, h_n = self.gru(packed)

        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]

        _, unsort_idx = sort_idx.sort()
        h_last = h_last[unsort_idx]

        out = self.dropout(h_last)
        logits = self.fc(out)

        return logits


class CNN1DClassifier(nn.Module):
    """1D CNN classifier with global max pooling."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_seq_len, input_dim)
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, num_classes)
        """
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # CNN layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (batch, hidden_dim)

        # Classification
        x = self.dropout(x)
        logits = self.fc(x)

        return logits


class MLPClassifier(nn.Module):
    """MLP using global pooling to handle variable-length sequences.

    Uses mean/max pooling over the sequence dimension to get fixed-size features,
    then applies MLP for classification.
    """

    def __init__(
        self,
        input_dim: int = 2,  # PPL + Entropy
        hidden_dims: List[int] = [64, 32],
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # After pooling: mean(2) + max(2) + min(2) = 6 features
        pooled_dim = input_dim * 3

        layers = []
        prev_dim = pooled_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_seq_len, 2) PPL and Entropy sequences
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, num_classes)
        """
        batch_size, max_len, input_dim = x.shape

        # Create mask for valid positions
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (batch, max_len, 1)

        # Masked mean pooling
        x_masked = x * mask
        mean_pool = x_masked.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)

        # Masked max pooling (set invalid positions to -inf)
        x_for_max = x.clone()
        x_for_max[~mask.bool().expand_as(x)] = float('-inf')
        max_pool = x_for_max.max(dim=1)[0]

        # Masked min pooling (set invalid positions to +inf)
        x_for_min = x.clone()
        x_for_min[~mask.bool().expand_as(x)] = float('inf')
        min_pool = x_for_min.min(dim=1)[0]

        # Concatenate: (batch, 6)
        features = torch.cat([mean_pool, max_pool, min_pool], dim=1)

        return self.mlp(features)


class AttentionClassifier(nn.Module):
    """Small self-attention based classifier for variable-length sequences.

    Uses positional encoding + multi-head self-attention + mean pooling.
    """

    def __init__(
        self,
        input_dim: int = 2,  # PPL + Entropy
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 2000,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_seq_len, input_dim)
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, num_classes)
        """
        batch_size, max_len, _ = x.shape

        # Project input to hidden dimension
        x = self.input_proj(x)  # (batch, max_len, hidden_dim)

        # Add positional encoding
        x = x + self.pos_encoding[:, :max_len, :]

        # Create attention mask for padding
        # True = masked (ignored), False = not masked
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling over valid positions
        # Create a mask for valid positions: (batch, max_len, 1)
        valid_mask = (~mask).unsqueeze(-1).float()
        x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        # Classification
        x = self.dropout(x)
        logits = self.fc(x)

        return logits


class ClassifierTrainer:
    """Trainer for sequence classifiers."""

    def __init__(
        self,
        model_type: str = "lstm",
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        lr: float = 0.001,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.num_classes = num_classes

        # Create model
        if model_type == "lstm":
            self.model = LSTMClassifier(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif model_type == "gru":
            self.model = GRUClassifier(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif model_type == "cnn":
            self.model = CNN1DClassifier(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif model_type == "mlp":
            self.model = MLPClassifier(
                input_dim=2,  # PPL + Entropy
                hidden_dims=[hidden_dim, hidden_dim // 2],
                num_classes=num_classes,
                dropout=dropout,
            )
        elif model_type == "attention":
            self.model = AttentionClassifier(
                input_dim=2,
                hidden_dim=hidden_dim,
                num_heads=4,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"Initialized {model_type.upper()} classifier on {self.device}")

    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        use_class_weights: bool = True,
    ) -> Dict:
        """Train the classifier."""
        train_dataset = SequenceDataset(train_data, self.num_classes)
        val_dataset = SequenceDataset(val_data, self.num_classes)

        # Calculate class weights to handle imbalanced data
        if use_class_weights:
            label_counts = {}
            for item in train_data:
                label = item['label']
                label_counts[label] = label_counts.get(label, 0) + 1

            total_samples = len(train_data)
            class_weights = []
            for i in range(self.num_classes):
                count = label_counts.get(i, 1)  # Avoid division by zero
                # Inverse frequency weighting
                weight = total_samples / (self.num_classes * count)
                class_weights.append(weight)

            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(features, lengths)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += len(labels)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Scheduler step
            self.scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if epoch % 10 == 0 or val_acc > best_val_acc:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        return history

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                logits = self.model(features, lengths)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * len(labels)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    def predict(self, ppl: List[float], entropy: List[float], threshold: float = 0.5) -> int:
        """Predict the optimal strategy for a single sample.

        For binary classification:
            - Returns 1 (sufficient) if P(class=1) >= threshold
            - Returns 0 (not sufficient) otherwise
        """
        self.model.eval()

        features = np.stack([np.array(ppl), np.array(entropy)], axis=1)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(ppl)]).to(self.device)

        with torch.no_grad():
            logits = self.model(features, lengths)
            if self.num_classes == 2:
                # Binary classification with threshold
                probs = torch.softmax(logits, dim=1)
                pred = 1 if probs[0, 1].item() >= threshold else 0
            else:
                pred = logits.argmax(dim=1).item()

        return pred

    def predict_proba(self, ppl: List[float], entropy: List[float]) -> np.ndarray:
        """Get class probabilities for a single sample.

        Returns:
            Array of shape (num_classes,) with class probabilities
        """
        self.model.eval()

        features = np.stack([np.array(ppl), np.array(entropy)], axis=1)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(ppl)]).to(self.device)

        with torch.no_grad():
            logits = self.model(features, lengths)
            probs = torch.softmax(logits, dim=1)

        return probs[0].cpu().numpy()

    def save(self, path: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def demo():
    """Demo with synthetic data."""
    logger.info("=== Sequence Classifier Demo ===")

    # Generate synthetic data
    np.random.seed(42)
    num_samples = 200

    data = []
    for i in range(num_samples):
        seq_len = np.random.randint(50, 500)
        ppl = np.random.lognormal(1.0, 0.5, seq_len).tolist()
        entropy = np.random.uniform(0.5, 3.0, seq_len).tolist()
        label = np.random.randint(0, 4)
        data.append({'ppl': ppl, 'entropy': entropy, 'label': label})

    # Split data
    train_data = data[:160]
    val_data = data[160:]

    # Train different classifiers
    for model_type in ["lstm", "gru", "cnn", "mlp", "attention"]:
        logger.info(f"\n--- Training {model_type.upper()} ---")
        trainer = ClassifierTrainer(model_type=model_type, hidden_dim=64, num_classes=4)
        history = trainer.train(train_data, val_data, epochs=50, batch_size=32)
        logger.info(f"{model_type.upper()} final val acc: {history['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    demo()
