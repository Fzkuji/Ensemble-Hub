#!/usr/bin/env python3
"""
LoRA fine-tuning for mentor sufficiency classification.

Architecture:
1. Load base model (DeepSeek-R1-Distill-Qwen-7B)
2. Add LoRA adapters
3. Add classification head on top of last token's hidden state
4. Train to predict is_correct (0/1)
"""

import argparse
import glob
import json
import logging
import os
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]

# Available subsets
SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MentorClassifierHead(nn.Module):
    """Small MLP classification head."""

    def __init__(self, hidden_size: int, num_stages: int = 4, dropout: float = 0.1):
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, 64)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, hidden_state: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_size] - last token's hidden state
            stage: [batch] - stage index (0-3)
        Returns:
            logits: [batch, 2]
        """
        stage_embed = self.stage_embedding(stage)  # [batch, 64]
        x = torch.cat([hidden_state, stage_embed], dim=-1)
        return self.classifier(x)


class MentorDataset(Dataset):
    """Dataset for mentor classification."""

    def __init__(
        self,
        data: Dict[int, List[Dict]],
        tokenizer,
        max_length: int = 2048,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Flatten all stages into samples
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            if tokens not in data:
                continue
            for item in data[tokens]:
                self.samples.append({
                    'question': item['question'],
                    'mentor_response': item.get('mentor_response', ''),
                    'label': 1 if item.get('is_correct', False) else 0,
                    'stage': stage_idx,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['question'] + sample['mentor_response']

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': sample['label'],
            'stage': sample['stage'],
        }


def collate_fn(batch, tokenizer):
    """Collate function with dynamic padding."""
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []
    stages = []

    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids.append(item['input_ids'] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append(item['attention_mask'] + [0] * pad_len)
        labels.append(item['label'])
        stages.append(item['stage'])

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'stages': torch.tensor(stages, dtype=torch.long),
    }


def load_json_data(data_dir: str, split: str = "train") -> Dict[int, List[Dict]]:
    """Load JSON data for all token levels.

    Expects structure: data_dir/{split}/tokens{0,100,500,1000}.json
    """
    data = {}
    split_dir = os.path.join(data_dir, split)

    if not os.path.exists(split_dir):
        # Fallback to data_dir directly
        split_dir = data_dir
        logger.warning(f"Split dir not found, using: {split_dir}")

    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(split_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
            logger.info(f"Loaded {len(data[tokens])} samples from {filepath}")
        else:
            logger.warning(f"File not found: {filepath}")

    return data


class LoRAClassifier(nn.Module):
    """Wrapper combining base model with LoRA and classification head."""

    def __init__(self, base_model, classifier_head):
        super().__init__()
        self.base_model = base_model
        self.classifier_head = classifier_head

    def forward(self, input_ids, attention_mask, stages):
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]

        # Get last valid token for each sample
        seq_lens = attention_mask.sum(dim=1) - 1  # [batch]
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, seq_lens]  # [batch, hidden]

        # Classify
        logits = self.classifier_head(last_hidden, stages)
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device, grad_accum_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        stages = batch['stages'].to(device)

        logits = model(input_ids, attention_mask, stages)
        loss = criterion(logits, labels)
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': total_loss / (step + 1), 'acc': correct / total})

    return total_loss / len(dataloader), correct / total


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_stages = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            stages = batch['stages'].to(device)

            logits = model(input_ids, attention_mask, stages)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_stages.extend(stages.cpu().tolist())

    return total_loss / len(dataloader), correct / total, all_preds, all_labels, all_stages


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for mentor classification")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Base directory with subset folders")
    parser.add_argument("--subset", type=str, default="algebra",
                        choices=SUBSETS + ["all"],
                        help="Which subset to train on (default: algebra)")
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save model (default: data_dir/{subset}/lora_model)")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Use 4-bit quantization")

    args = parser.parse_args()
    device = args.device

    # Determine subset directory
    if args.subset == "all":
        subset_dir = args.data_dir
        # Use all_train / all_test folders
        train_split = "all_train"
        test_split = "all_test"
    else:
        subset_dir = os.path.join(args.data_dir, args.subset)
        train_split = "train"
        test_split = "test"

    # Validate data directory exists
    if not os.path.exists(subset_dir):
        logger.error(f"Data directory not found: {subset_dir}")
        logger.error(f"Please check --data-dir and --subset arguments")
        return

    if args.output_dir is None:
        args.output_dir = os.path.join(subset_dir, "lora_model")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Subset: {args.subset}")
    logger.info(f"Data dir: {subset_dir}")

    # Load train and test data separately (already split)
    logger.info("Loading training data...")
    train_data = load_json_data(subset_dir, split=train_split)
    if not train_data:
        logger.error("No training data found!")
        return

    logger.info("Loading test data...")
    test_data = load_json_data(subset_dir, split=test_split)
    if not test_data:
        logger.warning("No test data found, will use train data for validation")
        # Fall back to train/val split
        from sklearn.model_selection import train_test_split
        n_samples = len(train_data[TOKEN_LEVELS[0]])
        train_idx, val_idx = train_test_split(
            np.arange(n_samples), test_size=0.2, random_state=42
        )
        val_data = {}
        for tokens in TOKEN_LEVELS:
            if tokens in train_data:
                val_data[tokens] = [train_data[tokens][i] for i in val_idx]
                train_data[tokens] = [train_data[tokens][i] for i in train_idx]
    else:
        val_data = test_data

    n_train = len(train_data[TOKEN_LEVELS[0]])
    n_val = len(val_data[TOKEN_LEVELS[0]])
    logger.info(f"Train: {n_train} samples, Val: {n_val} samples")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optional quantization
    logger.info(f"Loading model from {args.model_path}...")
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

    # Add LoRA
    logger.info(f"Adding LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Create classification head
    hidden_size = base_model.config.hidden_size
    classifier_head = MentorClassifierHead(hidden_size).to(device)

    # Combine into single model
    model = LoRAClassifier(base_model, classifier_head)

    # Create datasets
    train_dataset = MentorDataset(train_data, tokenizer, args.max_length)
    val_dataset = MentorDataset(val_data, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Class weights
    train_labels = torch.tensor([s['label'] for s in train_dataset.samples])
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer - only train LoRA params and classifier head
    trainable_params = [
        {'params': base_model.parameters(), 'lr': args.lr},
        {'params': classifier_head.parameters(), 'lr': args.lr * 10},  # Higher LR for head
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=0.01)

    # Training loop
    best_val_acc = 0
    best_state = None

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_accum
        )
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'lora': base_model.state_dict(),
                'classifier': classifier_head.state_dict(),
            }
            logger.info(f"New best! Saving...")

    # Save best model
    if best_state:
        torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))
        base_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")

    # Final evaluation with per-stage accuracy
    logger.info("\nFinal Evaluation:")
    logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")

    # Save results
    results = {
        'subset': args.subset,
        'n_train': n_train,
        'n_val': n_val,
        'best_val_acc': float(best_val_acc),
        'args': vars(args),
    }
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
