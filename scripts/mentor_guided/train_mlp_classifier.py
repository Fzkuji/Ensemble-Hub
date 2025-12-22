#!/usr/bin/env python3
"""
MLP classifier using frozen LLM hidden states.

Comparison baseline for LoRA fine-tuning:
1. Load base model (frozen, no LoRA)
2. Extract last token's hidden state
3. Train only MLP classification head

Usage:
    # Single GPU
    python train_mlp_classifier.py --subset algebra

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 train_mlp_classifier.py --ddp --subset algebra
"""

import argparse
import json
import logging
import os
from itertools import product
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to hold memory lock tensors (prevent garbage collection)
_memory_lock_tensors = {}


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


TOKEN_LEVELS = [0, 100, 500, 1000]

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MLPClassifierHead(nn.Module):
    """MLP classification head (same architecture as LoRA version)."""

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
        stage_embed = self.stage_embedding(stage)
        x = torch.cat([hidden_state, stage_embed], dim=-1)
        return self.classifier(x)


class MentorDataset(Dataset):
    """Dataset for mentor classification."""

    def __init__(
        self,
        data: Dict[int, List[Dict]],
        tokenizer,
        max_length: int = 2048,
        filter_uniform: bool = True,
        verbose: bool = True,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        n_samples = len(data[TOKEN_LEVELS[0]])

        if filter_uniform:
            varied_indices = []
            all_correct_count = 0
            all_wrong_count = 0

            for i in range(n_samples):
                labels = [1 if data[tokens][i].get('is_correct', False) else 0
                          for tokens in TOKEN_LEVELS if tokens in data]
                if all(l == 1 for l in labels):
                    all_correct_count += 1
                elif all(l == 0 for l in labels):
                    all_wrong_count += 1
                else:
                    varied_indices.append(i)

            if verbose:
                logger.info(f"Filtering: {all_correct_count} all-correct, {all_wrong_count} all-wrong, "
                           f"{len(varied_indices)} varied (kept)")
            indices_to_use = varied_indices
        else:
            indices_to_use = list(range(n_samples))

        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            if tokens not in data:
                continue
            for i in indices_to_use:
                item = data[tokens][i]
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
    """Load JSON data for all token levels."""
    data = {}
    split_dir = os.path.join(data_dir, split)

    if not os.path.exists(split_dir):
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


class FrozenLLMClassifier(nn.Module):
    """Frozen LLM + trainable MLP classifier."""

    def __init__(self, base_model, classifier_head, pooling_mode="last"):
        super().__init__()
        self.base_model = base_model
        self.classifier_head = classifier_head
        self.pooling_mode = pooling_mode

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, stages):
        # Get hidden states (no gradient for base model)
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]

        if self.pooling_mode == "mean_logits":
            # Per-token classification, then average logits (no for loop)
            batch_size, seq_len, hidden_size = hidden_states.shape

            # Expand stage embedding to all tokens: [batch] -> [batch, seq]
            stages_expanded = stages.unsqueeze(1).expand(-1, seq_len)
            stage_embed = self.classifier_head.stage_embedding(stages_expanded)  # [batch, seq, 64]

            # Concatenate hidden states with stage embedding
            combined = torch.cat([hidden_states.detach(), stage_embed], dim=-1)  # [batch, seq, hidden+64]

            # Reshape to [batch*seq, hidden+64], pass through classifier
            combined_flat = combined.view(batch_size * seq_len, -1)
            logits_flat = self.classifier_head.classifier(combined_flat)  # [batch*seq, 2]
            logits_all = logits_flat.view(batch_size, seq_len, -1)  # [batch, seq, 2]

            # Masked mean over sequence
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            logits_sum = (logits_all * mask).sum(dim=1)  # [batch, 2]
            seq_lens = mask.sum(dim=1)  # [batch, 1]
            logits = logits_sum / seq_lens  # [batch, 2]

        elif self.pooling_mode == "mean":
            # Mean pooling over all valid tokens, then classify once
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            sum_hidden = (hidden_states * mask).sum(dim=1)  # [batch, hidden]
            seq_lens = mask.sum(dim=1)  # [batch, 1]
            pooled = sum_hidden / seq_lens  # [batch, hidden]
            logits = self.classifier_head(pooled.detach(), stages)
        else:
            # Last token pooling (default)
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_indices, seq_lens]  # [batch, hidden]
            logits = self.classifier_head(pooled.detach(), stages)

        return logits


def train_epoch(model, dataloader, optimizer, criterion, device, grad_accum_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", disable=not is_main_process())
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
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process()):
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


def eval_cascade_on_val(
    model,
    val_data: Dict[int, List[Dict]],
    tokenizer,
    max_length: int,
    device: str,
) -> Tuple[float, List[float], Dict]:
    """Evaluate cascade accuracy on validation set with threshold search."""
    model.eval()
    n_samples = len(val_data[TOKEN_LEVELS[0]])

    all_probs = {tokens: [] for tokens in TOKEN_LEVELS}
    gt = {tokens: [] for tokens in TOKEN_LEVELS}

    for i in tqdm(range(n_samples), desc="Cascade eval", disable=not is_main_process()):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = val_data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            if mentor_response:
                text = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
            else:
                text = f"Question: {question}\n\nAnswer:"

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            stages = torch.tensor([stage_idx], device=device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask, stages)
                prob = torch.softmax(logits, dim=1)[0, 1].item()

            all_probs[tokens].append(prob)
            gt[tokens].append(1 if item.get('is_correct', False) else 0)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    def compute_cascade_acc(thresholds, all_probs, gt, n_samples):
        correct = 0
        for i in range(n_samples):
            decided = False
            stage_probs = []
            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                prob = all_probs[tokens][i]
                stage_probs.append((tokens, prob))
                if prob >= thresholds[stage_idx]:
                    correct += gt[tokens][i]
                    decided = True
                    break
            if not decided:
                best_tokens, _ = max(stage_probs, key=lambda x: x[1])
                correct += gt[best_tokens][i]
        return correct / n_samples

    threshold_candidates = [round(0.05 + i * 0.05, 2) for i in range(19)]
    best_acc = 0
    best_thresholds = None

    for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
        thresholds = list(combo)
        acc = compute_cascade_acc(thresholds, all_probs, gt, n_samples)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    oracle_correct = 0
    for i in range(n_samples):
        for tokens in TOKEN_LEVELS:
            if gt[tokens][i] == 1:
                oracle_correct += 1
                break
    oracle_acc = oracle_correct / n_samples

    stage_auc = {}
    for tokens in TOKEN_LEVELS:
        try:
            auc = roc_auc_score(gt[tokens], all_probs[tokens])
            stage_auc[tokens] = auc
        except ValueError:
            stage_auc[tokens] = 0.5

    detailed = {
        'oracle': oracle_acc,
        'baseline': {tokens: sum(gt[tokens]) / n_samples for tokens in TOKEN_LEVELS},
        'auc': stage_auc,
    }

    return best_acc, best_thresholds, detailed


def main():
    parser = argparse.ArgumentParser(description="MLP classifier with frozen LLM")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Base directory with subset folders")
    parser.add_argument("--subset", type=str, default="algebra",
                        choices=SUBSETS + ["all"],
                        help="Which subset to train on")
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save model (default: data_dir/{subset}/mlp_model)")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10,
                        help="More epochs since only training MLP")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Can use larger batch since no gradient for LLM")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Higher LR for MLP-only training")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Use 4-bit quantization for memory efficiency")
    parser.add_argument("--ddp", action="store_true",
                        help="Use DistributedDataParallel")
    parser.add_argument("--no-filter", action="store_true",
                        help="Don't filter out all-correct/all-wrong samples")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--reserve-memory", type=float, default=0,
                        help="Pre-allocate GPU memory in GB to prevent others from using it (released after model load)")
    parser.add_argument("--memory-lock", type=float, default=0,
                        help="Lock GPU memory at this fraction (0.0-1.0, e.g., 0.9 for 90%%). Keeps memory occupied throughout training.")
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean", "mean_logits"],
                        help="Pooling strategy for hidden states: last (last token) or mean (mean of all tokens)")

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    use_ddp = args.ddp or world_size > 1

    if use_ddp:
        device = f"cuda:{local_rank}"
        device_id = local_rank
    else:
        device = args.device
        device_id = int(device.split(':')[-1]) if ':' in device else 0

    # Lock GPU memory IMMEDIATELY at script startup (right after setup, before any other operations)
    # This prevents other processes from grabbing memory while we're still initializing
    if args.memory_lock > 0:
        total_mem = torch.cuda.get_device_properties(device_id).total_memory
        target_bytes = int(total_mem * args.memory_lock)
        free_mem = torch.cuda.mem_get_info(device_id)[0]
        buffer = 2 * 1024**3
        max_allocatable = free_mem - buffer
        fill_bytes = min(target_bytes, max_allocatable)
        
        if fill_bytes > 0:
            fill_elements = fill_bytes // 4  # float32 = 4 bytes
            lock_tensor = torch.empty(fill_elements, dtype=torch.float32, device=device)
            del lock_tensor  # Keep memory in cache, don't call empty_cache()
            _memory_lock_tensors[device_id] = True
            
            total_gb = total_mem / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device_id) / 1024**3
            if is_main_process():
                logger.info(f"[SCRIPT START] Memory locked immediately: {reserved_gb:.1f} GB reserved / {total_gb:.1f} GB total (target: {args.memory_lock*100:.0f}%)")
                logger.info(f"  Locked {fill_bytes/1024**3:.2f} GB in PyTorch cache (prevents other processes, reusable by training)")
        else:
            if is_main_process():
                logger.warning(f"Not enough free memory to lock at startup (free: {free_mem/1024**3:.1f} GB)")

    # Pre-allocate GPU memory if requested (will be released after model loading)
    _reserved_memory = None
    if args.reserve_memory > 0:
        reserve_bytes = int(args.reserve_memory * 1024**3)
        reserve_elements = reserve_bytes // 4  # float32 = 4 bytes
        _reserved_memory = torch.empty(reserve_elements, dtype=torch.float32, device=device)
        if is_main_process():
            logger.info(f"Pre-allocated {args.reserve_memory:.1f} GB GPU memory on {device} (will release after model load)")

    if args.subset == "all":
        subset_dir = os.path.join(args.data_dir, "all")
    else:
        subset_dir = os.path.join(args.data_dir, args.subset)

    if not os.path.exists(subset_dir):
        if is_main_process():
            logger.error(f"Data directory not found: {subset_dir}")
        cleanup_distributed()
        return

    if args.output_dir is None:
        args.output_dir = os.path.join(subset_dir, "mlp_model")
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process():
        logger.info(f"=== MLP Classifier (Frozen LLM) ===")
        logger.info(f"Subset: {args.subset}")
        logger.info(f"Data dir: {subset_dir}")
        logger.info(f"DDP: {use_ddp}, World size: {world_size}")

    # Load train data
    if is_main_process():
        logger.info("Loading training data...")
    train_data = load_json_data(subset_dir, split="train")
    if not train_data:
        if is_main_process():
            logger.error("No training data found!")
        cleanup_distributed()
        return

    # Split train/val
    from sklearn.model_selection import train_test_split as sk_split
    n_samples = len(train_data[TOKEN_LEVELS[0]])
    train_idx, val_idx = sk_split(
        np.arange(n_samples), test_size=args.val_ratio, random_state=42
    )

    val_data = {}
    actual_train_data = {}
    for tokens in TOKEN_LEVELS:
        if tokens in train_data:
            val_data[tokens] = [train_data[tokens][i] for i in val_idx]
            actual_train_data[tokens] = [train_data[tokens][i] for i in train_idx]
    train_data = actual_train_data

    n_train = len(train_data[TOKEN_LEVELS[0]])
    n_val = len(val_data[TOKEN_LEVELS[0]])
    if is_main_process():
        logger.info(f"Train: {n_train}, Val: {n_val}")

    # Load tokenizer
    if is_main_process():
        logger.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (frozen)
    if is_main_process():
        logger.info(f"Loading model (frozen) from {args.model_path}...")
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
            torch_dtype=torch.bfloat16,
        ).to(device)

    # Freeze base model
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    if is_main_process():
        total_params = sum(p.numel() for p in base_model.parameters())
        logger.info(f"Base model: {total_params/1e9:.2f}B params (all frozen)")

    # Create MLP classifier head
    hidden_size = base_model.config.hidden_size
    classifier_head = MLPClassifierHead(hidden_size, dropout=args.dropout).to(device)

    if is_main_process():
        mlp_params = sum(p.numel() for p in classifier_head.parameters())
        logger.info(f"MLP classifier: {mlp_params:,} params (trainable)")
        logger.info(f"Using pooling mode: {args.pooling}")

    # Combine
    model = FrozenLLMClassifier(base_model, classifier_head, pooling_mode=args.pooling)

    # Wrap classifier with DDP if needed
    if use_ddp:
        classifier_head = DDP(classifier_head, device_ids=[local_rank])
        model = FrozenLLMClassifier(base_model, classifier_head, pooling_mode=args.pooling)

    # Release pre-allocated memory now that model is loaded
    if _reserved_memory is not None:
        del _reserved_memory
        torch.cuda.empty_cache()
        if is_main_process():
            logger.info("Released pre-allocated GPU memory")

    # After model loading: if training uses more memory than locked, lock the additional memory too
    # This ensures all memory used by training is locked and won't be released
    if args.memory_lock > 0:
        device_id = local_rank if use_ddp else int(device.split(':')[-1]) if ':' in device else 0
        total_mem = torch.cuda.get_device_properties(device_id).total_memory
        initial_target_bytes = int(total_mem * args.memory_lock)
        
        # Check actual reserved memory after model loading
        current_reserved = torch.cuda.memory_reserved(device_id)
        free_mem = torch.cuda.mem_get_info(device_id)[0]
        
        # If training uses more than initial lock, lock the additional memory too
        if current_reserved > initial_target_bytes:
            # Calculate how much additional memory to lock
            additional_needed = current_reserved - initial_target_bytes
            # Lock up to available free memory (leave 500MB buffer)
            buffer = 500 * 1024 * 1024
            max_additional = free_mem - buffer
            additional_lock_bytes = min(additional_needed, max_additional)
            
            if additional_lock_bytes > 0:
                additional_elements = additional_lock_bytes // 4  # float32 = 4 bytes
                additional_tensor = torch.empty(additional_elements, dtype=torch.float32, device=device)
                del additional_tensor  # Keep in cache, don't call empty_cache()
                
                if is_main_process():
                    final_reserved = torch.cuda.memory_reserved(device_id)
                    total_gb = total_mem / 1024**3
                    logger.info(f"[AFTER MODEL LOAD] Training uses more memory than initial lock")
                    logger.info(f"  Initial lock: {initial_target_bytes/1024**3:.2f} GB, Actual used: {current_reserved/1024**3:.2f} GB")
                    logger.info(f"  Additional {additional_lock_bytes/1024**3:.2f} GB locked to prevent release")
                    logger.info(f"  Total locked: {final_reserved/1024**3:.2f} GB / {total_gb:.1f} GB (all memory will be kept)")
        
        if is_main_process():
            allocated_gb = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device_id) / 1024**3
            total_gb = total_mem / 1024**3
            logger.info(f"[AFTER MODEL LOAD] Memory status: {allocated_gb:.1f} GB allocated, {reserved_gb:.1f} GB reserved / {total_gb:.1f} GB total")
            if device_id in _memory_lock_tensors:
                logger.info(f"  All used memory is locked and will not be released")

    # Create datasets (filter train only, keep val unfiltered for consistent evaluation)
    filter_uniform = not args.no_filter
    verbose = is_main_process()
    train_dataset = MentorDataset(train_data, tokenizer, args.max_length, filter_uniform=filter_uniform, verbose=verbose)
    val_dataset = MentorDataset(val_data, tokenizer, args.max_length, filter_uniform=False, verbose=verbose)

    if verbose:
        logger.info(f"Training: {len(train_dataset)} samples (filtered={filter_uniform})")
        logger.info(f"Validation: {len(val_dataset)} samples (unfiltered)")

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Class weights
    train_labels = torch.tensor([s['label'] for s in train_dataset.samples])
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer (only MLP params)
    classifier_params = classifier_head.module.parameters() if use_ddp else classifier_head.parameters()
    optimizer = torch.optim.AdamW(classifier_params, lr=args.lr, weight_decay=0.01)

    # Training
    best_cascade_acc = 0
    best_thresholds = None
    best_state = None

    for epoch in range(args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_accum
        )
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion, device)

        if is_main_process():
            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Cascade evaluation
        if is_main_process():
            logger.info("Running cascade evaluation...")
        cascade_acc, thresholds, detailed = eval_cascade_on_val(
            model, val_data, tokenizer, args.max_length, device
        )

        if is_main_process():
            logger.info(f"Cascade Acc: {cascade_acc:.4f} (Oracle: {detailed['oracle']:.4f})")
            logger.info(f"Thresholds: {thresholds}")
            auc_str = ", ".join([f"T{t}={detailed['auc'][t]:.4f}" for t in TOKEN_LEVELS])
            logger.info(f"AUC: {auc_str}")

        if cascade_acc > best_cascade_acc:
            best_cascade_acc = cascade_acc
            best_thresholds = thresholds
            if is_main_process():
                classifier_state = classifier_head.module.state_dict() if use_ddp else classifier_head.state_dict()
                best_state = {
                    'classifier': classifier_state,
                    'thresholds': thresholds,
                }
                logger.info(f"New best! Saving...")

    # Save best model
    if is_main_process() and best_state:
        torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))
        logger.info(f"Best model saved to {args.output_dir}")

    # Save last model
    if is_main_process():
        classifier_state = classifier_head.module.state_dict() if use_ddp else classifier_head.state_dict()
        torch.save({'classifier': classifier_state}, os.path.join(args.output_dir, "last_model.pt"))

    # Final cascade evaluation on val (unfiltered) for consistent comparison
    if is_main_process():
        logger.info("\nFinal cascade evaluation on val (unfiltered)...")

    final_cascade_acc, final_thresholds, final_detailed = eval_cascade_on_val(
        model, val_data, tokenizer, args.max_length, device
    )
    if is_main_process():
        logger.info(f"Val Cascade Acc: {final_cascade_acc:.4f} (Oracle: {final_detailed['oracle']:.4f})")

    if is_main_process():
        logger.info("\n=== Final Results ===")
        logger.info(f"Val Cascade Accuracy: {final_cascade_acc:.4f}")
        logger.info(f"Thresholds: {final_thresholds}")

        results = {
            'method': 'mlp_frozen',
            'subset': args.subset,
            'n_train': n_train,
            'n_val': n_val,
            'best_cascade_acc': float(final_cascade_acc),
            'best_thresholds': final_thresholds,
            'oracle_acc': final_detailed['oracle'],
            'per_stage_auc': final_detailed['auc'],
            'per_stage_baseline_acc': final_detailed['baseline'],
            'args': vars(args),
        }
        with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_dir}/results.json")

    cleanup_distributed()


if __name__ == "__main__":
    main()
