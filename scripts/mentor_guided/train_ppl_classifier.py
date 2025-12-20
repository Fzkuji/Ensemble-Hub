#!/usr/bin/env python3
"""
PPL/Entropy-based classifier for mentor sufficiency prediction.

Method:
1. Feed mentor hint + question to intern model
2. Extract PPL and entropy statistics from model output
3. Train a simple regression model (LogisticRegression/XGBoost) on these features

Usage:
    # Single GPU
    python train_ppl_classifier.py --subset algebra --data-dir /path/to/data

    # Multi-GPU with torchrun
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_ppl_classifier.py \
        --ddp --subset algebra --data-dir /path/to/data
"""

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split as sk_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def load_json_data(data_dir: str, split: str = "train") -> Dict[int, List[Dict]]:
    """Load JSON data for all token levels."""
    data = {}
    split_dir = os.path.join(data_dir, split)

    if not os.path.exists(split_dir):
        split_dir = data_dir
        if is_main_process():
            logger.warning(f"Split dir not found, using: {split_dir}")

    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(split_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
            if is_main_process():
                logger.info(f"Loaded {len(data[tokens])} samples from {filepath}")
        else:
            if is_main_process():
                logger.warning(f"File not found: {filepath}")

    return data


def compute_trend_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute trend statistics for a sequence of values."""
    n = len(values)
    if n < 2:
        return {
            'slope': 0.0,
            'increase_ratio': 0.5,
            'decrease_ratio': 0.5,
            'last_quarter_mean': float(values[0]) if n > 0 else 0.0,
            'first_quarter_mean': float(values[0]) if n > 0 else 0.0,
            'trend_change': 0.0,
        }

    # Linear regression slope
    x = np.arange(n)
    slope = np.polyfit(x, values, 1)[0]

    # Ratio of increasing/decreasing transitions
    diffs = np.diff(values)
    increase_ratio = np.mean(diffs > 0)
    decrease_ratio = np.mean(diffs < 0)

    # Compare first and last quarters
    quarter = max(1, n // 4)
    first_quarter_mean = np.mean(values[:quarter])
    last_quarter_mean = np.mean(values[-quarter:])
    trend_change = last_quarter_mean - first_quarter_mean

    return {
        'slope': float(slope),
        'increase_ratio': float(increase_ratio),
        'decrease_ratio': float(decrease_ratio),
        'last_quarter_mean': float(last_quarter_mean),
        'first_quarter_mean': float(first_quarter_mean),
        'trend_change': float(trend_change),
    }


def compute_stats(token_logprobs: List[float], token_entropies: List[float]) -> Dict[str, float]:
    """Compute statistics from token log probabilities and entropies."""
    if not token_logprobs or len(token_logprobs) == 0:
        return {
            'ppl': 1.0,
            'log_ppl': 0.0,
            'entropy_mean': 0.0,
            'entropy_std': 0.0,
            'entropy_max': 0.0,
            'entropy_min': 0.0,
            'entropy_slope': 0.0,
            'entropy_increase_ratio': 0.5,
            'entropy_decrease_ratio': 0.5,
            'entropy_first_quarter': 0.0,
            'entropy_last_quarter': 0.0,
            'entropy_trend_change': 0.0,
            'log_prob_mean': 0.0,
            'log_prob_std': 0.0,
            'log_prob_max': 0.0,
            'log_prob_min': 0.0,
            'log_prob_slope': 0.0,
            'log_prob_increase_ratio': 0.5,
            'log_prob_decrease_ratio': 0.5,
            'log_prob_first_quarter': 0.0,
            'log_prob_last_quarter': 0.0,
            'log_prob_trend_change': 0.0,
            'seq_len': 0,
        }

    logprobs = np.array(token_logprobs)
    entropies = np.array(token_entropies)

    # PPL from mean log prob
    mean_logprob = np.mean(logprobs)
    ppl = np.exp(-mean_logprob)

    # Compute trend statistics
    logprob_trend = compute_trend_stats(logprobs)
    entropy_trend = compute_trend_stats(entropies)

    return {
        'ppl': float(ppl),
        'log_ppl': float(np.log(ppl + 1e-10)),
        # True entropy stats: H = -sum(p * log(p))
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
        'entropy_max': float(np.max(entropies)),
        'entropy_min': float(np.min(entropies)),
        # Entropy trend stats
        'entropy_slope': entropy_trend['slope'],
        'entropy_increase_ratio': entropy_trend['increase_ratio'],
        'entropy_decrease_ratio': entropy_trend['decrease_ratio'],
        'entropy_first_quarter': entropy_trend['first_quarter_mean'],
        'entropy_last_quarter': entropy_trend['last_quarter_mean'],
        'entropy_trend_change': entropy_trend['trend_change'],
        # Log prob basic stats
        'log_prob_mean': float(np.mean(logprobs)),
        'log_prob_std': float(np.std(logprobs)),
        'log_prob_max': float(np.max(logprobs)),
        'log_prob_min': float(np.min(logprobs)),
        # Log prob trend stats
        'log_prob_slope': logprob_trend['slope'],
        'log_prob_increase_ratio': logprob_trend['increase_ratio'],
        'log_prob_decrease_ratio': logprob_trend['decrease_ratio'],
        'log_prob_first_quarter': logprob_trend['first_quarter_mean'],
        'log_prob_last_quarter': logprob_trend['last_quarter_mean'],
        'log_prob_trend_change': logprob_trend['trend_change'],
        # Sequence info
        'seq_len': len(logprobs),
    }


def extract_features(
    model,
    tokenizer,
    data: Dict[int, List[Dict]],
    device: str,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract PPL/entropy features using HuggingFace model.
    """
    all_features = []
    all_labels = []
    all_stages = []

    n_samples = len(data[TOKEN_LEVELS[0]])

    for i in tqdm(range(n_samples), desc="Extracting features", disable=not is_main_process()):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            if mentor_response:
                text = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
            else:
                text = f"Question: {question}\n\nAnswer:"

            # Tokenize
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded['input_ids'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                logits = outputs.logits

                # Shift for next token prediction
                shifted_logits = logits[:, :-1, :]  # [1, seq_len-1, vocab_size]
                shifted_input_ids = input_ids[:, 1:]  # [1, seq_len-1]

                # Get log probs
                log_probs = torch.log_softmax(shifted_logits, dim=-1)

                # Token log probs (for PPL)
                token_log_probs = log_probs.gather(
                    dim=-1,
                    index=shifted_input_ids.unsqueeze(-1)
                ).squeeze(-1)
                token_logprobs = token_log_probs[0].float().cpu().numpy().tolist()

                # True entropy: H = -sum(p * log(p))
                probs = torch.softmax(shifted_logits, dim=-1)  # [1, seq_len-1, vocab_size]
                # Clamp to avoid log(0)
                log_probs_clamped = torch.log(probs + 1e-10)
                entropy = -torch.sum(probs * log_probs_clamped, dim=-1)  # [1, seq_len-1]
                token_entropies = entropy[0].float().cpu().numpy().tolist()

            stats = compute_stats(token_logprobs, token_entropies)

            features = [
                stats['ppl'],
                stats['log_ppl'],
                stats['entropy_mean'],
                stats['entropy_std'],
                stats['entropy_max'],
                stats['entropy_min'],
                stats['entropy_slope'],
                stats['entropy_increase_ratio'],
                stats['entropy_decrease_ratio'],
                stats['entropy_first_quarter'],
                stats['entropy_last_quarter'],
                stats['entropy_trend_change'],
                stats['log_prob_mean'],
                stats['log_prob_std'],
                stats['log_prob_max'],
                stats['log_prob_min'],
                stats['log_prob_slope'],
                stats['log_prob_increase_ratio'],
                stats['log_prob_decrease_ratio'],
                stats['log_prob_first_quarter'],
                stats['log_prob_last_quarter'],
                stats['log_prob_trend_change'],
                stats['seq_len'],
                stage_idx,
                tokens,
            ]

            all_features.append(features)
            all_labels.append(1 if item.get('is_correct', False) else 0)
            all_stages.append(stage_idx)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    return np.array(all_features), np.array(all_labels), np.array(all_stages)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "gb",
) -> Tuple[object, StandardScaler, Dict]:
    """Train a classifier on PPL/entropy features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if model_type == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

    clf.fit(X_train_scaled, y_train)

    train_pred = clf.predict(X_train_scaled)
    val_pred = clf.predict(X_val_scaled)
    train_proba = clf.predict_proba(X_train_scaled)[:, 1]
    val_proba = clf.predict_proba(X_val_scaled)[:, 1]

    results = {
        'train_acc': accuracy_score(y_train, train_pred),
        'val_acc': accuracy_score(y_val, val_pred),
        'train_auc': roc_auc_score(y_train, train_proba),
        'val_auc': roc_auc_score(y_val, val_proba),
    }

    return clf, scaler, results


def eval_cascade(
    clf,
    scaler: StandardScaler,
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, List[float], Dict]:
    """Evaluate cascade accuracy with threshold search."""
    from itertools import product

    n_stages = len(TOKEN_LEVELS)
    n_samples = len(features) // n_stages

    features_scaled = scaler.transform(features)
    probs = clf.predict_proba(features_scaled)[:, 1]

    probs = probs.reshape(n_samples, n_stages)
    gt = labels.reshape(n_samples, n_stages)

    def compute_cascade_acc(thresholds):
        correct = 0
        for i in range(n_samples):
            decided = False
            stage_probs = []
            for stage_idx in range(n_stages):
                prob = probs[i, stage_idx]
                stage_probs.append((stage_idx, prob))
                if prob >= thresholds[stage_idx]:
                    correct += gt[i, stage_idx]
                    decided = True
                    break
            if not decided:
                best_stage, _ = max(stage_probs, key=lambda x: x[1])
                correct += gt[i, best_stage]
        return correct / n_samples

    threshold_candidates = [round(i * 0.05, 2) for i in range(21)]
    best_acc = 0
    best_thresholds = None

    for combo in product(threshold_candidates, repeat=n_stages):
        thresholds = list(combo)
        acc = compute_cascade_acc(thresholds)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    oracle_correct = sum(1 for i in range(n_samples) if any(gt[i, :] == 1))
    oracle_acc = oracle_correct / n_samples

    stage_acc = {}
    stage_auc = {}
    for stage_idx, tokens in enumerate(TOKEN_LEVELS):
        stage_labels = gt[:, stage_idx]
        stage_probs_flat = probs[:, stage_idx]
        stage_acc[tokens] = np.mean(stage_labels)
        try:
            stage_auc[tokens] = roc_auc_score(stage_labels, stage_probs_flat)
        except ValueError:
            stage_auc[tokens] = 0.5

    detailed = {
        'oracle': oracle_acc,
        'baseline': stage_acc,
        'auc': stage_auc,
    }

    return best_acc, best_thresholds, detailed


def main():
    parser = argparse.ArgumentParser(description="PPL/Entropy-based classifier")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Base directory with subset folders")
    parser.add_argument("--subset", type=str, default="algebra",
                        choices=SUBSETS + ["all"])
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--classifier", type=str, default="gb",
                        choices=["lr", "gb"])
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--ddp", action="store_true",
                        help="Use DDP mode (with torchrun)")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reserve-memory", type=float, default=0,
                        help="Pre-allocate GPU memory in GB to prevent others from using it (released after model load)")
    parser.add_argument("--memory-lock", type=float, default=0,
                        help="Lock GPU memory at this fraction (0.0-1.0, e.g., 0.9 for 90%%). Keeps memory occupied throughout training.")

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    use_ddp = args.ddp or world_size > 1

    if use_ddp:
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    # Pre-allocate GPU memory if requested (will be released after model loading)
    _reserved_memory = None
    if args.reserve_memory > 0:
        reserve_bytes = int(args.reserve_memory * 1024**3)
        reserve_elements = reserve_bytes // 4  # float32 = 4 bytes
        _reserved_memory = torch.empty(reserve_elements, dtype=torch.float32, device=device)
        if is_main_process():
            logger.info(f"Pre-allocated {args.reserve_memory:.1f} GB GPU memory on {device} (will release after model load)")

    subset_dir = os.path.join(args.data_dir, args.subset)
    if not os.path.exists(subset_dir):
        if is_main_process():
            logger.error(f"Data directory not found: {subset_dir}")
        cleanup_distributed()
        return

    if args.output_dir is None:
        args.output_dir = os.path.join(subset_dir, "ppl_model")
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process():
        logger.info(f"Subset: {args.subset}")
        logger.info(f"Data dir: {subset_dir}")
        logger.info(f"Classifier: {args.classifier}")
        logger.info(f"DDP: {use_ddp}, World size: {world_size}")

    # Load model
    if is_main_process():
        logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Release pre-allocated memory now that model is loaded
    if _reserved_memory is not None:
        del _reserved_memory
        torch.cuda.empty_cache()
        if is_main_process():
            logger.info("Released pre-allocated GPU memory")

    # Lock GPU memory at target fraction (prevents other processes from using it)
    # This allocates a filler tensor to occupy remaining memory up to target fraction
    _memory_lock_tensor = None
    if args.memory_lock > 0:
        device_id = local_rank if use_ddp else int(device.split(':')[-1]) if ':' in device else 0
        total_mem = torch.cuda.get_device_properties(device_id).total_memory
        target_bytes = int(total_mem * args.memory_lock)
        current_allocated = torch.cuda.memory_allocated(device_id)
        # Leave some buffer (500MB) for temporary allocations during inference
        buffer_bytes = 500 * 1024 * 1024
        fill_bytes = target_bytes - current_allocated - buffer_bytes
        if fill_bytes > 0:
            fill_elements = fill_bytes // 4  # float32 = 4 bytes
            _memory_lock_tensor = torch.empty(fill_elements, dtype=torch.float32, device=device)
            if is_main_process():
                locked_gb = (current_allocated + fill_bytes) / 1024**3
                total_gb = total_mem / 1024**3
                logger.info(f"Memory locked: {locked_gb:.1f} GB / {total_gb:.1f} GB ({args.memory_lock*100:.0f}% target) on {device}")
        else:
            if is_main_process():
                logger.info(f"Model already uses {current_allocated/1024**3:.1f} GB, no additional locking needed")

    # Load data
    train_data = load_json_data(subset_dir, split="train")
    if not train_data:
        if is_main_process():
            logger.error("No training data found!")
        cleanup_distributed()
        return

    # Split train/val
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

    if is_main_process():
        logger.info(f"Train: {len(train_data[TOKEN_LEVELS[0]])} samples")
        logger.info(f"Val: {len(val_data[TOKEN_LEVELS[0]])} samples")

    # Filter uniform samples (only filter train_data, keep val_data unfiltered for evaluation)
    if not args.no_filter:
        def filter_varied(data):
            n = len(data[TOKEN_LEVELS[0]])
            varied_indices = []
            for i in range(n):
                labels = [1 if data[tokens][i].get('is_correct', False) else 0
                          for tokens in TOKEN_LEVELS]
                if not (all(l == 1 for l in labels) or all(l == 0 for l in labels)):
                    varied_indices.append(i)
            filtered = {}
            for tokens in TOKEN_LEVELS:
                filtered[tokens] = [data[tokens][i] for i in varied_indices]
            return filtered

        train_data = filter_varied(train_data)
        # val_data stays unfiltered for consistent evaluation
        if is_main_process():
            logger.info(f"After filtering train: {len(train_data[TOKEN_LEVELS[0]])} samples (val unfiltered: {len(val_data[TOKEN_LEVELS[0]])})")

    # For DDP, shard data across processes
    if use_ddp:
        n_train = len(train_data[TOKEN_LEVELS[0]])
        n_val = len(val_data[TOKEN_LEVELS[0]])

        train_shard_size = (n_train + world_size - 1) // world_size
        val_shard_size = (n_val + world_size - 1) // world_size

        train_start = rank * train_shard_size
        train_end = min(train_start + train_shard_size, n_train)
        val_start = rank * val_shard_size
        val_end = min(val_start + val_shard_size, n_val)

        train_shard = {}
        val_shard = {}
        for tokens in TOKEN_LEVELS:
            train_shard[tokens] = train_data[tokens][train_start:train_end]
            val_shard[tokens] = val_data[tokens][val_start:val_end]

        if is_main_process():
            logger.info(f"Rank {rank}: Train shard [{train_start}:{train_end}], Val shard [{val_start}:{val_end}]")
    else:
        train_shard = train_data
        val_shard = val_data

    # Extract features
    if is_main_process():
        logger.info("Extracting features...")
    X_train, y_train, stages_train = extract_features(
        model, tokenizer, train_shard, device, args.max_length
    )
    X_val, y_val, stages_val = extract_features(
        model, tokenizer, val_shard, device, args.max_length
    )

    # Gather results from all ranks
    if use_ddp:
        X_train_list = [None] * world_size
        y_train_list = [None] * world_size
        X_val_list = [None] * world_size
        y_val_list = [None] * world_size

        dist.all_gather_object(X_train_list, X_train)
        dist.all_gather_object(y_train_list, y_train)
        dist.all_gather_object(X_val_list, X_val)
        dist.all_gather_object(y_val_list, y_val)

        # Filter out empty arrays (some ranks may have no data)
        X_train_list = [x for x in X_train_list if len(x) > 0]
        y_train_list = [y for y in y_train_list if len(y) > 0]
        X_val_list = [x for x in X_val_list if len(x) > 0]
        y_val_list = [y for y in y_val_list if len(y) > 0]

        X_train = np.vstack(X_train_list) if X_train_list else np.array([])
        y_train = np.concatenate(y_train_list) if y_train_list else np.array([])
        X_val = np.vstack(X_val_list) if X_val_list else np.array([])
        y_val = np.concatenate(y_val_list) if y_val_list else np.array([])

    if is_main_process():
        logger.info(f"Train features shape: {X_train.shape}")
        logger.info(f"Val features shape: {X_val.shape}")

        # Train classifier
        logger.info(f"Training {args.classifier} classifier...")
        clf, scaler, train_results = train_classifier(
            X_train, y_train, X_val, y_val, args.classifier
        )

        logger.info(f"Train Acc: {train_results['train_acc']:.4f}, Train AUC: {train_results['train_auc']:.4f}")
        logger.info(f"Val Acc: {train_results['val_acc']:.4f}, Val AUC: {train_results['val_auc']:.4f}")

        # Cascade evaluation on val (unfiltered) for consistent comparison
        logger.info("Running cascade evaluation on val (unfiltered)...")

        cascade_acc, thresholds, detailed = eval_cascade(clf, scaler, X_val, y_val)

        logger.info(f"Val Cascade Accuracy: {cascade_acc:.4f} (Oracle: {detailed['oracle']:.4f})")
        logger.info(f"Thresholds: {thresholds}")

        auc_str = ", ".join([f"T{t}={detailed['auc'][t]:.4f}" for t in TOKEN_LEVELS])
        logger.info(f"Per-stage AUC: {auc_str}")

        baseline_str = ", ".join([f"T{t}={detailed['baseline'][t]:.4f}" for t in TOKEN_LEVELS])
        logger.info(f"Per-stage baseline acc: {baseline_str}")

        # Save model
        model_path_out = os.path.join(args.output_dir, "classifier.pkl")
        with open(model_path_out, 'wb') as f:
            pickle.dump({'classifier': clf, 'scaler': scaler, 'thresholds': thresholds}, f)
        logger.info(f"Model saved to {model_path_out}")

        # Save results
        results = {
            'subset': args.subset,
            'classifier': args.classifier,
            'train_acc': train_results['train_acc'],
            'val_acc': train_results['val_acc'],
            'train_auc': train_results['train_auc'],
            'val_auc': train_results['val_auc'],
            'best_cascade_acc': cascade_acc,
            'best_thresholds': thresholds,
            'oracle_acc': detailed['oracle'],
            'per_stage_auc': detailed['auc'],
            'per_stage_baseline_acc': detailed['baseline'],
        }

        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
