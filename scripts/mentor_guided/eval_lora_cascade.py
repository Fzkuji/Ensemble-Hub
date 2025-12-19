#!/usr/bin/env python3
"""
Evaluate trained LoRA classifier with cascade inference.

Usage:
    # Single GPU
    python eval_lora_cascade.py --model-dir /path/to/lora_model --subset algebra

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 eval_lora_cascade.py --subset algebra
"""

import argparse
import json
import logging
import os
from itertools import product
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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
    """Initialize distributed evaluation if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed evaluation."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


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

    def forward(self, hidden_state, stage_idx):
        stage_embed = self.stage_embedding(stage_idx)
        combined = torch.cat([hidden_state, stage_embed], dim=-1)
        return self.classifier(combined)


def load_json_data(data_dir: str, split: str = "test") -> Dict[int, List[Dict]]:
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

    return data


def compute_oracle_accuracy(data: Dict[int, List[Dict]]) -> float:
    """Compute oracle accuracy."""
    if not data:
        return 0.0

    n_samples = len(data[TOKEN_LEVELS[0]])
    oracle_correct = 0

    for i in range(n_samples):
        for tokens in TOKEN_LEVELS:
            if data[tokens][i].get('is_correct', False):
                oracle_correct += 1
                break

    return oracle_correct / n_samples


def get_hidden_state(model, tokenizer, question: str, mentor_response: str, max_length: int, device: str):
    """Get last token hidden state from model."""
    # Build prompt
    if mentor_response:
        prompt = f"Question: {question}\n\nHint: {mentor_response}\n\nAnswer:"
    else:
        prompt = f"Question: {question}\n\nAnswer:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Get last layer hidden state of the last token
        last_hidden = outputs.hidden_states[-1]
        seq_len = attention_mask.sum().item()
        hidden_state = last_hidden[0, seq_len - 1, :]

    return hidden_state


def evaluate_cascade_distributed(
    model,
    classifier_head,
    tokenizer,
    test_data: Dict[int, List[Dict]],
    max_length: int,
    device: str,
    rank: int,
    world_size: int,
    fixed_thresholds: List[float] = None,
) -> Dict:
    """Evaluate cascade accuracy with threshold search (distributed version)."""
    model.eval()
    classifier_head.eval()

    n_samples = len(test_data[TOKEN_LEVELS[0]])

    # Distribute samples across processes
    samples_per_rank = (n_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, n_samples)
    local_indices = list(range(start_idx, end_idx))

    # Get ground truth (all processes need this for threshold search)
    gt = {tokens: [item.get('is_correct', False) for item in test_data[tokens]]
          for tokens in TOKEN_LEVELS}

    # Pre-compute classifier predictions for local samples
    if is_main_process():
        logger.info(f"Pre-computing classifier predictions (distributed across {world_size} GPUs)...")

    local_probs = {tokens: {} for tokens in TOKEN_LEVELS}

    desc = f"GPU {rank}" if world_size > 1 else "Computing predictions"
    for i in tqdm(local_indices, desc=desc, disable=not is_main_process()):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = test_data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            hidden = get_hidden_state(model, tokenizer, question, mentor_response, max_length, device)
            stage_tensor = torch.tensor([stage_idx], device=device)

            with torch.no_grad():
                logits = classifier_head(hidden.unsqueeze(0), stage_tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                local_probs[tokens][i] = prob

        if (i - start_idx) % 50 == 0:
            torch.cuda.empty_cache()

    # Gather predictions from all processes
    if world_size > 1:
        if is_main_process():
            logger.info("Gathering predictions from all GPUs...")

        # Convert to tensor for gathering
        all_probs = {tokens: [0.0] * n_samples for tokens in TOKEN_LEVELS}

        for tokens in TOKEN_LEVELS:
            local_tensor = torch.zeros(n_samples, device=device)
            for idx, prob in local_probs[tokens].items():
                local_tensor[idx] = prob

            # All-reduce to gather all probabilities
            dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)

            all_probs[tokens] = local_tensor.cpu().tolist()
    else:
        all_probs = {tokens: [0.0] * n_samples for tokens in TOKEN_LEVELS}
        for tokens in TOKEN_LEVELS:
            for idx, prob in local_probs[tokens].items():
                all_probs[tokens][idx] = prob

    def compute_cascade_acc(thresholds, return_decisions=False):
        """Compute cascade accuracy for given thresholds."""
        correct = 0
        decisions = []  # which token level each sample is assigned to
        for i in range(n_samples):
            decided = False
            stage_probs = []
            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                prob = all_probs[tokens][i]
                stage_probs.append((tokens, prob))
                if prob >= thresholds[stage_idx]:
                    correct += int(gt[tokens][i])
                    decisions.append(tokens)
                    decided = True
                    break
            if not decided:
                best_tokens, _ = max(stage_probs, key=lambda x: x[1])
                correct += int(gt[best_tokens][i])
                decisions.append(best_tokens)
        if return_decisions:
            return correct / n_samples, decisions
        return correct / n_samples

    # Use fixed thresholds or search
    if fixed_thresholds is not None:
        if is_main_process():
            logger.info(f"Using saved thresholds: {fixed_thresholds}")
        best_acc = compute_cascade_acc(fixed_thresholds)
        best_thresholds = fixed_thresholds
    else:
        if is_main_process():
            logger.info("Searching thresholds...")
        threshold_candidates = [round(i * 0.05, 2) for i in range(21)]  # 0.0 to 1.0, step 0.05
        best_acc = 0
        best_thresholds = None

        for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
            thresholds = list(combo)
            acc = compute_cascade_acc(thresholds)
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds

    # Get cascade decisions with best thresholds
    _, cascade_decisions = compute_cascade_acc(best_thresholds, return_decisions=True)

    # Compute Oracle decisions (lowest token level that gets correct answer)
    oracle_decisions = []
    for i in range(n_samples):
        chosen = TOKEN_LEVELS[-1]  # default to highest
        for tokens in TOKEN_LEVELS:
            if gt[tokens][i]:  # correct at this level
                chosen = tokens
                break
        oracle_decisions.append(chosen)

    # Compute per-stage AUC
    stage_auc = {}
    for tokens in TOKEN_LEVELS:
        try:
            auc = roc_auc_score(gt[tokens], all_probs[tokens])
            stage_auc[tokens] = auc
        except ValueError:
            stage_auc[tokens] = 0.5

    return {
        'best_accuracy': float(best_acc),
        'best_thresholds': best_thresholds,
        'auc': stage_auc,
        'cascade_decisions': cascade_decisions,
        'oracle_decisions': oracle_decisions,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA classifier cascade")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Base directory with subset folders")
    parser.add_argument("--subset", type=str, default="algebra",
                        choices=SUBSETS + ["all", "math500"],
                        help="Which subset to evaluate")
    parser.add_argument("--test-data-dir", type=str, default=None,
                        help="Override test data directory (e.g., for cross-dataset evaluation)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory with trained LoRA model (default: data_dir/{subset}/lora_model)")
    parser.add_argument("--base-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--search-thresholds", action="store_true",
                        help="Search for best thresholds instead of using saved ones")

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    use_distributed = world_size > 1

    if use_distributed:
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    # Determine model and test data directories
    if args.subset == "all":
        subset_dir = args.data_dir
        test_subdir = "all"
        test_split = "test"
    elif args.subset == "math500":
        subset_dir = args.data_dir
        test_subdir = "math500"
        test_split = "test"
    else:
        subset_dir = os.path.join(args.data_dir, args.subset)
        test_subdir = args.subset
        test_split = "test"

    if args.model_dir is None:
        if args.subset in ["all", "math500"]:
            args.model_dir = os.path.join(args.data_dir, "all", "lora_model")
        else:
            args.model_dir = os.path.join(subset_dir, "lora_model")

    # Override test data directory if specified
    if args.test_data_dir:
        test_data_path = args.test_data_dir
    else:
        test_data_path = os.path.join(args.data_dir, test_subdir)

    # Check model exists
    model_path = os.path.join(args.model_dir, "best_model.pt")
    if not os.path.exists(model_path):
        if is_main_process():
            logger.error(f"Model not found: {model_path}")
        cleanup_distributed()
        return

    if is_main_process():
        logger.info(f"Subset: {args.subset}")
        logger.info(f"Model dir: {args.model_dir}")
        logger.info(f"Test data: {test_data_path}")
        logger.info(f"Distributed: {use_distributed}, World size: {world_size}")

    # Load test data
    if is_main_process():
        logger.info("Loading test data...")
    test_data = load_json_data(test_data_path, split="test")
    if not test_data:
        if is_main_process():
            logger.error("No test data found!")
        cleanup_distributed()
        return

    n_test = len(test_data[TOKEN_LEVELS[0]])
    if is_main_process():
        logger.info(f"Test samples: {n_test}")

    # Baseline accuracy
    if is_main_process():
        logger.info("\nBaseline accuracy:")
    baseline_acc = {}
    for tokens in TOKEN_LEVELS:
        if tokens in test_data:
            correct = sum(1 for item in test_data[tokens] if item.get('is_correct', False))
            acc = correct / n_test
            baseline_acc[tokens] = acc
            if is_main_process():
                logger.info(f"  Tokens {tokens}: {acc:.4f} ({acc*100:.1f}%)")

    # Oracle accuracy
    oracle_acc = compute_oracle_accuracy(test_data)
    if is_main_process():
        logger.info(f"  Oracle: {oracle_acc:.4f} ({oracle_acc*100:.1f}%)")

    # Load tokenizer
    if is_main_process():
        logger.info(f"\nLoading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    if is_main_process():
        logger.info(f"Loading base model from {args.base_model}...")

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
        ).to(device)

    # Load LoRA weights
    if is_main_process():
        logger.info(f"Loading LoRA from {args.model_dir}...")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()

    # Load classifier head
    if is_main_process():
        logger.info("Loading classifier head...")
    hidden_size = model.config.hidden_size
    classifier_head = MentorClassifierHead(hidden_size).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    classifier_head.load_state_dict(checkpoint['classifier'])
    classifier_head.eval()

    # Get saved thresholds from checkpoint (if available and not searching)
    saved_thresholds = checkpoint.get('thresholds', None)
    if saved_thresholds and not args.search_thresholds:
        if is_main_process():
            logger.info(f"Found saved thresholds in checkpoint: {saved_thresholds}")
        fixed_thresholds = saved_thresholds
    else:
        if is_main_process():
            if args.search_thresholds:
                logger.info("--search-thresholds specified, will search for best thresholds")
            else:
                logger.info("No saved thresholds found, will search for best thresholds")
        fixed_thresholds = None

    # Synchronize before evaluation
    if use_distributed:
        dist.barrier()

    # Evaluate cascade
    if is_main_process():
        logger.info("\nEvaluating cascade...")
    result = evaluate_cascade_distributed(
        model, classifier_head, tokenizer, test_data,
        args.max_length, device, rank, world_size,
        fixed_thresholds=fixed_thresholds,
    )

    # Only main process prints results and saves
    if is_main_process():
        logger.info(f"\n{'='*60}")
        logger.info("Results Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Subset: {args.subset}")
        logger.info(f"Test samples: {n_test}")
        logger.info(f"\nBaseline:")
        for tokens, acc in baseline_acc.items():
            logger.info(f"  T{tokens}: {acc:.4f}")
        logger.info(f"Oracle: {oracle_acc:.4f}")
        logger.info(f"\nLoRA Cascade:")
        logger.info(f"  Best Accuracy: {result['best_accuracy']:.4f} ({result['best_accuracy']*100:.1f}%)")
        logger.info(f"  Thresholds: {result['best_thresholds']}")
        auc_str = ", ".join([f"T{t}={result['auc'][t]:.4f}" for t in TOKEN_LEVELS])
        logger.info(f"  Per-stage AUC: {auc_str}")

        # Gap analysis
        gap_to_oracle = oracle_acc - result['best_accuracy']
        gap_to_best_baseline = result['best_accuracy'] - max(baseline_acc.values())
        logger.info(f"\nGap to Oracle: {gap_to_oracle:.4f} ({gap_to_oracle*100:.1f}%)")
        logger.info(f"Improvement over best baseline: {gap_to_best_baseline:.4f} ({gap_to_best_baseline*100:.1f}%)")

        # Compute length statistics for Oracle and Cascade decisions
        def get_item_length(item):
            """Get mentor and intern length from an item."""
            m_len = item.get('mentor_length', 0)
            if not m_len and 'mentor_response' in item and item['mentor_response']:
                m_len = len(item['mentor_response']) // 4  # estimate tokens
            i_len = item.get('intern_length', item.get('num_tokens', 0))
            if not i_len and 'response' in item:
                i_len = len(item['response']) // 4
            return m_len, i_len

        # Compute Oracle length stats
        oracle_m_lens, oracle_i_lens = [], []
        for i, decision in enumerate(result['oracle_decisions']):
            item = test_data[decision][i]
            m_len, i_len = get_item_length(item)
            oracle_m_lens.append(m_len)
            oracle_i_lens.append(i_len)

        oracle_length = {
            'mentor_mean': float(np.mean(oracle_m_lens)) if oracle_m_lens else 0,
            'intern_mean': float(np.mean(oracle_i_lens)) if oracle_i_lens else 0,
        }

        # Compute Cascade length stats
        cascade_m_lens, cascade_i_lens = [], []
        for i, decision in enumerate(result['cascade_decisions']):
            item = test_data[decision][i]
            m_len, i_len = get_item_length(item)
            cascade_m_lens.append(m_len)
            cascade_i_lens.append(i_len)

        cascade_length = {
            'mentor_mean': float(np.mean(cascade_m_lens)) if cascade_m_lens else 0,
            'intern_mean': float(np.mean(cascade_i_lens)) if cascade_i_lens else 0,
        }

        logger.info(f"\nLength Statistics:")
        logger.info(f"  Oracle  - M_Len: {oracle_length['mentor_mean']:.1f}, I_Len: {oracle_length['intern_mean']:.1f}")
        logger.info(f"  Cascade - M_Len: {cascade_length['mentor_mean']:.1f}, I_Len: {cascade_length['intern_mean']:.1f}")

        # Save results
        output_file = os.path.join(args.model_dir, "cascade_eval.json")
        results = {
            'subset': args.subset,
            'n_test': n_test,
            'baseline': baseline_acc,
            'oracle': oracle_acc,
            'cascade_accuracy': result['best_accuracy'],
            'thresholds': result['best_thresholds'],
            'auc': {str(k): v for k, v in result['auc'].items()},
            'oracle_length': oracle_length,
            'cascade_length': cascade_length,
        }
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_file}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
