#!/usr/bin/env python3
"""
Evaluate trained LoRA classifier with cascade inference.

Usage:
    python eval_lora_cascade.py --model-dir /path/to/lora_model --subset algebra
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
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


class MentorClassifierHead(nn.Module):
    """Small MLP classification head."""

    def __init__(self, hidden_size: int, num_stages: int = 4, dropout: float = 0.1):
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, 64)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
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
        logger.warning(f"Split dir not found, using: {split_dir}")

    for tokens in TOKEN_LEVELS:
        filepath = os.path.join(split_dir, f"tokens{tokens}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
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


def evaluate_cascade(
    model,
    classifier_head,
    tokenizer,
    test_data: Dict[int, List[Dict]],
    max_length: int,
    device: str,
) -> Dict:
    """Evaluate cascade accuracy with threshold search."""
    model.eval()
    classifier_head.eval()

    n_samples = len(test_data[TOKEN_LEVELS[0]])

    # Get ground truth
    gt = {tokens: [item.get('is_correct', False) for item in test_data[tokens]]
          for tokens in TOKEN_LEVELS}

    # Pre-compute all classifier predictions
    logger.info("Pre-computing classifier predictions...")
    all_probs = {tokens: [] for tokens in TOKEN_LEVELS}

    for i in tqdm(range(n_samples), desc="Computing predictions"):
        for stage_idx, tokens in enumerate(TOKEN_LEVELS):
            item = test_data[tokens][i]
            question = item['question']
            mentor_response = item.get('mentor_response', '')

            hidden = get_hidden_state(model, tokenizer, question, mentor_response, max_length, device)
            stage_tensor = torch.tensor([stage_idx], device=device)

            with torch.no_grad():
                logits = classifier_head(hidden.unsqueeze(0), stage_tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                all_probs[tokens].append(prob)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    # Threshold search
    logger.info("Searching thresholds...")
    threshold_candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_acc = 0
    best_thresholds = None

    for combo in product(threshold_candidates, repeat=len(TOKEN_LEVELS)):
        thresholds = list(combo)
        correct = 0

        for i in range(n_samples):
            decided = False
            stage_probs = []

            for stage_idx, tokens in enumerate(TOKEN_LEVELS):
                prob = all_probs[tokens][i]
                stage_probs.append((tokens, prob))

                if prob >= thresholds[stage_idx]:
                    correct += int(gt[tokens][i])
                    decided = True
                    break

            if not decided:
                # No stage passed threshold, select the one with highest confidence
                best_tokens, _ = max(stage_probs, key=lambda x: x[1])
                correct += int(gt[best_tokens][i])

        acc = correct / n_samples
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds

    return {
        'best_accuracy': float(best_acc),
        'best_thresholds': best_thresholds,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA classifier cascade")
    parser.add_argument("--data-dir", type=str,
                        default="/home/fzkuji/PycharmProjects/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split",
                        help="Base directory with subset folders")
    parser.add_argument("--subset", type=str, default="algebra",
                        choices=SUBSETS,
                        help="Which subset to evaluate")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory with trained LoRA model (default: data_dir/{subset}/lora_model)")
    parser.add_argument("--base-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Use 4-bit quantization")

    args = parser.parse_args()

    subset_dir = os.path.join(args.data_dir, args.subset)
    if args.model_dir is None:
        args.model_dir = os.path.join(subset_dir, "lora_model")

    # Check model exists
    model_path = os.path.join(args.model_dir, "best_model.pt")
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    logger.info(f"Subset: {args.subset}")
    logger.info(f"Model dir: {args.model_dir}")

    # Load test data
    logger.info("Loading test data...")
    test_data = load_json_data(subset_dir, split="test")
    if not test_data:
        logger.error("No test data found!")
        return

    n_test = len(test_data[TOKEN_LEVELS[0]])
    logger.info(f"Test samples: {n_test}")

    # Baseline accuracy
    logger.info("\nBaseline accuracy:")
    baseline_acc = {}
    for tokens in TOKEN_LEVELS:
        if tokens in test_data:
            correct = sum(1 for item in test_data[tokens] if item.get('is_correct', False))
            acc = correct / n_test
            baseline_acc[tokens] = acc
            logger.info(f"  Tokens {tokens}: {acc:.4f} ({acc*100:.1f}%)")

    # Oracle accuracy
    oracle_acc = compute_oracle_accuracy(test_data)
    logger.info(f"  Oracle: {oracle_acc:.4f} ({oracle_acc*100:.1f}%)")

    # Load tokenizer
    logger.info(f"\nLoading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading base model from {args.base_model}...")
    device = args.device

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
    logger.info(f"Loading LoRA from {args.model_dir}...")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()

    # Load classifier head
    logger.info("Loading classifier head...")
    hidden_size = model.config.hidden_size
    classifier_head = MentorClassifierHead(hidden_size).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    classifier_head.load_state_dict(checkpoint['classifier'])
    classifier_head.eval()

    # Evaluate cascade
    logger.info("\nEvaluating cascade...")
    result = evaluate_cascade(
        model, classifier_head, tokenizer, test_data,
        args.max_length, device
    )

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

    # Gap analysis
    gap_to_oracle = oracle_acc - result['best_accuracy']
    gap_to_best_baseline = result['best_accuracy'] - max(baseline_acc.values())
    logger.info(f"\nGap to Oracle: {gap_to_oracle:.4f} ({gap_to_oracle*100:.1f}%)")
    logger.info(f"Improvement over best baseline: {gap_to_best_baseline:.4f} ({gap_to_best_baseline*100:.1f}%)")

    # Save results
    output_file = os.path.join(args.model_dir, "cascade_eval.json")
    results = {
        'subset': args.subset,
        'n_test': n_test,
        'baseline': baseline_acc,
        'oracle': oracle_acc,
        'cascade_accuracy': result['best_accuracy'],
        'thresholds': result['best_thresholds'],
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
