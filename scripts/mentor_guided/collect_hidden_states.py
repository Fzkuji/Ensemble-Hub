#!/usr/bin/env python3
"""
Collect Hidden States for ACT-E Experiments

Instead of using PPL/Entropy, this script collects hidden states from the intern model
when processing mentor's response. These hidden states are used to predict whether
more mentor tokens are needed.

Flow:
1. Load existing collected data (mentor responses already generated)
2. Intern model processes: question + mentor_response
3. Extract hidden states at last token position
4. Train a lightweight classifier on hidden states to predict if more tokens needed
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HiddenStateExtractor:
    """Extract hidden states from intern model."""

    def __init__(
        self,
        intern_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        device: str = "cuda:0",
        hidden_layers: List[int] = None,
    ):
        self.device = device
        self.hidden_layers = hidden_layers or [-1]  # Default: last layer only

        logger.info(f"Loading intern model: {intern_model} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(intern_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            intern_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            output_hidden_states=True,
        ).to(device)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        logger.info(f"Hidden dim: {self.hidden_dim}, Layers to extract: {self.hidden_layers}")

    def extract(
        self,
        question: str,
        mentor_response: str,
        pool_method: str = "last",
    ) -> torch.Tensor:
        """
        Extract hidden states for mentor response portion.

        Args:
            question: Input question
            mentor_response: Mentor's response
            pool_method: "last", "mean", or "max"

        Returns:
            Tensor of shape [num_layers, hidden_dim]
        """
        if not mentor_response:
            return torch.zeros(len(self.hidden_layers), self.hidden_dim)

        full_text = question + mentor_response

        # Get token boundaries
        question_ids = self.tokenizer(question, return_tensors="pt")["input_ids"]
        full_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"].to(self.device)

        question_len = question_ids.shape[1]
        total_len = full_ids.shape[1]
        mentor_len = total_len - question_len

        if mentor_len <= 0:
            return torch.zeros(len(self.hidden_layers), self.hidden_dim)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(full_ids, output_hidden_states=True)

        all_hidden = outputs.hidden_states  # Tuple of [batch, seq, hidden]

        # Extract and pool
        collected = []
        for layer_idx in self.hidden_layers:
            layer_hidden = all_hidden[layer_idx][0]  # [seq, hidden]
            mentor_hidden = layer_hidden[question_len:total_len]  # [mentor_len, hidden]

            if pool_method == "last":
                pooled = mentor_hidden[-1]
            elif pool_method == "mean":
                pooled = mentor_hidden.mean(dim=0)
            elif pool_method == "max":
                pooled = mentor_hidden.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown pool method: {pool_method}")

            collected.append(pooled)

        return torch.stack(collected).cpu().float()


def load_collected_data(data_dir: str) -> Dict[int, List[Dict]]:
    """Load existing collected data from JSON files."""
    data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and 'tokens' in filename:
            if 'mentor_only' in filename:
                continue
            # Extract token count from filename
            # Format: hendrycks_math_all_tokens100.json
            try:
                tokens = int(filename.split('tokens')[1].split('.')[0])
            except:
                continue

            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                data[tokens] = json.load(f)
            logger.info(f"Loaded {len(data[tokens])} samples for {tokens} tokens")

    return data


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from collected data")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with collected JSON files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for hidden states")
    parser.add_argument("--intern-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pool-method", type=str, default="last", choices=["last", "mean", "max"])
    parser.add_argument("--hidden-layers", type=str, default="-1",
                        help="Comma-separated layer indices, e.g., '-1' or '-1,-2,-3,-4'")
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()

    # Parse hidden layers
    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "hidden_states")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load existing data
    collected_data = load_collected_data(args.data_dir)
    if not collected_data:
        logger.error(f"No data found in {args.data_dir}")
        return

    # Initialize extractor
    extractor = HiddenStateExtractor(
        intern_model=args.intern_model,
        device=args.device,
        hidden_layers=hidden_layers,
    )

    # Process each token level
    for tokens, samples in collected_data.items():
        logger.info(f"\n=== Extracting hidden states for {tokens} tokens ===")

        hidden_list = []
        labels = []

        for sample in tqdm(samples, desc=f"tokens={tokens}"):
            try:
                question = sample['question']
                mentor_response = sample['mentor_response']
                is_correct = sample['is_correct']

                hidden = extractor.extract(question, mentor_response, args.pool_method)
                hidden_list.append(hidden)
                labels.append(1 if is_correct else 0)

            except Exception as e:
                logger.warning(f"Error: {e}")
                # Add zeros for failed samples
                hidden_list.append(torch.zeros(len(hidden_layers), extractor.hidden_dim))
                labels.append(0)

        # Stack and save
        hidden_tensor = torch.stack(hidden_list)  # [num_samples, num_layers, hidden_dim]
        labels_tensor = torch.tensor(labels)

        # Save
        torch.save({
            'hidden_states': hidden_tensor,
            'labels': labels_tensor,
            'tokens': tokens,
            'pool_method': args.pool_method,
            'hidden_layers': hidden_layers,
        }, os.path.join(args.output_dir, f"tokens{tokens}.pt"))

        logger.info(f"Saved: shape={hidden_tensor.shape}, correct={labels_tensor.sum()}/{len(labels_tensor)}")

    logger.info(f"\nDone! Hidden states saved to {args.output_dir}")


if __name__ == "__main__":
    main()
