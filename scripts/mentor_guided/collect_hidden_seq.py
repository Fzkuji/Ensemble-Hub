#!/usr/bin/env python3
"""
Collect full sequence of hidden states (not pooled) for transformer classifier.

Instead of pooling mentor tokens to a single vector, we save the full sequence
so a transformer can learn which positions are important.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HiddenSeqExtractor:
    """Extract full sequence of hidden states from intern model."""

    def __init__(
        self,
        model_path: str,
        hidden_layers: List[int] = None,
        max_mentor_tokens: int = 512,
        device: str = "cuda",
    ):
        self.device = device
        self.max_mentor_tokens = max_mentor_tokens

        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            output_hidden_states=True,
        )
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        # Default: use last layer only
        self.hidden_layers = hidden_layers or [-1]
        logger.info(f"Hidden dim: {self.hidden_dim}, layers: {self.hidden_layers}")

    def extract(
        self,
        question: str,
        mentor_response: str,
    ) -> torch.Tensor:
        """
        Extract hidden states sequence for mentor tokens.

        Returns:
            Tensor of shape [seq_len, hidden_dim] or [seq_len, num_layers * hidden_dim]
        """
        # Tokenize
        question_ids = self.tokenizer.encode(question, return_tensors="pt").to(self.device)
        full_text = question + mentor_response
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

        question_len = question_ids.shape[1]
        total_len = full_ids.shape[1]
        mentor_len = total_len - question_len

        if mentor_len <= 0:
            # Return empty tensor with correct shape
            return torch.zeros(1, len(self.hidden_layers) * self.hidden_dim)

        # Truncate if too long
        if mentor_len > self.max_mentor_tokens:
            full_ids = full_ids[:, :question_len + self.max_mentor_tokens]
            mentor_len = self.max_mentor_tokens

        # Forward pass
        with torch.no_grad():
            outputs = self.model(full_ids, output_hidden_states=True)

        all_hidden = outputs.hidden_states  # Tuple of [batch, seq, hidden]

        # Extract mentor portion from each layer
        collected_layers = []
        for layer_idx in self.hidden_layers:
            layer_hidden = all_hidden[layer_idx][0]  # [seq, hidden]
            mentor_hidden = layer_hidden[question_len:total_len]  # [mentor_len, hidden]
            collected_layers.append(mentor_hidden)

        # Concatenate layers: [mentor_len, num_layers * hidden_dim]
        if len(collected_layers) == 1:
            result = collected_layers[0]
        else:
            result = torch.cat(collected_layers, dim=-1)

        return result.cpu().float()


def load_collected_data(data_dir: str, token_level: int) -> List[Dict]:
    """Load existing collected JSON data."""
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and f'tokens{token_level}' in filename:
            if 'mentor_only' in filename:
                continue
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description="Collect hidden state sequences")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with collected JSON files")
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Intern model path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data_dir/hidden_seq)")
    parser.add_argument("--token-levels", type=int, nargs='+',
                        default=[0, 100, 500, 1000],
                        help="Token levels to process")
    parser.add_argument("--max-mentor-tokens", type=int, default=512,
                        help="Max mentor tokens to keep")
    parser.add_argument("--hidden-layers", type=int, nargs='+',
                        default=[-1],
                        help="Which layers to extract (default: last layer only)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "hidden_seq")
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize extractor
    extractor = HiddenSeqExtractor(
        model_path=args.model_path,
        hidden_layers=args.hidden_layers,
        max_mentor_tokens=args.max_mentor_tokens,
        device=args.device,
    )

    for token_level in args.token_levels:
        logger.info(f"\nProcessing token level {token_level}...")

        # Load data
        data = load_collected_data(args.data_dir, token_level)
        if not data:
            logger.warning(f"No data found for tokens{token_level}")
            continue

        logger.info(f"Found {len(data)} samples")

        # Extract hidden sequences
        all_hidden_seqs = []
        all_labels = []
        all_seq_lens = []

        for item in tqdm(data, desc=f"tokens{token_level}"):
            question = item.get('question', '')
            mentor_response = item.get('mentor_response', '')
            is_correct = item.get('is_correct', False)

            hidden_seq = extractor.extract(question, mentor_response)
            all_hidden_seqs.append(hidden_seq)
            all_labels.append(1 if is_correct else 0)
            all_seq_lens.append(hidden_seq.shape[0])

        # Pad sequences to same length
        max_len = max(all_seq_lens)
        hidden_dim = all_hidden_seqs[0].shape[1]

        padded_hidden = torch.zeros(len(all_hidden_seqs), max_len, hidden_dim)
        attention_mask = torch.zeros(len(all_hidden_seqs), max_len)

        for i, (seq, seq_len) in enumerate(zip(all_hidden_seqs, all_seq_lens)):
            padded_hidden[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1

        labels = torch.tensor(all_labels, dtype=torch.long)

        # Save
        save_path = os.path.join(args.output_dir, f"tokens{token_level}.pt")
        torch.save({
            'hidden_states': padded_hidden,  # [N, max_len, hidden_dim]
            'attention_mask': attention_mask,  # [N, max_len]
            'labels': labels,  # [N]
            'seq_lens': torch.tensor(all_seq_lens),  # [N]
        }, save_path)

        logger.info(f"Saved {len(data)} samples to {save_path}")
        logger.info(f"  Shape: {padded_hidden.shape}")
        logger.info(f"  Max seq len: {max_len}, Mean: {sum(all_seq_lens)/len(all_seq_lens):.1f}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
