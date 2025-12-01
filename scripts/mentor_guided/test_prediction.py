#!/usr/bin/env python3
"""
Test two approaches:
1. Predict whether mentor help will be useful (before running)
2. Dynamic mentor: generate until student confidence is high

方向1：提前预测是否需要帮助
方向2：动态决定何时停止帮助
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_boxed_content(text: str) -> str:
    start = text.find(r'\boxed{')
    if start == -1:
        return ""
    i = start + len(r'\boxed{')
    depth = 1
    content = ""
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        if depth > 0:
            content += text[i]
        i += 1
    return content.strip()


class PredictionTester:
    """Test prediction and dynamic mentor approaches."""

    def __init__(self, mentor_model_name: str, student_model_name: str, device: str = "cuda"):
        self.device = device

        logger.info(f"Loading mentor: {mentor_model_name}")
        self.mentor_tokenizer = AutoTokenizer.from_pretrained(mentor_model_name, trust_remote_code=True)
        self.mentor_model = AutoModelForCausalLM.from_pretrained(
            mentor_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.mentor_model.eval()

        logger.info(f"Loading student: {student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.student_model.eval()

        for tok in [self.mentor_tokenizer, self.student_tokenizer]:
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        probs = F.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy().astype(np.float64)
        probs_np = probs_np[probs_np > 1e-10]
        if len(probs_np) == 0:
            return 0.0
        probs_np = probs_np / probs_np.sum()
        entropy = -np.sum(probs_np * np.log2(probs_np))
        return float(entropy) if not (np.isnan(entropy) or np.isinf(entropy)) else 0.0

    # ============ Approach 1: Prediction Features ============
    def get_prediction_features(self, prompt: str) -> Dict[str, float]:
        """
        Extract features to predict if mentor help will be useful.

        Features:
        - initial_entropy: student's entropy at start (high = uncertain = might need help)
        - top1_prob: student's confidence in first token
        - entropy_variance: variance in first N tokens
        """
        inputs = self.student_tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.student_model.device)

        # Get initial entropy
        with torch.no_grad():
            outputs = self.student_model(input_ids)
            last_logits = outputs.logits[0, -1, :]

        initial_entropy = self.calculate_entropy(last_logits)
        probs = F.softmax(last_logits, dim=-1)
        top1_prob = probs.max().item()

        # Generate a few tokens and get entropy trajectory
        entropies = [initial_entropy]
        current_ids = input_ids

        for _ in range(10):  # First 10 tokens
            with torch.no_grad():
                outputs = self.student_model(current_ids)
                logits = outputs.logits[0, -1, :]
                entropy = self.calculate_entropy(logits)
                entropies.append(entropy)

                next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                if next_token.item() == self.student_tokenizer.eos_token_id:
                    break
                current_ids = torch.cat([current_ids, next_token], dim=1)

        return {
            "initial_entropy": initial_entropy,
            "top1_prob": top1_prob,
            "mean_entropy_10": float(np.mean(entropies)),
            "max_entropy_10": float(np.max(entropies)),
            "entropy_std_10": float(np.std(entropies)),
        }

    # ============ Approach 2: Dynamic Mentor ============
    def dynamic_mentor_generate(
        self,
        prompt: str,
        entropy_threshold: float = 1.0,
        max_mentor_tokens: int = 300,
        min_mentor_tokens: int = 20,
    ) -> Tuple[str, int, List[float]]:
        """
        Mentor generates until student's entropy drops below threshold.

        Returns: (mentor_text, actual_tokens_used, entropy_trajectory)
        """
        inputs = self.mentor_tokenizer(prompt, return_tensors="pt", truncation=True)
        mentor_ids = inputs["input_ids"].to(self.mentor_model.device)

        generated_tokens = []
        entropy_trajectory = []

        for i in range(max_mentor_tokens):
            # Mentor generates one token
            with torch.no_grad():
                outputs = self.mentor_model(mentor_ids)
                logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(logits).item()

            if next_token == self.mentor_tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)
            mentor_ids = torch.cat([mentor_ids, torch.tensor([[next_token]], device=mentor_ids.device)], dim=1)

            # Check student's entropy with current mentor output
            if i >= min_mentor_tokens - 1:  # After minimum tokens
                current_text = prompt + self.mentor_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                student_inputs = self.student_tokenizer(current_text, return_tensors="pt", truncation=True)
                student_ids = student_inputs["input_ids"].to(self.student_model.device)

                with torch.no_grad():
                    student_outputs = self.student_model(student_ids)
                    student_logits = student_outputs.logits[0, -1, :]

                entropy = self.calculate_entropy(student_logits)
                entropy_trajectory.append(entropy)

                # Stop if student is confident enough
                if entropy < entropy_threshold:
                    logger.info(f"Dynamic stop at token {i+1}, entropy={entropy:.3f}")
                    break

        mentor_text = self.mentor_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return mentor_text, len(generated_tokens), entropy_trajectory

    def generate_answer(self, prompt: str, max_tokens: int = 2048) -> str:
        inputs = self.student_tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.student_model.device)

        with torch.no_grad():
            outputs = self.student_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.student_tokenizer.pad_token_id,
            )

        new_ids = outputs[0, input_ids.shape[1]:]
        return self.student_tokenizer.decode(new_ids, skip_special_tokens=True)

    def test_sample(
        self,
        problem: str,
        ground_truth: str,
        entropy_threshold: float = 1.0,
    ) -> Dict[str, Any]:
        """Test both approaches on one sample."""

        prompt = f"""Solve the following math problem. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

        # Get prediction features
        features = self.get_prediction_features(prompt)

        # Baseline: student alone
        baseline_answer = self.generate_answer(prompt)
        baseline_boxed = extract_boxed_content(baseline_answer)
        baseline_correct = grade_answer(baseline_boxed, ground_truth)

        # Dynamic mentor approach
        mentor_text, tokens_used, entropy_traj = self.dynamic_mentor_generate(
            prompt, entropy_threshold=entropy_threshold
        )
        dynamic_prompt = prompt + mentor_text
        dynamic_answer = self.generate_answer(dynamic_prompt)
        dynamic_full = mentor_text + dynamic_answer
        dynamic_boxed = extract_boxed_content(dynamic_full)
        dynamic_correct = grade_answer(dynamic_boxed, ground_truth)

        return {
            "ground_truth": ground_truth,
            "features": features,
            "baseline": {
                "correct": baseline_correct,
                "predicted": baseline_boxed,
            },
            "dynamic": {
                "correct": dynamic_correct,
                "predicted": dynamic_boxed,
                "mentor_tokens": tokens_used,
                "final_entropy": entropy_traj[-1] if entropy_traj else None,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-32B-Instruct')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', default='HuggingFaceH4/MATH-500')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--difficulty', type=str, default='5')
    parser.add_argument('--entropy-threshold', type=float, default=1.0,
                       help='Stop mentor when student entropy drops below this')
    parser.add_argument('--output-file', default='prediction_results.json')

    args = parser.parse_args()

    difficulty_levels = [int(d.strip()) for d in args.difficulty.split(',')]

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset, split="test")
    except:
        dataset = load_dataset(args.dataset, split="train")

    if 'level' in dataset.column_names:
        dataset = dataset.filter(lambda x: x['level'] in difficulty_levels)
        logger.info(f"Filtered to {len(dataset)} samples")

    # Initialize tester
    tester = PredictionTester(args.mentor_model, args.student_model)

    results = []
    num_samples = min(args.num_samples, len(dataset))

    for i in tqdm(range(num_samples), desc="Testing"):
        sample = dataset[i]
        problem = sample.get("problem", sample.get("question", ""))

        if "solution" in sample:
            ground_truth = extract_boxed_content(sample["solution"])
        else:
            ground_truth = str(sample.get("answer", ""))

        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {i+1}/{num_samples}")
        logger.info(f"GT: {ground_truth}")

        result = tester.test_sample(problem, ground_truth, args.entropy_threshold)
        result["idx"] = i
        results.append(result)

        b_status = "✓" if result["baseline"]["correct"] else "✗"
        d_status = "✓" if result["dynamic"]["correct"] else "✗"
        logger.info(f"Baseline: {b_status} | Dynamic ({result['dynamic']['mentor_tokens']} tokens): {d_status}")

    # Analysis
    baseline_correct = sum(1 for r in results if r["baseline"]["correct"])
    dynamic_correct = sum(1 for r in results if r["dynamic"]["correct"])

    rescued = sum(1 for r in results if not r["baseline"]["correct"] and r["dynamic"]["correct"])
    hurt = sum(1 for r in results if r["baseline"]["correct"] and not r["dynamic"]["correct"])

    avg_tokens = np.mean([r["dynamic"]["mentor_tokens"] for r in results])

    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Baseline accuracy: {baseline_correct}/{num_samples} ({100*baseline_correct/num_samples:.1f}%)")
    logger.info(f"Dynamic accuracy: {dynamic_correct}/{num_samples} ({100*dynamic_correct/num_samples:.1f}%)")
    logger.info(f"Rescued: {rescued}, Hurt: {hurt}, Net: {rescued - hurt:+d}")
    logger.info(f"Average mentor tokens used: {avg_tokens:.1f}")

    # Feature analysis for prediction
    logger.info("\n" + "="*60)
    logger.info("FEATURE ANALYSIS (for predicting when help is useful)")
    logger.info("="*60)

    helped_samples = [r for r in results if not r["baseline"]["correct"] and r["dynamic"]["correct"]]
    hurt_samples = [r for r in results if r["baseline"]["correct"] and not r["dynamic"]["correct"]]

    if helped_samples:
        logger.info("Samples where dynamic HELPED:")
        for key in ["initial_entropy", "mean_entropy_10", "max_entropy_10"]:
            vals = [r["features"][key] for r in helped_samples]
            logger.info(f"  {key}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    if hurt_samples:
        logger.info("Samples where dynamic HURT:")
        for key in ["initial_entropy", "mean_entropy_10", "max_entropy_10"]:
            vals = [r["features"][key] for r in hurt_samples]
            logger.info(f"  {key}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    # Save
    output_path = os.path.join(script_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
