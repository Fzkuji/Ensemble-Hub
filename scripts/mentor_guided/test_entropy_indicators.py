#!/usr/bin/env python3
"""
Test Different Entropy-based Indicators for Mentor-Guided Inference.

测试三个方向：
1. 熵变化模式：有大模型帮助 vs 无帮助时，熵轨迹的差异
2. 高熵token收敛能力：小模型遇到高熵token后能否收敛
3. 续写一致性：小模型续写与大模型输出的相似度

用 AIME25 数据集测试，评估哪个指标能预测答案正确性。
"""

import argparse
import json
import logging
import os
import sys
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_boxed_content(text: str) -> str:
    """Extract content inside \\boxed{} command."""
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


@dataclass
class IndicatorResult:
    """Results for one sample."""
    problem_id: int
    ground_truth: str

    # For each mentor length
    length_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class EntropyIndicatorTester:
    """Test different entropy-based indicators."""

    def __init__(
        self,
        mentor_model_name: str,
        student_model_name: str,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load models
        logger.info(f"Loading mentor: {mentor_model_name}")
        self.mentor_tokenizer = AutoTokenizer.from_pretrained(mentor_model_name, trust_remote_code=True)
        self.mentor_model = self._load_model(mentor_model_name)

        logger.info(f"Loading student: {student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
        self.student_model = self._load_model(student_model_name)

        for tok in [self.mentor_tokenizer, self.student_tokenizer]:
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

    def _load_model(self, model_name: str):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Loading with 8-bit: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, load_in_8bit=True, device_map="auto", trust_remote_code=True
            )
        model.eval()
        return model

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculate entropy in bits."""
        probs = F.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy().astype(np.float64)
        probs_np = probs_np[probs_np > 1e-10]
        if len(probs_np) == 0:
            return 0.0
        probs_np = probs_np / probs_np.sum()
        entropy = -np.sum(probs_np * np.log(probs_np)) / np.log(2)
        return float(entropy) if not (np.isnan(entropy) or np.isinf(entropy)) else 0.0

    def get_entropy_trajectory(
        self,
        model,
        tokenizer,
        text: str,
        max_new_tokens: int = 100
    ) -> Tuple[List[float], List[int]]:
        """Generate tokens and track entropy trajectory."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        entropies = []
        generated_ids = []
        current_ids = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(current_ids)
                last_logits = outputs.logits[0, -1, :]

            entropy = self.calculate_entropy(last_logits)
            entropies.append(entropy)

            # Greedy decode for consistency
            next_token = torch.argmax(last_logits).unsqueeze(0).unsqueeze(0)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return entropies, generated_ids

    def generate_mentor_tokens(self, prompt: str, num_tokens: int) -> Tuple[str, List[int]]:
        """Generate fixed number of tokens from mentor."""
        if num_tokens == 0:
            return "", []

        inputs = self.mentor_tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.mentor_model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=self.mentor_tokenizer.pad_token_id,
            )

        new_ids = outputs[0, input_ids.shape[1]:].tolist()
        new_text = self.mentor_tokenizer.decode(new_ids, skip_special_tokens=True)
        return new_text, new_ids

    def generate_student_answer(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate complete answer from student."""
        inputs = self.student_tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.student_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.student_tokenizer.pad_token_id,
            )

        new_ids = outputs[0, input_ids.shape[1]:]
        return self.student_tokenizer.decode(new_ids, skip_special_tokens=True)

    # ========== Indicator 1: Entropy Change Pattern ==========
    def compute_entropy_change_pattern(
        self,
        prompt: str,
        mentor_text: str,
        num_tokens: int = 50
    ) -> Dict[str, float]:
        """
        Compare entropy trajectory: with vs without mentor help.

        Returns metrics about the difference pattern.
        """
        # Without help
        entropies_base, _ = self.get_entropy_trajectory(
            self.student_model, self.student_tokenizer,
            prompt, num_tokens
        )

        # With mentor help
        entropies_helped, _ = self.get_entropy_trajectory(
            self.student_model, self.student_tokenizer,
            prompt + mentor_text, num_tokens
        )

        if not entropies_base or not entropies_helped:
            return {"entropy_reduction_mean": 0, "entropy_reduction_max": 0}

        # Align lengths
        min_len = min(len(entropies_base), len(entropies_helped))
        base = np.array(entropies_base[:min_len])
        helped = np.array(entropies_helped[:min_len])

        diff = base - helped  # Positive = entropy reduced (good)

        return {
            "entropy_reduction_mean": float(np.mean(diff)),
            "entropy_reduction_max": float(np.max(diff)) if len(diff) > 0 else 0,
            "entropy_reduction_std": float(np.std(diff)),
            "base_mean": float(np.mean(base)),
            "helped_mean": float(np.mean(helped)),
        }

    # ========== Indicator 2: High-Entropy Token Convergence ==========
    def compute_convergence_ability(
        self,
        prompt: str,
        mentor_text: str,
        num_tokens: int = 100,
        high_entropy_threshold: float = 2.0,
        window_after: int = 10
    ) -> Dict[str, float]:
        """
        Check if student can recover after high-entropy tokens.

        After each high-entropy token, check if entropy drops in next N tokens.
        """
        entropies, _ = self.get_entropy_trajectory(
            self.student_model, self.student_tokenizer,
            prompt + mentor_text, num_tokens
        )

        if len(entropies) < window_after + 1:
            return {"convergence_rate": 0, "avg_recovery": 0}

        high_entropy_indices = [i for i, e in enumerate(entropies) if e > high_entropy_threshold]

        if not high_entropy_indices:
            return {"convergence_rate": 1.0, "avg_recovery": 0, "num_high_entropy": 0}

        recoveries = []
        for idx in high_entropy_indices:
            if idx + window_after < len(entropies):
                initial = entropies[idx]
                after_window = entropies[idx + 1:idx + window_after + 1]
                min_after = min(after_window)
                recovery = (initial - min_after) / initial if initial > 0 else 0
                recoveries.append(recovery)

        convergence_rate = len([r for r in recoveries if r > 0.5]) / len(recoveries) if recoveries else 0

        return {
            "convergence_rate": float(convergence_rate),
            "avg_recovery": float(np.mean(recoveries)) if recoveries else 0,
            "num_high_entropy": len(high_entropy_indices),
            "high_entropy_ratio": len(high_entropy_indices) / len(entropies),
        }

    # ========== Indicator 3: Continuation Consistency ==========
    def compute_continuation_consistency(
        self,
        prompt: str,
        mentor_text: str,
        continuation_length: int = 50
    ) -> Dict[str, float]:
        """
        Compare student's continuation with mentor's full output.

        If student continues in the same direction as mentor would,
        it suggests student understands the path.
        """
        # Get mentor's full continuation
        mentor_full, mentor_ids = self.get_entropy_trajectory(
            self.mentor_model, self.mentor_tokenizer,
            prompt + mentor_text, continuation_length
        )

        # Get student's continuation
        student_cont, student_ids = self.get_entropy_trajectory(
            self.student_model, self.student_tokenizer,
            prompt + mentor_text, continuation_length
        )

        if not mentor_ids or not student_ids:
            return {"token_overlap": 0, "direction_consistency": 0}

        # Token overlap (how many tokens match)
        min_len = min(len(mentor_ids), len(student_ids))
        matches = sum(1 for i in range(min_len) if mentor_ids[i] == student_ids[i])
        token_overlap = matches / min_len if min_len > 0 else 0

        # First token match (most important)
        first_match = 1.0 if mentor_ids[0] == student_ids[0] else 0.0

        # Entropy correlation
        min_ent_len = min(len(mentor_full), len(student_cont))
        if min_ent_len > 1:
            corr = np.corrcoef(mentor_full[:min_ent_len], student_cont[:min_ent_len])[0, 1]
            entropy_corr = float(corr) if not np.isnan(corr) else 0
        else:
            entropy_corr = 0

        return {
            "token_overlap": float(token_overlap),
            "first_token_match": first_match,
            "entropy_correlation": entropy_corr,
        }

    def evaluate_sample(
        self,
        problem: str,
        ground_truth: str,
        mentor_lengths: List[int],
        student_max_tokens: int = 2048,
    ) -> IndicatorResult:
        """Evaluate one sample with all indicators."""

        # Build prompt
        prompt = f"""Solve the following math problem. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

        result = IndicatorResult(
            problem_id=0,
            ground_truth=ground_truth,
        )

        for length in mentor_lengths:
            logger.info(f"  Testing mentor length: {length}")

            # Generate mentor tokens
            mentor_text, _ = self.generate_mentor_tokens(prompt, length)

            # Generate student's full answer
            full_prompt = prompt + mentor_text
            student_answer = self.generate_student_answer(full_prompt, student_max_tokens)
            full_answer = mentor_text + student_answer

            # Check correctness
            predicted_boxed = extract_boxed_content(full_answer)
            is_correct = grade_answer(predicted_boxed, ground_truth)

            # Compute indicators
            indicator1 = self.compute_entropy_change_pattern(prompt, mentor_text)
            indicator2 = self.compute_convergence_ability(prompt, mentor_text)
            indicator3 = self.compute_continuation_consistency(prompt, mentor_text)

            result.length_results[length] = {
                "mentor_text": mentor_text[:200],
                "student_answer": student_answer[:200],
                "predicted_boxed": predicted_boxed,
                "is_correct": is_correct,
                # Indicators
                "ind1_entropy_change": indicator1,
                "ind2_convergence": indicator2,
                "ind3_consistency": indicator3,
            }

        return result


def compute_correlations(results: List[IndicatorResult]) -> Dict[str, float]:
    """Compute correlation between each indicator and correctness."""

    # Collect data points
    data_points = []
    for result in results:
        for length, lr in result.length_results.items():
            data_points.append({
                "length": length,
                "correct": 1 if lr["is_correct"] else 0,
                # Indicator 1
                "ent_red_mean": lr["ind1_entropy_change"].get("entropy_reduction_mean", 0),
                "ent_red_max": lr["ind1_entropy_change"].get("entropy_reduction_max", 0),
                # Indicator 2
                "convergence_rate": lr["ind2_convergence"].get("convergence_rate", 0),
                "avg_recovery": lr["ind2_convergence"].get("avg_recovery", 0),
                # Indicator 3
                "token_overlap": lr["ind3_consistency"].get("token_overlap", 0),
                "first_match": lr["ind3_consistency"].get("first_token_match", 0),
                "ent_corr": lr["ind3_consistency"].get("entropy_correlation", 0),
            })

    if not data_points:
        return {}

    correct = np.array([d["correct"] for d in data_points])

    correlations = {}
    for key in ["ent_red_mean", "ent_red_max", "convergence_rate", "avg_recovery",
                "token_overlap", "first_match", "ent_corr"]:
        values = np.array([d[key] for d in data_points])
        if np.std(values) > 0 and np.std(correct) > 0:
            corr = np.corrcoef(correct, values)[0, 1]
            correlations[key] = float(corr) if not np.isnan(corr) else 0
        else:
            correlations[key] = 0

    return correlations


def main():
    parser = argparse.ArgumentParser(description='Test Entropy-based Indicators')

    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-32B-Instruct')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', default='lighteval/MATH')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--difficulty', type=str, default='Level 2,Level 3',
                       help='Comma-separated difficulty levels (Level 1-5)')
    parser.add_argument('--lengths', type=str, default='0,50,100,200')
    parser.add_argument('--output-file', default='indicator_results.json')
    parser.add_argument('--student-max-tokens', type=int, default=2048)

    args = parser.parse_args()

    mentor_lengths = [int(x) for x in args.lengths.split(',')]

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    difficulty_levels = [d.strip() for d in args.difficulty.split(',')]
    logger.info(f"Filtering for difficulty: {difficulty_levels}")

    try:
        dataset = load_dataset(args.dataset, split="test")
    except:
        dataset = load_dataset(args.dataset, split="train")

    # Filter by difficulty if MATH dataset
    if 'level' in dataset.column_names:
        dataset = dataset.filter(lambda x: x['level'] in difficulty_levels)
        logger.info(f"Filtered to {len(dataset)} samples with difficulty {difficulty_levels}")

    # Initialize tester
    tester = EntropyIndicatorTester(
        mentor_model_name=args.mentor_model,
        student_model_name=args.student_model,
    )

    # Test samples
    results = []
    num_samples = min(args.num_samples, len(dataset))

    for i in tqdm(range(num_samples), desc="Testing samples"):
        sample = dataset[i]

        # Extract problem and answer (handle different dataset formats)
        if "problem" in sample:
            problem = sample["problem"]
        elif "question" in sample:
            problem = sample["question"]
        else:
            problem = str(sample)

        # For MATH dataset, answer is in 'solution' field with \boxed{}
        if "solution" in sample:
            ground_truth = extract_boxed_content(sample["solution"])
        elif "answer" in sample:
            ground_truth = str(sample["answer"])
        else:
            ground_truth = ""

        # Get difficulty level if available
        level = sample.get("level", "unknown")

        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {i+1}/{num_samples} [{level}]")
        logger.info(f"Problem: {problem[:100]}...")
        logger.info(f"Ground truth: {ground_truth}")

        result = tester.evaluate_sample(
            problem=problem,
            ground_truth=ground_truth,
            mentor_lengths=mentor_lengths,
            student_max_tokens=args.student_max_tokens,
        )
        result.problem_id = i
        results.append(result)

        # Print per-length results
        for length, lr in result.length_results.items():
            status = "✓" if lr["is_correct"] else "✗"
            logger.info(f"  Length {length}: {status} (pred={lr['predicted_boxed']})")

    # Compute correlations
    correlations = compute_correlations(results)

    logger.info("\n" + "="*60)
    logger.info("CORRELATION WITH CORRECTNESS")
    logger.info("="*60)
    for key, corr in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        logger.info(f"  {key}: {corr:.4f}")

    # Compute accuracy by length
    logger.info("\n" + "="*60)
    logger.info("ACCURACY BY MENTOR LENGTH")
    logger.info("="*60)
    for length in mentor_lengths:
        correct = sum(1 for r in results if r.length_results[length]["is_correct"])
        acc = correct / len(results) * 100
        logger.info(f"  Length {length}: {acc:.1f}% ({correct}/{len(results)})")

    # Save results
    output_path = os.path.join(script_dir, args.output_file)
    save_data = {
        "config": {
            "mentor_model": args.mentor_model,
            "student_model": args.student_model,
            "dataset": args.dataset,
            "num_samples": num_samples,
            "mentor_lengths": mentor_lengths,
        },
        "correlations": correlations,
        "results": [
            {
                "problem_id": r.problem_id,
                "ground_truth": r.ground_truth,
                "length_results": {
                    str(k): {
                        "is_correct": v["is_correct"],
                        "predicted_boxed": v["predicted_boxed"],
                        "mentor_text": v["mentor_text"],
                        "ind1": v["ind1_entropy_change"],
                        "ind2": v["ind2_convergence"],
                        "ind3": v["ind3_consistency"],
                    }
                    for k, v in r.length_results.items()
                }
            }
            for r in results
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
