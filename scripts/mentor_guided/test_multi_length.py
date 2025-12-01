#!/usr/bin/env python3
"""
Test Mentor-Guided Inference with Different Mentor Lengths.

核心测试逻辑：
1. 大模型推理不同长度的token（如 0, 10, 20, 50, 100, 200 tokens）
2. 小模型在每个长度后接续推理
3. 比较不同长度下：
   - 小模型接续时的初始熵
   - 小模型生成答案的质量
   - 总体效果

这样可以找到最优的"切换点"——即大模型需要提供多少帮助才足够。
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
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LengthTestResult:
    """Results for a single length test."""
    mentor_length: int
    mentor_text: str
    student_text: str
    full_output: str
    # Entropy metrics
    student_initial_entropy: float  # Student's entropy when starting to continue
    student_entropy_trajectory: List[float] = field(default_factory=list)
    # Quality metrics (can be extended)
    output_length: int = 0


class MultiLengthTester:
    """
    Test mentor-guided inference with different mentor lengths.
    """

    def __init__(
        self,
        mentor_model_name: str,
        student_model_name: str,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load mentor model
        logger.info(f"Loading mentor model: {mentor_model_name}")
        self.mentor_tokenizer = AutoTokenizer.from_pretrained(mentor_model_name, trust_remote_code=True)
        self.mentor_model = self._load_model(mentor_model_name)
        self.mentor_model_name = mentor_model_name

        # Load student model
        logger.info(f"Loading student model: {student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
        self.student_model = self._load_model(student_model_name)
        self.student_model_name = student_model_name

        # Ensure pad tokens
        for tokenizer in [self.mentor_tokenizer, self.student_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    def _load_model(self, model_name: str):
        """Load model with appropriate settings."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            if not torch.cuda.is_available():
                model = model.to(self.device)
        except Exception as e:
            logger.warning(f"Error loading with default settings: {e}")
            logger.info("Trying with 8-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        model.eval()
        return model

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculate Shannon entropy from logits (in bits)."""
        probs = F.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy().astype(np.float64)
        probs_np = probs_np[probs_np > 1e-10]

        if len(probs_np) == 0 or np.sum(probs_np) == 0:
            return 0.0

        probs_np = probs_np / probs_np.sum()
        log_probs = np.log(probs_np)
        entropy = -np.sum(probs_np * log_probs)
        entropy_bits = entropy / np.log(2)

        if np.isnan(entropy_bits) or np.isinf(entropy_bits) or entropy_bits < 0:
            return 0.0

        return float(entropy_bits)

    def generate_mentor_tokens(
        self,
        prompt: str,
        num_tokens: int,
        temperature: float = 0.7,
    ) -> Tuple[str, List[int]]:
        """Generate specific number of tokens from mentor."""
        if num_tokens == 0:
            return "", []

        inputs = self.mentor_tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.mentor_model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.mentor_tokenizer.pad_token_id,
            )

        new_token_ids = outputs[0, input_ids.shape[1]:].tolist()
        new_text = self.mentor_tokenizer.decode(new_token_ids, skip_special_tokens=True)

        return new_text, new_token_ids

    def get_student_entropy_at_position(self, full_text: str) -> Tuple[float, float, int]:
        """
        Get student's entropy for predicting the next token after full_text.

        Returns: (entropy, top1_prob, top1_token_id)
        """
        inputs = self.student_tokenizer(full_text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.student_model(input_ids)
            last_logits = outputs.logits[0, -1, :]

        entropy = self.calculate_entropy(last_logits)
        probs = F.softmax(last_logits, dim=-1)
        top1_prob, top1_token = torch.max(probs, dim=-1)

        return entropy, top1_prob.item(), top1_token.item()

    def generate_student_continuation(
        self,
        prompt_with_mentor: str,
        max_tokens: int,
        temperature: float = 0.7,
        track_entropy: bool = True,
    ) -> Tuple[str, List[float]]:
        """
        Generate continuation from student model.

        Returns: (generated_text, entropy_trajectory)
        """
        inputs = self.student_tokenizer(prompt_with_mentor, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        entropy_trajectory = []

        if track_entropy:
            # Generate token by token to track entropy
            generated_ids = []
            current_ids = input_ids

            for _ in range(max_tokens):
                with torch.no_grad():
                    outputs = self.student_model(current_ids)
                    last_logits = outputs.logits[0, -1, :]

                entropy = self.calculate_entropy(last_logits)
                entropy_trajectory.append(entropy)

                # Sample next token
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for EOS
                if next_token.item() == self.student_tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

            generated_text = self.student_tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            # Fast generation without tracking
            with torch.no_grad():
                outputs = self.student_model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.student_tokenizer.pad_token_id,
                )

            generated_ids = outputs[0, input_ids.shape[1]:]
            generated_text = self.student_tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, entropy_trajectory

    def test_single_length(
        self,
        prompt: str,
        mentor_length: int,
        student_max_tokens: int,
        temperature: float = 0.7,
        track_entropy: bool = True,
    ) -> LengthTestResult:
        """Test a single mentor length configuration."""

        # Generate mentor tokens
        mentor_text, mentor_ids = self.generate_mentor_tokens(prompt, mentor_length, temperature)

        # Create prompt with mentor context
        full_prompt = prompt + mentor_text

        # Get student's initial entropy (before generating)
        initial_entropy, initial_prob, _ = self.get_student_entropy_at_position(full_prompt)

        # Generate student continuation
        student_text, entropy_traj = self.generate_student_continuation(
            full_prompt,
            student_max_tokens,
            temperature,
            track_entropy,
        )

        return LengthTestResult(
            mentor_length=mentor_length,
            mentor_text=mentor_text,
            student_text=student_text,
            full_output=mentor_text + student_text,
            student_initial_entropy=initial_entropy,
            student_entropy_trajectory=entropy_traj,
            output_length=len(mentor_text) + len(student_text),
        )

    def run_multi_length_test(
        self,
        prompt: str,
        mentor_lengths: List[int],
        student_max_tokens: int = 200,
        temperature: float = 0.7,
        track_entropy: bool = True,
    ) -> Dict[str, Any]:
        """
        Run tests with multiple mentor lengths.

        Returns comprehensive results for analysis.
        """
        results = []
        baseline_entropy = None

        for length in tqdm(mentor_lengths, desc="Testing mentor lengths"):
            result = self.test_single_length(
                prompt, length, student_max_tokens, temperature, track_entropy
            )
            results.append(result)

            # First result (length=0) is baseline
            if length == 0:
                baseline_entropy = result.student_initial_entropy

            logger.info(f"Length {length}: initial_entropy={result.student_initial_entropy:.4f}")

        # Calculate entropy reduction for each length
        analysis = {
            "prompt": prompt,
            "mentor_lengths": mentor_lengths,
            "baseline_entropy": baseline_entropy,
            "results": [],
            "summary": {},
        }

        for result in results:
            reduction = None
            if baseline_entropy and baseline_entropy > 0:
                reduction = (baseline_entropy - result.student_initial_entropy) / baseline_entropy

            result_dict = {
                "mentor_length": result.mentor_length,
                "mentor_text": result.mentor_text,
                "student_text": result.student_text[:500] + "..." if len(result.student_text) > 500 else result.student_text,
                "full_output_length": result.output_length,
                "student_initial_entropy": result.student_initial_entropy,
                "entropy_reduction": reduction,
                "entropy_trajectory_stats": {
                    "mean": float(np.mean(result.student_entropy_trajectory)) if result.student_entropy_trajectory else None,
                    "std": float(np.std(result.student_entropy_trajectory)) if result.student_entropy_trajectory else None,
                    "min": float(min(result.student_entropy_trajectory)) if result.student_entropy_trajectory else None,
                    "max": float(max(result.student_entropy_trajectory)) if result.student_entropy_trajectory else None,
                }
            }
            analysis["results"].append(result_dict)

        # Summary statistics
        initial_entropies = [r.student_initial_entropy for r in results]
        analysis["summary"] = {
            "baseline_entropy": baseline_entropy,
            "min_entropy_achieved": min(initial_entropies),
            "best_length": mentor_lengths[np.argmin(initial_entropies)],
            "entropy_by_length": dict(zip(mentor_lengths, initial_entropies)),
        }

        return analysis


def main():
    parser = argparse.ArgumentParser(description='Test Multi-Length Mentor-Guided Inference')

    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-1.5B-Instruct',
                       help='Mentor (larger) model')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Student (smaller) model')
    parser.add_argument('--lengths', type=str, default='0,10,20,50,100,200',
                       help='Comma-separated mentor token lengths to test')
    parser.add_argument('--student-max-tokens', type=int, default=200,
                       help='Max tokens for student to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--output-file', default='multi_length_results.json',
                       help='Output file')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt')
    parser.add_argument('--no-track-entropy', action='store_true',
                       help='Disable per-token entropy tracking (faster)')

    args = parser.parse_args()

    # Parse lengths
    mentor_lengths = [int(x) for x in args.lengths.split(',')]
    logger.info(f"Testing mentor lengths: {mentor_lengths}")

    # Default prompt
    if args.prompt is None:
        args.prompt = """Solve the following math problem step by step:

Problem: Find all real solutions to the equation x^2 - 5x + 6 = 0.

Solution:"""

    # Initialize tester
    tester = MultiLengthTester(
        mentor_model_name=args.mentor_model,
        student_model_name=args.student_model,
    )

    # Run test
    logger.info("=" * 60)
    logger.info("MULTI-LENGTH MENTOR-GUIDED INFERENCE TEST")
    logger.info("=" * 60)
    logger.info(f"Prompt: {args.prompt[:200]}...")
    logger.info("=" * 60)

    results = tester.run_multi_length_test(
        prompt=args.prompt,
        mentor_lengths=mentor_lengths,
        student_max_tokens=args.student_max_tokens,
        temperature=args.temperature,
        track_entropy=not args.no_track_entropy,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Baseline entropy (no mentor help): {results['summary']['baseline_entropy']:.4f}")
    logger.info(f"Best length: {results['summary']['best_length']} tokens")
    logger.info(f"Min entropy achieved: {results['summary']['min_entropy_achieved']:.4f}")

    logger.info("\nEntropy by mentor length:")
    for length, entropy in results['summary']['entropy_by_length'].items():
        reduction = (results['summary']['baseline_entropy'] - entropy) / results['summary']['baseline_entropy'] * 100
        logger.info(f"  {length:4d} tokens: entropy={entropy:.4f}, reduction={reduction:+.1f}%")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.output_file)

    # Add config to results
    results["config"] = {
        "mentor_model": args.mentor_model,
        "student_model": args.student_model,
        "mentor_lengths": mentor_lengths,
        "student_max_tokens": args.student_max_tokens,
        "temperature": args.temperature,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_path}")

    # Print sample outputs
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE OUTPUTS")
    logger.info("=" * 60)

    for r in results["results"]:
        logger.info(f"\n--- Mentor length: {r['mentor_length']} tokens ---")
        logger.info(f"Initial entropy: {r['student_initial_entropy']:.4f}")
        if r['entropy_reduction'] is not None:
            logger.info(f"Entropy reduction: {r['entropy_reduction']*100:+.1f}%")
        logger.info(f"Mentor: {r['mentor_text'][:200]}..." if len(r['mentor_text']) > 200 else f"Mentor: {r['mentor_text']}")
        logger.info(f"Student: {r['student_text'][:200]}..." if len(r['student_text']) > 200 else f"Student: {r['student_text']}")


if __name__ == "__main__":
    main()
