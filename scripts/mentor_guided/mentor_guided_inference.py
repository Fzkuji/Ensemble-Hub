#!/usr/bin/env python3
"""
Mentor-Guided Adaptive Inference Framework (导师引导的自适应推理框架)

核心逻辑：
大模型（导师）不再机械地写八股文（大纲），而是像导师一样，只负责攻克最难的"起步阶段"，
通过"熵减"指标验证其对小模型的实际帮助，从而实现真正的高效交接。

具体实现：
1. 大模型推理，streaming的方式去输出token
2. 这些token也连续输入小模型
3. 根据小模型对于这些输入token，输出的预测分布，来判断这些大模型的内容是否对小模型有益
4. 如果小模型之前回答不出来一个问题，但是大模型推理一段内容之后，
   我们可以预测出小模型就可以根据这段内容继续推理出正确的答案，那就说明大模型有帮助
"""

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EntropyMetrics:
    """Metrics for a single token position."""
    token_id: int
    token_text: str
    entropy: float
    top1_prob: float
    perplexity: float


@dataclass
class InferenceState:
    """State tracking for mentor-guided inference."""
    # Generated tokens so far
    mentor_tokens: List[int] = field(default_factory=list)
    mentor_text: str = ""

    # Student model's entropy measurements
    student_entropies: List[float] = field(default_factory=list)

    # Baseline entropy (student without help)
    baseline_entropy: Optional[float] = None

    # Phase tracking
    phase: str = "mentor"  # "mentor" or "student"
    total_tokens_generated: int = 0
    mentor_tokens_used: int = 0

    def get_entropy_reduction(self) -> Optional[float]:
        """Calculate entropy reduction from baseline."""
        if self.baseline_entropy is None or not self.student_entropies:
            return None
        current = self.student_entropies[-1]
        return (self.baseline_entropy - current) / self.baseline_entropy if self.baseline_entropy > 0 else 0.0


class MentorGuidedInference:
    """
    Implementation of Mentor-Guided Adaptive Inference.

    The key insight: We use entropy as a signal to determine when the student model
    can continue independently. Lower entropy = higher confidence = student can proceed.
    """

    def __init__(
        self,
        mentor_model_name: str,
        student_model_name: str,
        device: str = None,
        entropy_threshold: float = 2.0,  # Entropy below this = student can continue
        entropy_reduction_threshold: float = 0.3,  # 30% reduction = helpful
        max_mentor_tokens: int = 100,  # Max tokens mentor can provide
        window_size: int = 5,  # Window for entropy averaging
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Entropy thresholds
        self.entropy_threshold = entropy_threshold
        self.entropy_reduction_threshold = entropy_reduction_threshold
        self.max_mentor_tokens = max_mentor_tokens
        self.window_size = window_size

        # Load models
        logger.info(f"Loading mentor model: {mentor_model_name}")
        self.mentor_tokenizer = AutoTokenizer.from_pretrained(mentor_model_name)
        self.mentor_model = self._load_model(mentor_model_name)

        logger.info(f"Loading student model: {student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.student_model = self._load_model(student_model_name)

        # Ensure pad tokens are set
        for tokenizer in [self.mentor_tokenizer, self.student_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    def _load_model(self, model_name: str):
        """Load a model with appropriate settings."""
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
            logger.warning(f"Error loading model with default settings: {e}")
            logger.info("Trying with 8-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        model.eval()
        return model

    def calculate_entropy_from_logits(self, logits: torch.Tensor, temperature: float = 1.0) -> float:
        """
        Calculate Shannon entropy from logits.

        H = -Σ(p * log(p))

        Returns entropy in bits (log base 2).
        """
        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Filter very small probabilities for numerical stability
        probs_np = probs.cpu().numpy().astype(np.float64)
        probs_np = probs_np[probs_np > 1e-10]

        if len(probs_np) == 0 or np.sum(probs_np) == 0:
            return 0.0

        # Normalize
        probs_np = probs_np / probs_np.sum()

        # Calculate entropy (natural log then convert to bits)
        log_probs = np.log(probs_np)
        entropy = -np.sum(probs_np * log_probs)
        entropy_bits = entropy / np.log(2)

        if np.isnan(entropy_bits) or np.isinf(entropy_bits) or entropy_bits < 0:
            return 0.0

        return float(entropy_bits)

    def get_student_entropy_for_next_token(
        self,
        prompt_text: str,
        context_text: str = ""
    ) -> Tuple[float, float, int]:
        """
        Get the student model's entropy for predicting the next token.

        Args:
            prompt_text: The original prompt/question
            context_text: Additional context (e.g., mentor's tokens)

        Returns:
            Tuple of (entropy, top1_probability, predicted_token_id)
        """
        # Combine prompt and context
        full_text = prompt_text
        if context_text:
            full_text = prompt_text + context_text

        # Tokenize
        inputs = self.student_tokenizer(full_text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.student_model(input_ids)
            # Get logits for the last position (predicting next token)
            last_logits = outputs.logits[0, -1, :]

        # Calculate entropy
        entropy = self.calculate_entropy_from_logits(last_logits)

        # Get top-1 probability and token
        probs = F.softmax(last_logits, dim=-1)
        top1_prob, top1_token = torch.max(probs, dim=-1)

        return entropy, top1_prob.item(), top1_token.item()

    def generate_mentor_token(
        self,
        prompt_text: str,
        context_text: str = "",
        temperature: float = 0.7,
    ) -> Tuple[int, str]:
        """
        Generate one token from the mentor model.

        Returns:
            Tuple of (token_id, token_text)
        """
        # Combine prompt and context
        full_text = prompt_text
        if context_text:
            full_text = prompt_text + context_text

        # Tokenize
        inputs = self.mentor_tokenizer(full_text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        # Generate one token
        with torch.no_grad():
            outputs = self.mentor_model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.mentor_tokenizer.pad_token_id,
            )

        # Extract the generated token
        new_token_id = outputs[0, -1].item()
        new_token_text = self.mentor_tokenizer.decode([new_token_id])

        return new_token_id, new_token_text

    def should_switch_to_student(self, state: InferenceState) -> Tuple[bool, str]:
        """
        Determine if the student model can continue independently.

        Returns:
            Tuple of (should_switch, reason)
        """
        if not state.student_entropies:
            return False, "no_entropy_data"

        current_entropy = state.student_entropies[-1]

        # Check 1: Absolute entropy threshold
        if current_entropy < self.entropy_threshold:
            return True, f"entropy_below_threshold ({current_entropy:.4f} < {self.entropy_threshold})"

        # Check 2: Entropy reduction from baseline
        reduction = state.get_entropy_reduction()
        if reduction is not None and reduction > self.entropy_reduction_threshold:
            return True, f"sufficient_reduction ({reduction:.2%} > {self.entropy_reduction_threshold:.2%})"

        # Check 3: Sliding window average entropy
        if len(state.student_entropies) >= self.window_size:
            window_avg = np.mean(state.student_entropies[-self.window_size:])
            if window_avg < self.entropy_threshold * 1.2:
                # Also check for downward trend
                if len(state.student_entropies) >= 2 * self.window_size:
                    prev_window = state.student_entropies[-2*self.window_size:-self.window_size]
                    curr_window = state.student_entropies[-self.window_size:]
                    if np.mean(curr_window) < np.mean(prev_window):
                        return True, f"downward_trend (avg={window_avg:.4f})"

        # Check 4: Max mentor tokens reached
        if state.mentor_tokens_used >= self.max_mentor_tokens:
            return True, f"max_mentor_tokens_reached ({state.mentor_tokens_used})"

        return False, "continue_mentor_assistance"

    def run_adaptive_inference(
        self,
        prompt: str,
        max_total_tokens: int = 512,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full mentor-guided adaptive inference.

        Args:
            prompt: The input prompt/question
            max_total_tokens: Maximum total tokens to generate
            temperature: Sampling temperature
            verbose: Whether to print progress

        Returns:
            Dictionary containing:
            - generated_text: The full generated text
            - mentor_text: Text generated by mentor
            - student_text: Text generated by student
            - mentor_tokens_used: Number of mentor tokens used
            - entropy_history: List of entropy values
            - switch_reason: Why we switched to student
        """
        state = InferenceState()

        # Step 1: Get baseline entropy (student without any help)
        baseline_entropy, _, _ = self.get_student_entropy_for_next_token(prompt)
        state.baseline_entropy = baseline_entropy

        if verbose:
            logger.info(f"Baseline student entropy: {baseline_entropy:.4f}")

        # Step 2: Mentor generation phase
        mentor_context = ""
        switch_reason = "max_tokens_reached"

        if verbose:
            logger.info("Starting mentor generation phase...")

        pbar = tqdm(range(self.max_mentor_tokens), desc="Mentor tokens", disable=not verbose)
        for i in pbar:
            # Generate one token from mentor
            token_id, token_text = self.generate_mentor_token(prompt, mentor_context, temperature)

            # Check for EOS
            if token_id == self.mentor_tokenizer.eos_token_id:
                switch_reason = "mentor_eos"
                break

            # Update mentor context
            mentor_context += token_text
            state.mentor_tokens.append(token_id)
            state.mentor_tokens_used += 1

            # Measure student's entropy with the new context
            entropy, top1_prob, _ = self.get_student_entropy_for_next_token(prompt, mentor_context)
            state.student_entropies.append(entropy)

            # Update progress bar
            reduction = state.get_entropy_reduction()
            reduction_str = f"{reduction:.1%}" if reduction else "N/A"
            pbar.set_postfix({
                'entropy': f'{entropy:.2f}',
                'reduction': reduction_str,
                'top1_p': f'{top1_prob:.2f}'
            })

            # Check if we should switch to student
            should_switch, reason = self.should_switch_to_student(state)
            if should_switch:
                switch_reason = reason
                break

        state.mentor_text = mentor_context

        if verbose:
            logger.info(f"Mentor phase complete. Reason: {switch_reason}")
            logger.info(f"Mentor tokens used: {state.mentor_tokens_used}")
            logger.info(f"Mentor text: {mentor_context[:200]}..." if len(mentor_context) > 200 else f"Mentor text: {mentor_context}")

        # Step 3: Student generation phase
        student_context = ""
        remaining_tokens = max_total_tokens - state.mentor_tokens_used

        if verbose:
            logger.info(f"Starting student generation phase (max {remaining_tokens} tokens)...")

        # Prepare full prompt with mentor context for student
        full_prompt = prompt + mentor_context

        # Generate with student model
        inputs = self.student_tokenizer(full_prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.student_model.generate(
                input_ids,
                max_new_tokens=remaining_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.student_tokenizer.pad_token_id,
            )

        # Extract student's generated text
        student_output_ids = outputs[0, input_ids.shape[1]:]
        student_text = self.student_tokenizer.decode(student_output_ids, skip_special_tokens=True)

        state.total_tokens_generated = state.mentor_tokens_used + len(student_output_ids)

        # Compile results
        result = {
            "generated_text": mentor_context + student_text,
            "mentor_text": mentor_context,
            "student_text": student_text,
            "mentor_tokens_used": state.mentor_tokens_used,
            "total_tokens": state.total_tokens_generated,
            "baseline_entropy": state.baseline_entropy,
            "final_entropy": state.student_entropies[-1] if state.student_entropies else None,
            "entropy_reduction": state.get_entropy_reduction(),
            "entropy_history": state.student_entropies,
            "switch_reason": switch_reason,
        }

        if verbose:
            logger.info(f"Total tokens generated: {state.total_tokens_generated}")
            if result["entropy_reduction"]:
                logger.info(f"Entropy reduction achieved: {result['entropy_reduction']:.2%}")

        return result


def main():
    parser = argparse.ArgumentParser(description='Mentor-Guided Adaptive Inference')

    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-7B-Instruct',
                       help='Mentor (large) model name')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Student (small) model name')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Input prompt for generation')
    parser.add_argument('--entropy-threshold', type=float, default=2.0,
                       help='Entropy threshold for switching to student')
    parser.add_argument('--reduction-threshold', type=float, default=0.3,
                       help='Minimum entropy reduction to consider helpful')
    parser.add_argument('--max-mentor-tokens', type=int, default=100,
                       help='Maximum tokens from mentor')
    parser.add_argument('--max-total-tokens', type=int, default=512,
                       help='Maximum total tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')

    args = parser.parse_args()

    # Default test prompt if none provided
    if args.prompt is None:
        args.prompt = """Solve the following math problem step by step:

Problem: Find all real numbers x such that x^2 - 5x + 6 = 0.

Solution:"""

    # Initialize the inference system
    inference = MentorGuidedInference(
        mentor_model_name=args.mentor_model,
        student_model_name=args.student_model,
        entropy_threshold=args.entropy_threshold,
        entropy_reduction_threshold=args.reduction_threshold,
        max_mentor_tokens=args.max_mentor_tokens,
    )

    # Run inference
    logger.info("=" * 60)
    logger.info("MENTOR-GUIDED ADAPTIVE INFERENCE")
    logger.info("=" * 60)
    logger.info(f"Prompt: {args.prompt[:200]}..." if len(args.prompt) > 200 else f"Prompt: {args.prompt}")
    logger.info("=" * 60)

    result = inference.run_adaptive_inference(
        prompt=args.prompt,
        max_total_tokens=args.max_total_tokens,
        temperature=args.temperature,
        verbose=True,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mentor tokens used: {result['mentor_tokens_used']}")
    logger.info(f"Total tokens: {result['total_tokens']}")
    logger.info(f"Switch reason: {result['switch_reason']}")
    logger.info(f"Baseline entropy: {result['baseline_entropy']:.4f}")
    if result['final_entropy']:
        logger.info(f"Final entropy: {result['final_entropy']:.4f}")
    if result['entropy_reduction']:
        logger.info(f"Entropy reduction: {result['entropy_reduction']:.2%}")
    logger.info("-" * 60)
    logger.info("Generated text:")
    logger.info(result['generated_text'])
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    main()
