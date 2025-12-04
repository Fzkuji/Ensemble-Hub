#!/usr/bin/env python3
"""
Collect Progressive Data for ACT-E Experiments

Collects PPL/Entropy sequences for different mentor token lengths:
- 0 (intern only)
- 100 tokens
- 500 tokens
- 1000 tokens

Supports both local models (DeepSeek) and API models (GPT via OpenRouter).
Uses the existing TokenAnalyzer class for PPL/Entropy computation.
"""

import argparse
import json
import logging
import os
import re
import sys
import signal
import contextlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Add current directory for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from openrouter_client import OpenRouterClient
from complete_token_analysis import TokenAnalyzer
from grader import grade_answer  # Use official grader for math answer checking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result for a single sample."""
    question: str
    ground_truth: str
    mentor_tokens: int
    mentor_response: str
    intern_response: str
    intern_ppl_sequence: List[float]
    intern_entropy_sequence: List[float]
    is_correct: bool
    mentor_length: int  # Actual mentor token count
    intern_length: int  # Actual intern token count


class LocalMentorModel:
    """Local mentor model using transformers."""

    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading mentor model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int) -> Tuple[str, int]:
        """Generate response with max_tokens limit."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        token_count = len(outputs[0]) - inputs['input_ids'].shape[1]

        return response, token_count


class InternModel:
    """Intern model for evaluation and PPL/Entropy computation.

    Uses the existing TokenAnalyzer class for PPL/Entropy computation.
    """

    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading intern model: {model_name}")

        # Use TokenAnalyzer for PPL/Entropy computation
        self.analyzer = TokenAnalyzer(model_name=model_name, device=self.device)
        self.tokenizer = self.analyzer.tokenizer
        self.model = self.analyzer.model

    def compute_ppl_entropy_for_context(
        self,
        question: str,
        mentor_response: str,
        max_tokens: int = 1000,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute PPL and Entropy for mentor_response tokens given question as context.

        Uses TokenAnalyzer.compute_answer_token_metrics for accurate computation.

        Returns:
            ppl_sequence: PPL for each mentor token
            entropy_sequence: Entropy for each mentor token
        """
        if not mentor_response:
            return [], []

        # Use TokenAnalyzer's compute_answer_token_metrics
        tokens, ppls, entropies = self.analyzer.compute_answer_token_metrics(
            problem_text=question,
            answer_text=mentor_response,
            n_tokens=max_tokens,
            entropy_method='full'
        )

        return ppls, entropies

    def generate(self, prompt: str, max_tokens: int = 4096) -> Tuple[str, int]:
        """Generate response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        token_count = len(outputs[0]) - inputs['input_ids'].shape[1]

        return response, token_count


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{}."""
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


def check_math_correctness(prediction: str, ground_truth: str) -> bool:
    """Check if math answer is correct using the official grader."""
    pred_answer = extract_boxed_answer(prediction)
    true_answer = extract_boxed_answer(ground_truth)

    if not pred_answer or not true_answer:
        return False

    # Use the official grader with sympy-based math equivalence checking
    return grade_answer(pred_answer, true_answer)


# ============ HumanEval Code Execution ============

class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager for timeout."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def extract_code_from_response(response: str) -> str:
    """Extract code from model response.

    Tries multiple patterns:
    1. Code in ```python ... ``` blocks
    2. Code in ``` ... ``` blocks
    3. Raw code (if no blocks found)
    """
    # Try to find code in ```python blocks
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return last match (usually the final answer)

    # Try to find code in ``` blocks
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # If no code blocks, try to extract function body
    # Look for lines that look like code (indented or starting with keywords)
    lines = response.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        # Skip thinking/reasoning lines
        if line.strip().startswith(('#', '//', '**', '-', '*')) and not line.strip().startswith('# '):
            continue
        # Detect code start
        if line.startswith('    ') or line.startswith('\t') or \
           line.strip().startswith(('def ', 'if ', 'for ', 'while ', 'return ', 'class ')):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    return response.strip()


def check_code_correctness(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout: float = 5.0
) -> bool:
    """Check if generated code passes the test cases.

    Args:
        prompt: The function signature and docstring
        completion: The model's generated code (function body)
        test: The test code with assertions
        entry_point: The function name to test
        timeout: Maximum execution time in seconds

    Returns:
        True if all tests pass, False otherwise
    """
    # Extract just the function body from completion
    code_body = extract_code_from_response(completion)

    # Build the full code: prompt + completion + test
    full_code = prompt + code_body + "\n\n" + test + f"\ncheck({entry_point})"

    try:
        with time_limit(timeout):
            exec_globals = {}
            exec(full_code, exec_globals)
        return True
    except TimeoutException:
        return False
    except AssertionError:
        return False
    except Exception as e:
        return False


class DataCollector:
    """Collect progressive data for ACT-E experiments."""

    def __init__(
        self,
        mentor_type: str = "local",  # "local" or "api"
        mentor_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        intern_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        api_key: Optional[str] = None,
        api_model: str = "gpt-4o",
        dataset_type: str = "math",  # "math" or "code"
    ):
        self.mentor_type = mentor_type
        self.dataset_type = dataset_type
        self.token_lengths = [0, 100, 500, 1000]

        # Initialize intern model (always local)
        logger.info("Initializing intern model...")
        self.intern = InternModel(intern_model)

        # Initialize mentor model
        if mentor_type == "local":
            logger.info("Initializing local mentor model...")
            self.mentor = LocalMentorModel(mentor_model)
        else:
            logger.info("Initializing API mentor model...")
            self.mentor = OpenRouterClient(api_key=api_key, default_model=api_model)

    def collect_sample(
        self,
        question: str,
        ground_truth: str,
        max_mentor_tokens: int,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> SampleResult:
        """Collect data for a single sample with specified mentor tokens."""

        # Generate mentor response
        if max_mentor_tokens == 0:
            mentor_response = ""
            mentor_length = 0
        elif self.mentor_type == "local":
            mentor_response, mentor_length = self.mentor.generate(question, max_mentor_tokens)
        else:
            response = self.mentor.generate_math_solution(question, max_tokens=max_mentor_tokens)
            mentor_response = response.text
            mentor_length = response.tokens_used

        # Compute PPL/Entropy for mentor tokens
        ppl_seq, entropy_seq = self.intern.compute_ppl_entropy_for_context(
            question, mentor_response
        )

        # Generate intern response (continuing from mentor)
        if mentor_response:
            intern_prompt = question + mentor_response
        else:
            intern_prompt = question

        intern_response, intern_length = self.intern.generate(intern_prompt)

        # Check correctness based on dataset type
        full_response = mentor_response + intern_response
        if self.dataset_type == "code" and test_code and entry_point:
            # For HumanEval: execute code and check against test cases
            is_correct = check_code_correctness(
                prompt=question,  # question contains the function signature
                completion=full_response,
                test=test_code,
                entry_point=entry_point,
            )
        else:
            # For math: use the grader
            is_correct = check_math_correctness(full_response, ground_truth)

        return SampleResult(
            question=question,
            ground_truth=ground_truth,
            mentor_tokens=max_mentor_tokens,
            mentor_response=mentor_response,
            intern_response=intern_response,
            intern_ppl_sequence=ppl_seq,
            intern_entropy_sequence=entropy_seq,
            is_correct=is_correct,
            mentor_length=mentor_length,
            intern_length=intern_length,
        )

    def collect_dataset(
        self,
        data: List[Dict],
        output_dir: str,
        dataset_name: str = "math500",
    ):
        """Collect data for entire dataset."""
        os.makedirs(output_dir, exist_ok=True)

        for token_limit in self.token_lengths:
            logger.info(f"\n=== Collecting data with {token_limit} mentor tokens ===")

            results = []
            correct_count = 0

            for item in tqdm(data, desc=f"Token limit: {token_limit}"):
                try:
                    question = item.get('input', item.get('prompt', ''))
                    ground_truth = item.get('output', item.get('canonical_solution', ''))
                    # HumanEval specific fields
                    test_code = item.get('test', None)
                    entry_point = item.get('entry_point', None)

                    result = self.collect_sample(
                        question, ground_truth, token_limit,
                        test_code=test_code, entry_point=entry_point
                    )
                    result_dict = {
                        'question': result.question,
                        'ground_truth': result.ground_truth,
                        'mentor_tokens': result.mentor_tokens,
                        'mentor_response': result.mentor_response,
                        'intern_response': result.intern_response,
                        'ppl': result.intern_ppl_sequence,
                        'entropy': result.intern_entropy_sequence,
                        'is_correct': result.is_correct,
                        'mentor_length': result.mentor_length,
                        'intern_length': result.intern_length,
                    }
                    # Add HumanEval specific fields if present
                    if test_code:
                        result_dict['test'] = test_code
                    if entry_point:
                        result_dict['entry_point'] = entry_point
                    results.append(result_dict)

                    if result.is_correct:
                        correct_count += 1

                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue

            # Save results
            output_file = os.path.join(output_dir, f"{dataset_name}_tokens{token_limit}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            accuracy = correct_count / len(results) if results else 0
            avg_mentor_len = np.mean([r['mentor_length'] for r in results]) if results else 0
            avg_intern_len = np.mean([r['intern_length'] for r in results]) if results else 0

            logger.info(f"Results saved to {output_file}")
            logger.info(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
            logger.info(f"Avg mentor length: {avg_mentor_len:.1f}, Avg intern length: {avg_intern_len:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Collect progressive data for ACT-E")
    parser.add_argument("--dataset", type=str, default="math500", choices=["math500", "humaneval"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--mentor-type", type=str, default="local", choices=["local", "api"])
    parser.add_argument("--mentor-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--intern-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--api-model", type=str, default="gpt-4o")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "acte_experiments")

    # Load data
    data_file = os.path.join(data_dir, args.dataset, f"{args.split}.json")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {data_file}")

    # Set output directory
    if args.output_dir is None:
        mentor_name = args.api_model if args.mentor_type == "api" else args.mentor_model.split('/')[-1]
        args.output_dir = os.path.join(data_dir, "collected", f"{args.dataset}_{args.split}_{mentor_name}")

    # Determine dataset type
    dataset_type = "code" if args.dataset == "humaneval" else "math"
    logger.info(f"Dataset type: {dataset_type}")

    # Initialize collector
    collector = DataCollector(
        mentor_type=args.mentor_type,
        mentor_model=args.mentor_model,
        intern_model=args.intern_model,
        api_key=args.api_key,
        api_model=args.api_model,
        dataset_type=dataset_type,
    )

    # Collect data
    collector.collect_dataset(data, args.output_dir, f"{args.dataset}_{args.split}")


if __name__ == "__main__":
    main()
