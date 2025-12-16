#!/usr/bin/env python3
"""
Collect Progressive Data for ACT-E Experiments

Collects PPL/Entropy sequences for different mentor token lengths:
- -1 (mentor only - mentor generates complete response)
- 0 (intern only)
- 100 tokens
- 500 tokens
- 1000 tokens

Supports both local models (DeepSeek) and API models (GPT via OpenRouter).
Supports data parallelism with multiple GPU workers for efficient processing.
"""

import argparse
import json
import logging
import os
import re
import sys
import signal
import contextlib
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy as scipy_entropy

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Add current directory for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from openrouter_client import OpenRouterClient
from grader import grade_answer  # Use official grader for math answer checking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_prompt(item: Dict[str, Any]) -> str:
    """Build full prompt from data item.

    For MATH dataset: combines 'instruction' + 'input'
    For HumanEval: uses 'prompt' directly
    """
    # Check if it's a MATH-style item with instruction
    if 'instruction' in item and 'input' in item:
        instruction = item['instruction']
        input_text = item['input']
        return f"{instruction}\n\n{input_text}"
    # Fallback for other formats
    return item.get('input', item.get('prompt', ''))


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
        """
        Args:
            model_name: HuggingFace model name
            device: Specific device like "cuda:0", "cuda:1", etc.
                    If None, uses device_map="auto" (distributed across all GPUs)
        """
        self.device = device
        logger.info(f"Loading mentor model: {model_name} on {device or 'auto'}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if device and device.startswith("cuda:"):
            # Load on specific GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(device)
        else:
            # Auto distribute across GPUs
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.device = "cuda"
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

    Implements its own PPL/Entropy computation with proper device support.
    """

    def __init__(self, model_name: str, device: str = None):
        """
        Args:
            model_name: HuggingFace model name
            device: Specific device like "cuda:0", "cuda:1", etc.
                    If None, uses device_map="auto" (distributed across all GPUs)
        """
        self.model_name = model_name
        logger.info(f"Loading intern model: {model_name} on {device or 'auto'}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if device and device.startswith("cuda:"):
            # Load on specific GPU
            self.device = device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(device)
        else:
            # Auto distribute across GPUs
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.device = "cuda"
        self.model.eval()

    def compute_ppl_entropy_for_context(
        self,
        question: str,
        mentor_response: str,
        max_tokens: int = 1000,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute PPL and Entropy for mentor_response tokens given question as context.

        Returns:
            ppl_sequence: PPL for each mentor token
            entropy_sequence: Entropy for each mentor token
        """
        if not mentor_response:
            return [], []

        try:
            # Create full text: question + mentor_response
            full_text = question + mentor_response

            # Tokenize full text and question separately
            full_inputs = self.tokenizer(full_text, return_tensors="pt", truncation=False)
            question_inputs = self.tokenizer(question, return_tensors="pt", truncation=False)

            full_ids = full_inputs["input_ids"].to(self.device)
            question_len = question_inputs["input_ids"].shape[1]

            # Number of answer tokens
            answer_len = full_ids.shape[1] - question_len
            if answer_len <= 0:
                return [], []

            # Limit to max_tokens
            n_tokens = min(answer_len, max_tokens)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits

            ppls = []
            entropies = []

            for i in range(n_tokens):
                # Position in full sequence: question_len - 1 + i (predict position question_len + i)
                pos = question_len - 1 + i
                if pos >= logits.shape[1]:
                    break

                current_logits = logits[0, pos, :]
                probs = torch.softmax(current_logits, dim=-1)

                # Actual next token
                actual_token = full_ids[0, question_len + i].item()

                # Perplexity: 1 / P(token)
                token_prob = probs[actual_token].item()
                ppl = 1.0 / token_prob if token_prob > 0 else 1000.0
                ppl = min(ppl, 1000.0)  # Cap at 1000

                # Entropy calculation
                probs_np = probs.cpu().numpy().astype(np.float64)
                probs_np = probs_np[probs_np > 1e-10]  # Filter small values

                if len(probs_np) == 0 or np.sum(probs_np) == 0:
                    token_entropy = 0.0
                else:
                    probs_np = probs_np / probs_np.sum()  # Normalize
                    log_probs = np.log(probs_np)
                    token_entropy = -np.sum(probs_np * log_probs)
                    token_entropy = token_entropy / np.log(2)  # Convert to bits

                    if np.isnan(token_entropy) or np.isinf(token_entropy) or token_entropy < 0:
                        token_entropy = 0.0

                ppls.append(ppl)
                entropies.append(token_entropy)

            return ppls, entropies

        except Exception as e:
            logger.error(f"Error computing PPL/Entropy: {e}")
            return [], []

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


def worker_process(
    worker_id: int,
    mentor_device: str,
    intern_device: str,
    mentor_type: str,
    mentor_model: str,
    intern_model: str,
    api_key: Optional[str],
    api_model: str,
    dataset_type: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """Worker process for parallel data collection.

    Each worker loads its own model instances on specified GPUs and processes samples.
    The device strings should be like "cuda:0", "cuda:1", etc.
    """
    # Configure logging for this worker
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - Worker {worker_id} - %(levelname)s - %(message)s'
    )

    logger.info(f"Worker {worker_id}: Initializing on mentor={mentor_device}, intern={intern_device}")

    # Initialize models for this worker
    if mentor_type == "local":
        mentor = LocalMentorModel(mentor_model, device=mentor_device)
    else:
        mentor = OpenRouterClient(api_key=api_key, default_model=api_model)

    intern = InternModel(intern_model, device=intern_device)

    logger.info(f"Worker {worker_id}: Models loaded, ready to process")

    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # Poison pill
                logger.info(f"Worker {worker_id}: Received shutdown signal")
                break

            item, token_limit = task

            question = build_prompt(item)
            ground_truth = item.get('output', item.get('canonical_solution', ''))
            test_code = item.get('test', None)
            entry_point = item.get('entry_point', None)

            # Process sample
            result = collect_single_sample(
                mentor=mentor,
                intern=intern,
                mentor_type=mentor_type,
                dataset_type=dataset_type,
                question=question,
                ground_truth=ground_truth,
                max_mentor_tokens=token_limit,
                test_code=test_code,
                entry_point=entry_point,
            )

            result_queue.put((item, token_limit, result))

        except Exception as e:
            if "timeout" not in str(e).lower():
                logger.warning(f"Worker {worker_id}: Error - {e}")
            continue

    logger.info(f"Worker {worker_id}: Shutting down")


def collect_single_sample(
    mentor,
    intern,
    mentor_type: str,
    dataset_type: str,
    question: str,
    ground_truth: str,
    max_mentor_tokens: int,
    test_code: Optional[str] = None,
    entry_point: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect data for a single sample.

    This function is used by both sequential and parallel modes.
    """
    # Handle mentor-only mode
    if max_mentor_tokens == -1:
        if mentor_type == "local":
            mentor_response, mentor_length = mentor.generate(question, 4096)
        else:
            response = mentor.generate_math_solution(question, max_tokens=4096)
            mentor_response = response.text
            mentor_length = response.tokens_used

        intern_response = ""
        intern_length = 0
        ppl_seq, entropy_seq = [], []
        full_response = mentor_response

    # Handle intern-only mode
    elif max_mentor_tokens == 0:
        mentor_response = ""
        mentor_length = 0
        intern_response, intern_length = intern.generate(question)
        ppl_seq, entropy_seq = [], []
        full_response = intern_response

    # Handle progressive mode
    else:
        if mentor_type == "local":
            mentor_response, mentor_length = mentor.generate(question, max_mentor_tokens)
        else:
            response = mentor.generate_math_solution(question, max_tokens=max_mentor_tokens)
            mentor_response = response.text
            mentor_length = response.tokens_used

        ppl_seq, entropy_seq = intern.compute_ppl_entropy_for_context(
            question, mentor_response
        )

        intern_prompt = question + mentor_response
        intern_response, intern_length = intern.generate(intern_prompt)
        full_response = mentor_response + intern_response

    # Check correctness
    if dataset_type == "code" and test_code and entry_point:
        is_correct = check_code_correctness(
            prompt=question,
            completion=full_response,
            test=test_code,
            entry_point=entry_point,
        )
    else:
        is_correct = check_math_correctness(full_response, ground_truth)

    return {
        'mentor_response': mentor_response,
        'intern_response': intern_response,
        'ppl': ppl_seq,
        'entropy': entropy_seq,
        'is_correct': is_correct,
        'mentor_length': mentor_length,
        'intern_length': intern_length,
    }


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
        mentor_device: str = None,  # Specific device for mentor
        intern_device: str = None,  # Specific device for intern
    ):
        self.mentor_type = mentor_type
        self.mentor_model = mentor_model
        self.intern_model = intern_model
        self.api_key = api_key
        self.api_model = api_model
        self.dataset_type = dataset_type
        self.mentor_device = mentor_device
        self.intern_device = intern_device
        self.token_lengths = [-1, 0, 100, 500, 1000]  # -1 = mentor only, 0 = intern only

        # Initialize intern model (always local)
        logger.info("Initializing intern model...")
        self.intern = InternModel(intern_model, device=intern_device)

        # Initialize mentor model
        if mentor_type == "local":
            logger.info("Initializing local mentor model...")
            self.mentor = LocalMentorModel(mentor_model, device=mentor_device)
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
        """Collect data for a single sample with specified mentor tokens.

        Args:
            max_mentor_tokens:
                -1 = mentor only (mentor generates complete response, no intern)
                 0 = intern only (no mentor, intern generates complete response)
                >0 = progressive (mentor generates prefix, intern continues)
        """
        result = collect_single_sample(
            mentor=self.mentor,
            intern=self.intern,
            mentor_type=self.mentor_type,
            dataset_type=self.dataset_type,
            question=question,
            ground_truth=ground_truth,
            max_mentor_tokens=max_mentor_tokens,
            test_code=test_code,
            entry_point=entry_point,
        )

        return SampleResult(
            question=question,
            ground_truth=ground_truth,
            mentor_tokens=max_mentor_tokens,
            mentor_response=result['mentor_response'],
            intern_response=result['intern_response'],
            intern_ppl_sequence=result['ppl'],
            intern_entropy_sequence=result['entropy'],
            is_correct=result['is_correct'],
            mentor_length=result['mentor_length'],
            intern_length=result['intern_length'],
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
            if token_limit == -1:
                mode_name = "mentor only"
            elif token_limit == 0:
                mode_name = "intern only"
            else:
                mode_name = f"{token_limit} mentor tokens"
            logger.info(f"\n=== Collecting data: {mode_name} ===")

            results = []
            correct_count = 0

            for item in tqdm(data, desc=mode_name):
                try:
                    question = build_prompt(item)
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
            if token_limit == -1:
                output_file = os.path.join(output_dir, f"{dataset_name}_mentor_only.json")
            else:
                output_file = os.path.join(output_dir, f"{dataset_name}_tokens{token_limit}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            accuracy = correct_count / len(results) if results else 0
            avg_mentor_len = np.mean([r['mentor_length'] for r in results]) if results else 0
            avg_intern_len = np.mean([r['intern_length'] for r in results]) if results else 0

            logger.info(f"Results saved to {output_file}")
            logger.info(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
            logger.info(f"Avg mentor length: {avg_mentor_len:.1f}, Avg intern length: {avg_intern_len:.1f}")


def collect_dataset_parallel(
    data: List[Dict],
    output_dir: str,
    dataset_name: str,
    mentor_type: str,
    mentor_model: str,
    intern_model: str,
    api_key: Optional[str],
    api_model: str,
    dataset_type: str,
    num_workers: int,
    mentor_gpus: List[str],
    intern_gpus: List[str],
):
    """Collect data using multiple parallel workers.

    Each worker loads its own model instance and processes samples independently.
    This achieves DDP-like data parallelism for efficient GPU utilization.

    Args:
        data: List of data items to process
        output_dir: Directory to save results
        dataset_name: Name prefix for output files
        mentor_type: "local" or "api"
        mentor_model: Model name for mentor
        intern_model: Model name for intern
        api_key: API key for API mentor
        api_model: Model name for API mentor
        dataset_type: "math" or "code"
        num_workers: Number of parallel workers
        mentor_gpus: List of GPU devices for mentor models (e.g., ["cuda:0", "cuda:1"])
        intern_gpus: List of GPU devices for intern models (e.g., ["cuda:4", "cuda:5"])
    """
    os.makedirs(output_dir, exist_ok=True)
    token_lengths = [-1, 0, 100, 500, 1000]

    # Ensure we have enough GPU assignments for workers
    if len(mentor_gpus) < num_workers:
        mentor_gpus = mentor_gpus * (num_workers // len(mentor_gpus) + 1)
    if len(intern_gpus) < num_workers:
        intern_gpus = intern_gpus * (num_workers // len(intern_gpus) + 1)

    for token_limit in token_lengths:
        if token_limit == -1:
            mode_name = "mentor only"
        elif token_limit == 0:
            mode_name = "intern only"
        else:
            mode_name = f"{token_limit} mentor tokens"
        logger.info(f"\n=== Collecting data (parallel, {num_workers} workers): {mode_name} ===")

        # Create task and result queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # Start worker processes
        workers = []
        for i in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    i,
                    mentor_gpus[i],
                    intern_gpus[i],
                    mentor_type,
                    mentor_model,
                    intern_model,
                    api_key,
                    api_model,
                    dataset_type,
                    task_queue,
                    result_queue,
                ),
            )
            p.start()
            workers.append(p)

        # Submit all tasks
        for item in data:
            task_queue.put((item, token_limit))

        # Send poison pills to stop workers
        for _ in range(num_workers):
            task_queue.put(None)

        # Collect results with progress bar
        results = []
        correct_count = 0
        pbar = tqdm(total=len(data), desc=mode_name)

        while len(results) < len(data):
            try:
                item, _, result = result_queue.get(timeout=300)  # 5 min timeout
                question = build_prompt(item)
                ground_truth = item.get('output', item.get('canonical_solution', ''))
                test_code = item.get('test', None)
                entry_point = item.get('entry_point', None)

                result_dict = {
                    'question': question,
                    'ground_truth': ground_truth,
                    'mentor_tokens': token_limit,
                    'mentor_response': result['mentor_response'],
                    'intern_response': result['intern_response'],
                    'ppl': result['ppl'],
                    'entropy': result['entropy'],
                    'is_correct': result['is_correct'],
                    'mentor_length': result['mentor_length'],
                    'intern_length': result['intern_length'],
                }
                if test_code:
                    result_dict['test'] = test_code
                if entry_point:
                    result_dict['entry_point'] = entry_point

                results.append(result_dict)
                if result['is_correct']:
                    correct_count += 1
                pbar.update(1)

            except Exception as e:
                logger.warning(f"Error collecting result: {e}")
                # Check if any workers are still alive
                alive_workers = sum(1 for p in workers if p.is_alive())
                if alive_workers == 0:
                    logger.error("All workers died, breaking out of collection loop")
                    break

        pbar.close()

        # Wait for all workers to finish
        for p in workers:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()

        # Save results
        if token_limit == -1:
            output_file = os.path.join(output_dir, f"{dataset_name}_mentor_only.json")
        else:
            output_file = os.path.join(output_dir, f"{dataset_name}_tokens{token_limit}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        accuracy = correct_count / len(results) if results else 0
        avg_mentor_len = np.mean([r['mentor_length'] for r in results]) if results else 0
        avg_intern_len = np.mean([r['intern_length'] for r in results]) if results else 0

        logger.info(f"Results saved to {output_file}")
        logger.info(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
        logger.info(f"Avg mentor length: {avg_mentor_len:.1f}, Avg intern length: {avg_intern_len:.1f}")


def load_hendrycks_math(split: str = "all") -> List[Dict[str, Any]]:
    """Load MATH data directly from HuggingFace EleutherAI/hendrycks_math.

    Args:
        split: "train", "test", or "all" (combines both)

    Returns:
        List of all problems from all 7 subsets
    """
    from datasets import load_dataset

    MATH_SUBSETS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    MATH_INSTRUCTION = """Solve the following math problem step by step. Structure your reasoning using the following framework:

1. **Goal**: Define the ultimate objective or question to be solved.
2. **Planning**: Outline the high-level reasoning strategy, including decomposition of subproblems.
3. **Retrieval**: Recall relevant knowledge, facts, or formulas necessary for problem solving.
4. **Action**: Execute concrete reasoning steps, calculations, or logical operations.

Write your reasoning clearly using LaTeX. Box the final answer using \\boxed{}."""

    all_data = []
    splits_to_load = ["train", "test"] if split == "all" else [split]

    for s in splits_to_load:
        for subset in MATH_SUBSETS:
            logger.info(f"Loading {subset} {s}...")
            dataset = load_dataset("EleutherAI/hendrycks_math", subset, split=s)

            for item in dataset:
                all_data.append({
                    'instruction': MATH_INSTRUCTION,
                    'input': item['problem'],
                    'output': item['solution'],
                    'type': item.get('type', subset),
                    'level': item.get('level', ''),
                    'subset': subset,
                })

            logger.info(f"  Loaded {len(dataset)} problems from {subset} {s}")

    logger.info(f"Total loaded: {len(all_data)} problems")
    return all_data


def main():
    parser = argparse.ArgumentParser(description="Collect progressive data for ACT-E")
    parser.add_argument("--dataset", type=str, default="hendrycks_math",
                        choices=["hendrycks_math", "math500", "humaneval"])
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument("--mentor-type", type=str, default="local", choices=["local", "api"])
    parser.add_argument("--mentor-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--intern-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--api-model", type=str, default="gpt-4o")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    # Parallel mode arguments
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel data collection with multiple GPU workers")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: number of GPUs)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated list of GPUs (e.g., '0,1,2,3,4,5,6,7'). "
                             "Each GPU loads both mentor and intern models.")
    # Legacy arguments for separate GPU allocation (deprecated)
    parser.add_argument("--mentor-gpus", type=str, default=None,
                        help="[DEPRECATED] Use --gpus instead. Comma-separated list of GPUs for mentor models")
    parser.add_argument("--intern-gpus", type=str, default=None,
                        help="[DEPRECATED] Use --gpus instead. Comma-separated list of GPUs for intern models")
    parser.add_argument("--mentor-device", type=str, default=None,
                        help="Single GPU device for mentor in sequential mode (e.g., 'cuda:0')")
    parser.add_argument("--intern-device", type=str, default=None,
                        help="Single GPU device for intern in sequential mode (e.g., 'cuda:1')")

    args = parser.parse_args()

    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "acte_experiments")

    # Load data
    if args.dataset == "hendrycks_math":
        # Load directly from HuggingFace
        data = load_hendrycks_math(args.split)
    else:
        # Load from local file (legacy datasets)
        if args.split == "all":
            # Try all.json first, otherwise merge train.json and test.json
            all_file = os.path.join(data_dir, args.dataset, "all.json")
            if os.path.exists(all_file):
                with open(all_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} samples from {all_file}")
            else:
                # Merge train and test
                data = []
                for split_name in ["train", "test"]:
                    split_file = os.path.join(data_dir, args.dataset, f"{split_name}.json")
                    if os.path.exists(split_file):
                        with open(split_file, 'r', encoding='utf-8') as f:
                            split_data = json.load(f)
                        data.extend(split_data)
                        logger.info(f"Loaded {len(split_data)} samples from {split_file}")
                logger.info(f"Total: {len(data)} samples (merged train + test)")
        else:
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

    if args.parallel:
        # Parse GPU list - each GPU loads both mentor and intern
        if args.gpus:
            gpus = [f"cuda:{g.strip()}" for g in args.gpus.split(",")]
        elif args.mentor_gpus and args.intern_gpus:
            # Legacy mode: use separate GPU lists (deprecated)
            mentor_gpus = [f"cuda:{g.strip()}" for g in args.mentor_gpus.split(",")]
            intern_gpus = [f"cuda:{g.strip()}" for g in args.intern_gpus.split(",")]
            logger.warning("Using deprecated --mentor-gpus/--intern-gpus. Consider using --gpus instead.")

            num_workers = args.num_workers or min(len(mentor_gpus), len(intern_gpus))
            logger.info(f"Parallel mode (legacy) with {num_workers} workers")
            logger.info(f"Mentor GPUs: {mentor_gpus[:num_workers]}")
            logger.info(f"Intern GPUs: {intern_gpus[:num_workers]}")

            mp.set_start_method('spawn', force=True)
            collect_dataset_parallel(
                data=data,
                output_dir=args.output_dir,
                dataset_name=f"{args.dataset}_{args.split}",
                mentor_type=args.mentor_type,
                mentor_model=args.mentor_model,
                intern_model=args.intern_model,
                api_key=args.api_key,
                api_model=args.api_model,
                dataset_type=dataset_type,
                num_workers=num_workers,
                mentor_gpus=mentor_gpus,
                intern_gpus=intern_gpus,
            )
        else:
            # Default: use all available GPUs
            num_gpus = torch.cuda.device_count()
            gpus = [f"cuda:{i}" for i in range(num_gpus)]

        if args.gpus or (not args.mentor_gpus and not args.intern_gpus):
            # New mode: each GPU loads both models
            num_workers = args.num_workers or len(gpus)
            if num_workers > len(gpus):
                logger.warning(f"Requested {num_workers} workers but only {len(gpus)} GPUs available. Using {len(gpus)} workers.")
                num_workers = len(gpus)

            logger.info(f"Parallel mode with {num_workers} workers (each GPU loads both mentor + intern)")
            logger.info(f"GPUs: {gpus[:num_workers]}")

            # Use multiprocessing spawn method for CUDA
            mp.set_start_method('spawn', force=True)

            collect_dataset_parallel(
                data=data,
                output_dir=args.output_dir,
                dataset_name=f"{args.dataset}_{args.split}",
                mentor_type=args.mentor_type,
                mentor_model=args.mentor_model,
                intern_model=args.intern_model,
                api_key=args.api_key,
                api_model=args.api_model,
                dataset_type=dataset_type,
                num_workers=num_workers,
                mentor_gpus=gpus,  # Same GPU for both
                intern_gpus=gpus,  # Same GPU for both
            )
    else:
        # Sequential mode
        collector = DataCollector(
            mentor_type=args.mentor_type,
            mentor_model=args.mentor_model,
            intern_model=args.intern_model,
            api_key=args.api_key,
            api_model=args.api_model,
            dataset_type=dataset_type,
            mentor_device=args.mentor_device,
            intern_device=args.intern_device,
        )

        collector.collect_dataset(data, args.output_dir, f"{args.dataset}_{args.split}")


if __name__ == "__main__":
    main()
