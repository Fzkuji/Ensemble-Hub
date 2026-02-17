#!/usr/bin/env python3
"""
Collect HumanEval Data with vLLM - Mentor/Intern Pipeline

Reuses the existing collect_data_vllm_think.py pipeline but adapted for:
- HumanEval code generation tasks (not math)
- Code execution correctness checking (not \\boxed{} extraction)
- Appropriate system prompt for code completion

Output format matches MATH pipeline (tokens{0,100,500,1000}.json with is_correct)
so the PPL classifier can directly consume it.

Usage:
    # Collect HumanEval data (all token levels)
    python collect_humaneval_vllm.py --split test

    # Specific token levels
    python collect_humaneval_vllm.py --token-levels 0,100,500,1000

    # Custom models
    python collect_humaneval_vllm.py --mentor-model /path/to/32B --intern-model /path/to/7B
"""

import argparse
import json
import logging
import os
import sys
import signal
import tempfile
import subprocess
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Reuse infrastructure from collect_data_vllm_think
from collect_data_vllm_think import (
    VLLMInference,
    collect_full_mentor_outputs,
    truncate_to_tokens,
    build_cross_model_prompt,
    detect_model_family,
    TOKEN_LEVELS as DEFAULT_TOKEN_LEVELS,
    MENTOR_ONLY_LEVEL,
)

# HumanEval-specific system prompt for code generation
SYSTEM_PROMPT_CODE = """You are a Python programming assistant. Complete the given function implementation.
Think step by step about the problem, then provide the complete function.
Put your final implementation in a Python code block like:
```python
def function_name(...):
    ...
```"""

# Simpler prompt that works better with DeepSeek-R1
SYSTEM_PROMPT_CODE_SIMPLE = """Complete the following Python function. Think step by step, then write the implementation."""


def extract_code_from_response(response: str, entry_point: str, prompt: str) -> str:
    """Extract Python code from model response.

    Tries multiple strategies:
    1. Extract from ```python ... ``` code blocks
    2. Extract from ``` ... ``` code blocks
    3. Look for the function definition
    4. Fall back to using the raw response

    Args:
        response: Model's full response (may include <think>...</think>)
        entry_point: Function name to look for
        prompt: Original function prompt/signature

    Returns:
        Complete Python code string (prompt + implementation)
    """
    import re

    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Strategy 1: Find ```python code blocks
    python_blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if python_blocks:
        # Use the last code block (usually the final version)
        code = python_blocks[-1].strip()
        # If the code contains the function definition, use it directly
        if f"def {entry_point}" in code:
            return code
        # Otherwise, it might just be the body - combine with prompt
        return prompt + code

    # Strategy 2: Find generic code blocks
    code_blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        code = code_blocks[-1].strip()
        if f"def {entry_point}" in code:
            return code
        return prompt + code

    # Strategy 3: Look for function definition in raw text
    func_pattern = rf'(def\s+{re.escape(entry_point)}\s*\(.*?\):.*?)(?:\n(?=\S)|\Z)'
    func_match = re.search(func_pattern, text, re.DOTALL)
    if func_match:
        return func_match.group(1)

    # Strategy 4: Try to find indented code after the prompt
    # The model might have just written the implementation body
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('return ') or (line and line[0] == ' '):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        code = '\n'.join(code_lines)
        if f"def {entry_point}" in code:
            return code
        return prompt + code

    # Fallback: just append the response to the prompt
    return prompt + text


def check_code_correctness(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> bool:
    """Check if generated code passes the test cases.

    Runs the code + test in a subprocess with timeout for safety.

    Args:
        code: Generated Python code
        test_code: HumanEval test code (check function)
        entry_point: Function entry point name
        timeout: Execution timeout in seconds

    Returns:
        True if all tests pass, False otherwise
    """
    # Combine code + test
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            os.unlink(tmp_path)
    except Exception:
        return False


def load_humaneval_data(data_dir: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load HumanEval data from JSON file.

    Args:
        data_dir: Directory containing train.json / test.json
        split: "train" or "test"

    Returns:
        List of HumanEval problems with fields:
        - question: The prompt (function signature + docstring)
        - ground_truth: canonical_solution
        - task_id, test, entry_point: For correctness checking
        - subset: "humaneval"
    """
    json_path = os.path.join(data_dir, f"{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"HumanEval data not found: {json_path}")

    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        data.append({
            'question': item['prompt'],
            'ground_truth': item['canonical_solution'],
            'task_id': item['task_id'],
            'test': item['test'],
            'entry_point': item['entry_point'],
            'subset': 'humaneval',
        })

    logger.info(f"Loaded {len(data)} HumanEval problems from {json_path}")
    return data


def collect_humaneval_for_token_level(
    mentor_model: "VLLMInference",
    intern_model: "VLLMInference",
    data: List[Dict[str, Any]],
    token_level: int,
    batch_size: int = 8,
    use_think: bool = True,
    mentor_cache: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Collect HumanEval data for a specific token level.

    Same logic as collect_data_for_token_level but with code execution checking.

    Args:
        mentor_model: VLLMInference for mentor
        intern_model: VLLMInference for intern
        data: List of HumanEval problems
        token_level: 0 for intern only, >0 for mentor+intern
        batch_size: Batch size
        use_think: Whether to use think mode
        mentor_cache: Optional cached mentor outputs

    Returns:
        List of results with is_correct from code execution
    """
    from tqdm import tqdm

    results = []
    total_batches = (len(data) + batch_size - 1) // batch_size
    level_desc = f"tokens={token_level}"

    for batch_start in tqdm(
        range(0, len(data), batch_size),
        desc=level_desc,
        total=total_batches,
        unit="batch",
        ncols=80,
    ):
        batch = data[batch_start:batch_start + batch_size]

        if token_level == MENTOR_ONLY_LEVEL:
            # Mentor only baseline
            prompts = [mentor_model.build_chat_prompt(item['question'], use_think=use_think) for item in batch]
            responses = mentor_model.generate(prompts)

            for item, response in zip(batch, responses):
                code = extract_code_from_response(response, item['entry_point'], item['question'])
                is_correct = check_code_correctness(code, item['test'], item['entry_point'])
                mentor_length = len(mentor_model.tokenizer.encode(response)) if response else 0
                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'task_id': item['task_id'],
                    'mentor_tokens': -1,
                    'mentor_response': response,
                    'response': response,
                    'is_correct': is_correct,
                    'mentor_length': mentor_length,
                    'intern_length': 0,
                    'subset': 'humaneval',
                })

        elif token_level == 0:
            # Intern only
            prompts = [intern_model.build_chat_prompt(item['question'], use_think=use_think) for item in batch]
            responses = intern_model.generate(prompts)

            for item, response in zip(batch, responses):
                code = extract_code_from_response(response, item['entry_point'], item['question'])
                is_correct = check_code_correctness(code, item['test'], item['entry_point'])
                intern_length = len(intern_model.tokenizer.encode(response)) if response else 0
                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'task_id': item['task_id'],
                    'mentor_tokens': 0,
                    'mentor_response': '',
                    'response': response,
                    'is_correct': is_correct,
                    'mentor_length': 0,
                    'intern_length': intern_length,
                    'subset': 'humaneval',
                })

        else:
            # Mentor generates first N tokens, intern continues
            mentor_prompts = [mentor_model.build_chat_prompt(item['question'], use_think=use_think) for item in batch]

            if mentor_cache is not None:
                mentor_outputs = []
                for item in batch:
                    full_response = mentor_cache.get(item['question'], '')
                    if full_response:
                        truncated = truncate_to_tokens(full_response, mentor_model.tokenizer, token_level)
                    else:
                        truncated = ''
                    mentor_outputs.append(truncated)
            else:
                mentor_outputs = mentor_model.generate_mentor_tokens(mentor_prompts, max_tokens=token_level)

            # Intern continues from mentor output
            is_cross_model = mentor_model.model_family != intern_model.model_family
            if is_cross_model:
                continued_prompts = [
                    build_cross_model_prompt(item['question'], mo, intern_model, use_think)
                    for item, mo in zip(batch, mentor_outputs)
                ]
            else:
                continued_prompts = [
                    prompt + mo
                    for prompt, mo in zip(mentor_prompts, mentor_outputs)
                ]

            intern_continuations = intern_model.generate(continued_prompts)

            for item, mentor_output, intern_cont in zip(batch, mentor_outputs, intern_continuations):
                full_response = mentor_output + intern_cont
                code = extract_code_from_response(full_response, item['entry_point'], item['question'])
                is_correct = check_code_correctness(code, item['test'], item['entry_point'])

                mentor_length = len(mentor_model.tokenizer.encode(mentor_output)) if mentor_output else 0
                intern_length = len(intern_model.tokenizer.encode(intern_cont)) if intern_cont else 0

                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'task_id': item['task_id'],
                    'mentor_tokens': token_level,
                    'mentor_response': mentor_output,
                    'response': full_response,
                    'is_correct': is_correct,
                    'mentor_length': mentor_length,
                    'intern_length': intern_length,
                    'subset': 'humaneval',
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect HumanEval data with vLLM mentor/intern pipeline")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Default model (use --mentor-model and --intern-model to override)")
    parser.add_argument("--mentor-model", type=str, default=None,
                        help="Mentor model name (large model)")
    parser.add_argument("--intern-model", type=str, default=None,
                        help="Intern model name (small model)")
    parser.add_argument("--data-dir", type=str,
                        default=None,
                        help="HumanEval data directory (containing train.json/test.json)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Data split")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for collected data")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length")
    parser.add_argument("--token-levels", type=str, default="0,100,500,1000",
                        help="Comma-separated token levels to collect")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--mentor-gpus", type=str, default=None,
                        help="GPU IDs for mentor model")
    parser.add_argument("--intern-gpus", type=str, default=None,
                        help="GPU IDs for intern model")
    parser.add_argument("--mentor-memory-util", type=float, default=0.5,
                        help="GPU memory utilization for mentor")
    parser.add_argument("--intern-memory-util", type=float, default=0.3,
                        help="GPU memory utilization for intern")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode")
    parser.add_argument("--force", action="store_true",
                        help="Force re-collection even if files exist")
    parser.add_argument("--exec-timeout", type=int, default=10,
                        help="Code execution timeout in seconds")

    args = parser.parse_args()

    # Determine models
    mentor_model_name = args.mentor_model if args.mentor_model else args.model
    intern_model_name = args.intern_model if args.intern_model else args.model

    logger.info(f"Mentor model: {mentor_model_name}")
    logger.info(f"Intern model: {intern_model_name}")

    use_think = not args.no_think
    token_levels = [int(x) for x in args.token_levels.split(",")]

    # Determine data directory (relative to this script: ../../data/acte_experiments/humaneval)
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        args.data_dir = os.path.join(repo_root, "data", "acte_experiments", "humaneval")
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(
                f"HumanEval data directory not found: {args.data_dir}\nSpecify --data-dir"
            )

    logger.info(f"Data directory: {args.data_dir}")

    # Load data
    data = load_humaneval_data(args.data_dir, args.split)

    # Determine output directory
    # Default: same parent as MATH data, i.e. ../../data/acte_experiments/collected/humaneval
    # Structure: {output_dir}/humaneval/{split}/tokens{level}.json
    # This matches MATH's: hendrycks_math_split/{subset}/{split}/tokens{level}.json
    if args.output_dir is None:
        base_dir = os.path.dirname(args.data_dir)  # .../acte_experiments
        args.output_dir = os.path.join(base_dir, "collected", "humaneval")

    # Output is stored in {output_dir}/humaneval/{split}/tokens{level}.json
    subset_output_dir = os.path.join(args.output_dir, "humaneval", args.split)
    os.makedirs(subset_output_dir, exist_ok=True)
    logger.info(f"Output directory: {subset_output_dir}")

    # Determine which models are needed
    need_mentor = any(t == -1 or t > 0 for t in token_levels)
    need_intern = any(t == 0 or t > 0 for t in token_levels)

    # Parse GPU IDs
    gpus = [int(g) for g in args.gpus.split(",")]
    mentor_gpu_ids = [int(g) for g in args.mentor_gpus.split(",")] if args.mentor_gpus else gpus
    intern_gpu_ids = [int(g) for g in args.intern_gpus.split(",")] if args.intern_gpus else gpus

    # Override system prompt for code generation
    import collect_data_vllm_think
    collect_data_vllm_think.SYSTEM_PROMPT = SYSTEM_PROMPT_CODE_SIMPLE

    # Load models
    mentor_model = None
    intern_model = None

    if need_mentor:
        logger.info(f"Loading mentor model on GPU {mentor_gpu_ids}...")
        mentor_model = VLLMInference(
            model_name=mentor_model_name,
            gpu_ids=mentor_gpu_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.mentor_memory_util,
        )

    if need_intern and intern_model_name != mentor_model_name:
        logger.info(f"Loading intern model on GPU {intern_gpu_ids}...")
        intern_model = VLLMInference(
            model_name=intern_model_name,
            gpu_ids=intern_gpu_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.intern_memory_util,
        )
    elif need_intern:
        # Same model for both - reuse
        intern_model = mentor_model if mentor_model else VLLMInference(
            model_name=intern_model_name,
            gpu_ids=intern_gpu_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.intern_memory_util,
        )

    # Collect full mentor outputs once (for truncation at different levels)
    mentor_cache = None
    hint_levels = [t for t in token_levels if t > 0]
    if hint_levels and mentor_model is not None:
        # Check if tokens-1.json exists to use as cache
        tokens_m1_file = os.path.join(subset_output_dir, "tokens-1.json")
        if os.path.exists(tokens_m1_file) and not args.force:
            logger.info("Loading mentor cache from existing tokens-1.json...")
            with open(tokens_m1_file, 'r') as f:
                existing = json.load(f)
            mentor_cache = {r['question']: r['mentor_response'] for r in existing}
            logger.info(f"Loaded mentor cache: {len(mentor_cache)} samples")

        if mentor_cache is None:
            logger.info(f"Collecting full mentor outputs for {len(data)} samples...")
            mentor_cache = collect_full_mentor_outputs(
                mentor_model, data, args.batch_size, use_think=use_think
            )
            logger.info(f"Mentor outputs collected ({len(mentor_cache)} samples)")

    # Collect data for each token level
    for token_level in token_levels:
        output_file = os.path.join(subset_output_dir, f"tokens{token_level}.json")

        if os.path.exists(output_file) and not args.force:
            logger.info(f"tokens={token_level} already exists, skipping (use --force to overwrite)")
            continue

        logger.info(f"Collecting tokens={token_level}...")
        cache_to_use = mentor_cache if token_level > 0 else None

        results = collect_humaneval_for_token_level(
            mentor_model=mentor_model,
            intern_model=intern_model,
            data=data,
            token_level=token_level,
            batch_size=args.batch_size,
            use_think=use_think,
            mentor_cache=cache_to_use,
        )

        # Stats
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / len(results) if results else 0
        logger.info(f"tokens={token_level}: {accuracy:.4f} ({correct}/{len(results)})")

        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"HumanEval Data Collection Summary")
    print(f"{'='*60}")
    print(f"Data: {len(data)} problems ({args.split})")
    print(f"Output: {subset_output_dir}")
    print()

    for token_level in token_levels:
        output_file = os.path.join(subset_output_dir, f"tokens{token_level}.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
            correct = sum(1 for r in results if r['is_correct'])
            accuracy = correct / len(results) if results else 0
            print(f"  tokens={token_level:>5d}: {accuracy:.4f} ({correct}/{len(results)})")

    print(f"{'='*60}")

    # Cleanup
    if mentor_model is not None:
        mentor_model.cleanup()
    if intern_model is not None and intern_model is not mentor_model:
        intern_model.cleanup()


if __name__ == "__main__":
    main()
