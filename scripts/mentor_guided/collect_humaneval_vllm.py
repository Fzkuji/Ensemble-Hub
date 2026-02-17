#!/usr/bin/env python3
"""
HumanEval Post-Processing: Re-evaluate correctness via code execution.

After collecting HumanEval data with collect_data_vllm_think.py --dataset humaneval,
the is_correct field uses the math grader (wrong for code). This script fixes it
by extracting code from responses and running HumanEval test cases.

Usage:
    # Re-evaluate correctness for collected HumanEval data
    python collect_humaneval_vllm.py --reeval \
        --data-dir ../../data/acte_experiments/collected/humaneval_think_model/humaneval/test

    # Re-evaluate with custom timeout
    python collect_humaneval_vllm.py --reeval \
        --data-dir ../../data/acte_experiments/collected/humaneval_think_model/humaneval/test \
        --exec-timeout 15

    # Re-evaluate both train and test splits
    python collect_humaneval_vllm.py --reeval \
        --data-dir ../../data/acte_experiments/collected/humaneval_think_model/humaneval/train
    python collect_humaneval_vllm.py --reeval \
        --data-dir ../../data/acte_experiments/collected/humaneval_think_model/humaneval/test
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import subprocess
import re
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]


def extract_code_from_response(response: str, entry_point: str, prompt: str) -> str:
    """Extract Python code from model response.

    Tries multiple strategies:
    1. Extract from ```python ... ``` code blocks
    2. Extract from ``` ... ``` code blocks
    3. Look for the function definition
    4. Fall back to using the raw response
    """
    # Remove <think>...</think> blocks (complete ones)
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Also handle unclosed <think> (truncated mentor thinking)
    # Remove everything from <think> to the end if no </think> found
    if '<think>' in text and '</think>' not in text:
        text = text[:text.index('<think>')].strip()

    # Strategy 1: Find ```python code blocks
    python_blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if python_blocks:
        code = python_blocks[-1].strip()
        if f"def {entry_point}" in code:
            return code
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
    """Check if generated code passes the test cases."""
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


def load_humaneval_metadata(data_dir: str = None) -> Dict[str, Dict]:
    """Load HumanEval test cases and entry points from raw data.

    Returns dict mapping task_id -> {test, entry_point, prompt}
    """
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        data_dir = os.path.join(repo_root, "data", "acte_experiments", "humaneval")

    metadata = {}
    for split_file in ["train.json", "test.json"]:
        json_path = os.path.join(data_dir, split_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
            for item in raw_data:
                metadata[item['task_id']] = {
                    'test': item['test'],
                    'entry_point': item['entry_point'],
                    'prompt': item['prompt'],
                }
    logger.info(f"Loaded metadata for {len(metadata)} HumanEval tasks")
    return metadata


def reeval_correctness(data_dir: str, exec_timeout: int = 10, humaneval_data_dir: str = None):
    """Re-evaluate is_correct for collected HumanEval data using code execution.

    Args:
        data_dir: Directory containing tokens{0,100,500,1000}.json
        exec_timeout: Timeout for code execution
        humaneval_data_dir: Directory containing raw HumanEval JSON (for test cases)
    """
    metadata = load_humaneval_metadata(humaneval_data_dir)

    for token_level in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, f"tokens{token_level}.json")
        if not os.path.exists(filepath):
            logger.warning(f"Not found: {filepath}")
            continue

        with open(filepath, 'r') as f:
            results = json.load(f)

        old_correct = sum(1 for r in results if r['is_correct'])
        updated = 0

        for r in results:
            task_id = r.get('task_id', '')
            response = r.get('response', '')

            if task_id and task_id in metadata:
                meta = metadata[task_id]
                entry_point = meta['entry_point']
                test_code = meta['test']
                prompt = meta['prompt']
            elif 'entry_point' in r and 'test' in r:
                # Data already has test info (collected with collect_humaneval_vllm.py directly)
                entry_point = r['entry_point']
                test_code = r['test']
                prompt = r['question']
            else:
                logger.warning(f"No test metadata for {task_id}, skipping")
                continue

            code = extract_code_from_response(response, entry_point, prompt)
            is_correct = check_code_correctness(code, test_code, entry_point, timeout=exec_timeout)
            r['is_correct'] = is_correct
            updated += 1

        new_correct = sum(1 for r in results if r['is_correct'])
        accuracy = new_correct / len(results) if results else 0

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"tokens={token_level}: {old_correct} -> {new_correct} correct "
            f"({accuracy:.4f}, {updated} re-evaluated)"
        )

    # Print summary
    print(f"\n{'='*60}")
    print(f"HumanEval Correctness Re-evaluation Summary")
    print(f"{'='*60}")
    print(f"Data dir: {data_dir}")
    print()

    for token_level in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, f"tokens{token_level}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results = json.load(f)
            correct = sum(1 for r in results if r['is_correct'])
            accuracy = correct / len(results) if results else 0
            print(f"  tokens={token_level:>5d}: {accuracy:.4f} ({correct}/{len(results)})")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="HumanEval post-processing: re-evaluate is_correct via code execution"
    )
    parser.add_argument("--reeval", action="store_true", default=True,
                        help="Re-evaluate correctness (default mode)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing tokens{0,100,500,1000}.json")
    parser.add_argument("--humaneval-data-dir", type=str, default=None,
                        help="Directory with raw HumanEval JSON (auto-detected if not specified)")
    parser.add_argument("--exec-timeout", type=int, default=10,
                        help="Code execution timeout in seconds")

    args = parser.parse_args()

    reeval_correctness(
        data_dir=args.data_dir,
        exec_timeout=args.exec_timeout,
        humaneval_data_dir=args.humaneval_data_dir,
    )


if __name__ == "__main__":
    main()
