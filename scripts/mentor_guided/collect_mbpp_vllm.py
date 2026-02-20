#!/usr/bin/env python3
"""
MBPP Post-Processing: Re-evaluate correctness via code execution.

After collecting MBPP data with collect_data_vllm_think.py --dataset mbpp,
the is_correct field uses the math grader (wrong for code). This script
fixes it by extracting code from responses and running MBPP test cases.

Usage:
    python collect_mbpp_vllm.py --reeval \
        --data-dir /path/to/collected/mbpp_think_.../mbpp/test

    python collect_mbpp_vllm.py --reeval \
        --data-dir /path/to/collected/mbpp_think_.../mbpp/train
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [-1, 0, 100, 500, 1000]


def extract_code_from_response(response: str, entry_point: str = None) -> str:
    """Extract Python code from model response.

    Tries multiple strategies on the full response first (to handle unclosed
    <think> tags), then on cleaned text:
    1. Extract from ```python ... ``` code blocks
    2. Extract from ``` ... ``` code blocks
    3. Look for function definition
    4. Fall back to using the raw response
    """
    # Strategy 1: Search for ```python blocks in the FULL response first.
    # This handles unclosed <think> tags where code appears inside the think block.
    python_blocks = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip()

    # Strategy 2: Generic code blocks in full response
    code_blocks = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # Now clean up think tags for remaining strategies
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # For unclosed <think>, take everything AFTER the tag (code often follows reasoning)
    if '<think>' in text and '</think>' not in text:
        after_think = text[text.index('<think>') + len('<think>'):]
        text = after_think.strip()

    # Strategy 3: Look for function definition by entry_point
    if entry_point:
        func_pattern = rf'((?:(?:import|from)\s+\S+.*\n)*def\s+{re.escape(entry_point)}\s*\(.*?\):.*?)(?:\n(?=\S)|\Z)'
        func_match = re.search(func_pattern, text, re.DOTALL)
        if func_match:
            return func_match.group(1)

    # Strategy 4: Find any function definitions
    func_match = re.search(r'((?:(?:import|from)\s+\S+.*\n)*def\s+\w+\s*\(.*?\):.*?)(?:\n(?=\S)|\Z)', text, re.DOTALL)
    if func_match:
        return func_match.group(1)

    # Strategy 5: Indented code lines
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(('def ', 'import ', 'from ', 'class ')):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return '\n'.join(code_lines)

    return text


def check_code_correctness(
    code: str,
    test_list: List[str],
    entry_point: str = None,
    timeout: int = 10,
) -> bool:
    """Check if generated code passes MBPP test cases.

    MBPP tests are assert statements (not wrapped in check() like HumanEval).
    """
    test_code = '\n'.join(test_list)
    full_code = f"{code}\n\n{test_code}\n"

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


def load_mbpp_metadata() -> Dict[str, Dict]:
    """Load MBPP test cases from HuggingFace.

    Returns dict mapping task_id -> {test_list, entry_point}
    """
    from datasets import load_dataset

    metadata = {}
    for split in ["train", "test", "validation"]:
        try:
            ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
            for item in ds:
                task_id = f"MBPP/{item['task_id']}"
                code = item['code']
                func_match = re.search(r'def\s+(\w+)\s*\(', code)
                entry_point = func_match.group(1) if func_match else "solution"

                metadata[task_id] = {
                    'test_list': item.get('test_list', []),
                    'entry_point': entry_point,
                    'code': code,
                }
        except Exception as e:
            logger.warning(f"Could not load MBPP split={split}: {e}")

    logger.info(f"Loaded metadata for {len(metadata)} MBPP tasks")
    return metadata


def reeval_correctness(data_dir: str, exec_timeout: int = 10):
    """Re-evaluate is_correct for collected MBPP data using code execution."""
    metadata = load_mbpp_metadata()

    for token_level in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, f"tokens{token_level}.json")
        if not os.path.exists(filepath):
            logger.warning(f"Not found: {filepath}")
            continue

        with open(filepath, 'r') as f:
            results = json.load(f)

        old_correct = sum(1 for r in results if r.get('is_correct', False))
        updated = 0

        correct_so_far = 0
        for i, r in enumerate(results):
            task_id = r.get('task_id', '')
            response = r.get('response', '')

            if task_id in metadata:
                meta = metadata[task_id]
                entry_point = meta['entry_point']
                test_list = meta['test_list']
            elif 'test_list' in r:
                entry_point = r.get('entry_point', None)
                test_list = r['test_list']
            else:
                logger.warning(f"No test metadata for {task_id}, skipping")
                continue

            code = extract_code_from_response(response, entry_point)
            is_correct = check_code_correctness(code, test_list, entry_point, timeout=exec_timeout)
            r['is_correct'] = is_correct
            updated += 1
            if is_correct:
                correct_so_far += 1

            if (i + 1) % 50 == 0 or i == len(results) - 1:
                print(f"\r  tokens={token_level}: {i+1}/{len(results)} "
                      f"({correct_so_far}/{updated} correct so far)", end="", flush=True)
        print()  # newline after progress

        new_correct = sum(1 for r in results if r.get('is_correct', False))
        accuracy = new_correct / len(results) if results else 0

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"tokens={token_level}: {old_correct} -> {new_correct} correct "
            f"({accuracy:.4f}, {updated} re-evaluated)"
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"MBPP Correctness Re-evaluation Summary")
    print(f"{'='*60}")
    print(f"Data dir: {data_dir}")
    print()

    for token_level in TOKEN_LEVELS:
        filepath = os.path.join(data_dir, f"tokens{token_level}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results = json.load(f)
            correct = sum(1 for r in results if r.get('is_correct', False))
            accuracy = correct / len(results) if results else 0
            print(f"  tokens={token_level:>5d}: {accuracy:.4f} ({correct}/{len(results)})")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="MBPP post-processing: re-evaluate is_correct via code execution"
    )
    parser.add_argument("--reeval", action="store_true", default=True)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing tokens{0,100,500,1000}.json")
    parser.add_argument("--exec-timeout", type=int, default=10,
                        help="Code execution timeout in seconds")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    reeval_correctness(data_dir=args.data_dir, exec_timeout=args.exec_timeout)


if __name__ == "__main__":
    main()
