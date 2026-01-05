#!/usr/bin/env python3
"""
Retry failed mentor API calls (empty responses) and update the JSON file.

Usage:
    python retry_failed_mentor.py --input /path/to/tokens-1.json --model deepseek/deepseek-r1
"""

import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

# Add scripts directory to path
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grader import grade_answer

try:
    from openrouter_inference import OpenRouterInference
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to tokens-1.json')
    parser.add_argument('--model', default='deepseek/deepseek-r1', help='Mentor model name')
    parser.add_argument('--api-key', default=None, help='OpenRouter API key (or set OPENROUTER_API_KEY env)')
    parser.add_argument('--max-workers', type=int, default=8, help='Max parallel API workers')
    parser.add_argument('--dry-run', action='store_true', help='Only show stats, do not retry')
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Find failed samples (empty mentor_response)
    failed_indices = []
    for i, d in enumerate(data):
        if not d.get('mentor_response', '').strip():
            failed_indices.append(i)

    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Failed (empty response): {len(failed_indices)}")

    if args.dry_run:
        logger.info("Dry run mode, exiting.")
        return

    if not failed_indices:
        logger.info("No failed samples to retry!")
        return

    if not OPENROUTER_AVAILABLE:
        logger.error("OpenRouter not available. Install openrouter_inference.py")
        return

    # Initialize API
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("No API key. Set OPENROUTER_API_KEY or use --api-key")
        return

    logger.info(f"Initializing OpenRouter API for {args.model}...")
    mentor = OpenRouterInference(args.model, api_key=api_key, max_workers=args.max_workers)

    # Retry failed samples
    logger.info(f"Retrying {len(failed_indices)} failed samples...")

    # Prepare prompts
    failed_samples = [data[i] for i in failed_indices]
    prompts = []
    for sample in failed_samples:
        question = sample['question']
        # Use thinking prompt format
        prompt = f"""You are a helpful assistant that solves problems step by step.

Please analyze this problem and provide your thinking process, then give the final answer.

Structure your response with these components:
1. **Goal**: What is the problem asking?
2. **Planning**: How will you approach this?
3. **Retrieval**: What knowledge/formulas do you need?
4. **Action**: Execute the solution step by step.

Problem: {question}

Please think through this carefully:"""
        prompts.append(prompt)

    # Batch generate
    responses = mentor.generate(prompts, max_tokens=4096, temperature=0.0)

    # Update data
    updated_count = 0
    newly_correct = 0
    for idx, (i, response) in enumerate(zip(failed_indices, responses)):
        if response and response.strip():
            data[i]['mentor_response'] = response
            # Re-grade
            answer = data[i].get('answer', '')
            # Extract answer from response (simple heuristic)
            is_correct = grade_answer(response, answer)
            old_correct = data[i].get('is_correct', False)
            data[i]['is_correct'] = is_correct
            updated_count += 1
            if is_correct and not old_correct:
                newly_correct += 1

    logger.info(f"Updated {updated_count}/{len(failed_indices)} samples")
    logger.info(f"Newly correct: {newly_correct}")

    # Save back
    backup_path = args.input + '.backup'
    logger.info(f"Saving backup to {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saving updated data to {args.input}")
    with open(args.input, 'w') as f:
        json.dump(data, f, indent=2)

    # Final stats
    total_correct = sum(1 for d in data if d.get('is_correct'))
    logger.info(f"Final accuracy: {total_correct}/{len(data)} = {total_correct/len(data):.4f}")


if __name__ == '__main__':
    main()
