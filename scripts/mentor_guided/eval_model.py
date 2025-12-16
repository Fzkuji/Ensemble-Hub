#!/usr/bin/env python3
"""
单独评测模型性能

评测指标：
- 准确率 (Accuracy)
- 生成长度统计 (avg/min/max tokens)
- 推理速度 (tokens/s)

Usage:
    # 使用 vLLM 评测（推荐，速度快）
    python eval_model.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --gpu 0

    # 评测特定子集
    python eval_model.py --model Qwen/Qwen2.5-7B-Instruct --subset algebra

    # 使用 HuggingFace transformers 评测
    python eval_model.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --backend hf

    # 自定义参数
    python eval_model.py --model xxx --max-tokens 2048 --temperature 0.7
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


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

    return grade_answer(pred_answer, true_answer)


def load_hendrycks_math_subset(subset: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load a specific subset of MATH dataset."""
    from datasets import load_dataset

    logger.info(f"Loading {subset} {split}...")
    dataset = load_dataset("EleutherAI/hendrycks_math", subset, split=split)

    data = []
    for item in dataset:
        data.append({
            'question': item['problem'],
            'ground_truth': item['solution'],
            'type': item.get('type', subset),
            'level': item.get('level', ''),
            'subset': subset,
        })

    logger.info(f"  Loaded {len(data)} problems from {subset} {split}")
    return data


class VLLMEvaluator:
    """vLLM-based evaluation."""

    def __init__(
        self,
        model_name: str,
        gpu_id: int = 0,
        max_model_len: int = 8192,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is required. Install with: pip install vllm")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logger.info(f"Loading model {model_name} with vLLM on GPU {gpu_id}...")

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = self.model.get_tokenizer()
        self.SamplingParams = SamplingParams
        self.model_name = model_name
        logger.info("Model loaded successfully")

    def build_prompt(self, question: str, use_cot: bool = True) -> str:
        """Build chat prompt."""
        if use_cot:
            system_prompt = "You are a mathematical reasoning expert. Think step by step and provide your final answer in \\boxed{}."
        else:
            system_prompt = "You are a mathematical reasoning expert. Provide your final answer in \\boxed{}."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {question}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> List[Dict]:
        """Generate responses and return with metadata."""
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        start_time = time.time()
        outputs = self.model.generate(prompts, sampling_params)
        total_time = time.time() - start_time

        results = []
        total_tokens = 0
        for output in outputs:
            response = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            total_tokens += num_tokens
            results.append({
                'response': response,
                'num_tokens': num_tokens,
            })

        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        return results, tokens_per_second


class HFEvaluator:
    """HuggingFace transformers-based evaluation."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model {model_name} with HuggingFace on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        self.model_name = model_name

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully")

    def build_prompt(self, question: str, use_cot: bool = True) -> str:
        """Build chat prompt."""
        if use_cot:
            system_prompt = "You are a mathematical reasoning expert. Think step by step and provide your final answer in \\boxed{}."
        else:
            system_prompt = "You are a mathematical reasoning expert. Provide your final answer in \\boxed{}."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {question}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> List[Dict]:
        """Generate responses one by one."""
        import torch

        results = []
        total_tokens = 0
        start_time = time.time()

        for prompt in tqdm(prompts, desc="Generating"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                if temperature == 0:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

            response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            num_tokens = outputs.shape[1] - input_len
            total_tokens += num_tokens

            results.append({
                'response': response,
                'num_tokens': num_tokens,
            })

            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        return results, tokens_per_second


def evaluate_subset(
    evaluator,
    data: List[Dict],
    max_tokens: int,
    temperature: float,
    batch_size: int,
    use_cot: bool,
) -> Dict:
    """Evaluate on a subset."""
    results = []
    all_tokens = []
    total_tokens_per_second = []

    # Process in batches
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
        batch = data[batch_start:batch_start + batch_size]
        prompts = [evaluator.build_prompt(item['question'], use_cot=use_cot) for item in batch]

        batch_results, tps = evaluator.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        total_tokens_per_second.append(tps)

        for item, result in zip(batch, batch_results):
            is_correct = check_math_correctness(result['response'], item['ground_truth'])
            results.append({
                'question': item['question'],
                'ground_truth': item['ground_truth'],
                'response': result['response'],
                'num_tokens': result['num_tokens'],
                'is_correct': is_correct,
                'level': item.get('level', ''),
            })
            all_tokens.append(result['num_tokens'])

    # Compute statistics
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / len(results) if results else 0

    token_stats = {
        'mean': float(np.mean(all_tokens)),
        'std': float(np.std(all_tokens)),
        'min': int(np.min(all_tokens)),
        'max': int(np.max(all_tokens)),
        'median': float(np.median(all_tokens)),
    }

    avg_tps = np.mean(total_tokens_per_second) if total_tokens_per_second else 0

    # Per-level accuracy
    level_acc = {}
    for r in results:
        level = r.get('level', 'unknown')
        if level not in level_acc:
            level_acc[level] = {'correct': 0, 'total': 0}
        level_acc[level]['total'] += 1
        if r['is_correct']:
            level_acc[level]['correct'] += 1

    for level in level_acc:
        level_acc[level]['accuracy'] = level_acc[level]['correct'] / level_acc[level]['total']

    return {
        'n_samples': len(results),
        'accuracy': accuracy,
        'correct': correct,
        'token_stats': token_stats,
        'tokens_per_second': float(avg_tps),
        'level_accuracy': level_acc,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance on MATH dataset")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--subset", type=str, default=None,
                        choices=SUBSETS + ["all"],
                        help="Subset to evaluate (default: all)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "hf"],
                        help="Inference backend (default: vllm)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for vLLM (ignored for HF)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model context length")
    parser.add_argument("--no-cot", action="store_true",
                        help="Disable chain-of-thought prompting")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate (for quick testing)")

    args = parser.parse_args()

    # Initialize evaluator
    if args.backend == "vllm":
        evaluator = VLLMEvaluator(
            model_name=args.model,
            gpu_id=args.gpu,
            max_model_len=args.max_model_len,
        )
    else:
        evaluator = HFEvaluator(
            model_name=args.model,
            device=f"cuda:{args.gpu}",
        )

    # Determine subsets to evaluate
    subsets = [args.subset] if args.subset and args.subset != "all" else SUBSETS

    # Output directory
    if args.output_dir is None:
        model_name = args.model.split('/')[-1]
        args.output_dir = f"/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/eval_results/{model_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for subset in subsets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {subset}")
        logger.info(f"{'='*60}")

        # Load data
        data = load_hendrycks_math_subset(subset, args.split)

        # Limit samples if specified
        if args.max_samples:
            data = data[:args.max_samples]

        # Evaluate
        result = evaluate_subset(
            evaluator,
            data,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size if args.backend == "vllm" else 1,
            use_cot=not args.no_cot,
        )

        all_results[subset] = {
            'n_samples': result['n_samples'],
            'accuracy': result['accuracy'],
            'correct': result['correct'],
            'token_stats': result['token_stats'],
            'tokens_per_second': result['tokens_per_second'],
            'level_accuracy': result['level_accuracy'],
        }

        # Print results
        logger.info(f"\nResults for {subset}:")
        logger.info(f"  Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['n_samples']})")
        logger.info(f"  Token stats: mean={result['token_stats']['mean']:.1f}, "
                   f"median={result['token_stats']['median']:.1f}, "
                   f"min={result['token_stats']['min']}, max={result['token_stats']['max']}")
        logger.info(f"  Speed: {result['tokens_per_second']:.1f} tokens/s")

        # Per-level accuracy
        logger.info(f"  Per-level accuracy:")
        for level, stats in sorted(result['level_accuracy'].items()):
            logger.info(f"    {level}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

        # Save detailed results
        output_file = os.path.join(args.output_dir, f"{subset}_{args.split}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved to {output_file}")

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"CoT: {not args.no_cot}")
    logger.info(f"\n{'Subset':<30} {'N':<8} {'Acc':<10} {'Avg Tokens':<12} {'Speed':<12}")
    logger.info("-" * 80)

    total_n = 0
    total_correct = 0
    total_tokens = 0

    for subset, result in all_results.items():
        logger.info(f"{subset:<30} {result['n_samples']:<8} {result['accuracy']:<10.4f} "
                   f"{result['token_stats']['mean']:<12.1f} {result['tokens_per_second']:<12.1f}")
        total_n += result['n_samples']
        total_correct += result['correct']
        total_tokens += result['token_stats']['mean'] * result['n_samples']

    if total_n > 0:
        logger.info("-" * 80)
        overall_acc = total_correct / total_n
        avg_tokens = total_tokens / total_n
        logger.info(f"{'TOTAL':<30} {total_n:<8} {overall_acc:<10.4f} {avg_tokens:<12.1f}")

    # Save summary
    summary = {
        'model': args.model,
        'backend': args.backend,
        'temperature': args.temperature,
        'use_cot': not args.no_cot,
        'max_tokens': args.max_tokens,
        'split': args.split,
        'results': all_results,
        'overall': {
            'n_samples': total_n,
            'accuracy': total_correct / total_n if total_n > 0 else 0,
            'avg_tokens': total_tokens / total_n if total_n > 0 else 0,
        }
    }

    summary_file = os.path.join(args.output_dir, f"summary_{args.split}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
