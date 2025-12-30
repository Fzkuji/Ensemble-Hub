#!/usr/bin/env python3
"""
计算 SLM 对 guidance 的 PPL 和 Entropy

对于每个样本，计算 intern 模型在给定 prompt + mentor guidance 情况下，
生成 intern response 的 perplexity 和 entropy。
"""

import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import torch.nn.functional as F


def load_model(model_path: str, device: str = "cuda"):
    """加载模型和 tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer


def compute_ppl_and_entropy(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = 4096,
) -> Tuple[float, float, float]:
    """
    计算给定 prompt 下生成 response 的 PPL 和 entropy

    Returns:
        ppl: perplexity
        avg_entropy: 平均 entropy
        max_entropy: 最大 entropy
    """
    # 构建完整输入
    full_text = prompt + response

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)

    # 获取 prompt 的长度，用于只计算 response 部分的 loss
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    if prompt_len >= input_ids.shape[1]:
        # Response 被截断了
        return float('inf'), float('inf'), float('inf')

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

        # 获取 logits
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # 只计算 response 部分
        # logits[i] 预测的是 token[i+1]
        # 所以 response 部分对应的 logits 是 [prompt_len-1 : -1]
        response_logits = logits[0, prompt_len-1:-1, :]  # [response_len, vocab_size]
        response_labels = input_ids[0, prompt_len:]  # [response_len]

        # 计算每个 token 的 log prob
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_labels.unsqueeze(1)).squeeze(1)  # [response_len]

        # PPL = exp(-mean(log_probs))
        avg_log_prob = token_log_probs.mean().item()
        ppl = np.exp(-avg_log_prob)

        # Entropy = -sum(p * log(p)) for each position
        probs = F.softmax(response_logits, dim=-1)
        entropies = -(probs * log_probs).sum(dim=-1)  # [response_len]

        avg_entropy = entropies.mean().item()
        max_entropy = entropies.max().item()

    return ppl, avg_entropy, max_entropy


def build_prompt(question: str, mentor_response: str, mentor_tokens: int) -> str:
    """构建 prompt（包含 mentor guidance）"""
    if mentor_tokens <= 0:
        # No mentor guidance (tokens=0 case)
        prompt = f"Question: {question}\n\nAnswer: "
    else:
        # With mentor guidance
        prompt = f"Question: {question}\n\nHint from expert: {mentor_response}\n\nAnswer: "
    return prompt


def process_subset(
    model,
    tokenizer,
    data_dir: str,
    subset: str,
    split: str,
    token_level: int,
    output_dir: str,
    max_samples: int = None,
):
    """处理单个子集的数据"""
    data_file = os.path.join(data_dir, subset, split, f"tokens{token_level}.json")
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return

    with open(data_file, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"\nProcessing {subset}/tokens{token_level}.json ({len(data)} samples)...")

    results = []
    for item in tqdm(data, desc=f"{subset}/T{token_level}"):
        question = item.get('question', '')
        mentor_response = item.get('mentor_response', '')
        intern_response = item.get('response', '')
        mentor_tokens = item.get('mentor_tokens', token_level)

        # 构建 prompt
        prompt = build_prompt(question, mentor_response, mentor_tokens)

        # 计算 PPL 和 entropy
        try:
            ppl, avg_entropy, max_entropy = compute_ppl_and_entropy(
                model, tokenizer, prompt, intern_response
            )
        except Exception as e:
            print(f"Error computing PPL: {e}")
            ppl, avg_entropy, max_entropy = float('inf'), float('inf'), float('inf')

        # 保存结果
        result = {
            'is_correct': item.get('is_correct', False),
            'ppl': ppl,
            'avg_entropy': avg_entropy,
            'max_entropy': max_entropy,
            'intern_length': item.get('intern_length', 0),
            'mentor_length': item.get('mentor_length', 0),
            'level': item.get('level', ''),
        }
        results.append(result)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{subset}_tokens{token_level}_ppl.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_file}")

    # 打印统计
    sufficient = [r for r in results if r['is_correct']]
    non_sufficient = [r for r in results if not r['is_correct']]

    print(f"\n  Sufficient (n={len(sufficient)}):")
    if sufficient:
        suff_ppl = [r['ppl'] for r in sufficient if r['ppl'] < float('inf')]
        suff_entropy = [r['avg_entropy'] for r in sufficient if r['avg_entropy'] < float('inf')]
        if suff_ppl:
            print(f"    PPL: mean={np.mean(suff_ppl):.2f}, median={np.median(suff_ppl):.2f}")
        if suff_entropy:
            print(f"    Entropy: mean={np.mean(suff_entropy):.2f}, median={np.median(suff_entropy):.2f}")

    print(f"  Non-sufficient (n={len(non_sufficient)}):")
    if non_sufficient:
        non_suff_ppl = [r['ppl'] for r in non_sufficient if r['ppl'] < float('inf')]
        non_suff_entropy = [r['avg_entropy'] for r in non_sufficient if r['avg_entropy'] < float('inf')]
        if non_suff_ppl:
            print(f"    PPL: mean={np.mean(non_suff_ppl):.2f}, median={np.median(non_suff_ppl):.2f}")
        if non_suff_entropy:
            print(f"    Entropy: mean={np.mean(non_suff_entropy):.2f}, median={np.median(non_suff_entropy):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Compute PPL and Entropy for intern responses")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Intern model path")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data-dir/ppl_analysis)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Specific subset to process (default: all)")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split: train or test")
    parser.add_argument("--token-levels", type=str, default="100,500,1000",
                        help="Comma-separated token levels to process")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per subset (for testing)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "ppl_analysis")

    token_levels = [int(x) for x in args.token_levels.split(',')]

    # 加载模型
    model, tokenizer = load_model(args.model, args.device)

    # 确定要处理的子集
    if args.subset:
        subsets = [args.subset]
    else:
        subsets = []
        for name in os.listdir(args.data_dir):
            subset_dir = os.path.join(args.data_dir, name, args.split)
            if os.path.isdir(subset_dir):
                token_file = os.path.join(subset_dir, f"tokens{token_levels[0]}.json")
                if os.path.exists(token_file):
                    subsets.append(name)
        subsets = sorted(subsets)

    print(f"Processing subsets: {subsets}")
    print(f"Token levels: {token_levels}")
    print(f"Output dir: {args.output_dir}")

    # 处理每个子集
    for subset in subsets:
        for token_level in token_levels:
            process_subset(
                model, tokenizer,
                args.data_dir, subset, args.split, token_level,
                args.output_dir, args.max_samples
            )

    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
