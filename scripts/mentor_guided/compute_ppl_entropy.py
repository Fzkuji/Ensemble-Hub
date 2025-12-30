#!/usr/bin/env python3
"""
计算 SLM 对 guidance 的 PPL 和 Entropy（支持 DDP）

对于每个样本，计算 intern 模型在给定 prompt + mentor guidance 情况下，
生成 intern response 的 perplexity 和 entropy。

Usage:
    # 单卡
    CUDA_VISIBLE_DEVICES=0 python compute_ppl_entropy.py --subset algebra

    # 多卡 DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 compute_ppl_entropy.py --ddp
"""

import argparse
import json
import os
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import torch.nn.functional as F


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), True
    return 0, 1, False


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否是主进程"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def load_model(model_path: str, device: str = "cuda"):
    """加载模型和 tokenizer"""
    if is_main_process():
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

    if is_main_process():
        print(f"Model loaded on {device}")
    return model, tokenizer


def compute_ppl_and_entropy(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = 4096,
    return_per_token: bool = False,
) -> Tuple[float, float, float, List[float]]:
    """
    计算给定 prompt 下生成 response 的 PPL 和 entropy

    Returns:
        ppl: perplexity
        avg_entropy: 平均 entropy
        max_entropy: 最大 entropy
        per_token_entropy: 每个token位置的entropy列表 (if return_per_token=True)
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
        return float('inf'), float('inf'), float('inf'), []

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

        # 每个token的entropy
        per_token_entropy = entropies.cpu().tolist() if return_per_token else []

    return ppl, avg_entropy, max_entropy, per_token_entropy


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
    world_size: int = 1,
    rank: int = 0,
    save_per_token: bool = False,
):
    """处理单个子集的数据"""
    data_file = os.path.join(data_dir, subset, split, f"tokens{token_level}.json")
    if not os.path.exists(data_file):
        if is_main_process():
            print(f"File not found: {data_file}")
        return

    with open(data_file, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    # DDP: 每个进程处理一部分数据
    total_samples = len(data)
    samples_per_rank = (total_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)
    local_data = data[start_idx:end_idx]

    if is_main_process():
        print(f"\nProcessing {subset}/tokens{token_level}.json ({total_samples} samples, {len(local_data)} on this rank)...")

    results = []
    pbar = tqdm(local_data, desc=f"{subset}/T{token_level}", disable=not is_main_process())

    for item in pbar:
        question = item.get('question', '')
        mentor_response = item.get('mentor_response', '')
        intern_response = item.get('response', '')
        mentor_tokens = item.get('mentor_tokens', token_level)

        # 构建 prompt
        prompt = build_prompt(question, mentor_response, mentor_tokens)

        # 计算 PPL 和 entropy
        try:
            ppl, avg_entropy, max_entropy, per_token_entropy = compute_ppl_and_entropy(
                model, tokenizer, prompt, intern_response, return_per_token=save_per_token
            )
        except Exception as e:
            if is_main_process():
                print(f"Error computing PPL: {e}")
            ppl, avg_entropy, max_entropy, per_token_entropy = float('inf'), float('inf'), float('inf'), []

        # 保存结果
        result = {
            'idx': start_idx + len(results),  # 全局索引
            'is_correct': item.get('is_correct', False),
            'ppl': ppl,
            'avg_entropy': avg_entropy,
            'max_entropy': max_entropy,
            'intern_length': item.get('intern_length', 0),
            'mentor_length': item.get('mentor_length', 0),
            'level': item.get('level', ''),
        }
        if save_per_token and per_token_entropy:
            result['per_token_entropy'] = per_token_entropy
        results.append(result)

    # DDP: 收集所有进程的结果
    if world_size > 1:
        # 每个进程保存自己的部分结果
        os.makedirs(output_dir, exist_ok=True)
        temp_file = os.path.join(output_dir, f"{subset}_tokens{token_level}_ppl_rank{rank}.json")
        with open(temp_file, 'w') as f:
            json.dump(results, f)

        # 同步
        dist.barrier()

        # 主进程合并结果
        if is_main_process():
            all_results = []
            for r in range(world_size):
                temp_file = os.path.join(output_dir, f"{subset}_tokens{token_level}_ppl_rank{r}.json")
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        all_results.extend(json.load(f))
                    os.remove(temp_file)  # 删除临时文件

            # 按全局索引排序
            all_results.sort(key=lambda x: x['idx'])
            # 移除索引字段
            for r in all_results:
                del r['idx']
            results = all_results
    else:
        # 移除索引字段
        for r in results:
            del r['idx']

    # 主进程保存最终结果
    if is_main_process():
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
    parser.add_argument("--ddp", action="store_true",
                        help="Use DDP for multi-GPU")
    parser.add_argument("--save-per-token", action="store_true",
                        help="Save per-token entropy for trend visualization")

    args = parser.parse_args()

    # 设置分布式
    if args.ddp:
        local_rank, world_size, use_ddp = setup_distributed()
        device = f"cuda:{local_rank}"
    else:
        local_rank, world_size, use_ddp = 0, 1, False
        device = "cuda"

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "ppl_analysis")

    token_levels = [int(x) for x in args.token_levels.split(',')]

    # 加载模型
    model, tokenizer = load_model(args.model, device)

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

    if is_main_process():
        print(f"Processing subsets: {subsets}")
        print(f"Token levels: {token_levels}")
        print(f"Output dir: {args.output_dir}")
        print(f"World size: {world_size}")

    # 处理每个子集
    for subset in subsets:
        for token_level in token_levels:
            process_subset(
                model, tokenizer,
                args.data_dir, subset, args.split, token_level,
                args.output_dir, args.max_samples,
                world_size, local_rank,
                save_per_token=args.save_per_token
            )

    if is_main_process():
        print(f"\nDone! Results saved to: {args.output_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
