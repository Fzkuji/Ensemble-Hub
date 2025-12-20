#!/usr/bin/env python3
"""
Check GPU memory requirements for classifier training.
Usage: python check_gpu_memory.py [--gpus 0,1,2,3]
"""

import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs to check")
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Only estimate, don't load model")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(',')]

    print("=" * 60)
    print("GPU Memory Check")
    print("=" * 60)

    # Check each GPU
    for gpu_id in gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            print(f"GPU {gpu_id}: Not available")
            continue

        props = torch.cuda.get_device_properties(gpu_id)
        total_gb = props.total_memory / 1024**3

        # Check current usage
        torch.cuda.set_device(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        free = total_gb - reserved

        print(f"\nGPU {gpu_id}: {props.name}")
        print(f"  Total:     {total_gb:.1f} GB")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Reserved:  {reserved:.1f} GB")
        print(f"  Free:      {free:.1f} GB")

    # Estimate model requirements
    print("\n" + "=" * 60)
    print("Model Memory Estimation")
    print("=" * 60)

    # 7B model in bfloat16: 7B * 2 bytes = 14GB
    model_params = 7  # billion
    bytes_per_param = 2  # bfloat16
    model_size_gb = model_params * bytes_per_param

    # Additional memory for:
    # - KV cache during inference: ~2-4GB for 1024 tokens
    # - Activations: ~1-2GB
    # - PyTorch overhead: ~1GB
    kv_cache_gb = 3
    activations_gb = 2
    overhead_gb = 1

    total_inference_gb = model_size_gb + kv_cache_gb + activations_gb + overhead_gb

    print(f"\nModel: {args.model_path}")
    print(f"  Model weights (bf16):  {model_size_gb:.1f} GB")
    print(f"  KV cache (est.):       {kv_cache_gb:.1f} GB")
    print(f"  Activations (est.):    {activations_gb:.1f} GB")
    print(f"  PyTorch overhead:      {overhead_gb:.1f} GB")
    print(f"  ---------------------------------")
    print(f"  Total inference:       {total_inference_gb:.1f} GB")

    # For training (LoRA)
    lora_overhead_gb = 2  # LoRA adapters + gradients
    optimizer_gb = 1  # AdamW states for LoRA params
    total_training_gb = total_inference_gb + lora_overhead_gb + optimizer_gb

    print(f"\n  + LoRA overhead:       {lora_overhead_gb:.1f} GB")
    print(f"  + Optimizer states:    {optimizer_gb:.1f} GB")
    print(f"  ---------------------------------")
    print(f"  Total training:        {total_training_gb:.1f} GB")

    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    for gpu_id in gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            continue
        props = torch.cuda.get_device_properties(gpu_id)
        total_gb = props.total_memory / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        free = total_gb - reserved

        min_fraction = total_inference_gb / total_gb
        safe_fraction = (total_training_gb + 1) / total_gb  # +1GB buffer

        print(f"\nGPU {gpu_id} ({total_gb:.0f} GB total, {free:.1f} GB free):")

        if free < total_inference_gb:
            print(f"  WARNING: Not enough free memory for inference!")
            print(f"  Need: {total_inference_gb:.1f} GB, Have: {free:.1f} GB")
        else:
            print(f"  Minimum --memory-lock: {min_fraction:.2f} ({min_fraction*100:.0f}%)")
            print(f"  Recommended:           {min(safe_fraction, 0.95):.2f} ({min(safe_fraction*100, 95):.0f}%)")

    if not args.estimate_only:
        print("\n" + "=" * 60)
        print("Actual Model Loading Test")
        print("=" * 60)

        gpu_id = gpu_ids[0]
        device = f"cuda:{gpu_id}"

        print(f"\nLoading model on GPU {gpu_id}...")

        before_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)

        after_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        model_actual_gb = after_allocated - before_allocated

        print(f"  Model loaded: {model_actual_gb:.2f} GB")

        # Test inference
        print("\nRunning inference test...")
        input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            _ = model(input_ids)

        peak_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        print(f"  Peak memory: {peak_allocated:.2f} GB")

        recommended = peak_allocated / (torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3)
        print(f"\n  Recommended --memory-lock: {min(recommended + 0.05, 0.95):.2f}")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
