#!/usr/bin/env python3
"""
Test script for GPU memory locking.
Usage:
    python test_memory_lock.py --fraction 0.9 --device cuda:0
"""

import argparse
import torch
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.9,
                        help="Fraction of GPU memory to lock (0.0-1.0)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gb", type=float, default=0,
                        help="Lock specific GB instead of fraction (overrides --fraction)")
    args = parser.parse_args()

    device = args.device
    device_id = int(device.split(':')[-1]) if ':' in device else 0

    # Get GPU info
    props = torch.cuda.get_device_properties(device_id)
    total_mem = props.total_memory
    total_gb = total_mem / 1024**3

    print(f"GPU {device_id}: {props.name}")
    print(f"Total memory: {total_gb:.2f} GB")

    # Calculate how much to allocate
    if args.gb > 0:
        target_bytes = int(args.gb * 1024**3)
        print(f"Target: {args.gb:.1f} GB (fixed)")
    else:
        target_bytes = int(total_mem * args.fraction)
        print(f"Target: {args.fraction*100:.0f}% = {target_bytes/1024**3:.2f} GB")

    # Leave 500MB buffer
    buffer_bytes = 500 * 1024 * 1024
    alloc_bytes = target_bytes - buffer_bytes

    if alloc_bytes <= 0:
        print("Nothing to allocate!")
        return

    # Allocate
    print(f"\nAllocating {alloc_bytes/1024**3:.2f} GB...")
    alloc_elements = alloc_bytes // 4  # float32
    tensor = torch.empty(alloc_elements, dtype=torch.float32, device=device)

    # Show current usage
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    print(f"Allocated: {allocated/1024**3:.2f} GB")
    print(f"Reserved:  {reserved/1024**3:.2f} GB")
    print(f"Free:      {(total_mem - reserved)/1024**3:.2f} GB")

    print("\n" + "="*50)
    print("Memory locked. Press Ctrl+C to release and exit.")
    print("="*50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nReleasing memory...")
        del tensor
        torch.cuda.empty_cache()
        print("Done.")


if __name__ == "__main__":
    main()
