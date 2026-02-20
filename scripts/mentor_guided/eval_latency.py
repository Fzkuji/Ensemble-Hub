#!/usr/bin/env python3
"""
Simple Latency Measurement for Tandem (Reviewer yUBt W5).

Approach: generate fixed # tokens (based on paper data averages), measure time.
No classifier, no feature extraction — just pure generation speed.

Two modes:
  1. --benchmark: load vLLM, generate fixed tokens, measure wall-clock time
  2. Default:     use pre-measured generation times from previous runs

Both modes compute Tandem latency from oracle cascade distribution.

Usage:
  python eval_latency.py                              # use pre-measured times
  python eval_latency.py --benchmark --intern-gpus 4  # re-measure 7B
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
from tqdm import tqdm

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]
STAGE_NAMES = {0: "T0", 100: "T100", 500: "T500", 1000: "T1000"}
SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

# Pre-measured generation times (s/sample) from previous vLLM runs
PREMEASURED_INTERN = {0: 6.59, 100: 4.48, 500: 3.93, 1000: 3.69}
PREMEASURED_MENTOR = 19.30

BASE_DIR = "/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected"
DEFAULT_DATA_DIR = os.path.join(
    BASE_DIR,
    "hendrycks_math_split_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B",
)
DEFAULT_MENTOR = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_INTERN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


# ============================================================================
# Step 1: Load data → avg token counts + oracle cascade distribution
# ============================================================================

def load_data_stats(data_dir: str, n_samples: int = 100, seed: int = 42):
    """Load test data and compute:
      - avg intern/mentor output token counts per stage
      - oracle cascade stop distribution (stop at first correct stage)
    """
    all_q = {}
    for subset in SUBSETS:
        for tl in TOKEN_LEVELS:
            path = os.path.join(data_dir, subset, "test", f"tokens{tl}.json")
            if not os.path.exists(path):
                log.warning(f"Missing: {path}")
                continue
            with open(path) as f:
                items = json.load(f)
            for item in items:
                q = item["question"].strip()
                if q not in all_q:
                    all_q[q] = {}
                all_q[q][tl] = {
                    "intern_length": item.get("intern_length", 0),
                    "mentor_length": item.get("mentor_length", 0),
                    "is_correct": item.get("is_correct", False),
                }

    complete = [s for s in all_q.values() if len(s) == len(TOKEN_LEVELS)]
    log.info(f"Complete samples (all 4 stages): {len(complete)}")

    rng = np.random.RandomState(seed)
    if n_samples < len(complete):
        idx = rng.choice(len(complete), size=n_samples, replace=False)
        complete = [complete[i] for i in sorted(idx)]

    # Average token counts per stage
    avg_intern = {}
    avg_mentor = {}
    for tl in TOKEN_LEVELS:
        avg_intern[tl] = int(np.mean([s[tl]["intern_length"] for s in complete]))
        avg_mentor[tl] = int(np.mean([s[tl]["mentor_length"] for s in complete]))

    # Oracle cascade: stop at FIRST stage where answer is correct
    oracle_stop = {tl: 0 for tl in TOKEN_LEVELS}
    for s in complete:
        for tl in TOKEN_LEVELS:
            if s[tl]["is_correct"]:
                oracle_stop[tl] += 1
                break
        else:
            oracle_stop[TOKEN_LEVELS[-1]] += 1  # never correct → stop at last

    n = len(complete)
    oracle_dist = {tl: oracle_stop[tl] / n for tl in TOKEN_LEVELS}

    return avg_intern, avg_mentor, oracle_dist, n


# ============================================================================
# Step 2: Benchmark generation (optional, with --benchmark)
# ============================================================================

def benchmark_vllm(model_name: str, gpu_ids: List[int],
                   token_targets: Dict[str, int], n_repeat: int = 20,
                   max_model_len: int = 8192, gpu_mem_util: float = 0.9):
    """Generate fixed # tokens with vLLM, measure avg wall-clock time.

    Args:
        token_targets: {label: n_tokens}
        n_repeat: repetitions per target
    Returns:
        {label: avg_time_s}
    """
    from collect_data_vllm_think import VLLMInference

    log.info(f"Loading {model_name} on GPUs {gpu_ids} ...")
    vllm_m = VLLMInference(model_name, gpu_ids=gpu_ids,
                           max_model_len=max_model_len,
                           gpu_memory_utilization=gpu_mem_util)

    prompt = vllm_m.build_chat_prompt(
        "Solve step by step: Find all real solutions to x^3 - 3x + 1 = 0.",
        use_think=True)

    # Warmup
    _ = vllm_m.generate([prompt], max_tokens=64, temperature=0.6)

    results = {}
    for label, n_tok in token_targets.items():
        if n_tok <= 0:
            results[label] = 0.0
            continue
        times = []
        for _ in tqdm(range(n_repeat), desc=f"{label}({n_tok}tok)", ncols=80):
            t0 = time.time()
            _ = vllm_m.generate([prompt], max_tokens=n_tok, temperature=0.6)
            t1 = time.time()
            times.append(t1 - t0)
        avg = float(np.mean(times))
        results[label] = avg
        log.info(f"  {label}: {avg:.3f}s ({n_tok} tokens)")

    vllm_m.cleanup()
    gc.collect()
    return results


# ============================================================================
# Step 3: Compute Tandem latency from cascade distribution
# ============================================================================

def compute_tandem(intern_times: Dict[int, float],
                   oracle_dist: Dict[int, float]) -> float:
    """Tandem cascade latency = weighted sum of cumulative per-stage times.

    Cascade is sequential: sample stopped at T_k has visited T0, ..., T_k.
    """
    avg_time = 0.0
    for i, stop_tl in enumerate(TOKEN_LEVELS):
        p = oracle_dist.get(stop_tl, 0.0)
        cum_time = sum(intern_times[TOKEN_LEVELS[j]] for j in range(i + 1))
        avg_time += p * cum_time
    return avg_time


# ============================================================================
# Step 4: Print rebuttal table
# ============================================================================

def print_table(mentor_time, intern_time_t0, tandem_time,
                avg_intern, oracle_dist, intern_times):
    print()
    print("=" * 70)
    print("  Wall-clock Latency (Reviewer yUBt W5)")
    print("=" * 70)
    fmt = "  {:<22} {:>14} {:>14}"
    print(fmt.format("Method", "Avg Latency (s)", "Avg Tokens"))
    print("  " + "-" * 52)
    print(fmt.format("Mentor Only (32B)", f"{mentor_time:.2f}", "—"))
    print(fmt.format("Intern Only (7B)", f"{intern_time_t0:.2f}",
                      str(avg_intern[0])))
    print(fmt.format("Tandem (Oracle)", f"{tandem_time:.2f}", "—"))
    print("  " + "-" * 52)

    print()
    print("  Per-stage 7B generation time:")
    for tl in TOKEN_LEVELS:
        print(f"    {STAGE_NAMES[tl]}: {intern_times[tl]:.2f}s "
              f"(avg {avg_intern[tl]} tokens)")

    print()
    print("  Oracle cascade distribution:")
    for tl in TOKEN_LEVELS:
        print(f"    {STAGE_NAMES[tl]}: {oracle_dist[tl]*100:.1f}%")

    # Tandem breakdown by stopped stage
    print()
    print("  Tandem per-stop-stage latency:")
    for i, tl in enumerate(TOKEN_LEVELS):
        cum = sum(intern_times[TOKEN_LEVELS[j]] for j in range(i + 1))
        pct = oracle_dist[tl] * 100
        print(f"    Stop@{STAGE_NAMES[tl]}: {cum:.2f}s "
              f"(cum. gen) × {pct:.1f}% samples")

    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Latency measurement (Reviewer yUBt W5)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--mentor-model", default=DEFAULT_MENTOR)
    p.add_argument("--intern-model", default=DEFAULT_INTERN)
    p.add_argument("--mentor-gpus", default="0,1,2,3")
    p.add_argument("--intern-gpus", default="4,5,6,7")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-repeat", type=int, default=20,
                   help="Repetitions per generation target (for --benchmark)")
    p.add_argument("--benchmark", action="store_true",
                   help="Re-measure generation times with vLLM (requires GPUs)")
    p.add_argument("--skip-mentor", action="store_true",
                   help="Skip mentor (32B) benchmark, use pre-measured time")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data_dir, "latency_results.json")

    mentor_gpus = [int(g) for g in args.mentor_gpus.split(",")]
    intern_gpus = [int(g) for g in args.intern_gpus.split(",")]

    # ---- Step 1: Load data stats ----
    log.info("Loading data stats ...")
    avg_intern, avg_mentor, oracle_dist, n = load_data_stats(
        args.data_dir, args.n_samples, args.seed)

    log.info(f"Avg intern tokens per stage: {avg_intern}")
    log.info(f"Oracle cascade distribution: "
             + ", ".join(f"{STAGE_NAMES[tl]}={oracle_dist[tl]*100:.1f}%"
                         for tl in TOKEN_LEVELS))

    # ---- Step 2: Get generation times ----
    if args.benchmark:
        # Re-measure with vLLM
        log.info("=" * 60)
        log.info("Benchmarking generation times ...")
        log.info("=" * 60)

        # Mentor (32B)
        if not args.skip_mentor:
            mentor_targets = {"mentor_full": 4096}  # generate up to 4096 tokens
            mentor_results = benchmark_vllm(
                args.mentor_model, mentor_gpus, mentor_targets,
                n_repeat=args.n_repeat)
            mentor_time = mentor_results["mentor_full"]
        else:
            mentor_time = PREMEASURED_MENTOR

        # Intern (7B) at each stage
        intern_targets = {STAGE_NAMES[tl]: avg_intern[tl] for tl in TOKEN_LEVELS}
        intern_results = benchmark_vllm(
            args.intern_model, intern_gpus, intern_targets,
            n_repeat=args.n_repeat)
        intern_times = {tl: intern_results[STAGE_NAMES[tl]] for tl in TOKEN_LEVELS}
    else:
        # Use pre-measured times
        log.info("Using pre-measured generation times (no --benchmark flag)")
        mentor_time = PREMEASURED_MENTOR
        intern_times = dict(PREMEASURED_INTERN)

    # ---- Step 3: Compute Tandem latency ----
    tandem_time = compute_tandem(intern_times, oracle_dist)

    # ---- Step 4: Print & save ----
    print_table(mentor_time, intern_times[0], tandem_time,
                avg_intern, oracle_dist, intern_times)

    save = {
        "n_samples": n,
        "avg_intern_tokens": {str(k): v for k, v in avg_intern.items()},
        "avg_mentor_tokens": {str(k): v for k, v in avg_mentor.items()},
        "oracle_distribution": {str(k): v for k, v in oracle_dist.items()},
        "mentor_time_s": mentor_time,
        "intern_times_s": {str(k): v for k, v in intern_times.items()},
        "intern_only_latency_s": intern_times[0],
        "tandem_oracle_latency_s": tandem_time,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(save, f, indent=2)
    log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
