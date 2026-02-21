#!/usr/bin/env python3
"""
Wall-clock Latency Measurement for Tandem (Reviewer yUBt W5).

Benchmarks actual vLLM generation times for both LLM (32B) and SLM (7B),
using token counts from the paper's Table 1 and the trained classifier's
cascade distribution.

Tandem cascade flow for a sample stopping at stage S:
  T0_check → (LLM gen 100 → T100_check →) ... → S_check → SLM generates at S

  - Each check: PPL/entropy forward pass + classifier inference (CLF_OVERHEAD)
  - Each stage transition: LLM generates incremental thinking tokens
  - Final step: SLM generates the answer using accumulated thinking tokens

Usage:
  python eval_latency.py --benchmark                                  # full benchmark
  python eval_latency.py --benchmark-mentor --mentor-gpus 0,1,2,3     # mentor only
  python eval_latency.py --benchmark-intern --intern-gpus 4,5,6,7     # intern only
  python eval_latency.py                                               # pre-measured
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

# ============================================================================
# Paper data (Table 1: DeepSeek-R1-Distill-Qwen-32B/7B on MATH, average)
# ============================================================================

STAGE_ORDER = ["T0", "T100", "T500", "T1000"]

# Total inference length per stage = LLM thinking tokens + SLM output tokens
STAGE_CONFIG = {
    "T0":    {"llm_tok": 0,    "total_tok": 2732},   # 7B standalone
    "T100":  {"llm_tok": 100,  "total_tok": 2735},   # 7B+32B (low)
    "T500":  {"llm_tok": 500,  "total_tok": 2853},   # 7B+32B (medium)
    "T1000": {"llm_tok": 1000, "total_tok": 2930},   # 7B+32B (high)
}
for s in STAGE_CONFIG.values():
    s["slm_tok"] = s["total_tok"] - s["llm_tok"]

# 32B standalone: avg 2630 output tokens (for Mentor Only baseline)
MENTOR_FULL_TOKENS = 2630

# Tandem cascade stage distribution (from trained classifier, Table in Q2)
CASCADE_DIST = {"T0": 0.0938, "T100": 0.1152, "T500": 0.3154, "T1000": 0.4756}

# Continual judgment overhead per stage (pre-measured: PPL/entropy forward + clf)
CLF_OVERHEAD = {"T0": 0.022, "T100": 0.029, "T500": 0.075, "T1000": 0.124}

DEFAULT_MENTOR = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_INTERN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_model(model_name: str, gpu_ids: List[int],
                    token_targets: Dict[str, int], n_repeat: int = 20,
                    max_model_len: int = 8192, gpu_mem_util: float = 0.9):
    """Benchmark vLLM generation for fixed token counts.

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
            times.append(time.time() - t0)
        avg = float(np.mean(times))
        results[label] = avg
        log.info(f"  {label}: {avg:.3f}s ({n_tok} tokens)")

    vllm_m.cleanup()
    gc.collect()
    return results


# ============================================================================
# Latency computation
# ============================================================================

def compute_cascade_latency(llm_times: Dict[int, float],
                            slm_times: Dict[str, float],
                            cascade_dist: Dict[str, float] = None):
    """Compute Tandem cascade latency with proper LLM + SLM accounting.

    Cascade flow for a sample stopping at stage S:
      - Visits all stages T0..S sequentially
      - At each stage: LLM generates incremental tokens + clf check
      - At stop stage: SLM generates the final answer

    Args:
        llm_times: {100: time_s, 500: time_s, 1000: time_s}
                   Cumulative LLM gen time for first N tokens.
        slm_times: {"T0": time_s, ...}
                   SLM gen time for the SLM output tokens at each stage.
        cascade_dist: {"T0": fraction, ...}

    Returns:
        cumulative: {stage: total_latency_if_stop_here}
        llm_incremental: {stage: incremental_llm_time}
        tandem_avg: weighted average latency
    """
    if cascade_dist is None:
        cascade_dist = CASCADE_DIST

    # LLM incremental times (from cumulative measurements)
    # T0: no LLM generation
    # T100: LLM generates first 100 tokens
    # T500: LLM generates tokens 100→500 (incremental)
    # T1000: LLM generates tokens 500→1000 (incremental)
    llm_incremental = {
        "T0":    0.0,
        "T100":  llm_times[100],
        "T500":  llm_times[500] - llm_times[100],
        "T1000": llm_times[1000] - llm_times[500],
    }

    # Cumulative time for samples stopping at each stage
    cumulative = {}
    running_time = 0.0
    for stage in STAGE_ORDER:
        # Add: LLM incremental generation + classifier check at this stage
        running_time += llm_incremental[stage] + CLF_OVERHEAD[stage]
        # If stopping here: also add SLM generation for the answer
        cumulative[stage] = running_time + slm_times[stage]

    # Weighted average
    tandem_avg = sum(cascade_dist[s] * cumulative[s] for s in STAGE_ORDER)

    return cumulative, llm_incremental, tandem_avg


# ============================================================================
# Output
# ============================================================================

def print_results(llm_times, slm_times, mentor_full_time):
    cumulative, llm_incr, tandem_avg = compute_cascade_latency(llm_times, slm_times)
    intern_only = slm_times["T0"]

    print()
    print("=" * 75)
    print("  Wall-clock Latency (Reviewer yUBt W5)")
    print("=" * 75)

    # Main comparison
    fmt = "  {:<22} {:>14} {:>10}"
    print(fmt.format("Method", "Latency (s)", "Speedup"))
    print("  " + "-" * 48)
    print(fmt.format("Mentor Only (32B)", f"{mentor_full_time:.2f}", "1.0x"))
    print(fmt.format("Tandem (Cascade)",  f"{tandem_avg:.2f}",
                      f"{mentor_full_time/tandem_avg:.1f}x"))
    print(fmt.format("Intern Only (7B)",  f"{intern_only:.2f}",
                      f"{mentor_full_time/intern_only:.1f}x"))

    # Per-stage breakdown
    print()
    print("  Per-stage breakdown (incremental cost per stage):")
    hdr = "  {:<8} {:>10} {:>10} {:>8} {:>10} {:>8}"
    print(hdr.format("Stage", "LLM Gen", "SLM Gen", "Clf", "Cumul", "Dist"))
    print("  " + "-" * 60)
    for stage in STAGE_ORDER:
        cfg = STAGE_CONFIG[stage]
        print(hdr.format(
            stage,
            f"{llm_incr[stage]:.2f}s",
            f"{slm_times[stage]:.2f}s",
            f"{CLF_OVERHEAD[stage]:.3f}s",
            f"{cumulative[stage]:.2f}s",
            f"{CASCADE_DIST[stage]*100:.1f}%",
        ))

    print()
    print(f"  Tandem weighted avg: {tandem_avg:.2f}s "
          f"({mentor_full_time/tandem_avg:.1f}x faster than Mentor Only)")

    # Token counts for reference
    print()
    print("  Token counts (from paper Table 1):")
    for stage in STAGE_ORDER:
        cfg = STAGE_CONFIG[stage]
        print(f"    {stage}: LLM={cfg['llm_tok']}, SLM={cfg['slm_tok']}, "
              f"Total={cfg['total_tok']}")

    print("=" * 75)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Wall-clock latency measurement for Tandem (Reviewer yUBt W5)")
    p.add_argument("--mentor-model", default=DEFAULT_MENTOR)
    p.add_argument("--intern-model", default=DEFAULT_INTERN)
    p.add_argument("--mentor-gpus", default="0,1,2,3")
    p.add_argument("--intern-gpus", default="4,5,6,7")
    p.add_argument("--n-repeat", type=int, default=20,
                   help="Repetitions per generation target")
    p.add_argument("--benchmark", action="store_true",
                   help="Benchmark both LLM and SLM (requires GPUs)")
    p.add_argument("--benchmark-mentor", action="store_true",
                   help="Benchmark LLM (32B) only")
    p.add_argument("--benchmark-intern", action="store_true",
                   help="Benchmark SLM (7B) only")
    p.add_argument("--output", default="latency_results.json")
    args = p.parse_args()

    mentor_gpus = [int(g) for g in args.mentor_gpus.split(",")]
    intern_gpus = [int(g) for g in args.intern_gpus.split(",")]

    do_mentor = args.benchmark or args.benchmark_mentor
    do_intern = args.benchmark or args.benchmark_intern

    # ---- Benchmark LLM (32B) ----
    if do_mentor:
        log.info("=" * 60)
        log.info("Benchmarking LLM (32B) generation")
        log.info("=" * 60)

        targets = {
            "LLM_100":  100,
            "LLM_500":  500,
            "LLM_1000": 1000,
            "LLM_full": MENTOR_FULL_TOKENS,
        }
        raw = benchmark_model(args.mentor_model, mentor_gpus, targets,
                              n_repeat=args.n_repeat)
        llm_times = {100: raw["LLM_100"], 500: raw["LLM_500"],
                     1000: raw["LLM_1000"]}
        mentor_full = raw["LLM_full"]

        # Save for future runs
        log.info(f"LLM times: 100={llm_times[100]:.3f}s, "
                 f"500={llm_times[500]:.3f}s, 1000={llm_times[1000]:.3f}s, "
                 f"full={mentor_full:.3f}s")
    else:
        log.info("Using pre-measured LLM times")
        log.info("(Run with --benchmark or --benchmark-mentor to re-measure)")
        # Will be filled after first benchmark run
        llm_times = None
        mentor_full = None

    # ---- Benchmark SLM (7B) ----
    if do_intern:
        log.info("=" * 60)
        log.info("Benchmarking SLM (7B) generation")
        log.info("=" * 60)

        targets = {}
        for stage in STAGE_ORDER:
            targets[f"SLM_{stage}"] = STAGE_CONFIG[stage]["slm_tok"]

        raw = benchmark_model(args.intern_model, intern_gpus, targets,
                              n_repeat=args.n_repeat)
        slm_times = {stage: raw[f"SLM_{stage}"] for stage in STAGE_ORDER}

        log.info(f"SLM times: " + ", ".join(
            f"{s}={slm_times[s]:.3f}s" for s in STAGE_ORDER))
    else:
        log.info("Using pre-measured SLM times")
        slm_times = None

    # ---- Load from previous results if needed ----
    if (llm_times is None or slm_times is None) and os.path.exists(args.output):
        log.info(f"Loading previous results from {args.output}")
        with open(args.output) as f:
            prev = json.load(f)
        if llm_times is None and "llm_gen_times_s" in prev:
            llm_times = {int(k): v for k, v in prev["llm_gen_times_s"].items()}
            mentor_full = prev["mentor_full_time_s"]
            log.info(f"Loaded LLM times from {args.output}")
        if slm_times is None and "slm_gen_times_s" in prev:
            slm_times = prev["slm_gen_times_s"]
            log.info(f"Loaded SLM times from {args.output}")

    if llm_times is None or slm_times is None:
        log.error("No generation times available. "
                  "Run with --benchmark to measure, or ensure previous "
                  f"results exist at {args.output}")
        return

    # ---- Compute and print results ----
    print_results(llm_times, slm_times, mentor_full)

    # ---- Save ----
    cumulative, llm_incr, tandem_avg = compute_cascade_latency(
        llm_times, slm_times)

    save = {
        "llm_gen_times_s": {str(k): v for k, v in llm_times.items()},
        "slm_gen_times_s": slm_times,
        "mentor_full_time_s": mentor_full,
        "llm_incremental_s": llm_incr,
        "clf_overhead_s": CLF_OVERHEAD,
        "cascade_dist": CASCADE_DIST,
        "stage_config": STAGE_CONFIG,
        "cumulative_latency_s": cumulative,
        "tandem_avg_latency_s": tandem_avg,
        "mentor_only_latency_s": mentor_full,
        "intern_only_latency_s": slm_times["T0"],
        "speedup_vs_mentor": mentor_full / tandem_avg,
    }
    with open(args.output, "w") as f:
        json.dump(save, f, indent=2)
    log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
