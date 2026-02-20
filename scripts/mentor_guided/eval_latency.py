#!/usr/bin/env python3
"""
Wall-clock Latency Measurement for Tandem (Response to Reviewer yUBt W5).

Measures end-to-end wall-clock time for three inference methods:
  1. Mentor-only (LLM):  32B generates full reasoning + answer
  2. Intern-only (SLM):  7B generates answer without guidance
  3. Tandem:  Multi-stage cascade with SLM feature-extraction overhead

Output table (matching rebuttal):
  | Method            | Avg Latency (s) | Total Tokens | Cost (TFLOPs) |

Measurement is split into phases to avoid GPU memory conflicts:
  Phase 1a: vLLM — mentor (32B) full generation timing
  Phase 1b: vLLM — intern (7B) generation at T0/T100/T500/T1000
  Phase 2:  transformers — intern feature extraction (PPL/entropy forward pass)
  Phase 3:  combine with classifier cascade routing
"""

import argparse
import gc
import json
import logging
import os
import pickle
import sys
import time
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TOKEN_LEVELS = [0, 100, 500, 1000]
STAGE_NAMES = {0: "T0", 100: "T100", 500: "T500", 1000: "T1000"}
SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]
PARAMS_7B = 7.0
PARAMS_32B = 32.0

# ======================== Concrete default paths ========================
BASE_DIR = "/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected"
DEFAULT_DATA_DIR = os.path.join(
    BASE_DIR,
    "hendrycks_math_split_think_mDeepSeek-R1-Distill-Qwen-32B_iDeepSeek-R1-Distill-Qwen-7B",
)
DEFAULT_CLASSIFIER_DIR = os.path.join(DEFAULT_DATA_DIR, "all", "ppl_model")
DEFAULT_MENTOR = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_INTERN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_OUTPUT = os.path.join(DEFAULT_DATA_DIR, "latency_results.json")


# ============================================================================
# Data Loading
# ============================================================================

def load_test_samples(data_dir: str, n_samples: int,
                      subsets: List[str] = None, seed: int = 42) -> List[Dict]:
    """Load a random sample of test problems with all 4 stages present."""
    if subsets is None:
        subsets = SUBSETS

    all_questions: Dict[str, Dict] = {}
    for subset in subsets:
        for tl in TOKEN_LEVELS:
            path = os.path.join(data_dir, subset, "test", f"tokens{tl}.json")
            if not os.path.exists(path):
                logger.warning(f"Missing: {path}")
                continue
            with open(path) as f:
                items = json.load(f)
            for item in items:
                q = item["question"].strip()
                if q not in all_questions:
                    all_questions[q] = {
                        "question": q,
                        "ground_truth": item.get("ground_truth", ""),
                        "subset": item.get("subset", subset),
                        "stages": {},
                    }
                all_questions[q]["stages"][tl] = {
                    "mentor_response": item.get("mentor_response", ""),
                    "is_correct": item.get("is_correct", False),
                    "intern_length": item.get("intern_length", 0),
                    "mentor_length": item.get("mentor_length", 0),
                }

    complete = [v for v in all_questions.values()
                if len(v["stages"]) == len(TOKEN_LEVELS)]
    logger.info(f"Found {len(complete)} complete samples (all 4 stages)")

    rng = np.random.RandomState(seed)
    if n_samples < len(complete):
        idx = rng.choice(len(complete), size=n_samples, replace=False)
        sampled = [complete[i] for i in sorted(idx)]
    else:
        sampled = complete
    logger.info(f"Selected {len(sampled)} samples for latency measurement")
    return sampled


# ============================================================================
# Feature extraction (PPL/Entropy) — same logic as eval_naive_cascade.py
# ============================================================================

def compute_stats(token_logprobs, token_entropies):
    """23 statistical features from token-level logprobs and entropies."""
    if not token_logprobs or not token_entropies:
        return [0.0] * 23

    n = len(token_logprobs)
    ent = np.array(token_entropies)
    lp = np.array(token_logprobs)

    def slope(a):
        return float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) >= 2 else 0.0
    def inc_ratio(a):
        return float(np.sum(np.diff(a) > 0) / max(len(a) - 1, 1))
    def dec_ratio(a):
        return float(np.sum(np.diff(a) < 0) / max(len(a) - 1, 1))
    def trend_ch(a):
        return float(np.sum(np.abs(np.diff(np.sign(np.diff(a)))) > 0)) if len(a) >= 3 else 0.0
    q1 = max(1, n // 4)
    avg_lp = float(np.mean(lp))

    return [
        float(np.exp(-avg_lp)), -avg_lp,
        float(np.mean(ent)), float(np.std(ent)),
        float(np.max(ent)), float(np.min(ent)),
        slope(ent), inc_ratio(ent), dec_ratio(ent),
        float(np.mean(ent[:q1])), float(np.mean(ent[-q1:])), trend_ch(ent),
        float(np.mean(lp)), float(np.std(lp)),
        float(np.max(lp)), float(np.min(lp)),
        slope(lp), inc_ratio(lp), dec_ratio(lp),
        float(np.mean(lp[:q1])), float(np.mean(lp[-q1:])), trend_ch(lp),
        float(n),
    ]


def extract_features_timed(model, tokenizer, text, device, max_length=1024):
    """Forward pass → PPL/entropy features. Returns (feat_23d, seconds)."""
    import torch
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"].to(device)

    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids=ids).logits
        shifted = logits[:, :-1, :]
        targets = ids[:, 1:]
        log_p = torch.log_softmax(shifted, dim=-1)
        tok_lp = log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.softmax(shifted, dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    torch.cuda.synchronize(device)
    t1 = time.time()

    feat = compute_stats(
        tok_lp[0].float().cpu().numpy().tolist(),
        ent[0].float().cpu().numpy().tolist(),
    )
    return feat, t1 - t0


# ============================================================================
# Phase 1 — vLLM generation timing
# ============================================================================

def phase1_vllm_timing(
    samples: List[Dict],
    model_name: str,
    gpu_ids: List[int],
    role: str,                   # "mentor" | "intern"
    token_levels: List[int],
    max_model_len: int = 8192,
    gpu_mem_util: float = 0.9,
) -> Dict[int, List[Dict]]:
    """Per-sample sequential vLLM generation timing at requested stages.

    Returns {token_level: [{time_s, n_output_tokens}, ...]}
    """
    from collect_data_vllm_think import VLLMInference, build_intern_prompt_with_insights

    logger.info(f"[Phase 1] Loading {role} ({model_name}) via vLLM on GPU {gpu_ids} ...")
    vllm = VLLMInference(model_name, gpu_ids=gpu_ids,
                         max_model_len=max_model_len,
                         gpu_memory_utilization=gpu_mem_util)

    # warm-up
    p0 = vllm.build_chat_prompt(samples[0]["question"], use_think=True)
    _ = vllm.generate([p0], max_tokens=64, temperature=0.0)

    results: Dict[int, List[Dict]] = {}

    for tl in token_levels:
        sname = STAGE_NAMES.get(tl, f"T{tl}")
        logger.info(f"[Phase 1] Timing {role} at {sname} ({len(samples)} samples) ...")

        # build all prompts
        prompts = []
        for s in samples:
            if role == "mentor":
                prompts.append(vllm.build_chat_prompt(s["question"], use_think=True))
            else:  # intern
                if tl == 0:
                    prompts.append(vllm.build_chat_prompt(s["question"], use_think=True))
                else:
                    hint = s["stages"].get(tl, {}).get("mentor_response", "")
                    prompts.append(build_intern_prompt_with_insights(
                        s["question"], hint, vllm, use_think=True))

        # sequential per-sample timing
        timings = []
        for prompt in tqdm(prompts, desc=f"{role} {sname}", ncols=80):
            t0 = time.time()
            resp = vllm.generate([prompt], max_tokens=4096, temperature=0.0)
            t1 = time.time()
            n_tok = len(vllm.tokenizer.encode(resp[0])) if resp and resp[0] else 0
            timings.append({"time_s": t1 - t0, "n_output_tokens": n_tok})
        results[tl] = timings

    vllm.cleanup()
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    return results


# ============================================================================
# Phase 2 — transformers feature extraction timing
# ============================================================================

def phase2_feature_timing(
    samples: List[Dict],
    intern_model_name: str,
    gpu_id: int,
    max_length: int = 1024,
) -> Dict[int, List[Dict]]:
    """Per-sample feature extraction timing at each stage.

    Returns {token_level: [{feat, time_s}, ...]}
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{gpu_id}"
    logger.info(f"[Phase 2] Loading intern ({intern_model_name}) via transformers on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(intern_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        intern_model_name, torch_dtype=torch.float16, device_map=device)
    model.eval()

    # warm-up
    _ = extract_features_timed(model, tokenizer, "warm up", device, max_length)

    results: Dict[int, List[Dict]] = {}
    for tl in TOKEN_LEVELS:
        sname = STAGE_NAMES[tl]
        logger.info(f"[Phase 2] Feature extraction at {sname} ({len(samples)} samples) ...")
        timings = []
        for s in tqdm(samples, desc=f"feat {sname}", ncols=80):
            if tl == 0:
                text = f"Question: {s['question']}\n\nAnswer:"
            else:
                hint = s["stages"].get(tl, {}).get("mentor_response", "")
                text = f"Question: {s['question']}\n\nHint: {hint}\n\nAnswer:"
            feat, fe_time = extract_features_timed(model, tokenizer, text, device, max_length)
            timings.append({"feat": feat, "time_s": fe_time})
        results[tl] = timings

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================================
# Phase 3 — classifier cascade routing + combine
# ============================================================================

def phase3_combine(
    samples: List[Dict],
    intern_gen: Dict[int, List[Dict]],
    feat_data: Dict[int, List[Dict]],
    classifier_dir: str,
    mentor_gen: Dict[int, List[Dict]] = None,
) -> Dict[str, Any]:
    """Walk the cascade per sample, sum actual times for stages visited."""
    clf_path = os.path.join(classifier_dir, "classifier.pkl")
    with open(clf_path, "rb") as f:
        saved = pickle.load(f)
    clf = saved["classifier"]
    scaler = saved["scaler"]
    thresholds = saved.get("thresholds", [0.5] * len(TOKEN_LEVELS))
    logger.info(f"Classifier: {type(clf).__name__}, thresholds={thresholds}")

    n = len(samples)
    per_sample = []

    for i in range(n):
        total_t = 0.0
        total_tok = 0
        stopped_at = None

        for stage_idx, tl in enumerate(TOKEN_LEVELS):
            # intern generation time at this stage
            gen_t = intern_gen[tl][i]["time_s"]
            gen_tok = intern_gen[tl][i]["n_output_tokens"]

            # mentor hint generation time (0 if cached or T0)
            mentor_t = 0.0
            if mentor_gen and tl in mentor_gen:
                mentor_t = mentor_gen[tl][i]["time_s"]

            # feature extraction time
            fe_t = feat_data[tl][i]["time_s"]
            feat = feat_data[tl][i]["feat"]

            # classifier inference
            t_c0 = time.time()
            feat_vec = feat + [stage_idx, tl]
            prob = clf.predict_proba(scaler.transform([feat_vec]))[0, 1]
            t_c1 = time.time()
            clf_t = t_c1 - t_c0

            total_t += mentor_t + gen_t + fe_t + clf_t
            total_tok += gen_tok

            # routing decision
            thresh = thresholds[stage_idx] if stage_idx < len(thresholds) else 0.5
            if prob >= thresh or tl == TOKEN_LEVELS[-1]:
                stopped_at = STAGE_NAMES[tl]
                break

        per_sample.append({
            "total_time_s": total_t,
            "total_tokens": total_tok,
            "stopped_at": stopped_at,
        })

    times = [r["total_time_s"] for r in per_sample]
    tokens = [r["total_tokens"] for r in per_sample]
    stop_dist = {}
    for r in per_sample:
        stop_dist[r["stopped_at"]] = stop_dist.get(r["stopped_at"], 0) + 1

    return {
        "avg_latency_s": float(np.mean(times)),
        "median_latency_s": float(np.median(times)),
        "total_time_s": float(np.sum(times)),
        "total_tokens": int(np.sum(tokens)),
        "avg_tokens_per_sample": float(np.mean(tokens)),
        "tokens_per_sec": float(np.sum(tokens) / np.sum(times)) if np.sum(times) > 0 else 0,
        "stop_distribution": stop_dist,
        "thresholds": [float(t) for t in thresholds],
    }


# ============================================================================
# Per-stage breakdown
# ============================================================================

def stage_breakdown(
    samples: List[Dict],
    intern_gen: Dict[int, List[Dict]],
    feat_data: Dict[int, List[Dict]],
    mentor_gen: Dict[int, List[Dict]] = None,
) -> Dict[str, Dict]:
    n = len(samples)
    out = {}
    for tl in TOKEN_LEVELS:
        sname = STAGE_NAMES[tl]
        gen_t = [intern_gen[tl][i]["time_s"] for i in range(n)]
        fe_t = [feat_data[tl][i]["time_s"] for i in range(n)]
        m_t = ([mentor_gen[tl][i]["time_s"] for i in range(n)]
               if mentor_gen and tl in mentor_gen else [0.0] * n)
        out[sname] = {
            "avg_mentor_hint_s": float(np.mean(m_t)),
            "avg_intern_gen_s": float(np.mean(gen_t)),
            "avg_feature_extract_s": float(np.mean(fe_t)),
            "avg_stage_total_s": float(np.mean(m_t) + np.mean(gen_t) + np.mean(fe_t)),
        }
    return out


# ============================================================================
# Print rebuttal table
# ============================================================================

def print_results(mentor_r, intern_r, tandem_r, breakdown):
    print()
    print("=" * 72)
    print("  Wall-clock Latency Results (Reviewer yUBt W5)")
    print("=" * 72)
    fmt = "  {:<20} {:>16} {:>14} {:>14}"
    print(fmt.format("Method", "Avg Latency (s)", "Total Tokens", "Cost (TFLOPs)"))
    print("  " + "-" * 66)

    if mentor_r:
        avg_tok = mentor_r["avg_tokens_per_sample"]
        tflops = 2 * PARAMS_32B * avg_tok / 1000.0
        print(fmt.format("Mentor Only (LLM)",
                          f"{mentor_r['avg_latency_s']:.2f}",
                          str(mentor_r["total_tokens"]),
                          f"{tflops:.2f}"))
    if intern_r:
        avg_tok = intern_r["avg_tokens_per_sample"]
        tflops = 2 * PARAMS_7B * avg_tok / 1000.0
        print(fmt.format("Intern Only (SLM)",
                          f"{intern_r['avg_latency_s']:.2f}",
                          str(intern_r["total_tokens"]),
                          f"{tflops:.2f}"))
    if tandem_r:
        print(fmt.format("Tandem",
                          f"{tandem_r['avg_latency_s']:.2f}",
                          str(tandem_r["total_tokens"]),
                          "—"))
    print("  " + "-" * 66)

    # tokens/sec
    print()
    print("  Throughput:")
    if mentor_r:
        print(f"    Mentor Only:  {mentor_r['tokens_per_sec']:.1f} tokens/s")
    if intern_r:
        print(f"    Intern Only:  {intern_r['tokens_per_sec']:.1f} tokens/s")
    if tandem_r:
        print(f"    Tandem:       {tandem_r['tokens_per_sec']:.1f} tokens/s")

    # stage breakdown
    if breakdown:
        print()
        print("  Tandem per-stage breakdown (avg seconds per sample):")
        print("  " + "-" * 66)
        hdr = "  {:<8} {:>12} {:>14} {:>14} {:>14}"
        print(hdr.format("Stage", "Mentor Hint", "SLM Generate", "Feature Ext.", "Stage Total"))
        print("  " + "-" * 66)
        for sname in ["T0", "T100", "T500", "T1000"]:
            if sname in breakdown:
                b = breakdown[sname]
                print(hdr.format(sname,
                                 f"{b['avg_mentor_hint_s']:.4f}",
                                 f"{b['avg_intern_gen_s']:.3f}",
                                 f"{b['avg_feature_extract_s']:.4f}",
                                 f"{b['avg_stage_total_s']:.3f}"))
        print("  " + "-" * 66)

    if tandem_r:
        print(f"\n  Cascade stop distribution: {tandem_r['stop_distribution']}")
    print("=" * 72)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Latency measurement (Reviewer yUBt W5)")
    p.add_argument("--data-dir",       default=DEFAULT_DATA_DIR)
    p.add_argument("--classifier-dir", default=DEFAULT_CLASSIFIER_DIR)
    p.add_argument("--mentor-model",   default=DEFAULT_MENTOR)
    p.add_argument("--intern-model",   default=DEFAULT_INTERN)
    p.add_argument("--mentor-gpus",    default="0,1,2,3")
    p.add_argument("--intern-gpus",    default="4,5,6,7")
    p.add_argument("--n-samples",      type=int, default=100)
    p.add_argument("--skip-mentor",    action="store_true",
                   help="Skip mentor-only and live mentor-hint timing (use cached hints)")
    p.add_argument("--subsets",        default=None)
    p.add_argument("--max-model-len",  type=int, default=8192)
    p.add_argument("--max-feat-len",   type=int, default=1024)
    p.add_argument("--gpu-mem-util",   type=float, default=0.9)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--output",         default=DEFAULT_OUTPUT)
    args = p.parse_args()

    mentor_gpus = [int(g) for g in args.mentor_gpus.split(",")]
    intern_gpus = [int(g) for g in args.intern_gpus.split(",")]
    subsets = [s.strip() for s in args.subsets.split(",")] if args.subsets else SUBSETS

    # ---- Load samples ----
    samples = load_test_samples(args.data_dir, args.n_samples, subsets, args.seed)
    if not samples:
        logger.error("No samples found — check --data-dir"); return

    # ---- Phase 1a: Mentor-only (32B) full generation ----
    mentor_only_r = None
    if not args.skip_mentor:
        logger.info("=" * 60)
        logger.info("Phase 1a: Mentor-only (32B) full generation")
        logger.info("=" * 60)
        mg = phase1_vllm_timing(samples, args.mentor_model, mentor_gpus,
                                role="mentor", token_levels=[0],
                                max_model_len=args.max_model_len,
                                gpu_mem_util=args.gpu_mem_util)
        tl = mg[0]
        mentor_only_r = {
            "avg_latency_s": float(np.mean([t["time_s"] for t in tl])),
            "total_tokens": int(sum(t["n_output_tokens"] for t in tl)),
            "avg_tokens_per_sample": float(np.mean([t["n_output_tokens"] for t in tl])),
            "tokens_per_sec": float(sum(t["n_output_tokens"] for t in tl)
                                    / sum(t["time_s"] for t in tl)),
        }

    # ---- Phase 1b: Intern (7B) generation at 4 stages ----
    logger.info("=" * 60)
    logger.info("Phase 1b: Intern (7B) generation at T0/T100/T500/T1000")
    logger.info("=" * 60)
    intern_gen = phase1_vllm_timing(samples, args.intern_model, intern_gpus,
                                    role="intern", token_levels=TOKEN_LEVELS,
                                    max_model_len=args.max_model_len,
                                    gpu_mem_util=args.gpu_mem_util)
    t0l = intern_gen[0]
    intern_only_r = {
        "avg_latency_s": float(np.mean([t["time_s"] for t in t0l])),
        "total_tokens": int(sum(t["n_output_tokens"] for t in t0l)),
        "avg_tokens_per_sample": float(np.mean([t["n_output_tokens"] for t in t0l])),
        "tokens_per_sec": float(sum(t["n_output_tokens"] for t in t0l)
                                / sum(t["time_s"] for t in t0l)),
    }

    # ---- Phase 2: Feature extraction (transformers) ----
    logger.info("=" * 60)
    logger.info("Phase 2: Feature extraction (transformers forward pass)")
    logger.info("=" * 60)
    feat_data = phase2_feature_timing(samples, args.intern_model,
                                      intern_gpus[0], args.max_feat_len)

    # ---- Phase 3: Cascade combine ----
    logger.info("=" * 60)
    logger.info("Phase 3: Cascade routing + combine")
    logger.info("=" * 60)
    # mentor hint generation timing at T100/T500/T1000
    mentor_gen_stages = None
    if not args.skip_mentor:
        logger.info("Timing mentor hint generation at T100/T500/T1000 ...")
        mentor_gen_stages = phase1_vllm_timing(
            samples, args.mentor_model, mentor_gpus,
            role="mentor", token_levels=[100, 500, 1000],
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util)
        mentor_gen_stages[0] = [{"time_s": 0.0, "n_output_tokens": 0}] * len(samples)

    tandem_r = phase3_combine(samples, intern_gen, feat_data,
                              args.classifier_dir, mentor_gen_stages)
    bkd = stage_breakdown(samples, intern_gen, feat_data, mentor_gen_stages)

    # ---- Print ----
    print_results(mentor_only_r, intern_only_r, tandem_r, bkd)

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save = {
        "n_samples": len(samples),
        "seed": args.seed,
        "mentor_model": args.mentor_model,
        "intern_model": args.intern_model,
        "mentor_only": mentor_only_r,
        "intern_only": intern_only_r,
        "tandem": tandem_r,
        "stage_breakdown": bkd,
    }
    with open(args.output, "w") as f:
        json.dump(save, f, indent=2, default=str)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
