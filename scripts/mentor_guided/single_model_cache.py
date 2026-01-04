#!/usr/bin/env python3
"""
Single Model Cache Manager

Manages cached inference results for single model runs (token level -1 and 0).
This avoids redundant computation when the same model is used across different
mentor-intern combinations.

Cache Structure:
    data/acte_experiments/single_model/{dataset}_{mode}/{model_name}/{subset}/{split}.json

Example:
    data/acte_experiments/single_model/hendrycks_math_split_think/gpt-4o/geometry/test.json
    data/acte_experiments/single_model/hendrycks_math_split_think/DeepSeek-R1-Distill-Qwen-7B/algebra/train.json

Usage:
    from single_model_cache import SingleModelCache

    cache = SingleModelCache(base_dir="/path/to/data/acte_experiments")

    # Check if cached results exist
    if cache.exists("hendrycks_math", "think", "gpt-4o", "geometry", "test"):
        results = cache.load(...)
    else:
        # Compute results...
        cache.save(results, ...)

    # Link cached results to target path
    cache.link_to_target(source_model, target_path, ...)
"""

import json
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Model name mapping for cleaner directory names
MODEL_NAME_MAP = {
    # OpenAI models
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-4-turbo": "gpt-4-turbo",
    "openai/o1": "o1",
    "openai/o1-mini": "o1-mini",
    "openai/o1-preview": "o1-preview",
    # Anthropic models
    "anthropic/claude-3-opus": "claude-3-opus",
    "anthropic/claude-3-sonnet": "claude-3-sonnet",
    "anthropic/claude-3.5-sonnet": "claude-3.5-sonnet",
    # DeepSeek models
    "deepseek/deepseek-r1": "deepseek-r1",
    "deepseek/deepseek-chat": "deepseek-chat",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-Qwen-32B",
    # Qwen models
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen/QwQ-32B": "QwQ-32B",
}


def get_model_short_name(model_name: str) -> str:
    """Get short name for model, used in directory paths."""
    if model_name in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[model_name]
    # Fallback: extract last part of path
    return model_name.split('/')[-1]


def get_dataset_prefix(dataset: str, mode: str) -> str:
    """Get dataset prefix for cache directory.

    Args:
        dataset: Dataset name (hendrycks_math, math500, gsm8k, etc.)
        mode: think or standard
    """
    if dataset == "math500":
        return f"math500_{mode}"
    elif dataset == "gsm8k":
        return f"gsm8k_{mode}"
    elif dataset == "hendrycks_math_all":
        return f"hendrycks_math_all_{mode}"
    else:
        return f"hendrycks_math_split_{mode}"


class SingleModelCache:
    """Manager for single model inference result caching."""

    def __init__(self, base_dir: str = "/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments"):
        """Initialize cache manager.

        Args:
            base_dir: Base directory for all experiment data
        """
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "single_model"

    def _get_cache_path(
        self,
        dataset: str,
        mode: str,
        model_name: str,
        subset: str,
        split: str,
    ) -> Path:
        """Get cache file path for a specific model+dataset+subset+split combination."""
        dataset_prefix = get_dataset_prefix(dataset, mode)
        model_short = get_model_short_name(model_name)
        return self.cache_dir / dataset_prefix / model_short / subset / f"{split}.json"

    def exists(
        self,
        dataset: str,
        mode: str,
        model_name: str,
        subset: str,
        split: str,
    ) -> bool:
        """Check if cached results exist for the given configuration."""
        cache_path = self._get_cache_path(dataset, mode, model_name, subset, split)
        return cache_path.exists()

    def load(
        self,
        dataset: str,
        mode: str,
        model_name: str,
        subset: str,
        split: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Load cached results.

        Returns:
            List of result dictionaries, or None if cache doesn't exist
        """
        cache_path = self._get_cache_path(dataset, mode, model_name, subset, split)
        if not cache_path.exists():
            return None

        logger.info(f"Loading cached results from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save(
        self,
        results: List[Dict[str, Any]],
        dataset: str,
        mode: str,
        model_name: str,
        subset: str,
        split: str,
    ) -> Path:
        """Save results to cache.

        Args:
            results: List of result dictionaries
            dataset, mode, model_name, subset, split: Cache key components

        Returns:
            Path to saved cache file
        """
        cache_path = self._get_cache_path(dataset, mode, model_name, subset, split)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(results)} results to cache: {cache_path}")
        return cache_path

    def link_to_target(
        self,
        dataset: str,
        mode: str,
        model_name: str,
        subset: str,
        split: str,
        target_path: Path,
        token_level: int,
        use_symlink: bool = False,
    ) -> bool:
        """Link or copy cached results to target experiment path.

        Args:
            dataset, mode, model_name, subset, split: Cache key components
            target_path: Target directory (e.g., output_subdir)
            token_level: Token level (-1 or 0)
            use_symlink: Use symlink instead of copy (default: copy for safety)

        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(dataset, mode, model_name, subset, split)
        if not cache_path.exists():
            return False

        target_file = Path(target_path) / f"tokens{token_level}.json"
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if target_file.exists():
            # Already exists, skip
            logger.info(f"Target already exists: {target_file}")
            return True

        if use_symlink:
            # Create relative symlink
            try:
                rel_path = os.path.relpath(cache_path, target_file.parent)
                target_file.symlink_to(rel_path)
                logger.info(f"Created symlink: {target_file} -> {rel_path}")
            except OSError as e:
                logger.warning(f"Symlink failed, falling back to copy: {e}")
                shutil.copy2(cache_path, target_file)
        else:
            # Copy file
            shutil.copy2(cache_path, target_file)
            logger.info(f"Copied cache to: {target_file}")

        return True

    def get_or_compute(
        self,
        dataset: str,
        mode: str,
        model_name: str,
        subset: str,
        split: str,
        compute_fn,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get cached results or compute and cache them.

        Args:
            dataset, mode, model_name, subset, split: Cache key components
            compute_fn: Function to call if cache miss (should return results list)
            force: Force recomputation even if cache exists

        Returns:
            List of result dictionaries
        """
        if not force and self.exists(dataset, mode, model_name, subset, split):
            results = self.load(dataset, mode, model_name, subset, split)
            if results is not None:
                logger.info(f"Cache hit for {model_name}/{subset}/{split}")
                return results

        logger.info(f"Cache miss for {model_name}/{subset}/{split}, computing...")
        results = compute_fn()

        if results:
            self.save(results, dataset, mode, model_name, subset, split)

        return results

    def list_cached_models(self, dataset: str, mode: str) -> List[str]:
        """List all cached models for a dataset+mode combination."""
        dataset_prefix = get_dataset_prefix(dataset, mode)
        dataset_dir = self.cache_dir / dataset_prefix
        if not dataset_dir.exists():
            return []
        return [d.name for d in dataset_dir.iterdir() if d.is_dir()]

    def get_cache_stats(self, dataset: str, mode: str, model_name: str) -> Dict[str, Any]:
        """Get statistics about cached data for a model."""
        dataset_prefix = get_dataset_prefix(dataset, mode)
        model_short = get_model_short_name(model_name)
        model_dir = self.cache_dir / dataset_prefix / model_short

        if not model_dir.exists():
            return {"exists": False}

        stats = {"exists": True, "subsets": {}}
        for subset_dir in model_dir.iterdir():
            if subset_dir.is_dir():
                subset_stats = {}
                for split_file in subset_dir.glob("*.json"):
                    split_name = split_file.stem
                    with open(split_file, 'r') as f:
                        data = json.load(f)
                    correct = sum(1 for r in data if r.get('is_correct'))
                    subset_stats[split_name] = {
                        "count": len(data),
                        "correct": correct,
                        "accuracy": correct / len(data) if data else 0,
                    }
                stats["subsets"][subset_dir.name] = subset_stats

        return stats


# Convenience function for use in collect_data_vllm_think.py
def check_and_use_cache(
    cache: SingleModelCache,
    dataset: str,
    mode: str,
    model_name: str,
    subset: str,
    split: str,
    target_dir: str,
    token_level: int,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Check cache and link to target if exists.

    Args:
        cache: SingleModelCache instance
        dataset, mode, model_name, subset, split: Cache key
        target_dir: Target directory for experiment output
        token_level: -1 for mentor-only, 0 for intern-only
        force: Force recomputation

    Returns:
        Stats dict if cache hit, None if cache miss
    """
    if force:
        return None

    if not cache.exists(dataset, mode, model_name, subset, split):
        return None

    # Load cached results to get stats
    results = cache.load(dataset, mode, model_name, subset, split)
    if results is None:
        return None

    # Link to target
    success = cache.link_to_target(
        dataset, mode, model_name, subset, split,
        target_dir, token_level
    )

    if not success:
        return None

    # Return stats
    correct = sum(1 for r in results if r.get('is_correct'))
    return {
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0,
        "output_file": os.path.join(target_dir, f"tokens{token_level}.json"),
        "from_cache": True,
    }


def save_to_cache(
    cache: SingleModelCache,
    results: List[Dict[str, Any]],
    dataset: str,
    mode: str,
    model_name: str,
    subset: str,
    split: str,
) -> None:
    """Save computed results to cache."""
    cache.save(results, dataset, mode, model_name, subset, split)


if __name__ == "__main__":
    # Test the cache
    import argparse

    parser = argparse.ArgumentParser(description="Single Model Cache Manager")
    parser.add_argument("--base-dir", type=str,
                        default="/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments")
    parser.add_argument("--list", action="store_true", help="List cached models")
    parser.add_argument("--stats", type=str, help="Show stats for a model")
    parser.add_argument("--dataset", type=str, default="hendrycks_math")
    parser.add_argument("--mode", type=str, default="think")
    args = parser.parse_args()

    cache = SingleModelCache(args.base_dir)

    if args.list:
        models = cache.list_cached_models(args.dataset, args.mode)
        print(f"Cached models for {args.dataset}/{args.mode}:")
        for m in models:
            print(f"  - {m}")
    elif args.stats:
        stats = cache.get_cache_stats(args.dataset, args.mode, args.stats)
        print(json.dumps(stats, indent=2))
    else:
        print("Use --list or --stats <model_name>")
