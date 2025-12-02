#!/usr/bin/env python3
"""
Test mentor-guided inference with Causal ORM for streaming evaluation.

Strategy:
1. Mentor generates tokens one by one
2. Causal ORM scores at each position: P(student correct | stop here)
3. Stop when score exceeds threshold, or use best stopping point

用 Causal ORM 做流式评估，实时决定何时停止 mentor 输出
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from grader import grade_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_boxed_content(text: str) -> str:
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


class CausalORM(nn.Module):
    """Causal ORM (same architecture as training)."""

    def __init__(self, model_name: str = "gpt2", hidden_size: int = 128):
        super().__init__()
        self.config = GPT2Config.from_pretrained(model_name)
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name).transformer

        self.score_head = nn.Sequential(
            nn.Linear(self.config.n_embd, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None, return_all_scores=False):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state
        all_scores = self.score_head(hidden_states).squeeze(-1)

        if return_all_scores:
            return all_scores

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            last_scores = all_scores[batch_indices, seq_lengths]
        else:
            last_scores = all_scores[:, -1]

        return last_scores


class CausalORMTester:
    """Test with Causal ORM streaming evaluation."""

    def __init__(
        self,
        mentor_model_name: str,
        student_model_name: str,
        orm_model_path: str,
        device: str = "cuda"
    ):
        self.device = device

        # Load Causal ORM
        logger.info(f"Loading Causal ORM from {orm_model_path}")
        checkpoint = torch.load(orm_model_path, map_location=device)
        base_model = checkpoint.get('base_model', 'gpt2')
        hidden_size = checkpoint.get('hidden_size', 128)

        self.orm_tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.orm_tokenizer.pad_token is None:
            self.orm_tokenizer.pad_token = self.orm_tokenizer.eos_token

        self.orm_model = CausalORM(base_model, hidden_size)
        self.orm_model.load_state_dict(checkpoint['model_state_dict'])
        self.orm_model.to(device)
        self.orm_model.eval()

        # Load mentor
        logger.info(f"Loading mentor: {mentor_model_name}")
        self.mentor_tokenizer = AutoTokenizer.from_pretrained(mentor_model_name, trust_remote_code=True)
        self.mentor_model = AutoModelForCausalLM.from_pretrained(
            mentor_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.mentor_model.eval()

        # Load student
        logger.info(f"Loading student: {student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.student_model.eval()

        for tok in [self.mentor_tokenizer, self.student_tokenizer]:
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

    def get_orm_score(self, problem: str, mentor_text: str) -> float:
        """Get ORM score for current prefix."""
        text = f"{problem[:400]} [SEP] {mentor_text}"

        encoding = self.orm_tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            score = self.orm_model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device)
            )

        return score.item()

    def streaming_mentor_generate(
        self,
        problem: str,
        prompt: str,
        max_tokens: int = 200,
        min_tokens: int = 10,
        score_threshold: float = 0.6,
        strategy: str = "threshold"  # "threshold", "best", "combined"
    ) -> Tuple[str, int, List[float]]:
        """
        Mentor generates with streaming ORM evaluation.

        Strategies:
        - "threshold": Stop when ORM score exceeds threshold
        - "best": Generate all tokens, use best stopping point
        - "combined": Stop at threshold, fallback to best if not reached

        Returns: (mentor_text, tokens_used, score_trajectory)
        """
        inputs = self.mentor_tokenizer(prompt, return_tensors="pt", truncation=True)
        mentor_ids = inputs["input_ids"].to(self.mentor_model.device)

        generated_tokens = []
        score_trajectory = []
        best_score = -1
        best_position = 0
        best_tokens = []

        for i in range(max_tokens):
            # Mentor generates one token
            with torch.no_grad():
                outputs = self.mentor_model(mentor_ids)
                logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(logits).item()

            if next_token == self.mentor_tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)
            mentor_ids = torch.cat([mentor_ids, torch.tensor([[next_token]], device=mentor_ids.device)], dim=1)

            # Check ORM score after minimum tokens
            if i >= min_tokens - 1:
                current_text = self.mentor_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                score = self.get_orm_score(problem, current_text)
                score_trajectory.append(score)

                # Track best position
                if score > best_score:
                    best_score = score
                    best_position = len(generated_tokens)
                    best_tokens = generated_tokens.copy()

                # Threshold strategy: stop immediately when score exceeds threshold
                if strategy == "threshold" and score >= score_threshold:
                    logger.info(f"Threshold stop at token {i+1}, score={score:.3f}")
                    break

        # Determine final output based on strategy
        if strategy == "best":
            # Use best stopping point
            final_tokens = best_tokens if best_tokens else generated_tokens
            logger.info(f"Best strategy: using position {best_position} with score {best_score:.3f}")
        elif strategy == "combined":
            # If threshold wasn't reached, use best position
            if not score_trajectory or score_trajectory[-1] < score_threshold:
                final_tokens = best_tokens if best_tokens else generated_tokens
                logger.info(f"Combined strategy (fallback to best): position {best_position}, score {best_score:.3f}")
            else:
                final_tokens = generated_tokens
        else:
            final_tokens = generated_tokens

        mentor_text = self.mentor_tokenizer.decode(final_tokens, skip_special_tokens=True)
        return mentor_text, len(final_tokens), score_trajectory

    def generate_student_answer(self, prompt: str, max_tokens: int = 2048) -> str:
        inputs = self.student_tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.student_model.device)

        with torch.no_grad():
            outputs = self.student_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.student_tokenizer.pad_token_id,
            )

        return self.student_tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

    def generate_mentor_tokens(self, prompt: str, max_tokens: int) -> str:
        """Generate fixed number of mentor tokens (for comparison)."""
        inputs = self.mentor_tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.mentor_model.device)

        with torch.no_grad():
            outputs = self.mentor_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.mentor_tokenizer.pad_token_id,
            )

        return self.mentor_tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

    def test_sample(
        self,
        problem: str,
        ground_truth: str,
        max_mentor_tokens: int = 200,
        score_threshold: float = 0.6,
        strategy: str = "threshold",
    ) -> Dict[str, Any]:
        """Test one sample with Causal ORM streaming evaluation."""

        prompt = f"""Solve the following math problem. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

        # Student alone (baseline)
        student_answer = self.generate_student_answer(prompt)
        student_boxed = extract_boxed_content(student_answer)
        student_correct = grade_answer(student_boxed, ground_truth)

        # Fixed mentor tokens (for comparison)
        fixed_mentor_text = self.generate_mentor_tokens(prompt, 100)
        fixed_prompt = prompt + fixed_mentor_text
        fixed_answer = self.generate_student_answer(fixed_prompt)
        fixed_full = fixed_mentor_text + fixed_answer
        fixed_boxed = extract_boxed_content(fixed_full)
        fixed_correct = grade_answer(fixed_boxed, ground_truth)

        # ORM-guided streaming
        orm_mentor_text, tokens_used, score_traj = self.streaming_mentor_generate(
            problem, prompt,
            max_tokens=max_mentor_tokens,
            score_threshold=score_threshold,
            strategy=strategy,
        )

        if tokens_used > 0:
            orm_prompt = prompt + orm_mentor_text
            orm_answer = self.generate_student_answer(orm_prompt)
            orm_full = orm_mentor_text + orm_answer
            orm_boxed = extract_boxed_content(orm_full)
            orm_correct = grade_answer(orm_boxed, ground_truth)
        else:
            # No mentor tokens used
            orm_boxed = student_boxed
            orm_correct = student_correct

        return {
            "ground_truth": ground_truth,
            "student_correct": student_correct,
            "fixed_mentor_correct": fixed_correct,
            "orm_guided_correct": orm_correct,
            "orm_tokens_used": tokens_used,
            "orm_final_score": score_traj[-1] if score_traj else None,
            "orm_max_score": max(score_traj) if score_traj else None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-32B-Instruct')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--orm-model', default='causal_orm_model/best_model.pt')
    parser.add_argument('--dataset', default='HuggingFaceH4/MATH-500')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--difficulty', type=str, default='5')
    parser.add_argument('--max-mentor-tokens', type=int, default=200)
    parser.add_argument('--score-threshold', type=float, default=0.6)
    parser.add_argument('--strategy', choices=['threshold', 'best', 'combined'], default='threshold')
    parser.add_argument('--output-file', default='causal_orm_results.json')

    args = parser.parse_args()

    difficulty_levels = [int(d.strip()) for d in args.difficulty.split(',')]

    # Load dataset
    try:
        dataset = load_dataset(args.dataset, split="test")
    except:
        dataset = load_dataset(args.dataset, split="train")

    if 'level' in dataset.column_names:
        dataset = dataset.filter(lambda x: x['level'] in difficulty_levels)

    # Initialize tester
    orm_path = os.path.join(script_dir, args.orm_model)
    tester = CausalORMTester(args.mentor_model, args.student_model, orm_path)

    results = []
    num_samples = min(args.num_samples, len(dataset))

    for i in tqdm(range(num_samples), desc="Testing"):
        sample = dataset[i]
        problem = sample.get("problem", sample.get("question", ""))
        if "solution" in sample:
            ground_truth = extract_boxed_content(sample["solution"])
        else:
            ground_truth = str(sample.get("answer", ""))

        result = tester.test_sample(
            problem, ground_truth,
            max_mentor_tokens=args.max_mentor_tokens,
            score_threshold=args.score_threshold,
            strategy=args.strategy,
        )
        result["idx"] = i
        results.append(result)

        s = "✓" if result["student_correct"] else "✗"
        f = "✓" if result["fixed_mentor_correct"] else "✗"
        o = "✓" if result["orm_guided_correct"] else "✗"
        logger.info(f"[{i}] S:{s} Fixed:{f} ORM({result['orm_tokens_used']}t):{o}")

    # Analysis
    student_acc = sum(1 for r in results if r["student_correct"]) / len(results)
    fixed_acc = sum(1 for r in results if r["fixed_mentor_correct"]) / len(results)
    orm_acc = sum(1 for r in results if r["orm_guided_correct"]) / len(results)
    avg_tokens = np.mean([r["orm_tokens_used"] for r in results])

    # Compare ORM-guided vs student
    rescued_vs_student = sum(1 for r in results
                            if not r["student_correct"] and r["orm_guided_correct"])
    hurt_vs_student = sum(1 for r in results
                         if r["student_correct"] and not r["orm_guided_correct"])

    # Compare ORM-guided vs fixed
    rescued_vs_fixed = sum(1 for r in results
                          if not r["fixed_mentor_correct"] and r["orm_guided_correct"])
    hurt_vs_fixed = sum(1 for r in results
                       if r["fixed_mentor_correct"] and not r["orm_guided_correct"])

    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Student alone:     {100*student_acc:.1f}%")
    logger.info(f"Fixed mentor (100t): {100*fixed_acc:.1f}%")
    logger.info(f"ORM-guided:        {100*orm_acc:.1f}%")
    logger.info(f"Average ORM tokens: {avg_tokens:.1f}")
    logger.info(f"\nvs Student: Rescued={rescued_vs_student}, Hurt={hurt_vs_student}, Net={rescued_vs_student-hurt_vs_student:+d}")
    logger.info(f"vs Fixed:   Rescued={rescued_vs_fixed}, Hurt={hurt_vs_fixed}, Net={rescued_vs_fixed-hurt_vs_fixed:+d}")

    # Save
    output_path = os.path.join(script_dir, args.output_file)
    save_data = {
        "config": vars(args),
        "summary": {
            "student_acc": student_acc,
            "fixed_mentor_acc": fixed_acc,
            "orm_guided_acc": orm_acc,
            "avg_tokens": avg_tokens,
        },
        "results": results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
