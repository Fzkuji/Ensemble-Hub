#!/usr/bin/env python3
"""
Test mentor-guided inference with ORM filtering.

Strategy:
1. Generate mentor output
2. Use ORM to score it
3. If score > threshold, use mentor output; otherwise, student solves alone

用ORM过滤：只使用被判断为有帮助的mentor输出
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
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


class ORMModel(nn.Module):
    """ORM model (same architecture as training)."""

    def __init__(self, encoder_name: str, hidden_size: int = 256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(encoder_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        score = self.classifier(pooled)
        return score.squeeze(-1)


class ORMFilteredTester:
    """Test with ORM-filtered mentor outputs."""

    def __init__(
        self,
        mentor_model_name: str,
        student_model_name: str,
        orm_model_path: str,
        device: str = "cuda"
    ):
        self.device = device

        # Load ORM
        logger.info(f"Loading ORM from {orm_model_path}")
        checkpoint = torch.load(orm_model_path, map_location=device)
        encoder_name = checkpoint['encoder_name']
        hidden_size = checkpoint['hidden_size']

        self.orm_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.orm_model = ORMModel(encoder_name, hidden_size)
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
        """Get ORM score for a (problem, mentor_text) pair."""
        text = f"{problem[:500]} [SEP] {mentor_text[:300]}"

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

    def generate_mentor_tokens(self, prompt: str, max_tokens: int) -> str:
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

    def test_sample(
        self,
        problem: str,
        ground_truth: str,
        mentor_tokens: int = 100,
        orm_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Test one sample with ORM filtering."""

        prompt = f"""Solve the following math problem. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

        # Student alone (baseline)
        student_answer = self.generate_student_answer(prompt)
        student_boxed = extract_boxed_content(student_answer)
        student_correct = grade_answer(student_boxed, ground_truth)

        # Generate mentor tokens
        mentor_text = self.generate_mentor_tokens(prompt, mentor_tokens)

        # Get ORM score
        orm_score = self.get_orm_score(problem, mentor_text)

        # Decide: use mentor or not
        if orm_score >= orm_threshold:
            # Use mentor
            full_prompt = prompt + mentor_text
            mentored_answer = self.generate_student_answer(full_prompt)
            full_answer = mentor_text + mentored_answer
            final_boxed = extract_boxed_content(full_answer)
            used_mentor = True
        else:
            # Skip mentor, use student alone
            final_boxed = student_boxed
            used_mentor = False

        final_correct = grade_answer(final_boxed, ground_truth)

        # Also test always-use-mentor for comparison
        if not used_mentor:
            full_prompt = prompt + mentor_text
            mentored_answer = self.generate_student_answer(full_prompt)
            full_answer = mentor_text + mentored_answer
            always_mentor_boxed = extract_boxed_content(full_answer)
        else:
            always_mentor_boxed = final_boxed
        always_mentor_correct = grade_answer(always_mentor_boxed, ground_truth)

        return {
            "ground_truth": ground_truth,
            "student_correct": student_correct,
            "always_mentor_correct": always_mentor_correct,
            "orm_score": orm_score,
            "used_mentor": used_mentor,
            "orm_filtered_correct": final_correct,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mentor-model', default='Qwen/Qwen2.5-32B-Instruct')
    parser.add_argument('--student-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--orm-model', default='orm_model/best_model.pt')
    parser.add_argument('--dataset', default='HuggingFaceH4/MATH-500')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--difficulty', type=str, default='5')
    parser.add_argument('--mentor-tokens', type=int, default=100)
    parser.add_argument('--orm-threshold', type=float, default=0.5)
    parser.add_argument('--output-file', default='orm_filtered_results.json')

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
    tester = ORMFilteredTester(args.mentor_model, args.student_model, orm_path)

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
            mentor_tokens=args.mentor_tokens,
            orm_threshold=args.orm_threshold,
        )
        result["idx"] = i
        results.append(result)

        s = "✓" if result["student_correct"] else "✗"
        m = "✓" if result["always_mentor_correct"] else "✗"
        f = "✓" if result["orm_filtered_correct"] else "✗"
        u = "Y" if result["used_mentor"] else "N"
        logger.info(f"[{i}] S:{s} M:{m} ORM:{result['orm_score']:.2f} Used:{u} Final:{f}")

    # Analysis
    student_acc = sum(1 for r in results if r["student_correct"]) / len(results)
    always_mentor_acc = sum(1 for r in results if r["always_mentor_correct"]) / len(results)
    orm_filtered_acc = sum(1 for r in results if r["orm_filtered_correct"]) / len(results)
    mentor_usage = sum(1 for r in results if r["used_mentor"]) / len(results)

    # Rescued and hurt by ORM filtering
    rescued_by_orm = sum(1 for r in results
                        if not r["student_correct"] and r["orm_filtered_correct"])
    hurt_by_orm = sum(1 for r in results
                     if r["student_correct"] and not r["orm_filtered_correct"])

    # Compare to always-mentor
    rescued_vs_mentor = sum(1 for r in results
                           if not r["always_mentor_correct"] and r["orm_filtered_correct"])
    hurt_vs_mentor = sum(1 for r in results
                        if r["always_mentor_correct"] and not r["orm_filtered_correct"])

    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Student alone: {100*student_acc:.1f}%")
    logger.info(f"Always mentor: {100*always_mentor_acc:.1f}%")
    logger.info(f"ORM filtered:  {100*orm_filtered_acc:.1f}%")
    logger.info(f"Mentor usage:  {100*mentor_usage:.1f}%")
    logger.info(f"\nvs Student: Rescued={rescued_by_orm}, Hurt={hurt_by_orm}, Net={rescued_by_orm-hurt_by_orm:+d}")
    logger.info(f"vs Always-Mentor: Rescued={rescued_vs_mentor}, Hurt={hurt_vs_mentor}, Net={rescued_vs_mentor-hurt_vs_mentor:+d}")

    # Save
    output_path = os.path.join(script_dir, args.output_file)
    save_data = {
        "config": vars(args),
        "summary": {
            "student_acc": student_acc,
            "always_mentor_acc": always_mentor_acc,
            "orm_filtered_acc": orm_filtered_acc,
            "mentor_usage": mentor_usage,
        },
        "results": results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
