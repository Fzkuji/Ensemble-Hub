#!/usr/bin/env python3
"""
OpenRouter API Client for Mentor Model

Provides a unified interface to call large language models (GPT-5, Claude, etc.)
via the OpenRouter API as the mentor model in our ACT-E framework.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MentorResponse:
    """Response from the mentor model."""
    text: str
    tokens_used: int
    model: str
    finish_reason: str
    latency_ms: float


class OpenRouterClient:
    """
    Client for calling LLMs via OpenRouter API.

    Supports:
    - GPT-5, GPT-4o, Claude, etc.
    - Token limit control for progressive inference
    - Streaming (optional)
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    # Model ID mapping
    MODEL_MAP = {
        "gpt-5": "openai/gpt-5",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-3-opus": "anthropic/claude-3-opus",
        "deepseek-r1": "deepseek/deepseek-r1",
        "deepseek-v3": "deepseek/deepseek-chat",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            default_model: Default model to use
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY or pass api_key.")

        self.default_model = self._resolve_model_id(default_model)
        self.timeout = timeout
        self.max_retries = max_retries

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ensemble-hub",  # For rankings
            "X-Title": "ACT-E Framework",
        }

        logger.info(f"OpenRouter client initialized with model: {self.default_model}")

    def _resolve_model_id(self, model: str) -> str:
        """Resolve model name to OpenRouter model ID."""
        return self.MODEL_MAP.get(model.lower(), model)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        model: Optional[str] = None,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> MentorResponse:
        """
        Generate text from the mentor model.

        Args:
            prompt: The input prompt (question + any context)
            max_tokens: Maximum tokens to generate
            model: Model to use (default: self.default_model)
            temperature: Sampling temperature
            stop: Stop sequences
            system_prompt: Optional system prompt

        Returns:
            MentorResponse with generated text and metadata
        """
        model_id = self._resolve_model_id(model) if model else self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                latency_ms = (time.time() - start_time) * 1000

                choice = data["choices"][0]
                usage = data.get("usage", {})

                return MentorResponse(
                    text=choice["message"]["content"],
                    tokens_used=usage.get("completion_tokens", 0),
                    model=data.get("model", model_id),
                    finish_reason=choice.get("finish_reason", "unknown"),
                    latency_ms=latency_ms,
                )

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def generate_thinking_insight(
        self,
        question: str,
        max_tokens: int = 100,
        model: Optional[str] = None,
    ) -> MentorResponse:
        """
        Generate thinking insights for a question.

        This is the core method for ACT-E framework:
        - Generates structured thinking insights (Goal, Planning, Retrieval, Action)
        - Truncated at max_tokens for progressive evaluation

        Args:
            question: The problem/question to solve
            max_tokens: Maximum tokens for thinking insight
            model: Model to use

        Returns:
            MentorResponse with thinking insights
        """
        system_prompt = """You are a helpful assistant that provides structured thinking insights.
When given a problem, provide your reasoning in this format:
1. **Goal**: What is the ultimate objective?
2. **Planning**: What is the high-level strategy?
3. **Retrieval**: What relevant knowledge/facts are needed?
4. **Action**: What concrete steps lead to the answer?

Be concise but thorough. Focus on the most critical insights first."""

        return self.generate(
            prompt=question,
            max_tokens=max_tokens,
            model=model,
            temperature=0.7,
            system_prompt=system_prompt,
        )

    def generate_math_solution(
        self,
        question: str,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> MentorResponse:
        """
        Generate math solution with step-by-step reasoning.

        Args:
            question: The math problem
            max_tokens: Maximum tokens for solution
            model: Model to use

        Returns:
            MentorResponse with solution
        """
        system_prompt = """You are a mathematical reasoning expert.
Solve the given problem step by step, showing all your work.
Put your final answer in \\boxed{}.
Think carefully and verify your calculations."""

        return self.generate(
            prompt=question,
            max_tokens=max_tokens,
            model=model,
            temperature=0.2,  # Lower temperature for math
            system_prompt=system_prompt,
        )

    def generate_code_solution(
        self,
        question: str,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> MentorResponse:
        """
        Generate code solution.

        Args:
            question: The coding problem (e.g., HumanEval prompt)
            max_tokens: Maximum tokens for solution
            model: Model to use

        Returns:
            MentorResponse with code solution
        """
        system_prompt = """You are an expert Python programmer.
Implement the requested function. Only output the Python code, no explanations.
Make sure the code is correct and handles edge cases."""

        return self.generate(
            prompt=question,
            max_tokens=max_tokens,
            model=model,
            temperature=0.2,
            system_prompt=system_prompt,
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models from OpenRouter."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/models",
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def check_credits(self) -> Optional[Dict[str, Any]]:
        """Check remaining credits/usage."""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check credits: {e}")
            return None


def test_client():
    """Test the OpenRouter client."""
    # Get API key from environment or use provided key
    api_key = os.environ.get("OPENROUTER_API_KEY", "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")

    client = OpenRouterClient(api_key=api_key, default_model="gpt-4o")

    # Check credits
    credits = client.check_credits()
    if credits:
        logger.info(f"Credits info: {credits}")

    # Test thinking insight generation
    question = "Solve for x: 2x + 3 = 7"

    for max_tokens in [100, 500]:
        logger.info(f"\n--- Testing with max_tokens={max_tokens} ---")
        response = client.generate_thinking_insight(question, max_tokens=max_tokens)
        logger.info(f"Model: {response.model}")
        logger.info(f"Tokens used: {response.tokens_used}")
        logger.info(f"Latency: {response.latency_ms:.0f}ms")
        logger.info(f"Response:\n{response.text}")


if __name__ == "__main__":
    test_client()
