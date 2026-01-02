#!/usr/bin/env python3
"""
OpenRouter API Inference for Closed-Source Models

Supports using closed-source models (GPT-4, Claude, etc.) via OpenRouter API
as mentor models to generate thinking/guidance content.

Usage:
    from openrouter_inference import OpenRouterInference

    mentor = OpenRouterInference(
        model_name="openai/gpt-4-turbo",
        api_key="your-api-key"
    )
    responses = mentor.generate(prompts)
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# System prompt (same as vLLM version)
SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""


class OpenRouterInference:
    """OpenRouter API-based inference for closed-source models."""

    # Popular models available on OpenRouter
    POPULAR_MODELS = {
        # OpenAI
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "o1": "openai/o1",
        "o1-mini": "openai/o1-mini",
        "o1-preview": "openai/o1-preview",
        # GPT-OSS (open source reasoning model from OpenAI)
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss": "openai/gpt-oss-20b",
        # Anthropic
        "claude-3-opus": "anthropic/claude-3-opus",
        "claude-3-sonnet": "anthropic/claude-3-sonnet",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
        # Google
        "gemini-pro": "google/gemini-pro",
        "gemini-1.5-pro": "google/gemini-1.5-pro",
        "gemini-1.5-flash": "google/gemini-1.5-flash",
        # DeepSeek (also available on OpenRouter)
        "deepseek-r1": "deepseek/deepseek-r1",
        "deepseek-chat": "deepseek/deepseek-chat",
        # Qwen
        "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
        "qwen-2.5-coder-32b": "qwen/qwen-2.5-coder-32b-instruct",
    }

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_workers: int = 8,
    ):
        """Initialize OpenRouter API client.

        Args:
            model_name: Model name (e.g., "openai/gpt-4-turbo" or shorthand "gpt-4-turbo")
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
            max_retries: Maximum retries on API errors
            retry_delay: Delay between retries (seconds)
            max_workers: Max concurrent API requests
        """
        # Resolve model name shorthand
        if model_name in self.POPULAR_MODELS:
            self.model_name = self.POPULAR_MODELS[model_name]
        else:
            self.model_name = model_name

        # Get API key
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )

        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers

        # Model family detection (for compatibility)
        self.model_family = self._detect_model_family()

        logger.info(f"Initialized OpenRouter client for model: {self.model_name}")

    def _detect_model_family(self) -> str:
        """Detect model family from model name."""
        model_lower = self.model_name.lower()
        if "deepseek-r1" in model_lower:
            return "deepseek-r1"
        elif "qwen" in model_lower:
            return "qwen3"
        elif "gpt-oss" in model_lower or "gptoss" in model_lower:
            return "gpt-oss"
        else:
            return "default"

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """Make a single API request with retries."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Fzkuji/Ensemble-Hub",
            "X-Title": "Ensemble-Hub Mentor Inference",
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"Unexpected response format: {result}")
                    return ""

            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"API request failed after {self.max_retries} attempts")
                    return ""

        return ""

    def build_chat_prompt(
        self,
        question: str,
        use_think: bool = True,
    ) -> List[Dict[str, str]]:
        """Build chat messages for the API.

        Note: OpenRouter uses message list format, not string prompts.
        For compatibility with vLLM interface, we return messages as a list.

        Handles different model families:
        - GPT-OSS: use "Reasoning: high/none" directive in system prompt
        - DeepSeek/Qwen: standard thinking prompts
        - Others: encourage step-by-step reasoning

        Args:
            question: The math problem
            use_think: Whether to encourage thinking (adds hint to system prompt)

        Returns:
            List of message dictionaries
        """
        if self.model_family == "gpt-oss":
            # GPT-OSS: control reasoning via "Reasoning: high/medium/low/none"
            reasoning_directive = "Reasoning: high" if use_think else "Reasoning: none"
            system_content = f"{reasoning_directive}\n\n{SYSTEM_PROMPT}"
        elif use_think:
            # Encourage step-by-step reasoning for other models
            system_content = "Please think through this problem step by step, showing your reasoning process. " + SYSTEM_PROMPT
        else:
            system_content = SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

        return messages

    def generate(
        self,
        prompts: List[Any],  # Can be string prompts or message lists
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of prompts (strings or message lists)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            List of generated responses
        """
        responses = [""] * len(prompts)

        def process_prompt(idx: int, prompt: Any) -> tuple:
            # Convert string prompt to messages if needed
            if isinstance(prompt, str):
                # Parse string prompt back to messages (best effort)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = prompt

            response = self._make_request(messages, max_tokens, temperature, top_p)
            return idx, response

        # Parallel API calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(process_prompt, idx, prompt)
                for idx, prompt in enumerate(prompts)
            ]

            for future in as_completed(futures):
                try:
                    idx, response = future.result()
                    responses[idx] = response
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")

        return responses

    def generate_mentor_tokens(
        self,
        prompts: List[Any],
        max_tokens: int,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate limited mentor tokens (for hint generation).

        Note: OpenRouter API doesn't support exact token limits like vLLM.
        We request max_tokens and let the model decide when to stop.
        For partial thinking, the model will generate up to max_tokens.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of partial responses (hints)
        """
        return self.generate(prompts, max_tokens=max_tokens, temperature=temperature)

    def cleanup(self):
        """Cleanup (no-op for API-based inference)."""
        pass

    @property
    def tokenizer(self):
        """Return a dummy tokenizer for compatibility.

        Note: For accurate token counting, you'd need model-specific tokenizers.
        This provides approximate counts using tiktoken (GPT-style).
        """
        return _DummyTokenizer()


class _DummyTokenizer:
    """Dummy tokenizer for approximate token counting."""

    def __init__(self):
        self._tiktoken = None

    def _get_tiktoken(self):
        if self._tiktoken is None:
            try:
                import tiktoken
                self._tiktoken = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not installed, using word-based approximation")
                self._tiktoken = False
        return self._tiktoken

    def encode(self, text: str) -> List[int]:
        """Encode text to approximate token IDs."""
        tiktoken_enc = self._get_tiktoken()
        if tiktoken_enc:
            return tiktoken_enc.encode(text)
        else:
            # Rough approximation: ~4 chars per token
            return list(range(len(text) // 4))


def test_openrouter():
    """Test OpenRouter API connection."""
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenRouter API")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="Model name (e.g., gpt-4-turbo, claude-3-sonnet, deepseek-chat)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    args = parser.parse_args()

    print(f"Testing OpenRouter with model: {args.model}")

    try:
        client = OpenRouterInference(
            model_name=args.model,
            api_key=args.api_key,
        )

        # Test simple math problem
        question = "What is 2 + 2? Put your answer in \\boxed{}."
        messages = client.build_chat_prompt(question)

        print(f"\nSending request...")
        responses = client.generate([messages], max_tokens=256)

        print(f"\nResponse:")
        print(responses[0])

        print("\nTest passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_openrouter()
