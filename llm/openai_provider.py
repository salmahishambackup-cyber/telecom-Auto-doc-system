"""llm/openai_provider.py — OpenAI SDK LLM provider."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import openai

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_DOCSTRING_SECTIONS = ("Args", "Returns", "Summary")
_MAX_RETRIES = 3
_BACKOFF_START = 1.0  # seconds


def _estimate_confidence(text: str) -> float:
    """Estimate docstring quality based on expected section presence.

    Args:
        text: The generated docstring text to evaluate.

    Returns:
        A float in [0.0, 1.0] where 1.0 means all expected sections found.
    """
    if not text or not text.strip():
        return 0.0
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    summary_present = len(lines) > 0
    args_present = "Args:" in text or "Arguments:" in text
    returns_present = "Returns:" in text or "Return:" in text
    found = sum([summary_present, args_present, returns_present])
    return round(found / len(_DOCSTRING_SECTIONS), 4)


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API.

    Args:
        model: OpenAI model name.
        api_key: OpenAI API key. Defaults to the ``OPENAI_API_KEY`` env var.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.timeout = timeout
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = openai.OpenAI(api_key=resolved_key, timeout=timeout)

    # ------------------------------------------------------------------
    # Internal call with retry / backoff on rate-limit
    # ------------------------------------------------------------------

    def _chat(self, prompt: str) -> tuple[str, float, int, int]:
        """Call Chat Completions with exponential back-off on rate limits.

        Args:
            prompt: The user prompt to send.

        Returns:
            Tuple of (response_text, latency_ms, prompt_tokens, completion_tokens).

        Raises:
            openai.RateLimitError: If all retries are exhausted.
            TimeoutError: If the API request times out.
        """
        delay = _BACKOFF_START
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            t0 = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                latency_ms = (time.monotonic() - t0) * 1000
                text = response.choices[0].message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = (
                    response.usage.completion_tokens if response.usage else 0
                )
                logger.info(
                    "OpenAI generate | model=%s prompt_tokens=%d "
                    "completion_tokens=%d latency_ms=%.1f",
                    self.model,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                )
                return text, latency_ms, prompt_tokens, completion_tokens
            except openai.RateLimitError as exc:
                last_exc = exc
                logger.warning(
                    "OpenAI rate limit (attempt %d/%d); backing off %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            except openai.APITimeoutError as exc:
                raise TimeoutError(
                    f"OpenAI request timed out after {self.timeout}s"
                ) from exc

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text response for the given prompt.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The generated text response.

        Raises:
            TimeoutError: If the API request times out.
            openai.RateLimitError: If all retries are exhausted.
        """
        text, _latency, _pt, _ct = self._chat(prompt)
        return text

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        """Generate a structured (JSON) response and parse it.

        Args:
            prompt: The input prompt text.
            schema: Unused schema hint (kept for interface compatibility).
            **kwargs: Additional keyword arguments.

        Returns:
            Parsed JSON dict from the LLM response, or empty dict on parse failure.

        Raises:
            TimeoutError: If the API request times out.
        """
        text = self.generate(prompt, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("OpenAI returned non-JSON response; returning empty dict")
            return {}

    def generate_with_confidence(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, float]:
        """Generate text and estimate a confidence score.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (generated_text, confidence_score) where confidence is
            in the range [0.0, 1.0].

        Raises:
            TimeoutError: If the API request times out.
        """
        text, _latency, _pt, _ct = self._chat(prompt)
        confidence = _estimate_confidence(text)
        return text, confidence
