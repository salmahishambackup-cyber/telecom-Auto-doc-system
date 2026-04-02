"""llm/anthropic_provider.py — Anthropic SDK LLM provider."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import anthropic

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_DOCSTRING_SECTIONS = ("Args", "Returns", "Summary")


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


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic Messages API.

    Args:
        model: Anthropic model name.
        api_key: Anthropic API key. Defaults to ``ANTHROPIC_API_KEY`` env var.
        timeout: Request timeout in seconds.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        timeout: int = 60,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=resolved_key, timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _messages_create(self, prompt: str) -> tuple[str, float, int, int]:
        """Call Anthropic Messages API.

        Args:
            prompt: The user prompt to send.

        Returns:
            Tuple of (response_text, latency_ms, input_tokens, output_tokens).

        Raises:
            TimeoutError: If the API request times out.
        """
        t0 = time.monotonic()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APITimeoutError as exc:
            raise TimeoutError(
                f"Anthropic request timed out after {self.timeout}s"
            ) from exc

        latency_ms = (time.monotonic() - t0) * 1000
        text = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        logger.info(
            "Anthropic generate | model=%s input_tokens=%d "
            "output_tokens=%d latency_ms=%.1f",
            self.model,
            input_tokens,
            output_tokens,
            latency_ms,
        )
        return text, latency_ms, input_tokens, output_tokens

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
        """
        text, _latency, _pt, _ct = self._messages_create(prompt)
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
            logger.warning("Anthropic returned non-JSON response; returning empty dict")
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
        text, _latency, _pt, _ct = self._messages_create(prompt)
        confidence = _estimate_confidence(text)
        return text, confidence
