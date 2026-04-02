"""llm/anthropic_provider.py — Anthropic API provider."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_EXPECTED_SECTIONS = ("args", "returns", "summary")


def _estimate_confidence(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    lower = text.lower()
    found = sum(1 for section in _EXPECTED_SECTIONS if section in lower)
    return found / len(_EXPECTED_SECTIONS)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        timeout: int = 60,
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic>=0.18"
            ) from exc

        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=resolved_key, timeout=float(timeout))

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        start = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000.0
        text = response.content[0].text if response.content else ""
        usage = response.usage
        logger.debug(
            json.dumps(
                {
                    "provider": "anthropic",
                    "model": self.model,
                    "input_tokens": usage.input_tokens if usage else 0,
                    "output_tokens": usage.output_tokens if usage else 0,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )
        return text

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        text = self.generate(prompt, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Anthropic response is not valid JSON: %s", exc)
            return {}

    def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
        start = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000.0
        text = response.content[0].text if response.content else ""
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        logger.debug(
            json.dumps(
                {
                    "provider": "anthropic",
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )
        confidence = _estimate_confidence(text)
        return text, confidence
