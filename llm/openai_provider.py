"""llm/openai_provider.py — OpenAI API provider."""
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


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: pip install openai>=1.0"
            ) from exc

        self.model = model
        self.timeout = timeout
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = openai.OpenAI(api_key=resolved_key, timeout=float(timeout))
        self._openai = openai

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_backoff(self, prompt: str, **kwargs: Any) -> Any:
        """Call the API with exponential backoff on rate-limit errors."""
        import time as _time  # noqa: PLC0415

        max_retries = 3
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            except self._openai.RateLimitError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    logger.warning(
                        "OpenAI rate limit hit (attempt %d/%d), backing off %.1fs",
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    _time.sleep(delay)
                    delay *= 2
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        start = time.monotonic()
        response = self._call_with_backoff(prompt, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000.0
        text = response.choices[0].message.content or ""
        usage = response.usage
        logger.debug(
            json.dumps(
                {
                    "provider": "openai",
                    "model": self.model,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
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
            logger.warning("OpenAI response is not valid JSON: %s", exc)
            return {}

    def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
        start = time.monotonic()
        response = self._call_with_backoff(prompt, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000.0
        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        logger.debug(
            json.dumps(
                {
                    "provider": "openai",
                    "model": self.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )
        confidence = _estimate_confidence(text)
        return text, confidence
