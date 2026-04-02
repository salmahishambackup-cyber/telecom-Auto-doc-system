"""OpenAI SDK LLM provider."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _confidence_heuristic(text: str) -> float:
    """Score docstring quality based on presence of key sections."""
    if not text or not text.strip():
        return 0.0
    sections_found = 0
    stripped = text.strip()
    lines = stripped.splitlines()
    if lines and lines[0].strip():
        sections_found += 1
    if "Args:" in stripped or "Parameters:" in stripped or "param " in stripped:
        sections_found += 1
    if "Returns:" in stripped or "return " in stripped.lower():
        sections_found += 1
    return sections_found / 3.0


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        import openai  # noqa: PLC0415

        self.model = model
        self.timeout = timeout
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = openai.OpenAI(api_key=resolved_key, timeout=timeout)
        self._openai = openai

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send a chat completion request and return the assistant's reply.

        Automatically retries up to 3 times with exponential back-off on
        rate-limit errors.

        Raises
        ------
        openai.RateLimitError
            If all retries are exhausted.
        """
        import openai  # noqa: PLC0415

        max_retries = 3
        delay = 1.0
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text: str = response.choices[0].message.content or ""
                usage = response.usage
                if usage:
                    logger.debug(
                        "OpenAI generate: model=%s prompt_tokens=%d completion_tokens=%d",
                        self.model,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    )
                return text
            except openai.RateLimitError as exc:
                last_exc = exc
                logger.warning(
                    "OpenAI rate limit hit (attempt %d/%d), backing off %.1fs",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

        raise last_exc  # type: ignore[misc]

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        """Generate text and parse JSON from the response."""
        import json

        text = self.generate(prompt, **kwargs).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            text = "\n".join(inner)
        return json.loads(text)

    def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
        """Generate text and return it with a confidence score."""
        text = self.generate(prompt, **kwargs)
        confidence = _confidence_heuristic(text)
        return text, confidence
