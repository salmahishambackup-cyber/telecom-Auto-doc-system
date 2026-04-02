"""Anthropic SDK LLM provider."""
from __future__ import annotations

import logging
import os
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


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        timeout: int = 60,
        max_tokens: int = 2048,
    ) -> None:
        import anthropic  # noqa: PLC0415

        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=resolved_key, timeout=timeout)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send a message and return the assistant's reply.

        Extracts token usage from the response and logs it.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text: str = response.content[0].text if response.content else ""
        usage = response.usage
        if usage:
            logger.debug(
                "Anthropic generate: model=%s input_tokens=%d output_tokens=%d",
                self.model,
                usage.input_tokens,
                usage.output_tokens,
            )
        return text

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
