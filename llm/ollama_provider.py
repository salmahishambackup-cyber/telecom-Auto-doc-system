"""Ollama REST API LLM provider."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _confidence_heuristic(text: str) -> float:
    """Score docstring quality based on presence of key sections.

    Checks for a summary line, 'Args:' section, and 'Returns:' section.
    Returns a score between 0.0 and 1.0.
    """
    if not text or not text.strip():
        return 0.0
    sections_found = 0
    stripped = text.strip()
    # Summary line: any non-empty first line
    lines = stripped.splitlines()
    if lines and lines[0].strip():
        sections_found += 1
    if "Args:" in stripped or "Parameters:" in stripped or "param " in stripped:
        sections_found += 1
    if "Returns:" in stripped or "return " in stripped.lower():
        sections_found += 1
    return sections_found / 3.0


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a locally-running Ollama instance."""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Health check — log a warning if the server is unreachable, don't crash.
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            logger.debug("Ollama health check OK at %s", self.base_url)
        except Exception as exc:
            logger.warning(
                "Ollama server unreachable at %s: %s — continuing anyway.",
                self.base_url,
                exc,
            )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to Ollama and return the response text.

        Raises
        ------
        TimeoutError
            If the request exceeds the configured timeout.
        """
        t0 = time.monotonic()
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s"
            ) from exc

        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        response_text: str = data.get("response", "")

        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response_text) // 4
        logger.debug(
            "Ollama generate: model=%s prompt_tokens≈%d completion_tokens≈%d latency_ms=%.1f",
            self.model_name,
            prompt_tokens,
            completion_tokens,
            latency_ms,
        )
        return response_text

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        """Generate text and parse JSON from the response."""
        text = self.generate(prompt, **kwargs)
        # Extract JSON from the response (handle markdown code blocks)
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove opening and closing fences
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
        """Generate text and return it with a confidence score.

        Returns
        -------
        tuple[str, float]
            ``(response_text, confidence)`` where confidence is in [0, 1].
        """
        text = self.generate(prompt, **kwargs)
        confidence = _confidence_heuristic(text)
        return text, confidence
