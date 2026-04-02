"""llm/ollama_provider.py — Ollama REST API LLM provider."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

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
    # Summary is present if there's any non-empty first line
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    summary_present = len(lines) > 0
    args_present = "Args:" in text or "Arguments:" in text
    returns_present = "Returns:" in text or "Return:" in text
    found = sum([summary_present, args_present, returns_present])
    return round(found / len(_DOCSTRING_SECTIONS), 4)


class OllamaProvider(BaseLLMProvider):
    """LLM provider that calls the Ollama local REST API.

    Args:
        model_name: Name of the Ollama model to use.
        base_url: Base URL for the Ollama API server.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Health check — warn but don't crash if unreachable
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            logger.info("Ollama health check OK (model=%s)", self.model_name)
        except Exception as exc:
            logger.warning(
                "Ollama health check failed — provider may be unavailable: %s", exc
            )

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _post_generate(self, prompt: str, **kwargs: Any) -> tuple[str, float, int, int]:
        """POST to /api/generate and return (text, latency_ms, prompt_tokens, completion_tokens).

        Args:
            prompt: The prompt text to send.
            **kwargs: Additional keyword arguments (unused, for API compatibility).

        Returns:
            Tuple of (response_text, latency_ms, prompt_tokens, completion_tokens).

        Raises:
            TimeoutError: If the request exceeds the configured timeout.
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        t0 = time.monotonic()
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
        text: str = data.get("response", "")

        # Ollama returns token counts in the response
        prompt_tokens: int = data.get("prompt_eval_count", len(prompt.split()))
        completion_tokens: int = data.get("eval_count", len(text.split()))

        logger.info(
            "Ollama generate | model=%s prompt_tokens=%d completion_tokens=%d latency_ms=%.1f",
            self.model_name,
            prompt_tokens,
            completion_tokens,
            latency_ms,
        )
        return text, latency_ms, prompt_tokens, completion_tokens

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
            TimeoutError: If the request exceeds the configured timeout.
        """
        text, _latency, _pt, _ct = self._post_generate(prompt, **kwargs)
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
            TimeoutError: If the request exceeds the configured timeout.
        """
        text = self.generate(prompt, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Ollama returned non-JSON response; returning empty dict")
            return {}

    def generate_with_confidence(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, float]:
        """Generate text and estimate a confidence score.

        Confidence is estimated heuristically based on whether the response
        contains expected Google-style docstring sections.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (generated_text, confidence_score) where confidence is
            in the range [0.0, 1.0].

        Raises:
            TimeoutError: If the request exceeds the configured timeout.
        """
        text, _latency, _pt, _ct = self._post_generate(prompt, **kwargs)
        confidence = _estimate_confidence(text)
        return text, confidence
