"""llm/ollama_provider.py — Ollama REST API provider."""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_EXPECTED_SECTIONS = ("args", "returns", "summary")


def _estimate_confidence(text: str) -> float:
    """Heuristically estimate docstring quality.

    Checks for the presence of expected Google-style docstring sections.
    Returns a score in [0.0, 1.0].
    """
    if not text or not text.strip():
        return 0.0
    lower = text.lower()
    found = sum(1 for section in _EXPECTED_SECTIONS if section in lower)
    return found / len(_EXPECTED_SECTIONS)


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a local Ollama instance."""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._check_health()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _check_health(self) -> None:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
            logger.debug("Ollama health check passed (%s)", self.base_url)
        except Exception as exc:
            logger.warning(
                "Ollama server unreachable at %s: %s — provider will fail at generation time.",
                self.base_url,
                exc,
            )

    # ------------------------------------------------------------------
    # Core HTTP call
    # ------------------------------------------------------------------

    def _post_generate(self, prompt: str, **kwargs: Any) -> tuple[str, int, int, float]:
        """Call POST /api/generate and return (text, prompt_tokens, completion_tokens, latency_ms)."""
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                **{k: v for k, v in kwargs.items() if k not in ("model", "prompt", "stream")},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except TimeoutError as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s"
            ) from exc
        except urllib.error.URLError as exc:
            if "timed out" in str(exc).lower():
                raise TimeoutError(
                    f"Ollama request timed out after {self.timeout}s"
                ) from exc
            raise

        latency_ms = (time.monotonic() - start) * 1000.0
        data = json.loads(raw)
        text = data.get("response", "")

        # Ollama may return token counts in the response dict
        prompt_tokens: int = data.get("prompt_eval_count", len(prompt) // 4)
        completion_tokens: int = data.get("eval_count", len(text) // 4)

        logger.debug(
            json.dumps(
                {
                    "provider": "ollama",
                    "model": self.model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )
        return text, prompt_tokens, completion_tokens, latency_ms

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        text, _, _, _ = self._post_generate(prompt, **kwargs)
        return text

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        text = self.generate(prompt, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Ollama response is not valid JSON: %s", exc)
            return {}

    def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
        """Generate text and return (text, confidence_score)."""
        text, _, _, _ = self._post_generate(prompt, **kwargs)
        confidence = _estimate_confidence(text)
        return text, confidence
