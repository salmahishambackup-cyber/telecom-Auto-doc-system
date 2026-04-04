"""Ollama REST API LLM provider."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from llm.base import BaseLLMProvider
from llm.utils import _confidence_heuristic

logger = logging.getLogger(__name__)

# Seconds to wait during the one-time health check at init.
_HEALTH_CHECK_TIMEOUT: int = 3


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a locally-running Ollama instance.

    On construction a fast health check is performed against the ``/api/tags``
    endpoint.  If the server is unreachable the provider still initialises
    (so the fallback router can try it later) but ``generate()`` will
    immediately raise ``ConnectionError`` instead of blocking on a doomed
    HTTP request.
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
        self._server_reachable: bool = True

        # Fast health check — short timeout to avoid blocking startup.
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=_HEALTH_CHECK_TIMEOUT,
            )
            resp.raise_for_status()
            logger.debug("Ollama health check OK at %s", self.base_url)
        except Exception as exc:
            self._server_reachable = False
            logger.warning(
                "Ollama server unreachable at %s: %s — "
                "generate() will raise immediately until the server is back.",
                self.base_url,
                exc,
            )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to Ollama and return the response text.

        Raises:
            ConnectionError: If the server was unreachable during the health
                check or the request fails due to a connection issue.
            TimeoutError: If the request exceeds the configured timeout.
        """
        if not self._server_reachable:
            raise ConnectionError(
                f"Ollama server at {self.base_url} was unreachable during init"
            )

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
        except requests.exceptions.ConnectionError as exc:
            self._server_reachable = False
            raise ConnectionError(
                f"Ollama server at {self.base_url} is unreachable"
            ) from exc
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
        """Generate text and parse JSON from the response.

        Args:
            prompt: The full input prompt to send to the model.
            schema: Expected schema type (kept for interface compatibility).

        Returns:
            Parsed JSON response from the model.

        Raises:
            json.JSONDecodeError: If the model output cannot be parsed as JSON.
        """
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

        Returns:
            A ``(response_text, confidence)`` tuple where *confidence* is in
            the range [0, 1].
        """
        text = self.generate(prompt, **kwargs)
        confidence = _confidence_heuristic(text)
        return text, confidence
