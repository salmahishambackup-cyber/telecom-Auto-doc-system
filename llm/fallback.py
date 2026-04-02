"""Confidence-based fallback router for LLM providers."""
from __future__ import annotations

import logging
from typing import Any

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class FallbackRouter:
    """Routes LLM requests with confidence-based fallback.

    If the primary provider's response confidence falls below
    *confidence_threshold*, the fallback provider is tried (when configured).
    The result with the higher confidence is returned.

    If the primary raises an exception, the fallback is tried automatically.

    Parameters
    ----------
    primary:
        The main LLM provider to use.
    fallback:
        An optional secondary provider to try on low confidence or errors.
    confidence_threshold:
        Minimum acceptable confidence score (0–1).  Responses below this
        threshold trigger the fallback provider.
    """

    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback: BaseLLMProvider | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.confidence_threshold = confidence_threshold

    def _provider_name(self, provider: BaseLLMProvider) -> str:
        return type(provider).__name__

    def generate_with_fallback(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, float, str, bool]:
        """Generate text with automatic fallback on low confidence or errors.

        Returns
        -------
        tuple[str, float, str, bool]
            ``(text, confidence, provider_name, fallback_was_used)``
        """
        primary_name = self._provider_name(self.primary)
        primary_text: str | None = None
        primary_confidence: float = 0.0
        primary_failed = False

        # --- Try primary ---
        try:
            primary_text, primary_confidence = self.primary.generate_with_confidence(
                prompt, **kwargs
            )
        except Exception as exc:
            primary_failed = True
            logger.warning(
                "Primary provider %s raised %s: %s — trying fallback.",
                primary_name,
                type(exc).__name__,
                exc,
            )

        # Sufficient confidence from primary → return immediately
        if not primary_failed and primary_confidence >= self.confidence_threshold:
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        # No fallback configured → return whatever primary gave us
        if self.fallback is None:
            if primary_failed:
                raise RuntimeError(
                    f"Primary provider {primary_name} failed and no fallback configured."
                )
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        # --- Try fallback ---
        fallback_name = self._provider_name(self.fallback)
        try:
            fallback_text, fallback_confidence = self.fallback.generate_with_confidence(
                prompt, **kwargs
            )
        except Exception as exc:
            logger.warning(
                "Fallback provider %s raised %s: %s.",
                fallback_name,
                type(exc).__name__,
                exc,
            )
            if primary_failed:
                raise
            # Fallback failed but primary succeeded (just low confidence)
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        # Return the result with higher confidence
        if primary_failed or fallback_confidence >= primary_confidence:
            logger.debug(
                "Fallback %s used: confidence %.2f vs primary %.2f",
                fallback_name,
                fallback_confidence,
                primary_confidence,
            )
            return fallback_text, fallback_confidence, fallback_name, True

        return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]
