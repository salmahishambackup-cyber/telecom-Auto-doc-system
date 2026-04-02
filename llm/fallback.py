"""llm/fallback.py — Confidence-based fallback router."""
from __future__ import annotations

import logging
from typing import Any

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class FallbackRouter:
    """Routes LLM calls to a primary provider with automatic fallback.

    If the primary provider returns a response with confidence below
    *confidence_threshold*, or raises any exception, the fallback provider
    is called.  The response with the higher confidence score is returned.
    """

    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback: BaseLLMProvider,
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
        """Call primary (and optionally fallback) and return the best result.

        Returns
        -------
        tuple[str, float, str, bool]
            ``(response_text, confidence, provider_name, fallback_was_used)``
        """
        primary_name = self._provider_name(self.primary)
        fallback_name = self._provider_name(self.fallback)

        primary_text: str = ""
        primary_confidence: float = 0.0
        primary_failed = False

        try:
            primary_text, primary_confidence = self.primary.generate_with_confidence(  # type: ignore[attr-defined]
                prompt, **kwargs
            )
        except Exception as exc:
            primary_failed = True
            logger.warning(
                "Primary provider %s failed: %s — triggering fallback.",
                primary_name,
                exc,
            )

        if not primary_failed and primary_confidence >= self.confidence_threshold:
            return primary_text, primary_confidence, primary_name, False

        # Call fallback
        try:
            fallback_text, fallback_confidence = self.fallback.generate_with_confidence(  # type: ignore[attr-defined]
                prompt, **kwargs
            )
        except Exception as exc:
            logger.warning("Fallback provider %s also failed: %s", fallback_name, exc)
            # Return whatever primary produced (may be empty)
            return primary_text, primary_confidence, primary_name, False

        reason = (
            f"primary exception"
            if primary_failed
            else f"primary_confidence={primary_confidence:.3f} < threshold={self.confidence_threshold}"
        )
        logger.info(
            "%s",
            {
                "event": "fallback_triggered",
                "reason": reason,
                "primary_confidence": primary_confidence,
                "fallback_confidence": fallback_confidence,
            },
        )

        # Return the higher-confidence result
        if not primary_failed and primary_confidence >= fallback_confidence:
            return primary_text, primary_confidence, primary_name, True
        return fallback_text, fallback_confidence, fallback_name, True
