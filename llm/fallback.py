"""llm/fallback.py — Confidence-based fallback router for LLM providers."""
from __future__ import annotations

import logging
from typing import Any

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class FallbackRouter:
    """Routes LLM calls through primary and optional fallback providers.

    If the primary provider returns a response below the confidence
    threshold, the fallback provider is called and the result with the
    higher confidence is returned.

    Args:
        primary: The primary LLM provider.
        fallback: Optional fallback LLM provider.
        confidence_threshold: Minimum confidence score to accept from the
            primary provider without triggering fallback.
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
        """Return a human-readable name for the provider.

        Args:
            provider: The provider instance to name.

        Returns:
            Class name of the provider.
        """
        return type(provider).__name__

    def generate_with_fallback(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, float, str, bool]:
        """Generate text using primary provider with fallback on low confidence.

        The method:
        1. Calls the primary provider.
        2. If confidence >= threshold, returns the primary result.
        3. If confidence < threshold and a fallback exists, calls the fallback.
        4. Returns whichever result has the higher confidence.
        5. If the primary raises an exception, calls the fallback and logs the reason.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional keyword arguments forwarded to providers.

        Returns:
            Tuple of (text, confidence, provider_name, fallback_used) where
            fallback_used is True when the fallback result was ultimately returned.

        Raises:
            RuntimeError: If both primary and fallback raise exceptions.
        """
        primary_name = self._provider_name(self.primary)
        primary_text: str | None = None
        primary_confidence: float = 0.0
        primary_failed = False

        # ------------------------------------------------------------------
        # Step 1: call primary
        # ------------------------------------------------------------------
        try:
            primary_text, primary_confidence = self.primary.generate_with_confidence(
                prompt, **kwargs
            )
        except Exception as exc:
            primary_failed = True
            logger.warning(
                "Primary provider %s raised %s: %s — attempting fallback",
                primary_name,
                type(exc).__name__,
                exc,
            )

        # ------------------------------------------------------------------
        # Step 2: check confidence threshold
        # ------------------------------------------------------------------
        if not primary_failed and primary_confidence >= self.confidence_threshold:
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        # ------------------------------------------------------------------
        # Step 3: attempt fallback
        # ------------------------------------------------------------------
        if self.fallback is None:
            if primary_failed:
                raise RuntimeError(
                    f"Primary provider {primary_name} failed and no fallback configured"
                )
            # Return primary result even if below threshold (no fallback available)
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        fallback_name = self._provider_name(self.fallback)
        reason = (
            "primary raised exception"
            if primary_failed
            else f"confidence {primary_confidence:.3f} < threshold {self.confidence_threshold}"
        )
        logger.info(
            "Fallback triggered (%s): switching from %s to %s",
            reason,
            primary_name,
            fallback_name,
        )

        try:
            fallback_text, fallback_confidence = self.fallback.generate_with_confidence(
                prompt, **kwargs
            )
        except Exception as exc:
            if primary_failed:
                raise RuntimeError(
                    f"Both primary ({primary_name}) and fallback ({fallback_name}) failed"
                ) from exc
            logger.warning(
                "Fallback provider %s raised %s: %s — using primary result",
                fallback_name,
                type(exc).__name__,
                exc,
            )
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        # ------------------------------------------------------------------
        # Step 4: return result with higher confidence
        # ------------------------------------------------------------------
        if not primary_failed and primary_confidence >= fallback_confidence:
            logger.info(
                "Using primary result (confidence %.3f >= fallback %.3f)",
                primary_confidence,
                fallback_confidence,
            )
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        logger.info(
            "Using fallback result (confidence %.3f > primary %.3f)",
            fallback_confidence,
            primary_confidence if not primary_failed else 0.0,
        )
        return fallback_text, fallback_confidence, fallback_name, True
