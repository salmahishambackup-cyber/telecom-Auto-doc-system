"""Confidence-based fallback router for LLM providers."""
from __future__ import annotations

import logging
from typing import Any

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Number of consecutive fallback failures before the circuit breaker opens
# and the fallback is disabled for the remainder of the session.
_CIRCUIT_BREAKER_THRESHOLD: int = 3


class FallbackRouter:
    """Routes LLM requests with confidence-based fallback.

    If the primary provider's response confidence falls below
    *confidence_threshold*, the fallback provider is tried (when configured).
    The result with the higher confidence is returned.

    If the primary raises an exception, the fallback is tried automatically.

    Built-in **circuit breakers** protect against slow / unreachable
    providers.  The *primary* provider is disabled immediately on a
    connection error so that subsequent requests skip the doomed attempt
    and go straight to the fallback.  The *fallback* provider is disabled
    after *circuit_breaker_threshold* consecutive failures (or immediately
    on a connection error).

    Parameters
    ----------
    primary:
        The main LLM provider to use.
    fallback:
        An optional secondary provider to try on low confidence or errors.
    confidence_threshold:
        Minimum acceptable confidence score (0–1).  Responses below this
        threshold trigger the fallback provider.
    circuit_breaker_threshold:
        Number of consecutive fallback failures before the fallback is
        disabled.  Defaults to ``_CIRCUIT_BREAKER_THRESHOLD`` (3).
        Connection errors disable the fallback immediately regardless
        of this setting.
    """

    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback: BaseLLMProvider | None = None,
        confidence_threshold: float = 0.7,
        circuit_breaker_threshold: int = _CIRCUIT_BREAKER_THRESHOLD,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.confidence_threshold = confidence_threshold

        # Circuit-breaker state — fallback
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._consecutive_fallback_failures: int = 0
        self._fallback_disabled: bool = False

        # Circuit-breaker state — primary
        self._primary_disabled: bool = False

    def _provider_name(self, provider: BaseLLMProvider) -> str:
        return type(provider).__name__

    @property
    def fallback_disabled(self) -> bool:
        """Whether the circuit breaker has disabled the fallback provider."""
        return self._fallback_disabled

    @property
    def primary_disabled(self) -> bool:
        """Whether the circuit breaker has disabled the primary provider."""
        return self._primary_disabled

    def _open_primary_circuit(self, reason: str) -> None:
        """Disable the primary provider for this session."""
        if not self._primary_disabled:
            self._primary_disabled = True
            logger.warning(
                "Circuit breaker OPEN — primary provider disabled: %s",
                reason,
            )

    def _open_circuit(self, reason: str) -> None:
        """Permanently disable the fallback for this session."""
        if not self._fallback_disabled:
            self._fallback_disabled = True
            logger.warning(
                "Circuit breaker OPEN — fallback provider disabled: %s",
                reason,
            )

    @staticmethod
    def _is_connection_error(exc: BaseException) -> bool:
        """Return True when *exc* indicates the remote server is unreachable."""
        # Built-in ConnectionError (a subclass of OSError) covers most cases.
        # requests.exceptions.ConnectionError also inherits from OSError, so
        # the isinstance check covers both.
        if isinstance(exc, OSError):
            return True
        # Some HTTP adapters wrap connection errors in a generic exception
        # whose __cause__ is an OSError.
        if exc.__cause__ is not None and isinstance(exc.__cause__, OSError):
            return True
        return False

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

        # --- Try primary (skip if circuit breaker is open) ---
        if self._primary_disabled:
            primary_failed = True
            logger.debug(
                "Primary provider %s skipped (circuit breaker open).",
                primary_name,
            )
        else:
            try:
                primary_text, primary_confidence = self.primary.generate_with_confidence(
                    prompt, **kwargs
                )
            except Exception as exc:
                primary_failed = True
                if self._is_connection_error(exc):
                    self._open_primary_circuit(
                        f"{primary_name} unreachable ({type(exc).__name__})"
                    )
                logger.warning(
                    "Primary provider %s raised %s: %s — trying fallback.",
                    primary_name,
                    type(exc).__name__,
                    exc,
                )

        # Sufficient confidence from primary → return immediately
        if not primary_failed and primary_confidence >= self.confidence_threshold:
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        # No fallback available → return whatever primary gave us
        if self.fallback is None or self._fallback_disabled:
            if primary_failed:
                raise RuntimeError(
                    f"Primary provider {primary_name} failed and no fallback available."
                )
            return primary_text, primary_confidence, primary_name, False  # type: ignore[return-value]

        # --- Try fallback ---
        fallback_name = self._provider_name(self.fallback)
        try:
            fallback_text, fallback_confidence = self.fallback.generate_with_confidence(
                prompt, **kwargs
            )
        except Exception as exc:
            # --- Circuit-breaker logic ---
            if self._is_connection_error(exc):
                self._open_circuit(
                    f"{fallback_name} unreachable ({type(exc).__name__})"
                )
            else:
                self._consecutive_fallback_failures += 1
                if self._consecutive_fallback_failures >= self._circuit_breaker_threshold:
                    self._open_circuit(
                        f"{self._consecutive_fallback_failures} consecutive failures"
                    )

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

        # Fallback succeeded — reset failure counter
        self._consecutive_fallback_failures = 0

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
