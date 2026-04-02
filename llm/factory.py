"""Factory helpers for instantiating LLM providers from a config dict."""
from __future__ import annotations

import logging
from typing import Any

from llm.base import BaseLLMProvider
from llm.fallback import FallbackRouter

logger = logging.getLogger(__name__)


def create_provider(config: dict[str, Any]) -> BaseLLMProvider:
    """Create an LLM provider from a config dict.

    Reads ``config["llm"]["provider"]`` to select the backend, then passes
    the provider-specific sub-section as keyword arguments to the constructor.

    Parameters
    ----------
    config:
        Top-level config dict (as loaded from ``config.yaml``).

    Returns
    -------
    BaseLLMProvider
        A fully initialised provider instance.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    """
    llm_config: dict[str, Any] = config.get("llm", {})
    provider_name: str = llm_config.get("provider", "ollama")

    if provider_name == "ollama":
        from llm.ollama_provider import OllamaProvider  # noqa: PLC0415

        settings = llm_config.get("ollama", {})
        return OllamaProvider(**settings)

    if provider_name == "huggingface":
        from llm.huggingface_provider import HuggingFaceProvider  # noqa: PLC0415

        settings = llm_config.get("huggingface", {})
        return HuggingFaceProvider(**settings)

    raise ValueError(f"Unknown LLM provider: {provider_name!r}")


def create_router(config: dict[str, Any]) -> FallbackRouter:
    """Create a :class:`FallbackRouter` from a config dict.

    Instantiates the primary provider via :func:`create_provider`.  If
    ``config["llm"]["fallback"]["enabled"]`` is ``true``, a secondary provider
    is also instantiated and attached as the fallback.

    Parameters
    ----------
    config:
        Top-level config dict (as loaded from ``config.yaml``).

    Returns
    -------
    FallbackRouter
        A router wrapping the primary (and optionally fallback) provider.
    """
    primary = create_provider(config)

    llm_config: dict[str, Any] = config.get("llm", {})
    fallback_cfg: dict[str, Any] = llm_config.get("fallback", {})
    confidence_threshold: float = fallback_cfg.get("confidence_threshold", 0.7)

    fallback_provider: BaseLLMProvider | None = None
    if fallback_cfg.get("enabled", False):
        fallback_name: str = fallback_cfg.get("provider", "")
        if fallback_name:
            # Temporarily override the provider name to build the fallback
            fallback_config: dict[str, Any] = {
                **config,
                "llm": {**llm_config, "provider": fallback_name},
            }
            try:
                fallback_provider = create_provider(fallback_config)
                logger.info(
                    "Fallback provider %r initialised successfully.", fallback_name
                )
            except Exception as exc:
                logger.warning(
                    "Could not initialise fallback provider %r: %s", fallback_name, exc
                )

    return FallbackRouter(
        primary=primary,
        fallback=fallback_provider,
        confidence_threshold=confidence_threshold,
    )
