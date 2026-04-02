"""HuggingFace Transformers LLM provider."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from llm.base import BaseLLMProvider
from llm.utils import _confidence_heuristic

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """LLM provider backed by a locally-loaded HuggingFace Transformers model.

    This provider is well-suited for environments like Google Colab where
    Ollama cannot be installed.  It loads the model weights on construction
    and runs inference entirely in-process via the ``transformers`` library.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``"Qwen/Qwen2.5-Coder-1.5B-Instruct"``).
    max_new_tokens:
        Maximum number of tokens to generate per request.
    device:
        Inference device.  ``"auto"`` (the default) selects ``"cuda"`` when a
        GPU is available and ``"cpu"`` otherwise.  Pass an explicit string to
        override the auto-detection.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        max_new_tokens: int = 512,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HuggingFaceProvider requires the 'transformers', 'torch', and "
                "'accelerate' packages.  Install them with:\n"
                "  pip install transformers torch accelerate sentencepiece"
            ) from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Auto-detect compute device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "HuggingFaceProvider: loading model %s on %s", model_name, self.device
        )
        t0 = time.monotonic()

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "auto" if self.device == "cuda" else None

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            model = model.to(self.device)

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=None if self.device == "cuda" else self.device,
            max_new_tokens=max_new_tokens,
            max_length=None,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "HuggingFaceProvider: model loaded in %.1fs", elapsed
        )

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Run inference and return the generated text.

        The echoed prompt is stripped from the output so only the model's
        continuation is returned.

        Parameters
        ----------
        prompt:
            The full input prompt to send to the model.

        Returns
        -------
        str
            The model's generated continuation (prompt prefix removed).
        """
        t0 = time.monotonic()
        outputs = self._pipeline(prompt, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        full_text: str = outputs[0]["generated_text"]

        # Strip the echoed prompt from the output
        if full_text.startswith(prompt):
            response_text = full_text[len(prompt):]
        else:
            response_text = full_text

        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response_text) // 4
        logger.debug(
            "HuggingFace generate: model=%s prompt_tokens≈%d "
            "completion_tokens≈%d latency_ms=%.1f",
            self.model_name,
            prompt_tokens,
            completion_tokens,
            latency_ms,
        )
        return response_text

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        """Generate text and parse the result as JSON.

        Handles responses wrapped in markdown code fences (e.g. triple-backtick json).

        Parameters
        ----------
        prompt:
            The full input prompt to send to the model.
        schema:
            The expected schema type (currently unused; kept for interface
            compatibility).

        Returns
        -------
        dict
            Parsed JSON response from the model.

        Raises
        ------
        json.JSONDecodeError
            If the model output cannot be parsed as JSON.
        """
        text = self.generate(prompt, **kwargs)
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner: list[str] = []
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
        """Generate text and return it with a heuristic confidence score.

        Returns
        -------
        tuple[str, float]
            ``(response_text, confidence)`` where *confidence* is in [0, 1].
        """
        text = self.generate(prompt, **kwargs)
        confidence = _confidence_heuristic(text)
        return text, confidence
