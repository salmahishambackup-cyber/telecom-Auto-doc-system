"""tests/test_phase3.py — Comprehensive tests for Phase 3 (Docstring Generation).

All LLM calls are mocked — no real API access required.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm.base import BaseLLMProvider
from llm.fallback import FallbackRouter
from models.schemas import (
    ASTNode,
    ASTNodes,
    CallGraph,
    CallGraphEntry,
    CalleeRef,
    CallerRef,
    DocstringEntry,
    DocstringFailure,
    Docstrings,
    ModuleDocstring,
    NodeType,
    PhaseArtifacts,
)
from phases.phase3_docstrings import (
    _build_callee_list,
    _build_caller_list,
    _build_function_id,
    _is_malformed,
    run_phase3,
)


# ---------------------------------------------------------------------------
# Helper: FakeLLMProvider
# ---------------------------------------------------------------------------

FULL_DOCSTRING = """\
Perform the main computation.

Args:
    x: Input value.
    y: Another value.

Returns:
    The sum of x and y.\
"""

LOW_CONFIDENCE_DOCSTRING = "Does something."

EMPTY_DOCSTRING = ""


class FakeLLMProvider(BaseLLMProvider):
    """A test double that returns canned responses without real API calls.

    Args:
        response: The canned text response to return.
        confidence: The canned confidence score to return.
        should_raise: If set, raising this exception on every call.
    """

    def __init__(
        self,
        response: str = FULL_DOCSTRING,
        confidence: float = 0.9,
        should_raise: Exception | None = None,
    ) -> None:
        self.response = response
        self.confidence = confidence
        self.should_raise = should_raise
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.call_count += 1
        if self.should_raise:
            raise self.should_raise
        return self.response

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        self.call_count += 1
        if self.should_raise:
            raise self.should_raise
        try:
            return json.loads(self.response)
        except json.JSONDecodeError:
            return {}

    def generate_with_confidence(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, float]:
        self.call_count += 1
        if self.should_raise:
            raise self.should_raise
        return self.response, self.confidence


# ---------------------------------------------------------------------------
# Confidence calculation tests
# ---------------------------------------------------------------------------


class TestConfidenceCalculation:
    """Tests for the heuristic confidence scoring in providers."""

    def test_full_docstring_high_confidence(self) -> None:
        from llm.ollama_provider import _estimate_confidence

        score = _estimate_confidence(FULL_DOCSTRING)
        assert score == 1.0

    def test_missing_args_and_returns_lower_confidence(self) -> None:
        from llm.ollama_provider import _estimate_confidence

        score = _estimate_confidence("Just a summary line.")
        assert 0.0 < score < 1.0

    def test_empty_string_zero_confidence(self) -> None:
        from llm.ollama_provider import _estimate_confidence

        assert _estimate_confidence("") == 0.0
        assert _estimate_confidence("   ") == 0.0

    def test_openai_provider_confidence(self) -> None:
        from llm.openai_provider import _estimate_confidence

        assert _estimate_confidence(FULL_DOCSTRING) == 1.0
        assert _estimate_confidence("") == 0.0

    def test_anthropic_provider_confidence(self) -> None:
        from llm.anthropic_provider import _estimate_confidence

        assert _estimate_confidence(FULL_DOCSTRING) == 1.0
        assert _estimate_confidence("") == 0.0


# ---------------------------------------------------------------------------
# Ollama provider tests
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    """Tests for OllamaProvider (all HTTP calls mocked)."""

    def _make_provider(self, mock_get: MagicMock) -> Any:
        """Create OllamaProvider with mocked health check."""
        from llm.ollama_provider import OllamaProvider

        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        return OllamaProvider()

    @patch("llm.ollama_provider.requests.get")
    @patch("llm.ollama_provider.requests.post")
    def test_generate_returns_text(self, mock_post: MagicMock, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": FULL_DOCSTRING,
            "prompt_eval_count": 50,
            "eval_count": 30,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        provider = self._make_provider(mock_get)
        result = provider.generate("some prompt")
        assert result == FULL_DOCSTRING
        mock_post.assert_called_once()

    @patch("llm.ollama_provider.requests.get")
    @patch("llm.ollama_provider.requests.post")
    def test_generate_with_confidence_scores(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": FULL_DOCSTRING}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        provider = self._make_provider(mock_get)
        text, confidence = provider.generate_with_confidence("prompt")
        assert text == FULL_DOCSTRING
        assert confidence == 1.0

    @patch("llm.ollama_provider.requests.get")
    @patch("llm.ollama_provider.requests.post")
    def test_timeout_raises_timeout_error(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        import requests as req

        mock_post.side_effect = req.exceptions.Timeout()
        provider = self._make_provider(mock_get)
        with pytest.raises(TimeoutError):
            provider.generate("prompt")

    @patch("llm.ollama_provider.requests.get")
    def test_health_check_warns_on_failure(self, mock_get: MagicMock) -> None:
        """Health check failure must not crash the constructor."""
        from llm.ollama_provider import OllamaProvider

        mock_get.side_effect = Exception("connection refused")
        # Should NOT raise
        provider = OllamaProvider()
        assert provider is not None

    @patch("llm.ollama_provider.requests.get")
    @patch("llm.ollama_provider.requests.post")
    def test_generate_structured_parses_json(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        payload = '{"key": "value"}'
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": payload}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        provider = self._make_provider(mock_get)
        result = provider.generate_structured("prompt", dict)
        assert result == {"key": "value"}

    @patch("llm.ollama_provider.requests.get")
    @patch("llm.ollama_provider.requests.post")
    def test_generate_structured_returns_empty_on_bad_json(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "not json"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        provider = self._make_provider(mock_get)
        result = provider.generate_structured("prompt", dict)
        assert result == {}


# ---------------------------------------------------------------------------
# OpenAI provider tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Tests for OpenAIProvider (SDK mocked)."""

    def _make_mock_response(self, text: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        choice = MagicMock()
        choice.message.content = text
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    @patch("llm.openai_provider.openai.OpenAI")
    def test_generate_returns_text(self, mock_openai_cls: MagicMock) -> None:
        from llm.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            FULL_DOCSTRING
        )

        provider = OpenAIProvider(api_key="fake-key")
        result = provider.generate("prompt")
        assert result == FULL_DOCSTRING

    @patch("llm.openai_provider.openai.OpenAI")
    def test_rate_limit_retries(self, mock_openai_cls: MagicMock) -> None:
        import openai as oai
        from llm.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Fail twice, succeed on third
        mock_client.chat.completions.create.side_effect = [
            oai.RateLimitError("rate limit", response=MagicMock(), body={}),
            oai.RateLimitError("rate limit", response=MagicMock(), body={}),
            self._make_mock_response(FULL_DOCSTRING),
        ]

        with patch("llm.openai_provider.time.sleep"):
            provider = OpenAIProvider(api_key="fake-key")
            result = provider.generate("prompt")

        assert result == FULL_DOCSTRING
        assert mock_client.chat.completions.create.call_count == 3

    @patch("llm.openai_provider.openai.OpenAI")
    def test_timeout_raises_timeout_error(self, mock_openai_cls: MagicMock) -> None:
        import openai as oai
        from llm.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = oai.APITimeoutError(
            request=MagicMock()
        )

        provider = OpenAIProvider(api_key="fake-key")
        with pytest.raises(TimeoutError):
            provider.generate("prompt")

    @patch("llm.openai_provider.openai.OpenAI")
    def test_token_extraction(self, mock_openai_cls: MagicMock) -> None:
        from llm.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            FULL_DOCSTRING, prompt_tokens=42, completion_tokens=17
        )

        provider = OpenAIProvider(api_key="fake-key")
        text, latency, pt, ct = provider._chat("prompt")
        assert pt == 42
        assert ct == 17


# ---------------------------------------------------------------------------
# Anthropic provider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Tests for AnthropicProvider (SDK mocked)."""

    def _make_mock_response(self, text: str, input_tokens: int = 10, output_tokens: int = 20) -> MagicMock:
        usage = MagicMock()
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        content_block = MagicMock()
        content_block.text = text
        response = MagicMock()
        response.content = [content_block]
        response.usage = usage
        return response

    @patch("llm.anthropic_provider.anthropic.Anthropic")
    def test_generate_returns_text(self, mock_anthropic_cls: MagicMock) -> None:
        from llm.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(
            FULL_DOCSTRING
        )

        provider = AnthropicProvider(api_key="fake-key")
        result = provider.generate("prompt")
        assert result == FULL_DOCSTRING

    @patch("llm.anthropic_provider.anthropic.Anthropic")
    def test_timeout_raises_timeout_error(self, mock_anthropic_cls: MagicMock) -> None:
        import anthropic as ant
        from llm.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = ant.APITimeoutError(
            request=MagicMock()
        )

        provider = AnthropicProvider(api_key="fake-key")
        with pytest.raises(TimeoutError):
            provider.generate("prompt")

    @patch("llm.anthropic_provider.anthropic.Anthropic")
    def test_token_extraction(self, mock_anthropic_cls: MagicMock) -> None:
        from llm.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(
            FULL_DOCSTRING, input_tokens=55, output_tokens=22
        )

        provider = AnthropicProvider(api_key="fake-key")
        _text, _latency, pt, ct = provider._messages_create("prompt")
        assert pt == 55
        assert ct == 22


# ---------------------------------------------------------------------------
# FallbackRouter tests
# ---------------------------------------------------------------------------


class TestFallbackRouter:
    """Tests for the FallbackRouter confidence-based routing logic."""

    def test_high_confidence_no_fallback(self) -> None:
        primary = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        fallback = FakeLLMProvider(response="Fallback.", confidence=0.5)
        router = FallbackRouter(primary, fallback, confidence_threshold=0.7)

        text, conf, provider, fb_used = router.generate_with_fallback("prompt")
        assert text == FULL_DOCSTRING
        assert conf == 0.9
        assert not fb_used
        assert fallback.call_count == 0

    def test_low_confidence_triggers_fallback(self) -> None:
        primary = FakeLLMProvider(response=LOW_CONFIDENCE_DOCSTRING, confidence=0.3)
        fallback = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        router = FallbackRouter(primary, fallback, confidence_threshold=0.7)

        text, conf, provider, fb_used = router.generate_with_fallback("prompt")
        assert fb_used
        assert conf == 0.9
        assert text == FULL_DOCSTRING

    def test_exception_triggers_fallback(self) -> None:
        primary = FakeLLMProvider(should_raise=RuntimeError("API down"))
        fallback = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        router = FallbackRouter(primary, fallback, confidence_threshold=0.7)

        text, conf, provider, fb_used = router.generate_with_fallback("prompt")
        assert fb_used
        assert text == FULL_DOCSTRING

    def test_returns_higher_confidence(self) -> None:
        # Primary: 0.5 (below threshold), Fallback: 0.4 — primary still wins
        primary = FakeLLMProvider(response="Primary.", confidence=0.5)
        fallback = FakeLLMProvider(response="Fallback.", confidence=0.4)
        router = FallbackRouter(primary, fallback, confidence_threshold=0.7)

        text, conf, provider, fb_used = router.generate_with_fallback("prompt")
        assert text == "Primary."
        assert conf == 0.5
        assert not fb_used

    def test_no_fallback_returns_primary_even_below_threshold(self) -> None:
        primary = FakeLLMProvider(response="Low.", confidence=0.2)
        router = FallbackRouter(primary, None, confidence_threshold=0.7)

        text, conf, provider, fb_used = router.generate_with_fallback("prompt")
        assert text == "Low."
        assert not fb_used

    def test_both_fail_raises_runtime_error(self) -> None:
        primary = FakeLLMProvider(should_raise=RuntimeError("primary down"))
        fallback = FakeLLMProvider(should_raise=RuntimeError("fallback down"))
        router = FallbackRouter(primary, fallback, confidence_threshold=0.7)

        with pytest.raises(RuntimeError, match="Both primary"):
            router.generate_with_fallback("prompt")

    def test_no_fallback_primary_exception_raises(self) -> None:
        primary = FakeLLMProvider(should_raise=RuntimeError("API down"))
        router = FallbackRouter(primary, None, confidence_threshold=0.7)

        with pytest.raises(RuntimeError):
            router.generate_with_fallback("prompt")


# ---------------------------------------------------------------------------
# Phase 3 integration tests
# ---------------------------------------------------------------------------


def _make_ast_nodes(*specs: tuple[str, str, NodeType, int, int]) -> ASTNodes:
    """Build ASTNodes from simple spec tuples."""
    nodes = []
    for file_path, name, ntype, ls, le in specs:
        nodes.append(
            ASTNode(
                file_path=file_path,
                node_type=ntype,
                name=name,
                line_start=ls,
                line_end=le,
            )
        )
    return ASTNodes(nodes=nodes)


def _make_artifacts_with_source(
    tmp_path: Path,
    functions: list[tuple[str, str]],
) -> tuple[PhaseArtifacts, list[str]]:
    """Write Python files and build matching ASTNodes + PhaseArtifacts.

    Args:
        tmp_path: Temporary directory.
        functions: List of (function_name, source_code) pairs.

    Returns:
        Tuple of (PhaseArtifacts, [file_paths]).
    """
    file_path = str(tmp_path / "module.py")
    source_lines = []
    nodes = []
    for func_name, src in functions:
        line_start = len(source_lines) + 1
        for line in src.splitlines():
            source_lines.append(line)
        line_end = len(source_lines)
        nodes.append(
            ASTNode(
                file_path=file_path,
                node_type=NodeType.FUNCTION,
                name=func_name,
                line_start=line_start,
                line_end=line_end,
            )
        )
    Path(file_path).write_text("\n".join(source_lines))
    artifacts = PhaseArtifacts(ast_nodes=ASTNodes(nodes=nodes))
    return artifacts, [file_path]


class TestPhase3Integration:
    """End-to-end integration tests for run_phase3."""

    def _run(
        self,
        artifacts: PhaseArtifacts,
        config: dict,
        fake_provider: FakeLLMProvider,
    ) -> Docstrings:
        with patch(
            "phases.phase3_docstrings._build_llm_provider",
            return_value=fake_provider,
        ), patch(
            "phases.phase3_docstrings._build_fallback_provider",
            return_value=None,
        ):
            run_phase3(config=config, artifacts=artifacts)
        assert artifacts.docstrings is not None
        return artifacts.docstrings

    def test_generates_docstrings_for_all_functions(self, tmp_path: Path) -> None:
        artifacts, _ = _make_artifacts_with_source(
            tmp_path,
            [
                ("add", "def add(x, y):\n    return x + y"),
                ("sub", "def sub(x, y):\n    return x - y"),
            ],
        )
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}
        fake = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        result = self._run(artifacts, config, fake)

        assert result.total_functions == 2
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.entries) == 2

    def test_generates_module_docstrings(self, tmp_path: Path) -> None:
        artifacts, fps = _make_artifacts_with_source(
            tmp_path, [("func", "def func():\n    pass")]
        )
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}
        fake = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        result = self._run(artifacts, config, fake)

        assert len(result.module_docstrings) >= 1

    def test_records_failure_on_file_read_error(self, tmp_path: Path) -> None:
        # Create node pointing to a non-existent file
        node = ASTNode(
            file_path=str(tmp_path / "missing.py"),
            node_type=NodeType.FUNCTION,
            name="ghost",
            line_start=1,
            line_end=5,
        )
        artifacts = PhaseArtifacts(ast_nodes=ASTNodes(nodes=[node]))
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}
        fake = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        result = self._run(artifacts, config, fake)

        assert result.failed >= 1
        assert any(f.error_type == "file_read_error" for f in result.failures)

    def test_correct_function_id_format(self, tmp_path: Path) -> None:
        artifacts, _ = _make_artifacts_with_source(
            tmp_path, [("my_func", "def my_func():\n    pass")]
        )
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}
        fake = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        result = self._run(artifacts, config, fake)

        assert len(result.entries) == 1
        entry = result.entries[0]
        assert "my_func" in entry.function_id
        assert "::" in entry.function_id

    def test_prompt_includes_callers_and_callees(self, tmp_path: Path) -> None:
        """Verify the generated prompt contains caller/callee info."""
        file_path = str(tmp_path / "a.py")
        Path(file_path).write_text("def foo():\n    bar()\n")

        node = ASTNode(
            file_path=file_path,
            node_type=NodeType.FUNCTION,
            name="foo",
            line_start=1,
            line_end=2,
        )
        call_graph = CallGraph(
            entries=[
                CallGraphEntry(
                    caller=CallerRef(file=file_path, function="foo"),
                    callees=[CalleeRef(file=file_path, function="bar", line=2)],
                )
            ]
        )
        artifacts = PhaseArtifacts(
            ast_nodes=ASTNodes(nodes=[node]), call_graph=call_graph
        )
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}

        captured_prompts: list[str] = []

        class CapturingFake(FakeLLMProvider):
            def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
                captured_prompts.append(prompt)
                return super().generate_with_confidence(prompt, **kwargs)

        fake = CapturingFake(response=FULL_DOCSTRING, confidence=0.9)
        with patch(
            "phases.phase3_docstrings._build_llm_provider", return_value=fake
        ), patch(
            "phases.phase3_docstrings._build_fallback_provider", return_value=None
        ):
            run_phase3(config=config, artifacts=artifacts)

        assert len(captured_prompts) >= 1
        func_prompt = next(
            (p for p in captured_prompts if "foo" in p or "Function source" in p), None
        )
        assert func_prompt is not None
        assert "bar" in func_prompt

    def test_saves_artifact_to_disk(self, tmp_path: Path) -> None:
        artifacts, _ = _make_artifacts_with_source(
            tmp_path, [("func", "def func():\n    pass")]
        )
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}
        fake = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        self._run(artifacts, config, fake)

        artifact_file = tmp_path / "docstrings.json"
        assert artifact_file.exists()
        data = json.loads(artifact_file.read_text())
        assert "entries" in data

    def test_summary_stats_correct(self, tmp_path: Path) -> None:
        artifacts, _ = _make_artifacts_with_source(
            tmp_path,
            [
                ("f1", "def f1():\n    pass"),
                ("f2", "def f2():\n    pass"),
            ],
        )
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}
        fake = FakeLLMProvider(response=FULL_DOCSTRING, confidence=1.0)
        result = self._run(artifacts, config, fake)

        assert result.total_functions == 2
        assert result.successful == 2
        assert result.failed == 0
        assert result.average_confidence == 1.0

    def test_handles_malformed_response_with_failure(self, tmp_path: Path) -> None:
        artifacts, _ = _make_artifacts_with_source(
            tmp_path, [("f", "def f():\n    pass")]
        )
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}
        fake = FakeLLMProvider(response=EMPTY_DOCSTRING, confidence=0.0)
        result = self._run(artifacts, config, fake)

        assert result.failed >= 1
        assert any(f.error_type == "malformed_response" for f in result.failures)

    def test_fallback_count_tracked(self, tmp_path: Path) -> None:
        artifacts, _ = _make_artifacts_with_source(
            tmp_path, [("f", "def f():\n    pass")]
        )
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}

        primary = FakeLLMProvider(response=LOW_CONFIDENCE_DOCSTRING, confidence=0.3)
        fallback = FakeLLMProvider(response=FULL_DOCSTRING, confidence=0.9)
        router = FallbackRouter(primary, fallback, confidence_threshold=0.7)

        with patch(
            "phases.phase3_docstrings._build_llm_provider", return_value=primary
        ), patch(
            "phases.phase3_docstrings._build_fallback_provider", return_value=fallback
        ), patch(
            "phases.phase3_docstrings.FallbackRouter", return_value=router
        ):
            run_phase3(config=config, artifacts=artifacts)

        assert artifacts.docstrings is not None
        assert artifacts.docstrings.fallback_count >= 1

    def test_raises_if_no_ast_nodes(self, tmp_path: Path) -> None:
        artifacts = PhaseArtifacts()
        config = {"output": {"artifacts_dir": str(tmp_path)}, "docstrings": {}}
        fake = FakeLLMProvider()
        with patch(
            "phases.phase3_docstrings._build_llm_provider", return_value=fake
        ), patch(
            "phases.phase3_docstrings._build_fallback_provider", return_value=None
        ):
            with pytest.raises(RuntimeError, match="Phase 2"):
                run_phase3(config=config, artifacts=artifacts)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for internal helper functions."""

    def test_build_function_id_plain_function(self) -> None:
        node = ASTNode(
            file_path="src/module.py",
            node_type=NodeType.FUNCTION,
            name="my_func",
            line_start=1,
            line_end=5,
        )
        fid = _build_function_id(node)
        assert fid == "src/module.py::my_func"

    def test_is_malformed_empty(self) -> None:
        assert _is_malformed("") is True
        assert _is_malformed("   \n  ") is True

    def test_is_malformed_valid(self) -> None:
        assert _is_malformed("Some summary.") is False

    def test_build_caller_list_found(self) -> None:
        entries = [
            CallGraphEntry(
                caller=CallerRef(file="a.py", function="caller_func"),
                callees=[CalleeRef(file="b.py", function="target", line=10)],
            )
        ]
        result = _build_caller_list("b.py::target", entries)
        assert "caller_func" in result

    def test_build_caller_list_none(self) -> None:
        result = _build_caller_list("x.py::unknown", [])
        assert result == "None"

    def test_build_callee_list_found(self) -> None:
        entries = [
            CallGraphEntry(
                caller=CallerRef(file="a.py", function="my_func"),
                callees=[CalleeRef(file="b.py", function="helper", line=5)],
            )
        ]
        result = _build_callee_list("a.py::my_func", entries)
        assert "helper" in result

    def test_build_callee_list_none(self) -> None:
        result = _build_callee_list("x.py::unknown", [])
        assert result == "None"
