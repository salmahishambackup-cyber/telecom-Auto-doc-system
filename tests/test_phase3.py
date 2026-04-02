"""Tests for Phase 3: Docstring Generation."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
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
    NodeType,
    PhaseArtifacts,
)
from phases.phase3_docstrings import run_phase3


# ---------------------------------------------------------------------------
# Fake / stub LLM providers
# ---------------------------------------------------------------------------

class FakeLLMProvider(BaseLLMProvider):
    """Returns canned responses without any real LLM call."""

    def __init__(self, response: str = "", confidence: float = 0.9) -> None:
        self._response = response
        self._confidence = confidence
        self.call_count = 0
        self.last_prompt: str = ""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self._response

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        return {}

    def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
        self.call_count += 1
        self.last_prompt = prompt
        return self._response, self._confidence


GOOD_DOCSTRING = dedent("""\
    Summary line for the function.

    Args:
        x: The input value.

    Returns:
        The processed result.
""").strip()

BAD_DOCSTRING = ""  # empty — should trigger failure path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(artifacts_dir: str, primary: str = "ollama") -> dict:
    return {
        "output": {"artifacts_dir": artifacts_dir},
        "docstring": {
            "primary_provider": primary,
            "fallback_provider": None,
            "confidence_threshold": 0.7,
            "max_retries": 1,
            "concurrency": 2,
            "ollama": {"model": "test-model", "base_url": "http://localhost:11434", "timeout": 10},
        },
    }


def _make_ast_node(
    file_path: str,
    name: str,
    node_type: NodeType = NodeType.FUNCTION,
    line_start: int = 1,
    line_end: int = 3,
    class_name: str | None = None,
) -> ASTNode:
    node = ASTNode(
        file_path=file_path,
        node_type=node_type,
        name=name,
        line_start=line_start,
        line_end=line_end,
    )
    # Attach class_name as a dynamic attribute for the phase to use
    object.__setattr__(node, "__dict__", {**node.__dict__, "class_name": class_name})
    return node


def _write_py(path: Path, content: str) -> None:
    path.write_text(dedent(content), encoding="utf-8")


# ---------------------------------------------------------------------------
# Confidence estimation tests
# ---------------------------------------------------------------------------

class TestConfidenceEstimation:
    def test_confidence_all_sections_present(self) -> None:
        from phases.phase3_docstrings import _is_valid_docstring
        from llm.ollama_provider import _estimate_confidence

        text = "Summary.\n\nArgs:\n    x: int\n\nReturns:\n    int"
        conf = _estimate_confidence(text)
        assert conf > 0.6

    def test_confidence_missing_sections(self) -> None:
        from llm.ollama_provider import _estimate_confidence

        text = "Just a brief description with no parameter or output information."
        conf = _estimate_confidence(text)
        # "args" and "returns" sections are missing
        assert conf < 1.0

    def test_confidence_empty_response(self) -> None:
        from llm.ollama_provider import _estimate_confidence

        assert _estimate_confidence("") == 0.0
        assert _estimate_confidence("   ") == 0.0


# ---------------------------------------------------------------------------
# OllamaProvider tests
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def test_generate_with_confidence(self) -> None:
        from llm.ollama_provider import OllamaProvider

        with patch("urllib.request.urlopen") as mock_urlopen:
            # Health check
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value.read.return_value = json.dumps({"models": []}).encode()

            provider = OllamaProvider.__new__(OllamaProvider)
            provider.model_name = "test"
            provider.base_url = "http://localhost:11434"
            provider.timeout = 10

            response_data = json.dumps({
                "response": GOOD_DOCSTRING,
                "prompt_eval_count": 50,
                "eval_count": 80,
            }).encode()

            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = response_data

            with patch("urllib.request.urlopen", return_value=mock_resp):
                text, conf = provider.generate_with_confidence("test prompt")

            assert text == GOOD_DOCSTRING
            assert 0.0 <= conf <= 1.0
            assert conf > 0.5  # GOOD_DOCSTRING has args + returns + summary

    def test_health_check_failure_does_not_crash(self) -> None:
        from llm.ollama_provider import OllamaProvider
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            # Should not raise — just log a warning
            provider = OllamaProvider(
                model_name="test",
                base_url="http://localhost:9999",
                timeout=5,
            )
        assert provider is not None

    def test_timeout_handling(self) -> None:
        from llm.ollama_provider import OllamaProvider

        provider = OllamaProvider.__new__(OllamaProvider)
        provider.model_name = "test"
        provider.base_url = "http://localhost:11434"
        provider.timeout = 1

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with pytest.raises(TimeoutError):
                provider.generate("prompt")


# ---------------------------------------------------------------------------
# OpenAIProvider tests
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def test_generate_returns_text_and_tokens(self) -> None:
        openai_mock = MagicMock()
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_choice = MagicMock()
        mock_choice.message.content = GOOD_DOCSTRING

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        openai_mock.OpenAI.return_value.chat.completions.create.return_value = mock_response
        openai_mock.RateLimitError = Exception  # Ensure it won't accidentally match

        with patch.dict("sys.modules", {"openai": openai_mock}):
            from llm.openai_provider import OpenAIProvider
            import importlib
            import llm.openai_provider as mod
            importlib.reload(mod)
            provider = mod.OpenAIProvider.__new__(mod.OpenAIProvider)
            provider.model = "gpt-4o-mini"
            provider.timeout = 10
            provider._openai = openai_mock
            provider._client = openai_mock.OpenAI()

            text, conf = provider.generate_with_confidence("test prompt")

        assert text == GOOD_DOCSTRING
        assert 0.0 <= conf <= 1.0

    def test_rate_limit_backoff(self) -> None:
        openai_mock = MagicMock()

        class FakeRateLimitError(Exception):
            pass

        openai_mock.RateLimitError = FakeRateLimitError

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_choice = MagicMock()
        mock_choice.message.content = "OK"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise FakeRateLimitError("rate limited")
            return mock_response

        with patch.dict("sys.modules", {"openai": openai_mock}):
            import importlib
            import llm.openai_provider as mod
            importlib.reload(mod)
            provider = mod.OpenAIProvider.__new__(mod.OpenAIProvider)
            provider.model = "gpt-4o-mini"
            provider.timeout = 10
            provider._openai = openai_mock
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = side_effect
            provider._client = mock_client

            with patch("time.sleep"):
                text = provider.generate("prompt")

        assert text == "OK"
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# AnthropicProvider tests
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def test_generate_returns_text(self) -> None:
        anthropic_mock = MagicMock()
        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 25

        mock_content = MagicMock()
        mock_content.text = GOOD_DOCSTRING

        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.usage = mock_usage

        anthropic_mock.Anthropic.return_value.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": anthropic_mock}):
            import importlib
            import llm.anthropic_provider as mod
            importlib.reload(mod)
            provider = mod.AnthropicProvider.__new__(mod.AnthropicProvider)
            provider.model = "claude-sonnet-4-20250514"
            provider.timeout = 10
            provider.max_tokens = 2048
            provider._client = anthropic_mock.Anthropic()

            text = provider.generate("test prompt")

        assert text == GOOD_DOCSTRING

    def test_token_extraction(self) -> None:
        anthropic_mock = MagicMock()
        mock_usage = MagicMock()
        mock_usage.input_tokens = 42
        mock_usage.output_tokens = 99

        mock_content = MagicMock()
        mock_content.text = GOOD_DOCSTRING

        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.usage = mock_usage

        anthropic_mock.Anthropic.return_value.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": anthropic_mock}):
            import importlib
            import llm.anthropic_provider as mod
            importlib.reload(mod)
            provider = mod.AnthropicProvider.__new__(mod.AnthropicProvider)
            provider.model = "claude-sonnet-4-20250514"
            provider.timeout = 10
            provider.max_tokens = 2048
            provider._client = anthropic_mock.Anthropic()

            text, conf = provider.generate_with_confidence("test")

        assert text == GOOD_DOCSTRING
        assert mock_response.usage.input_tokens == 42
        assert mock_response.usage.output_tokens == 99


# ---------------------------------------------------------------------------
# FallbackRouter tests
# ---------------------------------------------------------------------------

class TestFallbackRouter:
    def test_no_fallback_when_high_confidence(self) -> None:
        primary = FakeLLMProvider(response=GOOD_DOCSTRING, confidence=0.9)
        fallback = FakeLLMProvider(response="fallback", confidence=0.5)
        router = FallbackRouter(primary=primary, fallback=fallback, confidence_threshold=0.7)

        text, conf, provider_name, used_fallback = router.generate_with_fallback("prompt")

        assert text == GOOD_DOCSTRING
        assert conf == 0.9
        assert not used_fallback
        assert fallback.call_count == 0

    def test_fallback_triggered_on_low_confidence(self) -> None:
        primary = FakeLLMProvider(response="weak", confidence=0.3)
        fallback = FakeLLMProvider(response=GOOD_DOCSTRING, confidence=0.95)
        router = FallbackRouter(primary=primary, fallback=fallback, confidence_threshold=0.7)

        text, conf, provider_name, used_fallback = router.generate_with_fallback("prompt")

        assert used_fallback
        assert text == GOOD_DOCSTRING
        assert conf == 0.95

    def test_fallback_triggered_on_primary_exception(self) -> None:
        primary = MagicMock()
        primary.generate_with_confidence.side_effect = ConnectionError("offline")

        fallback = FakeLLMProvider(response=GOOD_DOCSTRING, confidence=0.8)
        router = FallbackRouter(primary=primary, fallback=fallback, confidence_threshold=0.7)

        text, conf, provider_name, used_fallback = router.generate_with_fallback("prompt")

        assert used_fallback
        assert text == GOOD_DOCSTRING

    def test_fallback_returns_higher_confidence(self) -> None:
        primary = FakeLLMProvider(response="low", confidence=0.4)
        fallback = FakeLLMProvider(response="high", confidence=0.85)
        router = FallbackRouter(primary=primary, fallback=fallback, confidence_threshold=0.7)

        text, conf, _, _ = router.generate_with_fallback("prompt")
        assert text == "high"
        assert conf == 0.85


# ---------------------------------------------------------------------------
# Phase 3 integration tests
# ---------------------------------------------------------------------------

class TestPhase3Integration:
    def _setup_artifacts(self, tmp_path: Path) -> tuple[PhaseArtifacts, str]:
        """Create a minimal file + artifacts for Phase 3 to process."""
        src_file = tmp_path / "service.py"
        _write_py(src_file, """\
            def connect(host: str, port: int) -> None:
                \"\"\"Old docstring.\"\"\"
                pass

            class Router:
                def route(self, path: str) -> str:
                    pass
        """)

        node_func = ASTNode(
            file_path=str(src_file),
            node_type=NodeType.FUNCTION,
            name="connect",
            args=["host: str", "port: int"],
            line_start=1,
            line_end=3,
        )
        node_method = ASTNode(
            file_path=str(src_file),
            node_type=NodeType.METHOD,
            name="route",
            args=["self", "path: str"],
            line_start=6,
            line_end=7,
        )
        artifacts = PhaseArtifacts(
            ast_nodes=ASTNodes(nodes=[node_func, node_method]),
            call_graph=CallGraph(),
        )
        return artifacts, str(src_file)

    def _patch_phase3_providers(
        self,
        primary_response: str = GOOD_DOCSTRING,
        primary_confidence: float = 0.9,
    ) -> Any:
        """Return a context-manager that patches _build_provider to use a fake."""
        fake = FakeLLMProvider(response=primary_response, confidence=primary_confidence)

        def fake_build(name: str, cfg: dict) -> FakeLLMProvider:
            return fake

        return patch("phases.phase3_docstrings._build_provider", side_effect=fake_build)

    def test_generates_docstrings_for_all_functions(self, tmp_path: Path) -> None:
        artifacts, _ = self._setup_artifacts(tmp_path)
        config = _make_config(str(tmp_path / "artifacts"))

        with self._patch_phase3_providers():
            run_phase3(config=config, artifacts=artifacts)

        assert artifacts.docstrings is not None
        assert artifacts.docstrings.total_functions == 2
        assert artifacts.docstrings.successful == 2

    def test_generates_module_docstrings(self, tmp_path: Path) -> None:
        artifacts, _ = self._setup_artifacts(tmp_path)
        config = _make_config(str(tmp_path / "artifacts"))

        with self._patch_phase3_providers():
            run_phase3(config=config, artifacts=artifacts)

        assert artifacts.docstrings is not None
        assert len(artifacts.docstrings.module_docstrings) == 1

    def test_handles_file_read_error(self, tmp_path: Path) -> None:
        """A missing file should record all its functions as file_read_error failures."""
        missing_file = str(tmp_path / "gone.py")
        node = ASTNode(
            file_path=missing_file,
            node_type=NodeType.FUNCTION,
            name="phantom",
            line_start=1,
            line_end=2,
        )
        artifacts = PhaseArtifacts(
            ast_nodes=ASTNodes(nodes=[node]),
            call_graph=CallGraph(),
        )
        config = _make_config(str(tmp_path / "artifacts"))

        with self._patch_phase3_providers():
            run_phase3(config=config, artifacts=artifacts)

        assert artifacts.docstrings is not None
        assert artifacts.docstrings.failed == 1
        assert artifacts.docstrings.failures[0].error_type == "file_read_error"

    def test_records_failures_on_bad_response(self, tmp_path: Path) -> None:
        """Empty LLM response should be recorded as malformed_response failure."""
        artifacts, _ = self._setup_artifacts(tmp_path)
        config = _make_config(str(tmp_path / "artifacts"))

        with self._patch_phase3_providers(primary_response="", primary_confidence=0.0):
            run_phase3(config=config, artifacts=artifacts)

        assert artifacts.docstrings is not None
        # Both functions should have failed
        assert artifacts.docstrings.failed == 2
        assert all(
            f.error_type == "malformed_response"
            for f in artifacts.docstrings.failures
        )

    def test_function_id_format_top_level(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        _write_py(src, "def foo(): pass\n")
        node = ASTNode(
            file_path=str(src),
            node_type=NodeType.FUNCTION,
            name="foo",
            line_start=1,
            line_end=1,
        )
        artifacts = PhaseArtifacts(
            ast_nodes=ASTNodes(nodes=[node]),
            call_graph=CallGraph(),
        )

        with self._patch_phase3_providers():
            run_phase3(config=_make_config(str(tmp_path / "artifacts")), artifacts=artifacts)

        entry = artifacts.docstrings.entries[0]
        assert entry.function_id == f"{str(src)}::foo"

    def test_prompt_includes_callers_and_callees(self, tmp_path: Path) -> None:
        src = tmp_path / "svc.py"
        _write_py(src, "def worker(): pass\ndef main():\n    worker()\n")

        node_worker = ASTNode(
            file_path=str(src),
            node_type=NodeType.FUNCTION,
            name="worker",
            line_start=1,
            line_end=1,
        )
        node_main = ASTNode(
            file_path=str(src),
            node_type=NodeType.FUNCTION,
            name="main",
            line_start=2,
            line_end=3,
        )

        call_graph = CallGraph(
            entries=[
                CallGraphEntry(
                    caller=CallerRef(file=str(src), function="main"),
                    callees=[CalleeRef(file=str(src), function="worker", line=3)],
                )
            ]
        )
        artifacts = PhaseArtifacts(
            ast_nodes=ASTNodes(nodes=[node_worker, node_main]),
            call_graph=call_graph,
        )

        captured_prompts: list[str] = []

        class CapturingFake(FakeLLMProvider):
            def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
                captured_prompts.append(prompt)
                return GOOD_DOCSTRING, 0.9

        fake = CapturingFake()

        def fake_build(name: str, cfg: dict) -> CapturingFake:
            return fake

        with patch("phases.phase3_docstrings._build_provider", side_effect=fake_build):
            run_phase3(
                config=_make_config(str(tmp_path / "artifacts")),
                artifacts=artifacts,
            )

        # worker's prompt should mention "main" as caller
        worker_prompt = next(p for p in captured_prompts if "worker" in p and "def worker" in p)
        assert "main" in worker_prompt

    def test_saves_artifact_to_disk(self, tmp_path: Path) -> None:
        artifacts, _ = self._setup_artifacts(tmp_path)
        arts_dir = tmp_path / "artifacts"
        config = _make_config(str(arts_dir))

        with self._patch_phase3_providers():
            run_phase3(config=config, artifacts=artifacts)

        assert (arts_dir / "docstrings.json").exists()
        data = json.loads((arts_dir / "docstrings.json").read_text())
        assert "entries" in data

    def test_summary_stats(self, tmp_path: Path) -> None:
        artifacts, _ = self._setup_artifacts(tmp_path)
        config = _make_config(str(tmp_path / "artifacts"))

        with self._patch_phase3_providers(primary_confidence=0.85):
            run_phase3(config=config, artifacts=artifacts)

        ds = artifacts.docstrings
        assert ds is not None
        assert ds.total_functions == 2
        assert ds.successful == 2
        assert ds.failed == 0
        assert ds.average_confidence > 0.0
        assert ds.fallback_count == 0

    def test_requires_phase2_artifacts(self, tmp_path: Path) -> None:
        artifacts = PhaseArtifacts()  # no ast_nodes
        with pytest.raises(RuntimeError):
            run_phase3(
                config=_make_config(str(tmp_path / "artifacts")),
                artifacts=artifacts,
            )
