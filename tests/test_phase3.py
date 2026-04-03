"""Comprehensive tests for Phase 3: Docstring Generation.

All LLM calls are mocked — no real HTTP requests are made.
"""
from __future__ import annotations

import json
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
    Docstrings,
    NodeType,
    PhaseArtifacts,
)


# ---------------------------------------------------------------------------
# Fake LLM provider (always returns a canned docstring)
# ---------------------------------------------------------------------------

_CANNED_DOCSTRING = (
    "Do something useful.\n\nArgs:\n    x: The input value.\n\nReturns:\n    A result."
)
_LOW_CONFIDENCE_DOCSTRING = "Does stuff."  # no Args/Returns → low confidence


class FakeLLMProvider(BaseLLMProvider):
    """A deterministic LLM provider for testing."""

    def __init__(self, response: str = _CANNED_DOCSTRING, name: str = "FakeLLM") -> None:
        self._response = response
        self.name = name
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        return self._response

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        return {}

    def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
        text = self.generate(prompt, **kwargs)
        from llm.utils import _confidence_heuristic  # shared heuristic

        return text, _confidence_heuristic(text)


class RaisingLLMProvider(BaseLLMProvider):
    """A provider that always raises an exception."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("LLM unavailable")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise self._exc

    def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
        raise self._exc

    def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
        raise self._exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase_artifacts(
    tmp_path: Path,
    *,
    source: str = "def foo(x):\n    return x\n",
    node_name: str = "foo",
    node_type: NodeType = NodeType.FUNCTION,
) -> PhaseArtifacts:
    """Create a minimal PhaseArtifacts with one AST node and a call graph."""
    py_file = tmp_path / "sample.py"
    py_file.write_text(source, encoding="utf-8")

    node = ASTNode(
        file_path=str(py_file),
        node_type=node_type,
        name=node_name,
        args=["x"],
        line_start=1,
        line_end=2,
    )
    caller = CallerRef(file=str(py_file), function="bar")
    callee = CalleeRef(file=str(py_file), function="foo", line=2)
    entry = CallGraphEntry(caller=caller, callees=[callee])

    return PhaseArtifacts(
        ast_nodes=ASTNodes(nodes=[node]),
        call_graph=CallGraph(entries=[entry]),
    )


def _make_config(artifacts_dir: str, provider: str = "fake") -> dict:
    return {
        "output": {"artifacts_dir": artifacts_dir},
        "docstring": {
            "primary_provider": provider,
            "confidence_threshold": 0.7,
            "max_retries": 2,
            "concurrency": 2,
        },
    }


# ---------------------------------------------------------------------------
# Tests: confidence heuristic
# ---------------------------------------------------------------------------

class TestConfidenceHeuristic:
    def test_all_sections_present_gives_high_score(self) -> None:
        from llm.ollama_provider import _confidence_heuristic

        text = "Do something.\n\nArgs:\n    x: input.\n\nReturns:\n    result."
        score = _confidence_heuristic(text)
        assert score == pytest.approx(1.0)

    def test_missing_args_and_returns_gives_lower_score(self) -> None:
        from llm.ollama_provider import _confidence_heuristic

        score = _confidence_heuristic("Just a summary line.")
        assert score == pytest.approx(1 / 3)

    def test_empty_string_gives_zero(self) -> None:
        from llm.ollama_provider import _confidence_heuristic

        assert _confidence_heuristic("") == 0.0
        assert _confidence_heuristic("   ") == 0.0

    def test_only_summary_and_returns(self) -> None:
        from llm.ollama_provider import _confidence_heuristic

        text = "Summary line.\n\nReturns:\n    A value."
        score = _confidence_heuristic(text)
        assert score == pytest.approx(2 / 3)

    def test_code_like_response_penalised(self) -> None:
        from llm.utils import _confidence_heuristic

        code = (
            "import os\nfrom pathlib import Path\n\n"
            "def foo(x):\n    self.x = x\n    return x"
        )
        assert _confidence_heuristic(code) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: _clean_docstring
# ---------------------------------------------------------------------------

class TestCleanDocstring:
    """Tests for the _clean_docstring post-processing function."""

    def test_passthrough_clean_docstring(self) -> None:
        from llm.utils import _clean_docstring

        text = "Do something useful.\n\nArgs:\n    x: The input."
        assert _clean_docstring(text) == text

    def test_strips_code_fences(self) -> None:
        from llm.utils import _clean_docstring

        raw = '```python\nDo something useful.\n\nArgs:\n    x: The input.\n```'
        assert _clean_docstring(raw) == "Do something useful.\n\nArgs:\n    x: The input."

    def test_extracts_docstring_from_code(self) -> None:
        from llm.utils import _clean_docstring

        raw = (
            '```python\n'
            'class Foo:\n'
            '    """Do something useful.\n\n'
            '    Args:\n'
            '        x: The input.\n'
            '    """\n'
            '    def __init__(self):\n'
            '        self.x = x\n'
            '```'
        )
        result = _clean_docstring(raw)
        assert "Do something useful." in result
        assert "Args:" in result
        assert "def __init__" not in result
        assert "self.x = x" not in result

    def test_deduplicates_repeated_content(self) -> None:
        from llm.utils import _clean_docstring

        block = "Summarise the module.\n\nArgs:\n    x: input."
        doubled = f"{block} {block}"
        result = _clean_docstring(doubled)
        assert result == block

    def test_strips_surrounding_triple_quotes(self) -> None:
        from llm.utils import _clean_docstring

        text = '"""Do something useful."""'
        assert _clean_docstring(text) == "Do something useful."

    def test_empty_input(self) -> None:
        from llm.utils import _clean_docstring

        assert _clean_docstring("") == ""
        assert _clean_docstring("   ") == ""

    def test_real_world_code_block_example(self) -> None:
        """Reproduce the exact failure mode from the bug report."""
        from llm.utils import _clean_docstring

        raw = (
            "```python\n"
            "class PropensitySubsetSelector:\n"
            '    """PSM-based candidate filtering.\"""\n'
            "    \n"
            "    def __init__(self, num_cols, cat_cols):\n"
            '        """\n'
            "        Initialize the PropensitySubsetSelector object.\n"
            "\n"
            "        Parameters:\n"
            "            num_cols (list): Numerical column names.\n"
            "            cat_cols (list): Categorical column names.\n"
            '        """\n'
            "        self.num_cols = num_cols\n"
            "        self.cat_cols = cat_cols\n"
            "```"
        )
        result = _clean_docstring(raw)
        assert "PSM-based candidate filtering." in result
        assert "self.num_cols" not in result
        assert "def __init__" not in result


# ---------------------------------------------------------------------------
# Tests: Issue 1 – function-definition wrapping
# ---------------------------------------------------------------------------

class TestStripWrappingDefinition:
    """Tests for _strip_wrapping_definition and the enhanced _looks_like_code."""

    def test_single_def_at_start_is_code(self) -> None:
        from llm.utils import _looks_like_code

        text = 'def foo(x, y):\n    """Does something."""\n    return x'
        assert _looks_like_code(text) is True

    def test_single_class_at_start_is_code(self) -> None:
        from llm.utils import _looks_like_code

        text = 'class Foo:\n    """A class."""\n    pass'
        assert _looks_like_code(text) is True

    def test_plain_prose_is_not_code(self) -> None:
        from llm.utils import _looks_like_code

        text = "Calculates the variance for specified columns.\n\nArgs:\n    x: input."
        assert _looks_like_code(text) is False

    def test_strip_wrapping_definition_extracts_body(self) -> None:
        from llm.utils import _strip_wrapping_definition

        raw = (
            'def _calc_categorical_variance(subset, whitelist, cat_cols):\n'
            '    """\n'
            '    Calculates the categorical variance.\n'
            '\n'
            '    Args:\n'
            '        subset: The dataframe.\n'
            '    """\n'
        )
        result = _strip_wrapping_definition(raw)
        assert "Calculates the categorical variance." in result
        assert "def _calc_categorical_variance" not in result

    def test_clean_docstring_removes_def_wrapper(self) -> None:
        from llm.utils import _clean_docstring

        raw = (
            'def _calc_categorical_variance(subset, whitelist, cat_cols):\n'
            '    """\n'
            '    Calculates the categorical variance for specified columns.\n'
            '\n'
            '    Args:\n'
            '        subset (pd.DataFrame): The dataframe.\n'
            '    """\n'
        )
        result = _clean_docstring(raw)
        assert "Calculates the categorical variance" in result
        assert "def _calc_categorical_variance" not in result

    def test_clean_docstring_removes_class_wrapper(self) -> None:
        from llm.utils import _clean_docstring

        raw = (
            'class MyEncoder:\n'
            '    """Encodes categorical columns.\n\n'
            '    Args:\n'
            '        cols: Column list.\n'
            '    """\n'
        )
        result = _clean_docstring(raw)
        assert "Encodes categorical columns." in result
        assert "class MyEncoder" not in result


# ---------------------------------------------------------------------------
# Tests: Issue 2 – escaped triple quotes
# ---------------------------------------------------------------------------

class TestStripEscapedQuotes:
    """Tests for _strip_escaped_quotes and its integration in _clean_docstring."""

    def test_strip_escaped_double_quotes(self) -> None:
        from llm.utils import _strip_escaped_quotes

        text = '\\"\\"\\"\\nCalculates variance.\\n\\"\\"\\"'
        result = _strip_escaped_quotes(text)
        assert '\\"' not in result
        assert '"' in result

    def test_strip_escaped_single_quotes(self) -> None:
        from llm.utils import _strip_escaped_quotes

        assert _strip_escaped_quotes("\\'hello\\'") == "'hello'"

    def test_clean_docstring_handles_escaped_triple_quotes(self) -> None:
        from llm.utils import _clean_docstring

        # LLM output with escaped triple-quotes wrapping a real docstring
        raw = (
            '\\\"\\\"\\\"\\n'
            'Calculates the categorical variance for specified columns.\\n'
            '\\\"\\\"\\\"'
        )
        result = _clean_docstring(raw)
        assert "Calculates the categorical variance" in result
        assert '\\"' not in result


# ---------------------------------------------------------------------------
# Tests: Issue 3 – degenerate / repetitive LLM output
# ---------------------------------------------------------------------------

class TestIsDegenerate:
    """Tests for _is_degenerate and its integration."""

    def test_backtick_spam_is_degenerate(self) -> None:
        from llm.utils import _is_degenerate

        spam = "``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ```"
        assert _is_degenerate(spam) is True

    def test_pure_whitespace_is_not_degenerate(self) -> None:
        """Empty/whitespace strings are handled by the not-text guard, not _is_degenerate."""
        from llm.utils import _is_degenerate

        assert _is_degenerate("") is False
        assert _is_degenerate("   ") is False

    def test_single_token_repetition_is_degenerate(self) -> None:
        from llm.utils import _is_degenerate

        spam = "the the the the the the the the the the the the"
        assert _is_degenerate(spam) is True

    def test_dominant_token_above_threshold_is_degenerate(self) -> None:
        from llm.utils import _is_degenerate

        # 18 "foo" + 2 other tokens → 90% dominant
        spam = " ".join(["foo"] * 18 + ["bar", "baz"])
        assert _is_degenerate(spam) is True

    def test_mixed_valid_text_is_not_degenerate(self) -> None:
        from llm.utils import _is_degenerate

        text = "Calculates variance.\n\nArgs:\n    x: int\n\nReturns:\n    float"
        assert _is_degenerate(text) is False

    def test_clean_docstring_returns_empty_for_backtick_spam(self) -> None:
        from llm.utils import _clean_docstring

        spam = " ".join(["```"] * 50)
        assert _clean_docstring(spam) == ""

    def test_confidence_heuristic_returns_zero_for_degenerate(self) -> None:
        from llm.utils import _confidence_heuristic

        spam = "``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ``` ```"
        assert _confidence_heuristic(spam) == 0.0

    def test_generate_for_node_retries_on_degenerate_output(
        self, tmp_path: Path
    ) -> None:
        """_generate_for_node must retry (not store) degenerate LLM output."""
        from phases.phase3_docstrings import _generate_for_node
        from models.schemas import DocstringFailure, NodeType, ASTNode

        backtick_spam = " ".join(["```"] * 60)

        py_file = tmp_path / "sample.py"
        py_file.write_text("def foo(x):\n    return x\n", encoding="utf-8")
        node = ASTNode(
            file_path=str(py_file),
            node_type=NodeType.FUNCTION,
            name="foo",
            args=["x"],
            line_start=1,
            line_end=2,
        )

        call_count = 0

        class SpamProvider(BaseLLMProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                return backtick_spam

            def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
                return {}

            def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
                nonlocal call_count
                call_count += 1
                return backtick_spam, 0.25

        provider = SpamProvider()
        router = FallbackRouter(primary=provider, fallback=None, confidence_threshold=0.7)

        result = _generate_for_node(
            node,
            py_file.read_text(encoding="utf-8").splitlines(keepends=True),
            router,
            {},
            {},
            max_retries=2,
        )
        # Should exhaust retries and return a failure, not a DocstringEntry
        assert isinstance(result, DocstringFailure)
        assert result.error_type == "malformed_response"
        # Provider should have been called 3 times (initial + 2 retries)
        assert call_count == 3


# ---------------------------------------------------------------------------
# Tests: OllamaProvider
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def test_health_check_warning_on_unreachable(self) -> None:
        """OllamaProvider must not crash if the server is unreachable."""
        with patch("requests.get", side_effect=ConnectionError("refused")):
            from llm.ollama_provider import OllamaProvider

            # Should not raise
            provider = OllamaProvider(base_url="http://localhost:9999")
            assert provider.base_url == "http://localhost:9999"

    def test_generate_returns_response(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "A docstring."}
        mock_resp.raise_for_status.return_value = None

        with (
            patch("requests.get"),  # health check
            patch("requests.post", return_value=mock_resp) as mock_post,
        ):
            from llm.ollama_provider import OllamaProvider

            provider = OllamaProvider(model_name="test-model")
            result = provider.generate("my prompt")

        assert result == "A docstring."
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["model"] == "test-model"
        assert payload["stream"] is False

    def test_generate_raises_timeout_error(self) -> None:
        import requests as req_lib

        with (
            patch("requests.get"),
            patch("requests.post", side_effect=req_lib.exceptions.Timeout),
        ):
            from llm.ollama_provider import OllamaProvider

            provider = OllamaProvider()
            with pytest.raises(TimeoutError):
                provider.generate("prompt")

    def test_generate_with_confidence_returns_tuple(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": _CANNED_DOCSTRING}
        mock_resp.raise_for_status.return_value = None

        with (
            patch("requests.get"),
            patch("requests.post", return_value=mock_resp),
        ):
            from llm.ollama_provider import OllamaProvider

            provider = OllamaProvider()
            text, confidence = provider.generate_with_confidence("prompt")

        assert text == _CANNED_DOCSTRING
        assert confidence == pytest.approx(1.0)

    def test_generate_structured_parses_json(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"key": "value"}'}
        mock_resp.raise_for_status.return_value = None

        with (
            patch("requests.get"),
            patch("requests.post", return_value=mock_resp),
        ):
            from llm.ollama_provider import OllamaProvider

            provider = OllamaProvider()
            result = provider.generate_structured("prompt", dict)

        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# Tests: FallbackRouter
# ---------------------------------------------------------------------------

class TestFallbackRouter:
    def test_high_confidence_no_fallback(self) -> None:
        """When primary confidence >= threshold, fallback must NOT be used."""
        primary = FakeLLMProvider(_CANNED_DOCSTRING)
        fallback = FakeLLMProvider("fallback text")
        router = FallbackRouter(primary, fallback, confidence_threshold=0.5)

        text, confidence, provider_name, fallback_used = router.generate_with_fallback("p")

        assert text == _CANNED_DOCSTRING
        assert provider_name == "FakeLLMProvider"
        assert not fallback_used
        assert len(fallback.calls) == 0

    def test_low_confidence_triggers_fallback(self) -> None:
        """When primary confidence < threshold, fallback is tried."""
        primary = FakeLLMProvider(_LOW_CONFIDENCE_DOCSTRING)
        fallback = FakeLLMProvider(_CANNED_DOCSTRING)
        router = FallbackRouter(primary, fallback, confidence_threshold=0.9)

        text, confidence, provider_name, fallback_used = router.generate_with_fallback("p")

        assert fallback_used
        assert text == _CANNED_DOCSTRING
        assert provider_name == "FakeLLMProvider"

    def test_exception_triggers_fallback(self) -> None:
        """When primary raises an exception, the fallback is tried automatically."""
        primary = RaisingLLMProvider(RuntimeError("crash"))
        fallback = FakeLLMProvider(_CANNED_DOCSTRING)
        router = FallbackRouter(primary, fallback, confidence_threshold=0.7)

        text, confidence, provider_name, fallback_used = router.generate_with_fallback("p")

        assert fallback_used
        assert text == _CANNED_DOCSTRING

    def test_higher_confidence_result_returned(self) -> None:
        """The result with higher confidence between primary and fallback is returned."""
        # primary: low confidence
        primary = FakeLLMProvider(_LOW_CONFIDENCE_DOCSTRING)
        # fallback: high confidence
        fallback = FakeLLMProvider(_CANNED_DOCSTRING)
        router = FallbackRouter(primary, fallback, confidence_threshold=0.9)

        text, confidence, provider_name, fallback_used = router.generate_with_fallback("p")

        assert text == _CANNED_DOCSTRING
        assert fallback_used

    def test_no_fallback_configured_returns_primary(self) -> None:
        primary = FakeLLMProvider(_LOW_CONFIDENCE_DOCSTRING)
        router = FallbackRouter(primary, fallback=None, confidence_threshold=0.99)

        text, confidence, provider_name, fallback_used = router.generate_with_fallback("p")

        assert text == _LOW_CONFIDENCE_DOCSTRING
        assert not fallback_used

    def test_both_providers_fail_raises(self) -> None:
        primary = RaisingLLMProvider()
        fallback = RaisingLLMProvider()
        router = FallbackRouter(primary, fallback, confidence_threshold=0.7)

        with pytest.raises(Exception):
            router.generate_with_fallback("p")


# ---------------------------------------------------------------------------
# Tests: Phase 3 integration
# ---------------------------------------------------------------------------

class TestPhase3Integration:
    def _run_phase3_with_fake(
        self,
        artifacts: PhaseArtifacts,
        artifacts_dir: str,
        provider: FakeLLMProvider | None = None,
    ) -> Docstrings:
        """Patch _build_provider and run_phase3."""
        from phases.phase3_docstrings import run_phase3

        fake = provider or FakeLLMProvider()
        cfg = {
            "output": {"artifacts_dir": artifacts_dir},
            "docstring": {
                "primary_provider": "fake",
                "confidence_threshold": 0.7,
                "max_retries": 2,
                "concurrency": 2,
            },
        }

        with patch("phases.phase3_docstrings._build_provider", return_value=fake):
            run_phase3(config=cfg, artifacts=artifacts)

        assert artifacts.docstrings is not None
        return artifacts.docstrings

    def test_generates_for_all_functions(self, tmp_path: Path) -> None:
        """All AST nodes should produce a DocstringEntry."""
        artifacts = _make_phase_artifacts(tmp_path)
        ds = self._run_phase3_with_fake(artifacts, str(tmp_path / "arts"))

        assert ds.total_functions == 1
        assert ds.successful == 1
        assert ds.failed == 0
        assert len(ds.entries) == 1

    def test_module_docstrings_generated(self, tmp_path: Path) -> None:
        artifacts = _make_phase_artifacts(tmp_path)
        ds = self._run_phase3_with_fake(artifacts, str(tmp_path / "arts"))

        assert len(ds.module_docstrings) == 1

    def test_stats_computed_correctly(self, tmp_path: Path) -> None:
        artifacts = _make_phase_artifacts(tmp_path)
        ds = self._run_phase3_with_fake(artifacts, str(tmp_path / "arts"))

        assert ds.successful + ds.failed == ds.total_functions
        assert 0.0 <= ds.average_confidence <= 1.0

    def test_docstring_json_saved(self, tmp_path: Path) -> None:
        artifacts = _make_phase_artifacts(tmp_path)
        arts_dir = tmp_path / "arts"
        self._run_phase3_with_fake(artifacts, str(arts_dir))

        out = arts_dir / "docstrings.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "entries" in data
        assert "failures" in data

    def test_correct_function_id(self, tmp_path: Path) -> None:
        artifacts = _make_phase_artifacts(tmp_path)
        ds = self._run_phase3_with_fake(artifacts, str(tmp_path / "arts"))

        entry = ds.entries[0]
        assert "foo" in entry.function_id
        assert "sample.py" in entry.function_id

    def test_prompt_contains_callers_and_callees(self, tmp_path: Path) -> None:
        fake = FakeLLMProvider()
        artifacts = _make_phase_artifacts(tmp_path)
        self._run_phase3_with_fake(artifacts, str(tmp_path / "arts"), provider=fake)

        # At least one call should mention callers/callees context
        assert len(fake.calls) >= 1
        combined = " ".join(fake.calls)
        assert "Called by:" in combined

    def test_error_recorded_as_failure_when_file_missing(self, tmp_path: Path) -> None:
        """Nodes pointing to a non-existent file should produce a DocstringFailure."""
        from phases.phase3_docstrings import run_phase3

        node = ASTNode(
            file_path="/nonexistent/path/missing.py",
            node_type=NodeType.FUNCTION,
            name="ghost",
            line_start=1,
            line_end=2,
        )
        artifacts = PhaseArtifacts(
            ast_nodes=ASTNodes(nodes=[node]),
            call_graph=CallGraph(entries=[]),
        )
        fake = FakeLLMProvider()
        cfg = {
            "output": {"artifacts_dir": str(tmp_path / "arts")},
            "docstring": {
                "primary_provider": "fake",
                "confidence_threshold": 0.7,
                "max_retries": 0,
                "concurrency": 1,
            },
        }
        with patch("phases.phase3_docstrings._build_provider", return_value=fake):
            run_phase3(config=cfg, artifacts=artifacts)

        assert artifacts.docstrings is not None
        assert artifacts.docstrings.failed == 1
        assert artifacts.docstrings.failures[0].error_type == "file_read_error"

    def test_raises_without_ast_nodes(self, tmp_path: Path) -> None:
        from phases.phase3_docstrings import run_phase3

        artifacts = PhaseArtifacts(call_graph=CallGraph(entries=[]))
        with pytest.raises(ValueError, match="ast_nodes"):
            run_phase3(
                config={"output": {"artifacts_dir": str(tmp_path)}},
                artifacts=artifacts,
            )

    def test_raises_without_call_graph(self, tmp_path: Path) -> None:
        from phases.phase3_docstrings import run_phase3

        artifacts = PhaseArtifacts(ast_nodes=ASTNodes(nodes=[]))
        with pytest.raises(ValueError, match="call_graph"):
            run_phase3(
                config={"output": {"artifacts_dir": str(tmp_path)}},
                artifacts=artifacts,
            )

    def test_fallback_count_tracked(self, tmp_path: Path) -> None:
        """When fallback is used, fallback_count must reflect that."""
        from phases.phase3_docstrings import run_phase3

        # Primary gives low confidence, fallback gives high
        primary = FakeLLMProvider(_LOW_CONFIDENCE_DOCSTRING, name="Primary")
        fallback = FakeLLMProvider(_CANNED_DOCSTRING, name="Fallback")
        router = FallbackRouter(primary, fallback, confidence_threshold=0.9)

        artifacts = _make_phase_artifacts(tmp_path)
        cfg = {
            "output": {"artifacts_dir": str(tmp_path / "arts")},
            "docstring": {
                "primary_provider": "fake",
                "confidence_threshold": 0.9,
                "max_retries": 0,
                "concurrency": 1,
            },
        }

        with patch("phases.phase3_docstrings._build_provider", return_value=primary):
            with patch("phases.phase3_docstrings.FallbackRouter", return_value=router):
                run_phase3(config=cfg, artifacts=artifacts)

        ds = artifacts.docstrings
        assert ds is not None
        assert ds.fallback_count >= 1

    def test_multiple_nodes_multiple_files(self, tmp_path: Path) -> None:
        """Phase 3 handles multiple files and nodes."""
        from phases.phase3_docstrings import run_phase3

        f1 = tmp_path / "mod1.py"
        f2 = tmp_path / "mod2.py"
        f1.write_text("def alpha():\n    pass\n")
        f2.write_text("def beta():\n    pass\n")

        nodes = [
            ASTNode(file_path=str(f1), node_type=NodeType.FUNCTION, name="alpha", line_start=1, line_end=2),
            ASTNode(file_path=str(f2), node_type=NodeType.FUNCTION, name="beta", line_start=1, line_end=2),
        ]
        artifacts = PhaseArtifacts(
            ast_nodes=ASTNodes(nodes=nodes),
            call_graph=CallGraph(entries=[]),
        )

        fake = FakeLLMProvider()
        cfg = {
            "output": {"artifacts_dir": str(tmp_path / "arts")},
            "docstring": {
                "primary_provider": "fake",
                "confidence_threshold": 0.7,
                "max_retries": 0,
                "concurrency": 2,
            },
        }
        with patch("phases.phase3_docstrings._build_provider", return_value=fake):
            run_phase3(config=cfg, artifacts=artifacts)

        ds = artifacts.docstrings
        assert ds is not None
        assert ds.total_functions == 2
        assert ds.successful == 2
        # One module docstring per file
        assert len(ds.module_docstrings) == 2
