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

    # Issue 4: residual markdown fences
    def test_backtick_spam_returns_empty(self) -> None:
        """Issue 4: docstring that is entirely backtick-spam returns empty."""
        from llm.utils import _clean_docstring

        spam = " ".join(["```"] * 200)
        assert _clean_docstring(spam) == ""

    def test_double_markdown_fence_prefix(self) -> None:
        """Issue 4: two consecutive ```markdown opening fences are stripped."""
        from llm.utils import _clean_docstring

        raw = "```markdown\n```markdown\nThe step_end method logs completion."
        result = _clean_docstring(raw)
        assert result == "The step_end method logs completion."
        assert "```" not in result

    def test_unclosed_fence_strips_marker(self) -> None:
        """Issue 4: an unclosed ``` fence marker is removed from the output."""
        from llm.utils import _clean_docstring

        raw = "```python\nDo something useful.\n\nArgs:\n    x: The input."
        result = _clean_docstring(raw)
        assert "Do something useful." in result
        assert "```" not in result

    def test_strip_inline_backtick_fences(self) -> None:
        """Issue 4: _strip_inline_backtick_fences removes residual markers."""
        from llm.utils import _strip_inline_backtick_fences

        text = "```markdown\nSome text with ```python fences```"
        result = _strip_inline_backtick_fences(text)
        assert "```" not in result
        assert "Some text with" in result

    # Issue 1: def/class wrapper
    def test_strips_def_wrapper(self) -> None:
        """Issue 1: leading 'def func():' wrapper is stripped."""
        from llm.utils import _clean_docstring

        raw = (
            'def _calc_categorical_variance(subset, whitelist, cat_cols):\n'
            '    """\n'
            '    Calculates the categorical variance.\n'
            '\n'
            '    Args:\n'
            '        subset (pd.DataFrame): The dataframe.\n'
            '    """\n'
        )
        result = _clean_docstring(raw)
        assert "Calculates the categorical variance." in result
        assert "def _calc_categorical_variance" not in result

    def test_strips_escaped_triple_quotes(self) -> None:
        """Issue 2: escaped triple-quotes are unescaped and stripped."""
        from llm.utils import _clean_docstring

        raw = r'\"\"\"Calculates the categorical variance.\"\"\"'
        result = _clean_docstring(raw)
        assert "Calculates the categorical variance." in result
        assert '"""' not in result

    # Issue 5: signature echo without def
    def test_strips_signature_echo(self) -> None:
        """Issue 5: function signature echo without 'def' is stripped."""
        from llm.utils import _clean_docstring

        raw = (
            "transform_features(\n"
            "    pl_df: pd.DataFrame,\n"
            "    wl_df: pd.DataFrame,\n"
            ") -> Tuple[csr_matrix, np.ndarray]:\n"
            '    """\n'
            "    Transforms the features of the dataframe.\n"
            "\n"
            "    Args:\n"
            "        pl_df: The dataframe.\n"
            '    """\n'
        )
        result = _clean_docstring(raw)
        assert "Transforms the features" in result
        assert "transform_features(" not in result
        assert "pd.DataFrame" not in result

    def test_strip_signature_echo_helper(self) -> None:
        """Issue 5: _strip_signature_echo helper removes signature prefix."""
        from llm.utils import _strip_signature_echo

        raw = (
            "plot_numerical_distributions(\n"
            "    df: pd.DataFrame,\n"
            "    cols: List[str],\n"
            "):\n"
            "    Plots numerical distributions.\n"
        )
        result = _strip_signature_echo(raw)
        assert "Plots numerical distributions." in result
        assert "plot_numerical_distributions(" not in result

    # Single-line signature echo (the exact bug from the issue report)
    def test_single_line_signature_echo_returns_empty(self) -> None:
        """A single-line function signature echo is cleaned to empty string."""
        from llm.utils import _clean_docstring

        raw = "evaluate(subset, whitelist, num_cols, cat_cols, logger: PipelineLogger)"
        assert _clean_docstring(raw) == ""

    def test_single_line_signature_with_return_type(self) -> None:
        """Single-line signature with return-type annotation is cleaned."""
        from llm.utils import _clean_docstring

        assert _clean_docstring("foo(x: int, y: str) -> bool") == ""
        assert _clean_docstring("transform_features(df) -> Tuple[csr_matrix, np.ndarray]") == ""

    def test_single_line_signature_not_confused_with_prose(self) -> None:
        """Legitimate prose that contains parentheses must survive cleaning."""
        from llm.utils import _clean_docstring

        # These should NOT be treated as signature echoes
        prose1 = "Processes data (with optional filtering)."
        assert _clean_docstring(prose1) == prose1
        prose2 = "Calls evaluate(x) and returns result."
        assert _clean_docstring(prose2) == prose2
        text = "Do something useful.\n\nArgs:\n    x: The input."
        assert _clean_docstring(text) == text


# ---------------------------------------------------------------------------
# Tests: new helper functions
# ---------------------------------------------------------------------------

class TestIsDegenerate:
    def test_empty_string(self) -> None:
        from llm.utils import _is_degenerate
        assert _is_degenerate("") is True

    def test_whitespace_only(self) -> None:
        from llm.utils import _is_degenerate
        assert _is_degenerate("   \n  ") is True

    def test_backtick_spam(self) -> None:
        from llm.utils import _is_degenerate
        spam = " ".join(["```"] * 100)
        assert _is_degenerate(spam) is True

    def test_normal_text(self) -> None:
        from llm.utils import _is_degenerate
        assert _is_degenerate("Do something useful.") is False


class TestStripEscapedQuotes:
    def test_replaces_escaped_double_quotes(self) -> None:
        from llm.utils import _strip_escaped_quotes
        assert _strip_escaped_quotes(r'say \"hello\"') == 'say "hello"'

    def test_replaces_escaped_single_quotes(self) -> None:
        from llm.utils import _strip_escaped_quotes
        assert _strip_escaped_quotes(r"it\'s fine") == "it's fine"


class TestConfidenceHeuristicNew:
    def test_degenerate_returns_zero(self) -> None:
        from llm.utils import _confidence_heuristic
        spam = " ".join(["```"] * 50)
        assert _confidence_heuristic(spam) == 0.0

    def test_text_with_fence_markers_penalised(self) -> None:
        from llm.utils import _confidence_heuristic
        text = "```markdown\nSomething useful."
        assert _confidence_heuristic(text) == pytest.approx(0.1)

    def test_signature_echo_penalised(self) -> None:
        from llm.utils import _confidence_heuristic
        text = (
            "transform_features(\n"
            "    pl_df: pd.DataFrame,\n"
            ") -> Tuple:\n"
            "    Transforms features.\n"
        )
        assert _confidence_heuristic(text) == pytest.approx(0.1)

    def test_single_line_signature_echo_returns_zero(self) -> None:
        from llm.utils import _confidence_heuristic
        text = "evaluate(subset, whitelist, num_cols, cat_cols, logger: PipelineLogger)"
        assert _confidence_heuristic(text) == 0.0


# ---------------------------------------------------------------------------
# Tests: confidence-based retry in _generate_for_node
# ---------------------------------------------------------------------------

class TestLowConfidenceRetry:
    """Verify that _generate_for_node retries on low-confidence results."""

    def test_low_confidence_triggers_retry(self, tmp_path: Path) -> None:
        """When confidence < 0.5, the node generator retries up to max_retries."""
        from phases.phase3_docstrings import _generate_for_node
        from llm.fallback import FallbackRouter

        # Provider always returns a minimal one-line response (confidence ≈ 0.33)
        call_count = 0

        class CountingProvider(BaseLLMProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                nonlocal call_count
                call_count += 1
                return "Does something."  # low confidence: only summary, no Args/Returns

            def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
                return {}

            def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
                text = self.generate(prompt)
                from llm.utils import _confidence_heuristic
                return text, _confidence_heuristic(text)

        provider = CountingProvider()
        router = FallbackRouter(primary=provider, fallback=None, confidence_threshold=0.7)

        py_file = tmp_path / "sample.py"
        py_file.write_text("def foo():\n    pass\n")
        node = ASTNode(
            file_path=str(py_file),
            node_type=NodeType.FUNCTION,
            name="foo",
            line_start=1,
            line_end=2,
        )
        source_lines = py_file.read_text().splitlines(keepends=True)

        result = _generate_for_node(
            node=node,
            source_lines=source_lines,
            router=router,
            caller_map={},
            callee_map={},
            max_retries=2,
        )
        # Should have been called 3 times (initial + 2 retries), then accepted
        assert call_count == 3
        # The result is a DocstringEntry (accepted on final attempt regardless)
        from models.schemas import DocstringEntry
        assert isinstance(result, DocstringEntry)
        assert result.retries == 2

    def test_high_confidence_no_retry(self, tmp_path: Path) -> None:
        """When confidence >= 0.5, no retry is triggered."""
        from phases.phase3_docstrings import _generate_for_node
        from llm.fallback import FallbackRouter

        call_count = 0

        class HighConfProvider(BaseLLMProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                nonlocal call_count
                call_count += 1
                return _CANNED_DOCSTRING

            def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
                return {}

            def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
                from llm.utils import _confidence_heuristic
                text = self.generate(prompt)
                return text, _confidence_heuristic(text)

        provider = HighConfProvider()
        router = FallbackRouter(primary=provider, fallback=None, confidence_threshold=0.7)

        py_file = tmp_path / "sample.py"
        py_file.write_text("def foo():\n    pass\n")
        node = ASTNode(
            file_path=str(py_file),
            node_type=NodeType.FUNCTION,
            name="foo",
            line_start=1,
            line_end=2,
        )
        source_lines = py_file.read_text().splitlines(keepends=True)

        result = _generate_for_node(
            node=node,
            source_lines=source_lines,
            router=router,
            caller_map={},
            callee_map={},
            max_retries=2,
        )
        assert call_count == 1  # no retry needed
        from models.schemas import DocstringEntry
        assert isinstance(result, DocstringEntry)
        assert result.retries == 0

    def test_signature_echo_triggers_failure(self, tmp_path: Path) -> None:
        """A single-line signature echo is cleaned to empty, producing a failure."""
        from phases.phase3_docstrings import _generate_for_node
        from llm.fallback import FallbackRouter
        from models.schemas import DocstringFailure

        call_count = 0

        class SigEchoProvider(BaseLLMProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                nonlocal call_count
                call_count += 1
                # Mimic the bug: LLM echoes the function signature
                return "evaluate(subset, whitelist, num_cols, cat_cols, logger: PipelineLogger)"

            def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
                return {}

            def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
                text = self.generate(prompt)
                from llm.utils import _confidence_heuristic
                return text, _confidence_heuristic(text)

        provider = SigEchoProvider()
        router = FallbackRouter(primary=provider, fallback=None, confidence_threshold=0.7)

        py_file = tmp_path / "sample.py"
        py_file.write_text("def evaluate(subset):\n    pass\n")
        node = ASTNode(
            file_path=str(py_file),
            node_type=NodeType.FUNCTION,
            name="evaluate",
            line_start=1,
            line_end=2,
        )
        source_lines = py_file.read_text().splitlines(keepends=True)

        result = _generate_for_node(
            node=node,
            source_lines=source_lines,
            router=router,
            caller_map={},
            callee_map={},
            max_retries=2,
        )
        # Cleaned to empty on every attempt → retries exhausted → DocstringFailure
        assert call_count == 3  # initial + 2 retries
        assert isinstance(result, DocstringFailure)
        assert result.error_type == "malformed_response"


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

    # --- Circuit-breaker tests ---

    def test_connection_error_disables_fallback_immediately(self) -> None:
        """A ConnectionError from the fallback should trip the circuit breaker."""
        primary = FakeLLMProvider(_LOW_CONFIDENCE_DOCSTRING)
        fallback = RaisingLLMProvider(ConnectionError("refused"))
        router = FallbackRouter(primary, fallback, confidence_threshold=0.9)

        # First call: fallback tried, fails with ConnectionError → disabled
        text, _, _, fallback_used = router.generate_with_fallback("p")
        assert not fallback_used  # fell back to primary result
        assert router.fallback_disabled

        # Second call: fallback NOT tried (circuit open), returns primary
        text2, _, _, fallback_used2 = router.generate_with_fallback("p2")
        assert not fallback_used2
        assert text2 == _LOW_CONFIDENCE_DOCSTRING

    def test_consecutive_failures_disable_fallback(self) -> None:
        """Non-connection errors disable fallback after threshold consecutive failures."""
        primary = FakeLLMProvider(_LOW_CONFIDENCE_DOCSTRING)
        fallback = RaisingLLMProvider(RuntimeError("model error"))
        threshold = 3
        router = FallbackRouter(
            primary, fallback,
            confidence_threshold=0.9,
            circuit_breaker_threshold=threshold,
        )

        # Failures 1 and 2: fallback still enabled
        for i in range(threshold - 1):
            router.generate_with_fallback(f"p{i}")
            assert not router.fallback_disabled, f"Should not be disabled after {i + 1} failures"

        # Failure 3: reaches threshold → circuit opens
        router.generate_with_fallback("final")
        assert router.fallback_disabled

    def test_successful_fallback_resets_failure_counter(self) -> None:
        """A successful fallback call should reset the failure counter."""
        call_count = 0

        class AlternatingProvider(BaseLLMProvider):
            """Fails on first N calls, then succeeds."""
            def __init__(self, fail_count: int) -> None:
                self._fail_count = fail_count
                self.calls: list[str] = []

            def generate(self, prompt: str, **kwargs: Any) -> str:
                nonlocal call_count
                call_count += 1
                self.calls.append(prompt)
                if call_count <= self._fail_count:
                    raise RuntimeError("temporary")
                return _CANNED_DOCSTRING

            def generate_structured(self, prompt: str, schema: type, **kwargs: Any) -> dict:
                return {}

            def generate_with_confidence(self, prompt: str, **kwargs: Any) -> tuple[str, float]:
                text = self.generate(prompt, **kwargs)
                from llm.utils import _confidence_heuristic
                return text, _confidence_heuristic(text)

        primary = FakeLLMProvider(_LOW_CONFIDENCE_DOCSTRING)
        # Fails twice, then succeeds
        fallback = AlternatingProvider(fail_count=2)
        router = FallbackRouter(
            primary, fallback,
            confidence_threshold=0.9,
            circuit_breaker_threshold=3,
        )

        # Two failures
        router.generate_with_fallback("a")
        router.generate_with_fallback("b")
        assert not router.fallback_disabled
        assert router._consecutive_fallback_failures == 2

        # Third call succeeds → counter resets
        call_count = 2  # already failed twice
        _, _, _, fallback_used = router.generate_with_fallback("c")
        assert fallback_used
        assert router._consecutive_fallback_failures == 0
        assert not router.fallback_disabled

    def test_oserror_subclass_disables_fallback(self) -> None:
        """OSError subclasses (requests.exceptions.ConnectionError) trip the breaker."""
        primary = FakeLLMProvider(_LOW_CONFIDENCE_DOCSTRING)
        fallback = RaisingLLMProvider(OSError("network unreachable"))
        router = FallbackRouter(primary, fallback, confidence_threshold=0.9)

        router.generate_with_fallback("p")
        assert router.fallback_disabled


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
