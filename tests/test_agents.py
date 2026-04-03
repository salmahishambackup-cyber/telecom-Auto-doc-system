"""Tests for Phase 4 quality-improvement agents.

Covers DocstringValidator, ArtifactCleaner, and ConfidenceImprover with
unit-level tests.  All tests are self-contained (no real file I/O for
source reading where avoidable).
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.artifact_cleaner import ArtifactCleaner, PYTHON_BUILTINS
from agents.confidence_improver import ConfidenceImprover
from agents.docstring_validator import DocstringValidator, ValidationReport
from models.schemas import (
    ASTNode,
    ASTNodes,
    CallGraph,
    CallGraphEntry,
    CalleeRef,
    CallerRef,
    DataFlow,
    DataFlowEntry,
    DocstringEntry,
    Docstrings,
    FlowType,
    ModuleDocstring,
    NodeType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    function_id: str = "file.py::my_func",
    docstring: str = "Does something.\n\nArgs:\n    x: A value.\n\nReturns:\n    int",
    confidence: float = 0.75,
    function_name: str = "my_func",
    file_path: str = "file.py",
) -> DocstringEntry:
    return DocstringEntry(
        function_id=function_id,
        file_path=file_path,
        function_name=function_name,
        node_type=NodeType.FUNCTION,
        docstring=docstring,
        confidence=confidence,
        provider_used="test",
    )


def _make_node(
    name: str = "my_func",
    file_path: str = "file.py",
    args: list[str] | None = None,
    line_start: int = 1,
    line_end: int = 10,
) -> ASTNode:
    return ASTNode(
        file_path=file_path,
        node_type=NodeType.FUNCTION,
        name=name,
        args=args or [],
        line_start=line_start,
        line_end=line_end,
    )


# ---------------------------------------------------------------------------
# DocstringValidator tests
# ---------------------------------------------------------------------------

class TestDocstringValidator:
    """Tests for DocstringValidator."""

    def _validate(
        self,
        docstring: str,
        args: list[str] | None = None,
        source_lines: list[str] | None = None,
        file_path: str = "file.py",
    ) -> ValidationReport:
        """Helper: validate a single entry."""
        fid = f"{file_path}::my_func"
        entry = _make_entry(
            function_id=fid,
            docstring=docstring,
            file_path=file_path,
        )
        node = _make_node(
            name="my_func",
            file_path=file_path,
            args=args or [],
            line_start=1,
            line_end=len(source_lines) if source_lines else 5,
        )
        docstrings = Docstrings(entries=[entry])
        ast_nodes = ASTNodes(nodes=[node])

        fake_source = "\n".join(source_lines) if source_lines else "def my_func(): pass"
        validator = DocstringValidator()
        with patch("builtins.open"), patch.object(
            Path, "read_text", return_value=fake_source
        ):
            return validator.validate(docstrings, ast_nodes)

    # ------------------------------------------------------------------
    # Stub detection
    # ------------------------------------------------------------------

    def test_stub_detection_short(self) -> None:
        report = self._validate("Too short.")
        issue_types = [i["issue_type"] for i in report.issues]
        assert "stub" in issue_types
        assert report.failed == 1

    def test_stub_detection_passes_long_docstring(self) -> None:
        report = self._validate(
            "Does something useful with a long description.",
            source_lines=["def my_func(): pass"],
        )
        stub_issues = [i for i in report.issues if i["issue_type"] == "stub"]
        assert not stub_issues

    # ------------------------------------------------------------------
    # Code contamination detection
    # ------------------------------------------------------------------

    def test_code_contamination_detected(self) -> None:
        contaminated = (
            "Does something.\n\n"
            "def helper():\n"
            "    pass\n"
            "self.x = 1\n"
        )
        report = self._validate(contaminated)
        issue_types = [i["issue_type"] for i in report.issues]
        assert "code_contamination" in issue_types

    def test_clean_docstring_no_contamination(self) -> None:
        clean = (
            "Does something useful.\n\nArgs:\n    x: A value.\n\nReturns:\n    int"
        )
        report = self._validate(clean, source_lines=["def my_func(x): return x"])
        contamination_issues = [
            i for i in report.issues if i["issue_type"] == "code_contamination"
        ]
        assert not contamination_issues

    # ------------------------------------------------------------------
    # Hallucinated Raises detection
    # ------------------------------------------------------------------

    def test_hallucinated_raises_detected(self) -> None:
        """Docstring claims ValueError but source has no raise statement."""
        docstring = (
            "Does something.\n\nArgs:\n    x: int\n\n"
            "Raises:\n    ValueError: If x is negative.\n\nReturns:\n    int"
        )
        source_lines = ["def my_func(x):", "    return x"]
        report = self._validate(docstring, args=["x"], source_lines=source_lines)
        raise_issues = [
            i for i in report.issues if i["issue_type"] == "hallucinated_raises"
        ]
        assert raise_issues
        assert "ValueError" in raise_issues[0]["details"]

    def test_valid_raises_not_flagged(self) -> None:
        """Docstring mentions ValueError and source actually raises it."""
        docstring = (
            "Does something.\n\nArgs:\n    x: int\n\n"
            "Raises:\n    ValueError: If x is negative.\n\nReturns:\n    int"
        )
        source_lines = [
            "def my_func(x):",
            "    if x < 0:",
            "        raise ValueError('negative')",
            "    return x",
        ]
        report = self._validate(docstring, args=["x"], source_lines=source_lines)
        raise_issues = [
            i for i in report.issues if i["issue_type"] == "hallucinated_raises"
        ]
        assert not raise_issues

    # ------------------------------------------------------------------
    # Args mismatch detection
    # ------------------------------------------------------------------

    def test_args_mismatch_extra_documented(self) -> None:
        """Docstring documents 'z' which is not in the function signature."""
        docstring = (
            "Does something.\n\nArgs:\n    x: int\n    z: str\n\nReturns:\n    None"
        )
        report = self._validate(
            docstring,
            args=["x"],
            source_lines=["def my_func(x): pass"],
        )
        mismatch = [i for i in report.issues if i["issue_type"] == "args_mismatch"]
        assert any("z" in i["details"] for i in mismatch)

    def test_args_self_ignored(self) -> None:
        """'self' should not be flagged as a mismatch."""
        docstring = (
            "Does something.\n\nArgs:\n    x: int\n\nReturns:\n    None"
        )
        report = self._validate(
            docstring,
            args=["self", "x"],
            source_lines=["def my_func(self, x): pass"],
        )
        mismatch = [i for i in report.issues if i["issue_type"] == "args_mismatch"]
        assert not mismatch

    # ------------------------------------------------------------------
    # Pass rate
    # ------------------------------------------------------------------

    def test_pass_rate_all_passing(self) -> None:
        report = ValidationReport(passed=10, failed=0)
        assert report.pass_rate == 1.0

    def test_pass_rate_mixed(self) -> None:
        report = ValidationReport(passed=3, failed=1)
        assert report.pass_rate == pytest.approx(0.75)

    def test_pass_rate_empty(self) -> None:
        report = ValidationReport()
        assert report.pass_rate == 1.0


# ---------------------------------------------------------------------------
# ArtifactCleaner tests
# ---------------------------------------------------------------------------

class TestArtifactCleaner:
    """Tests for ArtifactCleaner."""

    # ------------------------------------------------------------------
    # clean_call_graph
    # ------------------------------------------------------------------

    def test_builtins_removed(self) -> None:
        caller = CallerRef(file="a.py", function="my_func")
        callees = [
            CalleeRef(file="a.py", function="len", line=1),
            CalleeRef(file="a.py", function="list", line=2),
            CalleeRef(file="b.py", function="helper", line=3),
        ]
        cg = CallGraph(entries=[CallGraphEntry(caller=caller, callees=callees)])
        result = ArtifactCleaner.clean_call_graph(cg)
        assert len(result.entries) == 1
        callee_names = [c.function for c in result.entries[0].callees]
        assert "len" not in callee_names
        assert "list" not in callee_names
        assert "helper" in callee_names

    def test_entry_removed_when_all_callees_are_builtins(self) -> None:
        caller = CallerRef(file="a.py", function="my_func")
        callees = [
            CalleeRef(file="a.py", function="len", line=1),
            CalleeRef(file="a.py", function="str", line=2),
        ]
        cg = CallGraph(entries=[CallGraphEntry(caller=caller, callees=callees)])
        result = ArtifactCleaner.clean_call_graph(cg)
        assert len(result.entries) == 0

    def test_project_callees_preserved(self) -> None:
        caller = CallerRef(file="a.py", function="my_func")
        callees = [
            CalleeRef(file="b.py", function="prepare_data", line=5),
            CalleeRef(file="c.py", function="run_model", line=10),
        ]
        cg = CallGraph(entries=[CallGraphEntry(caller=caller, callees=callees)])
        result = ArtifactCleaner.clean_call_graph(cg)
        assert len(result.entries) == 1
        assert len(result.entries[0].callees) == 2

    def test_all_known_builtins_in_set(self) -> None:
        """Spot-check that key builtins are present in the set."""
        for name in ("len", "list", "set", "int", "float", "print", "range"):
            assert name in PYTHON_BUILTINS

    # ------------------------------------------------------------------
    # clean_data_flow
    # ------------------------------------------------------------------

    @staticmethod
    def _make_df_entry(details: str, flow_type: FlowType = FlowType.DB_READ) -> DataFlowEntry:
        return DataFlowEntry(
            file="a.py",
            function="f",
            flow_type=flow_type,
            details=details,
            line=1,
        )

    def test_false_positive_select_removed(self) -> None:
        """'selected_subset' should not be treated as a DB read."""
        entry = self._make_df_entry(
            "Pattern matched: selected_subset = df[mask]"
        )
        df = DataFlow(entries=[entry])
        result = ArtifactCleaner.clean_data_flow(df)
        assert len(result.entries) == 0

    def test_real_sql_select_kept(self) -> None:
        """Genuine SQL SELECT … FROM should be kept."""
        entry = self._make_df_entry(
            "Pattern matched: SELECT id, name FROM users"
        )
        df = DataFlow(entries=[entry])
        result = ArtifactCleaner.clean_data_flow(df)
        assert len(result.entries) == 1

    def test_non_db_read_entries_unchanged(self) -> None:
        """FILE_IO entries should pass through unmodified."""
        entry = self._make_df_entry(
            "Pattern matched: open('file.txt')", flow_type=FlowType.FILE_IO
        )
        df = DataFlow(entries=[entry])
        result = ArtifactCleaner.clean_data_flow(df)
        assert len(result.entries) == 1

    # ------------------------------------------------------------------
    # clean_docstrings
    # ------------------------------------------------------------------

    def test_code_fences_stripped(self) -> None:
        raw = "```python\nDoes something useful.\n```"
        entry = _make_entry(docstring=raw)
        docstrings = Docstrings(entries=[entry])
        result = ArtifactCleaner.clean_docstrings(docstrings)
        assert "```" not in result.entries[0].docstring

    def test_module_docstrings_cleaned(self) -> None:
        mod = ModuleDocstring(
            file_path="a.py",
            docstring="```\nA module.\n```",
            confidence=0.8,
            provider_used="test",
        )
        docstrings = Docstrings(module_docstrings=[mod])
        result = ArtifactCleaner.clean_docstrings(docstrings)
        assert "```" not in result.module_docstrings[0].docstring


# ---------------------------------------------------------------------------
# ConfidenceImprover tests
# ---------------------------------------------------------------------------

class TestConfidenceImprover:
    """Tests for ConfidenceImprover."""

    # ------------------------------------------------------------------
    # find_weak_entries
    # ------------------------------------------------------------------

    def test_low_confidence_flagged(self) -> None:
        entry = _make_entry(confidence=0.3)
        docstrings = Docstrings(entries=[entry])
        improver = ConfidenceImprover()
        weak = improver.find_weak_entries(docstrings, threshold=0.5)
        assert len(weak) == 1
        assert "0.30" in weak[0]["reason"]

    def test_high_confidence_not_flagged(self) -> None:
        entry = _make_entry(
            confidence=0.8,
            docstring="Does something useful.\n\nArgs:\n    x: int\n\nReturns:\n    int",
        )
        docstrings = Docstrings(entries=[entry])
        improver = ConfidenceImprover()
        weak = improver.find_weak_entries(docstrings, threshold=0.5)
        assert len(weak) == 0

    def test_stub_docstring_flagged(self) -> None:
        entry = _make_entry(confidence=0.9, docstring="Short.")
        docstrings = Docstrings(entries=[entry])
        improver = ConfidenceImprover()
        weak = improver.find_weak_entries(docstrings, threshold=0.5)
        assert any("stub" in w["reason"] for w in weak)

    def test_validation_issue_flagged(self) -> None:
        entry = _make_entry(
            function_id="file.py::my_func",
            confidence=0.8,
            docstring="Does something useful with lots of detail.",
        )
        docstrings = Docstrings(entries=[entry])
        issues = [{"function_id": "file.py::my_func", "issue_type": "hallucinated_raises", "details": "x"}]
        improver = ConfidenceImprover()
        weak = improver.find_weak_entries(docstrings, threshold=0.5, validation_issues=issues)
        assert len(weak) == 1
        assert "validation" in weak[0]["reason"]

    def test_max_regenerations_cap(self) -> None:
        entries = [_make_entry(function_id=f"f.py::func{i}", confidence=0.1) for i in range(100)]
        docstrings = Docstrings(entries=entries)
        improver = ConfidenceImprover(max_regenerations=10)
        weak = improver.find_weak_entries(docstrings)
        assert len(weak) == 10

    # ------------------------------------------------------------------
    # build_improved_prompt
    # ------------------------------------------------------------------

    def test_prompt_includes_function_id(self) -> None:
        entry = {
            "function_id": "mymodule.py::my_func",
            "reason": "low confidence",
            "current_confidence": 0.2,
            "current_docstring": "Does stuff.",
        }
        prompt = ConfidenceImprover.build_improved_prompt(entry)
        assert "mymodule.py::my_func" in prompt

    def test_prompt_includes_raises_feedback(self) -> None:
        entry = {
            "function_id": "f.py::func",
            "reason": "validation failed",
            "current_confidence": 0.5,
            "current_docstring": "Does something.",
        }
        issues = [
            {
                "function_id": "f.py::func",
                "issue_type": "hallucinated_raises",
                "details": "ValueError not in source",
            }
        ]
        prompt = ConfidenceImprover.build_improved_prompt(entry, validation_issues=issues)
        assert "Do NOT document Raises" in prompt

    def test_prompt_includes_args_feedback(self) -> None:
        entry = {
            "function_id": "f.py::func",
            "reason": "validation failed",
            "current_confidence": 0.5,
            "current_docstring": "Does something.",
        }
        issues = [
            {
                "function_id": "f.py::func",
                "issue_type": "args_mismatch",
                "details": "Extra arg 'z' documented",
            }
        ]
        prompt = ConfidenceImprover.build_improved_prompt(entry, validation_issues=issues)
        assert "Fix Args section" in prompt

    def test_prompt_includes_reason(self) -> None:
        entry = {
            "function_id": "f.py::func",
            "reason": "confidence 0.30 below threshold 0.50",
            "current_confidence": 0.3,
            "current_docstring": "Meh.",
        }
        prompt = ConfidenceImprover.build_improved_prompt(entry)
        assert "confidence 0.30" in prompt
