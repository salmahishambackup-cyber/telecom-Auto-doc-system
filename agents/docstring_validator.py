"""Docstring validation agent.

Validates generated docstrings against source code AST to detect
hallucinations, argument mismatches, code contamination, and stub entries.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from models.schemas import ASTNode, ASTNodes, Docstrings


@dataclass
class ValidationReport:
    """Result of a docstring validation pass.

    Attributes
    ----------
    issues:
        List of dicts, each containing ``function_id``, ``issue_type``,
        and ``details``.
    passed:
        Number of docstring entries that passed all checks.
    failed:
        Number of docstring entries that failed at least one check.
    pass_rate:
        ``passed / (passed + failed)`` or 1.0 when there are no entries.
    """

    issues: list[dict] = field(default_factory=list)
    passed: int = 0
    failed: int = 0

    @property
    def pass_rate(self) -> float:
        """Return the fraction of entries that passed validation."""
        total = self.passed + self.failed
        return 1.0 if total == 0 else self.passed / total


# ---------------------------------------------------------------------------
# Code contamination pattern (single indicator suffices)
# ---------------------------------------------------------------------------

_CODE_PATTERNS = [
    re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE),
    re.compile(r"^\s*class\s+\w+[\s(:]", re.MULTILINE),
    re.compile(r"^\s*import\s+\w+", re.MULTILINE),
    re.compile(r"^\s*self\.\w+\s*=", re.MULTILINE),
]


def _has_code_contamination(text: str) -> bool:
    """Return True if *text* contains at least one Python code indicator."""
    return any(pat.search(text) for pat in _CODE_PATTERNS)


def _extract_raise_types(source: str) -> set[str]:
    """Parse *source* and return the set of exception class names raised.

    Parameters
    ----------
    source:
        Python source code of a function body (or any code block).

    Returns
    -------
    set[str]
        Exception names found in ``raise ExceptionType(...)`` statements.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            # ``raise ValueError(...)`` → Call node whose func is a Name
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                names.add(exc.func.id)
            # ``raise ValueError`` (bare, no call)
            elif isinstance(exc, ast.Name):
                names.add(exc.id)
            # ``raise module.Error(...)``
            elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Attribute):
                names.add(exc.func.attr)
    return names


def _parse_raises_section(docstring: str) -> list[str]:
    """Extract exception type names documented in the ``Raises:`` section.

    Parameters
    ----------
    docstring:
        The docstring text to parse.

    Returns
    -------
    list[str]
        Exception names listed under ``Raises:``.
    """
    names: list[str] = []
    in_raises = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped == "Raises:":
            in_raises = True
            continue
        if in_raises:
            # A new top-level section ends the Raises block
            if stripped and not stripped.startswith(" ") and stripped.endswith(":"):
                break
            # Lines like ``    ValueError: if ...`` or ``    ValueError``
            m = re.match(r"^\s+(\w+)(?:\s*:.*)?$", line)
            if m:
                names.append(m.group(1))
    return names


def _parse_args_section(docstring: str) -> set[str]:
    """Extract parameter names documented in the ``Args:`` or ``Parameters:`` section.

    Parameters
    ----------
    docstring:
        The docstring text to parse.

    Returns
    -------
    set[str]
        Parameter names listed under ``Args:`` or ``Parameters:``.
    """
    names: set[str] = set()
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped in ("Args:", "Parameters:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith(" ") and stripped.endswith(":"):
                break
            m = re.match(r"^\s+(\w+)\s*(?:\(.*?\))?\s*:", line)
            if m:
                names.add(m.group(1))
    return names


class DocstringValidator:
    """Validate generated docstrings against source code AST.

    All checks are stateless — the class exists only to group related
    validation logic.  Call :meth:`validate` to run all checks at once.
    """

    def validate(
        self,
        docstrings: Docstrings,
        ast_nodes: ASTNodes,
    ) -> ValidationReport:
        """Run all validation checks and return a report.

        Parameters
        ----------
        docstrings:
            The generated docstrings to validate.
        ast_nodes:
            The AST nodes extracted from source code.  Used to cross-
            reference function signatures and source lines.

        Returns
        -------
        ValidationReport
        """
        # Build a fast lookup: function_id → ASTNode
        node_map: dict[str, ASTNode] = {}
        for node in ast_nodes.nodes:
            fid = f"{node.file_path}::{node.name}"
            node_map[fid] = node

        report = ValidationReport()
        for entry in docstrings.entries:
            entry_issues: list[dict] = []
            fid = entry.function_id
            node = node_map.get(fid)

            # ------------------------------------------------------------------
            # 1. Stub detection
            # ------------------------------------------------------------------
            if not entry.docstring or len(entry.docstring.strip()) < 20:
                entry_issues.append({
                    "function_id": fid,
                    "issue_type": "stub",
                    "details": "Docstring is shorter than 20 characters.",
                })

            # ------------------------------------------------------------------
            # 2. Code contamination
            # ------------------------------------------------------------------
            if entry.docstring and _has_code_contamination(entry.docstring):
                entry_issues.append({
                    "function_id": fid,
                    "issue_type": "code_contamination",
                    "details": "Docstring contains Python code patterns.",
                })

            if node is not None:
                # ------------------------------------------------------------------
                # 3. Hallucinated Raises detection
                # ------------------------------------------------------------------
                documented_raises = _parse_raises_section(entry.docstring or "")
                if documented_raises:
                    # Read the source file and extract the function body
                    actual_raises = self._get_actual_raises(node)
                    for exc_name in documented_raises:
                        if exc_name not in actual_raises:
                            entry_issues.append({
                                "function_id": fid,
                                "issue_type": "hallucinated_raises",
                                "details": (
                                    f"'{exc_name}' is documented in Raises: but "
                                    "no matching raise statement was found in the source."
                                ),
                            })

                # ------------------------------------------------------------------
                # 4. Args mismatch detection
                # ------------------------------------------------------------------
                documented_args = _parse_args_section(entry.docstring or "")
                if documented_args:
                    # Strip type annotations to get bare names
                    actual_args = {
                        a.split(":")[0].lstrip("*").strip()
                        for a in node.args
                    } - {"self", "cls"}
                    extra = documented_args - actual_args - {"self", "cls"}
                    missing = actual_args - documented_args
                    if extra:
                        entry_issues.append({
                            "function_id": fid,
                            "issue_type": "args_mismatch",
                            "details": f"Args documented but not in signature: {sorted(extra)}",
                        })
                    if missing:
                        entry_issues.append({
                            "function_id": fid,
                            "issue_type": "args_mismatch",
                            "details": f"Signature args missing from docstring: {sorted(missing)}",
                        })

            if entry_issues:
                report.issues.extend(entry_issues)
                report.failed += 1
            else:
                report.passed += 1

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_actual_raises(node: ASTNode) -> set[str]:
        """Return exception names actually raised inside *node*'s source lines.

        Parameters
        ----------
        node:
            The :class:`~models.schemas.ASTNode` whose source file and
            line range define the function body.

        Returns
        -------
        set[str]
            Names of exceptions raised in the function, or an empty set if
            the source cannot be read or parsed.
        """
        try:
            from pathlib import Path
            source = Path(node.file_path).read_text(encoding="utf-8", errors="replace")
            lines = source.splitlines()
            body_lines = lines[node.line_start - 1: node.line_end]
            body_source = "\n".join(body_lines)
        except OSError:
            return set()
        return _extract_raise_types(body_source)
