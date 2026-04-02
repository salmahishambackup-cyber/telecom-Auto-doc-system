"""Call-graph builder.

Builds caller → callee relationships for every function in a set of Python
files.  Both intra-file and cross-file (via import resolution) calls are
tracked.  Dynamic dispatch patterns (``getattr``, ``eval``,
``importlib.import_module``) are flagged as ``unresolved_dynamic``.
"""
from __future__ import annotations

import ast
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from models.schemas import CallGraph, CallGraphEntry, CalleeRef, CallerRef

logger = logging.getLogger(__name__)

_DYNAMIC_PATTERNS = frozenset({"getattr", "eval", "exec", "importlib.import_module"})


def _dotted(node: ast.expr) -> Optional[str]:
    """Return a dotted-name string for attribute / name nodes, or ``None``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _dotted(node.value)
        return f"{value}.{node.attr}" if value else node.attr
    return None


class _FunctionScope:
    """Tracks the current function name and collected callees during a walk."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.callees: list[tuple[str, int]] = []  # (callee_name, line)


class _FileCallVisitor(ast.NodeVisitor):
    """Walk a single file's AST and record all call sites per function."""

    def __init__(
        self,
        file_path: str,
        import_map: dict[str, str],  # local_name → qualified name or file path
    ) -> None:
        self.file_path = file_path
        self.import_map = import_map
        # Map: function_name → list of (callee, line)
        self.call_sites: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self._scope_stack: list[_FunctionScope] = []
        # Current class context for resolving self.method() calls
        self._class_stack: list[str] = []

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _current_scope(self) -> Optional[_FunctionScope]:
        return self._scope_stack[-1] if self._scope_stack else None

    # ------------------------------------------------------------------
    # Visitors
    # ------------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        scope = _FunctionScope(name=node.name)
        self._scope_stack.append(scope)
        self.generic_visit(node)
        self._scope_stack.pop()
        # Merge into file-level map (last writer wins for duplicates — use
        # extend so that all call sites from all occurrences are kept)
        self.call_sites[scope.name].extend(scope.callees)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        scope = self._current_scope()
        if scope is None:
            self.generic_visit(node)
            return

        callee_name: Optional[str] = None
        func = node.func

        if isinstance(func, ast.Name):
            callee_name = func.id
        elif isinstance(func, ast.Attribute):
            # self.method() → record as method name
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                callee_name = func.attr
            else:
                callee_name = _dotted(func)

        if callee_name is not None:
            # Check for dynamic patterns
            if callee_name in _DYNAMIC_PATTERNS:
                callee_name = f"unresolved_dynamic:{callee_name}"
            # Resolve through import map if possible
            resolved = self.import_map.get(callee_name, callee_name)
            scope.callees.append((resolved, node.lineno))

        self.generic_visit(node)


def _build_import_map(tree: ast.Module, file_path: str, project_root: str) -> dict[str, str]:
    """Build a mapping of local name → qualified import target.

    For intra-project imports the value is set to the *module path* so that
    the call-graph resolver can later match it against known file paths.
    """
    import_map: dict[str, str] = {}
    root = Path(project_root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                import_map[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                import_map[local] = f"{module}.{alias.name}" if module else alias.name
    return import_map


def build_call_graph(
    files: list[str | Path],
    project_root: str | Path,
) -> CallGraph:
    """Build a :class:`CallGraph` from the given list of Python *files*.

    Parameters
    ----------
    files:
        Python source files to analyse.
    project_root:
        Root of the project (used to distinguish internal from external calls).

    Returns
    -------
    CallGraph
    """
    root = str(project_root)
    # First pass: parse all files and collect per-file call sites
    # file_path → {function_name → [(callee, line), ...]}
    file_call_sites: dict[str, dict[str, list[tuple[str, int]]]] = {}

    # Also build a set of all known function names per file for resolution
    # file_path → set of function names defined in that file
    file_functions: dict[str, set[str]] = defaultdict(set)

    for fp in files:
        path = Path(fp)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            logger.warning("Skipping %s for call graph: %s", path, exc)
            continue

        import_map = _build_import_map(tree, str(path), root)
        visitor = _FileCallVisitor(file_path=str(path), import_map=import_map)
        visitor.visit(tree)
        file_call_sites[str(path)] = dict(visitor.call_sites)

        # Collect defined function names
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                file_functions[str(path)].add(node.name)

    # Build a reverse lookup: function_name → [file_paths that define it]
    func_to_files: dict[str, list[str]] = defaultdict(list)
    for fp, funcs in file_functions.items():
        for fn in funcs:
            func_to_files[fn].append(fp)

    entries: list[CallGraphEntry] = []
    for caller_file, call_map in file_call_sites.items():
        for caller_fn, callees_raw in call_map.items():
            callee_refs: list[CalleeRef] = []
            for callee_name, line in callees_raw:
                # Try to resolve to a file
                callee_file = caller_file  # default: same file
                pure_name = callee_name.split(".")[-1] if "." in callee_name else callee_name
                if callee_name.startswith("unresolved_dynamic:"):
                    callee_file = "unresolved_dynamic"
                    pure_name = callee_name
                elif pure_name in func_to_files:
                    candidates = func_to_files[pure_name]
                    # Prefer the same file; otherwise use the first match
                    if caller_file in candidates:
                        callee_file = caller_file
                    else:
                        callee_file = candidates[0]
                callee_refs.append(
                    CalleeRef(file=callee_file, function=pure_name, line=line)
                )

            entries.append(
                CallGraphEntry(
                    caller=CallerRef(file=caller_file, function=caller_fn),
                    callees=callee_refs,
                )
            )

    logger.info(
        "Call graph built: %d caller entries across %d files",
        len(entries),
        len(file_call_sites),
    )
    return CallGraph(entries=entries)
