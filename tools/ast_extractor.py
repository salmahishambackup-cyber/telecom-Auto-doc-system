"""AST node extraction tool.

Parses Python source files using the ``ast`` module and extracts classes,
functions, methods, and module-level variables into :class:`ASTNode` records.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

from models.schemas import ASTNode, ASTNodes, NodeType

logger = logging.getLogger(__name__)


def _get_arg_annotation(arg: ast.arg) -> str:
    """Return 'name: Type' if annotated, or just the bare argument name."""
    if arg.annotation is not None:
        return f"{arg.arg}: {ast.unparse(arg.annotation)}"
    return arg.arg


def _get_decorator_names(decorators: list[ast.expr]) -> list[str]:
    """Return string representations of decorator expressions."""
    result: list[str] = []
    for dec in decorators:
        try:
            result.append(ast.unparse(dec))
        except Exception:
            result.append("<unknown>")
    return result


def _get_return_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
    if node.returns is not None:
        try:
            return ast.unparse(node.returns)
        except Exception:
            return None
    return None


def _get_docstring(node: ast.AST) -> Optional[str]:
    return ast.get_docstring(node)  # type: ignore[arg-type]


def _build_arg_list(args: ast.arguments) -> list[str]:
    """Collect all argument names with type annotations as 'name: Type' strings."""
    result: list[str] = []
    # positional-only, regular, *args, keyword-only, **kwargs
    all_args: list[ast.arg] = (
        args.posonlyargs + args.args
    )
    for arg in all_args:
        result.append(_get_arg_annotation(arg))
    if args.vararg:
        result.append(f"*{_get_arg_annotation(args.vararg)}")
    for arg in args.kwonlyargs:
        result.append(_get_arg_annotation(arg))
    if args.kwarg:
        result.append(f"**{_get_arg_annotation(args.kwarg)}")
    return result


class _NodeCollector(ast.NodeVisitor):
    """Walk an AST tree and collect all class/function/method nodes."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.nodes: list[ASTNode] = []
        # Stack tracking whether we are inside a class body
        self._class_stack: list[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _end_lineno(self, node: ast.AST) -> int:
        return getattr(node, "end_lineno", getattr(node, "lineno", 1))

    # ------------------------------------------------------------------
    # Visitors
    # ------------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        decorators = _get_decorator_names(node.decorator_list)
        self.nodes.append(
            ASTNode(
                file_path=self.file_path,
                node_type=NodeType.CLASS,
                name=node.name,
                args=[],
                return_type=None,
                decorators=decorators,
                line_start=node.lineno,
                line_end=self._end_lineno(node),
                docstring_existing=_get_docstring(node),
            )
        )
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        decorators = _get_decorator_names(node.decorator_list)
        node_type = NodeType.METHOD if self._class_stack else NodeType.FUNCTION
        self.nodes.append(
            ASTNode(
                file_path=self.file_path,
                node_type=node_type,
                name=node.name,
                class_name=self._class_stack[-1] if self._class_stack else None,
                args=_build_arg_list(node.args),
                return_type=_get_return_annotation(node),
                decorators=decorators,
                line_start=node.lineno,
                line_end=self._end_lineno(node),
                docstring_existing=_get_docstring(node),
            )
        )
        # Push a *function* scope — nested functions inside a function are still
        # NodeType.FUNCTION (not METHOD), but nested functions inside a class
        # are methods only at the direct child level.  We keep the class stack
        # unchanged here so that nested classes inside functions still work.
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def extract_nodes(file_path: str | Path, source: Optional[str] = None) -> list[ASTNode]:
    """Parse *file_path* and return a list of :class:`ASTNode` records.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the Python file.
    source:
        Optional pre-loaded source string.  If ``None``, the file is read from
        disk.

    Returns
    -------
    list[ASTNode]
        Empty list if the file cannot be parsed (syntax error / read error).
    """
    path = Path(file_path)
    if source is None:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        logger.warning("Syntax error in %s (line %s): %s", path, exc.lineno, exc.msg)
        return []
    except Exception as exc:  # pragma: no cover
        logger.warning("Unexpected parse error in %s: %s", path, exc)
        return []

    collector = _NodeCollector(file_path=str(path), source_lines=source.splitlines())
    collector.visit(tree)
    return collector.nodes


def extract_nodes_from_directory(directory: str | Path) -> ASTNodes:
    """Recursively extract AST nodes from all ``*.py`` files under *directory*.

    Returns
    -------
    ASTNodes
        Aggregated result across all files.
    """
    root = Path(directory)
    all_nodes: list[ASTNode] = []
    for py_file in sorted(root.rglob("*.py")):
        nodes = extract_nodes(py_file)
        all_nodes.extend(nodes)
        logger.debug("Extracted %d nodes from %s", len(nodes), py_file)
    logger.info("Total AST nodes extracted: %d", len(all_nodes))
    return ASTNodes(nodes=all_nodes)
