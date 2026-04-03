"""Data flow pattern detector.

Uses a hybrid AST + regex approach to identify data-flow patterns such as
database reads/writes, external API calls, file I/O, environment variable
access, message-queue operations, and socket usage.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Optional

from models.schemas import DataFlow, DataFlowEntry, FlowType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default patterns (can be overridden via config)
# ---------------------------------------------------------------------------

DEFAULT_PATTERNS: dict[FlowType, list[str]] = {
    FlowType.DB_READ: [
        "session.query",
        "cursor.execute",
        r"\bSELECT\b\s+\w+.*\bFROM\b",
        r"\.find\(",
        r"\.find_one\(",
        r"objects\.all\(",
        r"objects\.filter\(",
        r"objects\.get\(",
    ],
    FlowType.DB_WRITE: [
        "session.add",
        "session.commit",
        "cursor.execute",
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bDELETE\b",
        r"\.insert_one\(",
        r"\.update_one\(",
        r"\.delete_one\(",
        r"session\.delete\(",
        r"objects\.create\(",
        r"objects\.update\(",
        r"objects\.delete\(",
        r"bulk_create\(",
        r"bulk_update\(",
    ],
    FlowType.API_CALL: [
        r"requests\.get\(",
        r"requests\.post\(",
        r"requests\.put\(",
        r"requests\.delete\(",
        r"requests\.patch\(",
        r"requests\.request\(",
        r"httpx\.",
        r"aiohttp\.",
        r"urllib\.request",
        r"urllib\.urlopen\(",
        r"http\.client",
        r"ClientSession\(",
        r"AsyncClient\(",
    ],
    FlowType.FILE_IO: [
        r"\bopen\(",
        r"pathlib\.Path",
        r"Path\(",
        r"shutil\.",
        r"os\.remove\(",
        r"os\.rename\(",
        r"os\.makedirs\(",
        r"csv\.reader\(",
        r"csv\.writer\(",
        r"json\.load\(",
        r"json\.dump\(",
        r"yaml\.load\(",
        r"yaml\.dump\(",
    ],
    FlowType.ENV_VAR: [
        r"os\.environ",
        r"os\.getenv\(",
        r"dotenv",
        r"load_dotenv\(",
        r"environ\.get\(",
    ],
    FlowType.MESSAGE_QUEUE: [
        r"pika\.",
        r"kombu\.",
        r"\bcelery\b",
        r"kafka\.",
        r"confluent_kafka",
        r"redis\.publish",
        r"redis\.subscribe",
        r"\.publish\(",
        r"\.subscribe\(",
        r"KafkaProducer\(",
        r"KafkaConsumer\(",
        r"Producer\(",
        r"Consumer\(",
    ],
    FlowType.SOCKET: [
        r"socket\.",
        r"zmq\.",
        r"websocket",
        r"WebSocket\(",
        r"create_connection\(",
        r"asyncio\.open_connection\(",
        r"asyncio\.start_server\(",
    ],
}


def _compile_patterns(
    patterns: dict[FlowType, list[str]],
) -> dict[FlowType, list[re.Pattern[str]]]:
    compiled: dict[FlowType, list[re.Pattern[str]]] = {}
    for flow_type, pattern_list in patterns.items():
        compiled[flow_type] = [
            re.compile(p, re.IGNORECASE) for p in pattern_list
        ]
    return compiled


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class _DataFlowVisitor(ast.NodeVisitor):
    """Walk a file's AST and detect data-flow patterns on a per-line basis."""

    def __init__(
        self,
        file_path: str,
        source_lines: list[str],
        compiled_patterns: dict[FlowType, list[re.Pattern[str]]],
    ) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.compiled_patterns = compiled_patterns
        self.entries: list[DataFlowEntry] = []
        self._function_stack: list[str] = []

    # ------------------------------------------------------------------

    def _current_function(self) -> str:
        return self._function_stack[-1] if self._function_stack else "<module>"

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        line_no = node.lineno
        if 1 <= line_no <= len(self.source_lines):
            line_text = self.source_lines[line_no - 1]
            self._check_line(line_text, line_no)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        # Catch bare expressions that may contain patterns (e.g. raw SQL strings)
        line_no = node.lineno
        if 1 <= line_no <= len(self.source_lines):
            line_text = self.source_lines[line_no - 1]
            self._check_line(line_text, line_no)
        self.generic_visit(node)

    def _check_line(self, line_text: str, line_no: int) -> None:
        for flow_type, patterns in self.compiled_patterns.items():
            for pat in patterns:
                if pat.search(line_text):
                    self.entries.append(
                        DataFlowEntry(
                            file=self.file_path,
                            function=self._current_function(),
                            flow_type=flow_type,
                            details=f"Pattern '{pat.pattern}' matched: {line_text.strip()[:120]}",
                            line=line_no,
                        )
                    )
                    # One match per flow_type per line is sufficient
                    break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_data_flows(
    files: list[str | Path],
    patterns: Optional[dict[FlowType, list[str]]] = None,
) -> DataFlow:
    """Detect data-flow patterns across all *files*.

    Parameters
    ----------
    files:
        Python source files to scan.
    patterns:
        Optional override for default patterns.  Keys are :class:`FlowType`
        values; values are lists of regex strings.

    Returns
    -------
    DataFlow
    """
    compiled = _compile_patterns(patterns or DEFAULT_PATTERNS)
    all_entries: list[DataFlowEntry] = []

    for fp in files:
        path = Path(fp)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            logger.warning("Skipping %s for data flow detection: %s", path, exc)
            continue

        lines = source.splitlines()
        visitor = _DataFlowVisitor(
            file_path=str(path),
            source_lines=lines,
            compiled_patterns=compiled,
        )
        visitor.visit(tree)

        # De-duplicate by (file, function, flow_type, line) before extending
        seen: set[tuple[str, str, FlowType, int]] = set()
        for entry in visitor.entries:
            key = (entry.file, entry.function, entry.flow_type, entry.line)
            if key not in seen:
                seen.add(key)
                all_entries.append(entry)

        logger.debug("Data flow: %d entries in %s", len(seen), path)

    logger.info("Data flow detection complete: %d total entries", len(all_entries))
    return DataFlow(entries=all_entries)
