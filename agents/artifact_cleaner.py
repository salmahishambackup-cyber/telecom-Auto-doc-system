"""Artifact cleaner agent.

Post-processes all pipeline artifacts to remove noise:

* :meth:`ArtifactCleaner.clean_call_graph` — filters Python builtins from
  callee lists.
* :meth:`ArtifactCleaner.clean_data_flow` — removes false-positive data-flow
  entries.
* :meth:`ArtifactCleaner.clean_docstrings` — strips markdown fences and LLM
  filler from docstring text.
"""
from __future__ import annotations

import re

from llm.utils import _clean_docstring
from models.schemas import (
    CallGraph,
    CallGraphEntry,
    DataFlow,
    DataFlowEntry,
    Docstrings,
    FlowType,
    ModuleDocstring,
)

# ---------------------------------------------------------------------------
# Python built-in names that clutter call graphs
# ---------------------------------------------------------------------------

PYTHON_BUILTINS: frozenset[str] = frozenset({
    "len", "list", "set", "int", "float", "range", "enumerate", "zip",
    "isinstance", "print", "copy", "items", "append", "tolist", "str",
    "dict", "tuple", "bool", "max", "min", "sum", "abs", "any", "all",
    "sorted", "reversed", "map", "filter", "type", "super", "hasattr",
    "getattr", "setattr", "delattr", "id", "hash", "round", "format",
    "repr", "chr", "ord", "hex", "oct", "bin", "iter", "next", "slice",
    "property", "staticmethod", "classmethod", "vars", "dir", "help",
    "input", "open", "exec", "eval", "compile", "globals", "locals",
    "breakpoint",
})

# SQL keywords that indicate genuine SQL context
_SQL_KEYWORDS_RE = re.compile(
    r"\b(FROM|WHERE|JOIN|GROUP\s+BY|ORDER\s+BY|INSERT|UPDATE|DELETE)\b",
    re.IGNORECASE,
)


class ArtifactCleaner:
    """Post-process pipeline artifacts to remove noise.

    All methods are static — the class exists only to group the related
    cleaning operations.
    """

    @staticmethod
    def clean_call_graph(call_graph: CallGraph) -> CallGraph:
        """Remove Python built-in callees from a call graph.

        Entries whose callee list becomes empty after filtering are dropped
        entirely.

        Parameters
        ----------
        call_graph:
            The :class:`~models.schemas.CallGraph` to clean.

        Returns
        -------
        CallGraph
            A new :class:`~models.schemas.CallGraph` with built-ins removed.
        """
        cleaned_entries: list[CallGraphEntry] = []
        for entry in call_graph.entries:
            filtered = [
                callee
                for callee in entry.callees
                if callee.function not in PYTHON_BUILTINS
            ]
            if filtered:
                cleaned_entries.append(
                    CallGraphEntry(caller=entry.caller, callees=filtered)
                )
        return CallGraph(entries=cleaned_entries)

    @staticmethod
    def clean_data_flow(data_flow: DataFlow) -> DataFlow:
        """Remove false-positive data-flow entries.

        For ``DB_READ`` entries, checks whether the matched line contains
        genuine SQL context (``FROM``, ``WHERE``, ``JOIN``, …).  Entries
        that look like plain Python assignments without SQL keywords are
        also removed.

        Parameters
        ----------
        data_flow:
            The :class:`~models.schemas.DataFlow` to clean.

        Returns
        -------
        DataFlow
            A new :class:`~models.schemas.DataFlow` with false positives
            removed.
        """
        cleaned: list[DataFlowEntry] = []
        for entry in data_flow.entries:
            if entry.flow_type == FlowType.DB_READ:
                # Keep only if there is genuine SQL context in the detail line
                if not _SQL_KEYWORDS_RE.search(entry.details):
                    continue
            cleaned.append(entry)
        return DataFlow(entries=cleaned)

    @staticmethod
    def clean_docstrings(docstrings: Docstrings) -> Docstrings:
        """Strip markdown code fences and LLM filler from docstring text.

        Applies :func:`~llm.utils._clean_docstring` to every
        :class:`~models.schemas.DocstringEntry` and
        :class:`~models.schemas.ModuleDocstring`.

        Parameters
        ----------
        docstrings:
            The :class:`~models.schemas.Docstrings` container to clean.

        Returns
        -------
        Docstrings
            A new :class:`~models.schemas.Docstrings` with cleaned text.
        """
        cleaned_entries = []
        for entry in docstrings.entries:
            cleaned_entries.append(
                entry.model_copy(update={"docstring": _clean_docstring(entry.docstring)})
            )

        cleaned_modules: list[ModuleDocstring] = []
        for mod in docstrings.module_docstrings:
            cleaned_modules.append(
                mod.model_copy(update={"docstring": _clean_docstring(mod.docstring)})
            )

        return docstrings.model_copy(
            update={
                "entries": cleaned_entries,
                "module_docstrings": cleaned_modules,
            }
        )
