"""phases/phase3_docstrings.py — Docstring Generation (Phase 3).

Generates Google-style docstrings for every function/class/module in the
codebase using LLM providers with confidence-based fallback routing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from llm.fallback import FallbackRouter
from models.schemas import (
    ASTNode,
    ASTNodes,
    CallGraph,
    DocstringEntry,
    DocstringFailure,
    Docstrings,
    ModuleDocstring,
    NodeType,
    PhaseArtifacts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_FUNCTION_PROMPT_TEMPLATE = """\
Generate a Google-style Python docstring for the following function.

Required sections:
- Summary (one line)
- Args (name, type, description for each parameter)
- Returns (type and description)
- Raises (exception type and condition, if applicable)
- Side Effects (any I/O, DB writes, state mutations — if applicable)

Context:
- This function is in file: {file_path}
- It is called by: {caller_list}
- It calls: {callee_list}
- Class context (if method): {class_context}

Function source code:
```python
{function_source}
```

Respond with ONLY the docstring text (no code fences, no extra commentary).
Start directly with the summary line."""

_STRICT_FUNCTION_PROMPT_TEMPLATE = """\
Generate a Google-style Python docstring for the following function.

IMPORTANT: Your response must start with a one-line summary sentence and include
Args, Returns, and Raises sections where applicable. No code fences or extra text.

Function source code:
```python
{function_source}
```

Respond with ONLY the docstring text."""

_MODULE_PROMPT_TEMPLATE = """\
Generate a Google-style module docstring summarizing what this Python file does.

File: {file_path}

This file contains the following functions and classes:
{function_list}

The file imports from: {import_list}
It is part of the component: {component_name}

Respond with ONLY the module docstring (2-4 sentences)."""


# ---------------------------------------------------------------------------
# LLM provider factory
# ---------------------------------------------------------------------------

def _build_provider(name: str, doc_cfg: dict[str, Any]) -> Any:
    """Instantiate an LLM provider from configuration."""
    if name == "ollama":
        from llm.ollama_provider import OllamaProvider  # noqa: PLC0415
        cfg = doc_cfg.get("ollama", {})
        return OllamaProvider(
            model_name=cfg.get("model", "qwen2.5-coder:7b"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            timeout=int(cfg.get("timeout", 60)),
        )
    if name == "openai":
        from llm.openai_provider import OpenAIProvider  # noqa: PLC0415
        cfg = doc_cfg.get("openai", {})
        return OpenAIProvider(
            model=cfg.get("model", "gpt-4o-mini"),
            timeout=int(cfg.get("timeout", 60)),
        )
    if name == "anthropic":
        from llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
        cfg = doc_cfg.get("anthropic", {})
        return AnthropicProvider(
            model=cfg.get("model", "claude-sonnet-4-20250514"),
            timeout=int(cfg.get("timeout", 60)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
        )
    raise ValueError(f"Unknown LLM provider: {name!r}")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_function_id(file_path: str, node: ASTNode) -> str:
    class_name = getattr(node, "class_name", None)
    if class_name:
        return f"{file_path}::{class_name}::{node.name}"
    return f"{file_path}::{node.name}"


def _read_source_lines(file_path: str, line_start: int, line_end: int) -> str:
    """Read source lines from a file (1-indexed, inclusive)."""
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[line_start - 1 : line_end])


def _callers_for(function_name: str, file_path: str, call_graph: CallGraph) -> list[str]:
    result: list[str] = []
    for entry in call_graph.entries:
        for callee in entry.callees:
            if callee.function == function_name and callee.file == file_path:
                result.append(f"{entry.caller.file}::{entry.caller.function}")
    return result


def _callees_for(function_name: str, file_path: str, call_graph: CallGraph) -> list[str]:
    for entry in call_graph.entries:
        if entry.caller.function == function_name and entry.caller.file == file_path:
            return [f"{c.file}::{c.function}" for c in entry.callees]
    return []


def _is_valid_docstring(text: str) -> bool:
    """Return True if the text looks like a real docstring (non-empty summary)."""
    stripped = text.strip()
    return bool(stripped) and len(stripped.splitlines()[0].strip()) > 3


# ---------------------------------------------------------------------------
# Per-node docstring generation
# ---------------------------------------------------------------------------

def _generate_for_node(
    node: ASTNode,
    file_path: str,
    source: str,
    call_graph: CallGraph,
    router: FallbackRouter,
    max_retries: int,
) -> DocstringEntry | DocstringFailure:
    """Generate a docstring for a single AST node (sync)."""
    function_id = _build_function_id(file_path, node)
    class_name = getattr(node, "class_name", None)

    callers = _callers_for(node.name, file_path, call_graph)
    callees = _callees_for(node.name, file_path, call_graph)

    class_context = (
        f"This is a method of class {class_name}" if class_name else "N/A (top-level function)"
    )
    prompt = _FUNCTION_PROMPT_TEMPLATE.format(
        file_path=file_path,
        caller_list=", ".join(callers) if callers else "none",
        callee_list=", ".join(callees) if callees else "none",
        class_context=class_context,
        function_source=source,
    )

    retries = 0
    start_ms = time.monotonic()

    for attempt in range(max_retries + 1):
        try:
            text, confidence, provider_used, fallback_used = router.generate_with_fallback(
                prompt if attempt == 0 else _STRICT_FUNCTION_PROMPT_TEMPLATE.format(
                    function_source=source
                )
            )
        except TimeoutError:
            if attempt < max_retries:
                retries += 1
                logger.warning("Timeout generating docstring for %s, retrying…", function_id)
                continue
            latency_ms = (time.monotonic() - start_ms) * 1000.0
            logger.warning("All retries exhausted for %s (timeout)", function_id)
            return DocstringFailure(
                function_id=function_id,
                file_path=file_path,
                function_name=node.name,
                reason="LLM timed out after all retries",
                provider=type(router.primary).__name__,
                error_type="timeout",
            )
        except Exception as exc:
            logger.warning("Error generating docstring for %s: %s", function_id, exc)
            return DocstringFailure(
                function_id=function_id,
                file_path=file_path,
                function_name=node.name,
                reason=str(exc),
                provider=type(router.primary).__name__,
                error_type="parse_error",
            )

        if _is_valid_docstring(text):
            latency_ms = (time.monotonic() - start_ms) * 1000.0
            logger.info(
                "Docstring generated for %s (confidence: %.2f)", function_id, confidence
            )
            return DocstringEntry(
                function_id=function_id,
                file_path=file_path,
                function_name=node.name,
                class_name=class_name,
                node_type=node.node_type,
                docstring=text.strip(),
                confidence=confidence,
                provider_used=provider_used,
                fallback_used=fallback_used,
                retries=retries,
                latency_ms=round(latency_ms, 2),
            )

        # Malformed response — retry with stricter prompt
        retries += 1
        logger.warning(
            "Malformed response for %s (attempt %d), retrying with stricter prompt…",
            function_id,
            attempt + 1,
        )

    # All retries exhausted with malformed responses
    return DocstringFailure(
        function_id=function_id,
        file_path=file_path,
        function_name=node.name,
        reason="LLM returned malformed response after all retries",
        provider=provider_used,  # type: ignore[possibly-undefined]
        error_type="malformed_response",
    )


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_file(
    file_path: str,
    nodes: list[ASTNode],
    call_graph: CallGraph,
    dependency_tree: Any,
    component_map: Any,
    router: FallbackRouter,
    max_retries: int,
    file_index: int,
    total_files: int,
) -> tuple[list[DocstringEntry], list[DocstringFailure], ModuleDocstring | None]:
    """Process all nodes in a single file and generate a module docstring."""
    logger.info(
        "Processing file %d of %d: %s (%d functions)",
        file_index,
        total_files,
        file_path,
        len(nodes),
    )

    entries: list[DocstringEntry] = []
    failures: list[DocstringFailure] = []

    # Read source file once; if it fails, record all nodes as file_read_error
    try:
        full_source_lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Cannot read file %s: %s — skipping.", file_path, exc)
        for node in nodes:
            function_id = _build_function_id(file_path, node)
            failures.append(
                DocstringFailure(
                    function_id=function_id,
                    file_path=file_path,
                    function_name=node.name,
                    reason=str(exc),
                    provider=type(router.primary).__name__,
                    error_type="file_read_error",
                )
            )
        return entries, failures, None

    for node in nodes:
        source = "\n".join(
            full_source_lines[node.line_start - 1 : node.line_end]
        )
        result = _generate_for_node(
            node, file_path, source, call_graph, router, max_retries
        )
        if isinstance(result, DocstringEntry):
            entries.append(result)
        else:
            failures.append(result)

    # Module docstring
    module_doc: ModuleDocstring | None = None
    function_summaries = "\n".join(
        f"- {e.function_name}: {e.docstring.splitlines()[0]}" for e in entries
    )
    if not function_summaries:
        function_summaries = "(no functions successfully documented)"

    # Determine component name
    component_name = "unknown"
    if component_map:
        for comp in component_map.components:
            if file_path in comp.files:
                component_name = comp.component_name
                break

    # Import list
    import_list = "none"
    if dependency_tree:
        imports = [
            imp.imported_module
            for imp in dependency_tree.internal_imports
            if imp.source_file == file_path
        ]
        if imports:
            import_list = ", ".join(imports)

    module_prompt = _MODULE_PROMPT_TEMPLATE.format(
        file_path=file_path,
        function_list=function_summaries,
        import_list=import_list,
        component_name=component_name,
    )

    try:
        mod_text, mod_confidence, mod_provider, _ = router.generate_with_fallback(
            module_prompt
        )
        if mod_text.strip():
            module_doc = ModuleDocstring(
                file_path=file_path,
                docstring=mod_text.strip(),
                confidence=mod_confidence,
                provider_used=mod_provider,
            )
    except Exception as exc:
        logger.warning("Failed to generate module docstring for %s: %s", file_path, exc)

    return entries, failures, module_doc


# ---------------------------------------------------------------------------
# Async wrapper for parallel file processing
# ---------------------------------------------------------------------------

async def _process_files_async(
    file_groups: dict[str, list[ASTNode]],
    call_graph: CallGraph,
    dependency_tree: Any,
    component_map: Any,
    router: FallbackRouter,
    max_retries: int,
    concurrency: int,
) -> tuple[list[DocstringEntry], list[DocstringFailure], list[ModuleDocstring]]:
    semaphore = asyncio.Semaphore(concurrency)
    all_entries: list[DocstringEntry] = []
    all_failures: list[DocstringFailure] = []
    all_module_docs: list[ModuleDocstring] = []

    file_paths = list(file_groups.keys())
    total_files = len(file_paths)

    async def _process_one(fp: str, idx: int) -> None:
        async with semaphore:
            loop = asyncio.get_event_loop()
            entries, failures, mod_doc = await loop.run_in_executor(
                None,
                _process_file,
                fp,
                file_groups[fp],
                call_graph,
                dependency_tree,
                component_map,
                router,
                max_retries,
                idx,
                total_files,
            )
            all_entries.extend(entries)
            all_failures.extend(failures)
            if mod_doc:
                all_module_docs.append(mod_doc)

    tasks = [_process_one(fp, i + 1) for i, fp in enumerate(file_paths)]
    await asyncio.gather(*tasks)

    return all_entries, all_failures, all_module_docs


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_phase3(*, config: dict[str, Any], artifacts: PhaseArtifacts) -> None:
    """Run Phase 3: Docstring Generation.

    Reads AST nodes from *artifacts*, generates Google-style docstrings for
    every function/class/module using the configured LLM providers, and writes
    ``docstrings.json`` to the artifacts directory.

    Parameters
    ----------
    config:
        Loaded configuration dictionary (from ``config.yaml``).
    artifacts:
        Shared artifact bag.  Reads ``ast_nodes`` and ``call_graph`` and
        writes ``docstrings``.

    Raises
    ------
    RuntimeError
        If Phase 2 artifacts (``ast_nodes``) are not present in *artifacts*.
    """
    if artifacts.ast_nodes is None:
        raise RuntimeError(
            "Phase 3 requires ast_nodes from Phase 2. Run Phase 2 first."
        )

    arts_dir = Path(config.get("output", {}).get("artifacts_dir", "output/artifacts"))
    arts_dir.mkdir(parents=True, exist_ok=True)

    doc_cfg: dict[str, Any] = config.get("docstring", {})
    primary_name: str = doc_cfg.get("primary_provider", "ollama")
    fallback_name: str | None = doc_cfg.get("fallback_provider") or None
    confidence_threshold: float = float(doc_cfg.get("confidence_threshold", 0.7))
    max_retries: int = int(doc_cfg.get("max_retries", 2))
    concurrency: int = int(doc_cfg.get("concurrency", 4))

    # ------------------------------------------------------------------
    # Build providers
    # ------------------------------------------------------------------
    primary = _build_provider(primary_name, doc_cfg)

    if fallback_name:
        fallback = _build_provider(fallback_name, doc_cfg)
    else:
        # Use a no-op fallback that mirrors the primary
        fallback = primary

    router = FallbackRouter(
        primary=primary,
        fallback=fallback,
        confidence_threshold=confidence_threshold,
    )

    # ------------------------------------------------------------------
    # Group AST nodes by file
    # ------------------------------------------------------------------
    ast_nodes: ASTNodes = artifacts.ast_nodes
    call_graph: CallGraph = artifacts.call_graph or CallGraph()
    dependency_tree = artifacts.dependency_tree
    component_map = artifacts.component_map

    file_groups: dict[str, list[ASTNode]] = {}
    for node in ast_nodes.nodes:
        file_groups.setdefault(node.file_path, []).append(node)

    logger.info("Phase 3: generating docstrings for %d files…", len(file_groups))

    # ------------------------------------------------------------------
    # Run async processing
    # ------------------------------------------------------------------
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    all_entries, all_failures, all_module_docs = loop.run_until_complete(
        _process_files_async(
            file_groups,
            call_graph,
            dependency_tree,
            component_map,
            router,
            max_retries,
            concurrency,
        )
    )

    # ------------------------------------------------------------------
    # Compute summary statistics
    # ------------------------------------------------------------------
    total_functions = len(ast_nodes.nodes)
    successful = len(all_entries)
    failed = len(all_failures)
    fallback_count = sum(1 for e in all_entries if e.fallback_used)
    average_confidence = (
        sum(e.confidence for e in all_entries) / successful if successful else 0.0
    )

    docstrings = Docstrings(
        entries=all_entries,
        module_docstrings=all_module_docs,
        failures=all_failures,
        total_functions=total_functions,
        successful=successful,
        failed=failed,
        average_confidence=round(average_confidence, 4),
        fallback_count=fallback_count,
    )

    # ------------------------------------------------------------------
    # Persist artifact
    # ------------------------------------------------------------------
    out_path = arts_dir / "docstrings.json"
    out_path.write_text(docstrings.model_dump_json(indent=2), encoding="utf-8")
    logger.info("docstrings.json written to %s", out_path)

    artifacts.docstrings = docstrings

    logger.info(
        "Phase 3 complete: %d/%d functions documented, %d failures, "
        "avg confidence=%.2f, fallback_count=%d",
        successful,
        total_functions,
        failed,
        average_confidence,
        fallback_count,
    )
