"""phases/phase3_docstrings.py — Phase 3: Docstring Generation.

Generates Google-style docstrings for every function, class, and module
in the analysed codebase using an LLM provider with confidence-based
fallback routing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from llm.base import BaseLLMProvider
from llm.fallback import FallbackRouter
from models.schemas import (
    ASTNode,
    DocstringEntry,
    DocstringFailure,
    Docstrings,
    ModuleDocstring,
    NodeType,
    PhaseArtifacts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_FUNCTION_PROMPT = """\
Generate a Google-style Python docstring for the following function.

Required sections:
- Summary (one line)
- Args (name, type, description for each parameter)
- Returns (type and description)
- Raises (exception type and condition, if applicable)
- Side Effects (any I/O, DB writes, state mutations — if applicable)

Context:
- This function is in file: {file_path}
- It is called by: {caller_list_or_none}
- It calls: {callee_list_or_none}
- Class context (if method): {class_context}

Function source code:
{function_source}

Respond with ONLY the docstring text (no code fences, no extra commentary).
Start directly with the summary line.\
"""

_FUNCTION_STRICT_PROMPT = """\
Generate a Google-style Python docstring. You MUST include these sections: \
Summary, Args, Returns.

Function source code:
{function_source}

Respond with ONLY the docstring text. No code fences. Start with the summary.\
"""

_MODULE_PROMPT = """\
Generate a Google-style module docstring summarizing what this Python file does.

File: {file_path}
This file contains: {list_of_function_summaries}
Imports from: {import_list}
Component: {component_name}

Respond with ONLY the module docstring (2-4 sentences).\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_source_slice(file_path: str, line_start: int, line_end: int) -> str:
    """Read a slice of a source file by line numbers.

    Args:
        file_path: Absolute or relative path to the Python file.
        line_start: First line to read (1-indexed, inclusive).
        line_end: Last line to read (1-indexed, inclusive).

    Returns:
        The source code slice as a string.

    Raises:
        OSError: If the file cannot be read.
    """
    lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
    # line numbers are 1-indexed
    return "\n".join(lines[line_start - 1 : line_end])


def _build_function_id(node: ASTNode) -> str:
    """Build a unique function identifier string.

    Args:
        node: The AST node representing the function or method.

    Returns:
        Identifier in the form ``"file_path::ClassName::func_name"`` for
        methods or ``"file_path::func_name"`` for module-level functions.
    """
    if node.node_type == NodeType.METHOD and hasattr(node, "class_name") and node.class_name:  # type: ignore[attr-defined]
        return f"{node.file_path}::{node.class_name}::{node.name}"  # type: ignore[attr-defined]
    return f"{node.file_path}::{node.name}"


def _build_caller_list(function_id: str, call_graph_entries: list[Any]) -> str:
    """Build a string listing all callers of a function.

    Args:
        function_id: The function's unique identifier (file::func).
        call_graph_entries: List of CallGraphEntry objects.

    Returns:
        Comma-separated caller names or ``"None"`` if no callers found.
    """
    # function_id format: "file_path::func_name"
    parts = function_id.rsplit("::", 1)
    func_name = parts[-1]
    callers = [
        f"{e.caller.file}::{e.caller.function}"
        for e in call_graph_entries
        for callee in e.callees
        if callee.function == func_name
    ]
    return ", ".join(callers) if callers else "None"


def _build_callee_list(function_id: str, call_graph_entries: list[Any]) -> str:
    """Build a string listing all callees of a function.

    Args:
        function_id: The function's unique identifier (file::func).
        call_graph_entries: List of CallGraphEntry objects.

    Returns:
        Comma-separated callee names or ``"None"`` if no callees found.
    """
    parts = function_id.rsplit("::", 1)
    func_name = parts[-1]
    callees = [
        f"{callee.file}::{callee.function}"
        for e in call_graph_entries
        if e.caller.function == func_name
        for callee in e.callees
    ]
    return ", ".join(callees) if callees else "None"


def _is_malformed(text: str) -> bool:
    """Check whether a docstring response is malformed.

    Args:
        text: The LLM response text to check.

    Returns:
        True if the response appears malformed (empty or only whitespace).
    """
    return not text or not text.strip()


def _build_llm_provider(config: dict[str, Any]) -> BaseLLMProvider:
    """Instantiate the primary LLM provider from config.

    Args:
        config: The parsed YAML config dict.

    Returns:
        An instantiated BaseLLMProvider.

    Raises:
        ValueError: If the configured provider is unknown.
    """
    ds_cfg = config.get("docstrings", {})
    provider_name: str = ds_cfg.get("provider", "ollama")

    if provider_name == "ollama":
        from llm.ollama_provider import OllamaProvider

        return OllamaProvider(
            model_name=ds_cfg.get("model", "qwen2.5-coder:7b"),
            base_url=ds_cfg.get("base_url", "http://localhost:11434"),
            timeout=ds_cfg.get("timeout", 60),
        )
    if provider_name == "openai":
        from llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=ds_cfg.get("model", "gpt-4o-mini"),
            timeout=ds_cfg.get("timeout", 60),
        )
    if provider_name == "anthropic":
        from llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=ds_cfg.get("model", "claude-sonnet-4-20250514"),
            timeout=ds_cfg.get("timeout", 60),
            max_tokens=ds_cfg.get("max_tokens", 2048),
        )
    raise ValueError(f"Unknown LLM provider: {provider_name!r}")


def _build_fallback_provider(config: dict[str, Any]) -> BaseLLMProvider | None:
    """Instantiate the fallback LLM provider from config, if configured.

    Args:
        config: The parsed YAML config dict.

    Returns:
        An instantiated BaseLLMProvider, or None if not configured.
    """
    ds_cfg = config.get("docstrings", {})
    fallback_name: str | None = ds_cfg.get("fallback_provider")
    if not fallback_name:
        return None

    if fallback_name == "openai":
        from llm.openai_provider import OpenAIProvider

        return OpenAIProvider(timeout=ds_cfg.get("timeout", 60))
    if fallback_name == "anthropic":
        from llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(timeout=ds_cfg.get("timeout", 60))
    if fallback_name == "ollama":
        from llm.ollama_provider import OllamaProvider

        return OllamaProvider(timeout=ds_cfg.get("timeout", 60))
    logger.warning("Unknown fallback provider %r — fallback disabled", fallback_name)
    return None


# ---------------------------------------------------------------------------
# Per-node processing
# ---------------------------------------------------------------------------


async def _process_node(
    node: ASTNode,
    router: FallbackRouter,
    call_graph_entries: list[Any],
    semaphore: asyncio.Semaphore,
) -> DocstringEntry | DocstringFailure:
    """Generate a docstring for a single AST node.

    Args:
        node: The AST node (function, method, or class) to document.
        router: The fallback-capable LLM router.
        call_graph_entries: All call graph entries for caller/callee lookup.
        semaphore: Concurrency-limiting semaphore.

    Returns:
        A DocstringEntry on success or a DocstringFailure on error.
    """
    function_id = _build_function_id(node)

    async with semaphore:
        # Read source code
        try:
            source = await asyncio.get_event_loop().run_in_executor(
                None,
                _read_source_slice,
                node.file_path,
                node.line_start,
                node.line_end,
            )
        except OSError as exc:
            logger.error("File read error for %s: %s", function_id, exc)
            return DocstringFailure(
                function_id=function_id,
                file_path=node.file_path,
                function_name=node.name,
                reason=str(exc),
                provider="none",
                error_type="file_read_error",
            )

        caller_list = _build_caller_list(function_id, call_graph_entries)
        callee_list = _build_callee_list(function_id, call_graph_entries)
        class_name: str | None = getattr(node, "class_name", None)
        class_context = (
            f"This is a method of class {class_name}"
            if class_name
            else "Not a method"
        )

        prompt = _FUNCTION_PROMPT.format(
            file_path=node.file_path,
            caller_list_or_none=caller_list,
            callee_list_or_none=callee_list,
            class_context=class_context,
            function_source=source,
        )

        retries = 0
        for attempt in range(3):  # original + 2 retries
            try:
                use_prompt = (
                    prompt
                    if attempt == 0
                    else _FUNCTION_STRICT_PROMPT.format(function_source=source)
                )
                text, confidence, provider_name, fallback_used = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda p=use_prompt: router.generate_with_fallback(p),
                )
            except TimeoutError as exc:
                if attempt < 1:
                    logger.warning(
                        "Timeout on %s (attempt %d), retrying: %s",
                        function_id,
                        attempt,
                        exc,
                    )
                    retries += 1
                    continue
                logger.error("Timeout on %s after retries", function_id)
                return DocstringFailure(
                    function_id=function_id,
                    file_path=node.file_path,
                    function_name=node.name,
                    reason=str(exc),
                    provider="unknown",
                    error_type="timeout",
                )
            except Exception as exc:
                logger.error("LLM error on %s: %s", function_id, exc)
                return DocstringFailure(
                    function_id=function_id,
                    file_path=node.file_path,
                    function_name=node.name,
                    reason=str(exc),
                    provider="unknown",
                    error_type="llm_error",
                )

            if _is_malformed(text):
                if attempt < 2:
                    logger.warning(
                        "Malformed response for %s (attempt %d), retrying",
                        function_id,
                        attempt,
                    )
                    retries += 1
                    continue
                logger.error("Malformed response for %s after retries", function_id)
                return DocstringFailure(
                    function_id=function_id,
                    file_path=node.file_path,
                    function_name=node.name,
                    reason="LLM returned empty or malformed response",
                    provider=provider_name,
                    error_type="malformed_response",
                )

            # Success
            logger.info(
                "Docstring generated for %s (confidence: %.3f)",
                function_id,
                confidence,
            )
            return DocstringEntry(
                function_id=function_id,
                file_path=node.file_path,
                function_name=node.name,
                class_name=class_name,
                node_type=node.node_type,
                docstring=text.strip(),
                confidence=confidence,
                provider_used=provider_name,
                fallback_used=fallback_used,
                retries=retries,
            )

        # Should not reach here, but guard anyway
        return DocstringFailure(
            function_id=function_id,
            file_path=node.file_path,
            function_name=node.name,
            reason="Exceeded max attempts",
            provider="unknown",
            error_type="malformed_response",
        )


# ---------------------------------------------------------------------------
# Module docstring generation
# ---------------------------------------------------------------------------


async def _process_module(
    file_path: str,
    nodes: list[ASTNode],
    router: FallbackRouter,
    semaphore: asyncio.Semaphore,
    dependency_tree: Any | None,
    component_map: Any | None,
) -> ModuleDocstring | None:
    """Generate a module-level docstring for one Python file.

    Args:
        file_path: Path to the Python file.
        nodes: All AST nodes belonging to this file.
        router: The fallback-capable LLM router.
        semaphore: Concurrency-limiting semaphore.
        dependency_tree: Optional dependency tree artifact for import info.
        component_map: Optional component map for component context.

    Returns:
        A ModuleDocstring on success or None on error.
    """
    func_names = [n.name for n in nodes if n.node_type != NodeType.CLASS]
    func_summary = ", ".join(func_names[:10]) or "no functions"

    import_list = "unknown"
    if dependency_tree is not None:
        imports = [
            imp.imported_module
            for imp in dependency_tree.internal_imports
            if imp.source_file == file_path
        ]
        import_list = ", ".join(imports) or "none"

    component_name = "unknown"
    if component_map is not None:
        for comp in component_map.components:
            if file_path in comp.files:
                component_name = comp.component_name
                break

    prompt = _MODULE_PROMPT.format(
        file_path=file_path,
        list_of_function_summaries=func_summary,
        import_list=import_list,
        component_name=component_name,
    )

    async with semaphore:
        try:
            text, confidence, provider_name, _fallback_used = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: router.generate_with_fallback(prompt),
            )
            if _is_malformed(text):
                return None
            return ModuleDocstring(
                file_path=file_path,
                docstring=text.strip(),
                confidence=confidence,
                provider_used=provider_name,
            )
        except Exception as exc:
            logger.error("Module docstring failed for %s: %s", file_path, exc)
            return None


# ---------------------------------------------------------------------------
# Main Phase 3 entry point
# ---------------------------------------------------------------------------


def run_phase3(*, config: dict[str, Any], artifacts: PhaseArtifacts) -> None:
    """Run Phase 3: generate docstrings for all functions, methods, and modules.

    Args:
        config: The parsed YAML configuration dictionary.
        artifacts: The shared PhaseArtifacts instance; both reads prior
            phases' outputs and writes the final ``docstrings`` field.

    Raises:
        RuntimeError: If no AST nodes are available (Phase 2 must run first).
    """
    asyncio.run(_run_phase3_async(config=config, artifacts=artifacts))


async def _run_phase3_async(
    *, config: dict[str, Any], artifacts: PhaseArtifacts
) -> None:
    """Async implementation of Phase 3.

    Args:
        config: The parsed YAML configuration dictionary.
        artifacts: The shared PhaseArtifacts instance.
    """
    ds_cfg = config.get("docstrings", {})
    concurrency: int = ds_cfg.get("concurrency", 4)
    artifacts_dir = Path(
        config.get("output", {}).get("artifacts_dir", "output/artifacts")
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if artifacts.ast_nodes is None:
        raise RuntimeError(
            "Phase 3 requires ast_nodes from Phase 2. Run Phase 2 first."
        )

    nodes = [
        n
        for n in artifacts.ast_nodes.nodes
        if n.node_type in (NodeType.FUNCTION, NodeType.METHOD, NodeType.CLASS)
    ]
    call_graph_entries = (
        artifacts.call_graph.entries if artifacts.call_graph is not None else []
    )

    # ------------------------------------------------------------------
    # Build LLM router
    # ------------------------------------------------------------------
    primary = _build_llm_provider(config)
    fallback = _build_fallback_provider(config)
    router = FallbackRouter(
        primary=primary,
        fallback=fallback,
        confidence_threshold=ds_cfg.get("confidence_threshold", 0.7),
    )

    # ------------------------------------------------------------------
    # Group nodes by file for progress reporting
    # ------------------------------------------------------------------
    files: dict[str, list[ASTNode]] = {}
    for node in nodes:
        files.setdefault(node.file_path, []).append(node)

    total_files = len(files)
    logger.info("Phase 3: %d function/method/class nodes across %d files", len(nodes), total_files)

    semaphore = asyncio.Semaphore(concurrency)
    entries: list[DocstringEntry] = []
    failures: list[DocstringFailure] = []

    # ------------------------------------------------------------------
    # Process nodes per file
    # ------------------------------------------------------------------
    all_file_paths = list(files.keys())
    for file_idx, fp in enumerate(all_file_paths, start=1):
        logger.info("Processing file %d of %d: %s", file_idx, total_files, fp)
        file_nodes = files[fp]
        tasks = [
            _process_node(node, router, call_graph_entries, semaphore)
            for node in file_nodes
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            if isinstance(result, DocstringEntry):
                entries.append(result)
            else:
                failures.append(result)

    # ------------------------------------------------------------------
    # Module docstrings
    # ------------------------------------------------------------------
    module_tasks = [
        _process_module(
            fp,
            files.get(fp, []),
            router,
            semaphore,
            artifacts.dependency_tree,
            artifacts.component_map,
        )
        for fp in all_file_paths
    ]
    module_results = await asyncio.gather(*module_tasks)
    module_docstrings = [m for m in module_results if m is not None]

    # ------------------------------------------------------------------
    # Build summary stats
    # ------------------------------------------------------------------
    total = len(nodes)
    successful = len(entries)
    failed = len(failures)
    avg_confidence = (
        sum(e.confidence for e in entries) / successful if successful else 0.0
    )
    fallback_count = sum(1 for e in entries if e.fallback_used)

    docstrings = Docstrings(
        entries=entries,
        module_docstrings=module_docstrings,
        failures=failures,
        total_functions=total,
        successful=successful,
        failed=failed,
        average_confidence=round(avg_confidence, 4),
        fallback_count=fallback_count,
    )
    artifacts.docstrings = docstrings

    # ------------------------------------------------------------------
    # Save artifact
    # ------------------------------------------------------------------
    out_path = artifacts_dir / "docstrings.json"
    out_path.write_text(
        json.dumps(json.loads(docstrings.model_dump_json()), indent=2),
        encoding="utf-8",
    )
    logger.info("Docstrings saved to %s", out_path)

    success_rate = (successful / total * 100) if total else 0.0
    logger.info(
        "Phase 3 summary | total=%d successful=%d failed=%d "
        "success_rate=%.1f%% avg_confidence=%.3f fallback_count=%d",
        total,
        successful,
        failed,
        success_rate,
        avg_confidence,
        fallback_count,
    )
