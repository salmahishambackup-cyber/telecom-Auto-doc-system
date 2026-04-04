"""Phase 3: Docstring Generation.

Uses an LLM (via a configurable provider + optional fallback) to generate
Google-style docstrings for every function, method, and class discovered
in Phase 2.  Also generates a module-level docstring for every source file.

All file I/O errors and LLM failures are captured as :class:`DocstringFailure`
records; the phase never raises on individual node failures.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from llm.base import BaseLLMProvider
from llm.fallback import FallbackRouter
from llm.utils import _clean_docstring, _confidence_heuristic
from models.schemas import (
    ASTNode,
    CallGraph,
    DocstringEntry,
    DocstringFailure,
    Docstrings,
    ModuleDocstring,
    NodeType,
    PhaseArtifacts,
)

logger = logging.getLogger(__name__)

# Confidence threshold below which a docstring generation attempt is retried.
_LOW_CONFIDENCE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_DOCSTRING_SYSTEM = (
    "You are a technical documentation expert.  Write a concise Google-style "
    "Python docstring for the provided code snippet.  Output ONLY the docstring "
    "text (without the surrounding triple-quotes).\n\n"
    "Follow the Google Python Style Guide for docstrings:\n"
    "- First line: a concise one-line summary ending with a period.\n"
    "- If the function has arguments, include an 'Args:' section.\n"
    "- If the function returns a value, include a 'Returns:' section.\n"
    "- If the function raises exceptions, include a 'Raises:' section.\n"
    "- If the function yields values, include a 'Yields:' section instead of 'Returns:'.\n\n"
    "IMPORTANT: Do NOT output any Python code, function definitions, class "
    "definitions, or import statements.  Do NOT repeat the source code.  "
    "Output ONLY the plain-text docstring content."
)

_MODULE_SYSTEM = (
    "You are a technical documentation expert.  Write a concise module-level "
    "Python docstring for the provided source file.  Output ONLY the docstring "
    "text (without the surrounding triple-quotes).  Summarise the module's "
    "purpose in one paragraph.\n\n"
    "IMPORTANT: Do NOT output any Python code, function definitions, class "
    "definitions, or import statements.  Do NOT repeat the source code.  "
    "Output ONLY the plain-text docstring content."
)


def _build_function_prompt(
    node: ASTNode,
    source_lines: list[str],
    callers: list[str],
    callees: list[str],
) -> str:
    """Build the LLM prompt for a single function / method / class."""
    start = max(0, node.line_start - 1)
    end = min(len(source_lines), node.line_end)
    snippet = "".join(source_lines[start:end])

    caller_str = ", ".join(callers) if callers else "none"
    callee_str = ", ".join(callees) if callees else "none"

    return (
        f"{_DOCSTRING_SYSTEM}\n\n"
        f"File: {node.file_path}\n"
        f"Node type: {node.node_type.value}\n"
        f"Called by: {caller_str}\n"
        f"Calls: {callee_str}\n\n"
        f"```python\n{snippet}\n```\n\n"
        "Write the docstring now:"
    )


def _build_module_prompt(file_path: str, source: str) -> str:
    """Build the LLM prompt for a module-level docstring."""
    # Limit source to first 200 lines for context
    lines = source.splitlines()[:200]
    preview = "\n".join(lines)
    return (
        f"{_MODULE_SYSTEM}\n\n"
        f"File: {file_path}\n\n"
        f"```python\n{preview}\n```\n\n"
        "Write the module docstring now:"
    )


# ---------------------------------------------------------------------------
# Call-graph helpers
# ---------------------------------------------------------------------------

def _build_caller_map(call_graph: CallGraph) -> dict[str, list[str]]:
    """Return a mapping of callee function key → list of caller names."""
    callee_to_callers: dict[str, list[str]] = {}
    for entry in call_graph.entries:
        for callee in entry.callees:
            key = f"{callee.file}::{callee.function}"
            callee_to_callers.setdefault(key, []).append(entry.caller.function)
    return callee_to_callers


def _build_callee_map(call_graph: CallGraph) -> dict[str, list[str]]:
    """Return a mapping of caller function key → list of callee names."""
    caller_to_callees: dict[str, list[str]] = {}
    for entry in call_graph.entries:
        key = f"{entry.caller.file}::{entry.caller.function}"
        caller_to_callees[key] = [c.function for c in entry.callees]
    return caller_to_callees


def _make_function_id(node: ASTNode) -> str:
    """Build a unique identifier for an AST node."""
    # Use just the filename portion for readability
    file_key = node.file_path
    if node.node_type == NodeType.METHOD:
        # Try to infer class name from the node name (e.g. "MyClass.my_method")
        if "." in node.name:
            class_part, func_part = node.name.rsplit(".", 1)
            return f"{file_key}::{class_part}::{func_part}"
    return f"{file_key}::{node.name}"


# ---------------------------------------------------------------------------
# LLM provider factory
# ---------------------------------------------------------------------------

def _build_provider(provider_name: str, cfg: dict[str, Any]) -> BaseLLMProvider:
    """Instantiate an LLM provider from *provider_name* and *cfg*."""
    if provider_name == "ollama":
        from llm.ollama_provider import OllamaProvider  # noqa: PLC0415

        ollama_cfg = cfg.get("ollama", {})
        return OllamaProvider(
            model_name=ollama_cfg.get("model", "qwen2.5-coder:7b"),
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            timeout=ollama_cfg.get("timeout", 60),
        )
    if provider_name == "huggingface":
        from llm.huggingface_provider import HuggingFaceProvider  # noqa: PLC0415

        hf_cfg = cfg.get("huggingface", {})
        return HuggingFaceProvider(
            model_name=hf_cfg.get("model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            max_new_tokens=hf_cfg.get("max_new_tokens", 512),
            device=hf_cfg.get("device", "auto"),
        )
    raise ValueError(f"Unknown LLM provider: {provider_name!r}")


# ---------------------------------------------------------------------------
# Per-node generation
# ---------------------------------------------------------------------------

def _generate_for_node(
    node: ASTNode,
    source_lines: list[str],
    router: FallbackRouter,
    caller_map: dict[str, list[str]],
    callee_map: dict[str, list[str]],
    max_retries: int,
) -> DocstringEntry | DocstringFailure:
    """Generate a docstring for a single AST node.

    Never raises; returns a :class:`DocstringFailure` on unrecoverable errors.
    """
    function_id = _make_function_id(node)
    # Determine class_name for method nodes
    class_name: str | None = None
    func_name = node.name
    if node.node_type == NodeType.METHOD and "." in node.name:
        class_name, func_name = node.name.rsplit(".", 1)

    # Look up callers and callees
    node_key = f"{node.file_path}::{node.name}"
    callers = caller_map.get(node_key, [])
    callees = callee_map.get(node_key, [])

    prompt = _build_function_prompt(node, source_lines, callers, callees)

    retries = 0
    last_error: str = ""
    last_error_type: str = "malformed_response"

    while retries <= max_retries:
        t0 = time.monotonic()
        try:
            text, confidence, provider_used, fallback_used = router.generate_with_fallback(
                prompt
            )
            latency_ms = (time.monotonic() - t0) * 1000

            if not text or not text.strip():
                retries += 1
                last_error = "Empty response from LLM"
                last_error_type = "malformed_response"
                logger.debug(
                    "Empty docstring for %s (attempt %d/%d)",
                    function_id,
                    retries,
                    max_retries + 1,
                )
                continue

            cleaned = _clean_docstring(text)

            if not cleaned:
                retries += 1
                last_error = "Empty docstring after cleaning"
                last_error_type = "malformed_response"
                logger.debug(
                    "Empty cleaned docstring for %s (attempt %d/%d)",
                    function_id,
                    retries,
                    max_retries + 1,
                )
                continue

            # Take the minimum of the router confidence and the heuristic
            # score so that a structurally poor response is always penalised
            # even when the router reported high confidence.
            effective_confidence = min(confidence, _confidence_heuristic(cleaned))

            if effective_confidence < _LOW_CONFIDENCE_THRESHOLD and retries < max_retries:
                retries += 1
                last_error = f"Low confidence {effective_confidence:.2f} < {_LOW_CONFIDENCE_THRESHOLD}"
                last_error_type = "low_confidence"
                logger.debug(
                    "Low confidence for %s (attempt %d/%d), retrying",
                    function_id,
                    retries,
                    max_retries + 1,
                )
                continue

            return DocstringEntry(
                function_id=function_id,
                file_path=node.file_path,
                function_name=func_name,
                class_name=class_name,
                node_type=node.node_type,
                docstring=cleaned,
                confidence=effective_confidence,
                provider_used=provider_used,
                fallback_used=fallback_used,
                retries=retries,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(text) // 4,
                latency_ms=latency_ms,
            )

        except TimeoutError as exc:
            retries += 1
            last_error = str(exc)
            last_error_type = "timeout"
            logger.warning("Timeout for %s (attempt %d): %s", function_id, retries, exc)
        except Exception as exc:
            retries += 1
            last_error = str(exc)
            last_error_type = "parse_error"
            logger.warning(
                "Error generating docstring for %s (attempt %d): %s",
                function_id,
                retries,
                exc,
            )

    return DocstringFailure(
        function_id=function_id,
        file_path=node.file_path,
        function_name=func_name,
        reason=last_error,
        provider=type(router.primary).__name__,
        error_type=last_error_type,
    )


def _generate_module_docstring(
    file_path: str,
    source: str,
    router: FallbackRouter,
) -> ModuleDocstring | None:
    """Generate a module-level docstring.  Returns None on failure."""
    prompt = _build_module_prompt(file_path, source)
    try:
        text, confidence, provider_used, _ = router.generate_with_fallback(prompt)
        if text and text.strip():
            return ModuleDocstring(
                file_path=file_path,
                docstring=_clean_docstring(text),
                confidence=confidence,
                provider_used=provider_used,
            )
    except Exception as exc:
        logger.warning("Failed to generate module docstring for %s: %s", file_path, exc)
    return None


# ---------------------------------------------------------------------------
# Async worker for node-level parallelism
# ---------------------------------------------------------------------------

def _read_source_file(file_path: str) -> tuple[str, list[str]] | None:
    """Read a source file and return ``(source, source_lines)`` or ``None``."""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return source, source.splitlines(keepends=True)
    except OSError as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return None


async def _process_node_async(
    node: ASTNode,
    source_lines: list[str],
    router: FallbackRouter,
    caller_map: dict[str, list[str]],
    callee_map: dict[str, list[str]],
    max_retries: int,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> DocstringEntry | DocstringFailure:
    """Generate a docstring for a single node, respecting the semaphore."""
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            _generate_for_node,
            node,
            source_lines,
            router,
            caller_map,
            callee_map,
            max_retries,
        )


async def _process_module_async(
    file_path: str,
    source: str,
    router: FallbackRouter,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> ModuleDocstring | None:
    """Generate a module-level docstring, respecting the semaphore."""
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor, _generate_module_docstring, file_path, source, router,
        )


# ---------------------------------------------------------------------------
# Phase entry-point
# ---------------------------------------------------------------------------

def run_phase3(*, config: dict[str, Any], artifacts: PhaseArtifacts) -> None:
    """Execute Phase 3: Docstring Generation.

    Reads AST nodes and call graph from *artifacts*, invokes the configured
    LLM provider(s) to generate docstrings, and writes ``docstrings.json`` to
    the artifacts directory.

    Mutates *artifacts* in place by setting ``artifacts.docstrings``.

    Raises
    ------
    ValueError
        If ``artifacts.ast_nodes`` or ``artifacts.call_graph`` is ``None``.
    """
    if artifacts.ast_nodes is None:
        raise ValueError(
            "Phase 3 requires ast_nodes artifact (run Phase 2 first)."
        )
    if artifacts.call_graph is None:
        raise ValueError(
            "Phase 3 requires call_graph artifact (run Phase 2 first)."
        )

    ds_cfg: dict[str, Any] = config.get("docstring", {})

    # The top-level "llm" section takes precedence when present.
    llm_cfg: dict[str, Any] = config.get("llm", {})
    if llm_cfg.get("provider"):
        primary_name: str = llm_cfg["provider"]
        # Merge llm sub-sections into ds_cfg so _build_provider can find them
        ds_cfg = {**ds_cfg, **llm_cfg}
        fallback_llm = llm_cfg.get("fallback", {})
        fallback_name: str | None = (
            fallback_llm.get("provider") if fallback_llm.get("enabled") else None
        )
        confidence_threshold: float = float(
            fallback_llm.get("confidence_threshold", ds_cfg.get("confidence_threshold", 0.7))
        )
    else:
        primary_name = ds_cfg.get("primary_provider", "ollama")
        fallback_name = ds_cfg.get("fallback_provider")
        confidence_threshold = float(ds_cfg.get("confidence_threshold", 0.7))

    max_retries: int = int(ds_cfg.get("max_retries", 2))
    concurrency: int = int(ds_cfg.get("concurrency", 4))

    # Build providers
    primary = _build_provider(primary_name, ds_cfg)
    fallback_provider: BaseLLMProvider | None = None
    if fallback_name and fallback_name != primary_name:
        try:
            fallback_provider = _build_provider(fallback_name, ds_cfg)
        except Exception as exc:
            logger.warning("Could not init fallback provider %r: %s", fallback_name, exc)

    router = FallbackRouter(
        primary=primary,
        fallback=fallback_provider,
        confidence_threshold=confidence_threshold,
    )

    # Build call-graph lookup tables
    call_graph = artifacts.call_graph
    caller_map = _build_caller_map(call_graph)
    callee_map = _build_callee_map(call_graph)

    # Group nodes by file
    nodes_by_file: dict[str, list[ASTNode]] = {}
    for node in artifacts.ast_nodes.nodes:
        nodes_by_file.setdefault(node.file_path, []).append(node)

    total_functions = len(artifacts.ast_nodes.nodes)
    logger.info(
        "Phase 3: generating docstrings for %d nodes across %d files",
        total_functions,
        len(nodes_by_file),
    )

    # Run async processing
    all_entries: list[DocstringEntry] = []
    all_failures: list[DocstringFailure] = []
    all_module_docs: list[ModuleDocstring] = []

    async def _run_all() -> None:
        semaphore = asyncio.Semaphore(concurrency)
        executor = ThreadPoolExecutor(max_workers=concurrency)

        # Pre-read source files (I/O) and build per-node tasks
        node_tasks: list[asyncio.Task[DocstringEntry | DocstringFailure]] = []
        module_tasks: list[asyncio.Task[ModuleDocstring | None]] = []

        for file_path, nodes in nodes_by_file.items():
            result = _read_source_file(file_path)
            if result is None:
                # File unreadable — record failures for every node
                for node in nodes:
                    all_failures.append(
                        DocstringFailure(
                            function_id=_make_function_id(node),
                            file_path=node.file_path,
                            function_name=node.name,
                            reason=f"Cannot read {file_path}",
                            provider=type(router.primary).__name__,
                            error_type="file_read_error",
                        )
                    )
                continue

            source, source_lines = result

            for node in nodes:
                node_tasks.append(
                    asyncio.create_task(
                        _process_node_async(
                            node,
                            source_lines,
                            router,
                            caller_map,
                            callee_map,
                            max_retries,
                            semaphore,
                            executor,
                        )
                    )
                )

            module_tasks.append(
                asyncio.create_task(
                    _process_module_async(
                        file_path, source, router, semaphore, executor,
                    )
                )
            )

        # Await all node tasks
        node_results = await asyncio.gather(*node_tasks, return_exceptions=False)
        for res in node_results:
            if isinstance(res, DocstringEntry):
                all_entries.append(res)
            else:
                all_failures.append(res)

        # Await all module tasks
        mod_results = await asyncio.gather(*module_tasks, return_exceptions=False)
        for mod_doc in mod_results:
            if mod_doc is not None:
                all_module_docs.append(mod_doc)

        executor.shutdown(wait=False)

    # In Jupyter / Colab an event loop is already running, so plain
    # ``asyncio.run()`` raises RuntimeError.  Use *nest_asyncio* to patch
    # the running loop, then run the coroutine inside it.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import nest_asyncio  # noqa: PLC0415

        nest_asyncio.apply()
        loop.run_until_complete(_run_all())
    else:
        asyncio.run(_run_all())

    # Compute summary stats
    successful = len(all_entries)
    failed = len(all_failures)
    fallback_count = sum(1 for e in all_entries if e.fallback_used)
    avg_confidence = (
        sum(e.confidence for e in all_entries) / successful if successful else 0.0
    )

    docstrings = Docstrings(
        entries=all_entries,
        module_docstrings=all_module_docs,
        failures=all_failures,
        total_functions=total_functions,
        successful=successful,
        failed=failed,
        average_confidence=round(avg_confidence, 4),
        fallback_count=fallback_count,
    )
    artifacts.docstrings = docstrings

    # Persist artifact
    artifacts_dir = Path(
        config.get("output", {}).get("artifacts_dir", "output/artifacts")
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "docstrings.json"
    out_path.write_text(docstrings.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "Phase 3 complete: %d successful, %d failed, avg_confidence=%.2f, "
        "fallback_used=%d times.  Written to %s",
        successful,
        failed,
        avg_confidence,
        fallback_count,
        out_path,
    )
