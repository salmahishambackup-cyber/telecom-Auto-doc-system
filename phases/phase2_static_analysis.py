"""Phase 2: Deep Static Analysis.

Orchestrates the four static-analysis tools:
  - AST extraction        → ast_nodes.json
  - Call-graph building   → call_graph.json
  - Dependency analysis   → dependency_tree.json
  - Data-flow detection   → data_flow.json

Additionally generates component_map.json by grouping files by top-level
directory and inferring entry points / component purposes.

Validates results after all tools have run.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from models.schemas import (
    ASTNodes,
    CallGraph,
    ComponentEntry,
    ComponentMap,
    DataFlow,
    DependencyTree,
    FlowType,
    PhaseArtifacts,
)
from tools.ast_extractor import extract_nodes
from tools.call_graph import build_call_graph
from tools.data_flow_detector import DEFAULT_PATTERNS, detect_data_flows, _compile_patterns
from tools.dependency_analyzer import analyse_dependencies

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry-point heuristics
# ---------------------------------------------------------------------------

_ENTRY_HINTS = (
    "if __name__",
    "app = FastAPI",
    "app = Flask",
    "application = Flask",
    "application = FastAPI",
    "cli =",
    "@click.command",
    "ArgumentParser",
)


def _is_entry_point(source: str) -> bool:
    return any(hint in source for hint in _ENTRY_HINTS)


# ---------------------------------------------------------------------------
# Component map builder
# ---------------------------------------------------------------------------

def _build_component_map(
    files: list[str],
    project_root: str,
) -> ComponentMap:
    """Group files by top-level directory relative to *project_root*."""
    root = Path(project_root)
    # component_name → ComponentEntry (use dict so we can accumulate)
    components: dict[str, ComponentEntry] = {}

    for fp in files:
        path = Path(fp)
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path

        parts = relative.parts
        if len(parts) == 0:
            continue

        # Files directly in root get component name "<root>"
        component_name = parts[0] if len(parts) > 1 else "<root>"
        # Directory is the component dir
        component_dir = str(root / component_name) if component_name != "<root>" else str(root)

        if component_name not in components:
            components[component_name] = ComponentEntry(
                component_name=component_name,
                directory=component_dir,
                files=[],
                purpose_hint=_infer_purpose(component_name),
                entry_points=[],
                dependencies=[],
            )

        components[component_name].files.append(fp)

        # Check for entry points
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            if _is_entry_point(source):
                components[component_name].entry_points.append(fp)
        except OSError:
            pass

    # Record cross-component dependencies (simplified: use directory names only)
    # A more precise version would use the dependency tree, which is available
    # after Phase 2 runs fully — this is a best-effort heuristic.

    return ComponentMap(components=list(components.values()))


def _infer_purpose(directory_name: str) -> str:
    """Return a human-readable purpose hint based on common directory names."""
    mapping = {
        "models": "Data models and schemas",
        "views": "View layer / request handlers",
        "controllers": "Business logic controllers",
        "services": "Service layer",
        "routes": "URL routing",
        "api": "API endpoints",
        "db": "Database access layer",
        "database": "Database access layer",
        "utils": "Utility functions",
        "helpers": "Helper functions",
        "tests": "Test suite",
        "config": "Configuration",
        "settings": "Application settings",
        "migrations": "Database migrations",
        "tasks": "Background tasks / celery workers",
        "workers": "Background workers",
        "cli": "Command-line interface",
        "scripts": "Utility scripts",
        "phases": "Pipeline phases",
        "tools": "Analysis tools",
        "llm": "LLM provider integrations",
        "output": "Output artifacts",
        "<root>": "Project root files",
    }
    return mapping.get(directory_name.lower(), f"Component: {directory_name}")


# ---------------------------------------------------------------------------
# Cross-component dependency wiring
# ---------------------------------------------------------------------------

def _wire_component_dependencies(
    component_map: ComponentMap,
    dep_tree: DependencyTree,
) -> None:
    """Fill in the ``dependencies`` field of each component based on imports."""
    # Build file → component mapping
    file_to_comp: dict[str, str] = {}
    for comp in component_map.components:
        for fp in comp.files:
            file_to_comp[fp] = comp.component_name

    for comp in component_map.components:
        comp_files = set(comp.files)
        dep_set: set[str] = set()
        for imp in dep_tree.internal_imports:
            if imp.source_file in comp_files:
                # Find which component the imported module belongs to
                for other_comp in component_map.components:
                    if other_comp.component_name == comp.component_name:
                        continue
                    for fp in other_comp.files:
                        if imp.imported_module in fp or fp.endswith(
                            imp.imported_module.replace(".", "/") + ".py"
                        ):
                            dep_set.add(other_comp.component_name)
        comp.dependencies = sorted(dep_set)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_artifacts(
    ast_nodes: ASTNodes,
    call_graph: CallGraph,
    file_inventory_files: list[str],
    processed_files: set[str],
) -> None:
    """Run post-Phase-2 validation checks and log warnings."""
    # Check 1: every function in ast_nodes appears in call_graph as caller or callee
    defined_functions: set[str] = {n.name for n in ast_nodes.nodes}
    caller_functions: set[str] = {e.caller.function for e in call_graph.entries}
    callee_functions: set[str] = {
        c.function for e in call_graph.entries for c in e.callees
    }
    referenced_functions = caller_functions | callee_functions
    orphans = defined_functions - referenced_functions
    if orphans:
        logger.debug(
            "Orphan functions (defined but not referenced in call graph): %s",
            sorted(orphans)[:20],
        )

    # Check 2: every file in inventory was processed (or skipped with reason)
    unprocessed = set(file_inventory_files) - processed_files
    if unprocessed:
        logger.warning(
            "%d file(s) from inventory were not processed by Phase 2: %s",
            len(unprocessed),
            sorted(unprocessed)[:10],
        )

    logger.info("Validation complete. Orphans: %d, Unprocessed: %d", len(orphans), len(unprocessed))


# ---------------------------------------------------------------------------
# Phase 2 orchestrator
# ---------------------------------------------------------------------------

def run_phase2(*, config: dict[str, Any], artifacts: PhaseArtifacts) -> None:
    """Execute Phase 2: Deep Static Analysis.

    Reads ``artifacts.file_inventory`` (populated by Phase 1) and runs all
    four static-analysis tools.  Mutates *artifacts* in place and writes JSON
    files to the configured artifacts directory.

    Raises
    ------
    RuntimeError
        If Phase 1 has not been run (no file inventory available).
    """
    if artifacts.file_inventory is None:
        raise RuntimeError(
            "Phase 2 requires Phase 1 output.  Run Phase 1 first or load "
            "file_inventory.json before invoking Phase 2."
        )

    file_paths: list[str] = [fm.path for fm in artifacts.file_inventory.files]
    project_root = artifacts.file_inventory.project_root

    if not file_paths:
        raise RuntimeError("File inventory is empty — nothing to analyse.")

    logger.info("Phase 2: analysing %d files", len(file_paths))

    artifacts_dir = Path(config.get("output", {}).get("artifacts_dir", "output/artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    processed_files: set[str] = set()

    # ------------------------------------------------------------------
    # AST extraction
    # ------------------------------------------------------------------
    all_nodes = []
    for fp in file_paths:
        nodes = extract_nodes(fp)
        all_nodes.extend(nodes)
        processed_files.add(fp)

    ast_nodes = ASTNodes(nodes=all_nodes)
    artifacts.ast_nodes = ast_nodes
    (artifacts_dir / "ast_nodes.json").write_text(
        ast_nodes.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("ast_nodes.json: %d nodes", len(ast_nodes.nodes))

    # ------------------------------------------------------------------
    # Call graph
    # ------------------------------------------------------------------
    call_graph = build_call_graph(file_paths, project_root)
    artifacts.call_graph = call_graph
    (artifacts_dir / "call_graph.json").write_text(
        call_graph.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("call_graph.json: %d entries", len(call_graph.entries))

    # ------------------------------------------------------------------
    # Dependency analysis
    # ------------------------------------------------------------------
    dep_tree = analyse_dependencies(file_paths, project_root)
    artifacts.dependency_tree = dep_tree
    (artifacts_dir / "dependency_tree.json").write_text(
        dep_tree.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info(
        "dependency_tree.json: %d internal, %d external",
        len(dep_tree.internal_imports),
        len(dep_tree.external_packages),
    )

    # ------------------------------------------------------------------
    # Data flow detection
    # ------------------------------------------------------------------
    # Allow pattern overrides from config
    analysis_cfg = config.get("analysis", {})
    raw_patterns = analysis_cfg.get("data_flow_patterns", {})
    if raw_patterns:
        patterns = {
            FlowType(key): value for key, value in raw_patterns.items()
        }
    else:
        patterns = None  # use defaults

    data_flow = detect_data_flows(file_paths, patterns=patterns)
    artifacts.data_flow = data_flow
    (artifacts_dir / "data_flow.json").write_text(
        data_flow.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("data_flow.json: %d entries", len(data_flow.entries))

    # ------------------------------------------------------------------
    # Component map
    # ------------------------------------------------------------------
    component_map = _build_component_map(file_paths, project_root)
    _wire_component_dependencies(component_map, dep_tree)
    artifacts.component_map = component_map
    (artifacts_dir / "component_map.json").write_text(
        component_map.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("component_map.json: %d components", len(component_map.components))

    # ------------------------------------------------------------------
    # Post-phase validation
    # ------------------------------------------------------------------
    _validate_artifacts(
        ast_nodes=ast_nodes,
        call_graph=call_graph,
        file_inventory_files=file_paths,
        processed_files=processed_files,
    )

    logger.info("Phase 2 complete.")
