"""main.py — Orchestrator for the Telecom Auto-Documentation System.

Designed for use from a Python notebook or script via :func:`run_pipeline`.

Example (notebook cell)::

    from main import run_pipeline
    artifacts = run_pipeline(target="/path/to/codebase")

The module also retains a thin CLI entry-point for convenience::

    python main.py --target /path/to/codebase --phases 1,2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from models.schemas import PhaseArtifacts
from phases.phase1_discovery import run_phase1
from phases.phase2_static_analysis import run_phase2
from phases.phase3_docstrings import run_phase3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------

_PHASE_FN = {
    "1": run_phase1,
    "2": run_phase2,
    "3": run_phase3,
}

_PHASE_ARTIFACT_KEYS = {
    "1": "file_inventory",
    "2": "ast_nodes",  # proxy: if ast_nodes exists, phase 2 ran
    "3": "docstrings",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML config file and return as dict.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  Defaults to ``config.yaml``
        in the current directory.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found at %s; using empty config.", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_existing_artifacts(
    artifacts_dir: Path,
    phases: list[str],
) -> PhaseArtifacts:
    """Load any existing JSON artifacts from a previous run."""
    from models.schemas import (
        ASTNodes,
        CallGraph,
        ComponentMap,
        DataFlow,
        DependencyTree,
        Docstrings,
        FileInventory,
    )

    pa = PhaseArtifacts()
    name_map = {
        "file_inventory": ("file_inventory.json", FileInventory),
        "ast_nodes": ("ast_nodes.json", ASTNodes),
        "call_graph": ("call_graph.json", CallGraph),
        "dependency_tree": ("dependency_tree.json", DependencyTree),
        "data_flow": ("data_flow.json", DataFlow),
        "component_map": ("component_map.json", ComponentMap),
        "docstrings": ("docstrings.json", Docstrings),
    }
    for attr, (filename, model_cls) in name_map.items():
        fp = artifacts_dir / filename
        if fp.exists():
            try:
                pa.__setattr__(attr, model_cls.model_validate_json(fp.read_text()))
                logger.info("Loaded existing artifact: %s", filename)
            except Exception as exc:
                logger.warning("Could not load %s: %s", filename, exc)
    return pa


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
        force=True,
    )


# ---------------------------------------------------------------------------
# Public notebook/script API
# ---------------------------------------------------------------------------

def run_pipeline(
    target: str,
    *,
    phases: str | list[str] = "1,2",
    config_path: str = "config.yaml",
    resume: bool = False,
    artifacts_dir: str = "",
    log_level: str = "INFO",
) -> PhaseArtifacts:
    """Run the analysis pipeline and return the collected artifacts.

    This is the primary entry-point for notebook and script usage.

    Parameters
    ----------
    target:
        Absolute or relative path to the Python codebase to analyse.
    phases:
        Phases to execute.  Either a comma-separated string (``"1,2"``)
        or a list of strings (``["1", "2"]``).
    config_path:
        Path to the YAML configuration file.
    resume:
        When ``True``, skip phases whose output artifacts already exist
        on disk.
    artifacts_dir:
        Override the output directory for JSON artifacts.  If empty, the
        value from ``config.yaml`` (or ``output/artifacts``) is used.
    log_level:
        Python logging level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, …).

    Returns
    -------
    PhaseArtifacts
        A Pydantic model containing every artifact produced by the
        requested phases (``file_inventory``, ``ast_nodes``,
        ``call_graph``, ``dependency_tree``, ``data_flow``,
        ``component_map``).

    Raises
    ------
    ValueError
        If an unknown phase is requested or the target path is empty.
    RuntimeError
        If a phase fails during execution.
    """
    _setup_logging(log_level)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config = load_config(config_path)

    # Apply overrides
    if target:
        config.setdefault("target", {})["path"] = target
    if artifacts_dir:
        config.setdefault("output", {})["artifacts_dir"] = artifacts_dir

    arts_dir = Path(
        config.get("output", {}).get("artifacts_dir", "output/artifacts")
    )

    # ------------------------------------------------------------------
    # Parse phases
    # ------------------------------------------------------------------
    if isinstance(phases, str):
        requested_phases = [p.strip() for p in phases.split(",") if p.strip()]
    else:
        requested_phases = list(phases)

    unknown = [p for p in requested_phases if p not in _PHASE_FN]
    if unknown:
        raise ValueError(
            f"Unknown phase(s): {unknown}.  Available: {list(_PHASE_FN.keys())}"
        )

    # ------------------------------------------------------------------
    # Load or initialise artifacts
    # ------------------------------------------------------------------
    if resume:
        artifacts = _load_existing_artifacts(arts_dir, requested_phases)
    else:
        artifacts = PhaseArtifacts()

    # ------------------------------------------------------------------
    # Run phases
    # ------------------------------------------------------------------
    for phase in requested_phases:
        if resume:
            proxy_key = _PHASE_ARTIFACT_KEYS.get(phase)
            if proxy_key and getattr(artifacts, proxy_key, None) is not None:
                logger.info("Phase %s: skipping (artifact already exists)", phase)
                continue

        logger.info("=" * 60)
        logger.info("Running Phase %s", phase)
        logger.info("=" * 60)
        _PHASE_FN[phase](config=config, artifacts=artifacts)

    logger.info("All requested phases completed successfully.")
    return artifacts


# ---------------------------------------------------------------------------
# Thin CLI wrapper (kept for convenience)
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Telecom Auto-Documentation System",
    )
    parser.add_argument(
        "--target",
        default="",
        help="Path to the Python codebase to analyse.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--phases",
        default="1,2",
        help="Comma-separated phases to run, e.g. '1,2' (default: 1,2).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases whose output artifacts already exist.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help="Override the artifacts output directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point.  Prefer :func:`run_pipeline` from notebooks."""
    args = _parse_args(argv)
    try:
        run_pipeline(
            target=args.target,
            phases=args.phases,
            config_path=args.config,
            resume=args.resume,
            artifacts_dir=args.artifacts_dir,
            log_level=args.log_level,
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
