"""main.py — CLI orchestrator for the Telecom Auto-Documentation System.

Usage
-----
    python main.py --target /path/to/codebase [--phases 1,2] [--resume]

Flags
-----
--target PATH       Path to the Python codebase to analyse.
--config PATH       Path to config.yaml (default: config.yaml in the same dir).
--phases LIST       Comma-separated list of phases to run (default: 1,2).
--resume            Skip phases whose output artifacts already exist.
--artifacts-dir DIR Override output/artifacts directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from models.schemas import PhaseArtifacts
from phases.phase1_discovery import run_phase1
from phases.phase2_static_analysis import run_phase2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------

_PHASE_FN = {
    "1": run_phase1,
    "2": run_phase2,
}

_PHASE_ARTIFACT_KEYS = {
    "1": "file_inventory",
    "2": "ast_nodes",  # proxy: if ast_nodes exists, phase 2 ran
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config file and return as dict."""
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
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Telecom Auto-Documentation System CLI",
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
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config = _load_config(args.config)

    # Apply CLI overrides
    if args.target:
        config.setdefault("target", {})["path"] = args.target
    if args.artifacts_dir:
        config.setdefault("output", {})["artifacts_dir"] = args.artifacts_dir

    artifacts_dir = Path(
        config.get("output", {}).get("artifacts_dir", "output/artifacts")
    )

    # ------------------------------------------------------------------
    # Parse phases
    # ------------------------------------------------------------------
    requested_phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in requested_phases if p not in _PHASE_FN]
    if unknown:
        logger.error("Unknown phase(s): %s.  Available: %s", unknown, list(_PHASE_FN.keys()))
        return 1

    # ------------------------------------------------------------------
    # Load or initialise artifacts
    # ------------------------------------------------------------------
    if args.resume:
        artifacts = _load_existing_artifacts(artifacts_dir, requested_phases)
    else:
        artifacts = PhaseArtifacts()

    # ------------------------------------------------------------------
    # Run phases
    # ------------------------------------------------------------------
    for phase in requested_phases:
        if args.resume:
            # Check if we can skip this phase
            proxy_key = _PHASE_ARTIFACT_KEYS.get(phase)
            if proxy_key and getattr(artifacts, proxy_key, None) is not None:
                logger.info("Phase %s: skipping (artifact already exists)", phase)
                continue

        logger.info("=" * 60)
        logger.info("Running Phase %s", phase)
        logger.info("=" * 60)
        try:
            _PHASE_FN[phase](config=config, artifacts=artifacts)
        except Exception as exc:
            logger.exception("Phase %s failed: %s", phase, exc)
            return 1

    logger.info("All requested phases completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
