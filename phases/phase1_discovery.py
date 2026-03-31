"""Phase 1: File Discovery & Metadata.

Recursively walks the target directory, collects metadata for every Python
file (size, line count, last-modified timestamp, encoding), and writes the
result to ``output/artifacts/file_inventory.json``.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chardet

from models.schemas import FileInventory, FileMetadata, PhaseArtifacts

logger = logging.getLogger(__name__)

# Directories to skip unconditionally
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".tox",
        ".eggs",
        "dist",
        "build",
    }
)


def _should_skip_dir(dir_name: str) -> bool:
    """Return True if a directory with *dir_name* should be skipped."""
    if dir_name in _SKIP_DIRS:
        return True
    if dir_name.endswith(".egg-info"):
        return True
    return False


def _detect_encoding(raw: bytes) -> str:
    """Detect the encoding of raw bytes.  Falls back to ``utf-8`` on failure."""
    try:
        result = chardet.detect(raw)
        encoding = result.get("encoding") or "utf-8"
        return encoding
    except Exception as exc:
        logger.warning("Encoding detection failed: %s", exc)
        return "utf-8"


def _count_lines(raw: bytes, encoding: str) -> int:
    """Count newlines in a byte string decoded with *encoding*."""
    try:
        text = raw.decode(encoding, errors="replace")
        return text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    except Exception:
        return raw.count(b"\n")


def _collect_files(
    root: Path,
    config: dict[str, Any],
) -> tuple[list[FileMetadata], int]:
    """Walk *root* and return ``(file_metadata_list, skipped_dir_count)``."""
    analysis_cfg = config.get("analysis", {})
    max_size: int = analysis_cfg.get("max_file_size_bytes", 5_242_880)
    detect_enc: bool = analysis_cfg.get("detect_encodings", True)

    metadata_list: list[FileMetadata] = []
    skipped_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune skip-dirs in-place so os.walk won't descend into them
        before = len(dirnames)
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        skipped_dirs += before - len(dirnames)

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            file_path = Path(dirpath) / filename

            try:
                stat = file_path.stat()
            except PermissionError:
                logger.warning("Permission denied: %s — skipping", file_path)
                continue
            except OSError as exc:
                logger.warning("Cannot stat %s: %s — skipping", file_path, exc)
                continue

            size_bytes = stat.st_size
            if size_bytes > max_size:
                logger.warning(
                    "File %s exceeds max size (%d bytes) — skipping",
                    file_path,
                    max_size,
                )
                continue

            try:
                raw = file_path.read_bytes()
            except PermissionError:
                logger.warning("Permission denied reading %s — skipping", file_path)
                continue
            except OSError as exc:
                logger.warning("Cannot read %s: %s — skipping", file_path, exc)
                continue

            encoding = _detect_encoding(raw) if detect_enc else "utf-8"
            line_count = _count_lines(raw, encoding)
            last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            metadata_list.append(
                FileMetadata(
                    path=str(file_path),
                    language="python",
                    size_bytes=size_bytes,
                    line_count=line_count,
                    last_modified=last_modified,
                    encoding=encoding,
                )
            )

    return metadata_list, skipped_dirs


def run_phase1(*, config: dict[str, Any], artifacts: PhaseArtifacts) -> None:
    """Execute Phase 1: File Discovery & Metadata.

    Mutates *artifacts* in place by setting ``artifacts.file_inventory``.
    Also writes ``file_inventory.json`` to the configured artifacts directory.

    Raises
    ------
    ValueError
        If no Python files are found in the target directory.
    """
    target_path = config.get("target", {}).get("path", "")
    if not target_path:
        raise ValueError("No target path configured.  Set 'target.path' in config.yaml or pass --target.")

    root = Path(target_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Target directory does not exist: {root}")

    logger.info("Phase 1: scanning %s", root)

    files, skipped_dirs = _collect_files(root, config)

    if not files:
        raise ValueError(
            f"No Python files found in '{root}'.  "
            "Check the target path and exclude patterns."
        )

    total_lines = sum(f.line_count for f in files)
    logger.info(
        "Phase 1 complete: %d files found, %d total lines, %d directories skipped",
        len(files),
        total_lines,
        skipped_dirs,
    )

    inventory = FileInventory(
        files=files,
        project_root=str(root),
        scan_timestamp=datetime.now(tz=timezone.utc),
        total_files=len(files),
    )
    artifacts.file_inventory = inventory

    # Persist artifact
    artifacts_dir = Path(config.get("output", {}).get("artifacts_dir", "output/artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "file_inventory.json"
    out_path.write_text(
        inventory.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("file_inventory.json written to %s", out_path)
