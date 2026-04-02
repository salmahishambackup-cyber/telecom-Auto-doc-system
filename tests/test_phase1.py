"""Tests for Phase 1: File Discovery & Metadata."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from models.schemas import PhaseArtifacts
from phases.phase1_discovery import run_phase1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(target_path: str, artifacts_dir: str) -> dict:
    return {
        "target": {"path": target_path},
        "analysis": {"max_file_size_bytes": 5_242_880, "detect_encodings": True},
        "output": {"artifacts_dir": artifacts_dir},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    def test_all_py_files_discovered(self, tmp_path: Path) -> None:
        """All .py files in the target directory must appear in the inventory."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("z = 3\n")

        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)

        assert artifacts.file_inventory is not None
        paths = {fm.path for fm in artifacts.file_inventory.files}
        assert str(tmp_path / "a.py") in paths
        assert str(tmp_path / "b.py") in paths
        assert str(sub / "c.py") in paths

    def test_total_files_count(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}\n")
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)
        assert artifacts.file_inventory is not None
        assert artifacts.file_inventory.total_files == 5

    def test_pycache_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("pass\n")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.py").write_text("cached = True\n")

        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)

        paths = {fm.path for fm in artifacts.file_inventory.files}
        assert not any("__pycache__" in p for p in paths)
        assert str(tmp_path / "real.py") in paths

    def test_venv_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("pass\n")
        venv = tmp_path / "venv"
        venv.mkdir()
        lib = venv / "lib"
        lib.mkdir()
        pkg_file = lib / "pkg.py"
        pkg_file.write_text("pass\n")

        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)

        paths = {fm.path for fm in artifacts.file_inventory.files}
        # The file inside venv/ must not be discovered
        assert str(pkg_file) not in paths
        # The file outside venv/ must be discovered
        assert str(tmp_path / "app.py") in paths

    def test_line_count_is_accurate(self, tmp_path: Path) -> None:
        content = "line1\nline2\nline3\n"  # 3 lines
        (tmp_path / "three.py").write_text(content)
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)

        inv = artifacts.file_inventory
        assert inv is not None
        fm = next(f for f in inv.files if f.path.endswith("three.py"))
        assert fm.line_count == 3

    def test_size_bytes_is_accurate(self, tmp_path: Path) -> None:
        content = "hello\n"
        p = tmp_path / "hello.py"
        p.write_bytes(content.encode("utf-8"))
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)

        inv = artifacts.file_inventory
        assert inv is not None
        fm = next(f for f in inv.files if f.path.endswith("hello.py"))
        assert fm.size_bytes == len(content.encode("utf-8"))

    def test_encoding_detection(self, tmp_path: Path) -> None:
        """Encoding field must be populated (not empty)."""
        (tmp_path / "enc.py").write_text("x = 1\n", encoding="utf-8")
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)

        inv = artifacts.file_inventory
        assert inv is not None
        fm = next(f for f in inv.files if f.path.endswith("enc.py"))
        assert fm.encoding  # non-empty string

    def test_last_modified_is_set(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("pass\n")
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)

        inv = artifacts.file_inventory
        assert inv is not None
        fm = next(f for f in inv.files if f.path.endswith("mod.py"))
        assert fm.last_modified is not None

    def test_language_is_python(self, tmp_path: Path) -> None:
        (tmp_path / "lang.py").write_text("pass\n")
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)
        inv = artifacts.file_inventory
        assert inv is not None
        fm = next(f for f in inv.files if f.path.endswith("lang.py"))
        assert fm.language == "python"

    def test_empty_codebase_raises_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        artifacts = PhaseArtifacts()
        with pytest.raises((ValueError, FileNotFoundError)):
            run_phase1(
                config=_make_config(str(empty), str(tmp_path / "artifacts")),
                artifacts=artifacts,
            )

    def test_artifact_json_written(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("pass\n")
        artifacts_dir = tmp_path / "artifacts"
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(artifacts_dir)), artifacts=artifacts)
        assert (artifacts_dir / "file_inventory.json").exists()

    def test_non_python_files_excluded(self, tmp_path: Path) -> None:
        (tmp_path / "script.py").write_text("pass\n")
        (tmp_path / "notes.txt").write_text("hello\n")
        (tmp_path / "data.json").write_text("{}\n")
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)
        inv = artifacts.file_inventory
        assert inv is not None
        assert all(f.path.endswith(".py") for f in inv.files)

    def test_project_root_is_set(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("pass\n")
        artifacts = PhaseArtifacts()
        run_phase1(config=_make_config(str(tmp_path), str(tmp_path / "artifacts")), artifacts=artifacts)
        assert artifacts.file_inventory is not None
        assert artifacts.file_inventory.project_root == str(tmp_path)
