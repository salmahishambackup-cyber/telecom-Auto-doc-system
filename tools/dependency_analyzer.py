"""Dependency analyser.

Separates internal imports (within the project) from external package imports,
resolves package versions from common lock/config files, maps which files use
which external packages, and detects circular internal imports.
"""
from __future__ import annotations

import ast
import configparser
import logging
import re
import tomllib
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from models.schemas import DependencyTree, ExternalPackage, InternalImport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version resolution helpers
# ---------------------------------------------------------------------------

_REQ_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9_\-\.]+)\s*([><=!~^][^\s;#]*)?"
)


def _parse_requirements_txt(path: Path) -> dict[str, str]:
    """Return {package_name_lower: version_spec} from a requirements.txt file."""
    versions: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            m = _REQ_LINE_RE.match(line)
            if m:
                name = m.group(1).lower().replace("-", "_")
                spec = (m.group(2) or "").strip()
                versions[name] = spec
    except OSError:
        pass
    return versions


def _parse_pyproject_toml(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
        deps: list[str] = []
        # PEP 517 / poetry
        project = data.get("project", {})
        deps.extend(project.get("dependencies", []))
        poetry_deps: dict = (
            data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        )
        for name, spec in poetry_deps.items():
            if name.lower() == "python":
                continue
            if isinstance(spec, str):
                deps.append(f"{name}{spec}")
            elif isinstance(spec, dict):
                deps.append(f"{name}{spec.get('version', '')}")
        for dep in deps:
            m = _REQ_LINE_RE.match(dep)
            if m:
                name = m.group(1).lower().replace("-", "_")
                spec = (m.group(2) or "").strip()
                versions[name] = spec
    except Exception:
        pass
    return versions


def _parse_pipfile(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        cp = configparser.ConfigParser()
        cp.read_string(path.read_text(encoding="utf-8", errors="replace"))
        for section in ("packages", "dev-packages"):
            if cp.has_section(section):
                for name, spec in cp.items(section):
                    key = name.lower().replace("-", "_")
                    versions[key] = spec.strip('"').strip("'")
    except Exception:
        pass
    return versions


def _collect_version_map(project_root: Path) -> dict[str, str]:
    """Aggregate version info from all known dependency files."""
    version_map: dict[str, str] = {}
    for name in ("requirements.txt", "requirements-dev.txt", "requirements_dev.txt"):
        f = project_root / name
        if f.exists():
            version_map.update(_parse_requirements_txt(f))
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        version_map.update(_parse_pyproject_toml(pyproject))
    pipfile = project_root / "Pipfile"
    if pipfile.exists():
        version_map.update(_parse_pipfile(pipfile))
    return version_map


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

def _module_to_path(module: str, project_root: Path, source_file: Path) -> Optional[Path]:
    """Try to resolve a dotted *module* name to a .py file within the project."""
    parts = module.split(".")
    # Absolute import
    candidate = project_root.joinpath(*parts)
    if (candidate.with_suffix(".py")).exists():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").exists():
        return candidate / "__init__.py"
    # Relative from source file directory
    candidate2 = source_file.parent.joinpath(*parts)
    if (candidate2.with_suffix(".py")).exists():
        return candidate2.with_suffix(".py")
    if (candidate2 / "__init__.py").exists():
        return candidate2 / "__init__.py"
    return None


def _is_stdlib(module: str) -> bool:
    """Heuristic: check if a module is part of the standard library."""
    import sys

    top = module.split(".")[0]
    # sys.stdlib_module_names is available from Python 3.10+
    stdlib_names: frozenset[str] = getattr(sys, "stdlib_module_names", frozenset())
    if stdlib_names:
        return top in stdlib_names
    # Fallback: try importing and check the origin
    import importlib.util

    try:
        spec = importlib.util.find_spec(top)
        if spec is None:
            return False
        origin = getattr(spec, "origin", None) or ""
        return "site-packages" not in str(origin)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Circular import detection
# ---------------------------------------------------------------------------

def _detect_circular_imports(
    internal_graph: dict[str, set[str]],
) -> list[list[str]]:
    """Return a list of cycles found in the internal import graph using DFS."""
    visited: set[str] = set()
    rec_stack: set[str] = set()
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in internal_graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
            elif neighbor in rec_stack:
                # Found a cycle — slice the path to show the cycle
                cycle_start = path.index(neighbor) if neighbor in path else 0
                cycles.append(path[cycle_start:] + [neighbor])
        rec_stack.discard(node)

    for node in list(internal_graph.keys()):
        if node not in visited:
            dfs(node, [node])

    return cycles


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

def analyse_dependencies(
    files: list[str | Path],
    project_root: str | Path,
) -> DependencyTree:
    """Analyse imports across all *files* and return a :class:`DependencyTree`.

    Parameters
    ----------
    files:
        Python source files to analyse.
    project_root:
        Root directory of the project.
    """
    root = Path(project_root).resolve()
    version_map = _collect_version_map(root)

    internal_imports: list[InternalImport] = []
    # external: package_name → set of files that import it
    ext_usage: dict[str, set[str]] = defaultdict(set)
    # internal graph: file_path → set of file_paths it imports
    internal_graph: dict[str, set[str]] = defaultdict(set)

    for fp in files:
        source_path = Path(fp).resolve()
        try:
            source = source_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(source_path))
        except (OSError, SyntaxError) as exc:
            logger.warning("Skipping %s for dependency analysis: %s", source_path, exc)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    _process_import(
                        module=module,
                        names=[module],
                        source_file=source_path,
                        root=root,
                        internal_imports=internal_imports,
                        ext_usage=ext_usage,
                        internal_graph=internal_graph,
                        version_map=version_map,
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                level = node.level or 0
                if level > 0:
                    # Relative import → always internal
                    _record_internal(
                        source_file=source_path,
                        module=("." * level) + module,
                        names=names,
                        root=root,
                        internal_imports=internal_imports,
                        internal_graph=internal_graph,
                    )
                else:
                    _process_import(
                        module=module,
                        names=names,
                        source_file=source_path,
                        root=root,
                        internal_imports=internal_imports,
                        ext_usage=ext_usage,
                        internal_graph=internal_graph,
                        version_map=version_map,
                    )

    # Build ExternalPackage list
    external_packages: list[ExternalPackage] = []
    for pkg_name, using_files in sorted(ext_usage.items()):
        norm = pkg_name.lower().replace("-", "_")
        version = version_map.get(norm) or version_map.get(pkg_name.lower())
        external_packages.append(
            ExternalPackage(
                name=pkg_name,
                version=version,
                used_by_files=sorted(using_files),
            )
        )

    # Detect circular imports
    cycles = _detect_circular_imports(internal_graph)
    if cycles:
        for cycle in cycles:
            logger.warning("Circular import detected: %s", " → ".join(cycle))

    logger.info(
        "Dependency analysis: %d internal imports, %d external packages",
        len(internal_imports),
        len(external_packages),
    )
    return DependencyTree(
        internal_imports=internal_imports,
        external_packages=external_packages,
    )


def _process_import(
    *,
    module: str,
    names: list[str],
    source_file: Path,
    root: Path,
    internal_imports: list[InternalImport],
    ext_usage: dict[str, set[str]],
    internal_graph: dict[str, set[str]],
    version_map: dict[str, str],
) -> None:
    top_level = module.split(".")[0] if module else ""
    resolved = _module_to_path(module, root, source_file) if module else None
    if resolved is not None:
        _record_internal(
            source_file=source_file,
            module=module,
            names=names,
            root=root,
            internal_imports=internal_imports,
            internal_graph=internal_graph,
            resolved_path=resolved,
        )
    elif top_level and not _is_stdlib(top_level):
        ext_usage[top_level].add(str(source_file))
    # stdlib imports are silently ignored


def _record_internal(
    *,
    source_file: Path,
    module: str,
    names: list[str],
    root: Path,
    internal_imports: list[InternalImport],
    internal_graph: dict[str, set[str]],
    resolved_path: Optional[Path] = None,
) -> None:
    internal_imports.append(
        InternalImport(
            source_file=str(source_file),
            imported_module=module,
            imported_names=names,
        )
    )
    target = str(resolved_path) if resolved_path else module
    internal_graph[str(source_file)].add(target)
