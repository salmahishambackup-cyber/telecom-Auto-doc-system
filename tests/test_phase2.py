"""Tests for Phase 2: Deep Static Analysis (and its sub-tools)."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from models.schemas import PhaseArtifacts
from phases.phase1_discovery import run_phase1
from phases.phase2_static_analysis import run_phase2
from tools.ast_extractor import extract_nodes
from tools.call_graph import build_call_graph
from tools.data_flow_detector import detect_data_flows
from tools.dependency_analyzer import analyse_dependencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(target_path: str, artifacts_dir: str) -> dict:
    return {
        "target": {"path": target_path},
        "analysis": {"max_file_size_bytes": 5_242_880, "detect_encodings": True},
        "output": {"artifacts_dir": artifacts_dir},
    }


def _write(path: Path, content: str) -> Path:
    path.write_text(dedent(content), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# AST Extractor tests
# ---------------------------------------------------------------------------

class TestASTExtractor:
    def test_extracts_function(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "funcs.py", """\
            def hello(name: str) -> str:
                return f"Hello, {name}"
        """)
        nodes = extract_nodes(str(p))
        assert any(n.name == "hello" for n in nodes)

    def test_extracts_class(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "classes.py", """\
            class MyClass:
                pass
        """)
        nodes = extract_nodes(str(p))
        assert any(n.name == "MyClass" for n in nodes)

    def test_extracts_method(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "methods.py", """\
            class Greeter:
                def greet(self, name: str) -> str:
                    return f"Hi {name}"
        """)
        nodes = extract_nodes(str(p))
        method_nodes = [n for n in nodes if n.name == "greet"]
        assert method_nodes
        from models.schemas import NodeType
        assert method_nodes[0].node_type == NodeType.METHOD

    def test_captures_docstring(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "docs.py", '''\
            def documented():
                """This is the docstring."""
                pass
        ''')
        nodes = extract_nodes(str(p))
        doc_node = next(n for n in nodes if n.name == "documented")
        assert doc_node.docstring_existing == "This is the docstring."

    def test_captures_return_type(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "typed.py", """\
            def add(a: int, b: int) -> int:
                return a + b
        """)
        nodes = extract_nodes(str(p))
        node = next(n for n in nodes if n.name == "add")
        assert node.return_type == "int"

    def test_captures_decorators(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "decs.py", """\
            class API:
                @staticmethod
                def ping() -> str:
                    return "pong"
        """)
        nodes = extract_nodes(str(p))
        node = next(n for n in nodes if n.name == "ping")
        assert "staticmethod" in node.decorators

    def test_syntax_error_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.py"
        p.write_text("def foo(:\n    pass\n")
        nodes = extract_nodes(str(p))
        assert nodes == []

    def test_args_extracted(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "args.py", """\
            def connect(host: str, port: int = 8080) -> None:
                pass
        """)
        nodes = extract_nodes(str(p))
        node = next(n for n in nodes if n.name == "connect")
        assert "host: str" in node.args or "host" in " ".join(node.args)

    def test_nested_class(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "nested.py", """\
            class Outer:
                class Inner:
                    def inner_method(self):
                        pass
        """)
        nodes = extract_nodes(str(p))
        names = {n.name for n in nodes}
        assert "Outer" in names
        assert "Inner" in names
        assert "inner_method" in names

    def test_line_range_captured(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "lines.py", """\
            def alpha():
                pass


            def beta():
                pass
        """)
        nodes = extract_nodes(str(p))
        alpha = next(n for n in nodes if n.name == "alpha")
        beta = next(n for n in nodes if n.name == "beta")
        assert alpha.line_start < beta.line_start


# ---------------------------------------------------------------------------
# Call Graph tests
# ---------------------------------------------------------------------------

class TestCallGraph:
    def test_intra_file_call(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "calls.py", """\
            def helper():
                pass

            def main():
                helper()
        """)
        cg = build_call_graph([str(p)], str(tmp_path))
        caller_map = {e.caller.function: e for e in cg.entries}
        assert "main" in caller_map
        callee_names = [c.function for c in caller_map["main"].callees]
        assert "helper" in callee_names

    def test_self_method_call(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "self_call.py", """\
            class Service:
                def _do_work(self):
                    pass

                def run(self):
                    self._do_work()
        """)
        cg = build_call_graph([str(p)], str(tmp_path))
        caller_map = {e.caller.function: e for e in cg.entries}
        assert "run" in caller_map
        callee_names = [c.function for c in caller_map["run"].callees]
        assert "_do_work" in callee_names

    def test_dynamic_call_flagged(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "dynamic.py", """\
            def dispatch(name, obj):
                return getattr(obj, name)()
        """)
        cg = build_call_graph([str(p)], str(tmp_path))
        all_callees = [
            c.function
            for e in cg.entries
            for c in e.callees
        ]
        assert any("unresolved_dynamic" in fn for fn in all_callees)

    def test_cross_file_call(self, tmp_path: Path) -> None:
        utils = _write(tmp_path / "utils.py", """\
            def format_number(n: int) -> str:
                return str(n)
        """)
        main = _write(tmp_path / "main_mod.py", """\
            from utils import format_number

            def run():
                result = format_number(42)
                return result
        """)
        cg = build_call_graph([str(utils), str(main)], str(tmp_path))
        caller_map = {e.caller.function: e for e in cg.entries}
        assert "run" in caller_map
        callee_names = [c.function for c in caller_map["run"].callees]
        assert "format_number" in callee_names

    def test_empty_file_list(self, tmp_path: Path) -> None:
        cg = build_call_graph([], str(tmp_path))
        assert cg.entries == []


# ---------------------------------------------------------------------------
# Dependency Analyser tests
# ---------------------------------------------------------------------------

class TestDependencyAnalyzer:
    def test_external_package_detected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "req.py", """\
            import requests

            def fetch(url):
                return requests.get(url)
        """)
        dt = analyse_dependencies([str(p)], str(tmp_path))
        pkg_names = {pkg.name for pkg in dt.external_packages}
        assert "requests" in pkg_names

    def test_internal_import_detected(self, tmp_path: Path) -> None:
        utils = _write(tmp_path / "utils.py", """\
            def helper():
                pass
        """)
        main = _write(tmp_path / "main_dep.py", """\
            from utils import helper
        """)
        dt = analyse_dependencies([str(utils), str(main)], str(tmp_path))
        internal_modules = {imp.imported_module for imp in dt.internal_imports}
        assert "utils" in internal_modules

    def test_stdlib_not_in_external(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "stdlib_use.py", """\
            import os
            import sys
            import json
        """)
        dt = analyse_dependencies([str(p)], str(tmp_path))
        pkg_names = {pkg.name for pkg in dt.external_packages}
        assert "os" not in pkg_names
        assert "sys" not in pkg_names
        assert "json" not in pkg_names

    def test_version_from_requirements(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text("pydantic>=2.0\n")
        p = _write(tmp_path / "use_pydantic.py", """\
            from pydantic import BaseModel
        """)
        dt = analyse_dependencies([str(p)], str(tmp_path))
        pydantic_pkg = next((pkg for pkg in dt.external_packages if pkg.name == "pydantic"), None)
        assert pydantic_pkg is not None
        assert pydantic_pkg.version is not None and "2" in pydantic_pkg.version

    def test_used_by_files_populated(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "use_yaml.py", """\
            import yaml
        """)
        dt = analyse_dependencies([str(p)], str(tmp_path))
        yaml_pkg = next((pkg for pkg in dt.external_packages if pkg.name == "yaml"), None)
        assert yaml_pkg is not None
        assert str(p) in yaml_pkg.used_by_files

    def test_relative_import(self, tmp_path: Path) -> None:
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "helpers.py").write_text("def h(): pass\n")
        main = _write(pkg / "main.py", """\
            from . import helpers
        """)
        dt = analyse_dependencies([str(main)], str(tmp_path))
        # Relative imports should be recorded as internal
        assert any(imp.source_file == str(main) for imp in dt.internal_imports)


# ---------------------------------------------------------------------------
# Data Flow Detector tests
# ---------------------------------------------------------------------------

class TestDataFlowDetector:
    def test_db_read_detected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "db_read.py", """\
            def get_users(session):
                return session.query(User).all()
        """)
        df = detect_data_flows([str(p)])
        from models.schemas import FlowType
        entries = [e for e in df.entries if e.flow_type == FlowType.DB_READ]
        assert entries

    def test_db_write_detected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "db_write.py", """\
            def create_user(session, user):
                session.add(user)
                session.commit()
        """)
        df = detect_data_flows([str(p)])
        from models.schemas import FlowType
        entries = [e for e in df.entries if e.flow_type == FlowType.DB_WRITE]
        assert entries

    def test_api_call_detected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "api.py", """\
            import requests

            def call_service(url):
                return requests.get(url)
        """)
        df = detect_data_flows([str(p)])
        from models.schemas import FlowType
        entries = [e for e in df.entries if e.flow_type == FlowType.API_CALL]
        assert entries

    def test_file_io_detected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "file_io.py", """\
            def read_config(path):
                with open(path) as f:
                    return f.read()
        """)
        df = detect_data_flows([str(p)])
        from models.schemas import FlowType
        entries = [e for e in df.entries if e.flow_type == FlowType.FILE_IO]
        assert entries

    def test_env_var_detected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "env.py", """\
            import os

            def get_db_url():
                return os.environ.get("DATABASE_URL")
        """)
        df = detect_data_flows([str(p)])
        from models.schemas import FlowType
        entries = [e for e in df.entries if e.flow_type == FlowType.ENV_VAR]
        assert entries

    def test_message_queue_detected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "mq.py", """\
            import pika

            def send_message(msg):
                conn = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
                channel = conn.channel()
                channel.queue_declare(queue="hello")
        """)
        df = detect_data_flows([str(p)])
        from models.schemas import FlowType
        entries = [e for e in df.entries if e.flow_type == FlowType.MESSAGE_QUEUE]
        assert entries

    def test_socket_detected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "sockets.py", """\
            import socket

            def connect(host, port):
                s = socket.socket()
                s.connect((host, port))
        """)
        df = detect_data_flows([str(p)])
        from models.schemas import FlowType
        entries = [e for e in df.entries if e.flow_type == FlowType.SOCKET]
        assert entries

    def test_function_name_captured(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "named.py", """\
            def fetch_data(url):
                import requests
                return requests.get(url)
        """)
        df = detect_data_flows([str(p)])
        from models.schemas import FlowType
        api_entries = [e for e in df.entries if e.flow_type == FlowType.API_CALL]
        assert any(e.function == "fetch_data" for e in api_entries)

    def test_line_number_positive(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "lines.py", """\
            import os

            def use_env():
                return os.getenv("KEY")
        """)
        df = detect_data_flows([str(p)])
        assert all(e.line >= 1 for e in df.entries)


# ---------------------------------------------------------------------------
# Phase 2 integration test
# ---------------------------------------------------------------------------

class TestPhase2Integration:
    def test_full_phase2_run(self, tmp_path: Path) -> None:
        """Phase 2 must complete without error and populate all artifact fields."""
        src = _write(tmp_path / "service.py", """\
            import os
            import requests

            class DataService:
                def fetch(self, url: str) -> dict:
                    \"\"\"Fetch data from external API.\"\"\"
                    return requests.get(url).json()

                def save(self, session, obj) -> None:
                    session.add(obj)
                    session.commit()

            def load_config() -> str:
                return os.getenv("CONFIG_PATH", "config.yaml")
        """)

        config = _make_config(str(tmp_path), str(tmp_path / "artifacts"))
        artifacts = PhaseArtifacts()
        run_phase1(config=config, artifacts=artifacts)
        run_phase2(config=config, artifacts=artifacts)

        assert artifacts.ast_nodes is not None
        assert artifacts.call_graph is not None
        assert artifacts.dependency_tree is not None
        assert artifacts.data_flow is not None
        assert artifacts.component_map is not None

    def test_component_map_groups_by_directory(self, tmp_path: Path) -> None:
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        _write(api_dir / "views.py", "def index(): pass\n")
        _write(db_dir / "models.py", "class User: pass\n")

        config = _make_config(str(tmp_path), str(tmp_path / "artifacts"))
        artifacts = PhaseArtifacts()
        run_phase1(config=config, artifacts=artifacts)
        run_phase2(config=config, artifacts=artifacts)

        assert artifacts.component_map is not None
        comp_names = {c.component_name for c in artifacts.component_map.components}
        assert "api" in comp_names
        assert "db" in comp_names

    def test_phase2_requires_phase1(self, tmp_path: Path) -> None:
        artifacts = PhaseArtifacts()  # no file_inventory
        config = _make_config(str(tmp_path), str(tmp_path / "artifacts"))
        with pytest.raises(RuntimeError):
            run_phase2(config=config, artifacts=artifacts)

    def test_artifacts_json_written(self, tmp_path: Path) -> None:
        _write(tmp_path / "mod.py", "def f(): pass\n")
        arts_dir = tmp_path / "artifacts"
        config = _make_config(str(tmp_path), str(arts_dir))
        artifacts = PhaseArtifacts()
        run_phase1(config=config, artifacts=artifacts)
        run_phase2(config=config, artifacts=artifacts)

        for filename in [
            "ast_nodes.json",
            "call_graph.json",
            "dependency_tree.json",
            "data_flow.json",
            "component_map.json",
        ]:
            assert (arts_dir / filename).exists(), f"{filename} not written"
