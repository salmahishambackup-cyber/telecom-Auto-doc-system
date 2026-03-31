# AI-Powered Telecom Codebase Documentation System

An intelligent system that analyzes Python codebases in the telecom domain and produces comprehensive business + technical documentation.

## Status

✅ Foundation, Phase 1 (File Discovery & Metadata), and Phase 2 (Deep Static Analysis) implemented.

## Project Structure

```
telecom-Auto-doc-system/
├── usage.ipynb                      # ⭐ Start here — Jupyter notebook
├── main.py                          # Pipeline API + optional CLI
├── config.yaml                      # Full configuration
├── requirements.txt                 # Dependencies
├── models/
│   ├── __init__.py
│   └── schemas.py                   # All Pydantic v2 data models
├── phases/
│   ├── __init__.py
│   ├── phase1_discovery.py          # Phase 1: File Discovery
│   └── phase2_static_analysis.py    # Phase 2: Static Analysis orchestrator
├── tools/
│   ├── __init__.py
│   ├── ast_extractor.py             # AST node extraction
│   ├── call_graph.py                # Call graph builder
│   ├── dependency_analyzer.py       # Import & dependency analysis
│   └── data_flow_detector.py        # Data flow pattern detection
├── llm/
│   ├── __init__.py
│   └── base.py                      # Abstract LLM provider (stub for future phases)
├── output/
│   └── artifacts/                   # JSON artifacts output directory
└── tests/
    ├── __init__.py
    ├── test_phase1.py               # Phase 1 tests
    └── test_phase2.py               # Phase 2 tests
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
jupyter notebook usage.ipynb
```

The notebook walks through the full workflow:
- Running Phase 1 & 2 against any Python codebase
- Exploring the returned artifacts (file inventory, AST nodes, call graph, dependencies, data flow, component map)
- Running individual phases
- Exporting artifacts as JSON
- Resuming from previously saved artifacts

### 3. Or use from any Python script / notebook cell

```python
from main import run_pipeline

artifacts = run_pipeline(target="/path/to/your/codebase", phases="1,2")

# Explore results
print(f"Files found: {artifacts.file_inventory.total_files}")
print(f"AST nodes:   {len(artifacts.ast_nodes.nodes)}")
print(f"Call graph:   {len(artifacts.call_graph.entries)} entries")
print(f"External deps: {len(artifacts.dependency_tree.external_packages)}")
print(f"Data flows:   {len(artifacts.data_flow.entries)}")
print(f"Components:   {len(artifacts.component_map.components)}")
```

### `run_pipeline` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target` | `str` | *(required)* | Path to the Python codebase to analyse |
| `phases` | `str \| list` | `"1,2"` | Which phases to run (comma-separated or list) |
| `config_path` | `str` | `"config.yaml"` | Path to the YAML configuration file |
| `resume` | `bool` | `False` | Skip phases whose artifacts already exist on disk |
| `artifacts_dir` | `str` | `""` | Override the output directory for JSON artifacts |
| `log_level` | `str` | `"INFO"` | Logging level (`DEBUG`, `INFO`, `WARNING`, …) |

## Running Tests

```bash
pytest tests/ -v
```

## Output Artifacts

After running, the following JSON files are written to `output/artifacts/`:

| File | Description |
|------|-------------|
| `file_inventory.json` | All discovered Python files with metadata |
| `ast_nodes.json` | All classes, functions, and methods |
| `call_graph.json` | Caller → callee relationships |
| `dependency_tree.json` | Internal imports and external packages |
| `data_flow.json` | DB, API, file I/O, env var, queue, socket patterns |
| `component_map.json` | Files grouped by directory with entry points |

## Configuration

Edit `config.yaml` to configure target path, file filters, analysis options, and output directory.
