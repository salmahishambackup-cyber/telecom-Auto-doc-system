# AI-Powered Telecom Codebase Documentation System

An intelligent system that analyzes Python codebases in the telecom domain and produces comprehensive business + technical documentation.

## Status

✅ Foundation, Phase 1 (File Discovery & Metadata), and Phase 2 (Deep Static Analysis) implemented.

## Project Structure

```
telecom-Auto-doc-system/
├── main.py                          # CLI orchestrator
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

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run against a codebase

```bash
python main.py --target /path/to/your/codebase --phases 1,2
```

### Run only Phase 1

```bash
python main.py --target /path/to/your/codebase --phases 1
```

### Resume from existing artifacts

```bash
python main.py --target /path/to/your/codebase --phases 1,2 --resume
```

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
