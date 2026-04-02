from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class FlowType(str, Enum):
    DB_READ = "db_read"
    DB_WRITE = "db_write"
    API_CALL = "api_call"
    FILE_IO = "file_io"
    ENV_VAR = "env_var"
    MESSAGE_QUEUE = "message_queue"
    SOCKET = "socket"


class FileMetadata(BaseModel):
    path: str
    language: str
    size_bytes: int = Field(ge=0)
    line_count: int = Field(ge=0)
    last_modified: datetime
    encoding: str = "utf-8"


class FileInventory(BaseModel):
    files: list[FileMetadata] = []
    project_root: str
    scan_timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    total_files: int = Field(default=0, ge=0)


class ASTNode(BaseModel):
    file_path: str
    node_type: NodeType
    name: str
    args: list[str] = []
    return_type: Optional[str] = None
    decorators: list[str] = []
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    docstring_existing: Optional[str] = None


class ASTNodes(BaseModel):
    nodes: list[ASTNode] = []


class CalleeRef(BaseModel):
    file: str
    function: str
    line: int = Field(ge=1)


class CallerRef(BaseModel):
    file: str
    function: str


class CallGraphEntry(BaseModel):
    caller: CallerRef
    callees: list[CalleeRef] = []


class CallGraph(BaseModel):
    entries: list[CallGraphEntry] = []


class InternalImport(BaseModel):
    source_file: str
    imported_module: str
    imported_names: list[str] = []


class ExternalPackage(BaseModel):
    name: str
    version: Optional[str] = None
    used_by_files: list[str] = []


class DependencyTree(BaseModel):
    internal_imports: list[InternalImport] = []
    external_packages: list[ExternalPackage] = []


class DataFlowEntry(BaseModel):
    file: str
    function: str
    flow_type: FlowType
    details: str
    line: int = Field(ge=1)


class DataFlow(BaseModel):
    entries: list[DataFlowEntry] = []


class ComponentEntry(BaseModel):
    component_name: str
    directory: str
    files: list[str] = []
    purpose_hint: str = ""
    entry_points: list[str] = []
    dependencies: list[str] = []


class ComponentMap(BaseModel):
    components: list[ComponentEntry] = []


class DocstringEntry(BaseModel):
    """A single generated docstring."""

    function_id: str
    file_path: str
    function_name: str
    class_name: Optional[str] = None
    node_type: NodeType
    docstring: str
    confidence: float = Field(ge=0.0, le=1.0)
    provider_used: str
    fallback_used: bool = False
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class DocstringFailure(BaseModel):
    """Records a failed docstring generation."""

    function_id: str
    file_path: str
    function_name: str
    reason: str
    provider: str
    error_type: str


class ModuleDocstring(BaseModel):
    """Module-level docstring for an entire file."""

    file_path: str
    docstring: str
    confidence: float = Field(ge=0.0, le=1.0)
    provider_used: str


class Docstrings(BaseModel):
    """Complete Phase 3 output artifact."""

    entries: list[DocstringEntry] = []
    module_docstrings: list[ModuleDocstring] = []
    failures: list[DocstringFailure] = []
    total_functions: int = 0
    successful: int = 0
    failed: int = 0
    average_confidence: float = 0.0
    fallback_count: int = 0


class PhaseArtifacts(BaseModel):
    file_inventory: Optional[FileInventory] = None
    ast_nodes: Optional[ASTNodes] = None
    call_graph: Optional[CallGraph] = None
    dependency_tree: Optional[DependencyTree] = None
    data_flow: Optional[DataFlow] = None
    component_map: Optional[ComponentMap] = None
    docstrings: Optional[Docstrings] = None
