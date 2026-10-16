from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "jarvis-agent-backend"
    version: str = "0.3.0"
    timestamp: datetime = Field(default_factory=utc_now)


class DatabaseHealthResponse(BaseModel):
    status: str  # "connected" or "disconnected"
    database: str
    latency_ms: Optional[float] = None
    uri: Optional[str] = None
    error: Optional[str] = None
    hint: Optional[str] = None


class GroqHealthResponse(BaseModel):
    status: str  # "connected", "disconnected", or "degraded"
    model: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    hint: Optional[str] = None


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    description: Optional[str] = Field(default="", max_length=2000, description="Task description")
    priority: Optional[str] = Field(default="medium", pattern="^(low|medium|high|critical)$")
    deadline: Optional[str] = Field(default=None, description="Optional ISO deadline string, e.g. 2026-10-20")


class TaskCreate(TaskBase):
    pass


class Task(TaskBase):
    id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolStepStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRIED = "retried"
    BLOCKED = "blocked"


class ToolStep(BaseModel):
    step_id: str
    run_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    status: ToolStepStatus
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AgentRun(BaseModel):
    id: str
    command: str
    task_id: Optional[str] = None
    status: RunStatus = RunStatus.RUNNING
    model: str
    steps: List[ToolStep] = Field(default_factory=list)
    output: Optional[str] = None
    spoken_summary: Optional[str] = None
    report_url: Optional[str] = None
    goal_achieved: Optional[bool] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None


class CommandRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=1000, description="Natural language prompt for Jarvis")
    task_id: Optional[str] = Field(default=None, description="Optional existing task ID to bind to")
    source: Optional[str] = Field(default="text", description="Source of command: 'text' or 'voice'")


class ReportInfo(BaseModel):
    filename: str
    file_path: str
    download_url: str
    size_bytes: int
    created_at: str


class IntegrationStatus(BaseModel):
    name: str
    status: str  # "ready", "pending_setup", "disabled"
    description: str
    auth_type: str
    docs_url: Optional[str] = None


class IntegrationsResponse(BaseModel):
    integrations: Dict[str, IntegrationStatus]
