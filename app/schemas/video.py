"""API schemas (Pydantic) and internal detection containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --------------------------------------------------------------------- envelope
class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = ""
    data: Optional[T] = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody


# ------------------------------------------------------------------ video / jobs
class VideoInfo(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: Optional[float] = None
    fps_assumed: bool = False


class UploadData(BaseModel):
    job_id: str
    filename: str
    size_bytes: int
    status: JobStatus
    video: VideoInfo


class ProcessData(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusData(BaseModel):
    job_id: str
    original_filename: str
    status: JobStatus
    progress: float = Field(description="Percent 0-100")
    frames_processed: int
    total_frames: int
    message: str
    error: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    video: Optional[VideoInfo] = None
    summary: Optional[Dict[str, Any]] = None


class DeleteData(BaseModel):
    job_id: str
    deleted: List[str]


# ------------------------------------------------------------------------ health
class ModelStatus(BaseModel):
    loaded: bool
    path: str
    path_exists: bool
    classes: Dict[int, str] = {}
    conf_threshold: float
    error: Optional[str] = None


class HealthData(BaseModel):
    status: str
    app: str
    device: str
    cuda_available: bool
    half_precision: bool
    torch_version: str
    ffmpeg_available: bool
    max_upload_mb: int
    models: Dict[str, ModelStatus]
    class_roles: Dict[str, Dict[str, List[str]]]
    jobs: Dict[str, int]
    warnings: List[str] = []


# ------------------------------------------------------------------------ report
class ViolationEvent(BaseModel):
    event_id: str
    category: str
    label: str
    status: str = "confirmed"
    verification_method: str
    track_id: Optional[int] = None
    first_frame: int
    first_timestamp_s: float
    last_frame: int
    last_timestamp_s: float
    hits: int
    max_confidence: float
    mean_confidence: float
    bbox: List[float]
    details: Dict[str, Any] = {}


class ModelRunInfo(BaseModel):
    key: str
    path: str
    loaded: bool
    used: bool
    class_names: Dict[int, str] = {}
    conf_threshold: float
    skipped_reason: Optional[str] = None


class CategorySummary(BaseModel):
    category: str
    confirmed_violations: int
    unconfirmed_candidates: int
    verification_method: str
    detections_by_class: Dict[str, int] = Field(
        default_factory=dict, description="Per-frame detection counts by class (not unique objects)"
    )
    notes: List[str] = []


class ProcessingInfo(BaseModel):
    device: str
    frames_processed: int
    frame_stride: int
    elapsed_s: float
    average_fps: float
    output_codec: str
    output_h264_reencoded: bool
    models: List[ModelRunInfo]


class ReportSummary(BaseModel):
    total_confirmed_violations: int
    categories: Dict[str, CategorySummary]


class Report(BaseModel):
    job_id: str
    source_filename: str
    generated_at: str
    video: VideoInfo
    processing: ProcessingInfo
    summary: ReportSummary
    violations: List[ViolationEvent]
    notes: List[str] = []


# ------------------------------------------------------- internal (not API) types
@dataclass
class Detection:
    """Common detection format shared by all three detectors."""

    category: str  # helmet | tripling | red_light
    source_model: str
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    frame_index: int
    timestamp_s: float
    role: str = "context"  # "violation" (a violation candidate/confirmed) or "context"
    status: str = "detected"  # detected | candidate | confirmed
    track_id: Optional[int] = None
    verification: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorOutput:
    detections: List[Detection] = field(default_factory=list)
    events: List[ViolationEvent] = field(default_factory=list)  # newly confirmed this frame
    signal_state: Optional[str] = None
    stop_line: Optional[Tuple[float, float, float, float]] = None  # normalized
