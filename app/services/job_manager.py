"""In-memory job registry with thread-safe state transitions and duplicate-processing protection."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from app.schemas.video import JobStatus, Report, VideoInfo


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    original_filename: str
    upload_path: Path
    video: VideoInfo
    status: JobStatus = JobStatus.UPLOADED
    progress: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    message: str = "Uploaded. Ready to process."
    error: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    output_video_path: Optional[Path] = None
    report_path: Optional[Path] = None
    report: Optional[Report] = None
    processing_lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_requested: bool = False


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def create_job(self, original_filename: str, upload_path: Path, video: VideoInfo) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            original_filename=original_filename,
            upload_path=upload_path,
            video=video,
            total_frames=video.frame_count,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def try_start_processing(self, job_id: str) -> tuple[bool, str]:
        """Atomically transition UPLOADED/FAILED -> QUEUED. Returns (started, reason_if_not)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, "not_found"
            if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                return False, "already_in_progress"
            if job.status == JobStatus.COMPLETED:
                return False, "already_completed"
            job.status = JobStatus.QUEUED
            job.cancel_requested = False
            job.message = "Queued for processing."
            job.updated_at = now_iso()
            return True, ""

    def mark_processing(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.PROCESSING
            job.started_at = now_iso()
            job.updated_at = job.started_at
            job.message = "Processing video."

    def update_progress(self, job_id: str, frames_processed: int, progress: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.frames_processed = frames_processed
            job.progress = min(100.0, max(0.0, progress))
            job.updated_at = now_iso()

    def mark_completed(self, job_id: str, output_video_path: Path, report_path: Path, report: Report) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.output_video_path = output_video_path
            job.report_path = report_path
            job.report = report
            job.message = "Processing complete."
            job.finished_at = now_iso()
            job.updated_at = job.finished_at

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.error = error
            job.message = "Processing failed."
            job.finished_at = now_iso()
            job.updated_at = job.finished_at

    def request_cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.cancel_requested = True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_requested)

    def delete(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def counts(self) -> Dict[str, int]:
        with self._lock:
            out = {s.value: 0 for s in JobStatus}
            for job in self._jobs.values():
                out[job.status.value] += 1
            out["total"] = len(self._jobs)
            return out
