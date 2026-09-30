from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse

from app.errors import JobError
from app.schemas.video import (
    ApiResponse, DeleteData, JobStatus, JobStatusData, ProcessData, Report, UploadData, VideoInfo,
)
from app.services.video_processor import VideoProcessor
from app.utils.video_utils import probe_video, safe_unlink

logger = logging.getLogger("civic_mirror.routes.videos")
router = APIRouter(prefix="/api/videos", tags=["videos"])

CHUNK_SIZE = 1024 * 1024


def _job_or_404(request: Request, job_id: str):
    job = request.app.state.ctx.jobs.get(job_id)
    if job is None:
        raise JobError("job_not_found", f"No job found with id '{job_id}'.", status_code=404)
    return job


def _status_data(job) -> JobStatusData:
    summary = None
    if job.report is not None:
        summary = {
            "total_confirmed_violations": job.report.summary.total_confirmed_violations,
            "categories": {k: v.confirmed_violations for k, v in job.report.summary.categories.items()},
        }
    return JobStatusData(
        job_id=job.job_id,
        original_filename=job.original_filename,
        status=job.status,
        progress=job.progress,
        frames_processed=job.frames_processed,
        total_frames=job.total_frames,
        message=job.message,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        video=job.video,
        summary=summary,
    )


@router.post("/upload", response_model=ApiResponse[UploadData], status_code=201)
async def upload_video(request: Request, file: UploadFile = File(...)):
    ctx = request.app.state.ctx
    settings, jobs = ctx.settings, ctx.jobs

    if not file.filename:
        raise JobError("invalid_filename", "Uploaded file has no filename.", 400)
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise JobError(
            "unsupported_format",
            f"Unsupported file extension '{ext}'. Allowed: {', '.join(settings.allowed_extensions)}.",
            415,
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_id = Path(file.filename).stem
    dest_path = settings.upload_dir / f"upload_{tmp_id}_{Path(file.filename).name}"
    counter = 1
    while dest_path.exists():
        dest_path = settings.upload_dir / f"upload_{tmp_id}_{counter}_{Path(file.filename).name}"
        counter += 1

    size = 0
    max_bytes = settings.max_upload_bytes
    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    out.close()
                    safe_unlink(dest_path)
                    raise JobError(
                        "file_too_large",
                        f"File exceeds the {settings.max_upload_mb} MB upload limit.",
                        413,
                    )
                out.write(chunk)
    except JobError:
        raise
    except Exception as exc:
        safe_unlink(dest_path)
        raise JobError("upload_failed", f"Could not save uploaded file: {exc}", 400) from exc
    finally:
        await file.close()

    if size == 0:
        safe_unlink(dest_path)
        raise JobError("empty_file", "Uploaded file is empty.", 400)

    try:
        video_info: VideoInfo = probe_video(dest_path)
    except Exception as exc:
        safe_unlink(dest_path)
        raise JobError("invalid_video", f"Uploaded file is not a readable video: {exc}", 400) from exc

    job = jobs.create_job(original_filename=file.filename, upload_path=dest_path, video=video_info)
    # Start the worker immediately after the validated upload. This makes the
    # single user action ("Analyze Traffic Video") atomic from the browser's
    # perspective and avoids relying on a second HTTP request to begin analysis.
    started, _ = jobs.try_start_processing(job.job_id)
    if not started:  # Defensive: a freshly created job must always be startable.
        raise JobError("job_start_failed", "Could not queue the uploaded video for processing.", 500)
    threading.Thread(
        target=ctx.processor.run,
        args=(job,),
        name=f"analysis-{job.job_id[:8]}",
        daemon=True,
    ).start()
    logger.info("Uploaded job %s (%s, %d bytes)", job.job_id, file.filename, size)
    return ApiResponse(
        message="Upload accepted and analysis started.",
        data=UploadData(job_id=job.job_id, filename=file.filename, size_bytes=size, status=job.status, video=video_info),
    )


@router.post("/{job_id}/process", response_model=ApiResponse[ProcessData])
def process_video(job_id: str, request: Request):
    ctx = request.app.state.ctx
    job = _job_or_404(request, job_id)

    started, reason = ctx.jobs.try_start_processing(job_id)
    if not started:
        messages = {
            "already_in_progress": ("Processing already in progress for this job.", 409),
            "already_completed": ("This job has already completed. Delete it to reprocess.", 409),
            "not_found": ("Job not found.", 404),
        }
        msg, code = messages.get(reason, ("Could not start processing.", 409))
        raise JobError("duplicate_or_invalid_state", msg, code)

    processor: VideoProcessor = ctx.processor
    # Do not use FastAPI BackgroundTasks here: it runs after the response body but
    # remains attached to the request lifecycle. A standalone worker thread lets
    # the browser receive the job id immediately while CPU inference continues.
    threading.Thread(
        target=processor.run,
        args=(job,),
        name=f"analysis-{job_id[:8]}",
        daemon=True,
    ).start()
    logger.info("Queued processing for job %s", job_id)
    return ApiResponse(
        message="Processing started.",
        data=ProcessData(job_id=job_id, status=JobStatus.QUEUED, message="Job queued for background processing."),
    )


@router.get("/{job_id}/status", response_model=ApiResponse[JobStatusData])
def get_status(job_id: str, request: Request):
    job = _job_or_404(request, job_id)
    return ApiResponse(message="ok", data=_status_data(job))


@router.get("/{job_id}/results", response_model=ApiResponse[Report])
def get_results(job_id: str, request: Request):
    job = _job_or_404(request, job_id)
    if job.status != JobStatus.COMPLETED or job.report is None:
        raise JobError(
            "not_ready", f"Job is not completed yet (current status: {job.status.value}).", 409
        )
    return ApiResponse(message="ok", data=job.report)


@router.get("/{job_id}/download")
def download_video(job_id: str, request: Request):
    job = _job_or_404(request, job_id)
    if job.status != JobStatus.COMPLETED or not job.output_video_path or not job.output_video_path.is_file():
        raise JobError("not_ready", "Annotated video is not available yet.", 409)
    return FileResponse(
        path=job.output_video_path,
        media_type="video/mp4",
        filename=f"{Path(job.original_filename).stem}_annotated.mp4",
    )


@router.get("/{job_id}/report")
def download_report(job_id: str, request: Request):
    job = _job_or_404(request, job_id)
    if job.status != JobStatus.COMPLETED or not job.report_path or not job.report_path.is_file():
        raise JobError("not_ready", "Report is not available yet.", 409)
    return FileResponse(
        path=job.report_path,
        media_type="application/json",
        filename=f"{Path(job.original_filename).stem}_report.json",
    )


@router.get("/{job_id}/evidence/{event_id}")
def get_evidence(job_id: str, event_id: str, request: Request):
    job = _job_or_404(request, job_id)
    if job.status != JobStatus.COMPLETED or job.report is None:
        raise JobError("not_ready", "Evidence is not available until processing is complete.", 409)
    event = next((item for item in job.report.violations if item.event_id == event_id), None)
    filename = event.details.get("evidence_image") if event else None
    path = request.app.state.ctx.settings.processed_dir / job_id / "evidence" / str(filename or "")
    if not filename or not path.is_file():
        raise JobError("evidence_not_found", "No evidence frame is available for this event.", 404)
    return FileResponse(path=path, media_type="image/jpeg", filename=f"{event_id}.jpg")


@router.delete("/{job_id}", response_model=ApiResponse[DeleteData])
def delete_video(job_id: str, request: Request):
    ctx = request.app.state.ctx
    job = _job_or_404(request, job_id)

    if job.status == JobStatus.PROCESSING:
        # Signal the background worker to stop; safe_unlink below retries briefly so it
        # can succeed once the worker releases its file handles (important on Windows).
        ctx.jobs.request_cancel(job_id)

    deleted = []
    for path in (job.upload_path, job.output_video_path, job.report_path,
                 ctx.settings.processed_dir / f"{job_id}_raw.mp4"):
        if path and Path(path).exists():
            if safe_unlink(Path(path)):
                deleted.append(str(path))
    evidence_dir = ctx.settings.processed_dir / job_id
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir, ignore_errors=True)
        deleted.append(str(evidence_dir))

    ctx.jobs.delete(job_id)
    logger.info("Deleted job %s (%d files removed)", job_id, len(deleted))
    return ApiResponse(message="Deleted.", data=DeleteData(job_id=job_id, deleted=deleted))
