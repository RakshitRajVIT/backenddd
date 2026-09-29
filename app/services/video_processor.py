"""Orchestrates the full pipeline: read video -> run 3 detectors -> annotate -> write output -> report."""
from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from app.config import Settings
from app.errors import JobCancelled, ProcessingError
from app.models.model_config import PipelineConfig
from app.models.model_loader import ModelRegistry
from app.schemas.video import (
    CategorySummary, Detection, ModelRunInfo, ProcessingInfo, Report, ReportSummary, ViolationEvent, VideoInfo,
)
from app.services.base_detector import BaseDetector
from app.services.helmet_detector import HelmetDetector
from app.services.job_manager import Job, JobManager, now_iso
from app.services.red_light_detector import RedLightDetector
from app.services.tripling_detector import TriplingDetector
from app.utils.video_utils import annotate_frame, open_writer, reencode_h264, safe_unlink

logger = logging.getLogger("civic_mirror.pipeline")

CATEGORY_CLASS = {"helmet": HelmetDetector, "tripling": TriplingDetector, "red_light": RedLightDetector}


def build_detectors(registry: ModelRegistry, pipeline_cfg: PipelineConfig) -> Dict[str, BaseDetector]:
    configs = {"helmet": pipeline_cfg.helmet, "tripling": pipeline_cfg.tripling, "red_light": pipeline_cfg.red_light}
    detectors: Dict[str, BaseDetector] = {}
    for key, cfg in configs.items():
        if registry.is_loaded(key):
            detectors[key] = CATEGORY_CLASS[key](registry, cfg)
    return detectors


class VideoProcessor:
    def __init__(self, settings: Settings, registry: ModelRegistry, pipeline_cfg: PipelineConfig, jobs: JobManager):
        self.settings = settings
        self.registry = registry
        self.pipeline_cfg = pipeline_cfg
        self.jobs = jobs

    def run(self, job: Job) -> None:
        job_id = job.job_id
        self.jobs.mark_processing(job_id)
        cap = None
        writer = None
        raw_output_path = self.settings.processed_dir / f"{job_id}_raw.mp4"
        final_output_path = self.settings.processed_dir / f"{job_id}.mp4"
        report_path = self.settings.processed_dir / f"{job_id}_report.json"
        try:
            if not job.upload_path.is_file():
                raise ProcessingError(f"Uploaded file is missing: {job.upload_path}")

            detectors = build_detectors(self.registry, self.pipeline_cfg)
            if not detectors:
                raise ProcessingError("No detection models are loaded; cannot process video.")
            if self.settings.require_all_models and len(detectors) < 3:
                missing = [k for k in CATEGORY_CLASS if k not in detectors]
                raise ProcessingError(f"REQUIRE_ALL_MODELS is set and these models failed to load: {missing}")

            cap = cv2.VideoCapture(str(job.upload_path))
            if not cap.isOpened():
                raise ProcessingError("Could not reopen the uploaded video for processing.")

            video = job.video
            writer, codec_used = open_writer(
                raw_output_path, video.fps, (video.width, video.height),
                [self.settings.output_codec, "mp4v", "avc1", "XVID"],
            )

            frame_index = 0
            confirmed_counts: Counter = Counter({k: 0 for k in detectors})
            last_signal_state: Optional[str] = None
            last_stop_line = None
            t0 = time.time()
            stride = self.settings.frame_stride
            last_outputs: Dict[str, list] = {k: [] for k in detectors}

            while True:
                if self.jobs.is_cancelled(job_id):
                    raise JobCancelled(f"Job {job_id} was cancelled.")
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                timestamp_s = frame_index / video.fps if video.fps > 0 else 0.0

                run_inference = (frame_index % stride == 0)
                all_detections: List[Detection] = []
                for key, det in detectors.items():
                    if run_inference:
                        out = det.process_frame(frame, frame_index, timestamp_s)
                        last_outputs[key] = out.detections
                        if out.signal_state is not None:
                            last_signal_state = out.signal_state
                        if out.stop_line is not None:
                            last_stop_line = out.stop_line
                    all_detections.extend(last_outputs[key])
                    confirmed_counts[key] = len(det.confirmed_events())

                annotate_frame(
                    frame, all_detections, frame_index, timestamp_s,
                    dict(confirmed_counts), last_signal_state, last_stop_line,
                    self.settings.draw_context_objects,
                )
                writer.write(frame)

                frame_index += 1
                if video.frame_count > 0:
                    progress = min(99.0, 100.0 * frame_index / video.frame_count)
                else:
                    progress = min(95.0, frame_index / 10.0)
                if frame_index % 10 == 0 or frame_index == video.frame_count:
                    self.jobs.update_progress(job_id, frame_index, progress)

            elapsed = time.time() - t0
            writer.release()
            writer = None
            cap.release()
            cap = None

            final_path, was_reencoded = self._finalize_output(raw_output_path, final_output_path)

            report = self._build_report(
                job=job, detectors=detectors, frames_processed=frame_index, elapsed=elapsed,
                codec_used=codec_used, was_reencoded=was_reencoded,
            )
            report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

            self.jobs.mark_completed(job_id, final_path, report_path, report)
            self.jobs.update_progress(job_id, frame_index, 100.0)
            logger.info("Job %s completed: %d frames in %.1fs", job_id, frame_index, elapsed)

        except JobCancelled as exc:
            logger.info(str(exc))
            self.jobs.mark_failed(job_id, "Cancelled by user.")
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self.jobs.mark_failed(job_id, str(exc))
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            safe_unlink(raw_output_path) if raw_output_path.exists() and raw_output_path != final_output_path and final_output_path.exists() else None

    def _finalize_output(self, raw_path: Path, final_path: Path) -> tuple[Path, bool]:
        if not self.settings.reencode_h264:
            raw_path.replace(final_path)
            return final_path, False
        ok, msg = reencode_h264(raw_path, final_path, self.settings.ffmpeg_path)
        if ok:
            safe_unlink(raw_path)
            logger.info(msg)
            return final_path, True
        logger.warning(msg)
        raw_path.replace(final_path)
        return final_path, False

    def _build_report(
        self, job: Job, detectors: Dict[str, BaseDetector], frames_processed: int,
        elapsed: float, codec_used: str, was_reencoded: bool,
    ) -> Report:
        violations: List[ViolationEvent] = []
        categories: Dict[str, CategorySummary] = {}
        for key, det in detectors.items():
            violations.extend(det.confirmed_events())
            categories[key] = det.summarize()

        model_runs: List[ModelRunInfo] = []
        for key, spec in self.pipeline_cfg.specs().items():
            loaded = self.registry.is_loaded(key)
            model_runs.append(ModelRunInfo(
                key=key, path=str(spec.path), loaded=loaded, used=key in detectors,
                class_names=self.registry.class_names(key) if loaded else {},
                conf_threshold=spec.conf,
                skipped_reason=None if loaded else (self.registry.error(key) or "Model not loaded."),
            ))

        violations.sort(key=lambda v: v.first_frame)
        notes: List[str] = []
        for det in detectors.values():
            notes.extend(f"[{det.category}] {n}" for n in det.notes)
        for key in CATEGORY_CLASS:
            if key not in detectors:
                notes.append(f"[{key}] Model unavailable; this category was skipped for this video.")

        return Report(
            job_id=job.job_id,
            source_filename=job.original_filename,
            generated_at=now_iso(),
            video=job.video,
            processing=ProcessingInfo(
                device=self.registry.device,
                frames_processed=frames_processed,
                frame_stride=self.settings.frame_stride,
                elapsed_s=round(elapsed, 3),
                average_fps=round(frames_processed / elapsed, 2) if elapsed > 0 else 0.0,
                output_codec=codec_used,
                output_h264_reencoded=was_reencoded,
                models=model_runs,
            ),
            summary=ReportSummary(
                total_confirmed_violations=len(violations),
                categories=categories,
            ),
            violations=violations,
            notes=notes,
        )
