"""Helmet-violation detector.

Detection alone is not treated as a confirmed violation: a candidate must be
observed on the same tracked object for `min_hits` frames before it is
reported as confirmed (temporal-persistence verification). This filters out
one-off false positives from a single noisy frame.
"""
from __future__ import annotations

from typing import List

import numpy as np

from app.models.model_config import HelmetConfig
from app.models.model_loader import ModelRegistry
from app.schemas.video import Detection, DetectorOutput, ViolationEvent
from app.services.base_detector import BaseDetector, PersistenceVerifier


class HelmetDetector(BaseDetector):
    category = "helmet"
    verification_description = "temporal_persistence (tracked across consecutive frames)"

    def __init__(self, registry: ModelRegistry, config: HelmetConfig):
        super().__init__(registry, config.spec)
        self.config = config
        self.verifier = PersistenceVerifier(
            category=self.category,
            method=self.verification_description,
            min_hits=config.min_hits,
            track_iou=config.track_iou,
            max_age=config.track_max_age,
        )
        if not config.violation.configured:
            self.notes.append(
                "HELMET_VIOLATION_CLASSES is not configured: all detections are shown as context only "
                "and no violations can be confirmed. Set this env var to the model's 'no helmet' class name(s)."
            )

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> DetectorOutput:
        raw = self.predict(frame)
        detections: List[Detection] = []
        candidates: List[Detection] = []
        for r in raw:
            is_violation = self.config.violation.configured and self.config.violation.matches(r.class_id, r.class_name)
            det = self.make_detection(r, frame_index, timestamp_s, role="violation" if is_violation else "context")
            detections.append(det)
            if is_violation:
                det.status = "candidate"
                det.verification = "pending_temporal_persistence"
                candidates.append(det)
        events = self.verifier.update(candidates, frame_index, timestamp_s)
        return DetectorOutput(detections=detections, events=events)

    def confirmed_events(self) -> List[ViolationEvent]:
        return list(self.verifier.events.values())

    def unconfirmed_candidates(self) -> int:
        return self.verifier.unconfirmed_tracks()
