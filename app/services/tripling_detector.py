"""Triple-riding detector.

Supports two independent, configurable paths (either or both may be active):

1. Direct: the model has its own "triple riding" class -> TRIPLING_VIOLATION_CLASSES.
2. Association: person detections are grouped onto motorcycle detections by
   spatial overlap; a motorcycle with >= min_riders overlapping people is a
   candidate -> TRIPLING_PERSON_CLASSES + TRIPLING_MOTORCYCLE_CLASSES.

Either path only produces a *candidate*; PersistenceVerifier confirms across
frames before anything is reported as a violation.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.models.model_config import TriplingConfig
from app.models.model_loader import ModelRegistry, RawDetection
from app.schemas.video import Detection, DetectorOutput, ViolationEvent
from app.services.base_detector import BaseDetector, PersistenceVerifier
from app.utils.video_utils import Box, box_area, intersection_area


def _expand_motorcycle_box(b: Box, expand_x: float, expand_up: float) -> Box:
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    return (x1 - w * expand_x, y1 - h * expand_up, x2 + w * expand_x, y2)


class TriplingDetector(BaseDetector):
    category = "tripling"
    verification_description = "temporal_persistence (tracked across consecutive frames)"

    def __init__(self, registry: ModelRegistry, config: TriplingConfig):
        super().__init__(registry, config.spec)
        self.config = config
        self.verifier = PersistenceVerifier(
            category=self.category,
            method=self.verification_description,
            min_hits=config.min_hits,
            track_iou=config.track_iou,
            max_age=config.track_max_age,
        )
        self.association_enabled = config.person.configured and config.motorcycle.configured
        if not config.violation.configured and not self.association_enabled:
            self.notes.append(
                "Neither TRIPLING_VIOLATION_CLASSES nor the "
                "TRIPLING_PERSON_CLASSES/TRIPLING_MOTORCYCLE_CLASSES pair is configured: all detections are "
                "shown as context only and no violations can be confirmed."
            )
        elif self.association_enabled:
            self.notes.append(
                f"Person-motorcycle association active (min_riders={config.min_riders}, "
                f"overlap>={config.assoc_overlap})."
            )

    def _direct_candidates(self, raw: List[RawDetection], frame_index: int, timestamp_s: float) -> List[Detection]:
        out = []
        for r in raw:
            if self.config.violation.configured and self.config.violation.matches(r.class_id, r.class_name):
                det = self.make_detection(r, frame_index, timestamp_s, role="violation")
                det.extra["method"] = "direct_class"
                out.append(det)
        return out

    def _association_candidates(
        self, raw: List[RawDetection], frame_index: int, timestamp_s: float
    ) -> List[Detection]:
        people = [r for r in raw if self.config.person.matches(r.class_id, r.class_name)]
        motos = [r for r in raw if self.config.motorcycle.matches(r.class_id, r.class_name)]
        out: List[Detection] = []
        for m in motos:
            zone = _expand_motorcycle_box(m.bbox, self.config.expand_x, self.config.expand_up)
            riders = []
            for p in people:
                area = box_area(p.bbox)
                if area <= 0:
                    continue
                overlap = intersection_area(zone, p.bbox) / area
                if overlap >= self.config.assoc_overlap:
                    riders.append(p)
            if len(riders) >= self.config.min_riders:
                det = self.make_detection(m, frame_index, timestamp_s, role="violation")
                det.class_name = f"{m.class_name} (+{len(riders)} riders)"
                det.extra["method"] = "person_motorcycle_association"
                det.extra["rider_count"] = len(riders)
                det.extra["rider_confidences"] = [round(p.confidence, 3) for p in riders]
                out.append(det)
        return out

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> DetectorOutput:
        raw = self.predict(frame)
        detections: List[Detection] = [self.make_detection(r, frame_index, timestamp_s) for r in raw]
        candidates = self._direct_candidates(raw, frame_index, timestamp_s)
        candidates += self._association_candidates(raw, frame_index, timestamp_s)
        for c in candidates:
            c.status, c.verification = "candidate", "pending_temporal_persistence"
        detections.extend(candidates)
        events = self.verifier.update(candidates, frame_index, timestamp_s)
        return DetectorOutput(detections=detections, events=events)

    def confirmed_events(self) -> List[ViolationEvent]:
        return list(self.verifier.events.values())

    def unconfirmed_candidates(self) -> int:
        return self.verifier.unconfirmed_tracks()
