"""Common detector base class and the temporal-persistence verifier."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List

import numpy as np

from app.models.model_config import ModelSpec
from app.models.model_loader import ModelRegistry, RawDetection
from app.schemas.video import CategorySummary, Detection, DetectorOutput, ViolationEvent
from app.utils.video_utils import IoUTracker


class PersistenceVerifier:
    """Tracks violation candidates and confirms them only after `min_hits` observations.

    A single-frame detection stays a *candidate* and is never reported as a confirmed violation.
    """

    def __init__(self, category: str, method: str, min_hits: int, track_iou: float, max_age: int):
        self.category = category
        self.method = method
        self.min_hits = min_hits
        self.tracker = IoUTracker(track_iou, max_age)
        self._stats: Dict[int, dict] = {}
        self.events: Dict[int, ViolationEvent] = {}
        self._counter = 0

    def update(self, candidates: List[Detection], frame_index: int, timestamp_s: float) -> List[ViolationEvent]:
        new_events: List[ViolationEvent] = []
        ids = self.tracker.update([c.bbox for c in candidates])
        for det, tid in zip(candidates, ids):
            st = self._stats.setdefault(
                tid, {"hits": 0, "conf_sum": 0.0, "max": 0.0, "first_frame": frame_index, "first_ts": timestamp_s}
            )
            st["hits"] += 1
            st["conf_sum"] += det.confidence
            st["max"] = max(st["max"], det.confidence)
            det.track_id = tid
            if st["hits"] < self.min_hits:
                det.status, det.verification = "candidate", "pending_temporal_persistence"
                continue
            det.status, det.verification = "confirmed", self.method
            event = self.events.get(tid)
            if event is None:
                self._counter += 1
                details = dict(det.extra)
                details["confirmed_at_frame"] = frame_index
                details["confirmed_at_timestamp_s"] = round(timestamp_s, 3)
                event = ViolationEvent(
                    event_id=f"{self.category}-{self._counter:04d}",
                    category=self.category,
                    label=det.class_name,
                    verification_method=self.method,
                    track_id=tid,
                    first_frame=st["first_frame"],
                    first_timestamp_s=round(st["first_ts"], 3),
                    last_frame=frame_index,
                    last_timestamp_s=round(timestamp_s, 3),
                    hits=st["hits"],
                    max_confidence=round(st["max"], 4),
                    mean_confidence=round(st["conf_sum"] / st["hits"], 4),
                    bbox=[round(v, 1) for v in det.bbox],
                    details=details,
                )
                self.events[tid] = event
                new_events.append(event)
            else:
                event.last_frame = frame_index
                event.last_timestamp_s = round(timestamp_s, 3)
                event.hits = st["hits"]
                event.max_confidence = round(st["max"], 4)
                event.mean_confidence = round(st["conf_sum"] / st["hits"], 4)
                event.bbox = [round(v, 1) for v in det.bbox]
        return new_events

    def unconfirmed_tracks(self) -> int:
        return sum(1 for tid in self._stats if tid not in self.events)


class BaseDetector(ABC):
    category: str = ""
    verification_description: str = ""

    def __init__(self, registry: ModelRegistry, spec: ModelSpec):
        self.registry = registry
        self.spec = spec
        self.class_counts: Counter = Counter()
        self.notes: List[str] = []

    def predict(self, frame: np.ndarray) -> List[RawDetection]:
        raw = self.registry.predict(self.spec.key, frame)
        for r in raw:
            self.class_counts[r.class_name] += 1
        return raw

    def make_detection(self, r: RawDetection, frame_index: int, timestamp_s: float, role: str = "context") -> Detection:
        return Detection(
            category=self.category,
            source_model=self.spec.key,
            class_id=r.class_id,
            class_name=r.class_name,
            confidence=r.confidence,
            bbox=r.bbox,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            role=role,
        )

    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> DetectorOutput: ...

    @abstractmethod
    def confirmed_events(self) -> List[ViolationEvent]: ...

    @abstractmethod
    def unconfirmed_candidates(self) -> int: ...

    def summarize(self) -> CategorySummary:
        return CategorySummary(
            category=self.category,
            confirmed_violations=len(self.confirmed_events()),
            unconfirmed_candidates=self.unconfirmed_candidates(),
            verification_method=self.verification_description,
            detections_by_class=dict(self.class_counts),
            notes=list(self.notes),
        )
