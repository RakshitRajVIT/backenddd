"""Red-light jumping detector.

A confirmed violation requires ALL of:
  1. The traffic signal is currently RED (majority vote over a sliding window
     of recent frames, from RED_LIGHT_RED_SIGNAL_CLASSES).
  2. A tracked vehicle (RED_LIGHT_VEHICLE_CLASSES) crosses the configured
     stop line (RED_LIGHT_STOP_LINE), in the configured direction if any.
  3. The vehicle's track has been observed for >= min_track_frames.

If the model additionally emits a direct "violation" class
(RED_LIGHT_VIOLATION_CLASSES), that is surfaced only as an unverified
candidate annotation and is never auto-confirmed, since a single detector
output cannot itself prove the signal state + line crossing.

If RED_LIGHT_STOP_LINE is not configured, no violation can ever be confirmed
here (by design) - the detector still runs and reports raw detections.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from app.models.model_config import RedLightConfig
from app.models.model_loader import ModelRegistry, RawDetection
from app.schemas.video import Detection, DetectorOutput, ViolationEvent
from app.services.base_detector import BaseDetector, PersistenceVerifier
from app.utils.video_utils import IoUTracker, projection_on_segment, side_of_line


class RedLightDetector(BaseDetector):
    category = "red_light"
    verification_description = "signal_state + stop_line_crossing (tracked vehicle)"

    def __init__(self, registry: ModelRegistry, config: RedLightConfig):
        super().__init__(registry, config.spec)
        self.config = config
        self.vehicle_tracker = IoUTracker(config.track_iou, config.track_max_age)
        self.verifier = PersistenceVerifier(
            category=self.category,
            method=self.verification_description,
            min_hits=1,  # confirmation already requires signal + crossing; no extra frame-count gate
            track_iou=config.track_iou,
            max_age=config.track_max_age,
        )
        self._signal_window: deque = deque(maxlen=config.signal_window)
        self._vehicle_state: Dict[int, dict] = {}

        if not config.stop_line:
            self.notes.append(
                "RED_LIGHT_STOP_LINE is not configured: no red-light violation can be confirmed. "
                "Set it to 'x1,y1,x2,y2' (normalized 0..1) to enable confirmation."
            )
        if not config.red_signal.configured:
            self.notes.append(
                "RED_LIGHT_RED_SIGNAL_CLASSES is not configured: signal state cannot be determined, "
                "so no violation can be confirmed."
            )
        if not config.vehicle.configured:
            self.notes.append(
                "RED_LIGHT_VEHICLE_CLASSES is not configured: vehicles cannot be tracked, "
                "so no violation can be confirmed."
            )
        if config.violation.configured:
            self.notes.append(
                "RED_LIGHT_VIOLATION_CLASSES is configured: those detections are drawn as UNVERIFIED "
                "candidates only (signal-state + stop-line crossing is required for confirmation)."
            )

    def _current_signal(self) -> Optional[str]:
        if not self._signal_window:
            return None
        red_votes = sum(1 for v in self._signal_window if v)
        if red_votes >= self.config.signal_min_red:
            return "red"
        return "red" if red_votes == len(self._signal_window) and red_votes > 0 else "not_red"

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> DetectorOutput:
        raw = self.predict(frame)
        detections: List[Detection] = [self.make_detection(r, frame_index, timestamp_s) for r in raw]

        # 1) signal state
        is_red_frame = any(self.config.red_signal.matches(r.class_id, r.class_name) for r in raw)
        if self.config.red_signal.configured:
            self._signal_window.append(is_red_frame)
        signal_state = self._current_signal() if self.config.red_signal.configured else None

        # 2) unverified direct-violation-class candidates (never auto-confirmed)
        for r in raw:
            if self.config.violation.configured and self.config.violation.matches(r.class_id, r.class_name):
                det = self.make_detection(r, frame_index, timestamp_s, role="violation")
                det.status, det.verification = "candidate", "model_class_only_not_confirmable"
                detections.append(det)

        # 3) vehicle tracking + stop-line crossing, gated on signal == red
        vehicles = [r for r in raw if self.config.vehicle.matches(r.class_id, r.class_name)]
        candidates: List[Detection] = []
        if vehicles:
            ids = self.vehicle_tracker.update([v.bbox for v in vehicles])
            for v, tid in zip(vehicles, ids):
                state = self._vehicle_state.setdefault(tid, {"frames": 0, "crossed": False})
                state["frames"] += 1
                if self.config.stop_line and not state["crossed"]:
                    x1, y1, x2, y2 = v.bbox
                    cx, cy = (x1 + x2) / 2.0, y2  # front/base of the vehicle bbox
                    h_ref, w_ref = frame.shape[0], frame.shape[1]
                    a = (self.config.stop_line[0] * w_ref, self.config.stop_line[1] * h_ref)
                    b = (self.config.stop_line[2] * w_ref, self.config.stop_line[3] * h_ref)
                    t = projection_on_segment(cx, cy, a, b)
                    if -self.config.line_margin <= t <= 1 + self.config.line_margin:
                        side = side_of_line(cx, cy, a, b)
                        prev_side = state.get("side")
                        state["side"] = side
                        crossed_now = prev_side is not None and prev_side != side and side != 0
                        direction_ok = True
                        if crossed_now and self.config.crossing_direction != "any":
                            direction_ok = (
                                (self.config.crossing_direction == "positive_to_negative" and prev_side > 0 and side < 0)
                                or (self.config.crossing_direction == "negative_to_positive" and prev_side < 0 and side > 0)
                            )
                        if crossed_now and direction_ok and state["frames"] >= self.config.min_track_frames:
                            if signal_state == "red":
                                state["crossed"] = True
                                det = self.make_detection(v, frame_index, timestamp_s, role="violation")
                                det.track_id = tid
                                det.class_name = f"{v.class_name} (red-light jump)"
                                det.extra["signal_state_at_crossing"] = "red"
                                candidates.append(det)
                    else:
                        state["side"] = state.get("side")
        detections.extend(candidates)
        for c in candidates:
            c.status, c.verification = "candidate", "pending_confirmation"
        events = self.verifier.update(candidates, frame_index, timestamp_s)
        return DetectorOutput(
            detections=detections, events=events, signal_state=signal_state, stop_line=self.config.stop_line
        )

    def confirmed_events(self) -> List[ViolationEvent]:
        return list(self.verifier.events.values())

    def unconfirmed_candidates(self) -> int:
        return self.verifier.unconfirmed_tracks()
