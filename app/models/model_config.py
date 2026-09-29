"""Per-model configuration: paths, thresholds and class-role mapping.

Class names are NEVER assumed. Each detector receives explicit, user-configured
lists of class names (or numeric class ids) for the roles it needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import BACKEND_DIR, Settings, env_float, env_int, env_list, env_str


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


class ClassMatcher:
    """Matches a detection's class against configured names / numeric ids.

    Names are compared case-insensitively ignoring spaces, underscores and dashes,
    so "No Helmet", "no_helmet" and "no-helmet" are equivalent.
    """

    def __init__(self, entries: Sequence[str]):
        self.entries: List[str] = [e.strip() for e in entries if e.strip()]
        self._ids = set()
        self._names = set()
        for entry in self.entries:
            if entry.lstrip("-").isdigit():
                self._ids.add(int(entry))
            else:
                self._names.add(_norm(entry))

    @property
    def configured(self) -> bool:
        return bool(self._ids or self._names)

    def matches(self, class_id: int, class_name: str) -> bool:
        return class_id in self._ids or _norm(class_name) in self._names


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    path: Path
    conf: float
    iou: float
    imgsz: int


@dataclass(frozen=True)
class HelmetConfig:
    spec: ModelSpec
    violation: ClassMatcher
    min_hits: int
    track_iou: float
    track_max_age: int


@dataclass(frozen=True)
class TriplingConfig:
    spec: ModelSpec
    violation: ClassMatcher
    person: ClassMatcher
    motorcycle: ClassMatcher
    min_riders: int
    assoc_overlap: float
    expand_x: float
    expand_up: float
    min_hits: int
    track_iou: float
    track_max_age: int


@dataclass(frozen=True)
class RedLightConfig:
    spec: ModelSpec
    violation: ClassMatcher
    vehicle: ClassMatcher
    red_signal: ClassMatcher
    stop_line: Optional[Tuple[float, float, float, float]]
    crossing_direction: str
    signal_window: int
    signal_min_red: int
    min_track_frames: int
    line_margin: float
    track_iou: float
    track_max_age: int


@dataclass(frozen=True)
class PipelineConfig:
    helmet: HelmetConfig
    tripling: TriplingConfig
    red_light: RedLightConfig

    def specs(self) -> Dict[str, ModelSpec]:
        return {
            "helmet": self.helmet.spec,
            "tripling": self.tripling.spec,
            "red_light": self.red_light.spec,
        }

    def role_summary(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "helmet": {"violation": self.helmet.violation.entries},
            "tripling": {
                "violation": self.tripling.violation.entries,
                "person": self.tripling.person.entries,
                "motorcycle": self.tripling.motorcycle.entries,
            },
            "red_light": {
                "violation": self.red_light.violation.entries,
                "vehicle": self.red_light.vehicle.entries,
                "red_signal": self.red_light.red_signal.entries,
            },
        }


def _spec(settings: Settings, key: str, prefix: str, display: str, filename: str) -> ModelSpec:
    raw = env_str(f"{prefix}_MODEL_PATH", "")
    if raw:
        path = Path(raw)
        path = path if path.is_absolute() else (BACKEND_DIR / path).resolve()
    else:
        path = settings.models_dir / filename
    return ModelSpec(
        key=key,
        display_name=display,
        path=path,
        conf=env_float(f"{prefix}_CONF", 0.40),
        iou=env_float(f"{prefix}_IOU", 0.45),
        imgsz=env_int(f"{prefix}_IMGSZ", 640),
    )


def _parse_stop_line(raw: str) -> Optional[Tuple[float, float, float, float]]:
    if not raw:
        return None
    try:
        values = [float(v) for v in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"RED_LIGHT_STOP_LINE must be four comma-separated numbers, got '{raw}'") from exc
    if len(values) != 4 or any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError("RED_LIGHT_STOP_LINE must be 'x1,y1,x2,y2' with every value in the range 0..1")
    if values[0] == values[2] and values[1] == values[3]:
        raise ValueError("RED_LIGHT_STOP_LINE endpoints must be different points")
    return (values[0], values[1], values[2], values[3])


def load_pipeline_config(settings: Settings) -> PipelineConfig:
    helmet = HelmetConfig(
        spec=_spec(settings, "helmet", "HELMET", "Helmet violation", "helmet.pt"),
        violation=ClassMatcher(env_list("HELMET_VIOLATION_CLASSES")),
        min_hits=max(1, env_int("HELMET_MIN_HITS", 3)),
        track_iou=env_float("HELMET_TRACK_IOU", 0.30),
        track_max_age=max(1, env_int("HELMET_TRACK_MAX_AGE", 15)),
    )
    tripling = TriplingConfig(
        spec=_spec(settings, "tripling", "TRIPLING", "Triple riding", "tripling.pt"),
        violation=ClassMatcher(env_list("TRIPLING_VIOLATION_CLASSES")),
        person=ClassMatcher(env_list("TRIPLING_PERSON_CLASSES")),
        motorcycle=ClassMatcher(env_list("TRIPLING_MOTORCYCLE_CLASSES")),
        min_riders=max(2, env_int("TRIPLING_MIN_RIDERS", 3)),
        assoc_overlap=env_float("TRIPLING_ASSOC_OVERLAP", 0.50),
        expand_x=env_float("TRIPLING_MOTO_EXPAND_X", 0.15),
        expand_up=env_float("TRIPLING_MOTO_EXPAND_UP", 0.80),
        min_hits=max(1, env_int("TRIPLING_MIN_HITS", 3)),
        track_iou=env_float("TRIPLING_TRACK_IOU", 0.30),
        track_max_age=max(1, env_int("TRIPLING_TRACK_MAX_AGE", 15)),
    )
    window = max(1, env_int("RED_LIGHT_SIGNAL_WINDOW", 5))
    direction = env_str("RED_LIGHT_CROSSING_DIRECTION", "any").lower()
    if direction not in {"any", "positive_to_negative", "negative_to_positive"}:
        raise ValueError("RED_LIGHT_CROSSING_DIRECTION must be any, positive_to_negative or negative_to_positive")
    red_light = RedLightConfig(
        spec=_spec(settings, "red_light", "RED_LIGHT", "Red-light violation", "red_light.pt"),
        violation=ClassMatcher(env_list("RED_LIGHT_VIOLATION_CLASSES")),
        vehicle=ClassMatcher(env_list("RED_LIGHT_VEHICLE_CLASSES")),
        red_signal=ClassMatcher(env_list("RED_LIGHT_RED_SIGNAL_CLASSES")),
        stop_line=_parse_stop_line(env_str("RED_LIGHT_STOP_LINE", "")),
        crossing_direction=direction,
        signal_window=window,
        signal_min_red=max(1, min(window, env_int("RED_LIGHT_SIGNAL_MIN_RED", 3))),
        min_track_frames=max(1, env_int("RED_LIGHT_MIN_TRACK_FRAMES", 3)),
        line_margin=env_float("RED_LIGHT_LINE_MARGIN", 0.10),
        track_iou=env_float("RED_LIGHT_TRACK_IOU", 0.30),
        track_max_age=max(1, env_int("RED_LIGHT_TRACK_MAX_AGE", 15)),
    )
    return PipelineConfig(helmet=helmet, tripling=tripling, red_light=red_light)
