"""Video I/O, geometry, tracking and annotation helpers."""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.schemas.video import Detection, VideoInfo

logger = logging.getLogger("civic_mirror.video")
Box = Tuple[float, float, float, float]

# ------------------------------------------------------------------- file helpers


def safe_unlink(path: Path) -> bool:
    for _ in range(5):
        try:
            if path.exists():
                path.unlink()
            return True
        except PermissionError:
            time.sleep(0.2)  # Windows may briefly hold file handles
        except OSError:
            return False
    return False


def safe_rmtree(path: Path) -> bool:
    for _ in range(5):
        try:
            if path.exists():
                shutil.rmtree(path)
            return True
        except (PermissionError, OSError):
            time.sleep(0.2)
    return not path.exists()


# ------------------------------------------------------------------- video probing


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("File could not be opened as a video (unsupported or corrupted).")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Video contains no decodable frames.")
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]
        fps_assumed = False
        if not math.isfinite(fps) or fps < 1e-3 or fps > 240:
            fps, fps_assumed = 25.0, True
        frames = max(frames, 0)
        return VideoInfo(
            width=width,
            height=height,
            fps=round(fps, 4),
            frame_count=frames,
            duration_s=round(frames / fps, 3) if frames > 0 else None,
            fps_assumed=fps_assumed,
        )
    finally:
        cap.release()


def open_writer(path: Path, fps: float, size: Tuple[int, int], codecs: Iterable[str]):
    """Try each fourcc in order and return (writer, codec_used)."""
    tried: List[str] = []
    for codec in dict.fromkeys(codecs):
        if len(codec) != 4:
            continue
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
        tried.append(codec)
    raise RuntimeError(f"Could not open a video writer (tried codecs: {', '.join(tried) or 'none'}).")


def reencode_h264(src: Path, dst: Path, ffmpeg_path: str) -> Tuple[bool, str]:
    """Re-encode to browser-playable H.264 with ffmpeg. Returns (success, message)."""
    exe = shutil.which(ffmpeg_path)
    if not exe:
        return False, "ffmpeg not found on PATH; output left as OpenCV-encoded MP4."
    cmd = [
        exe, "-y", "-loglevel", "error", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"ffmpeg re-encode failed: {exc}"
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        safe_unlink(dst)
        return False, f"ffmpeg re-encode failed: {proc.stderr.strip()[:300]}"
    return True, "Re-encoded to H.264."


# ------------------------------------------------------------------------ geometry


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_area(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: Sequence[Box]) -> Box:
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def side_of_line(px: float, py: float, a: Tuple[float, float], b: Tuple[float, float]) -> int:
    """+1 / -1 / 0: which side of the directed line a->b the point lies on (image coordinates)."""
    cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
    return 1 if cross > 0 else (-1 if cross < 0 else 0)


def projection_on_segment(px: float, py: float, a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Parameter t of the point's projection on segment a->b (0 at a, 1 at b)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    return ((px - a[0]) * dx + (py - a[1]) * dy) / length_sq if length_sq > 0 else 0.0


# ------------------------------------------------------------------------- tracker


@dataclass
class _Track:
    track_id: int
    bbox: Box
    missed: int = 0
    hits: int = 1


class IoUTracker:
    """Lightweight greedy IoU tracker (class-agnostic). max_age counts update() calls."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def update(self, boxes: Sequence[Box]) -> List[int]:
        ids = [-1] * len(boxes)
        pairs = []
        for tid, track in self.tracks.items():
            for bi, box in enumerate(boxes):
                score = iou(track.bbox, box)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, bi))
        pairs.sort(reverse=True)
        used_t, used_b = set(), set()
        for _, tid, bi in pairs:
            if tid in used_t or bi in used_b:
                continue
            used_t.add(tid)
            used_b.add(bi)
            track = self.tracks[tid]
            track.bbox, track.missed, track.hits = tuple(boxes[bi]), 0, track.hits + 1
            ids[bi] = tid
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_age:
                    del self.tracks[tid]
        for bi, box in enumerate(boxes):
            if bi not in used_b:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = _Track(tid, tuple(box))
                ids[bi] = tid
        return ids


# ---------------------------------------------------------------------- annotation

FONT = cv2.FONT_HERSHEY_SIMPLEX
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {  # BGR
    "helmet": (0, 140, 255),     # orange
    "tripling": (255, 0, 200),   # magenta
    "red_light": (0, 0, 255),    # red
}
CATEGORY_TITLES = {"helmet": "NO HELMET", "tripling": "TRIPLE RIDING", "red_light": "RED-LIGHT VIOLATION"}
CATEGORY_HUD = {"helmet": "Helmet", "tripling": "Tripling", "red_light": "Red-light"}


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes):02d}:{secs:05.2f}"


def _context_color(category: str) -> Tuple[int, int, int]:
    base = CATEGORY_COLORS.get(category, (200, 200, 200))
    return tuple(int(0.35 * c + 0.65 * 150) for c in base)  # muted version of the category color


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], scale: float = 0.5) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    top = y - th - base - 4
    if top < 0:
        top = y + 2
    fg = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)
    cv2.rectangle(frame, (x, top), (x + tw + 6, top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x + 3, top + th + 1), FONT, scale, fg, 1, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[Detection],
    frame_index: int,
    timestamp_s: float,
    confirmed_counts: Dict[str, int],
    signal_state: Optional[str],
    stop_line: Optional[Tuple[float, float, float, float]],
    draw_context: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    thick = max(2, int(round(min(h, w) / 360)))
    scale = max(0.45, min(h, w) / 1200)

    if stop_line is not None:
        p1 = (int(stop_line[0] * w), int(stop_line[1] * h))
        p2 = (int(stop_line[2] * w), int(stop_line[3] * h))
        cv2.line(frame, p1, p2, (0, 255, 255), thick, cv2.LINE_AA)
        _draw_label(frame, "STOP LINE", p1[0], p1[1], (0, 255, 255), scale)

    ordered = sorted(detections, key=lambda d: {"context": 0, "candidate": 1, "confirmed": 2}.get(
        d.status if d.role == "violation" else "context", 0))
    for det in ordered:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        if det.role != "violation":
            if not draw_context:
                continue
            color = _context_color(det.category)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            _draw_label(frame, f"{det.class_name} {det.confidence:.2f}", x1, y1, color, scale * 0.85)
            continue
        color = CATEGORY_COLORS.get(det.category, (0, 255, 0))
        title = CATEGORY_TITLES.get(det.category, det.category.upper())
        tid = f" #{det.track_id}" if det.track_id is not None else ""
        extra = f" ({det.extra['rider_count']} riders)" if "rider_count" in det.extra else ""
        if det.status == "confirmed":
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick + 1)
            label = f"{title}{extra}{tid} {det.confidence:.2f}"
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thick - 1))
            label = f"{title}? unverified{extra}{tid} {det.confidence:.2f}"
        _draw_label(frame, label, x1, y1, color, scale)

    # HUD
    lines = [f"t={format_timestamp(timestamp_s)}  frame {frame_index}"]
    lines.append("  ".join(f"{CATEGORY_HUD[k]}: {v}" for k, v in confirmed_counts.items()) + "  (confirmed)")
    if signal_state is not None:
        lines.append(f"Signal: {signal_state.upper()}")
    y = int(22 * scale * 1.6)
    for i, text in enumerate(lines):
        color = (0, 0, 255) if text.startswith("Signal: RED") else (255, 255, 255)
        (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
        cv2.rectangle(frame, (6, y - th - 4), (12 + tw, y + base), (0, 0, 0), -1)
        cv2.putText(frame, text, (9, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += int(th + base + 10)
    return frame
