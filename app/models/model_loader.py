"""Loads the three independent YOLOv8 models once and exposes thread-safe inference."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from app.config import Settings
from app.models.model_config import ModelSpec

logger = logging.getLogger("civic_mirror.models")


@dataclass
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class ModelRegistry:
    def __init__(self, settings: Settings, specs: Dict[str, ModelSpec]):
        self.settings = settings
        self.specs = specs
        self.device = "cpu"
        self.half = False
        self._models: Dict[str, Any] = {}
        self._names: Dict[str, Dict[int, str]] = {}
        self._errors: Dict[str, str] = {}
        # Ultralytics predictors are not guaranteed to be thread-safe: serialize per model.
        self._locks: Dict[str, threading.Lock] = {k: threading.Lock() for k in specs}

    # ------------------------------------------------------------------ device
    def _resolve_device(self) -> str:
        requested = self.settings.device.strip().lower()
        cuda_ok = torch.cuda.is_available()
        if requested in ("", "auto"):
            return "cuda:0" if cuda_ok else "cpu"
        if requested == "cpu":
            return "cpu"
        if requested.isdigit():
            requested = f"cuda:{requested}"
        if requested.startswith("cuda"):
            if cuda_ok:
                return requested
            logger.warning("DEVICE=%s requested but CUDA is unavailable; falling back to CPU.", requested)
            return "cpu"
        logger.warning("Unknown DEVICE '%s'; falling back to CPU.", requested)
        return "cpu"

    # ------------------------------------------------------------------ loading
    def load_all(self) -> None:
        self.device = self._resolve_device()
        self.half = self.device.startswith("cuda") and self.settings.use_half
        logger.info("Inference device: %s (half precision: %s)", self.device, self.half)

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on environment
            msg = f"Ultralytics could not be imported: {exc}"
            logger.error(msg)
            for key in self.specs:
                self._errors[key] = msg
            return

        for key, spec in self.specs.items():
            if not spec.path.is_file():
                self._errors[key] = f"Model file not found: {spec.path}"
                logger.warning("%s", self._errors[key])
                continue
            try:
                model = YOLO(str(spec.path))
                task = getattr(model, "task", "detect")
                if task not in ("detect", "segment"):
                    raise RuntimeError(f"Unsupported model task '{task}'; a YOLOv8 detection model is required.")
                model.to(self.device)
                # Warm-up: surfaces problems at startup and initialises CUDA kernels.
                dummy = np.zeros((spec.imgsz, spec.imgsz, 3), dtype=np.uint8)
                model.predict(dummy, imgsz=spec.imgsz, device=self.device, half=self.half, verbose=False)
                names = getattr(model, "names", None) or {}
                self._names[key] = {int(k): str(v) for k, v in dict(names).items()}
                self._models[key] = model
                logger.info("Loaded '%s' from %s with classes: %s", key, spec.path, self._names[key])
            except Exception as exc:
                self._errors[key] = f"Failed to load model: {exc}"
                logger.exception("Failed to load model '%s'", key)

    def release(self) -> None:
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ queries
    def is_loaded(self, key: str) -> bool:
        return key in self._models

    def loaded_keys(self) -> List[str]:
        return [k for k in self.specs if k in self._models]

    def error(self, key: str) -> str:
        return self._errors.get(key, "")

    def class_names(self, key: str) -> Dict[int, str]:
        return dict(self._names.get(key, {}))

    # ---------------------------------------------------------------- inference
    def predict(self, key: str, frame: np.ndarray) -> List[RawDetection]:
        if key not in self._models:
            raise RuntimeError(f"Model '{key}' is not loaded")
        spec = self.specs[key]
        with self._locks[key]:
            results = self._models[key].predict(
                source=frame,
                conf=spec.conf,
                iou=spec.iou,
                imgsz=spec.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        names = self._names.get(key, {})
        detections: List[RawDetection] = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
            cid = int(cls)
            detections.append(
                RawDetection(
                    class_id=cid,
                    class_name=names.get(cid, str(cid)),
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections
