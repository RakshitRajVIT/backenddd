"""Application settings, loaded from environment variables (a .env file is supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got '{raw}'") from exc


def env_float(name: str, default: float) -> float:
    raw = env_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got '{raw}'") from exc


def env_list(name: str, default: str = "") -> List[str]:
    return [item.strip() for item in env_str(name, default).split(",") if item.strip()]


def env_path(name: str, default: str) -> Path:
    path = Path(env_str(name, default))
    return path if path.is_absolute() else (BACKEND_DIR / path).resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str
    log_level: str
    upload_dir: Path
    processed_dir: Path
    models_dir: Path
    max_upload_mb: int
    allowed_extensions: List[str]
    cors_origins: List[str]
    device: str
    use_half: bool
    frame_stride: int
    max_concurrent_jobs: int
    require_all_models: bool
    output_codec: str
    reencode_h264: bool
    ffmpeg_path: str
    draw_context_objects: bool

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    extensions = [
        e.lower() if e.startswith(".") else f".{e.lower()}"
        for e in env_list("ALLOWED_EXTENSIONS", ".mp4,.avi,.mov,.mkv,.webm,.m4v")
    ]
    return Settings(
        app_name=env_str("APP_NAME", "Civic Mirror"),
        log_level=env_str("LOG_LEVEL", "INFO").upper(),
        upload_dir=env_path("UPLOAD_DIR", "uploads"),
        processed_dir=env_path("PROCESSED_DIR", "processed"),
        models_dir=env_path("MODELS_DIR", "models"),
        max_upload_mb=max(1, env_int("MAX_UPLOAD_MB", 500)),
        allowed_extensions=extensions,
        cors_origins=env_list("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"),
        device=env_str("DEVICE", "auto"),
        use_half=env_bool("USE_HALF", True),
        frame_stride=max(1, env_int("FRAME_STRIDE", 1)),
        max_concurrent_jobs=max(1, env_int("MAX_CONCURRENT_JOBS", 1)),
        require_all_models=env_bool("REQUIRE_ALL_MODELS", False),
        output_codec=env_str("OUTPUT_CODEC", "mp4v"),
        reencode_h264=env_bool("REENCODE_H264", True),
        ffmpeg_path=env_str("FFMPEG_PATH", "ffmpeg"),
        draw_context_objects=env_bool("DRAW_CONTEXT_OBJECTS", True),
    )
