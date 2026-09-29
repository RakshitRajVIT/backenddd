"""Civic Mirror backend entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.errors import JobError
from app.models.model_config import load_pipeline_config
from app.models.model_loader import ModelRegistry
from app.routes.health import router as health_router
from app.routes.videos import router as videos_router
from app.schemas.video import ErrorBody, ErrorResponse
from app.services.job_manager import JobManager
from app.services.video_processor import VideoProcessor

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("civic_mirror.main")


@dataclass
class AppContext:
    settings: object
    registry: ModelRegistry
    pipeline_cfg: object
    jobs: JobManager
    processor: VideoProcessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = load_pipeline_config(settings)
    registry = ModelRegistry(settings, pipeline_cfg.specs())
    logger.info("Loading models...")
    registry.load_all()
    loaded = registry.loaded_keys()
    if not loaded:
        logger.warning(
            "No models loaded. Place helmet.pt, tripling.pt, red_light.pt under '%s' "
            "(or set *_MODEL_PATH env vars) and restart. The API will still start; "
            "/api/health reports per-model status and /process will fail until a model loads.",
            settings.models_dir,
        )
    else:
        logger.info("Loaded models: %s", loaded)

    jobs = JobManager()
    processor = VideoProcessor(settings, registry, pipeline_cfg, jobs)
    app.state.ctx = AppContext(settings=settings, registry=registry, pipeline_cfg=pipeline_cfg, jobs=jobs, processor=processor)

    yield

    registry.release()


app = FastAPI(
    title=settings.app_name,
    description="AI-powered traffic violation detection backend (helmet, triple-riding, red-light).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(JobError)
async def job_error_handler(request: Request, exc: JobError):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=ErrorBody(code=exc.code, message=exc.message)).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=ErrorBody(code="internal_error", message="An unexpected error occurred.")).model_dump(),
    )


app.include_router(health_router)
app.include_router(videos_router)


@app.get("/")
def root():
    return {"app": settings.app_name, "docs": "/docs", "health": "/api/health"}
