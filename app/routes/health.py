from __future__ import annotations

import shutil

import torch
from fastapi import APIRouter, Request

from app.schemas.video import ApiResponse, HealthData, ModelStatus

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=ApiResponse[HealthData])
def health(request: Request):
    ctx = request.app.state.ctx
    settings, registry, pipeline_cfg, jobs = ctx.settings, ctx.registry, ctx.pipeline_cfg, ctx.jobs

    models = {}
    warnings = []
    specs = pipeline_cfg.specs()
    for key, spec in specs.items():
        loaded = registry.is_loaded(key)
        models[key] = ModelStatus(
            loaded=loaded,
            path=str(spec.path),
            path_exists=spec.path.is_file(),
            classes=registry.class_names(key) if loaded else {},
            conf_threshold=spec.conf,
            error=registry.error(key) or None,
        )
        if not loaded:
            warnings.append(f"Model '{key}' is not loaded: {registry.error(key) or 'unknown reason'}")

    roles = pipeline_cfg.role_summary()
    if not pipeline_cfg.helmet.violation.configured:
        warnings.append("HELMET_VIOLATION_CLASSES not configured; helmet violations cannot be confirmed.")
    if not pipeline_cfg.tripling.violation.configured and not (
        pipeline_cfg.tripling.person.configured and pipeline_cfg.tripling.motorcycle.configured
    ):
        warnings.append("Tripling classes not configured; tripling violations cannot be confirmed.")
    if not pipeline_cfg.red_light.stop_line:
        warnings.append("RED_LIGHT_STOP_LINE not configured; red-light violations cannot be confirmed.")

    data = HealthData(
        status="ok" if models else "degraded",
        app=settings.app_name,
        device=registry.device,
        cuda_available=torch.cuda.is_available(),
        half_precision=registry.half,
        torch_version=torch.__version__,
        ffmpeg_available=shutil.which(settings.ffmpeg_path) is not None,
        max_upload_mb=settings.max_upload_mb,
        models=models,
        class_roles=roles,
        jobs=jobs.counts(),
        warnings=warnings,
    )
    return ApiResponse(message="ok", data=data)
