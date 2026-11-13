# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT.
"""FastAPI server for Hunyuan3D-2.1.

Existing JSON endpoints are retained:
  POST /generate       synchronous generation (texture is controlled by request.texture)
  POST /send           asynchronous generation
  GET  /status/{uid}   task status (legacy base64 result field retained)

New one-image-to-textured-GLB endpoints:
  POST /generate-textured  synchronous Shape -> unload -> Paint -> GLB
  POST /send-textured      asynchronous Shape -> unload -> Paint -> GLB
  GET  /result/{uid}       download final GLB

The new generation endpoints accept either the existing JSON/base64 payload or
raw image bytes (Content-Type: image/png, image/jpeg, etc.).
"""
import argparse
import asyncio
import base64
import logging
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.concurrency import run_in_threadpool

from api_models import GenerationRequest, GenerationResponse, StatusResponse, HealthResponse
from logger_utils import build_logger
from constants import (
    SERVER_ERROR_MSG,
    DEFAULT_SAVE_DIR,
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_CONTACT,
    API_LICENSE_INFO,
    API_TAGS_METADATA,
)
from model_worker import ModelWorker

SAVE_DIR = DEFAULT_SAVE_DIR
worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")
worker = None
model_semaphore = None

# In-memory task metadata. Generated files remain the source of truth for results.
_TASKS: Dict[str, Dict[str, Any]] = {}
_TASKS_LOCK = threading.Lock()

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE_INFO,
    tags_metadata=API_TAGS_METADATA,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_to_dict(model):
    """Support both Pydantic v1 and v2."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _set_task(uid: str, **updates):
    with _TASKS_LOCK:
        task = _TASKS.setdefault(
            uid,
            {
                "status": "processing",
                "stage": "queued",
                "progress": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        task.update(updates)
        task["updated_at"] = datetime.now(timezone.utc).isoformat()


def _get_task(uid: str):
    with _TASKS_LOCK:
        task = _TASKS.get(uid)
        return dict(task) if task else None


def _task_callback(uid: str):
    def callback(stage: str, progress: int):
        status_value = "completed" if stage == "completed" else (
            "texturing" if stage == "paint" else "processing"
        )
        _set_task(uid, stage=stage, progress=int(progress), status=status_value)
    return callback



async def _read_payload(request: Request, force_texture: bool = False):
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
            payload = GenerationRequest(**body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON generation request: {exc}") from exc
        params = _model_to_dict(payload)
    else:
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="Request body is empty; upload an image")
        params = _model_to_dict(
            GenerationRequest(image=base64.b64encode(raw).decode("ascii"))
        )
    if force_texture:
        params["texture"] = True
    return params


def _run_background(uid: str, params: dict):
    """Background thread target for /send and /send-textured."""
    _set_task(uid, status="processing", stage="queued", progress=0)
    params["_status_callback"] = _task_callback(uid)
    try:
        file_path, _ = worker.generate(uid, params)
        _set_task(
            uid,
            status="completed",
            stage="completed",
            progress=100,
            result_path=file_path,
        )
    except Exception as exc:
        logger.exception("Background generation failed for %s", uid)
        _set_task(uid, status="error", stage="error", error=str(exc))


def _result_path(uid: str):
    # Prefer the textured file; otherwise the legacy Shape-only GLB.
    textured = os.path.join(SAVE_DIR, f"{uid}_textured.glb")
    initial = os.path.join(SAVE_DIR, f"{uid}_initial.glb")
    if os.path.isfile(textured):
        return textured
    if os.path.isfile(initial):
        return initial
    return None


@app.post("/generate", tags=["generation"])
async def generate_3d_model(request: GenerationRequest):
    """Legacy synchronous endpoint. Set texture=true to run both stages."""
    uid = str(uuid.uuid4())
    params = _model_to_dict(request)
    _set_task(uid, status="processing", stage="queued", progress=0)
    params["_status_callback"] = _task_callback(uid)
    try:
        file_path, _ = await run_in_threadpool(worker.generate, uid, params)
        _set_task(uid, status="completed", stage="completed", progress=100, result_path=file_path)
        return FileResponse(file_path, filename=os.path.basename(file_path))
    except ValueError as exc:
        _set_task(uid, status="error", stage="error", error=str(exc))
        logger.exception("Generation ValueError")
        return JSONResponse({"text": SERVER_ERROR_MSG, "error_code": 1, "detail": str(exc)}, status_code=400)
    except Exception as exc:
        _set_task(uid, status="error", stage="error", error=str(exc))
        logger.exception("Generation failed")
        return JSONResponse({"text": SERVER_ERROR_MSG, "error_code": 1, "detail": str(exc)}, status_code=500)


@app.post("/generate-textured", tags=["generation"])
async def generate_textured_model(request: Request):
    """Upload one image and synchronously return the final textured GLB.

    Accepts either raw image bytes or the original JSON/base64 GenerationRequest.
    """
    uid = str(uuid.uuid4())
    try:
        params = await _read_payload(request, force_texture=True)
        params["_status_callback"] = _task_callback(uid)
        _set_task(uid, status="processing", stage="queued", progress=0)
        file_path, _ = await run_in_threadpool(worker.generate, uid, params)
        _set_task(uid, status="completed", stage="completed", progress=100, result_path=file_path)
        return FileResponse(
            file_path,
            media_type="model/gltf-binary",
            filename=f"{uid}.glb",
        )
    except HTTPException:
        raise
    except ValueError as exc:
        _set_task(uid, status="error", stage="error", error=str(exc))
        logger.exception("Textured generation failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _set_task(uid, status="error", stage="error", error=str(exc))
        logger.exception("Textured generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/send", response_model=GenerationResponse, tags=["generation"])
async def send_generation_task(request: GenerationRequest):
    """Legacy asynchronous endpoint."""
    uid = str(uuid.uuid4())
    params = _model_to_dict(request)
    _set_task(uid, status="processing", stage="queued", progress=0)
    try:
        thread = threading.Thread(target=_run_background, args=(uid, params), daemon=True)
        thread.start()
        return JSONResponse({"uid": uid}, status_code=200)
    except Exception as exc:
        logger.exception("Failed to start generation thread")
        _set_task(uid, status="error", stage="error", error=str(exc))
        return JSONResponse({"error": "Failed to start generation", "detail": str(exc)}, status_code=500)


@app.post("/send-textured", response_model=GenerationResponse, tags=["generation"])
async def send_textured_generation_task(request: Request):
    """Asynchronously run Shape -> unload -> Paint -> final GLB."""
    uid = str(uuid.uuid4())
    params = await _read_payload(request, force_texture=True)
    _set_task(uid, status="processing", stage="queued", progress=0)
    try:
        thread = threading.Thread(target=_run_background, args=(uid, params), daemon=True)
        thread.start()
        return JSONResponse({"uid": uid}, status_code=200)
    except Exception as exc:
        logger.exception("Failed to start textured generation thread")
        _set_task(uid, status="error", stage="error", error=str(exc))
        return JSONResponse({"error": "Failed to start textured generation", "detail": str(exc)}, status_code=500)


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    return JSONResponse({"status": "healthy", "worker_id": worker_id}, status_code=200)


@app.get("/status/{uid}", response_model=StatusResponse, tags=["status"])
async def status(uid: str):
    """Task state; model_base64 is preserved for existing clients."""
    task = _get_task(uid)
    if task and task.get("status") == "error":
        return JSONResponse(
            {
                "status": "error",
                "message": task.get("error", "Generation failed"),
                "stage": task.get("stage", "error"),
                "progress": task.get("progress", 0),
            },
            status_code=200,
        )

    path = _result_path(uid)
    if task and task.get("status") == "completed" and path:
        try:
            with open(path, "rb") as file:
                encoded = base64.b64encode(file.read()).decode("ascii")
            return JSONResponse(
                {
                    "status": "completed",
                    "model_base64": encoded,
                    "stage": "completed",
                    "progress": 100,
                    "result_url": f"/result/{uid}",
                }
            )
        except Exception as exc:
            logger.exception("Could not read result for %s", uid)
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)

    if task:
        return JSONResponse(
            {
                "status": task.get("status", "processing"),
                "stage": task.get("stage", "queued"),
                "progress": task.get("progress", 0),
            }
        )

    # Backward-compatible file-based fallback for tasks created before server restart.
    if path:
        with open(path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("ascii")
        return JSONResponse(
            {
                "status": "completed",
                "model_base64": encoded,
                "stage": "completed",
                "progress": 100,
                "result_url": f"/result/{uid}",
            }
        )
    initial_path = os.path.join(SAVE_DIR, f"{uid}_initial.glb")
    if os.path.isfile(initial_path):
        return JSONResponse({"status": "texturing", "stage": "paint", "progress": 55})
    return JSONResponse({"status": "processing", "stage": "shape", "progress": 0})


@app.get("/result/{uid}", tags=["generation"])
async def download_result(uid: str):
    """Download the completed GLB without embedding it in a JSON response."""
    task = _get_task(uid)
    if task and task.get("status") == "error":
        raise HTTPException(status_code=409, detail=task.get("error", "Generation failed"))
    path = _result_path(uid)
    if not path:
        raise HTTPException(status_code=404, detail="Result is not ready")
    if task and task.get("status") not in ("completed", None):
        raise HTTPException(status_code=409, detail="Generation is still running")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{uid}.glb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mc_algo", type=str, default="mc")
    parser.add_argument("--limit-model-concurrency", type=int, default=1)
    parser.add_argument("--enable_flashvdm", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument("--cache-path", type=str, default="./gradio_cache")
    parser.add_argument("--paint-max-num-view", type=int, default=4)
    parser.add_argument("--paint-resolution", type=int, default=512)
    args = parser.parse_args()
    logger.info("args: %s", args)

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(
        model_path=args.model_path,
        subfolder=args.subfolder,
        device=args.device,
        low_vram_mode=args.low_vram_mode,
        worker_id=worker_id,
        model_semaphore=model_semaphore,
        save_dir=SAVE_DIR,
        mc_algo=args.mc_algo,
        enable_flashvdm=args.enable_flashvdm,
        compile=args.compile,
        paint_max_num_view=args.paint_max_num_view,
        paint_resolution=args.paint_resolution,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
