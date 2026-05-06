"""
FastAPI ALIA RAG API, mounted by Django ASGI (see alia.asgi).
"""
from __future__ import annotations

import traceback
from pathlib import Path
import requests
from asgiref.sync import sync_to_async
from django.conf import settings
from fastapi import FastAPI, HTTPException, Request
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apps.modeling.metrics import get_system_metrics, log_request_metrics
from time import perf_counter

from apps.modeling.handlers import ask_alia_json, debug_search_json, health_json, listen_json
from apps.modeling.rendering import render_modeling_index_sync
from apps.modeling.runtime import get_runtime


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_root = Path(settings.MODELING_STATIC_DIR)
    (static_root / "audio").mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")

    class Query(BaseModel):
        text: str

    class ModeBody(BaseModel):
        mode: str = "commercial"

    @app.post("/reset")
    async def reset_endpoint():
        rt = get_runtime()
        rt.alia.history = []
        return {"status": "ok"}

    @app.post("/set_mode")
    async def set_mode_endpoint(body: ModeBody):
        if body.mode not in ("commercial", "training"):
            raise HTTPException(status_code=400, detail="Invalid mode")
        rt = get_runtime()
        await sync_to_async(rt.alia.set_mode)(body.mode)
        return {"status": "ok", "mode": body.mode}

    @app.middleware("http")
    async def request_metrics_middleware(request: Request, call_next):
        start = perf_counter()
        response = await call_next(request)
        latency_s = perf_counter() - start
        log_request_metrics(str(request.url.path), request.method, latency_s, response.status_code)
        return response

    @app.post("/ask_alia")
    async def ask_alia_endpoint(request: Request):
        body = await request.body()
        print(f"[DEBUG] Raw request body: {body}")
        print(f"[DEBUG] Content-Type: {request.headers.get('content-type')}")
        
        # Parse manually to debug
        try:
            data = await request.json()
            print(f"[DEBUG] Parsed JSON: {data}")
            print(f"[DEBUG] text value: '{data.get('text')}'")
        except:
            print("[DEBUG] Could not parse as JSON")
        query = Query(text=data.get('text', ''))
        try:
            return await ask_alia_json(query.text)
        except Exception as e:
            print(f"[modeling] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {e!s}"
            ) from e

    @app.post("/listen")
    async def listen_endpoint(audio: UploadFile = File(...)):
        try:
            audio_bytes = await audio.read()
            return await listen_json(audio_bytes)
        except Exception as e:
            print(f"[modeling] LISTEN ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {e!s}"
            ) from e

    @app.get("/")
    async def index():
        resp = await sync_to_async(render_modeling_index_sync)()
        return HTMLResponse(
            content=resp.content.decode("utf-8"),
            media_type="text/html; charset=utf-8",
        )

    @app.get("/health")
    async def health():
        return health_json()

    @app.post("/debug/search")
    async def debug_search(query: Query):
        return debug_search_json(query.text)

    @app.get("/debug/metrics")
    async def debug_metrics():
        return get_system_metrics()

    class VideoBody(BaseModel):
        product_name: str
 
    @app.post("/generate_video")
    async def generate_video_endpoint(body: VideoBody):
        """
        Generate a PPTX + avatar MP4 video for a product.
        Returns a JSON with video_url pointing to the served file,
        and presentation_url pointing to the .pptx file.
        """
        from apps.modeling.video_generation import generate_video_for_product
        import asyncio as _asyncio
        try:
            video_path = await _asyncio.to_thread(
                generate_video_for_product,
                body.product_name,
            )
            # Move outputs to the static directory so they are served
            static_root = Path(settings.MODELING_STATIC_DIR)
            videos_dir  = static_root / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
 
            dest_mp4  = videos_dir / video_path.name
            dest_pptx = videos_dir / video_path.with_suffix(".pptx").name
 
            import shutil
            shutil.copy2(str(video_path), str(dest_mp4))
            if video_path.with_suffix(".pptx").exists():
                shutil.copy2(str(video_path.with_suffix(".pptx")), str(dest_pptx))
 
            prefix = api_prefix()
            return {
                "status": "ok",
                "product": body.product_name,
                "video_url":        f"{prefix}/static/videos/{dest_mp4.name}",
                "presentation_url": f"{prefix}/static/videos/{dest_pptx.name}",
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            print(f"[modeling] VIDEO ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from e
    return app
