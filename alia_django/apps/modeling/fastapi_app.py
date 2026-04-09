"""
FastAPI ALIA RAG API, mounted by Django ASGI (see alia.asgi).
"""
from __future__ import annotations

import traceback
from pathlib import Path

from asgiref.sync import sync_to_async
from django.conf import settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apps.modeling.handlers import ask_alia_json, debug_search_json, health_json
from apps.modeling.rendering import render_modeling_index_sync


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

    @app.post("/ask_alia")
    async def ask_alia_endpoint(query: Query):
        try:
            return await ask_alia_json(query.text)
        except Exception as e:
            print(f"[modeling] ERROR: {type(e).__name__}: {e}")
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

    return app
