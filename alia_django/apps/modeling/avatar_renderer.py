# ==========================================================
# FILE 1: apps/modeling/avatar_renderer.py
# Render ONE long talking avatar video (transparent greenscreen)
# ==========================================================

from __future__ import annotations

import asyncio
import http.server
import socketserver
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SERVER_PORT = 8765

def start_server():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def log_message(self, format, *args):
            pass

    class Reusable(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = Reusable(("127.0.0.1", SERVER_PORT), Handler)

    thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True
    )
    thread.start()

    return httpd

async def render_avatar_video(
    seconds: float,
    out_path: Path,
    audio_url: str = ""
) -> float:
    from playwright.async_api import async_playwright

    server = start_server()
    offset_sec = 0.0

    try:
        async with async_playwright() as p:

            browser = await p.chromium.launch(
                headless=True,
                args=["--use-gl=angle", "--enable-gpu"]
            )

            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=str(out_path.parent),
                record_video_size={"width": 1280, "height": 720}
            )

            # Playwright starts recording natively the moment the page context is created.
            t0 = time.time()
            page = await context.new_page()

            url = f"http://127.0.0.1:{SERVER_PORT}/apps/modeling/render_avatar.html"
            if audio_url:
                url += f"?audio={audio_url}"

            await page.goto(url)

            await page.wait_for_function(
                "() => window.avatarReady === true",
                timeout=120000
            )

            # Mark exactly when the avatar begins moving its lips
            await page.evaluate("window.startTalk()")
            t1 = time.time()
            
            # The exact dead time recorded at the start of the video
            offset_sec = t1 - t0

            await page.wait_for_timeout(
                int(seconds * 1000)
            )

            await context.close()
            await browser.close()

            raw = max(
                out_path.parent.glob("*.webm"),
                key=lambda x: x.stat().st_mtime
            )

            raw.rename(out_path)

    finally:
        server.shutdown()
        server.server_close()
        
    return offset_sec

async def render_sync(seconds: float, out_path: Path, audio_url: str = "") -> float:
    return await render_avatar_video(seconds, out_path, audio_url)