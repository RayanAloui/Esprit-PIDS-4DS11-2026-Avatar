# ==========================================================
# FILE 3: apps/modeling/video_generation.py
# FULL UPDATED VERSION
# Natural PPTX slide sorting + Exact audio offset syncing + Duplicate fix
# ==========================================================

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from apps.modeling.avatar_renderer import render_sync

MODELING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODELING_DIR.parent.parent

VOICE = "fr-FR-HenriNeural"
import ssl

# Create SSL context that doesn't verify certificates
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# ==========================================================
# PPTX EXPORT
# ==========================================================

def natural_sort_key(s):
    """
    Sorts strings containing numbers logically (e.g., Slide2 comes before Slide10).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def pptx_to_images(pptx_path: Path, out_dir: Path):
    import pythoncom
    import win32com.client
 
    # COM must be initialized on every thread that uses it (thread pool threads are not initialized)
    pythoncom.CoInitialize()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
 
        app = win32com.client.Dispatch("PowerPoint.Application")
        app.Visible = 1
 
        deck = app.Presentations.Open(str(pptx_path), WithWindow=False)
        deck.Export(str(out_dir), "PNG")
        deck.Close()
        app.Quit()
    finally:
        pythoncom.CoUninitialize()
 
    images = []
    for p in out_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            images.append(p)
 
    return sorted(images, key=natural_sort_key)

# ==========================================================
# TTS
# ==========================================================

async def _tts(text: str, dst: str):
    import edge_tts
    t = edge_tts.Communicate(text or ".", VOICE)
    await t.save(dst)

async def texts_to_audio(texts, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []

    for i, txt in enumerate(texts):
        p = out_dir / f"slide_{i:03d}.mp3"
        print(f"Generating audio {i+1}/{len(texts)}")
        await _tts(txt, str(p))
        files.append(p)

    return files

# ==========================================================
# EXACT AUDIO DURATION
# ==========================================================

def get_duration(path: Path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]

    out = subprocess.check_output(cmd)\
        .decode()\
        .strip()

    return float(out)

# ==========================================================
# BUILD SEGMENT
# ==========================================================

def build_segment(
    slide_img: Path,
    audio_path: Path,
    avatar_video: Path,
    avatar_start_time: float,
    duration: float,
    out_path: Path
):
    cmd = [
        "ffmpeg",
        "-y",

        "-loop", "1",
        "-t", str(duration),
        "-i", str(slide_img),

        # Use the computed start time to perfectly bypass the 3D loading phase
        "-ss", str(avatar_start_time),
        "-t", str(duration),
        "-i", str(avatar_video),

        "-i", str(audio_path),

        "-filter_complex",

        "[1:v]"
        "colorkey=0x00FF00:0.35:0.15,"
        "scale=620:-2[av];"

        "[0:v][av]"
        "overlay=W-w+185:H-h+20[v]",

        "-map", "[v]",
        "-map", "2:a",

        "-r", "30",

        "-c:v", "h264_nvenc",
        "-preset", "fast",

        "-c:a", "aac",
        "-shortest",

        str(out_path)
    ]

    subprocess.run(cmd, check=True)

# ==========================================================
# CONCAT
# ==========================================================

def concat_segments(files, output):
    txt = output.parent / "concat.txt"

    lines = [
        f"file '{f.as_posix()}'"
        for f in files
    ]

    txt.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(txt),
        "-c", "copy",
        str(output)
    ]

    subprocess.run(cmd, check=True)

# ==========================================================
# MAIN
# ==========================================================

async def generate_avatar_video(
    pptx_path: Path,
    slide_texts: list[str],
    output_path: Optional[Path] = None
):
    if output_path is None:
        output_path = pptx_path.with_suffix(".mp4")

    tmp = PROJECT_ROOT / "static" / "alia_tmp"

    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    tmp.mkdir(parents=True, exist_ok=True)

    slide_dir = tmp / "slides"
    audio_dir = tmp / "audio"
    seg_dir = tmp / "segments"

    slide_dir.mkdir()
    audio_dir.mkdir()
    seg_dir.mkdir()

    print("Exporting slides...")
    slides = pptx_to_images(
        pptx_path,
        slide_dir
    )

    # Pad texts if the generator missed some narrations (prevents slides from dropping out)
    if len(slide_texts) < len(slides):
        print(f"Padding missing narrations: {len(slides) - len(slide_texts)} extra slides found.")
        while len(slide_texts) < len(slides):
            slide_texts.append(".")  # Brief silent text so TTS still generates a valid mp3

    print("Generating TTS...")
    audios = await texts_to_audio(
        slide_texts,
        audio_dir
    )

    n = min(len(slides), len(audios))

    durations = []

    for a in audios:
        durations.append(get_duration(a))

    total = sum(durations)

    print("Building master audio track for lipsync...")
    master_audio = tmp / "master_audio.mp3"
    concat_txt = tmp / "audio_concat.txt"
    concat_lines = [f"file '{a.resolve().as_posix()}'" for a in audios]
    concat_txt.write_text("\n".join(concat_lines), encoding="utf-8")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        str(master_audio)
    ], check=True)

    avatar_video = tmp / "avatar_master.webm"

    print(f"Rendering one avatar video ({round(total,1)} sec)...")

    # Capture the calculated precise loading offset from Python
    video_start_offset = await render_sync(
        total,
        avatar_video,
        audio_url="/static/alia_tmp/master_audio.mp3"
    )
    print(f"Detected loading phase dead-time: {video_start_offset:.3f}s")

    segments = []
    cursor = 0.0

    for i in range(n):
        print(f"Building segment {i+1}/{n}")
        seg = seg_dir / f"seg_{i:03d}.mp4"

        # Shift the ffmpeg start pointer to skip the green screen loading phase
        avatar_start_time = cursor + video_start_offset

        build_segment(
            slides[i],
            audios[i],
            avatar_video,
            avatar_start_time,
            durations[i],
            seg
        )

        segments.append(seg)
        cursor += durations[i]

    print("Final concatenation...")

    concat_segments(
        segments,
        output_path
    )

    shutil.rmtree(tmp, ignore_errors=True)
    print("Done:", output_path)

    return output_path

# ==========================================================
# DJANGO ENTRY
# ==========================================================

def generate_video_for_product(
    product_name: str,
    csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
):
    from apps.modeling.powerpoint_generation import (
        generate_presentation_for_product_with_narrations,
    )

    pptx, texts = generate_presentation_for_product_with_narrations(
        product_name,
        csv_path,
        output_dir
    )

    return asyncio.run(
        generate_avatar_video(
            pptx,
            texts,
            pptx.with_suffix(".mp4")
        )
    )