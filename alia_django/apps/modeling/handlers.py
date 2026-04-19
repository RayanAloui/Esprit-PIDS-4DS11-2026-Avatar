"""Shared ALIA RAG HTTP logic (FastAPI + Django WSGI)."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from django.conf import settings

from apps.modeling.runtime import get_runtime

_whisper_model = None
_whisper_lock = threading.Lock()
_ffmpeg_checked = False


def _ensure_ffmpeg_available() -> None:
    """
    Ensure ffmpeg is available for Whisper audio decoding.
    On Windows, use imageio-ffmpeg as a fallback when ffmpeg is not in PATH.
    """
    global _ffmpeg_checked
    if _ffmpeg_checked:
        return

    if shutil.which("ffmpeg"):
        _ffmpeg_checked = True
        return

    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ffmpeg introuvable. Installez ffmpeg (PATH) ou `pip install imageio-ffmpeg`."
        ) from e

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = str(Path(ffmpeg_exe).parent)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg reste introuvable apres fallback imageio-ffmpeg."
        )

    _ffmpeg_checked = True


def _ensure_whisper_loaded():
    """Load Whisper once; safe across threads and per-request ``asyncio.run()`` loops."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        try:
            import whisper
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Whisper is not installed. Run: pip install openai-whisper"
            ) from e
        _ensure_ffmpeg_available()
        _whisper_model = whisper.load_model("small", device="cpu")
        return _whisper_model

# Voix edge-tts utilisées pour ALIA — une voix par langue prise en charge
EDGE_TTS_VOICES = {
    "fr": "fr-FR-HenriNeural",
    "en": "en-US-GuyNeural",
    "es": "es-ES-AlvaroNeural",
    "ar": "ar-SA-HamedNeural",
}
EDGE_TTS_DEFAULT_LANG = "fr"


def _resolve_tts_voice(lang: str | None = None) -> str:
    """Return the edge-tts voice name for the given language code."""
    if lang and lang in EDGE_TTS_VOICES:
        return EDGE_TTS_VOICES[lang]
    # Try 2-letter prefix (e.g. 'fr-FR' -> 'fr')
    if lang:
        short = lang[:2].lower()
        if short in EDGE_TTS_VOICES:
            return EDGE_TTS_VOICES[short]
    return EDGE_TTS_VOICES[EDGE_TTS_DEFAULT_LANG]

def api_prefix() -> str:
    return settings.MODELING_API_MOUNT_PATH.rstrip("/") or "/alia-api"


def clean_for_tts(text: str) -> str:
    text = text or ""
    text = text.replace("*", "").replace("#", "")
    text = text.replace("**", "").replace("__", "")
    text = text.replace('"', "")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


async def _get_whisper_model():
    if _whisper_model is not None:
        return _whisper_model
    return await asyncio.to_thread(_ensure_whisper_loaded)


async def _synthesize_to_file(text: str, destination: Path, lang: str | None = None) -> None:
    """
    Synthèse vocale avec edge-tts (Microsoft Edge Neural TTS).
    Aucune clé API requise — fonctionne comme le navigateur Edge.
    Nécessite une connexion internet.
    ``lang`` : code ISO-639-1 ("fr", "en", "es", "ar") pour choisir la voix.

    Installation : pip install edge-tts
    """
    try:
        import edge_tts
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "edge-tts n'est pas installé. Lancez : pip install edge-tts"
        ) from e

    voice = _resolve_tts_voice(lang)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(destination))


async def listen_json(audio_bytes: bytes) -> dict:
    """Transcribe audio using Whisper with automatic language detection.
    Returns {"text": "...", "detected_lang": "fr"|"en"|"es"|"ar"|...}
    """
    if not audio_bytes:
        return {"text": "", "detected_lang": "fr"}
    _ensure_ffmpeg_available()
    model = await _get_whisper_model()
    fd, tmp_path = tempfile.mkstemp(suffix=".webm")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
        result = await asyncio.to_thread(
            model.transcribe,
            tmp_path,
            language=None,   # auto-detect language
            fp16=False,
            verbose=False,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    text = (result.get("text") or "").strip()
    detected_lang = (result.get("language") or "fr").strip()
    # Normalise to 2-letter code and default to 'fr' if unsupported
    detected_lang = detected_lang[:2].lower()
    return {"text": text, "detected_lang": detected_lang}


# Fallback messages per language
_FALLBACK_MESSAGES = {
    "fr": "Je n'ai pas compris votre question.",
    "en": "I didn't understand your question.",
    "es": "No he entendido su pregunta.",
    "ar": "لم أفهم سؤالك.",
}
_FALLBACK_EMPTY = {
    "fr": "Désolée, je n'ai pas pu générer une réponse. Pouvez-vous reformuler ?",
    "en": "Sorry, I couldn't generate a response. Could you rephrase?",
    "es": "Lo siento, no pude generar una respuesta. ¿Puede reformular?",
    "ar": "عذرًا، لم أتمكن من إنشاء إجابة. هل يمكنك إعادة الصياغة؟",
}


async def ask_alia_json(text: str, lang: str = "fr") -> dict:
    print(f"\n{'=' * 60}\n[modeling] Query ({lang}): {text}\n{'=' * 60}")
    rt = get_runtime()
    result = await rt.alia.generate(text, lang=lang)
    intent = result.get("intent", "unknown")
    print(f"[modeling] Response intent: {intent}")

    text_response = result.get("text", _FALLBACK_MESSAGES.get(lang, _FALLBACK_MESSAGES["fr"]))
    if not text_response or not text_response.strip():
        text_response = _FALLBACK_EMPTY.get(lang, _FALLBACK_EMPTY["fr"])

    # Video generation already includes its own audio — no TTS needed
    if intent == "presentation_generated":
        return {
            "text": text_response,
            "audio_url": None,
            "intent": intent,
            "detected_lang": lang,
            "video_url": result.get("video_url"),
            "presentation_url": result.get("presentation_url"),
        }

    speech_text = clean_for_tts(text_response)
    filename = f"{uuid.uuid4()}.mp3"
    path = Path(settings.MODELING_AUDIO_DIR) / filename
    prefix = api_prefix()

    try:
        await _synthesize_to_file(speech_text, path, lang=lang)
        print(f"[modeling] Audio généré ({lang}): {filename}")
    except Exception as audio_error:
        print(f"[modeling] Audio generation failed: {audio_error}")
        return {
            "text": text_response,
            "audio_url": None,
            "intent": intent,
            "detected_lang": lang,
            "error": "Audio generation failed",
        }

    return {
        "text": text_response,
        "audio_url": f"{prefix}/static/audio/{filename}",
        "intent": intent,
        "detected_lang": lang,
    }


def health_json() -> dict:
    try:
        rt = get_runtime()
        return {
            "status": "ok",
            "alia_loaded": rt.alia is not None,
            "kb_loaded": rt.kb is not None,
            "num_products": len(rt.df) if rt.df is not None else 0,
            "module": rt.rag_mod.__name__,
            "tts_engine": f"edge-tts (multilingual: {list(EDGE_TTS_VOICES.keys())})",
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


def debug_search_json(text: str) -> dict:
    try:
        rt = get_runtime()
        docs = rt.kb.retriever.invoke(text)
        results = []
        for i, doc in enumerate(docs[:5]):
            results.append(
                {
                    "rank": i + 1,
                    "product": doc.metadata.get("name", "Unknown"),
                    "score": "N/A",
                    "preview": doc.page_content[:200],
                }
            )
        return {
            "query": text,
            "num_results": len(docs),
            "top_results": results,
        }
    except Exception as e:
        return {"error": str(e)}
