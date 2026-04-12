"""Shared ALIA RAG HTTP logic (FastAPI + Django WSGI)."""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import uuid
from pathlib import Path

from django.conf import settings

from apps.modeling.runtime import get_runtime

_whisper_model = None
_whisper_lock = threading.Lock()


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
        _whisper_model = whisper.load_model("small", device="cpu")
        return _whisper_model

# Voix edge-tts utilisée pour ALIA
# Autres options françaises disponibles :
#   fr-FR-HenriNeural      (voix masculine)
#   fr-FR-EloiseNeural     (voix féminine, plus douce)
#   fr-BE-GerardNeural     (accent belge)
EDGE_TTS_VOICE = "fr-FR-DeniseNeural"

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


async def _synthesize_to_file(text: str, destination: Path) -> None:
    """
    Synthèse vocale avec edge-tts (Microsoft Edge Neural TTS).
    Aucune clé API requise — fonctionne comme le navigateur Edge.
    Nécessite une connexion internet.
 
    Installation : pip install edge-tts
    """
    try:
        import edge_tts
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "edge-tts n'est pas installé. Lancez : pip install edge-tts"
        ) from e
 
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(str(destination))


async def listen_json(audio_bytes: bytes) -> dict:
    if not audio_bytes:
        return {"text": ""}
    model = await _get_whisper_model()
    fd, tmp_path = tempfile.mkstemp(suffix=".webm")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
        result = await asyncio.to_thread(
            model.transcribe,
            tmp_path,
            language="fr",
            fp16=False,
            verbose=False,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    text = (result.get("text") or "").strip()
    return {"text": text}


async def ask_alia_json(text: str) -> dict:
    print(f"\n{'=' * 60}\n[modeling] Query: {text}\n{'=' * 60}")
    rt = get_runtime()
    result = await rt.alia.generate(text)
    print(f"[modeling] Response intent: {result.get('intent')}")

    text_response = result.get("text", "Je n'ai pas compris votre question.")
    if not text_response or not text_response.strip():
        text_response = (
            "Désolée, je n'ai pas pu générer une réponse. Pouvez-vous reformuler ?"
        )

    speech_text = clean_for_tts(text_response)
    filename = f"{uuid.uuid4()}.mp3"
    path = Path(settings.MODELING_AUDIO_DIR) / filename
    prefix = api_prefix()

    try:
        await _synthesize_to_file(speech_text, path)
        print(f"[modeling] Audio généré : {filename}")
    except Exception as audio_error:
        print(f"[modeling] Audio generation failed: {audio_error}")
        return {
            "text": text_response,
            "audio_url": None,
            "intent": result.get("intent", "unknown"),
            "error": "Audio generation failed",
        }

    return {
        "text": text_response,
        "audio_url": f"{prefix}/static/audio/{filename}",
        "intent": result.get("intent", "unknown"),
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
            "tts_engine": f"edge-tts ({EDGE_TTS_VOICE})",
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
