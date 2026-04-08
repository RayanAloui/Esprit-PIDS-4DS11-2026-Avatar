"""Shared ALIA RAG HTTP logic (FastAPI + Django WSGI)."""
from __future__ import annotations

import uuid
from pathlib import Path

import edge_tts
from django.conf import settings

from apps.modeling.runtime import get_runtime


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
        communicate = edge_tts.Communicate(speech_text, "fr-FR-HenriNeural")
        await communicate.save(str(path))
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
