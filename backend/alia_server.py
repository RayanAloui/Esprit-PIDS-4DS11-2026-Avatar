import os
import uuid
import edge_tts
import pandas as pd
import traceback
import tempfile
import whisper

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import RAG_gemini as rag_mod

# =========================
# INIT APP
# =========================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# =========================
# LOAD DATA + RAG
# =========================
CSV_PATH = "vital_products.csv"
if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"CSV introuvable : {CSV_PATH}")

print("[SERVER] Chargement CSV...")
df = pd.read_csv(CSV_PATH).fillna("")
df.columns = df.columns.str.strip().str.lstrip("\ufeff")
print(f"[SERVER] ✓ {len(df)} produits")

print("[SERVER] Construction documents...")
docs = rag_mod.DataProcessor.build_documents(df)

print("[SERVER] Initialisation base de connaissances...")
kb = rag_mod.KnowledgeManager()
kb.load_or_create(docs)
print("[SERVER] ✓ Base prête!")

print("[SERVER] Initialisation ALIA...")
alia = rag_mod.AliaOrchestrator(kb)
print("[SERVER] ✓ ALIA prête!")

# =========================
# LOAD WHISPER STT
# =========================
print("[SERVER] Chargement Whisper (small)...")
stt_model = whisper.load_model("small", device="cpu")
print("[SERVER] ✓ Whisper prêt!")

# =========================
# STATIC FILES
# =========================
os.makedirs("./static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# MODELS
# =========================
class Query(BaseModel):
    text: str
    mode: Optional[str] = "commercial"  # "commercial" ou "training"

class ModeRequest(BaseModel):
    mode: str  # "commercial" ou "training"

# =========================
# PROMPTS PAR MODE
# =========================
PROMPT_COMMERCIAL = """
Tu es ALIA, pharmacienne experte de Vital Tunisie.
Tu parles à un médecin ou pharmacien qui veut des informations sur les produits.

QUESTION: "{query}"

PRODUITS DISPONIBLES:
{context}

RÈGLES:
1. Réponds de façon professionnelle et concise (2-3 phrases max pour l'oral)
2. Recommande le produit le plus pertinent avec son indication principale
3. Mentionne le mode d'emploi si pertinent
4. Ton naturel, comme en vrai conversation avec un professionnel de santé

Réponse (courte, adaptée à l'oral):
"""

PROMPT_TRAINING = """
Tu es ALIA, formatrice experte en pharmacologie pour Vital Tunisie.
Tu simules une vraie visite médicale pour former un délégué médical.
Tu joues le rôle d'un médecin ou pharmacien exigeant et réaliste.

CONTEXTE DE LA VISITE: Le délégué présente les produits Vital Tunisie.
PRODUITS DISPONIBLES: {context}
MESSAGE DU DÉLÉGUÉ: "{query}"

TON RÔLE:
- Pose des questions difficiles comme un vrai médecin/pharmacien le ferait
- Réagis de façon réaliste (sceptique, curieux, demande des preuves)
- Si le délégué répond bien → valide et approfondis
- Si la réponse est incorrecte → corrige avec bienveillance
- Simule des objections réelles: prix, efficacité, effets secondaires, concurrence
- Garde un ton professionnel mais exigeant

Réponse (courte, naturelle, comme en vrai conversation sur le terrain):
"""

# =========================
# HELPER TTS
# =========================
def clean_for_tts(text: str) -> str:
    text = (text or "").replace("*","").replace("#","").replace("**","").replace("__","").replace('"','')
    return " ".join(text.replace("\n"," ").split()).strip()

async def generate_audio(text: str) -> Optional[str]:
    """Génère un fichier mp3 avec edge_tts et retourne l'URL"""
    try:
        filename = f"{uuid.uuid4()}.mp3"
        path = f"./static/audio/{filename}"
        await edge_tts.Communicate(clean_for_tts(text), "fr-FR-HenriNeural").save(path)
        return f"/static/audio/{filename}"
    except Exception as e:
        print(f"[TTS] ✗ Erreur audio: {e}")
        return None

# =========================
# ENDPOINT 1 — STT /listen
# Audio webm → Whisper → texte
# =========================
@app.post("/listen")
async def listen_endpoint(audio: UploadFile = File(...)):
    try:
        suffix = ".webm" if "webm" in (audio.content_type or "") else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        print(f"[STT] Transcription...")
        result = stt_model.transcribe(tmp_path, language="fr", fp16=False, verbose=False)
        os.unlink(tmp_path)

        text = result["text"].strip()
        print(f"[STT] ✓ '{text}'")
        return {"text": text}

    except Exception as e:
        print(f"[STT] ✗ {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ENDPOINT 2 — RAG+TTS /ask_alia
# Texte + mode → RAG (prompt adapté) → edge_tts → texte + mp3
# =========================
@app.post("/ask_alia")
async def ask_alia_endpoint(query: Query):
    print(f"\n[RAG/{query.mode.upper()}] '{query.text}'")
    try:
        # Injecter le bon prompt selon le mode
        original_prompt = alia.prompt
        if query.mode == "training":
            from langchain_core.prompts import ChatPromptTemplate
            alia.prompt = ChatPromptTemplate.from_template(PROMPT_TRAINING)
            alia.chain  = alia.prompt | alia.llm | __import__('langchain_core.output_parsers', fromlist=['StrOutputParser']).StrOutputParser()
        else:
            from langchain_core.prompts import ChatPromptTemplate
            alia.prompt = ChatPromptTemplate.from_template(PROMPT_COMMERCIAL)
            alia.chain  = alia.prompt | alia.llm | __import__('langchain_core.output_parsers', fromlist=['StrOutputParser']).StrOutputParser()

        result = await alia.generate(query.text)
        text_response = result.get("text") or "Désolée, je n'ai pas pu générer une réponse."
        print(f"[RAG] ✓ '{text_response[:80]}...'")

        # Restaurer prompt original
        alia.prompt = original_prompt
        alia.chain  = alia.prompt | alia.llm | __import__('langchain_core.output_parsers', fromlist=['StrOutputParser']).StrOutputParser()

        # Générer audio
        audio_url = await generate_audio(text_response)

        return {
            "text": text_response,
            "audio_url": audio_url,
            "mode": query.mode,
            "intent": result.get("intent", "unknown")
        }

    except Exception as e:
        print(f"[RAG] ✗ {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ENDPOINT 3 — Message de démarrage Training
# ALIA initie la simulation de visite
# =========================
@app.post("/start_training")
async def start_training():
    """ALIA lance la simulation en posant la première question"""
    try:
        opening = (
            "Bonjour, je suis le Docteur Martin. "
            "Vous avez rendez-vous avec moi pour me présenter les produits de votre laboratoire. "
            "Je vous écoute, qu'est-ce que vous avez à me proposer aujourd'hui ?"
        )
        audio_url = await generate_audio(opening)
        return {"text": opening, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ROOT
# =========================
@app.get("/")
async def index():
    return FileResponse("../frontend/index.html")


# =========================
# HEALTH
# =========================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "produits": len(df),
        "alia": alia is not None,
        "whisper": stt_model is not None,
        "modes": ["commercial", "training"]
    }


# =========================
# DEBUG
# =========================
@app.post("/debug/search")
async def debug_search(query: Query):
    try:
        results = kb.retriever.invoke(query.text)
        return {
            "query": query.text,
            "nb": len(results),
            "top5": [{"rang": i+1, "produit": d.metadata.get('name'), "apercu": d.page_content[:150]}
                     for i, d in enumerate(results[:5])]
        }
    except Exception as e:
        return {"error": str(e)}


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  ALIA — Mode Commercial + Training")
    print("  STT (Whisper) + RAG + TTS (edge_tts)")
    print("  http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)