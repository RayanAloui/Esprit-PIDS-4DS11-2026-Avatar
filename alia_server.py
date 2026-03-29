import os
import uuid
import edge_tts
import importlib
import pandas as pd
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# IMPORT RAG MODULE
# =========================
try:
    # Use the IMPROVED RAG module
    import RAG_gemini as rag_mod
    print("[SERVER] ✓ Improved RAG module loaded successfully!")
except ImportError as e:
    print(f"[SERVER] Failed to import improved RAG module: {e}")
    print("[SERVER] Trying original module...")
    try:
        import RAG_gemini as rag_mod
        print("[SERVER] ✓ Original RAG module loaded")
    except ImportError:
        raise ImportError("No RAG module found. Make sure RAG_gemini_improved.py or RAG_gemini.py exists")

# =========================
# INIT APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# LOAD DATA + RAG
# =========================
CSV_PATH = "vital_products.csv"

print(f"[SERVER] Looking for CSV at: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"CSV file not found at {CSV_PATH}")

print(f"[SERVER] Loading CSV...")
df = pd.read_csv(CSV_PATH).fillna("")
df.columns = df.columns.str.strip().str.lstrip("\ufeff")
print(f"[SERVER] ✓ Loaded {len(df)} products")

# Show sample products
print("\n[SERVER] Sample products in database:")
for i, row in df.head(3).iterrows():
    name = row.get('name', 'Unknown')
    indications = str(row.get('indications', ''))[:40]
    print(f"  • {name} - {indications}...")
print()

print("[SERVER] Building documents...")
docs = rag_mod.DataProcessor.build_documents(df)
print(f"[SERVER] ✓ Built {len(docs)} documents")

print("[SERVER] Initializing knowledge base...")
kb = rag_mod.KnowledgeManager()
kb.load_or_create(docs)
print("[SERVER] ✓ Knowledge base ready!")

print("[SERVER] Initializing ALIA orchestrator...")
alia = rag_mod.AliaOrchestrator(kb)
print("[SERVER] ✓ ALIA ready!")

# =========================
# STATIC (AUDIO)
# =========================
os.makedirs("./static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# REQUEST MODEL
# =========================
class Query(BaseModel):
    text: str

# =========================
# TEXT CLEAN FOR TTS
# =========================
def clean_for_tts(text: str) -> str:
    text = text or ""
    # Remove markdown formatting
    text = text.replace("*", "").replace("#", "")
    text = text.replace("**", "").replace("__", "")
    # Remove quotes used for product names
    text = text.replace('"', '')
    # Normalize whitespace
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

# =========================
# ENDPOINT
# =========================
@app.post("/ask_alia")
async def ask_alia_endpoint(query: Query):
    print(f"\n{'='*60}")
    print(f"[ENDPOINT] Query: {query.text}")
    print('='*60)
    
    try:
        # Generate response from ALIA
        result = await alia.generate(query.text)
        print(f"[ENDPOINT] Response intent: {result.get('intent')}")

        text_response = result.get("text", "Je n'ai pas compris votre question.")
        
        if not text_response or len(text_response.strip()) == 0:
            text_response = "Désolée, je n'ai pas pu générer une réponse. Pouvez-vous reformuler ?"
        
        print(f"[ENDPOINT] Response: {text_response[:100]}...")
        
        # Clean text for TTS
        speech_text = clean_for_tts(text_response)

        # Generate audio
        filename = f"{uuid.uuid4()}.mp3"
        path = f"./static/audio/{filename}"
        
        try:
            communicate = edge_tts.Communicate(
                speech_text,
                "fr-FR-HenriNeural"
            )
            await communicate.save(path)
            print(f"[ENDPOINT] ✓ Audio saved")
        except Exception as audio_error:
            print(f"[ENDPOINT] ⚠ Audio generation failed: {audio_error}")
            # Continue without audio
            return {
                "text": text_response,
                "audio_url": None,
                "intent": result.get("intent", "unknown"),
                "error": "Audio generation failed"
            }

        response_data = {
            "text": text_response,
            "audio_url": f"/static/audio/{filename}",
            "intent": result.get("intent", "unknown")
        }
        
        print(f"[ENDPOINT] ✓ Response ready\n")
        return response_data

    except Exception as e:
        error_msg = f"[ENDPOINT] ✗ ERROR: {type(e).__name__}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# =========================
# ROOT
# =========================
@app.get("/")
async def index():
    return FileResponse("index.html")

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "alia_loaded": alia is not None,
        "kb_loaded": kb is not None,
        "num_products": len(df) if df is not None else 0,
        "module": "improved" if "improved" in rag_mod.__name__ else "original"
    }

# =========================
# DEBUG ENDPOINT - Test retrieval
# =========================
@app.post("/debug/search")
async def debug_search(query: Query):
    """Debug endpoint to test document retrieval"""
    try:
        docs = kb.retriever.invoke(query.text)
        results = []
        for i, doc in enumerate(docs[:5]):
            results.append({
                "rank": i + 1,
                "product": doc.metadata.get('name', 'Unknown'),
                "score": "N/A",
                "preview": doc.page_content[:200]
            })
        return {
            "query": query.text,
            "num_results": len(docs),
            "top_results": results
        }
    except Exception as e:
        return {"error": str(e)}

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting ALIA Server (IMPROVED)")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)