from __future__ import annotations

from pathlib import Path
import shutil
import threading
import pandas as pd
from django.conf import settings

_runtime = None
_runtime_lock = threading.Lock()


class _Runtime:
    def __init__(self) -> None:
        from apps.modeling import rag_gemini as rag_mod

        self.rag_mod = rag_mod
        csv_path = Path(settings.MODELING_CSV)
        if not csv_path.is_file():
            raise RuntimeError(f"CSV file not found: {csv_path}")
        print(f"[modeling] Loading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path).fillna("")
        self.df.columns = self.df.columns.str.strip().str.lstrip("\ufeff")
        docs = rag_mod.DataProcessor.build_documents(self.df)
        # Load enriched knowledge base (best responses + avatars)
        data_dir = Path(settings.MODELING_CSV).parent  # same folder as the CSV
        self.kb_loader = rag_mod.KnowledgeBaseLoader(data_dir)
        self.kb_loader.load()
        
        self.kb = rag_mod.KnowledgeManager(persist_dir=str(settings.MODELING_KB_DIR))
        self.kb.load_or_create(docs)
        self.alia = rag_mod.AliaOrchestrator(self.kb,self.kb_loader)

        # Préchargement de Whisper au démarrage pour éviter le délai
        # de 5-10 secondes au premier appel micro
        print("[modeling] Préchargement du modèle Whisper...")
        try:
            import whisper
            self.whisper_model = whisper.load_model("small", device="cpu")
            # Partage le modèle avec handlers.py pour éviter un double chargement
            import apps.modeling.handlers as _handlers
            _handlers._whisper_model = self.whisper_model
            print("[modeling] Whisper prêt.")
        except Exception as e:
            print(f"[modeling] Whisper non disponible au démarrage : {e}")


def get_runtime() -> _Runtime:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _Runtime()
            # Verify alia was properly initialized
            if not hasattr(_runtime, 'alia') or _runtime.alia is None:
                print("[ERROR] Alia orchestrator failed to initialize")
                raise RuntimeError("Failed to initialize Alia orchestrator")
    return _runtime
