from __future__ import annotations

from pathlib import Path

import pandas as pd
from django.conf import settings

_runtime = None


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
        self.kb = rag_mod.KnowledgeManager(persist_dir=str(settings.MODELING_KB_DIR))
        self.kb.load_or_create(docs)
        self.alia = rag_mod.AliaOrchestrator(self.kb)


def get_runtime() -> _Runtime:
    global _runtime
    if _runtime is None:
        _runtime = _Runtime()
    return _runtime
