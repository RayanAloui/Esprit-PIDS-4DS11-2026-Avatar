"""
ALIA RAG — rag_gemini.py
Deux modes :
  commercial : ALIA assiste la déléguée en visite (médecin / pharmacien)
  training   : ALIA JOUE le rôle du médecin ou pharmacien et évalue la déléguée
"""
from __future__ import annotations

import asyncio
import os
import random
import re
from pathlib import Path

import pandas as pd
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from langchain_community.retrievers import BM25Retriever


MODEL_SETTINGS = {
    "main": "llama3.2:latest",
    "embed": "nomic-embed-text:latest",
    "temperature": 0.5, 
    "num_predict": 60
}

_DATA_DIR = Path(__file__).resolve().parent / "data"
PERSIST_DIR = str(_DATA_DIR / "alia_knowledge_db")

PERSONAS = [
    {
        "prenom": "Dr. Martin",
        "role": "médecin généraliste, plutôt ouvert aux compléments alimentaires mais demande des preuves"
    },
    {
        "prenom": "Dr. Bernard",
        "role": "médecin généraliste, sceptique et très orienté médicaments classiques"
    },
    {
        "prenom": "Dr. Petit",
        "role": "pédiatre, très prudent, ne recommande que des produits spécifiquement pour enfants"
    },
    {
        "prenom": "Dr. Dubois",
        "role": "pharmacien, connaît bien les produits mais demande des arguments différenciants"
    },
    {
        "prenom": "Dr. Lambert",
        "role": "médecin généraliste, ouvert d'esprit mais très occupé, va droit au but"
    },
    {
        "prenom": "Dr. Moreau",
        "role": "rhumatologue, s'intéresse particulièrement aux produits pour articulations et os"
    },
    {
        "prenom": "Dr. Simon",
        "role": "dermatologue, très exigeant sur la qualité et l'efficacité des produits pour la peau"
    }
]
# MODE TRAINING : ALIA joue le médecin/pharmacien
# Le délégué doit convaincre le "médecin" de prescrire / recommander les produits Vital SA
PROMPT_TRAINING = """Tu es {prenom}, {persona}. Tu reçois une déléguée médicale.

Historique de la conversation:
{history}

Ce que la déléguée vient de dire: "{query}"

Contexte des produits Vital SA (information pour toi, médecin): 
{context}

RÈGLES IMPORTANTES POUR TON RÔLE:
1. Si la déléguée se présente ("Bonjour docteur..."), réponds simplement: "Bonjour, je vous écoute. Quels produits souhaitez-vous me présenter ?"

2. Si la déléguée mentionne un produit spécifique (ex: "Ferson", "VitalFer"), réponds en posant UNE question pertinente sur ce produit:
   - "Quels sont les avantages de ce produit par rapport aux génériques ?"
   - "Avez-vous des études cliniques ?"
   - "Quels sont les effets secondaires ?"
   - "Quelle est la posologie ?"
   - "Est-ce remboursé ?"

3. Ne répète PAS la même question que dans l'historique.

4. Reste professionnel mais engage la conversation.

5. Réponds en UNE SEULE PHRASE courte, sans sauter de ligne.

{prenom}:"""


PROMPT_TRAINING_OBJECTIONS = {
    "prix": "Le prix me semble élevé par rapport aux génériques.",
    "efficacité": "Avez-vous des études cliniques qui prouvent l'efficacité ?",
    "effets_secondaires": "Quels sont les effets secondaires ?",
    "remboursement": "Est-ce remboursé par la sécurité sociale ?",
    "concurrent": "En quoi ce produit est-il meilleur que le produit X ?",
}

# =========================
# DATA PROCESSING
# =========================
class DataProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"[•–\-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def infer_usage(forme: str) -> str:
        forme = str(forme).lower()
        if any(x in forme for x in ["gélule", "capsule", "sirop", "comprimé", "sachet"]):
            return "oral"
        if any(x in forme for x in ["crème", "gel", "lotion", "pommade"]):
            return "topique"
        return "oral"

    @staticmethod
    def infer_target(name: str) -> str:
        name = str(name).lower()
        if "junior" in name or "enfant" in name or "bébé" in name or "pédiatrique" in name:
            return "enfant"
        return "adulte"

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """Extract important medical keywords from text"""
        text = str(text).lower()
        keywords = []
        
        # Common medical conditions
        conditions = [
            "anémie", "fatigue", "stress", "sommeil", "digestion",
            "constipation", "diarrhée", "ballonnement", "acné", 
            "peau", "cheveux", "ongles", "articulation", "os",
            "immunité", "rhume", "grippe", "toux", "allergie",
            "mémoire", "concentration", "vision", "circulation"
        ]
        
        for condition in conditions:
            if condition in text:
                keywords.append(condition)
        
        return keywords

    @classmethod
    def build_documents(cls, df: pd.DataFrame) -> List[Document]:
        docs = []

        for _, r in df.iterrows():
            name = str(r.get('name', '')).strip()
            categories = cls.clean_text(r.get('categories', ''))
            indications = cls.clean_text(r.get('indications', ''))
            forme = str(r.get('forme', '')).strip()
            classe = str(r.get('classe', '')).strip()
            infos = cls.clean_text(r.get('infos_produit', ''))
            
            # Extract keywords from indications and categories
            keywords = cls.extract_keywords(indications + " " + categories + " " + classe)
            
            # CRITICAL FIX: Ensure keywords is never empty
            if not keywords:
                keywords = ["general"]  # Default keyword for products without specific conditions
            
            keywords_str = ", ".join(keywords) if keywords else ""
            
            # Build rich content with better structure
            content = f"""
    Produit: {name}

    Indications principales: {indications}

    Catégories: {categories}

    Classe thérapeutique: {classe}

    Forme galénique: {forme}

    Utilisation: {cls.infer_usage(forme)}

    Population cible: {cls.infer_target(name)}

    Description: {infos}

    Mots-clés: {keywords_str}
            """.strip()

            # Add metadata for better filtering
            metadata = {
                "name": name,
                "forme": forme,
                "target": cls.infer_target(name),
                "usage": cls.infer_usage(forme),
                "keywords": keywords  # Now guaranteed to be non-empty
            }

            docs.append(Document(
                page_content=content,
                metadata=metadata
            ))

        return docs


class _WeightedRRFRetriever:
    """
    Weighted reciprocal-rank fusion of several LangChain retrievers.
    Replaces langchain_classic.EnsembleRetriever to avoid importing transformers/TensorFlow
    (broken with NumPy 2 on many Windows installs).
    """

    def __init__(self, retrievers, weights, rrf_k: int = 60, limit: int = 16):
        self.retrievers = list(retrievers)
        self.weights = list(weights)
        self.rrf_k = rrf_k
        self.limit = limit
        

    def invoke(self, query: str) -> List[Document]:
        scores: Dict[str, Tuple[Document, float]] = {}
        for retriever, w in zip(self.retrievers, self.weights):
            for rank, doc in enumerate(retriever.invoke(query)):
                key = doc.page_content[:500] + "\0" + str(doc.metadata.get("name", ""))
                contrib = w / (self.rrf_k + rank + 1)
                if key in scores:
                    prev_doc, prev_s = scores[key]
                    scores[key] = (prev_doc, prev_s + contrib)
                else:
                    scores[key] = (doc, contrib)
        ranked = sorted(scores.values(), key=lambda x: -x[1])
        return [d for d, _ in ranked[: self.limit]]


# =========================
# KNOWLEDGE MANAGER
# =========================
class KnowledgeManager:
    def __init__(self, persist_dir=PERSIST_DIR):
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model=MODEL_SETTINGS["embed"])

        self.vectorstore = None
        self.retriever = None

    def load_or_create(self, docs: List[Document]):
        print(f"[KB] Loading/creating knowledge base with {len(docs)} documents...")
        
        # Utilise un client en mémoire (EphemeralClient) pour éviter
        # l'incompatibilité ChromaDB Rust + Python 3.14
        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self.vectorstore = Chroma(
                client=client,
                collection_name="alia_kb",
                embedding_function=self.embeddings,
            )
            # Populate if empty
            if self.vectorstore._collection.count() == 0:
                print(f"[KB] Populating collection with {len(docs)} documents...")
                self.vectorstore.add_documents(docs)
        except Exception as e:
            print(f"[KB] EphemeralClient failed ({e}), fallback to from_documents...")
            self.vectorstore = Chroma.from_documents(docs, self.embeddings)
 
        self._build_retriever(docs)
        print("[KB] Knowledge base ready!")

    def _build_retriever(self, docs):
        # Increased k for better coverage
        vector_ret = self.vectorstore.as_retriever(search_kwargs={"k": 8})

        bm25_ret = BM25Retriever.from_documents(docs)
        bm25_ret.k = 8

        self.retriever = _WeightedRRFRetriever(
            retrievers=[vector_ret, bm25_ret],
            weights=[0.5, 0.5],
            limit=16,
        )


# =========================
# QUERY UNDERSTANDING
# =========================
class QueryAnalyzer:
    @staticmethod
    def analyze(query: str) -> Dict:
        q = query.lower()
        
        # Medical conditions mapping
        conditions = {
            "anémie": ["anémie", "fer", "hémoglobine", "fatigue extrême"],
            "fatigue": ["fatigue", "tonus", "énergie", "vitalité"],
            "stress": ["stress", "anxiété", "nervosité", "tension"],
            "sommeil": ["sommeil", "insomnie", "dormir", "repos"],
            "digestion": ["digestion", "digestif", "estomac", "intestin"],
            "peau": ["peau", "acné", "dermatologique", "cutané"],
            "immunité": ["immunité", "défense", "rhume", "grippe","gorge", "angine"],
            "articulation": ["articulation", "os", "cartilage", "mobilité"]
        }
        
        detected_conditions = []
        for condition, keywords in conditions.items():
            if any(kw in q for kw in keywords):
                detected_conditions.append(condition)

        return {
            "target": "enfant" if any(x in q for x in ["enfant", "junior", "bébé", "pédiatrique"]) else "adulte",
            "conditions": detected_conditions,
            "is_treatment_query": any(x in q for x in ["traitement", "soigner", "guérir", "médicament", "produit"])
        }


# =========================
# ORCHESTRATOR
# =========================
# =========================
# MODE D'ALIA
# =========================
# "training"   : ALIA évalue le délégué, corrige ses réponses, l'encourage
# "commercial" : ALIA assiste le délégué en visite chez un médecin/pharmacien
ALIA_MODE = "training"   # valeur par défaut
 
PROMPT_COMMERCIAL = """Tu es ALIA, assistante pharmaceutique d'une délégué médicale.
Tu aides la délégué pendant sa visite chez un médecin ou un pharmacien.
 
HISTORIQUE DE LA CONVERSATION:
{history}
 
QUESTION ACTUELLE: "{query}"
 
PRODUITS DISPONIBLES:
{context}
 
RÈGLES:
1. Vérifie que les produits correspondent EXACTEMENT à la question.
2. Si aucun produit ne correspond, dis clairement qu'il n'y en a pas dans la base actuelle.
3. Si un produit correspond, recommande LE MEILLEUR (1 seul) avec :
   - Nom exact du produit
   - Argument de vente en 1-2 phrases (bénéfice patient)
   - Mode d'administration
   - Version enfant si elle existe
4. Tiens compte de l'historique : ne répète pas ce qui a déjà été dit.
5. Reste concise et professionnelle.
6. Si la question est trop vague ,demande des précisions : "Pouvez-vous préciser quel type de produit vous intéresse ?"
 
Réponse:
"""

class AliaOrchestrator:
    HISTORY_MAX_TURNS = 5
    _MEDICAL_KW = [
        "fatigue", "sommeil", "stress", "anémie", "digestion", "peau", "acné",
        "cheveux", "ongles", "articulation", "os", "immunité", "rhume", "grippe",
        "toux", "gorge", "angine", "allergie", "mémoire", "concentration", "vision",
        "circulation", "douleur", "mal", "vitamine", "complément", "traitement",
        "médicament", "produit", "sirop", "comprimé", "gélule", "crème", "gel",
        "fer", "calcium", "magnésium", "zinc",
    ]

    def __init__(self, manager: KnowledgeManager, mode: str = ALIA_MODE):
        self.manager = manager
        self.mode = mode  # "commercial" ou "training"
        self.history: List[Dict] = []  # [{"role": "user"|"alia", "text": "..."}]
        self._turn = 0
        self._persona = random.choice(PERSONAS)
 
        print(f"[ALIA] Initializing LLM with model: {MODEL_SETTINGS['main']} | mode: {self.mode}")
        self.llm = OllamaLLM(
            model=MODEL_SETTINGS["main"],
            temperature=MODEL_SETTINGS["temperature"],
            num_predict=MODEL_SETTINGS["num_predict"]
        )
 
        self._build_chain()
 
    def _build_chain(self):
        template = PROMPT_TRAINING if self.mode == "training" else PROMPT_COMMERCIAL
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()
 


    def set_mode(self, mode: str):
        import random as _random
        if mode not in ("training", "commercial"):
            raise ValueError("mode doit être 'training' ou 'commercial'")
        self.mode     = mode
        self.history  = []
        self._turn    = 0
        self._persona = _random.choice(PERSONAS)
        self._build_chain()
        print(f"[ALIA] Mode: {mode} | Persona: {self._persona['prenom']}")
 
    def reset(self):
        import random as _random
        self.history  = []
        self._turn    = 0
        self._persona = _random.choice(PERSONAS)
        print(f"[ALIA] Reset | Persona: {self._persona['prenom']}")
 
    def _is_vague_commercial(self, query: str) -> bool:
        q = query.lower()
        greetings = ["bonjour", "docteur", "déléguée", "présenter", "enchantée", "ravi"]
        if any(g in q for g in greetings) and len(q.split()) < 20:
            return True
        if any(kw in q for kw in self._MEDICAL_KW):
            return False
        vague = ["quoi de neuf", "qu'est-ce que vous avez", "vous avez quoi", "montrez-moi"]
        return any(p in q for p in vague)
 
    def _format_history(self) -> str:
        turns = self.history[-(self.HISTORY_MAX_TURNS * 2):]
        if not turns:
            return "Début de visite."
        lines = []
        for msg in turns:
            if self.mode == "training":
                prefix = "Déléguée" if msg["role"] == "user" else self._persona["prenom"]
            else:
                prefix = "Déléguée" if msg["role"] == "user" else "ALIA"
            lines.append(f"{prefix}: {msg['text']}")
        return "\n".join(lines)
 
    def _add_to_history(self, role: str, text: str):
        self.history.append({"role": role, "text": text})
        # Limite la taille en mémoire
        max_entries = self.HISTORY_MAX_TURNS * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def _clean(self, text: str) -> str:
        # Remove any thinking tags or markup
        text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
        text = re.sub(r"</?[^>]+>", "", text)
        text = re.sub(r"Bonjour,?\s*(madame|monsieur)?\s*la?\s*délégu[eé]e?[.,]?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"merci de me présenter[^.]*\.\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"je vois que vous présentez[^.]*\.\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"pouvez-vous me dire ce qui vous amène[^?]*\?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"quels sont les produits[^?]*\?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _humanize(self, text: str) -> str:
        text = text.replace("Il est recommandé", "Je vous conseille")
        text = text.replace("Vous pouvez utiliser", "Vous pouvez prendre")
        text = text.replace("Je vous recommande", "Je vous conseille")
        return text

    def _check_relevance(self, query: str, docs: List[Document]) -> bool:
        """Check if retrieved documents are relevant to the query"""
        if not docs:
            return False
        
        # Extract key terms from query
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        
        # Check if any document has significant overlap
        for doc in docs[:3]:  # Check top 3 docs
            doc_lower = doc.page_content.lower()
            doc_terms = set(re.findall(r'\b\w+\b', doc_lower))
            
            # Calculate overlap
            overlap = query_terms & doc_terms
            if len(overlap) >= 2:  # At least 2 matching terms
                return True
        
        return False

    def _smart_fallback(self, query: str, docs: List[Document], analysis: Dict) -> str:
        """Provide intelligent fallback when no relevant products found"""
        
        conditions = analysis.get("conditions", [])
        
        if conditions:
            conditions_str = ", ".join(conditions)
            return f"Je n'ai pas de produit spécifique pour {conditions_str} dans ma base de données actuelle. Pourriez-vous me donner plus de détails sur vos besoins, ou puis-je vous aider avec un autre type de produit ?"
        
        return "Je n'ai pas trouvé de produit correspondant exactement à votre demande. Pourriez-vous reformuler ou préciser votre besoin ?"

    async def generate(self, query: str) -> Dict:
    try:
        self._turn += 1
        print(f"[ALIA] turn={self._turn} mode={self.mode} | {query[:80]}")
        
        # Mode training
        if self.mode == "training":
            # Premier message : accueil
            if self._turn == 1:
                rep = "Bonjour, je suis le Dr. Martin. Quel produit souhaitez-vous me présenter aujourd'hui ?"
                self._add_to_history("user", query)
                self._add_to_history("alia", rep)
                return {"text": rep, "intent": "greeting"}
            
            # Pour les tours suivants : toujours chercher les produits pertinents
            docs = await asyncio.to_thread(self.manager.retriever.invoke, query)
            print(f"[ALIA] docs trouvés: {len(docs)}")
            
            # Construire le contexte
            if docs:
                context = "\n\n---\n\n".join(d.page_content for d in docs[:3])
            else:
                context = "Aucun produit spécifique trouvé dans la base de données."
            
            # Appeler le LLM avec le prompt training
            prompt = PROMPT_TRAINING.format(
                prenom=self._persona["prenom"],
                persona=self._persona["role"],
                history=self._format_history(),
                query=query,
                context=context
            )
            
            response = await asyncio.wait_for(
                asyncio.to_thread(self.llm.invoke, prompt),
                timeout=30,
            )
            
            cleaned = self._clean(response)
            
            # Validation de la réponse
            if not cleaned or len(cleaned.split()) < 3:
                # Questions de secours
                fallback_questions = [
                    "Quels sont les avantages de ce produit ?",
                    "Avez-vous des études cliniques ?",
                    "Quels sont les effets secondaires ?",
                    "Quelle est la posologie ?",
                    "Est-ce remboursé ?",
                ]
                cleaned = random.choice(fallback_questions)
            
            self._add_to_history("user", query)
            self._add_to_history("alia", cleaned)
            return {"text": cleaned, "intent": "response"}
        
        else:  # Mode commercial - gardez votre code existant
            # Vague check — commercial seulement
            if self.mode == "commercial" and self._is_vague_commercial(query):
                rep = "Pouvez-vous préciser votre besoin ? Fatigue, sommeil, gorge, peau, ou autre pathologie ?"
                self._add_to_history("user", query)
                self._add_to_history("alia", rep)
                return {"text": rep, "intent": "clarification"}

            # Retrieval
            docs = await asyncio.to_thread(self.manager.retriever.invoke, query)
            print(f"[ALIA] docs={len(docs)}")
            
            if not docs:
                analysis = QueryAnalyzer.analyze(query)
                conditions = analysis.get("conditions", [])
                rep = (
                    f"Je n'ai pas de produit pour {', '.join(conditions)}. Pouvez-vous préciser ?"
                    if conditions else
                    "Je n'ai pas trouvé de produit correspondant. Pouvez-vous préciser ?"
                )
                self._add_to_history("user", query)
                self._add_to_history("alia", rep)
                return {"text": rep, "intent": "no_results"}
            
            context = "\n\n---\n\n".join(d.page_content for d in docs[:5])
            
            kwargs = {
                "context": context,
                "query": query,
                "history": self._format_history(),
            }
            
            response = await asyncio.wait_for(
                asyncio.to_thread(self.chain.invoke, kwargs),
                timeout=120,
            )
            
            cleaned = self._clean(response)
            is_relevant = self._check_relevance(query, docs)
            
            if not cleaned or len(cleaned.split()) < 5 or not is_relevant:
                analysis = QueryAnalyzer.analyze(query)
                conditions = analysis.get("conditions", [])
                cleaned = (
                    f"Je n'ai pas trouvé de produit exact pour {', '.join(conditions)}. Pouvez-vous préciser ?"
                    if conditions else "Pouvez-vous préciser votre demande ?"
                )
            else:
                cleaned = self._humanize(cleaned)
            
            self._add_to_history("user", query)
            self._add_to_history("alia", cleaned)
            return {"text": cleaned, "intent": "response"}
 
    except asyncio.TimeoutError:
        return {"text": "Délai dépassé. Veuillez reformuler.", "intent": "timeout"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"text": "Erreur technique. Veuillez réessayer.", "intent": "error"}

# =========================
# MAIN (USAGE)
# =========================
async def main():
    csv_path = _DATA_DIR / "vital_products.csv"
    df = pd.read_csv(csv_path).fillna("")
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")

    docs = DataProcessor.build_documents(df)

    manager = KnowledgeManager()
    manager.load_or_create(docs)

    alia = AliaOrchestrator(manager)

    print("\n=== ALIA Ready ===\n")

    # Test with anemia query
    test_queries = [
        "Quel est votre traitement pour l'anémie ?",
        "produit pour la fatigue",
        "quelque chose pour le rhume"
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print('='*60)
        res = await alia.generate(q)
        print(f"\nALIA: {res['text']}\n")

    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit\n")
    
    while True:
        q = input("\nUser: ")
        if q.lower() in ['quit', 'exit', 'q']:
            break
            
        res = await alia.generate(q)
        print("ALIA:", res["text"])


if __name__ == "__main__":
    asyncio.run(main())