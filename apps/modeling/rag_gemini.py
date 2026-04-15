"""
ALIA RAG — rag_gemini.py  (v2 — enrichi avec prepared_data)
============================================================
Deux modes :
  commercial : ALIA assiste la déléguée en visite (médecin / pharmacien)
  training   : ALIA JOUE le rôle du médecin ou pharmacien et évalue la déléguée
 
Nouveautés v2 :
  - Base de connaissances construite depuis conversations_dso2_knowledge_base.csv
    (meilleures réponses réelles aux objections, scorées et validées)
  - Détection automatique du produit et du type d'objection dans le discours
    de la déléguée → réponse modèle issue des données
  - Avatars de médecins/pharmaciens chargés depuis pharmacy_doctor_avatar_profiles.json
  - Feedback training : score estimé + conseil personnalisé
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import json
from pathlib import Path

import pandas as pd
from typing import Dict, List, Tuple,Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from langchain_community.retrievers import BM25Retriever


MODEL_SETTINGS = {
    "main": "mistral:7b-instruct",
    "embed": "nomic-embed-text:latest",
    "temperature": 0.5, 
    "num_predict": 120
}

_DATA_DIR = Path(__file__).resolve().parent / "data"
PERSIST_DIR = str(_DATA_DIR / "alia_knowledge_db")

# ══════════════════════════════════════════════════════════════════════
# PRODUCT KNOWLEDGE — chargé depuis prepared_data
# ══════════════════════════════════════════════════════════════════════
 
# Catalogue produits Vital SA avec indications détaillées
PRODUCT_CATALOG: Dict[str, Dict] = {
    "OLIGOVIT Trio": {
        "indications": "multivitamines, carences nutritionnelles, fatigue générale, déficit en oligoéléments",
        "classe": "Complément alimentaire multivitaminé",
        "forme": "Comprimés",
        "target": "adulte",
        "differentiants": "formulation biodisponible brevetée, absorption supérieure de 40% vs génériques",
        "posologie": "1 comprimé par jour au cours du repas",
        "effets_secondaires": "Profil de sécurité excellent. Aucun effet indésirable significatif dans les études.",
        "keywords": ["vitamine", "oligoélément", "fatigue", "carence", "multivitamine"],
    },
    "LV CYSPROTECT": {
        "indications": "infections urinaires récidivantes, cystites, protection de la muqueuse vésicale",
        "classe": "Complément alimentaire urologique",
        "forme": "Gélules",
        "target": "adulte",
        "differentiants": "D-mannose + cranberry + probiotiques, action triple sur la muqueuse urinaire",
        "posologie": "2 gélules par jour, cure de 3 mois recommandée",
        "effets_secondaires": "Tolérance excellente. Pas d'interaction médicamenteuse connue.",
        "keywords": ["infection urinaire", "cystite", "voies urinaires", "urinaire", "récidivante"],
    },
    "LV Fersang junior": {
        "indications": "anémie ferriprive chez l'enfant, carence en fer pédiatrique, fatigue de l'enfant",
        "classe": "Supplément en fer pédiatrique",
        "forme": "Sirop",
        "target": "enfant",
        "differentiants": "fer bisglycinate hautement biodisponible, sans goût métallique, tolérance digestive optimale",
        "posologie": "Selon âge : 5 à 10 ml par jour selon prescription pédiatrique",
        "effets_secondaires": "Bien toléré. Pas de constipation ni de nausées contrairement aux sels ferreux classiques.",
        "keywords": ["anémie", "fer", "enfant", "pédiatrique", "junior", "hémoglobine", "ferriprive"],
    },
    "Omevie Omega 3 – 1000": {
        "indications": "hypertriglycéridémie, santé cardiovasculaire, inflammation chronique, santé cérébrale",
        "classe": "Acides gras oméga-3 concentrés",
        "forme": "Capsules molles",
        "target": "adulte",
        "differentiants": "EPA/DHA concentrés 1000 mg, pureté certifiée IFOS 5 étoiles, sans goût de poisson",
        "posologie": "2 capsules par jour au cours des repas",
        "effets_secondaires": "Très bonne tolérance. Légère anticoagulation à surveiller avec AVK.",
        "keywords": ["omega", "oméga", "cardiovasculaire", "triglycérides", "cerveau", "inflammation"],
    },
    "LV Vitamine A": {
        "indications": "carence en vitamine A, santé oculaire, protection cutanée, immunité",
        "classe": "Supplément vitaminique",
        "forme": "Capsules",
        "target": "adulte",
        "differentiants": "rétinol naturel d3 haute biodisponibilité, formulation liposomale",
        "posologie": "1 capsule par jour, cure de 1 mois",
        "effets_secondaires": "Respecter les doses recommandées. Attention en grossesse (tératogène à forte dose).",
        "keywords": ["vitamine A", "vision", "peau", "immunité", "carence", "oculaire"],
    },
    "Echinacée, Zinc, Vit.C": {
        "indications": "immunité, rhume, grippe, infections ORL, prévention hivernale, angine",
        "classe": "Immunostimulant naturel",
        "forme": "Gélules",
        "target": "adulte",
        "differentiants": "synergie triple Echinacée + Zinc + Vitamine C, action préventive et curative",
        "posologie": "1 gélule matin et soir pendant les repas",
        "effets_secondaires": "Excellent profil de sécurité. +50 000 patients sans incident.",
        "keywords": ["immunité", "rhume", "grippe", "ORL", "gorge", "angine", "prévention", "hiver", "infection"],
    },
    "Vitonic Tonus": {
        "indications": "fatigue chronique, asthénie, manque de tonus, récupération physique, stress",
        "classe": "Tonique et énergisant",
        "forme": "Ampoules buvables",
        "target": "adulte",
        "differentiants": "association ginseng + vitamines B + magnésium, résultats visibles en 4 semaines",
        "posologie": "1 ampoule par jour le matin, cure de 4 semaines",
        "effets_secondaires": "Bonne tolérance générale. Déconseillé le soir (effet stimulant).",
        "keywords": ["fatigue", "tonus", "énergie", "asthénie", "stress", "récupération", "vitalité"],
    },
    "PHYTOFANE Lotion anti chute": {
        "indications": "chute de cheveux, alopécie, fragilité capillaire, densification des cheveux",
        "classe": "Soin capillaire médicalisé",
        "forme": "Lotion topique",
        "target": "adulte",
        "differentiants": "complexe phyto-kératinisant breveté, résultats dès 6 semaines, sans hormone",
        "posologie": "Application locale 2 fois par semaine, massage 3 minutes",
        "effets_secondaires": "Usage externe uniquement. Aucun effet systémique.",
        "keywords": ["cheveux", "chute", "alopécie", "capillaire", "densité", "calvitie"],
    },
}
 
# Types d'objections reconnus
OBJECTION_TYPES = [
    "Efficacité douteuse",
    "Prix élevé",
    "Effets Secondaires",
    "Posologie complexe",
    "Marque inconnue",
    "Stock suffisant",
    "Demande Faible",
    "Concurrence",
]
 
# Mots-clés pour détecter chaque type d'objection
OBJECTION_KEYWORDS: Dict[str, List[str]] = {
    "Efficacité douteuse": ["efficacité", "preuves", "études", "fonctionne", "résultats", "scientifique", "démontré", "doute"],
    "Prix élevé": ["prix", "cher", "coût", "coûte", "tarif", "remboursement", "remboursé", "budget"],
    "Effets Secondaires": ["effets", "secondaires", "danger", "risque", "sécurité", "tolérance", "contre-indication"],
    "Posologie complexe": ["posologie", "dose", "dosage", "complexe", "compliqué", "comment prendre", "administration"],
    "Marque inconnue": ["marque", "laboratoire", "connu", "réputation", "fiable", "sérieux"],
    "Stock suffisant": ["stock", "j'en ai", "déjà", "assez", "suffisant"],
    "Demande Faible": ["demande", "patients", "prescrivent", "vendent", "vente", "faible"],
    "Concurrence": ["concurrent", "générique", "similaire", "moins cher", "laboratoire X", "autre marque", "comparaison"],
}

# ══════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE LOADER — depuis prepared_data
# ══════════════════════════════════════════════════════════════════════
 
class KnowledgeBaseLoader:
    """Charge et structure la base de connaissances depuis les CSV/JSON prepared_data."""
 
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.best_responses: Dict[Tuple[str, str], Dict] = {}
        self.avatars: List[Dict] = []
        self._loaded = False
 
    def load(self):
        if self._loaded:
            return
        self._load_kb_csv()
        self._load_avatars()
        self._loaded = True
        print(f"[KB] Loaded {len(self.best_responses)} product×objection pairs | {len(self.avatars)} avatars")
 
    def _load_kb_csv(self):
        """Charge conversations_dso2_knowledge_base.csv — meilleures réponses scorées."""
        csv_path = self.data_dir / "conversations_dso2_knowledge_base.csv"
        if not csv_path.exists():
            print(f"[KB] WARNING: {csv_path} not found, using defaults")
            return
 
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lstrip("\ufeff")
 
        # Pour chaque produit × objection, garder la meilleure réponse
        best = (
            df.sort_values("overall_score", ascending=False)
            .groupby(["product", "objection_type"])
            .first()
            .reset_index()
        )
 
        for _, row in best.iterrows():
            key = (str(row["product"]), str(row["objection_type"]))
            self.best_responses[key] = {
                "response": str(row["rep_response"]),
                "score": float(row.get("overall_score", 8.0)),
                "has_empathy": bool(row.get("has_empathy", 0)),
                "has_data_args": bool(row.get("has_data_args", 0)),
                "skill_level": str(row.get("skill_level", "Expert")),
            }
 
    def _load_avatars(self):
        """Charge pharmacy_doctor_avatar_profiles.json."""
        json_path = self.data_dir / "pharmacy_doctor_avatar_profiles.json"
        if not json_path.exists():
            print(f"[KB] WARNING: {json_path} not found, using PERSONAS defaults")
            return
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            self.avatars = raw
        elif isinstance(raw, dict):
            self.avatars = list(raw.values())
 
    def get_best_response(self, product: str, objection_type: str) -> Optional[str]:
        key = (product, objection_type)
        entry = self.best_responses.get(key)
        return entry["response"] if entry else None
 
    def get_random_avatar(self) -> Dict:
        if self.avatars:
            return random.choice(self.avatars)
        return {}
 
 
# ══════════════════════════════════════════════════════════════════════
# PERSONAS (fallback si pas d'avatars)
# ══════════════════════════════════════════════════════════════════════
 
PERSONAS = [
    {"prenom": "Dr. Martin", "role": "médecin généraliste, ouvert aux compléments alimentaires mais demande des preuves"},
    {"prenom": "Dr. Bernard", "role": "médecin généraliste, sceptique, très orienté médicaments classiques"},
    {"prenom": "Dr. Petit", "role": "pédiatre, très prudent, ne recommande que des produits spécifiquement pour enfants"},
    {"prenom": "Dr. Dubois", "role": "pharmacien, connaît bien les produits mais demande des arguments différenciants"},
    {"prenom": "Dr. Lambert", "role": "médecin généraliste, ouvert d'esprit mais très occupé, va droit au but"},
    {"prenom": "Dr. Moreau", "role": "rhumatologue, s'intéresse particulièrement aux produits pour articulations et os"},
    {"prenom": "Dr. Simon", "role": "dermatologue, très exigeant sur qualité et efficacité des produits pour la peau"},
]
 
 
# ══════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════

# MODE TRAINING : ALIA joue le médecin/pharmacien
# Le délégué doit convaincre le "médecin" de prescrire / recommander les produits Vital SA
PROMPT_TRAINING = """Tu es {prenom}, {role}.
La déléguée dit: "{query}"
Produit concerné: {context}
Conversation récente:
{history}
 
Règles:
- Réponds en UNE SEULE PHRASE courte et naturelle (médecin réaliste)
- Ne dis pas "Bonjour" après le premier échange
- Pose UNE question pertinente sur le produit (efficacité, études, effets secondaires, prix, posologie)
- Ne répète pas les questions déjà posées dans l'historique
 
{prenom}:"""

PROMPT_TRAINING_OBJECTIONS = {
    "prix": "Le prix me semble élevé par rapport aux génériques.",
    "efficacité": "Avez-vous des études cliniques qui prouvent l'efficacité ?",
    "effets_secondaires": "Quels sont les effets secondaires ?",
    "remboursement": "Est-ce remboursé par la sécurité sociale ?",
    "concurrent": "En quoi ce produit est-il meilleur que le produit X ?",
}

PROMPT_COMMERCIAL = """Tu es ALIA, assistante pharmaceutique d'une déléguée médicale.
Tu aides la déléguée pendant sa visite chez un médecin ou un pharmacien.

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

Réponse d'ALIA:"""
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

@classmethod
def build_documents_from_csv(cls, csv_path: Path) -> List[Document]:
    """Construit des documents à partir du CSV des produits"""
    if not csv_path.exists():
        print(f"[KB] CSV not found: {csv_path}")
        return []
    
    df = pd.read_csv(csv_path).fillna("")
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    return cls.build_documents(df)

@classmethod
def build_documents_from_catalog(cls) -> List[Document]:
    """Construit des documents à partir du catalogue PRODUCT_CATALOG"""
    docs = []
    for product_name, product_info in PRODUCT_CATALOG.items():
        content = f"""
Produit: {product_name}
Indications: {product_info.get('indications', '')}
Classe: {product_info.get('classe', '')}
Forme: {product_info.get('forme', '')}
Points différenciants: {product_info.get('differentiants', '')}
Posologie: {product_info.get('posologie', '')}
Effets secondaires: {product_info.get('effets_secondaires', '')}
        """.strip()
        
        docs.append(Document(
            page_content=content,
            metadata={
                'name': product_name,
                'indications': product_info.get('indications', ''),
                'keywords': product_info.get('keywords', [])
            }
        ))
    return docs
# ══════════════════════════════════════════════════════════════════════
# TRAINING EVALUATOR — évalue la réponse de la déléguée
# ══════════════════════════════════════════════════════════════════════
 
class TrainingEvaluator:
    """Évalue la qualité de la réponse de la déléguée et fournit un feedback."""
 
    @staticmethod
    def evaluate(rep_response: str, product: str, objection_type: Optional[str], kb_loader: KnowledgeBaseLoader) -> Dict:
        score = 5.0
        tips = []
 
        resp_lower = rep_response.lower()
        resp_words = len(rep_response.split())
 
        # Critère 1 : empathie
        empathy_phrases = ["je comprends", "c'est une bonne question", "excellente question",
                           "je vois", "tout à fait", "vous avez raison"]
        if any(ep in resp_lower for ep in empathy_phrases):
            score += 1.5
        else:
            tips.append("💡 Commencez par montrer de l'empathie (ex: 'Je comprends votre préoccupation...')")
 
        # Critère 2 : données / chiffres
        if re.search(r"\d+\s*%", rep_response) or any(kw in resp_lower for kw in ["étude", "clinique", "prouvé", "démontré"]):
            score += 1.5
        else:
            tips.append("📊 Appuyez-vous sur des données chiffrées ou des études cliniques")
 
        # Critère 3 : solution concrète / invitation
        if any(kw in resp_lower for kw in ["échantillon", "puis-je", "je peux vous", "proposer", "montrer", "laisser"]):
            score += 1.0
        else:
            tips.append("🤝 Terminez par une proposition concrète (échantillon, documentation, rendez-vous)")
 
        # Critère 4 : longueur suffisante
        if resp_words >= 30:
            score += 0.5
        elif resp_words < 15:
            score -= 0.5
            tips.append("📝 Développez davantage votre argumentation")
 
        score = round(min(max(score, 0), 10), 1)
 
        # Réponse modèle depuis la KB
        model_response = None
        if product and objection_type:
            model_response = kb_loader.get_best_response(product, objection_type)
 
        return {
            "score": score,
            "tips": tips,
            "model_response": model_response,
            "label": "Excellent" if score >= 8.5 else "Bon" if score >= 7.0 else "À améliorer",
        }

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
    def detect_product(text: str) -> Optional[str]:
        """Détecte le produit mentionné dans le texte de la déléguée."""
        text_lower = text.lower()
        # Corrections des noms mal prononcés
        corrections = {
            "phytophan lossuant antichute": "PHYTOFANE Lotion anti chute",
            "phytophan": "PHYTOFANE Lotion anti chute",
            "phytofane": "PHYTOFANE Lotion anti chute",
            "multiband calcium vit détroit": "MULTIBON Calcium Vit.D3",
            "élevé cis-protect": "LV CYSPROTECT",
            "au mévis omega 3": "Omevie Omega 3",
        }
        
        for wrong, correct in corrections.items():
            if wrong in text_lower:
                print(f"[ALIA] Correction produit: '{wrong}' -> '{correct}'")
                return correct
        
        # Recherche normale
        for product_name in PRODUCT_CATALOG:
            if product_name.lower() in text_lower:
                return product_name
        # Fuzzy matching sur les mots-clés du nom
        for product_name , info in PRODUCT_CATALOG.items():
            for keyword in info.get('keywords', []):
                if keyword in text_lower:
                    return product_name
        return None
 
    @staticmethod
    def detect_objection_type(text: str) -> Optional[str]:
        """Détecte le type d'objection dans le texte du médecin/client."""
        text_lower = text.lower()
        best_match = None
        best_count = 0
        for obj_type, keywords in OBJECTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_match = obj_type
        return best_match if best_count > 0 else None

    @staticmethod
    def analyze(query: str) -> Dict:
        q = query.lower()
        conditions_map = {
            "anémie": ["anémie", "fer", "hémoglobine"],
            "fatigue": ["fatigue", "tonus", "énergie", "vitalité", "asthénie"],
            "stress": ["stress", "anxiété", "nervosité"],
            "sommeil": ["sommeil", "insomnie", "dormir"],
            "digestion": ["digestion", "digestif", "estomac", "intestin"],
            "peau": ["peau", "acné", "dermatologique"],
            "immunité": ["immunité", "défense", "rhume", "grippe", "gorge", "angine"],
            "articulation": ["articulation", "os", "cartilage", "mobilité"],
            "cheveux": ["cheveux", "chute", "alopécie", "capillaire"],
            "cardiovasculaire": ["cardiovasculaire", "triglycérides", "oméga", "omega"],
        }
        detected = [cond for cond, kws in conditions_map.items() if any(kw in q for kw in kws)]
        return {
            "target": "enfant" if any(x in q for x in ["enfant", "junior", "bébé", "pédiatrique"]) else "adulte",
            "conditions": detected,
            "product": QueryAnalyzer.detect_product(query),
            "objection_type": QueryAnalyzer.detect_objection_type(query),
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
 

class AliaOrchestrator:
    HISTORY_MAX_TURNS = 6
    _MEDICAL_KW = [
        "fatigue", "sommeil", "stress", "anémie", "digestion", "peau", "acné",
        "cheveux", "ongles", "articulation", "os", "immunité", "rhume", "grippe",
        "toux", "gorge", "angine", "allergie", "mémoire", "concentration", "vision",
        "circulation", "douleur", "vitamine", "complément", "traitement",
        "sirop", "comprimé", "gélule", "crème", "fer", "calcium", "magnésium", "zinc",
        "omega", "oméga", "urinaire", "cystite", "capillaire",
    ]

    # Questions du médecin en mode training — cycle varié
    _DOCTOR_QUESTIONS = [
        "Avez-vous des études cliniques qui prouvent son efficacité ?",
        "Quels sont les avantages par rapport aux génériques du marché ?",
        "Quels sont les effets secondaires possibles ?",
        "Quelle est la posologie recommandée pour un adulte ?",
        "Est-ce que ce produit est remboursé par la sécurité sociale ?",
        "En quoi est-il différent des autres compléments similaires ?",
        "Quel est le prix public conseillé pour le patient ?",
        "Y a-t-il des contre-indications importantes à connaître ?",
        "Quelle est la durée de traitement recommandée ?",
        "Existe-t-il une version adaptée pour les enfants ?",
        "Pouvez-vous me laisser de la documentation scientifique ?",
        "Quels types de patients bénéficieraient le plus de ce produit ?",
    ]

    def __init__(self, manager: KnowledgeManager, kb_loader: KnowledgeBaseLoader,mode: str = ALIA_MODE):
        self.manager = manager
        self.kb_loader = kb_loader
        self.mode = mode  # "commercial" ou "training"
        self.history: List[Dict] = []  # [{"role": "user"|"alia", "text": "..."}]
        self._turn = 0
        self._persona = self._pick_persona()
        self._asked_questions: List[str] = []
        self._current_product: Optional[str] = None
        self._last_objection_type: Optional[str] = None
        self._greeted = False

        print(f"[ALIA] Initializing LLM with model: {MODEL_SETTINGS['main']} | mode: {self.mode}")
        self.llm = OllamaLLM(
            model=MODEL_SETTINGS["main"],
            temperature=MODEL_SETTINGS["temperature"],
            num_predict=MODEL_SETTINGS["num_predict"]
        )
 
        self._build_chain()

    def _pick_persona(self) -> Dict:
        avatar = self.kb_loader.get_random_avatar() if self.kb_loader.avatars else {}
        if avatar:
            prenom = avatar.get("client_name", "Dr. Martin")
            specialty = avatar.get("specialty", "médecin généraliste")
            engagement = avatar.get("engagement_level", "Moyen")
            if engagement == "Faible":
                role = f"{specialty}, peu disponible et sceptique"
            elif engagement == "Fort":
                role = f"{specialty}, ouvert et intéressé par les nouveautés"
            else:
                role = f"{specialty}, attentif mais demande des preuves"
            return {"prenom": prenom, "role": role}
        return random.choice(PERSONAS)

 
    def _build_chain(self):
        template = PROMPT_TRAINING if self.mode == "training" else PROMPT_COMMERCIAL
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()
 


    def set_mode(self, mode: str):
        if mode not in ("training", "commercial"):
            raise ValueError("mode doit être 'training' ou 'commercial'")
        self.mode = mode
        self._greeted = True
        self.history = []
        self._turn = 0
        self._asked_questions = []
        self._current_product = None
        self._last_objection_type = None
        self._persona = self._pick_persona()
        
        print(f"[ALIA] Mode: {mode} | Persona: {self._persona['prenom']}")
 
    def get_greeting(self) -> Dict:
        """Appelé UNE SEULE FOIS par set_mode_view pour obtenir le message d'accueil."""
        self._greeted = True
        rep = (
            f"Bonjour, je suis {self._persona['prenom']}, {self._persona['role']}. "
            f"Quel produit souhaitez-vous me présenter aujourd'hui ?"
        )
        self._add_to_history("alia", rep)
        return {"text": rep, "intent": "greeting", "persona": self._persona["prenom"]}

    def reset(self):
        self.history = []
        self._turn = 0
        self._asked_questions = []
        self._current_product = None
        self._last_objection_type = None
        self._persona = self._pick_persona()
        print(f"[ALIA] Reset | Persona: {self._persona['prenom']}")
 
    def _is_vague_commercial(self, query: str) -> bool:
        q = query.lower()
        greetings = ["bonjour", "docteur", "déléguée", "présenter", "enchantée", "ravi","monsieur","madame"]
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
                prefix = "Délégué" if msg["role"] == "user" else self._persona["prenom"]
            else:
                prefix = "Délégué" if msg["role"] == "user" else "ALIA"
            lines.append(f"{prefix}: {msg['text']}")
        return "\n".join(lines)
 
    def _add_to_history(self, role: str, text: str):
        self.history.append({"role": role, "text": text})
        # Limite la taille en mémoire
        max_entries = self.HISTORY_MAX_TURNS * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def _clean(self, text: str) -> str:
        text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
        text = re.sub(r"</?[^>]+>", "", text)
        text = re.sub(r"Bonjour,?\s*(madame|monsieur)?\s*la?\s*délégu[eé]e?[.,]?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _next_doctor_question(self) -> str:
        for q in self._DOCTOR_QUESTIONS:
            if q not in self._asked_questions:
                self._asked_questions.append(q)
                return q
        self._asked_questions = []
        return self._DOCTOR_QUESTIONS[0]
 
    def _build_training_feedback(self, rep_response: str) -> Optional[Dict]:
        """Génère un feedback structuré sur la réponse de la déléguée."""
        if not self._current_product:
            return None
        return TrainingEvaluator.evaluate(
            rep_response,
            self._current_product,
            self._last_objection_type,
            self.kb_loader,
        )


    def _humanize(self, text: str) -> str:
        text = text.replace("Il est recommandé", "Je vous conseille")
        text = text.replace("Vous pouvez utiliser", "Vous pouvez prendre")
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
        #self._turn += 1
        print(f"[ALIA] turn={self._turn} mode={self.mode} | {query[:80]}")
    
        # Vérifier si l'utilisateur parle de Vital SA
        query_lower = query.lower()
        if "vital" not in query_lower and "vita" not in query_lower:
            if "vita lessa" in query_lower:
                query = query.replace("Vita Lessa", "Vital SA").replace("vita lessa", "Vital SA")
                query = query.replace("vitales", "Vital SA")
                print(f"[ALIA] Correction: {query}")

        try:
            # ── MODE TRAINING ──────────────────────────────────────────
            if self.mode == "training":
                if self._turn == 0 and len(self.history) == 0:
                    self._turn = 1
                    rep = (
                        f"Bonjour, je suis {self._persona['prenom']}, {self._persona['role']}. "
                        f"Quel produit Vital SA souhaitez-vous me présenter aujourd'hui ?"
                    )
                    self._add_to_history("user", query)
                    self._add_to_history("alia", rep)
                    return {"text": rep, "intent": "greeting"}
                
                # Incrémenter le tour seulement pour les vrais messages
                self._turn += 1
                print(f"[ALIA] turn={self._turn} mode={self.mode} | {query[:80]}")

                # Détecter le produit dans la réponse de la déléguée
                detected_product = QueryAnalyzer.detect_product(query)
                if detected_product:
                    self._current_product = detected_product
                    print(f"[ALIA] Produit détecté: {detected_product}")
                
                # Le médecin pose la prochaine question
                if self._current_product:
                    idx = (self._turn - 2) % len(self._DOCTOR_QUESTIONS)
                    rep = self._DOCTOR_QUESTIONS[idx]
                    self._last_objection_type = QueryAnalyzer.detect_objection_type(rep)
                else:
                    product_found = False
                    for product_name in PRODUCT_CATALOG:
                        if product_name.lower() in query_lower:
                            self._current_product = product_name
                            rep = self._DOCTOR_QUESTIONS[0]
                            product_found = True
                            break
                    if not product_found:
                        rep = (
                            "Je ne connais pas ce produit dans la gamme Vital SA. "
                            "Pouvez-vous me présenter un produit de notre catalogue ? "
                            "Nous avons par exemple : OLIGOVIT Trio, LV CYSPROTECT, LV Fersang junior, Omevie Omega 3."
                        )
                
                self._add_to_history("user", query)
                self._add_to_history("alia", rep)
                print(f"[ALIA] Doctor says: {rep}")
                
                result: Dict = {"text": rep, "intent": "response", "persona": self._persona["prenom"]}
                return result
    
            # ── MODE COMMERCIAL ────────────────────────────────────────
            else:
                if self._is_vague_commercial(query):
                    rep = "Pouvez-vous préciser votre besoin ? Fatigue, sommeil, gorge, peau, infections urinaires, ou autre pathologie ?"
                    self._add_to_history("user", query)
                    self._add_to_history("alia", rep)
                    return {"text": rep, "intent": "clarification"}
    
                docs = await asyncio.to_thread(self.manager.retriever.invoke, query)
                print(f"[ALIA] docs retrieved: {len(docs)}")
    
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
                kwargs = {"context": context, "query": query, "history": self._format_history()}
    
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.chain.invoke, kwargs),
                    timeout=120,
                )
                cleaned = self._clean(response)
    
                if not cleaned or len(cleaned.split()) < 5:
                    analysis = QueryAnalyzer.analyze(query)
                    conditions = analysis.get("conditions", [])
                    cleaned = (
                        f"Je n'ai pas trouvé de produit exact pour {', '.join(conditions)}."
                        if conditions else "Pouvez-vous préciser votre demande ?"
                    )
                else:
                    cleaned = self._humanize(cleaned)
    
                self._add_to_history("user", query)
                self._add_to_history("alia", cleaned)
                return {"text": cleaned, "intent": "response"}
    
        except asyncio.TimeoutError:
            fallback = random.choice([
                "Pouvez-vous me donner plus de détails sur ce produit ?",
                "Quels sont les résultats des études cliniques ?",
                "Comment se compare-t-il aux autres produits ?",
            ])
            return {"text": fallback, "intent": "timeout"}
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return {"text": "Je n'ai pas bien compris. Pouvez-vous reformuler ?", "intent": "error"}


# ══════════════════════════════════════════════════════════════════════
# FACTORY — point d'entrée pour Django et CLI
# ══════════════════════════════════════════════════════════════════════
 
def build_alia(
    data_dir: Path = _DATA_DIR,
    mode: str = ALIA_MODE,
    csv_name: str = "vital_products.csv",
) -> AliaOrchestrator:
    """
    Construit et retourne une instance AliaOrchestrator prête à l'emploi.
 
    Args:
        data_dir  : répertoire contenant vital_products.csv et/ou les fichiers prepared_data
        mode      : 'training' ou 'commercial'
        csv_name  : nom du fichier CSV principal des produits (optionnel)
    """
    # 1. Charger la KB enrichie depuis prepared_data
    kb_loader = KnowledgeBaseLoader(data_dir)
    kb_loader.load()
 
    # 2. Construire les documents vectoriels
    csv_path = data_dir / csv_name
    docs = []
    
    if csv_path.exists():
        docs = DataProcessor.build_documents_from_csv(csv_path)
    
    if not docs:
        docs = DataProcessor.build_documents_from_catalog()
    
    print(f"[build_alia] {len(docs)} product documents built")
 
    # 3. Indexer
    manager = KnowledgeManager()
    if docs:
        manager.load_or_create(docs)
 
    # 4. Créer l'orchestrateur
    alia = AliaOrchestrator(manager, kb_loader, mode=mode)
    return alia
 
 
# =========================
# MAIN (USAGE)
# =========================
async def main():
    import sys
 
    data_dir = _DATA_DIR
    # Si lancé depuis prepared_data pour test, utiliser ce répertoire
    if (Path(".") / "conversations_dso2_knowledge_base.csv").exists():
        data_dir = Path(".")
 
    alia = build_alia(data_dir=data_dir, mode="training")
 
    print("\n=== ALIA v2 Ready ===")
    print(f"Persona: {alia._persona['prenom']} — {alia._persona['role']}\n")
 
    if "--interactive" in sys.argv or "-i" in sys.argv:
        print("Mode interactif. Tapez 'quit' pour quitter, 'reset' pour recommencer.\n")
        while True:
            q = input("Déléguée: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q.lower() == "reset":
                alia.reset()
                print(f"[Reset] Nouveau persona: {alia._persona['prenom']}\n")
                continue
            res = await alia.generate(q)
            print(f"\n{alia._persona['prenom']}: {res['text']}")
            if res.get("feedback"):
                fb = res["feedback"]
                print(f"\n  ── FEEDBACK ALIA ──")
                print(f"  Score estimé : {fb['score']}/10 ({fb['label']})")
                for tip in fb["tips"]:
                    print(f"  {tip}")
                if fb.get("model_response"):
                    print(f"  📌 Réponse modèle : {fb['model_response'][:200]}...")
                print()
    else:
        # Démo automatique
        scenarios = [
            ("Je viens vous présenter Vitonic Tonus pour la fatigue.", "training"),
            ("J'ai des doutes sur son efficacité, avez-vous des preuves ?", "training"),
            ("Le prix me semble élevé par rapport aux génériques.", "training"),
            ("Quels produits avez-vous pour les infections urinaires ?", "commercial"),
            ("Un produit pour l'anémie chez l'enfant ?", "commercial"),
        ]
 
        for query, mode in scenarios:
            print(f"\n{'='*60}")
            if alia.mode != mode:
                alia.set_mode(mode)
            print(f"[{mode.upper()}] Déléguée: {query}")
            res = await alia.generate(query)
            speaker = alia._persona["prenom"] if mode == "training" else "ALIA"
            print(f"{speaker}: {res['text']}")
            if res.get("feedback"):
                fb = res["feedback"]
                print(f"  → Score: {fb['score']}/10 | {fb['label']}")
                for tip in fb["tips"]:
                    print(f"  → {tip}")


if __name__ == "__main__":
    asyncio.run(main())