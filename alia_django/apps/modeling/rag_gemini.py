import asyncio
import os
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
    "temperature": 0.1, 
    "num_predict": 400
}

_DATA_DIR = Path(__file__).resolve().parent / "data"
PERSIST_DIR = str(_DATA_DIR / "alia_knowledge_db")


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
        
        if os.path.exists(self.persist_dir):
            print(f"[KB] Found existing database at {self.persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            print(f"[KB] Creating new database at {self.persist_dir}")
            self.vectorstore = Chroma.from_documents(
                docs,
                self.embeddings,
                persist_directory=self.persist_dir
            )

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
            "immunité": ["immunité", "défense", "rhume", "grippe"],
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
class AliaOrchestrator:
    def __init__(self, manager: KnowledgeManager):
        self.manager = manager

        print(f"[ALIA] Initializing LLM with model: {MODEL_SETTINGS['main']}")
        self.llm = OllamaLLM(
            model=MODEL_SETTINGS["main"],
            temperature=MODEL_SETTINGS["temperature"],
            num_predict=MODEL_SETTINGS["num_predict"]
        )

        self.prompt = ChatPromptTemplate.from_template("""
Tu es ALIA, assistant pharmaceutique.

ANALYSE LA QUESTION: "{query}"

PRODUITS DISPONIBLES:
{context}

RÈGLES STRICTES:
1. Vérifie que les produits ci-dessus correspondent EXACTEMENT à la question
2. Si AUCUN produit ne correspond à la condition demandée, dis: "Je n'ai pas de produit spécifique pour [condition] dans ma base actuelle."
3. Si des produits correspondent, recommande LE MEILLEUR (1 seul)
4. Structure ta réponse:
   - Produit recommandé: "[Nom exact]"
   - Pourquoi: [1-2 phrases]
   - Mode d'emploi: [oral/topique]
   - Note si version enfant existe

JAMAIS de réponse hors-sujet. Si la question est sur l'anémie, ne parle PAS de rhume.

Réponse:
""")

        self.chain = self.prompt | self.llm | StrOutputParser()

    def _clean(self, text: str) -> str:
        # Remove any thinking tags or markup
        text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
        text = re.sub(r"</?[^>]+>", "", text)
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

    async def generate(self, query: str):
        try:
            print(f"[ALIA] Processing query: {query}")
            
            # Analyze query first
            analysis = QueryAnalyzer.analyze(query)
            print(f"[ALIA] Query analysis: {analysis}")
            
            # Retrieve documents
            print("[ALIA] Retrieving relevant documents...")
            docs = await asyncio.to_thread(self.manager.retriever.invoke, query)
            print(f"[ALIA] Retrieved {len(docs)} documents")

            if not docs:
                print("[ALIA] No documents found!")
                return {
                    "text": self._smart_fallback(query, [], analysis),
                    "intent": "no_results"
                }

            # Check relevance
            is_relevant = self._check_relevance(query, docs)
            print(f"[ALIA] Documents relevant: {is_relevant}")
            
            # Show top retrieved products for debugging
            print("[ALIA] Top 3 retrieved products:")
            for i, doc in enumerate(docs[:3]):
                product_name = doc.metadata.get('name', 'Unknown')
                print(f"  {i+1}. {product_name}")

            context = "\n\n---\n\n".join(d.page_content for d in docs[:5])
            print(f"[ALIA] Context length: {len(context)} chars")

            # Generate response with timeout
            print("[ALIA] Generating response...")
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.chain.invoke,
                    {"context": context, "query": query}
                ),
                timeout=30
            )
            
            print(f"[ALIA] Raw response: {response[:200]}...")

            # Clean and humanize
            cleaned = self._clean(response)
            
            # If response is too short or seems off-topic, use fallback
            if not cleaned or len(cleaned.split()) < 10:
                print("[ALIA] Response too short, using fallback")
                cleaned = self._smart_fallback(query, docs, analysis)
            elif not is_relevant:
                print("[ALIA] Documents not relevant enough, using fallback")
                cleaned = self._smart_fallback(query, docs, analysis)

            cleaned = self._humanize(cleaned)
            print(f"[ALIA] Final response: {cleaned}")

            return {
                "text": cleaned,
                "intent": "recommendation"
            }

        except asyncio.TimeoutError:
            print("[ALIA] Timeout error!")
            return {
                "text": "Désolée, le traitement de votre demande a pris trop de temps. Pouvez-vous reformuler ?",
                "intent": "timeout"
            }
        except Exception as e:
            print(f"[ALIA] Error during generation: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "text": "Désolée, j'ai rencontré une erreur technique. Veuillez réessayer.",
                "intent": "error"
            }


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