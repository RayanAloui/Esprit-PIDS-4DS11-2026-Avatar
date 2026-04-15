import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional  # Added Optional import

import pandas as pd

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
    "num_predict": 150
}

_DATA_DIR = Path(__file__).resolve().parent / "data"
PERSIST_DIR = str(_DATA_DIR / "alia_knowledge_db")

sys.path.append(str(Path(__file__).resolve().parent))

# Import the PowerPoint generation module
try:
    from powerpoint_generation import generate_presentation_for_product
    PPT_AVAILABLE = True
except ImportError as e:
    print(f"[ALIA] PowerPoint generation not available: {e}")
    PPT_AVAILABLE = False

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

    def find_product_by_name(self, product_name: str) -> Optional[Document]:
        try:
            if not product_name:
                return None
                
            product_name_lower = product_name.lower().strip()
            print(f"[KB] Searching for product: '{product_name_lower}'")
            
            # Get all products from vectorstore (up to 50 for comprehensive search)
            results = self.vectorstore.similarity_search(
                product_name_lower,
                k=50,
                filter=None
            )
            
            if not results:
                print("[KB] No products found in vectorstore")
                return None
            
            # Extract unique products with their metadata
            unique_products = {}
            for doc in results:
                name = doc.metadata.get('name', '')
                if name and name not in unique_products:
                    unique_products[name] = doc
            
            print(f"[KB] Found {len(unique_products)} unique products to search through")
            
            # Try different matching strategies
            matches = []
            
            for doc_name, doc in unique_products.items():
                doc_name_lower = doc_name.lower()
                score = 0
                
                # Strategy 1: Exact match (highest score)
                if doc_name_lower == product_name_lower:
                    matches.append((doc, 1.0, "exact"))
                    continue
                
                # Strategy 2: Product name is substring of document name
                if product_name_lower in doc_name_lower:
                    score = len(product_name_lower) / len(doc_name_lower)
                    matches.append((doc, score, "substring"))
                
                # Strategy 3: Document name is substring of product name
                elif doc_name_lower in product_name_lower:
                    score = len(doc_name_lower) / len(product_name_lower)
                    matches.append((doc, score, "contains"))
                
                # Strategy 4: Word overlap (Jaccard similarity)
                else:
                    product_words = set(product_name_lower.split())
                    doc_words = set(doc_name_lower.split())
                    
                    if product_words and doc_words:
                        intersection = product_words & doc_words
                        union = product_words | doc_words
                        
                        if intersection:
                            # Jaccard similarity
                            jaccard = len(intersection) / len(union)
                            
                            # Also check if all product words appear in doc (in any order)
                            all_words_present = all(word in doc_name_lower for word in product_words)
                            
                            if all_words_present:
                                score = 0.7 + (0.3 * jaccard)  # Boost if all words are present
                            else:
                                score = 0.4 * jaccard  # Lower score for partial word matches
                            
                            if score > 0.3:  # Minimum threshold
                                matches.append((doc, score, f"word_overlap_{len(intersection)}"))
                
                # Strategy 5: Levenshtein distance for similar strings
                if not matches or max(m[1] for m in matches) < 0.5:
                    # Only compute if we don't have good matches yet
                    distance_ratio = self._string_similarity(product_name_lower, doc_name_lower)
                    if distance_ratio > 0.6:  # 60% similar
                        matches.append((doc, distance_ratio * 0.8, "similar"))  # Slightly lower weight
            
            # Sort matches by score (descending)
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Log all potential matches for debugging
            print(f"[KB] Top matches for '{product_name_lower}':")
            for doc, score, strategy in matches[:5]:
                print(f"  - {doc.metadata.get('name')} (score: {score:.3f}, strategy: {strategy})")
            
            # Return the best match if score is above threshold
            if matches and matches[0][1] >= 0.4:  # 40% confidence threshold
                best_doc, best_score, best_strategy = matches[0]
                print(f"[KB] Selected: {best_doc.metadata.get('name')} (score: {best_score:.3f}, strategy: {best_strategy})")
                return best_doc
            
            print(f"[KB] No suitable match found for '{product_name_lower}' (best score: {matches[0][1] if matches else 0:.3f})")
            return None
        
        except Exception as e:
            print(f"[KB] Error finding product: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _string_similarity(self, s1: str, s2: str) -> float:
        try:
            # Simple Levenshtein implementation
            if len(s1) < len(s2):
                s1, s2 = s2, s1
            
            if len(s2) == 0:
                return 0.0
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            distance = previous_row[-1]
            max_len = max(len(s1), len(s2))
            similarity = 1 - (distance / max_len)
            return similarity
        except:
            return 0.0


# =========================
# QUERY UNDERSTANDING
# =========================
class QueryAnalyzer:
    @staticmethod
    def analyze(query: str) -> Dict:
        q = query.lower().strip()

        greeting_keywords = [
            "bonjour", "salut", "bonsoir", "hello", "hi", "hey",
            "ça va", "cv", "salam"
        ]

        casual_keywords = [
            "merci", "ok", "d'accord", "oui", "non", "bien",
            "super", "parfait"
        ]

        presentation_keywords = [
            "présentation", "powerpoint", "ppt", "diaporama", "slides",
            "créer", "générer", "présenter", "générez", "créez"
        ]

        greeting_matches = [word for word in greeting_keywords if word in q]

        is_greeting = (
            len(greeting_matches) > 0
            and len(q.split()) <= 3
        )
        is_casual = any(word in q for word in casual_keywords)
        is_presentation_request = any(
    re.search(rf"\b{re.escape(word)}\b", q)
    for word in presentation_keywords
)

        product_name = None
        if is_presentation_request:
            product_name = QueryAnalyzer._extract_product_name_regex(q)

        conditions = {
            "anémie": ["anémie", "fer", "hémoglobine", "fatigue extrême"],
            "fatigue": ["fatigue", "tonus", "énergie", "vitalité"],
            "stress": ["stress", "anxiété", "nervosité", "tension"],
            "sommeil": ["sommeil", "insomnie", "dormir", "repos"],
            "digestion": ["digestion", "digestif", "estomac", "intestin"],
            "peau": ["peau", "acné", "dermatologique", "cutané"],
            "immunité": ["immunité", "défense", "rhume", "grippe", "gorge", "angine"],
            "articulation": ["articulation", "os", "cartilage", "mobilité"]
        }

        detected_conditions = []
        for condition, keywords in conditions.items():
            if any(kw in q for kw in keywords):
                detected_conditions.append(condition)

        is_medical_query = (
            len(detected_conditions) > 0
            or any(x in q for x in [
                "traitement", "soigner", "guérir",
                "médicament", "produit", "symptôme"
            ])
        )

        return {
            "is_greeting": is_greeting,
            "is_casual": is_casual,
            "is_medical_query": is_medical_query,
            "is_presentation_request": is_presentation_request,
            "product_name": product_name,
            "conditions": detected_conditions,
            "target": "enfant" if any(
                x in q for x in ["enfant", "junior", "bébé", "pédiatrique"]
            ) else "adulte"
        }
    
    @staticmethod
    def _extract_product_name_regex(text: str) -> Optional[str]:
        """Extract product name using various regex patterns"""
        text = text.lower().strip()
        
        # List of patterns to try, from most specific to most general
        patterns = [
            # Pattern 1: "présentation [preposition] X" where X can be anything
            r'(?:présentation|powerpoint|ppt|diaporama)\s+(?:pour|de|du|des|sur|d\')\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            
            # Pattern 2: "présenter X"
            r'présenter\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            
            # Pattern 3: "créer/générer [une] [présentation] [pour] X"
            r'(?:créer|générer|créez?|générez?)\s+(?:une\s+)?(?:présentation|powerpoint|ppt)?\s*(?:pour|de|du|des|sur|d\')?\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            
            # Pattern 4: "je veux/voudrais une présentation [pour] X"
            r'(?:je\s+)?(?:veux|voudrais|aimerais|souhaite)\s+(?:une\s+)?(?:présentation|powerpoint|ppt)\s+(?:pour|de|du|des|sur|d\')\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            
            # Pattern 5: "peux-tu me faire une présentation sur X"
            r'(?:peux-?tu|pouvez-?vous)\s+(?:me\s+)?(?:faire|créer|générer)\s+(?:une\s+)?(?:présentation|powerpoint|ppt)\s+(?:pour|de|du|des|sur|d\')\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            
            # Pattern 6: Anything after "présentation" that's not a stop word
            r'présentation\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_name = match.group(1).strip()
                
                # Clean up the extracted name
                # Remove leading articles and prepositions
                raw_name = re.sub(r'^(une|un|la|le|les|des|pour|de|du|sur|d\')\s+', '', raw_name, flags=re.IGNORECASE)
                
                # Remove trailing punctuation
                raw_name = raw_name.rstrip('?.!,; ')
                
                if raw_name and len(raw_name) > 1:
                    print(f"[QueryAnalyzer] Pattern {i+1} matched: '{raw_name}'")
                    return raw_name
        
        return None


# =========================
# ORCHESTRATOR
# =========================
ALIA_MODE = "commercial"   # valeur par défaut
 
PROMPT_COMMERCIAL = """Tu es ALIA, assistant pharmaceutique d'ne délégué médicale.
Tu aides le délégué pendant sa visite chez un médecin ou un pharmacien.
 
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
 
PROMPT_TRAINING = """Tu es ALIA, coach de formation pour délégués médicaux.
Tu joues le rôle d'un médecin ou pharmacien exigeant.
Tu évalues les réponses du délégué et tu l'aides à progresser.
 
HISTORIQUE DE LA CONVERSATION:
{history}
 
CE QUE DIT LE DÉLÉGUÉ: "{query}"
 
PRODUITS DISPONIBLES (pour vérifier si le délégué a raison):
{context}
 
RÈGLES:
1. Si le délégué donne une bonne information sur un produit → félicite-le brièvement et enrichis.
2. Si le délégué fait une erreur → corrige avec bienveillance et donne la bonne réponse.
3. Si le délégué hésite → encourage-le et guide-le avec une question ou un indice.
4. Pose une question de suivi pour tester davantage ses connaissances.
5. Tiens compte de l'historique pour ne pas répéter les mêmes corrections.
 
Réponse:
"""


class AliaOrchestrator:
    HISTORY_MAX_TURNS = 5
    
    def __init__(self, manager: KnowledgeManager, mode: str = ALIA_MODE, csv_path: Optional[Path] = None):
        self.manager = manager
        self.mode = mode
        self.history: List[Dict] = []
        self.csv_path = csv_path or _DATA_DIR / "vital_products.csv"
        
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
        """Change le mode à chaud et réinitialise l'historique."""
        if mode not in ("training", "commercial"):
            raise ValueError("mode doit être 'training' ou 'commercial'")
        self.mode = mode
        self.history = []
        self._build_chain()
        print(f"[ALIA] Mode changé : {mode}")
 
    def _format_history(self) -> str:
        """Formate les N derniers tours pour le prompt."""
        turns = self.history[-(self.HISTORY_MAX_TURNS * 2):]
        if not turns:
            return "Aucun historique (début de conversation)."
        lines = []
        for msg in turns:
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

    async def generate_presentation(self, product_name: str) -> Dict:
        """Generate a PowerPoint presentation for a specific product"""
        if not PPT_AVAILABLE:
            return {
                "text": "Désolée, la génération de présentations PowerPoint n'est pas disponible actuellement.",
                "intent": "presentation_error",
                "presentation_path": None
            }
        
        try:
            print(f"[ALIA] Generating presentation for: {product_name}")
            
            # Generate the presentation
            output_path = await asyncio.to_thread(
                generate_presentation_for_product,
                product_name,
                self.csv_path
            )
            
            # Create a user-friendly response
            response = f"J'ai généré la présentation PowerPoint pour {product_name} !\n\n"
            response += "La présentation comprend :\n"
            response += "• Une slide d'accroche avec le positionnement produit\n"
            response += "• L'analyse de la problématique patient\n"
            response += "• La solution experte Vital Labs\n"
            response += "• Les 4 piliers stratégiques\n"
            response += "• Les spécifications techniques\n"
            response += "• Une conclusion avec appel à l'action\n\n"
            response += "Voulez-vous que je vous présente le contenu ou que je génère une présentation pour un autre produit ?"
            
            return {
                "text": response,
                "intent": "presentation_generated",
                "presentation_path": str(output_path)
            }
            
        except ValueError as e:
            # Product not found
            return {
                "text": f"Je n'ai pas trouvé le produit '{product_name}' dans notre catalogue. "
                       f"Pouvez-vous vérifier le nom exact du produit ?",
                "intent": "presentation_error",
                "presentation_path": None
            }
        except Exception as e:
            print(f"[ALIA] Presentation generation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "Désolée, une erreur s'est produite lors de la génération de la présentation. "
                       "Veuillez réessayer avec un autre produit.",
                "intent": "presentation_error",
                "presentation_path": None
            }
    def _find_similar_products(self, product_name: str, limit: int = 5) -> List[str]:
        """Find similar products when exact match fails"""
        try:
            if not product_name:
                return []
            
            product_name_lower = product_name.lower().strip()
            
            # Get a sample of products from the vectorstore
            all_docs = self.manager.vectorstore.similarity_search(
                product_name_lower,
                k=20,
                filter=None
            )
            
            if not all_docs:
                return []
            
            # Extract unique product names
            unique_products = {}
            for doc in all_docs:
                name = doc.metadata.get('name', '')
                if name and name not in unique_products:
                    unique_products[name] = doc
            
            # Score each product for similarity
            scored_products = []
            search_terms = set(product_name_lower.split())
            
            for name, doc in unique_products.items():
                name_lower = name.lower()
                score = 0
                
                # Check word overlap
                name_terms = set(name_lower.split())
                overlap = search_terms & name_terms
                
                if overlap:
                    # Score based on number of matching words
                    score = len(overlap) * 0.5
                    
                    # Boost for products containing "fer" if searching for iron-related
                    if "fer" in search_terms and "fer" in name_lower:
                        score += 1.0
                    
                    # Boost for products containing "sang" if searching for blood-related
                    if "sang" in search_terms and "sang" in name_lower:
                        score += 1.0
                
                # Also check Levenshtein similarity for typos
                similarity = self.manager._string_similarity(product_name_lower, name_lower)
                score += similarity * 0.3
                
                if score > 0.3:  # Minimum threshold
                    scored_products.append((name, score))
            
            # Sort by score and return top results
            scored_products.sort(key=lambda x: x[1], reverse=True)
            return [name for name, _ in scored_products[:limit]]
            
        except Exception as e:
            print(f"[ALIA] Error finding similar products: {e}")
            return []

    async def generate(self, query: str):
        try:
            analysis = QueryAnalyzer.analyze(query)
            print(f"[ALIA] Query analysis: {analysis}")

            # =========================
            # GREETING
            # =========================
            if analysis.get("is_greeting"):
                response = "Bonjour, je suis ALIA. Comment puis-je vous aider aujourd’hui ?"
                self._add_to_history("user", query)
                self._add_to_history("alia", response)
                return {"text": response, "intent": "greeting"}

            # =========================
            # CASUAL
            # =========================
            if analysis.get("is_casual"):
                response = "Je vous en prie. Avez-vous besoin d’informations sur un produit ou d’une présentation ?"
                self._add_to_history("user", query)
                self._add_to_history("alia", response)
                return {"text": response, "intent": "casual"}

            # =========================
            # PRESENTATION REQUEST
            # =========================
            if analysis.get("is_presentation_request"):
                product_name = analysis.get("product_name")

                if not product_name:
                    response = "Pour quel produit souhaitez-vous générer une présentation PowerPoint ?"
                    self._add_to_history("user", query)
                    self._add_to_history("alia", response)
                    return {"text": response, "intent": "clarification"}

                product_doc = self.manager.find_product_by_name(product_name)

                if not product_doc:
                    similar_products = self._find_similar_products(product_name)

                    response = f"Je n'ai pas trouvé exactement '{product_name}' dans notre base.\n\n"

                    if similar_products:
                        response += "Produits similaires disponibles :\n"
                        for prod in similar_products[:5]:
                            response += f"• {prod}\n"
                        response += "\nSouhaitez-vous une présentation pour l'un de ces produits ?"
                    else:
                        response += "Pouvez-vous vérifier le nom du produit ?"

                    self._add_to_history("user", query)
                    self._add_to_history("alia", response)
                    return {"text": response, "intent": "clarification"}

                actual_product_name = product_doc.metadata.get("name")
                self._add_to_history("user", query)

                result = await self.generate_presentation(actual_product_name)

                self._add_to_history("alia", result["text"])
                return result

            # =========================
            # VAGUE QUERY DETECTION
            # =========================
            mots_vagues = [
                "nouveau",
                "nouveauté",
                "quoi de neuf",
                "qu'est-ce que vous avez",
                "vous avez quoi",
                "un produit",
                "quelque chose"
            ]

            query_lower = query.lower()

            est_vague = (
                len(query.split()) < 12
                and any(m in query_lower for m in mots_vagues)
                and not any(
                    m in query_lower
                    for m in [
                        "cheveux",
                        "fatigue",
                        "peau",
                        "sommeil",
                        "anémie",
                        "digestion",
                        "stress",
                        "os",
                        "gorge",
                        "toux",
                        "rhume",
                        "grippe",
                        "articulation",
                        "vision",
                        "mémoire"
                    ]
                )
            )

            if est_vague:
                response = (
                    "Pouvez-vous préciser votre besoin ? "
                    "Par exemple : avez-vous un problème particulier comme "
                    "la fatigue, le sommeil, la peau, ou une pathologie spécifique ?"
                )

                self._add_to_history("user", query)
                self._add_to_history("alia", response)

                return {"text": response, "intent": "clarification"}

            # =========================
            # PRODUCT RECOMMENDATION FLOW
            # =========================
            print(f"[ALIA] Processing query: {query}")
            print("[ALIA] Retrieving relevant documents...")

            docs = await asyncio.to_thread(self.manager.retriever.invoke, query)

            print(f"[ALIA] Retrieved {len(docs)} documents")

            if not docs:
                response = self._smart_fallback(query, [], analysis)

                self._add_to_history("user", query)
                self._add_to_history("alia", response)

                return {"text": response, "intent": "no_results"}

            is_relevant = self._check_relevance(query, docs)

            print(f"[ALIA] Documents relevant: {is_relevant}")

            print("[ALIA] Top 3 retrieved products:")
            for i, doc in enumerate(docs[:3]):
                print(f"  {i+1}. {doc.metadata.get('name', 'Unknown')}")

            context = "\n\n---\n\n".join(d.page_content for d in docs[:5])

            history_str = self._format_history()

            print(f"[ALIA] Context length: {len(context)} chars")
            print("[ALIA] Generating response...")

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.chain.invoke,
                    {
                        "context": context,
                        "query": query,
                        "history": history_str
                    }
                ),
                timeout=120
            )

            print(f"[ALIA] Raw response: {response[:200]}...")

            cleaned = self._clean(response)

            if not cleaned or len(cleaned.split()) < 10:
                print("[ALIA] Response too short, using fallback")
                cleaned = self._smart_fallback(query, docs, analysis)

            elif not is_relevant:
                print("[ALIA] Documents not relevant enough, using fallback")
                cleaned = self._smart_fallback(query, docs, analysis)

            cleaned = self._humanize(cleaned)

            print(f"[ALIA] Final response: {cleaned}")

            self._add_to_history("user", query)
            self._add_to_history("alia", cleaned)

            return {"text": cleaned, "intent": "recommendation"}

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

    alia = AliaOrchestrator(manager, csv_path=csv_path)

    print("\n=== ALIA Ready ===\n")

    # Test with various queries including presentation
    test_queries = [
        "Quel est votre traitement pour l'anémie ?",
        "produit pour la fatigue",
        "quelque chose pour le rhume",
        "Peux-tu me créer une présentation powerpoint pour VITAL JUNIOR ?",
        "Je voudrais une présentation du produit VITAL FORCE",
        "Générez une présentation pour alcohol de montes"  # Test with the query from logs
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print('='*60)
        res = await alia.generate(q)
        print(f"\nALIA: {res['text']}")
        if res.get('presentation_path'):
            print(f"\nPresentation saved to: {res['presentation_path']}")

    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit")
    print("You can ask for presentations like: 'Crée une présentation pour [produit]'\n")
    
    while True:
        q = input("\nUser: ")
        if q.lower() in ['quit', 'exit', 'q']:
            break
            
        res = await alia.generate(q)
        print("ALIA:", res["text"])
        if res.get('presentation_path'):
            print(f"\n📊 Presentation saved to: {res['presentation_path']}")


if __name__ == "__main__":
    asyncio.run(main())