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
            # FR
            "bonjour", "salut", "bonsoir", "coucou",
            # EN
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            # ES
            "hola", "buenos días", "buenas tardes", "buenas noches",
            # AR
            "مرحبا", "السلام عليكم", "أهلا", "صباح الخير", "مساء الخير",
            # common
            "ça va", "cv", "salam",
        ]

        casual_keywords = [
            # FR
            "merci", "ok", "d'accord", "oui", "non", "bien", "super", "parfait",
            # EN
            "thank you", "thanks", "yes", "no", "okay", "great", "perfect", "fine",
            # ES
            "gracias", "sí", "no", "de acuerdo", "perfecto", "genial",
            # AR
            "شكرا", "نعم", "لا", "حسنا", "ممتاز", "تمام",
        ]

        presentation_keywords = [
            # FR
            "présentation", "powerpoint", "ppt", "diaporama", "slides",
            "créer", "générer", "présenter", "générez", "créez",
            # EN
            "presentation", "create", "generate", "present", "slideshow",
            # ES
            "presentación", "crear", "generar", "diapositivas",
            # AR
            "تقديم", "عرض", "إنشاء", "شرائح",
        ]

        greeting_matches = [word for word in greeting_keywords if word in q]

        is_greeting = (
            len(greeting_matches) > 0
            and len(q.split()) <= 4
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
            "anémie": ["anémie", "fer", "hémoglobine", "fatigue extrême",
                       "anemia", "iron", "hemoglobin",  # EN
                       "hierro", "hemoglobina",  # ES
                       "فقر الدم", "حديد",  # AR
                       ],
            "fatigue": ["fatigue", "tonus", "énergie", "vitalité",
                        "tiredness", "energy", "vitality", "exhaustion",  # EN
                        "cansancio", "energía", "vitalidad",  # ES
                        "تعب", "إرهاق", "طاقة",  # AR
                        ],
            "stress": ["stress", "anxiété", "nervosité", "tension",
                       "anxiety", "nervousness",  # EN
                       "ansiedad", "nerviosismo", "estrés",  # ES
                       "توتر", "قلق",  # AR
                       ],
            "sommeil": ["sommeil", "insomnie", "dormir", "repos",
                        "sleep", "insomnia",  # EN
                        "sueño", "insomnio", "dormir",  # ES
                        "نوم", "أرق",  # AR
                        ],
            "digestion": ["digestion", "digestif", "estomac", "intestin",
                          "stomach", "intestine", "digestive",  # EN
                          "estómago", "intestino", "digestión",  # ES
                          "هضم", "معدة", "أمعاء",  # AR
                          ],
            "peau": ["peau", "acné", "dermatologique", "cutané",
                     "skin", "acne", "dermatological",  # EN
                     "piel", "acné", "dermatológico",  # ES
                     "جلد", "بشرة", "حب الشباب",  # AR
                     ],
            "immunité": ["immunité", "défense", "rhume", "grippe", "gorge", "angine",
                         "immunity", "defense", "cold", "flu", "throat",  # EN
                         "inmunidad", "defensa", "resfriado", "gripe", "garganta",  # ES
                         "مناعة", "دفاع", "برد", "إنفلونزا", "حلق",  # AR
                         ],
            "articulation": ["articulation", "os", "cartilage", "mobilité",
                             "joint", "bone", "cartilage", "mobility",  # EN
                             "articulación", "hueso", "cartílago", "movilidad",  # ES
                             "مفصل", "عظم", "غضروف",  # AR
                             ],
        }

        detected_conditions = []
        for condition, keywords in conditions.items():
            if any(kw in q for kw in keywords):
                detected_conditions.append(condition)

        is_medical_query = (
            len(detected_conditions) > 0
            or any(x in q for x in [
                # FR
                "traitement", "soigner", "guérir", "médicament", "produit", "symptôme",
                # EN
                "treatment", "cure", "heal", "medicine", "product", "symptom",
                # ES
                "tratamiento", "curar", "medicamento", "producto", "síntoma",
                # AR
                "علاج", "دواء", "منتج", "أعراض",
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
                x in q for x in [
                    "enfant", "junior", "bébé", "pédiatrique",
                    "child", "children", "baby", "pediatric",  # EN
                    "niño", "niña", "bebé", "pediátrico",  # ES
                    "طفل", "أطفال", "رضيع",  # AR
                ]
            ) else "adulte"
        }
    
    @staticmethod
    def _extract_product_name_regex(text: str) -> Optional[str]:
        """Extract product name using various regex patterns (FR + EN + ES + AR)"""
        text = text.lower().strip()
        
        # List of patterns to try, from most specific to most general
        patterns = [
            # ── FRENCH ──
            # Pattern FR1: "présentation [preposition] X"
            r'(?:présentation|powerpoint|ppt|diaporama)\s+(?:pour|de|du|des|sur|d\')\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            # Pattern FR2: "présenter X"
            r'présenter\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            # Pattern FR3: "créer/générer [une] [présentation] [pour] X"
            r'(?:créer|générer|créez?|générez?)\s+(?:une\s+)?(?:présentation|powerpoint|ppt)?\s*(?:pour|de|du|des|sur|d\')?\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            # Pattern FR4: "je veux/voudrais une présentation [pour] X"
            r'(?:je\s+)?(?:veux|voudrais|aimerais|souhaite)\s+(?:une\s+)?(?:présentation|powerpoint|ppt)\s+(?:pour|de|du|des|sur|d\')\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',
            # Pattern FR5: Anything after "présentation" that's not a stop word
            r'présentation\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+et|\s+ou)',

            # ── ENGLISH ──
            # Pattern EN1: "presentation for/about/on X"
            r'(?:presentation|powerpoint|ppt|slideshow|slides)\s+(?:for|about|on|of)\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+and|\s+or)',
            # Pattern EN2: "present X"
            r'\bpresent\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+and|\s+or)',
            # Pattern EN3: "create/generate [a] [presentation] [for] X"
            r'(?:create|generate|make)\s+(?:a\s+)?(?:presentation|powerpoint|ppt)?\s*(?:for|about|on|of)?\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+and|\s+or)',
            # Pattern EN4: "I want/would like a presentation for X"
            r'(?:i\s+)?(?:want|need|would like)\s+(?:a\s+)?(?:presentation|powerpoint|ppt)\s+(?:for|about|on|of)\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+and|\s+or)',

            # ── SPANISH ──
            # Pattern ES1: "presentación para/de/sobre X"
            r'(?:presentación|powerpoint|ppt|diapositivas)\s+(?:para|de|del|sobre)\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+y|\s+o)',
            # Pattern ES2: "crear/generar presentación para X"
            r'(?:crear|generar)\s+(?:una\s+)?(?:presentación|powerpoint)?\s*(?:para|de|del|sobre)?\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+y|\s+o)',
            # Pattern ES3: "quiero una presentación para X"
            r'(?:quiero|necesito|deseo)\s+(?:una\s+)?(?:presentación|powerpoint)\s+(?:para|de|del|sobre)\s+([^?.!,;]+?)(?:\s*$|\s*[?.!,;]|\s+y|\s+o)',

            # ── ARABIC ── (simplified patterns)
            r'(?:\u062a\u0642\u062f\u064a\u0645|\u0639\u0631\u0636)\s+(?:\u0639\u0646|\u062d\u0648\u0644|\u0644)\s+([^?.!,;\u061f]+?)(?:\s*$|\s*[?.!,;\u061f])',
            r'(?:\u0625\u0646\u0634\u0627\u0621|\u062a\u0648\u0644\u064a\u062f)\s+(?:\u062a\u0642\u062f\u064a\u0645|\u0639\u0631\u0636)?\s*(?:\u0639\u0646|\u062d\u0648\u0644|\u0644)?\s+([^?.!,;\u061f]+?)(?:\s*$|\s*[?.!,;\u061f])',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_name = match.group(1).strip()
                
                # Clean up the extracted name
                # Remove leading articles and prepositions (FR + EN + ES)
                raw_name = re.sub(r'^(une|un|la|le|les|des|pour|de|du|sur|d\'|a|an|the|for|about|on|of|una|el|los|las|para|del|sobre)\s+', '', raw_name, flags=re.IGNORECASE)
                
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

# ══════════════════════════════════════════════════════════════════════
# FULLY MULTILINGUAL PROMPT TEMPLATES
# Each language gets its OWN complete template — this is critical for
# small LLMs like llama3.2 which follow the prompt language, not a
# single instruction line.
# ══════════════════════════════════════════════════════════════════════

_PROMPTS_COMMERCIAL = {
    "fr": """Tu es ALIA, assistant pharmaceutique d'un délégué médical.
Tu aides le délégué pendant sa visite chez un médecin ou un pharmacien.
Réponds UNIQUEMENT en FRANÇAIS.

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
6. Si la question est trop vague, demande des précisions.

Réponse:
""",

    "en": """You are ALIA, a pharmaceutical assistant for a medical sales representative.
You help the representative during their visit with a doctor or pharmacist.
You MUST respond ONLY in ENGLISH. Every word must be in English. Do NOT use French.

CONVERSATION HISTORY:
{history}

CURRENT QUESTION: "{query}"

AVAILABLE PRODUCTS (reference data in French — translate info to English):
{context}

RULES:
1. Check that the products match the question EXACTLY.
2. If no product matches, clearly say there is none in the current database.
3. If a product matches, recommend THE BEST one (only 1) with:
   - Exact product name
   - Sales argument in 1-2 sentences (patient benefit)
   - Administration method
   - Children's version if available
4. Consider the history: do not repeat what was already said.
5. Stay concise and professional.
6. If the question is too vague, ask for clarification.
7. CRITICAL: Respond ONLY in ENGLISH. Not French.

Response:
""",

    "es": """Eres ALIA, asistente farmacéutica de un delegado médico.
Ayudas al delegado durante su visita con un médico o farmacéutico.
DEBES responder SOLO en ESPAÑOL. Cada palabra debe ser en español. NO uses francés.

HISTORIAL DE LA CONVERSACIÓN:
{history}

PREGUNTA ACTUAL: "{query}"

PRODUCTOS DISPONIBLES (datos de referencia en francés — traduce al español):
{context}

REGLAS:
1. Verifica que los productos correspondan EXACTAMENTE a la pregunta.
2. Si ningún producto corresponde, dilo claramente.
3. Si un producto corresponde, recomienda EL MEJOR (solo 1) con:
   - Nombre exacto del producto
   - Argumento de venta en 1-2 frases (beneficio para el paciente)
   - Modo de administración
   - Versión infantil si existe
4. Considera el historial: no repitas lo que ya se dijo.
5. Sé concisa y profesional.
6. Si la pregunta es demasiado vaga, pide aclaraciones.
7. CRÍTICO: Responde SOLO en ESPAÑOL. No en francés.

Respuesta:
""",

    "ar": """أنتِ أليا، مساعدة صيدلانية لمندوب طبي.
تساعدين المندوب أثناء زيارته لطبيب أو صيدلي.
يجب أن تردي حصرياً بالعربية. لا تستخدمي الفرنسية أبداً.

سجل المحادثة:
{history}

السؤال الحالي: "{query}"

المنتجات المتاحة (بيانات مرجعية بالفرنسية — ترجمي المعلومات إلى العربية):
{context}

القواعد:
1. تحققي من أن المنتجات تتطابق تماماً مع السؤال.
2. إذا لم يتطابق أي منتج، قولي ذلك بوضوح.
3. إذا تطابق منتج، أوصي بالأفضل (واحد فقط) مع:
   - الاسم الدقيق للمنتج
   - حجة البيع في جملة أو جملتين (فائدة المريض)
   - طريقة الاستخدام
   - نسخة الأطفال إن وجدت
4. راعي السجل: لا تكرري ما قيل.
5. كوني موجزة ومهنية.
6. إذا كان السؤال غامضاً، اطلبي توضيحاً.
7. حاسم: ردي بالعربية فقط. ليس بالفرنسية.

الرد:
""",
}

_PROMPTS_TRAINING = {
    "fr": """Tu es ALIA, coach de formation pour délégués médicaux.
Tu joues le rôle d'un médecin ou pharmacien exigeant.
Tu évalues les réponses du délégué et tu l'aides à progresser.
Réponds UNIQUEMENT en FRANÇAIS.

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
""",

    "en": """You are ALIA, a training coach for medical sales representatives.
You play the role of a demanding doctor or pharmacist.
You evaluate the representative's answers and help them improve.
You MUST respond ONLY in ENGLISH. Every word must be in English. Do NOT use French.

CONVERSATION HISTORY:
{history}

WHAT THE REPRESENTATIVE SAYS: "{query}"

AVAILABLE PRODUCTS (reference data in French — use to verify, respond in English):
{context}

RULES:
1. If the representative gives correct product information → briefly congratulate and enrich.
2. If the representative makes a mistake → correct kindly and give the right answer.
3. If the representative hesitates → encourage and guide with a question or hint.
4. Ask a follow-up question to further test their knowledge.
5. Consider history to avoid repeating the same corrections.
6. CRITICAL: Respond ONLY in ENGLISH. Not French.

Response:
""",

    "es": """Eres ALIA, coach de formación para delegados médicos.
Juegas el rol de un médico o farmacéutico exigente.
Evalúas las respuestas del delegado y le ayudas a progresar.
DEBES responder SOLO en ESPAÑOL. Cada palabra debe ser en español. NO uses francés.

HISTORIAL DE LA CONVERSACIÓN:
{history}

LO QUE DICE EL DELEGADO: "{query}"

PRODUCTOS DISPONIBLES (datos de referencia en francés — usa para verificar, responde en español):
{context}

REGLAS:
1. Si el delegado da información correcta → felicítalo brevemente y enriquece.
2. Si el delegado comete un error → corrige con amabilidad y da la respuesta correcta.
3. Si el delegado duda → anímalo y guíalo con una pregunta o pista.
4. Haz una pregunta de seguimiento para probar más sus conocimientos.
5. Considera el historial para no repetir las mismas correcciones.
6. CRÍTICO: Responde SOLO en ESPAÑOL. No en francés.

Respuesta:
""",

    "ar": """أنتِ أليا، مدربة تكوين لمندوبين طبيين.
تلعبين دور طبيب أو صيدلي متطلب.
تقيّمين إجابات المندوب وتساعدينه على التقدم.
يجب أن تردي حصرياً بالعربية. لا تستخدمي الفرنسية أبداً.

سجل المحادثة:
{history}

ما يقوله المندوب: "{query}"

المنتجات المتاحة (بيانات مرجعية بالفرنسية — استخدميها للتحقق، ردي بالعربية):
{context}

القواعد:
1. إذا أعطى المندوب معلومة صحيحة → هنئيه باختصار وأثري.
2. إذا أخطأ المندوب → صححي بلطف وأعطي الإجابة الصحيحة.
3. إذا تردد المندوب → شجعيه ووجهيه بسؤال أو تلميح.
4. اطرحي سؤال متابعة لاختبار معارفه أكثر.
5. راعي السجل لعدم تكرار نفس التصحيحات.
6. حاسم: ردي بالعربية فقط. ليس بالفرنسية.

الرد:
""",
}


def _get_prompt_template(mode: str, lang: str) -> str:
    """Return the full prompt template for the given mode and language."""
    prompts = _PROMPTS_TRAINING if mode == "training" else _PROMPTS_COMMERCIAL
    return prompts.get(lang, prompts["fr"])


# ── Multilingual hardcoded responses ──────────────────────────────────
_LANG_RESPONSES = {
    "greeting": {
        "fr": "Bonjour, je suis ALIA. Comment puis-je vous aider aujourd'hui ?",
        "en": "Hello, I'm ALIA. How can I help you today?",
        "es": "Hola, soy ALIA. ¿Cómo puedo ayudarle hoy?",
        "ar": "مرحبًا، أنا أليا. كيف يمكنني مساعدتك اليوم؟",
    },
    "casual": {
        "fr": "Je vous en prie. Avez-vous besoin d'informations sur un produit ou d'une présentation ?",
        "en": "You're welcome. Do you need information about a product or a presentation?",
        "es": "De nada. ¿Necesita información sobre un producto o una presentación?",
        "ar": "على الرحب والسعة. هل تحتاج معلومات عن منتج أو عرض تقديمي؟",
    },
    "clarify_product": {
        "fr": "Pour quel produit souhaitez-vous générer une présentation PowerPoint ?",
        "en": "For which product would you like to generate a PowerPoint presentation?",
        "es": "¿Para qué producto desea generar una presentación PowerPoint?",
        "ar": "لأي منتج ترغب في إنشاء عرض تقديمي؟",
    },
    "vague": {
        "fr": "Pouvez-vous préciser votre besoin ? Par exemple : avez-vous un problème particulier comme la fatigue, le sommeil, la peau, ou une pathologie spécifique ?",
        "en": "Could you specify your need? For example: do you have a particular issue like fatigue, sleep, skin, or a specific condition?",
        "es": "¿Puede precisar su necesidad? Por ejemplo: ¿tiene un problema particular como cansancio, sueño, piel, o una patología específica?",
        "ar": "هل يمكنك تحديد حاجتك؟ مثلاً: هل لديك مشكلة معينة كالتعب أو النوم أو البشرة أو حالة مرضية محددة؟",
    },
}


def _lr(key: str, lang: str = "fr") -> str:
    """Get a language-specific hardcoded response."""
    responses = _LANG_RESPONSES.get(key, {})
    return responses.get(lang, responses.get("fr", ""))


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
        # Chain is now built dynamically per-language in generate()
        # Default chain for compatibility (French)
        template = _get_prompt_template(self.mode, "fr")
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _build_chain_for_lang(self, lang: str):
        """Build a LangChain chain for a specific language."""
        template = _get_prompt_template(self.mode, lang)
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()
 
    def set_mode(self, mode: str):
        """Change le mode à chaud et réinitialise l'historique."""
        if mode not in ("training", "commercial"):
            raise ValueError("mode doit être 'training' ou 'commercial'")
        self.mode = mode
        self.history = []
        self._build_chain()
        print(f"[ALIA] Mode changé : {mode}")
 
    def _format_history(self, lang: str = "fr") -> str:
        """Formate les N derniers tours pour le prompt."""
        _EMPTY_HISTORY = {
            "fr": "Aucun historique (début de conversation).",
            "en": "No history (start of conversation).",
            "es": "Sin historial (inicio de conversación).",
            "ar": "لا يوجد سجل (بداية المحادثة).",
        }
        _USER_LABEL = {
            "fr": "Délégué", "en": "Representative",
            "es": "Delegado", "ar": "المندوب",
        }
        turns = self.history[-(self.HISTORY_MAX_TURNS * 2):]
        if not turns:
            return _EMPTY_HISTORY.get(lang, _EMPTY_HISTORY["fr"])
        lines = []
        user_label = _USER_LABEL.get(lang, "Délégué")
        for msg in turns:
            prefix = user_label if msg["role"] == "user" else "ALIA"
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

    def _humanize(self, text: str, lang: str = "fr") -> str:
        """Apply language-specific text humanization."""
        if lang == "fr":
            text = text.replace("Il est recommandé", "Je vous conseille")
            text = text.replace("Vous pouvez utiliser", "Vous pouvez prendre")
            text = text.replace("Je vous recommande", "Je vous conseille")
        elif lang == "en":
            text = text.replace("It is recommended", "I suggest")
            text = text.replace("You can use", "You can take")
        elif lang == "es":
            text = text.replace("Se recomienda", "Le sugiero")
            text = text.replace("Puede utilizar", "Puede tomar")
        return text

    def _validate_response_language(self, text: str, expected_lang: str) -> str:
        """Post-generation safety net: detect if response is in wrong language.
        If the user asked in English but the model responded in French,
        return a language-appropriate fallback instead of wrong-language gibberish.
        """
        if expected_lang == "fr":
            return text  # Default is French, no check needed

        # Common French function words (high frequency, language-specific)
        french_markers = {
            "je", "vous", "pour", "est", "les", "des", "une", "dans",
            "que", "qui", "avec", "sur", "par", "plus", "mais", "peut",
            "sont", "cette", "être", "avoir", "nous", "leur", "ces",
        }

        words = text.lower().split()
        if len(words) < 5:
            return text  # Too short to judge

        french_count = sum(1 for w in words if w in french_markers)
        french_ratio = french_count / len(words)

        if french_ratio > 0.15:
            # Response is likely still French — return a polite note
            print(f"[ALIA] WARNING: Response appears French (ratio={french_ratio:.2f}) "
                  f"but expected '{expected_lang}'. Using fallback.")
            _WRONG_LANG_FALLBACK = {
                "en": "I found relevant products in our database, but I had trouble generating a response in English. "
                      "Could you rephrase your question? I'll do my best to respond in English.",
                "es": "Encontré productos relevantes en nuestra base de datos, pero tuve dificultad para generar "
                      "una respuesta en español. ¿Puede reformular su pregunta?",
                "ar": "وجدت منتجات ذات صلة في قاعدة بياناتنا، لكن واجهت صعوبة في إنشاء رد بالعربية. "
                      "هل يمكنك إعادة صياغة سؤالك؟",
            }
            return _WRONG_LANG_FALLBACK.get(expected_lang, text)

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

    def _smart_fallback(self, query: str, docs: List[Document], analysis: Dict, lang: str = "fr") -> str:
        """Provide intelligent fallback when no relevant products found"""
        _FALLBACK_CONDITIONS = {
            "fr": "Je n'ai pas de produit spécifique pour {cond} dans ma base de données actuelle. Pourriez-vous me donner plus de détails sur vos besoins, ou puis-je vous aider avec un autre type de produit ?",
            "en": "I don't have a specific product for {cond} in my current database. Could you give me more details about your needs, or can I help you with another type of product?",
            "es": "No tengo un producto específico para {cond} en mi base de datos actual. ¿Puede darme más detalles sobre sus necesidades, o puedo ayudarle con otro tipo de producto?",
            "ar": "ليس لدي منتج محدد لـ {cond} في قاعدة بياناتي الحالية. هل يمكنك إعطائي مزيداً من التفاصيل حول احتياجاتك؟",
        }
        _FALLBACK_GENERIC = {
            "fr": "Je n'ai pas trouvé de produit correspondant exactement à votre demande. Pourriez-vous reformuler ou préciser votre besoin ?",
            "en": "I didn't find a product matching your request exactly. Could you rephrase or specify your need?",
            "es": "No he encontrado un producto que corresponda exactamente a su solicitud. ¿Podría reformular o precisar su necesidad?",
            "ar": "لم أجد منتجاً يتطابق تماماً مع طلبك. هل يمكنك إعادة الصياغة أو تحديد حاجتك؟",
        }
        conditions = analysis.get("conditions", [])
        
        if conditions:
            conditions_str = ", ".join(conditions)
            template = _FALLBACK_CONDITIONS.get(lang, _FALLBACK_CONDITIONS["fr"])
            return template.format(cond=conditions_str)
        
        return _FALLBACK_GENERIC.get(lang, _FALLBACK_GENERIC["fr"])

    async def generate_presentation(self, product_name: str, lang: str = "fr") -> Dict:
        """Generate a PowerPoint presentation for a specific product"""
        _PPT_UNAVAIL = {
            "fr": "Désolée, la génération de présentations PowerPoint n'est pas disponible actuellement.",
            "en": "Sorry, PowerPoint presentation generation is not currently available.",
            "es": "Lo siento, la generación de presentaciones PowerPoint no está disponible actualmente.",
            "ar": "عذراً، إنشاء عروض PowerPoint غير متاح حالياً.",
        }
        _PPT_SUCCESS = {
            "fr": (
                "J'ai généré la présentation PowerPoint pour {name} !\n\n"
                "La présentation comprend :\n"
                "• Une slide d'accroche avec le positionnement produit\n"
                "• L'analyse de la problématique patient\n"
                "• La solution experte Vital Labs\n"
                "• Les 4 piliers stratégiques\n"
                "• Les spécifications techniques\n"
                "• Une conclusion avec appel à l'action\n\n"
                "Voulez-vous que je vous présente le contenu ou que je génère une présentation pour un autre produit ?"
            ),
            "en": (
                "I've generated the PowerPoint presentation for {name}!\n\n"
                "The presentation includes:\n"
                "• An opening slide with product positioning\n"
                "• Patient problem analysis\n"
                "• The Vital Labs expert solution\n"
                "• The 4 strategic pillars\n"
                "• Technical specifications\n"
                "• A conclusion with call to action\n\n"
                "Would you like me to present the content or generate a presentation for another product?"
            ),
            "es": (
                "¡He generado la presentación PowerPoint para {name}!\n\n"
                "La presentación incluye:\n"
                "• Una diapositiva de apertura con posicionamiento del producto\n"
                "• Análisis de la problemática del paciente\n"
                "• La solución experta Vital Labs\n"
                "• Los 4 pilares estratégicos\n"
                "• Especificaciones técnicas\n"
                "• Una conclusión con llamada a la acción\n\n"
                "¿Desea que le presente el contenido o que genere una presentación para otro producto?"
            ),
            "ar": (
                "لقد أنشأت عرض PowerPoint لـ {name}!\n\n"
                "يتضمن العرض:\n"
                "• شريحة افتتاحية مع تموضع المنتج\n"
                "• تحليل مشكلة المريض\n"
                "• حل Vital Labs المتخصص\n"
                "• الركائز الاستراتيجية الأربع\n"
                "• المواصفات التقنية\n"
                "• خاتمة مع دعوة للعمل\n\n"
                "هل تريد أن أقدم لك المحتوى أو أنشئ عرضاً لمنتج آخر؟"
            ),
        }
        _PPT_NOT_FOUND = {
            "fr": "Je n'ai pas trouvé le produit '{name}' dans notre catalogue. Pouvez-vous vérifier le nom exact du produit ?",
            "en": "I didn't find the product '{name}' in our catalog. Can you check the exact product name?",
            "es": "No he encontrado el producto '{name}' en nuestro catálogo. ¿Puede verificar el nombre exacto del producto?",
            "ar": "لم أجد المنتج '{name}' في كتالوجنا. هل يمكنك التحقق من الاسم الدقيق للمنتج؟",
        }
        _PPT_ERROR = {
            "fr": "Désolée, une erreur s'est produite lors de la génération de la présentation. Veuillez réessayer avec un autre produit.",
            "en": "Sorry, an error occurred during presentation generation. Please try again with another product.",
            "es": "Lo siento, ocurrió un error durante la generación de la presentación. Por favor, intente con otro producto.",
            "ar": "عذراً، حدث خطأ أثناء إنشاء العرض. يرجى المحاولة مع منتج آخر.",
        }

        if not PPT_AVAILABLE:
            return {
                "text": _PPT_UNAVAIL.get(lang, _PPT_UNAVAIL["fr"]),
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
            
            response = _PPT_SUCCESS.get(lang, _PPT_SUCCESS["fr"]).format(name=product_name)
            
            return {
                "text": response,
                "intent": "presentation_generated",
                "presentation_path": str(output_path)
            }
            
        except ValueError as e:
            return {
                "text": _PPT_NOT_FOUND.get(lang, _PPT_NOT_FOUND["fr"]).format(name=product_name),
                "intent": "presentation_error",
                "presentation_path": None
            }
        except Exception as e:
            print(f"[ALIA] Presentation generation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": _PPT_ERROR.get(lang, _PPT_ERROR["fr"]),
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



    async def generate(self, query: str, lang: str = "fr"):
        try:
            analysis = QueryAnalyzer.analyze(query)
            print(f"[ALIA] Query analysis: {analysis}")
            print(f"[ALIA] Language received: {lang}")

            # =========================
            # GREETING
            # =========================
            if analysis.get("is_greeting"):
                response = _lr("greeting", lang)
                self._add_to_history("user", query)
                self._add_to_history("alia", response)
                return {"text": response, "intent": "greeting"}

            # =========================
            # CASUAL
            # =========================
            if analysis.get("is_casual"):
                response = _lr("casual", lang)
                self._add_to_history("user", query)
                self._add_to_history("alia", response)
                return {"text": response, "intent": "casual"}

            # =========================
            # PRESENTATION REQUEST
            # =========================
            if analysis.get("is_presentation_request"):
                product_name = analysis.get("product_name")

                if not product_name:
                    response = _lr("clarify_product", lang)
                    self._add_to_history("user", query)
                    self._add_to_history("alia", response)
                    return {"text": response, "intent": "clarification"}

                product_doc = self.manager.find_product_by_name(product_name)

                if not product_doc:
                    similar_products = self._find_similar_products(product_name)

                    _NOT_FOUND_HEADER = {
                        "fr": "Je n'ai pas trouvé exactement '{name}' dans notre base.\n\n",
                        "en": "I didn't find exactly '{name}' in our database.\n\n",
                        "es": "No he encontrado exactamente '{name}' en nuestra base.\n\n",
                        "ar": "لم أجد بالضبط '{name}' في قاعدة بياناتنا.\n\n",
                    }
                    _SIMILAR_HEADER = {
                        "fr": "Produits similaires disponibles :\n",
                        "en": "Similar products available:\n",
                        "es": "Productos similares disponibles:\n",
                        "ar": "منتجات مشابهة متاحة:\n",
                    }
                    _SIMILAR_FOOTER = {
                        "fr": "\nSouhaitez-vous une présentation pour l'un de ces produits ?",
                        "en": "\nWould you like a presentation for one of these products?",
                        "es": "\n¿Desea una presentación para uno de estos productos?",
                        "ar": "\nهل تريد عرضاً تقديمياً لأحد هذه المنتجات؟",
                    }
                    _CHECK_NAME = {
                        "fr": "Pouvez-vous vérifier le nom du produit ?",
                        "en": "Can you check the product name?",
                        "es": "¿Puede verificar el nombre del producto?",
                        "ar": "هل يمكنك التحقق من اسم المنتج؟",
                    }

                    response = _NOT_FOUND_HEADER.get(lang, _NOT_FOUND_HEADER["fr"]).format(name=product_name)

                    if similar_products:
                        response += _SIMILAR_HEADER.get(lang, _SIMILAR_HEADER["fr"])
                        for prod in similar_products[:5]:
                            response += f"• {prod}\n"
                        response += _SIMILAR_FOOTER.get(lang, _SIMILAR_FOOTER["fr"])
                    else:
                        response += _CHECK_NAME.get(lang, _CHECK_NAME["fr"])

                    self._add_to_history("user", query)
                    self._add_to_history("alia", response)
                    return {"text": response, "intent": "clarification"}

                actual_product_name = product_doc.metadata.get("name")
                self._add_to_history("user", query)

                result = await self.generate_presentation(actual_product_name, lang=lang)

                self._add_to_history("alia", result["text"])
                return result

            # =========================
            # VAGUE QUERY DETECTION
            # =========================
            mots_vagues = [
                # FR
                "nouveau", "nouveauté", "quoi de neuf",
                "qu'est-ce que vous avez", "vous avez quoi",
                "un produit", "quelque chose",
                # EN
                "what do you have", "something new", "new product",
                "what's new", "anything", "a product",
                # ES
                "qué tienen", "algo nuevo", "nuevo producto",
                "qué hay de nuevo", "un producto",
                # AR
                "ماذا لديكم", "شيء جديد", "منتج جديد",
            ]

            query_lower = query.lower()

            specific_terms = [
                # FR
                "cheveux", "fatigue", "peau", "sommeil", "anémie",
                "digestion", "stress", "os", "gorge", "toux",
                "rhume", "grippe", "articulation", "vision", "mémoire",
                # EN
                "hair", "skin", "sleep", "anemia", "iron",
                "cold", "flu", "joint", "bone", "energy",
                # ES
                "cabello", "piel", "sueño", "hierro",
                "gripe", "articulación",
                # AR
                "شعر", "بشرة", "نوم", "حديد", "تعب",
            ]

            est_vague = (
                len(query.split()) < 12
                and any(m in query_lower for m in mots_vagues)
                and not any(m in query_lower for m in specific_terms)
            )

            if est_vague:
                response = _lr("vague", lang)

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
                response = self._smart_fallback(query, [], analysis, lang=lang)

                self._add_to_history("user", query)
                self._add_to_history("alia", response)

                return {"text": response, "intent": "no_results"}

            is_relevant = self._check_relevance(query, docs)

            print(f"[ALIA] Documents relevant: {is_relevant}")

            print("[ALIA] Top 3 retrieved products:")
            for i, doc in enumerate(docs[:3]):
                print(f"  {i+1}. {doc.metadata.get('name', 'Unknown')}")

            context = "\n\n---\n\n".join(d.page_content for d in docs[:5])

            history_str = self._format_history(lang=lang)

            print(f"[ALIA] Context length: {len(context)} chars")
            print(f"[ALIA] Generating response in '{lang}'...")

            # Build a chain for the specific language
            lang_chain = self._build_chain_for_lang(lang)

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lang_chain.invoke,
                    {
                        "context": context,
                        "query": query,
                        "history": history_str,
                    }
                ),
                timeout=120
            )

            print(f"[ALIA] Raw response: {response[:200]}...")

            cleaned = self._clean(response)

            if not cleaned or len(cleaned.split()) < 10:
                print("[ALIA] Response too short, using fallback")
                cleaned = self._smart_fallback(query, docs, analysis, lang=lang)

            elif not is_relevant:
                print("[ALIA] Documents not relevant enough, using fallback")
                cleaned = self._smart_fallback(query, docs, analysis, lang=lang)

            cleaned = self._humanize(cleaned, lang=lang)

            # Post-generation language validation
            cleaned = self._validate_response_language(cleaned, lang)

            print(f"[ALIA] Final response ({lang}): {cleaned}")

            self._add_to_history("user", query)
            self._add_to_history("alia", cleaned)

            return {"text": cleaned, "intent": "recommendation"}

        except asyncio.TimeoutError:
            print("[ALIA] Timeout error!")
            _TIMEOUT = {
                "fr": "Désolée, le traitement de votre demande a pris trop de temps. Pouvez-vous reformuler ?",
                "en": "Sorry, your request took too long to process. Could you rephrase?",
                "es": "Lo siento, su solicitud tardó demasiado en procesarse. ¿Puede reformular?",
                "ar": "عذراً، استغرق طلبك وقتاً طويلاً. هل يمكنك إعادة الصياغة؟",
            }
            return {
                "text": _TIMEOUT.get(lang, _TIMEOUT["fr"]),
                "intent": "timeout"
            }

        except Exception as e:
            print(f"[ALIA] Error during generation: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            _ERROR = {
                "fr": "Désolée, j'ai rencontré une erreur technique. Veuillez réessayer.",
                "en": "Sorry, I encountered a technical error. Please try again.",
                "es": "Lo siento, encontré un error técnico. Por favor, inténtelo de nuevo.",
                "ar": "عذراً، واجهت خطأ تقنياً. يرجى المحاولة مرة أخرى.",
            }
            return {
                "text": _ERROR.get(lang, _ERROR["fr"]),
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