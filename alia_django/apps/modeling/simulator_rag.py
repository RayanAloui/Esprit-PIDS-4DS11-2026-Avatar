"""
simulator_rag.py
================
RAG dédié au mode Simulator — distinct du RAG commercial (modeling/rag_gemini.py).

Rôle : enrichir chaque tour de conversation avec des données produit précises
extraites de la base VITAL SA, ET générer la décision finale du médecin
(commande / refus / conditions) en intégrant le score global NLP+LSTM+VM.

Architecture :
  - SimulatorRAG.get_product_context()  → contexte RAG pour chaque réponse médecin
  - SimulatorRAG.generate_final_decision() → décision finale pondérée par score global
  - SimulatorRAG.build_doctor_objection_prompt() → prompt système enrichi par les
    données réelles du produit depuis la KB

Le RAG simulator utilise le même KnowledgeManager que modeling (singleton via
get_runtime()), évitant un double chargement de Chroma + embeddings.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# ACCÈS À LA BASE DE CONNAISSANCES (partagée avec modeling)
# ══════════════════════════════════════════════════════════════════════

def _get_kb():
    """Retourne le KnowledgeManager partagé (singleton modeling runtime)."""
    try:
        from apps.modeling.runtime import get_runtime
        return get_runtime().kb
    except Exception as e:
        log.warning(f"[SimRAG] KB indisponible : {e}")
        return None


def get_product_context(product_name: str, query: str, n_docs: int = 3) -> str:
    """
    Récupère le contexte produit depuis la base VITAL SA.
    Retourne une chaîne formatée prête à injecter dans le prompt médecin.
    """
    kb = _get_kb()
    if not kb:
        return ""
    try:
        search_query = f"{product_name} {query}"
        docs = kb.retriever.invoke(search_query)
        if not docs:
            return ""
        parts = []
        for doc in docs[:n_docs]:
            name    = doc.metadata.get("name", "")
            content = doc.page_content[:400].replace("\n", " ").strip()
            parts.append(f"• {name} : {content}")
        return "\n".join(parts)
    except Exception as e:
        log.warning(f"[SimRAG] Erreur retrieval : {e}")
        return ""


def get_full_product_data(product_name: str) -> Optional[Dict]:
    """
    Récupère les données complètes d'un produit depuis la KB.
    Utilisé pour enrichir le prompt système initial du médecin.
    """
    kb = _get_kb()
    if not kb:
        return None
    try:
        doc = kb.find_product_by_name(product_name)
        if not doc:
            return None
        return {
            "name"    : doc.metadata.get("name", product_name),
            "content" : doc.page_content[:600],
            "forme"   : doc.metadata.get("forme", ""),
            "target"  : doc.metadata.get("target", "adulte"),
            "usage"   : doc.metadata.get("usage", "oral"),
        }
    except Exception as e:
        log.warning(f"[SimRAG] Erreur product lookup : {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# MULTILINGUAL PROMPT STRUCTURE
# ══════════════════════════════════════════════════════════════════════

_SYSTEM_LABELS = {
    "fr": {
        "personality": "PERSONNALITÉ",
        "visit_context": "CONTEXTE DE LA VISITE",
        "product_presented": "Produit présenté",
        "current_turn": "Tour actuel",
        "openness_level": "Ton niveau d'ouverture actuel",
        "real_data": "DONNÉES RÉELLES DU PRODUIT — Base VITAL SA",
        "product_section": "PRODUIT PRÉSENTÉ",
        "indication": "Indication",
        "key_argument": "Argument clé",
        "data_note": "Tu connais ces données. Tes objections et questions doivent être cohérentes avec elles.",
        "absolute_rule": "RÈGLE ABSOLUE",
        "react_directly": "Réagis DIRECTEMENT au dernier message du délégué.",
        "no_repeat": "Ne répète jamais une formulation déjà utilisée.",
        "one_objection": "Pose une seule objection ou question à la fois.",
        "progression": "PROGRESSION",
        "turn_1_2": "Tour 1-2 : accueil réservé + première objection selon ta personnalité",
        "turn_3_4": "Tour 3-4 : objection précise sur le produit (données, effets secondaires, prix, concurrence)",
        "turn_5_6": "Tour 5-6 : position claire selon ton ouverture",
        "turn_7_8": "Tour 7-8 : conclusion (signal d'achat ou refus)",
        "style": "STYLE : 2-3 phrases max. Ton",
        "respond_only": "Réponds UNIQUEMENT avec tes paroles. Pas de guillemets, pas de préfixe.",
        "typical_objections": "OBJECTIONS TYPIQUES QUE TU UTILISES",
        "lang_rule": "",
    },
    "en": {
        "personality": "PERSONALITY",
        "visit_context": "VISIT CONTEXT",
        "product_presented": "Product presented",
        "current_turn": "Current turn",
        "openness_level": "Your current openness level",
        "real_data": "REAL PRODUCT DATA — VITAL SA Database",
        "product_section": "PRODUCT PRESENTED",
        "indication": "Indication",
        "key_argument": "Key argument",
        "data_note": "You know this data. Your objections and questions must be consistent with it.",
        "absolute_rule": "ABSOLUTE RULE",
        "react_directly": "React DIRECTLY to the delegate's last message.",
        "no_repeat": "Never repeat a formulation already used.",
        "one_objection": "Ask only one objection or question at a time.",
        "progression": "PROGRESSION",
        "turn_1_2": "Turn 1-2: reserved welcome + first objection based on your personality",
        "turn_3_4": "Turn 3-4: precise objection about the product (data, side effects, price, competition)",
        "turn_5_6": "Turn 5-6: clear position based on your openness",
        "turn_7_8": "Turn 7-8: conclusion (buying signal or refusal)",
        "style": "STYLE: 2-3 sentences max. Tone",
        "respond_only": "Respond ONLY with your words. No quotes, no prefix.",
        "typical_objections": "TYPICAL OBJECTIONS YOU USE",
        "lang_rule": "CRITICAL: You MUST respond ONLY in ENGLISH. Every word must be in English. Do NOT use French.",
    },
    "es": {
        "personality": "PERSONALIDAD",
        "visit_context": "CONTEXTO DE LA VISITA",
        "product_presented": "Producto presentado",
        "current_turn": "Turno actual",
        "openness_level": "Tu nivel de apertura actual",
        "real_data": "DATOS REALES DEL PRODUCTO — Base VITAL SA",
        "product_section": "PRODUCTO PRESENTADO",
        "indication": "Indicación",
        "key_argument": "Argumento clave",
        "data_note": "Conoces estos datos. Tus objeciones y preguntas deben ser coherentes con ellos.",
        "absolute_rule": "REGLA ABSOLUTA",
        "react_directly": "Reacciona DIRECTAMENTE al último mensaje del delegado.",
        "no_repeat": "Nunca repitas una formulación ya utilizada.",
        "one_objection": "Haz solo una objeción o pregunta a la vez.",
        "progression": "PROGRESIÓN",
        "turn_1_2": "Turno 1-2: acogida reservada + primera objeción según tu personalidad",
        "turn_3_4": "Turno 3-4: objeción precisa sobre el producto (datos, efectos secundarios, precio, competencia)",
        "turn_5_6": "Turno 5-6: posición clara según tu apertura",
        "turn_7_8": "Turno 7-8: conclusión (señal de compra o rechazo)",
        "style": "ESTILO: 2-3 frases máx. Tono",
        "respond_only": "Responde ÚNICAMENTE con tus palabras. Sin comillas, sin prefijo.",
        "typical_objections": "OBJECIONES TÍPICAS QUE USAS",
        "lang_rule": "CRÍTICO: DEBES responder SOLO en ESPAÑOL. Cada palabra debe ser en español. NO uses francés.",
    },
    "tn": {
        "personality": "الشخصية",
        "visit_context": "سياق الزيارة",
        "product_presented": "المنتج المقدم",
        "current_turn": "الدور الحالي",
        "openness_level": "مستوى انفتاحك الحالي",
        "real_data": "بيانات المنتج الحقيقية — قاعدة VITAL SA",
        "product_section": "المنتج المقدم",
        "indication": "دواعي الاستعمال",
        "key_argument": "الحجة الرئيسية",
        "data_note": "أنت تعرف هذه البيانات. اعتراضاتك وأسئلتك يجب أن تكون متسقة معها.",
        "absolute_rule": "قاعدة مطلقة",
        "react_directly": "تفاعل مباشرة مع آخر رسالة للمندوب.",
        "no_repeat": "لا تكرر صياغة سبق استخدامها.",
        "one_objection": "اطرح اعتراضاً أو سؤالاً واحداً فقط في كل مرة.",
        "progression": "التقدم",
        "turn_1_2": "الدور 1-2: استقبال متحفظ + أول اعتراض حسب شخصيتك",
        "turn_3_4": "الدور 3-4: اعتراض دقيق حول المنتج",
        "turn_5_6": "الدور 5-6: موقف واضح حسب انفتاحك",
        "turn_7_8": "الدور 7-8: خاتمة (إشارة شراء أو رفض)",
        "style": "الأسلوب: 2-3 جمل كحد أقصى. النبرة",
        "respond_only": "رد فقط بكلامك. بدون علامات اقتباس أو بادئة.",
        "typical_objections": "الاعتراضات النموذجية التي تستخدمها",
        "lang_rule": "حاسم: يجب أن ترد حصرياً باللغة العربية. لا تستخدم الفرنسية.",
    },
}


def _get_sys_labels(lang: str) -> dict:
    return _SYSTEM_LABELS.get(lang, _SYSTEM_LABELS["fr"])


def build_enriched_doctor_system(
    doctor: Dict,
    product: Dict,
    turn: int,
    openness: float,
    max_turns: int = 8,
    lang: str = "fr",
) -> str:
    """
    Construit le prompt système du médecin enrichi par les données réelles
    de la base VITAL SA. Remplace _build_doctor_system() dans engine.py.
    Multilingual: structural parts use the target language.
    """
    ll = _get_sys_labels(lang)
    # Récupérer les données produit réelles depuis la KB
    kb_data = get_full_product_data(product["nom"])

    # Section données produit enrichies
    if kb_data:
        product_section = f"""
[{ll['real_data']}]
{kb_data['name']}
{kb_data['content']}
Forme : {kb_data['forme']} | Usage : {kb_data['usage']} | Cible : {kb_data['target']}

{ll['data_note']}
"""
    else:
        product_section = f"""
[{ll['product_section']}]
{product['nom']} ({product['categorie']})
{ll['indication']} : {product['indication']}
{ll['key_argument']} : {product['argument_cle']}
"""

    lang_rule = f"\n{ll['lang_rule']}\n" if ll['lang_rule'] else ""

    return f"""{lang_rule}Tu es {doctor['nom']}, {doctor['specialite']} à {doctor['ville']}.

{ll['personality']} :
{doctor['personnalite']}

{ll['visit_context']} :
- {ll['product_presented']} : {product['nom']} ({product['categorie']})
- {ll['current_turn']} : {turn}/{max_turns}
- {ll['openness_level']} : {openness:.1f}/5
{product_section}
{ll['absolute_rule']} :
{ll['react_directly']}
{ll['no_repeat']}
{ll['one_objection']}

{ll['progression']} ({_openness_label(openness, lang)}) :
- {ll['turn_1_2']}
- {ll['turn_3_4']}
- {ll['turn_5_6']}
- {ll['turn_7_8']}

{ll['style']} : {doctor['description_ui']}.
{ll['respond_only']}

{ll['typical_objections']} :
{chr(10).join(f'- {o}' for o in doctor['objections_favorites'])}
"""


def _openness_label(openness: float, lang: str = "fr") -> str:
    _LABELS = {
        "fr": [
            (4.2, "Très ouvert — questions positives sur la mise en pratique"),
            (3.5, "Ouvert — demande des précisions sans s'engager"),
            (2.5, "Neutre — objection modérée, en attente d'arguments"),
            (1.5, "Sceptique — doute direct sur le dernier argument"),
            (0.0, "Fermé — impatience, fin de visite imminente"),
        ],
        "en": [
            (4.2, "Very open — positive questions about implementation"),
            (3.5, "Open — asking for details without committing"),
            (2.5, "Neutral — moderate objection, waiting for arguments"),
            (1.5, "Skeptical — direct doubt about the last argument"),
            (0.0, "Closed — impatient, visit ending soon"),
        ],
        "es": [
            (4.2, "Muy abierto — preguntas positivas sobre la implementación"),
            (3.5, "Abierto — pide detalles sin comprometerse"),
            (2.5, "Neutral — objeción moderada, esperando argumentos"),
            (1.5, "Escéptico — duda directa sobre el último argumento"),
            (0.0, "Cerrado — impaciente, visita terminándose"),
        ],
        "tn": [
            (4.2, "منفتح جداً — أسئلة إيجابية"),
            (3.5, "منفتح — يطلب تفاصيل"),
            (2.5, "محايد — اعتراض معتدل"),
            (1.5, "متشكك — شك مباشر"),
            (0.0, "مغلق — نفاد الصبر"),
        ],
    }
    labels = _LABELS.get(lang, _LABELS["fr"])
    for threshold, label in labels:
        if openness >= threshold:
            return label
    return labels[-1][1]


# ══════════════════════════════════════════════════════════════════════
# PROMPT PHARMACIEN ENRICHI PAR RAG
# ══════════════════════════════════════════════════════════════════════

def build_enriched_pharmacist_system(
    pharmacist: Dict,
    product: Dict,
    turn: int,
    openness: float,
    max_turns: int = 8,
    lang: str = "fr",
) -> str:
    """
    Construit le prompt système du pharmacien enrichi par les données réelles
    de la base VITAL SA. Les données sont orientées officine :
    marge, rotation, conditionnement, PLV, DLC.
    """
    ll = _get_sys_labels(lang)
    # Récupérer les données produit réelles depuis la KB
    kb_data = get_full_product_data(product["nom"])

    if kb_data:
        product_section = f"""
[{ll['real_data']} (vue officine)]
{kb_data['name']}
{kb_data['content']}
Forme : {kb_data['forme']} | Usage : {kb_data['usage']} | Cible : {kb_data['target']}

En tant que pharmacien, tu évalues ce produit selon :
- La marge brute et le prix d'achat pharmacien (PPA)
- La rotation en rayon et la demande spontanée
- Le conditionnement et la DLC
- Le support terrain (PLV, animations, fiches conseil)
"""
    else:
        product_section = f"""
[{ll['product_section']}]
{product['nom']} ({product['categorie']})
{ll['indication']} : {product['indication']}
{ll['key_argument']} : {product['argument_cle']}
"""

    lang_rule = f"\n{ll['lang_rule']}\n" if ll['lang_rule'] else ""

    return f"""{lang_rule}Tu es {pharmacist['nom']}, {pharmacist['specialite']} à {pharmacist['ville']}.

{ll['personality']} :
{pharmacist['personnalite']}

{ll['visit_context']} :
- {ll['product_presented']} : {product['nom']} ({product['categorie']})
- {ll['current_turn']} : {turn}/{max_turns}
- {ll['openness_level']} : {openness:.1f}/5
{product_section}
CADRE DE LA VISITE OFFICINE (DIFFÉRENT D'UN MÉDECIN) :
- Tu penses en PHARMACIEN : marge brute, rotation de stock, DLC, conditionnement, PLV.
- Tu ne prescris pas — tu recommandes et tu vends au comptoir.
- Tes objections portent sur : le prix d'achat, la concurrence en rayon, le conditionnement,
  les conditions de retour, les remises de volume, la demande spontanée de tes clients.
- Tu n'es PAS intéressé par les mécanismes d'action cliniques (laisse ça au médecin).
- Tu veux savoir : est-ce que ça se vend bien ? Quelle marge ? Quel support terrain ?

{ll['absolute_rule']} :
{ll['react_directly']}
{ll['no_repeat']}
{ll['one_objection']}

{ll['progression']} ({_openness_label(openness, lang)}) :
- {ll['turn_1_2']}
- {ll['turn_3_4']}
- {ll['turn_5_6']}
- {ll['turn_7_8']}

{ll['style']} : {pharmacist['description_ui']}.
{ll['respond_only']}

{ll['typical_objections']} :
{chr(10).join(f'- {o}' for o in pharmacist['objections_favorites'])}
"""


# ══════════════════════════════════════════════════════════════════════
# PROMPT GÉNÉRALISTE ENRICHI PAR RAG
# ══════════════════════════════════════════════════════════════════════

def build_enriched_generalist_system(
    interlocutor: Dict,
    product: Dict,
    turn: int,
    openness: float,
    max_turns: int = 8,
    lang: str = "fr",
) -> str:
    """
    Construit le prompt système pour le mode généraliste.
    Contrairement au mode classique (un produit précis), le mode généraliste
    couvre toute la gamme VITAL SA, avec des questions plus ouvertes
    et un focus sur la capacité d'adaptation du délégué.
    """
    ll = _get_sys_labels(lang)
    kb = _get_kb()
    gamme_section = ""
    if kb:
        try:
            docs = kb.retriever.invoke("gamme complète VITAL SA produits")
            if docs:
                parts = []
                for doc in docs[:5]:
                    name    = doc.metadata.get("name", "")
                    content = doc.page_content[:250].replace("\n", " ").strip()
                    parts.append(f"• {name} : {content}")
                gamme_section = (
                    "\n[GAMME VITAL SA — KB Data]\n"
                    + "\n".join(parts) + "\n"
                )
        except Exception as e:
            log.warning(f"[SimRAG] Gamme retrieval error : {e}")

    lang_rule = f"\n{ll['lang_rule']}\n" if ll['lang_rule'] else ""

    return f"""{lang_rule}Tu es {interlocutor['nom']}, {interlocutor['specialite']} à {interlocutor['ville']}.

{ll['personality']} :
{interlocutor['personnalite']}

{ll['visit_context']} :
- {ll['current_turn']} : {turn}/{max_turns}
- {ll['openness_level']} : {openness:.1f}/5
{gamme_section}
COMPORTEMENT EN MODE GÉNÉRALISTE :
- Tu peux aborder PLUSIEURS produits de la gamme VITAL SA au fil de la conversation.
- Tu poses des questions ouvertes sur la gamme : « Qu'avez-vous de nouveau ? »,
  « Comment se positionne tel produit ? », « Que recommandez-vous pour tel profil ? »
- Tu testes l'adaptabilité du délégué : changements de sujet, objections inattendues.
- Tu juges la MÉTHODE plus que le contenu produit.

RÈGLE ABSOLUE :
Réagis DIRECTEMENT au dernier message du délégué.
Ne répète jamais une formulation déjà utilisée.

PROGRESSION ({_openness_label(openness)}) :
- Tour 1-2 : accueil + question ouverte sur la gamme.
- Tour 3-4 : approfondir un produit + objection sur la méthode.
- Tour 5-6 : tester l'adaptabilité (changer de sujet / nouveau produit).
- Tour 7-8 : conclusion selon impression globale.

STYLE : 2-3 phrases max. Ton : {interlocutor['description_ui']}.
Réponds UNIQUEMENT avec tes paroles. Pas de guillemets, pas de préfixe.

OBJECTIONS TYPIQUES :
{chr(10).join(f'- {o}' for o in interlocutor['objections_favorites'])}
"""


# ══════════════════════════════════════════════════════════════════════
# DÉCISION FINALE — COMMANDE / REFUS / CONDITIONS
# ══════════════════════════════════════════════════════════════════════

# Seuils de décision
SEUIL_COMMANDE    = 7.5   # score global ≥ 7.5 + openness ≥ 4.0 → commande ferme
SEUIL_CONDITIONS  = 5.5   # score global ≥ 5.5 + openness ≥ 3.0 → commande conditionnelle
SEUIL_REFUS_DUR   = 4.0   # score global < 4.0 OU openness < 1.5 → refus sec


def compute_final_decision(
    global_score: float,
    openness: float,
    doctor: Dict,
    product: Dict,
    vm_steps_done: List[int],
    turns: int,
    lang: str = "fr",
) -> Dict:
    """
    Calcule la décision finale de l'interlocuteur (médecin ou pharmacien)
    en combinant :
    - Le score global (NLP 40% + LSTM 25% + conformité 20% + étapes VM 15%)
    - L'ouverture finale
    - Les étapes VM complétées
    - La difficulté du profil
    """
    is_pharm = doctor.get("type") == "pharmacist"
    
    _LABELS = {
        "fr": {
            "type_pharm": "pharmacien", "type_doc": "médecin",
            "cmd_essai": "Commande d'essai passée — {qty} unités", "cmd_ferme": "Commande passée — {qty} unités",
            "cond_pharm": "Intéressé — demande des conditions commerciales", "cond_doc": "Intéressé sous conditions",
            "refus": "Refus — {t} non convaincu",
            "engage_partiel": "Engagement partiel — relance dans 2 semaines",
            "rsn_cmd": "Score {s:.1f}/10 + ouverture {o:.1f}/5",
            "rsn_cond": "Score {s:.1f}/10 — suivi nécessaire",
            "rsn_refus": "Score {s:.1f}/10 — arguments insuffisants",
            "rsn_eng": "Score {s:.1f}/10 + ouverture {o:.1f}/5"
        },
        "en": {
            "type_pharm": "pharmacist", "type_doc": "doctor",
            "cmd_essai": "Trial order placed — {qty} units", "cmd_ferme": "Order placed — {qty} units",
            "cond_pharm": "Interested — asks for commercial terms", "cond_doc": "Interested under conditions",
            "refus": "Refusal — {t} not convinced",
            "engage_partiel": "Partial engagement — follow up in 2 weeks",
            "rsn_cmd": "Score {s:.1f}/10 + openness {o:.1f}/5",
            "rsn_cond": "Score {s:.1f}/10 — follow-up needed",
            "rsn_refus": "Score {s:.1f}/10 — insufficient arguments",
            "rsn_eng": "Score {s:.1f}/10 + openness {o:.1f}/5"
        },
        "es": {
            "type_pharm": "farmacéutico", "type_doc": "médico",
            "cmd_essai": "Pedido de prueba realizado — {qty} unidades", "cmd_ferme": "Pedido realizado — {qty} unidades",
            "cond_pharm": "Interesado — pide condiciones comerciales", "cond_doc": "Interesado bajo condiciones",
            "refus": "Rechazo — {t} no convencido",
            "engage_partiel": "Compromiso parcial — seguimiento en 2 semanas",
            "rsn_cmd": "Puntuación {s:.1f}/10 + apertura {o:.1f}/5",
            "rsn_cond": "Puntuación {s:.1f}/10 — seguimiento necesario",
            "rsn_refus": "Puntuación {s:.1f}/10 — argumentos insuficientes",
            "rsn_eng": "Puntuación {s:.1f}/10 + apertura {o:.1f}/5"
        },
        "tn": {
            "type_pharm": "صيدلي", "type_doc": "طبيب",
            "cmd_essai": "طلب تجريبي تم وضعه — {qty} وحدة", "cmd_ferme": "طلب تم وضعه — {qty} وحدة",
            "cond_pharm": "مهتم — يطلب شروطاً تجارية", "cond_doc": "مهتم بشروط",
            "refus": "رفض — {t} غير مقتنع",
            "engage_partiel": "التزام جزئي — متابعة خلال أسبوعين",
            "rsn_cmd": "النتيجة {s:.1f}/10 + الانفتاح {o:.1f}/5",
            "rsn_cond": "النتيجة {s:.1f}/10 — بحاجة لمتابعة",
            "rsn_refus": "النتيجة {s:.1f}/10 — حجج غير كافية",
            "rsn_eng": "النتيجة {s:.1f}/10 + الانفتاح {o:.1f}/5"
        }
    }
    ll = _LABELS.get(lang, _LABELS["fr"])
    label_type = ll["type_pharm"] if is_pharm else ll["type_doc"]

    # Malus selon difficulté (profil très difficile = seuils plus exigeants)
    diff_malus = {1: 0.0, 2: 0.3, 3: 0.7, 4: 1.2}.get(
        doctor.get("difficulte_n", 2), 0.5
    )

    # Score ajusté
    adjusted_score = global_score - diff_malus

    # Bonus étapes VM complètes
    n_steps  = len(vm_steps_done)
    vm_bonus = min(n_steps * 0.2, 1.0)  # max +1.0

    effective_score = adjusted_score + vm_bonus

    log.info(
        f"[SimRAG] Décision | type={label_type} | global={global_score:.2f} "
        f"| diff_malus={diff_malus} | vm_bonus={vm_bonus:.2f} "
        f"| effective={effective_score:.2f} | openness={openness:.1f}"
    )

    # ── Arbre de décision ──────────────────────────────────────────
    if effective_score >= SEUIL_COMMANDE and openness >= 4.0:
        qty = _estimate_order_qty(effective_score, doctor)
        label = ll["cmd_essai"].format(qty=qty) if is_pharm else ll["cmd_ferme"].format(qty=qty)
        return {
            "decision"     : "commande",
            "label"        : label,
            "reason"       : ll["rsn_cmd"].format(s=effective_score, o=openness),
            "commande_qty" : qty,
            "color"        : "green",
            "icon"         : "🏆",
        }

    elif effective_score >= SEUIL_CONDITIONS and openness >= 3.0:
        label = ll["cond_pharm"] if is_pharm else ll["cond_doc"]
        return {
            "decision"     : "conditions",
            "label"        : label,
            "reason"       : ll["rsn_cond"].format(s=effective_score),
            "commande_qty" : 0,
            "color"        : "gold",
            "icon"         : "⚠️",
        }

    elif effective_score < SEUIL_REFUS_DUR or openness < 1.5:
        label = ll["refus"].format(t=label_type)
        return {
            "decision"     : "refus",
            "label"        : label,
            "reason"       : ll["rsn_refus"].format(s=effective_score),
            "commande_qty" : 0,
            "color"        : "red",
            "icon"         : "❌",
        }

    else:
        return {
            "decision"     : "conditions",
            "label"        : ll["engage_partiel"],
            "reason"       : ll["rsn_eng"].format(s=effective_score, o=openness),
            "commande_qty" : 0,
            "color"        : "gold",
            "icon"         : "🔄",
        }


def _estimate_order_qty(score: float, doctor: Dict) -> int:
    """Estime la quantité commandée selon le score et le profil."""
    base = 10
    if score >= 9.0:
        base = 30
    elif score >= 8.0:
        base = 20
    elif score >= 7.5:
        base = 10
    # Réduction si profil difficile
    diff = doctor.get("difficulte_n", 2)
    factor = {1: 1.5, 2: 1.0, 3: 0.7, 4: 0.5}.get(diff, 1.0)
    return max(5, int(base * factor))


# ══════════════════════════════════════════════════════════════════════
# GÉNÉRATION TEXTE DÉCISION FINALE VIA OLLAMA
# ══════════════════════════════════════════════════════════════════════

# Delegate labels per language
_DELEGATE_LABELS = {"fr": "Délégué", "en": "Delegate", "es": "Delegado", "tn": "المندوب"}
_CLOSING_LANG_INSTRUCTION = {
    "fr": "",
    "en": "\nLANGUAGE RULE: You MUST respond ONLY in English.\n",
    "es": "\nREGLA DE IDIOMA: DEBES responder SOLO en espa\u00f1ol.\n",
    "tn": "\n\u0642\u0627\u0639\u062f\u0629 \u0627\u0644\u0644\u063a\u0629: \u064a\u062c\u0628 \u0623\u0646 \u062a\u0631\u062f \u062d\u0635\u0631\u064a\u064b\u0627 \u0628\u0627\u0644\u0639\u0631\u0628\u064a\u0629.\n",
}


def generate_closing_message(
    doctor: Dict,
    product: Dict,
    decision: Dict,
    global_score: float,
    openness: float,
    conversation_summary: str,
    lang: str = "fr",
) -> str:
    """
    Génère le message de clôture de l'interlocuteur via Ollama,
    cohérent avec la décision calculée et la personnalité.
    Adapté au type (médecin ou pharmacien).
    Fallback statique si Ollama indisponible.
    """
    decision_type = decision["decision"]
    is_pharm = doctor.get("type") == "pharmacist"
    label = "Pharmacien" if is_pharm else "Médecin"

    if decision_type == "commande":
        if is_pharm:
            instruction = (
                f"Tu viens de décider de passer une commande d'essai de {decision['commande_qty']} unités "
                f"de {product['nom']}. Conclus positivement en 2-3 phrases. "
                "Mentionne le conditionnement, les conditions de livraison et demande un bon de commande."
            )
        else:
            instruction = (
                f"Tu viens de décider de passer une commande de {decision['commande_qty']} unités "
                f"de {product['nom']}. Conclus positivement en 2-3 phrases. "
                "Mentionne que tu vas tester sur quelques patients et que tu rappelleras le délégué."
            )
    elif decision_type == "conditions":
        if is_pharm:
            instruction = (
                f"Tu es intéressé par {product['nom']} mais tu veux de meilleures conditions. "
                "Conclus en 2 phrases en demandant une remise de volume ou un accord-cadre."
            )
        else:
            instruction = (
                f"Tu es intéressé par {product['nom']} mais pas encore prêt à commander. "
                "Conclus en 2 phrases en précisant ce qu'il te faudrait encore "
                "(données cliniques, essai gratuit, ou information complémentaire)."
            )
    else:  # refus
        if is_pharm:
            instruction = (
                f"Tu as décidé de ne pas référencer {product['nom']} pour le moment. "
                "Conclus poliment en 2 phrases. Mentionne que ton stock actuel te suffit."
            )
        else:
            instruction = (
                f"Tu as décidé de ne pas commander {product['nom']} pour le moment. "
                "Conclus poliment mais fermement en 2 phrases. "
                "Reste dans ta personnalité de médecin exigeant."
            )

    lang_instr = _CLOSING_LANG_INSTRUCTION.get(lang, "")
    delegate_label = _DELEGATE_LABELS.get(lang, "Delegue")

    system_prompt = (
        f"Tu es {doctor['nom']}, {doctor['specialite']}. "
        f"{doctor['personnalite']} "
        f"Reponds UNIQUEMENT avec tes paroles. Pas de guillemets ni de prefixe."
        f"{lang_instr}"
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"[FIN DE VISITE]\n{instruction}\n\n"
        f"{label} :"
    )

    try:
        payload = json.dumps({
            "model" : "llama3.2:latest",
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 120,
                "stop": [f"\n{delegate_label}", f"\n{label}", f"{delegate_label} :"],
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data     = json.loads(resp.read())
            response = data.get("response", "").strip()
            response = response.split(f"\n{delegate_label}")[0].strip()
            response = response.split(f"\n{label}")[0].strip()
            if response:
                return response
    except Exception as e:
        log.warning(f"[SimRAG] Ollama closing fallback : {e}")

    # Fallback statique selon décision + type + lang
    _FB_PHARM = {
        "fr": {
            "commande": f"Bon, on va essayer. Envoyez-moi {decision.get('commande_qty', 10)} unités de {product['nom']} avec les conditions qu'on a discutées.",
            "conditions": "C'est intéressant mais je voudrais de meilleures conditions. Revenez avec une offre réseau et on en reparle.",
            "refus": "Merci mais j'ai déjà un stock suffisant de produits sur ce segment. Peut-être la prochaine fois."
        },
        "en": {
            "commande": f"Okay, let's try it. Send me {decision.get('commande_qty', 10)} units of {product['nom']} with the terms we discussed.",
            "conditions": "It's interesting, but I'd like better terms. Come back with a network offer and we'll talk.",
            "refus": "Thank you, but I already have enough stock in this segment. Maybe next time."
        },
        "es": {
            "commande": f"Bien, vamos a probar. Envíeme {decision.get('commande_qty', 10)} unidades de {product['nom']} con las condiciones que acordamos.",
            "conditions": "Es interesante, pero me gustarían mejores condiciones. Vuelva con una oferta de red y hablaremos.",
            "refus": "Gracias, pero ya tengo suficiente stock en este segmento. Quizás la próxima vez."
        },
        "tn": {
            "commande": f"حسناً، لنجرب. أرسل لي {decision.get('commande_qty', 10)} وحدة من {product['nom']} بالشروط التي ناقشناها.",
            "conditions": "هذا مثير للاهتمام، لكنني أريد شروطًا أفضل. عد بعرض شبكة وسنتحدث.",
            "refus": "شكرًا لك، لكن لدي مخزون كافٍ في هذا القطاع. ربما في المرة القادمة."
        }
    }
    _FB_DOC = {
        "fr": {
            "commande": f"Bien. Je vais essayer {product['nom']} sur quelques-uns de mes patients. Passez me voir dans un mois avec les retours terrain.",
            "conditions": "C'est intéressant. Envoyez-moi les données cliniques complètes et nous en reparlerons.",
            "refus": "Merci pour votre présentation. Pour l'instant je reste sur mes prescriptions habituelles."
        },
        "en": {
            "commande": f"Good. I will try {product['nom']} on a few of my patients. Come see me in a month with field feedback.",
            "conditions": "That's interesting. Send me the full clinical data and we'll talk about it.",
            "refus": "Thank you for your presentation. For now, I will stick to my usual prescriptions."
        },
        "es": {
            "commande": f"Bien. Probaré {product['nom']} en algunos de mis pacientes. Venga a verme en un mes con los resultados.",
            "conditions": "Es interesante. Envíeme los datos clínicos completos y hablaremos.",
            "refus": "Gracias por su presentación. Por ahora seguiré con mis prescripciones habituales."
        },
        "tn": {
            "commande": f"جيد. سأجرب {product['nom']} على بعض مرضاي. تعال لتراني بعد شهر مع ملاحظات ميدانية.",
            "conditions": "هذا مثير للاهتمام. أرسل لي البيانات السريرية الكاملة وسنتحدث عنها.",
            "refus": "شكرًا لعرضك. في الوقت الحالي، سألتزم بوصفاتي المعتادة."
        }
    }

    if is_pharm:
        fallbacks = _FB_PHARM.get(lang, _FB_PHARM["fr"])
    else:
        fallbacks = _FB_DOC.get(lang, _FB_DOC["fr"])

    return fallbacks.get(decision_type, fallbacks["refus"])
