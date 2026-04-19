"""
engine.py
=========
Moteur de simulation intégré — NLP + LSTM + RAG + SessionState.

v2 — Ajouts Bloc 1 :
  - Support pharmaciens : prompt système adapté (marge, rotation, officine)
  - generate_suggestion() : suggestion de reformulation par tour via Claude
  - Mode généraliste : produit fictif générique, médecin choisit le contexte
  - get_visit_steps() dynamique selon le type d'interlocuteur
"""

import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing  import Dict, List, Optional

from .profiles       import (
    get_product, get_doctor, get_pharmacist, get_interlocutor,
    get_visit_steps, is_pharmacist, VISIT_STEPS,
)
from .session_state  import SessionState
from apps.modeling.simulator_rag import (
    get_product_context,
    build_enriched_doctor_system,
    build_enriched_pharmacist_system as rag_pharmacist_system,
    build_enriched_generalist_system as rag_generalist_system,
    compute_final_decision,
    generate_closing_message,
)

log = logging.getLogger(__name__)

MAX_TURNS      = 8
MAX_TOKENS_RPT = 1200


# ══════════════════════════════════════════════════════════════════════
# NLP MODEL
# ══════════════════════════════════════════════════════════════════════

def _get_nlp_model():
    try:
        from django.conf import settings
        models_dir = str(settings.MODELS_AI_DIR)
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        import nlp_scoring_train_v2  # noqa
        from nlp_scoring_model_v2 import NLPScoringModel
        bundle = Path(settings.MODELS_AI_DIR) / 'nlp_scoring_bundle_v2.pkl'
        return NLPScoringModel.load(str(bundle))
    except Exception as e:
        log.warning(f"NLPScoringModel non disponible : {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# RAG
# ══════════════════════════════════════════════════════════════════════

def _get_rag_context(product_name: str, delegate_text: str) -> str:
    """Délègue au SimulatorRAG dédié (données KB VITAL SA enrichies)."""
    return get_product_context(product_name, delegate_text, n_docs=3)


# ══════════════════════════════════════════════════════════════════════
# OLLAMA — avec fix anti-répétition
# ══════════════════════════════════════════════════════════════════════

# ── Language labels for multilingual Ollama prompts ───────────────────
_LANG_LABELS = {
    "fr": {"delegate": "Délégué", "forbidden_header": "PHRASES QUE TU AS DÉJÀ DITES \u2014 INTERDITES",
           "forbidden_footer": "Ne commence PAS ta réponse par l'une de ces phrases. Varie obligatoirement.",
           "rag_header": "DONNÉES PRODUIT \u2014 Base VITAL SA",
           "rag_footer": "Utilise ces données si pertinent. Ne les récite pas."},
    "en": {"delegate": "Delegate", "forbidden_header": "PHRASES YOU ALREADY SAID \u2014 FORBIDDEN",
           "forbidden_footer": "Do NOT start your response with any of these phrases. Always vary.",
           "rag_header": "PRODUCT DATA \u2014 VITAL SA Database",
           "rag_footer": "Use this data if relevant. Do not recite it."},
    "es": {"delegate": "Delegado", "forbidden_header": "FRASES QUE YA HAS DICHO \u2014 PROHIBIDAS",
           "forbidden_footer": "NO empieces tu respuesta con ninguna de estas frases. Var\u00eda siempre.",
           "rag_header": "DATOS DEL PRODUCTO \u2014 Base VITAL SA",
           "rag_footer": "Usa estos datos si es pertinente. No los recites."},
    "ar": {"delegate": "\u0627\u0644\u0645\u0646\u062f\u0648\u0628", "forbidden_header": "\u0639\u0628\u0627\u0631\u0627\u062a \u0642\u0644\u062a\u0647\u0627 \u0645\u0633\u0628\u0642\u064b\u0627 \u2014 \u0645\u0645\u0646\u0648\u0639\u0629",
           "forbidden_footer": "\u0644\u0627 \u062a\u0628\u062f\u0623 \u0631\u062f\u0643 \u0628\u0623\u064a \u0645\u0646 \u0647\u0630\u0647 \u0627\u0644\u0639\u0628\u0627\u0631\u0627\u062a. \u0646\u0648\u0651\u0639 \u062f\u0627\u0626\u0645\u064b\u0627.",
           "rag_header": "\u0628\u064a\u0627\u0646\u0627\u062a \u0627\u0644\u0645\u0646\u062a\u062c \u2014 \u0642\u0627\u0639\u062f\u0629 VITAL SA",
           "rag_footer": "\u0627\u0633\u062a\u062e\u062f\u0645 \u0647\u0630\u0647 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a \u0625\u0646 \u0643\u0627\u0646\u062a \u0630\u0627\u062a \u0635\u0644\u0629. \u0644\u0627 \u062a\u0633\u0631\u062f\u0647\u0627 \u062d\u0631\u0641\u064a\u064b\u0627."},
}

_LANG_INSTRUCTION = {
    "fr": "",  # default, no extra instruction needed
    "en": (
        "CRITICAL LANGUAGE RULE — YOU MUST FOLLOW THIS:\n"
        "The delegate is speaking ENGLISH.\n"
        "You MUST respond ONLY in ENGLISH. Every single word must be in English.\n"
        "Do NOT use any French words. Do NOT respond in French.\n"
        "If you respond in French, you have FAILED.\n"
        "RESPOND IN ENGLISH ONLY."
    ),
    "es": (
        "REGLA DE IDIOMA CRÍTICA — DEBES SEGUIR ESTA REGLA:\n"
        "El delegado habla ESPAÑOL.\n"
        "DEBES responder SOLO en ESPAÑOL. Cada palabra debe ser en español.\n"
        "NO uses palabras en francés. NO respondas en francés.\n"
        "Si respondes en francés, has FALLADO.\n"
        "RESPONDE SOLO EN ESPAÑOL."
    ),
    "ar": (
        "\u0642\u0627\u0639\u062f\u0629 \u0627\u0644\u0644\u063a\u0629 \u0627\u0644\u062d\u0627\u0633\u0645\u0629 \u2014 \u064a\u062c\u0628 \u0627\u062a\u0628\u0627\u0639 \u0647\u0630\u0647 \u0627\u0644\u0642\u0627\u0639\u062f\u0629:\n"
        "\u0627\u0644\u0645\u0646\u062f\u0648\u0628 \u064a\u062a\u062d\u062f\u062b \u0627\u0644\u0639\u0631\u0628\u064a\u0629.\n"
        "\u064a\u062c\u0628 \u0623\u0646 \u062a\u0631\u062f \u062d\u0635\u0631\u064a\u064b\u0627 \u0628\u0627\u0644\u0639\u0631\u0628\u064a\u0629. \u0643\u0644 \u0643\u0644\u0645\u0629 \u064a\u062c\u0628 \u0623\u0646 \u062a\u0643\u0648\u0646 \u0628\u0627\u0644\u0639\u0631\u0628\u064a\u0629.\n"
        "\u0644\u0627 \u062a\u0633\u062a\u062e\u062f\u0645 \u0623\u064a \u0643\u0644\u0645\u0627\u062a \u0641\u0631\u0646\u0633\u064a\u0629. \u0644\u0627 \u062a\u0631\u062f \u0628\u0627\u0644\u0641\u0631\u0646\u0633\u064a\u0629.\n"
        "RESPOND IN ARABIC ONLY."
    ),
}


def _call_ollama(system: str, history: List[Dict],
                 rag_context: str, recent_msgs: List[str],
                 interlocutor_label: str = "Médecin",
                 lang: str = "fr") -> str:
    ll = _LANG_LABELS.get(lang, _LANG_LABELS["fr"])
    delegate_label = ll["delegate"]

    history_text = ""
    for msg in history:
        role = delegate_label if msg["role"] == "user" else interlocutor_label
        history_text += f"\n{role} : {msg['content']}"

    rag_section = ""
    if rag_context:
        rag_section = (
            f"\n\n[{ll['rag_header']}]\n{rag_context}\n"
            f"{ll['rag_footer']}\n"
        )

    forbidden_section = ""
    if recent_msgs:
        forbidden_list = "\n".join(f"  - \"{m[:80]}\"" for m in recent_msgs)
        forbidden_section = (
            f"\n\n[{ll['forbidden_header']}]\n"
            f"{forbidden_list}\n"
            f"{ll['forbidden_footer']}\n"
        )

    # Build language prefix — placed FIRST so LLM prioritises it
    # AND language suffix — placed RIGHT BEFORE generation marker
    # (sandwiching the French content forces the model to stay in target lang)
    lang_prefix = ""
    lang_suffix = ""
    if lang and lang != "fr":
        lang_prefix = _LANG_INSTRUCTION.get(lang, "")
        if lang_prefix:
            lang_prefix = lang_prefix.strip() + "\n\n"
            # Short reminder right before the generation point
            _LANG_REMINDER = {
                "en": "\n[REMINDER: RESPOND IN ENGLISH ONLY — NOT FRENCH]\n",
                "es": "\n[RECORDATORIO: RESPONDE SOLO EN ESPAÑOL — NO EN FRANCÉS]\n",
                "ar": "\n[تذكير: رد بالعربية فقط — ليس بالفرنسية]\n",
            }
            lang_suffix = _LANG_REMINDER.get(lang, "")

    print(f"[Ollama] lang={lang} | label={interlocutor_label} | delegate={delegate_label}")

    full_prompt = (
        f"{lang_prefix}"
        f"{system}"
        f"{rag_section}"
        f"{forbidden_section}"
        f"{lang_suffix}"
        f"\n\n[CONVERSATION]\n{history_text}\n{interlocutor_label} :"
    )

    stop_tokens = [f"\n{delegate_label}", f"\n{interlocutor_label}", f"{delegate_label} :"]

    payload = json.dumps({
        "model" : "llama3.2:latest",
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.75,
            "num_predict": 180,
            "stop": stop_tokens,
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=payload, headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data     = json.loads(resp.read())
        response = data.get('response', '').strip()
        response = response.split(f'\n{delegate_label}')[0].strip()
        response = response.split(f'\n{interlocutor_label}')[0].strip()
        return response


def _call_claude_doctor(system: str, history: List[Dict]) -> str:
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514", "max_tokens": 400,
        "system": system, "messages": history,
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=payload, headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read())['content'][0]['text'].strip()


def _call_claude_report(messages: List[Dict], system: str) -> str:
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514", "max_tokens": MAX_TOKENS_RPT,
        "system": system, "messages": messages,
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=payload, headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())['content'][0]['text'].strip()


def _generate_interlocutor_response(system: str, history: List[Dict],
                                     rag_context: str, recent_msgs: List[str],
                                     fallback_fn, label: str = "Médecin",
                                     lang: str = "fr") -> tuple:
    try:
        msg = _call_ollama(system, history, rag_context, recent_msgs, label, lang=lang)
        if not msg:
            raise ValueError("Réponse Ollama vide")
        return msg, "ollama+rag" if rag_context else "ollama"
    except Exception as e_ollama:
        log.info(f"Ollama → fallback Claude ({e_ollama})")
        try:
            return _call_claude_doctor(system, history), "claude_fallback"
        except Exception:
            return fallback_fn(), "static_fallback"


# ══════════════════════════════════════════════════════════════════════
# PROMPT PHARMACIEN
# ══════════════════════════════════════════════════════════════════════

def _build_pharmacist_system(pharmacist: dict, product: dict,
                               turn: int, openness: float) -> str:
    """
    Prompt système spécifique à la visite officine.
    Le pharmacien raisonne en termes de marge, stock, rotation, DLC,
    conditionnement et relation commerciale — pas en termes cliniques.
    """
    product_section = (
        f"Produit présenté : {product['nom']} ({product['categorie']})\n"
        f"Indication : {product['indication']}\n"
        f"Argument clé : {product['argument_cle']}\n"
    )

    return f"""Tu es {pharmacist['nom']}, {pharmacist['specialite']} à {pharmacist['ville']}.

PERSONNALITÉ :
{pharmacist['personnalite']}

CONTEXTE DE LA VISITE OFFICINE :
{product_section}
Tour actuel : {turn}/{MAX_TURNS}
Ouverture : {openness:.1f}/5

CADRE DE LA VISITE OFFICINE (DIFFÉRENT D'UN MÉDECIN) :
- Tu penses en PHARMACIEN : marge brute, rotation de stock, DLC, conditionnement, PLV.
- Tu ne prescris pas — tu recommandes et tu vends au comptoir.
- Tes objections portent sur : le prix d'achat, la concurrence en rayon, le conditionnement,
  les conditions de retour, les remises de volume, la demande spontanée de tes clients.
- Tu n'es PAS intéressé par les mécanismes d'action cliniques (laisse ça au médecin).
- Tu veux savoir : est-ce que ça se vend bien ? Quelle marge ? Quel support terrain ?

PROGRESSION ({_openness_label(openness)}) :
- Tour 1-2 : accueil neutre + première objection commerciale.
- Tour 3-4 : creuser les conditions commerciales ou la demande marché.
- Tour 5-6 : position claire selon ton intérêt.
- Tour 7-8 : décision (commande d'essai ou refus).

STYLE : 2-3 phrases max. Ton direct, professionnel, commercial.
Réponds UNIQUEMENT avec tes paroles. Pas de guillemets, pas de préfixe.

OBJECTIONS TYPIQUES :
{chr(10).join(f'- {o}' for o in pharmacist['objections_favorites'])}
"""


def _openness_label(openness: float) -> str:
    if openness >= 4.2:
        return "Clairement intéressé — questions positives"
    elif openness >= 3.5:
        return "En train de s'ouvrir — demande des précisions"
    elif openness >= 2.5:
        return "Neutre — objection modérée"
    elif openness >= 1.5:
        return "Sceptique — doute direct"
    else:
        return "Très fermé — fin de visite imminente"


def _build_doctor_system(doctor: dict, product: dict,
                          turn: int, openness: float) -> str:
    return f"""Tu es {doctor['nom']}, {doctor['specialite']} à {doctor['ville']}.

PERSONNALITÉ :
{doctor['personnalite']}

CONTEXTE :
- Produit présenté : {product['nom']} ({product['categorie']})
- Tour actuel : {turn}/{MAX_TURNS}
- Ouverture : {openness:.1f}/5

RÈGLE ABSOLUE — RÉAGIS AU DERNIER MESSAGE PRÉCISEMENT :
Lis ce que vient de dire le délégué et réponds-y DIRECTEMENT.
Ne répète jamais une formulation déjà utilisée dans cette conversation.

PROGRESSION ({_openness_label(openness)})

Tour 1-2 : accueil + première objection/question.
Tour 3-4 : approfondissement, objection précise.
Tour 5-6 : position claire selon ton ouverture.
Tour 7-8 : conclusion.

STYLE : 2-3 phrases max. Ton : {doctor['description_ui']}.
Réponds UNIQUEMENT avec tes paroles. Pas de guillemets, pas de préfixe.

OBJECTIONS TYPIQUES :
{chr(10).join(f'- {o}' for o in doctor['objections_favorites'])}
"""


def _build_report_system(lang: str = "fr") -> str:
    lang_map = {
        "fr": "Français", "en": "English", "es": "Español", "ar": "العربية"
    }
    lang_name = lang_map.get(lang, "Français")
    return (
        "Tu es ALIA, évaluateur IA de VITAL SA. "
        "Génère un rapport de visite concis et actionnable. "
        f"IMPORTANT: Rédige le contenu du rapport EXACTEMENT et ENTIÈREMENT en {lang_name}. "
        "Réponds UNIQUEMENT en JSON valide, sans markdown."
    )


# ══════════════════════════════════════════════════════════════════════
# SUGGESTION DE REFORMULATION (Bloc 1 — nouvelle feature)
# ══════════════════════════════════════════════════════════════════════

# ── Pool de suggestions statiques variées (fallback) ─────────────
_LANG_NAMES = {"fr": "Français", "en": "English", "es": "Español", "ar": "العربية"}

_SUGGESTION_POOL_DOCTOR = {
    "fr": [
        "💡 Reformulez avec une donnée clinique : « Les études montrent une réduction de {pct}% des effets secondaires… »",
        "💡 Utilisez la technique du miroir : « Si je comprends bien, votre préoccupation porte sur… »",
        "💡 Appuyez sur le bénéfice patient : « Pour vos patients {indication}, cela signifie concrètement… »",
        "💡 Posez une question de validation : « Est-ce que cela répond à votre interrogation, Docteur ? »",
        "💡 Renforcez avec un cas concret : « Un confrère a observé chez ses patients… »",
    ],
    "en": [
        "💡 Rephrase with clinical data: 'Studies show a {pct}% reduction in side effects...'",
        "💡 Use the mirror technique: 'If I understand correctly, your concern is about...'",
        "💡 Emphasize patient benefit: 'For your patients with {indication}, this practically means...'",
        "💡 Ask a validation question: 'Does this answer your question, Doctor?'",
        "💡 Reinforce with a concrete case: 'A colleague observed in their patients...'",
    ],
    "es": [
        "💡 Reformular con datos clínicos: 'Los estudios muestran una reducción del {pct}% en efectos secundarios...'",
        "💡 Usar la técnica del espejo: 'Si entiendo bien, su preocupación es sobre...'",
        "💡 Enfatizar el beneficio para el paciente: 'Para sus pacientes con {indication}, esto significa...'",
        "💡 Hacer una pregunta de validación: '¿Responde esto a su pregunta, Doctor?'",
        "💡 Reforzar con un caso concreto: 'Un colega observó en sus pacientes...'",
    ],
    "ar": [
        "💡 أعد الصياغة ببيانات سريرية: 'تظهر الدراسات انخفاضاً بنسبة {pct}% في الآثار الجانبية...'",
        "💡 استخدم تقنية المرآة: 'إذا كنت أفهم بشكل صحيح، فإن قلقك يتعلق بـ...'",
        "💡 أكد على فائدة المريض: 'بالنسبة لمرضاك الذين يعانون من {indication}، هذا يعني عملياً...'",
        "💡 اطرح سؤال تحقق: 'هل هذا يجيب على تساؤلك، دكتور؟'",
        "💡 عزز بحالة ملموسة: 'لاحظ أحد الزملاء لدى مرضاه...'",
    ],
}

_SUGGESTION_POOL_PHARMACIST = {
    "fr": [
        "💡 Parlez de la marge : « Avec un PPA de X DT et un prix public de Y DT, votre marge est de Z%. »",
        "💡 Citez des données de rotation : « Dans les officines pilotes, le sell-out moyen est de N unités/mois. »",
        "💡 Proposez un soutien terrain : « Nous fournissons des présentoirs PLV et des fiches conseil patient. »",
        "💡 Répondez sur la DLC : « La DLC est de 24 mois, ce qui vous laisse une marge confortable. »",
        "💡 Valorisez la demande : « La demande spontanée est en hausse sur ce segment. »",
    ],
    "en": [
        "💡 Talk about the margin: 'With a wholesale price of X and public price of Y, your margin is Z%.'",
        "💡 Quote turnover data: 'In pilot pharmacies, the average sell-out is N units/month.'",
        "💡 Offer field support: 'We provide POS displays and patient advice sheets.'",
        "💡 Reply about the shelf life: 'The shelf life is 24 months, leaving you a comfortable margin.'",
        "💡 Emphasize demand: 'Spontaneous demand is increasing in this segment.'",
    ],
    "es": [
        "💡 Hablar del margen: 'Con un precio de compra de X y un precio al público de Y, su margen es del Z%.'",
        "💡 Citar datos de rotación: 'En farmacias piloto, la venta promedio es de N unidades/mes.'",
        "💡 Ofrecer apoyo en el terreno: 'Proporcionamos exhibidores de punto de venta y fichas de consejos clínicos.'",
        "💡 Responder sobre la caducidad: 'La caducidad es de 24 meses, lo que le deja un margen cómodo.'",
        "💡 Destacar la demanda: 'La demanda espontánea está aumentando en este segmento.'",
    ],
    "ar": [
        "💡 تحدث عن هامش الربح: 'مع سعر جملة X وسعر بيع للجمهور Y، هامش ربحك هو Z%.'",
        "💡 اذكر بيانات دورة المبيعات: 'في الصيدليات الرائدة، متوسط المبيعات هو N وحدة/شهر.'",
        "💡 اعرض الدعم الميداني: 'نوفر منصات عرض ونشرات نصائح للمرضى.'",
        "💡 أجب حول تاريخ الصلاحية: 'تاريخ الصلاحية هو 24 شهراً، مما يترك لك هامشاً مريحاً.'",
        "💡 أبرز الطلب: 'الطلب التلقائي يتزايد في هذا القطاع.'",
    ],
}


def generate_suggestion(
    delegate_text: str,
    doctor_msg: str,
    score: float,
    acrv: int,
    conformite: bool,
    step: Optional[int],
    product: dict,
    interlocutor: dict,
    previous_suggestions: Optional[List[str]] = None,
    lang: str = "fr",
) -> str:
    """
    Génère une suggestion de reformulation concrète pour le délégué.
    Chaque suggestion est :
      - en rapport DIRECT avec le dernier message de l'interlocuteur
      - différente des suggestions précédentes (anti-répétition)
      - variée dans ses axes (données cliniques, A-C-R-V, closing, etc.)
      - dans la langue cible detectée (lang)

    Retourne : str (1-2 phrases max, directement actionnables)
    """
    import random

    previous_suggestions = previous_suggestions or []

    # ── Axe de coaching aléatoire pour variété ────────────────────
    axes = []
    if not conformite:
        axes.append("CONFORMITÉ : le délégué a utilisé un mot tueur. Suggère une reformulation conforme.")
    if acrv < 2:
        axes.append("A-C-R-V : le délégué n'applique pas la méthode. Suggère comment structurer sa réponse en Accueil→Clarification→Réponse→Validation.")
    if score < 5.0:
        axes.append("ARGUMENTATION FAIBLE : le score est bas. Suggère l'utilisation de l'argument clé du produit ou de données chiffrées.")
    if score >= 7.0:
        axes.append("CLOSING : bonne performance. Suggère une technique de micro-engagement ou de validation.")
    if step is None:
        axes.append("MÉTHODE DE VISITE : aucune étape détectée. Suggère d'intégrer une étape (permission, sondage, preuve…).")
    # Toujours au moins un axe générique
    axes.append("REFORMULATION : propose une meilleure façon de formuler la réponse en se basant précisément sur ce que l'interlocuteur vient de dire.")

    chosen_axis = random.choice(axes)
    visit_type = "pharmacien" if is_pharmacist(interlocutor) else "médecin"

    # ── Section anti-répétition ───────────────────────────────────
    anti_repeat = ""
    if previous_suggestions:
        prev_list = "\n".join(f"  - \"{s[:100]}\"" for s in previous_suggestions[-5:])
        anti_repeat = f"""\n\n[SUGGESTIONS DÉJÀ DONNÉES — NE PAS RÉPÉTER]
{prev_list}
Tu DOIS proposer une suggestion COMPLÈTEMENT DIFFÉRENTE de celles ci-dessus."""

    lang_name = _LANG_NAMES.get(lang, "Français")

    prompt = f"""Tu es ALIA, coach IA de délégués médicaux VITAL SA.

Un délégué vient de répondre à un {visit_type} lors d'une simulation de visite.

PRODUIT : {product['nom']} ({product['categorie']})
INDICATION : {product['indication']}
ARGUMENT CLÉ : {product['argument_cle']}

DERNIER MESSAGE DU {visit_type.upper()} : "{doctor_msg}"
RÉPONSE DU DÉLÉGUÉ : "{delegate_text}"

SCORE NLP : {score:.1f}/10
A-C-R-V : {acrv}/4
CONFORMITÉ : {"✅ OK" if conformite else "❌ MOT TUEUR DÉTECTÉ"}

AXE DE COACHING À SUIVRE : {chosen_axis}
{anti_repeat}

RÈGLES :
1. Ta suggestion DOIT répondre DIRECTEMENT à ce que le {visit_type} vient de dire.
2. Commence par "💡" puis une formulation concrète que le délégué aurait pu utiliser.
3. Sois spécifique au produit {product['nom']} et au contexte {visit_type}.
4. 1-2 phrases MAXIMUM. Pas d'introduction, pas de conclusion.
5. Varie le style : parfois une reformulation, parfois une question, parfois une donnée chiffrée.
6. EXTRÊMEMENT IMPORTANT: Rédige la suggestion ENTIÈREMENT et EXCLUSIVEMENT en {lang_name}."""

    try:
        payload = json.dumps({
            "model"     : "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "messages"  : [{"role": "user", "content": prompt}],
        }).encode('utf-8')
        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            suggestion = result['content'][0]['text'].strip()
            if suggestion:
                return suggestion
    except Exception as e:
        log.warning(f"[Suggestion] Claude indisponible : {e}")

    # ── Fallback statique varié ────────────────────────────────────
    pool_dict = _SUGGESTION_POOL_PHARMACIST if is_pharmacist(interlocutor) else _SUGGESTION_POOL_DOCTOR
    pool = pool_dict.get(lang, pool_dict["fr"])
    # Filtrer celles déjà données
    available = [s for s in pool if s not in previous_suggestions]
    if not available:
        available = pool  # reset si toutes utilisées
    suggestion = random.choice(available)
    # Remplacer les placeholders
    suggestion = suggestion.replace("{produit}", product['nom'])
    suggestion = suggestion.replace("{argument_cle}", product['argument_cle'])
    suggestion = suggestion.replace("{indication}", product['indication'])
    # fallback values for other missing format keys just in case
    suggestion = suggestion.replace("{pct}", str(random.choice([15, 20, 30, 45])))
    return suggestion


# ══════════════════════════════════════════════════════════════════════
# SIMULATION SESSION — UNIFIÉE médecin + pharmacien
# ══════════════════════════════════════════════════════════════════════

class SimulationSession:
    """
    Session de simulation intégrée.
    Supporte médecins et pharmaciens avec des prompts adaptés.
    Tous les modules (NLP, LSTM, RAG) alimentent SessionState.
    """

    def __init__(self, interlocutor_id: str, product_id: str, niveau_alia: str):
        self.interlocutor   = get_interlocutor(interlocutor_id)
        self.product        = get_product(product_id)
        self.niveau_alia    = niveau_alia
        self.turn           = 0
        self.openness       = float(self.interlocutor['ouverture_initiale'])
        self.history        = []
        self.scores_history = []
        self.step_history   = []
        self.is_finished    = False
        self.nlp_model      = _get_nlp_model()
        self.final_decision = None
        self.suggestion_history = []   # ← anti-répétition des suggestions
        self.language       = "fr"     # ← langue détectée (FR par défaut)

        # Détecter le type pour adapter le comportement
        self._is_pharmacist = is_pharmacist(self.interlocutor)
        self._is_generalist = self.product.get('_is_generic', False)
        self._visit_steps   = get_visit_steps(self.interlocutor)
        self._label         = "Pharmacien" if self._is_pharmacist else "Médecin"

        # Compatibilité avec l'ancien code (doctor = interlocuteur)
        self.doctor = self.interlocutor

        self.state = SessionState()

    def _recent_msgs(self, n: int = 4) -> List[str]:
        msgs = [m["content"] for m in self.history if m["role"] == "assistant"]
        return msgs[-n:] if msgs else []

    # ── Prompt système unifié ─────────────────────────────────────

    def _get_system_prompt(self, turn: int, openness: float) -> str:
        lang = self.language or "fr"
        # Mode généraliste : prompt dédié
        if self._is_generalist:
            try:
                return rag_generalist_system(
                    self.interlocutor, self.product, turn, openness, MAX_TURNS, lang=lang)
            except Exception:
                pass  # fallback ci-dessous
        # Pharmacien : utiliser le RAG enrichi pharmacien
        if self._is_pharmacist:
            try:
                return rag_pharmacist_system(
                    self.interlocutor, self.product, turn, openness, MAX_TURNS, lang=lang)
            except Exception:
                return _build_pharmacist_system(
                    self.interlocutor, self.product, turn, openness)
        # Médecin : utiliser le RAG enrichi
        return build_enriched_doctor_system(
            self.interlocutor, self.product, turn, openness, MAX_TURNS, lang=lang)

    # ── Premier message ──────────────────────────────────────────

    def first_message(self) -> Dict:
        system  = self._get_system_prompt(0, self.openness)
        rag_ctx = _get_rag_context(
            self.product['nom'],
            f"présentation {self.product['categorie']} {self.product['indication']}"
        )

        lang = self.language or "fr"

        # Multilingual opening content — the trigger that sets the LLM's language
        _OPENING_PHARMA = {
            "fr": (
                f"Un délégué de VITAL SA entre dans ton officine pour te présenter "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"Accueille-le brièvement selon ta personnalité de pharmacien."
            ),
            "en": (
                f"A VITAL SA representative enters your pharmacy to present "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"Welcome them briefly according to your pharmacist personality. RESPOND IN ENGLISH."
            ),
            "es": (
                f"Un delegado de VITAL SA entra en tu farmacia para presentarte "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"Recíbelo brevemente según tu personalidad de farmacéutico. RESPONDE EN ESPAÑOL."
            ),
            "ar": (
                f"مندوب VITAL SA يدخل صيدليتك لتقديم "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"رحب به باختصار حسب شخصيتك كصيدلي. رد بالعربية."
            ),
        }
        _OPENING_DOCTOR = {
            "fr": (
                f"Un délégué de VITAL SA entre pour présenter "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"Accueille-le en 1-2 phrases selon ta personnalité."
            ),
            "en": (
                f"A VITAL SA representative enters to present "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"Welcome them in 1-2 sentences according to your personality. RESPOND IN ENGLISH."
            ),
            "es": (
                f"Un delegado de VITAL SA entra para presentar "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"Recíbelo en 1-2 frases según tu personalidad. RESPONDE EN ESPAÑOL."
            ),
            "ar": (
                f"مندوب VITAL SA يدخل لتقديم "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"رحب به في جملة أو اثنتين حسب شخصيتك. رد بالعربية."
            ),
        }
        _FALLBACK = {
            "fr": "Oui, entrez. Je vous écoute.",
            "en": "Yes, come in. I'm listening.",
            "es": "Sí, pase. Le escucho.",
            "ar": "نعم، تفضل. أنا أستمع.",
        }

        if self._is_pharmacist:
            opening_content = _OPENING_PHARMA.get(lang, _OPENING_PHARMA["fr"])
        else:
            opening_content = _OPENING_DOCTOR.get(lang, _OPENING_DOCTOR["fr"])

        opening = {"role": "user", "content": opening_content}
        msg, engine = _generate_interlocutor_response(
            system=system, history=[opening],
            rag_context=rag_ctx, recent_msgs=[],
            fallback_fn=lambda: _FALLBACK.get(lang, _FALLBACK["fr"]),
            label=self._label,
            lang=lang,
        )
        self.history.append({"role": "assistant", "content": msg})
        self.state.push_rag_hit(0, self.product['nom'], bool(rag_ctx), engine)

        _COACH_START = {
            "fr": "La visite commence — demandez la permission (Étape 1).",
            "en": "The visit begins — ask for permission (Step 1).",
            "es": "La visita comienza — pida permiso (Etapa 1).",
            "ar": "بدأت الزيارة — اطلب الإذن (المرحلة 1).",
        }

        log.info(f"[Sim] Start — type={self._label} lang={lang} engine={engine}")
        return {
            "message"         : msg,
            "turn"            : self.turn,
            "openness"        : round(self.openness, 1),
            "step"            : None,
            "score"           : None,
            "coach"           : _COACH_START.get(lang, _COACH_START["fr"]),
            "suggestion"      : None,
            "is_first"        : True,
            "engine"          : engine,
            "rag_used"        : bool(rag_ctx),
            "interlocutor_type": self._label,
        }

    # ── Tour de conversation ──────────────────────────────────────

    def process_delegate_response(self, delegate_text: str, lang: str = None, user_id: int = None) -> Dict:
        # Update session language if provided
        if lang:
            self.language = lang
        self.turn += 1

        # ── 1. NLP ────────────────────────────────────────────────
        nlp        = self._evaluate_nlp(delegate_text)
        score      = nlp.get('overall_score', 5.0)
        acrv       = nlp.get('acrv_score', 0)
        acrv_det   = nlp.get('acrv_detail', {})
        conformite = nlp.get('conformite', True)
        quality    = nlp.get('quality', 'Bon')
        scores_det = nlp.get('scores', {})
        feedback   = nlp.get('feedback_coaching', [])

        self.scores_history.append({
            'turn': self.turn, 'score': score, 'quality': quality,
            'acrv': acrv, 'conformite': conformite,
        })

        # ── 2. Ouverture + SessionState ───────────────────────────
        delta      = self._delta_openness(score, conformite, quality)
        self.openness = max(1.0, min(5.0, self.openness + delta))
        self.state.push_nlp_turn(
            turn=self.turn, score=score, quality=quality,
            acrv=acrv, conformite=conformite, scores_det=scores_det,
            feedback=feedback, openness=self.openness,
        )

        # ── 3. Étapes VM ──────────────────────────────────────────
        step = self._detect_step(delegate_text)
        self.step_history.append(step)
        self.state.push_vm_step(step)

        # ── 4. Coaching ───────────────────────────────────────────
        coach = self._build_coaching(score, acrv, conformite, step, quality, feedback)

        # ── 5. Suggestion de reformulation (contextuelle + anti-répétition)
        last_interlocutor_msg = next(
            (m["content"] for m in reversed(self.history) if m["role"] == "assistant"), ""
        )
        suggestion = generate_suggestion(
            delegate_text=delegate_text,
            doctor_msg=last_interlocutor_msg,
            score=score,
            acrv=acrv,
            conformite=conformite,
            step=step,
            product=self.product,
            interlocutor=self.interlocutor,
            previous_suggestions=self.suggestion_history,
            lang=self.language,
        )
        self.suggestion_history.append(suggestion)

        # ── Sauvegarde BDD Analytics (Métrique Utilisateur) ───────
        if user_id:
            try:
                from apps.avatar.models import NLPAnalysis
                from django.contrib.auth import get_user_model
                User_model = get_user_model()
                u = User_model.objects.get(id=user_id)
                NLPAnalysis.objects.create(
                    user=u,
                    objection=last_interlocutor_msg,
                    response=delegate_text,
                    niveau_alia_input=self.niveau_alia,
                    quality=quality,
                    overall_score=score,
                    score_scientific=float(scores_det.get('scientific', 0)),
                    score_clarity=float(scores_det.get('clarity', 0)),
                    score_objection=float(scores_det.get('objection', 0)),
                    sentiment=nlp.get('sentiment', 'Neutre'),
                    niveau_alia_pred=nlp.get('niveau_pred', 'Junior'),
                    visit_format=nlp.get('format', 'Standard'),
                    conformite=conformite,
                    acrv_score=acrv,
                )
            except Exception as e:
                log.error(f"[Sim] Erreur sauvegarde NLPAnalysis: {e}")

        # ── 6. Fin de visite ? ────────────────────────────────────
        should_close = (
            self.turn >= MAX_TURNS
            or (self.openness >= 4.2 and self.turn >= 4)
            or (self.openness <= 1.2 and self.turn >= 3)
        )

        # ── 7. RAG + prompt ───────────────────────────────────────
        self.history.append({"role": "user", "content": delegate_text})
        rag_ctx = _get_rag_context(self.product['nom'], delegate_text)
        system  = self._get_system_prompt(self.turn, self.openness)

        # ── 8. Réponse interlocuteur ──────────────────────────────
        recent = self._recent_msgs(n=4)
        if should_close:
            global_score_now = self.state.global_score
            vm_steps_now     = list(self.state.vm_steps_done)
            conv_summary     = " | ".join(
                m["content"][:60] for m in self.history[-4:] if m["role"] == "user"
            )
            self.final_decision = compute_final_decision(
                global_score=global_score_now,
                openness=self.openness,
                doctor=self.interlocutor,
                product=self.product,
                vm_steps_done=vm_steps_now,
                turns=self.turn,
                lang=self.language,
            )
            msg    = generate_closing_message(
                doctor=self.interlocutor, product=self.product,
                decision=self.final_decision,
                global_score=global_score_now,
                openness=self.openness,
                conversation_summary=conv_summary,
                lang=self.language,
            )
            engine = "sim_rag_decision"
            log.info(f"[Sim] Décision : {self.final_decision['decision']} | {self.final_decision['label']}")
        else:
            msg, engine = _generate_interlocutor_response(
                system=system, history=self.history,
                rag_context=rag_ctx, recent_msgs=recent,
                fallback_fn=lambda: self._static_fallback(False),
                label=self._label,
                lang=self.language,
            )

        self.history.append({"role": "assistant", "content": msg})
        self.state.push_rag_hit(self.turn, self.product['nom'], bool(rag_ctx), engine)

        if should_close:
            self.is_finished = True

        global_score  = self.state.global_score
        global_niveau = self.state.global_niveau

        log.info(
            f"[Sim] Tour {self.turn} | NLP={score:.2f} | open={self.openness:.1f} "
            f"| global={global_score:.2f} | engine={engine}"
        )

        return {
            "message"        : msg,
            "turn"           : self.turn,
            "openness"       : round(self.openness, 1),
            "step"           : step,
            "score"          : round(score, 2),
            "quality"        : quality,
            "acrv"           : acrv,
            "acrv_detail"    : acrv_det,
            "conformite"     : conformite,
            "scores"         : scores_det,
            "coach"          : coach,
            "suggestion"     : suggestion,       # ← NOUVEAU
            "is_finished"    : self.is_finished,
            "delta_open"     : round(delta, 2),
            "engine"         : engine,
            "rag_used"       : bool(rag_ctx),
            "global_score"   : global_score,
            "global_niveau"  : global_niveau,
            "final_decision" : self.final_decision,
            "interlocutor_type": self._label,
        }

    # ── Rapport final ────────────────────────────────────────────

    def generate_report(self) -> Dict:
        n = len(self.scores_history)
        if n == 0:
            return {"error": "Aucune donnée de visite."}

        scores    = [s['score'] for s in self.scores_history]
        avg_score = round(sum(scores) / n, 2)
        max_score = round(max(scores), 2)
        acrv_avg  = round(sum(s['acrv'] for s in self.scores_history) / n, 2)
        taux_conf = round(sum(1 for s in self.scores_history if s['conformite']) / n * 100)
        n_exc     = sum(1 for s in self.scores_history if s['quality'] == 'Excellent')

        ll = {
            "fr": {
                "conf": f"Engagement confirmé — le {self._label.lower()} va tester le produit",
                "part": "Engagement partiel — intérêt manifesté, suivi nécessaire",
                "non": f"Visite non concluante — {self._label.lower()} non convaincu"
            },
            "en": {
                "conf": f"Confirmed engagement — the {self._label.lower()} will test the product",
                "part": "Partial engagement — interest shown, follow-up needed",
                "non": f"Inconclusive visit — {self._label.lower()} not convinced"
            },
            "es": {
                "conf": f"Compromiso confirmado — el {self._label.lower()} probará el producto",
                "part": "Compromiso parcial — interés mostrado, seguimiento necesario",
                "non": f"Visita no concluyente — {self._label.lower()} no convencido"
            },
            "ar": {
                "conf": "التزام مؤكد — سيختبر المنتج",
                "part": "التزام جزئي — أظهر اهتماماً، يحتاج لمتابعة",
                "non": "زيارة غير حاسمة — غير مقتنع"
            }
        }.get(self.language, {
            "conf": f"Engagement confirmé — le {self._label.lower()} va tester le produit",
            "part": "Engagement partiel — intérêt manifesté, suivi nécessaire",
            "non": f"Visite non concluante — {self._label.lower()} non convaincu"
        })

        if self.openness >= 4.2:
            resultat, ico, col = (ll["conf"], "🏆", "green")
        elif self.openness >= 3.0:
            resultat, ico, col = (ll["part"], "⚠️", "gold")
        else:
            resultat, ico, col = (ll["non"], "❌", "red")

        steps_done         = list(set(s for s in self.step_history if s))
        niveau_final       = self._infer_niveau(avg_score)
        global_score_final = self.state.global_score

        summary = {
            "produit"       : self.product['nom'],
            "interlocuteur" : self.interlocutor['nom'],
            "type"          : self._label,
            "difficulte"    : self.interlocutor['difficulte'],
            "tours"         : n,
            "score_nlp"     : avg_score,
            "score_global"  : global_score_final,
            "acrv_moyen"    : acrv_avg,
            "conformite"    : f"{taux_conf}%",
            "ouverture_fin" : round(self.openness, 1),
            "resultat"      : resultat,
            "lstm_score"    : self.state.lstm_posture_score,
            "vm_steps"      : sorted(list(self.state.vm_steps_done)),
            "rag_hits"      : len([h for h in self.state.rag_hits if h['context_used']]),
        }

        try:
            prompt = (
                f"Analyse cette visite médicale intégrée (NLP + LSTM + RAG) :\n"
                f"{json.dumps(summary, ensure_ascii=False)}\n\n"
                f'Retourne UNIQUEMENT :\n'
                f'{{ "points_forts": ["...", "..."], '
                f'"axes_travail": ["...", "..."], '
                f'"conseil_final": "..." }}'
            )
            raw      = _call_claude_report(
                messages=[{"role": "user", "content": prompt}],
                system=_build_report_system(lang=self.language),
            )
            analysis = json.loads(raw.replace('```json','').replace('```','').strip())
        except Exception:
            _fb = {
                "fr": {
                    "points_forts": ["Engagement dans la simulation"],
                    "axes_travail": ["Améliorer le closing", "Renforcer l'A-C-R-V"],
                    "conseil_final": f"Continuez à pratiquer sur ce profil {self._label.lower()}."
                },
                "en": {
                    "points_forts": ["Engagement in the simulation"],
                    "axes_travail": ["Improve closing", "Reinforce A-C-R-V"],
                    "conseil_final": f"Keep practicing on this {self._label.lower()} profile."
                },
                "es": {
                    "points_forts": ["Compromiso en la simulación"],
                    "axes_travail": ["Mejorar el cierre", "Reforzar el A-C-R-V"],
                    "conseil_final": f"Siga practicando en este perfil de {self._label.lower()}."
                },
                "ar": {
                    "points_forts": ["المشاركة في المحاكاة"],
                    "axes_travail": ["تحسين الإغلاق", "تعزيز A-C-R-V"],
                    "conseil_final": f"استمر في التدرب على هذا الملف {self._label.lower()}."
                }
            }
            analysis = _fb.get(self.language, _fb["fr"])

        return {
            "doctor"            : self.interlocutor,   # compat ancien code
            "interlocutor"      : self.interlocutor,
            "interlocutor_type" : self._label,
            "product"           : self.product,
            "turns"             : n,
            "avg_score"         : avg_score,
            "max_score"         : max_score,
            "global_score"      : global_score_final,
            "acrv_avg"          : acrv_avg,
            "taux_conforme"     : taux_conf,
            "pct_excellent"     : round(n_exc / n * 100),
            "ouverture_fin"     : round(self.openness, 1),
            "niveau_final"      : niveau_final,
            "resultat"          : resultat,
            "resultat_ico"      : ico,
            "resultat_col"      : col,
            "steps_done"        : steps_done,
            "scores_history"    : self.scores_history,
            "points_forts"      : analysis.get('points_forts', []),
            "axes_travail"      : analysis.get('axes_travail', []),
            "conseil_final"     : analysis.get('conseil_final', ''),
            "lstm_score"        : self.state.lstm_posture_score,
            "rag_hits"          : len([h for h in self.state.rag_hits if h['context_used']]),
            "vm_steps_done"     : sorted(list(self.state.vm_steps_done)),
            "state_snapshot"    : self.state.dashboard_data(),
            "final_decision"    : self.final_decision,
        }

    # ── Helpers ──────────────────────────────────────────────────

    def _evaluate_nlp(self, delegate_text: str) -> Dict:
        if self.nlp_model is None:
            return {
                'overall_score': 5.0, 'quality': 'Bon', 'acrv_score': 2,
                'acrv_detail': {}, 'conformite': True,
                'feedback_coaching': [], 'scores': {},
            }
        last_msg = next(
            (m["content"] for m in reversed(self.history) if m["role"] == "assistant"), ""
        )
        try:
            return self.nlp_model.predict(last_msg, delegate_text)
        except Exception:
            return {
                'overall_score': 5.0, 'quality': 'Bon', 'acrv_score': 2,
                'acrv_detail': {}, 'conformite': True,
                'feedback_coaching': [], 'scores': {},
            }

    def _delta_openness(self, score: float, conformite: bool, quality: str) -> float:
        d = 0.0
        if score >= 8.5:   d += 0.8
        elif score >= 7.0: d += 0.4
        elif score >= 5.5: d += 0.1
        elif score >= 4.0: d -= 0.2
        else:              d -= 0.5
        if not conformite:         d -= 0.8
        if quality == 'Excellent': d += 0.2
        if quality == 'Faible':    d -= 0.3
        return d

    def _detect_step(self, text: str) -> Optional[int]:
        tl = text.lower()
        for step in self._visit_steps:
            if any(kw in tl for kw in step['keywords']):
                return step['num']
        return None

    def _build_coaching(self, score, acrv, conformite, step, quality, feedback) -> str:
        lang = self.language or "fr"
        msgs = []
        if lang == "fr":
            if not conformite:
                msgs.append("🚨 MOT TUEUR — évitez les promesses !")
            elif score >= 8.0:
                msgs.append(f"✅ Excellente réponse ({score:.1f}/10)")
            elif score >= 6.5:
                msgs.append(f"👍 Bonne réponse ({score:.1f}/10)")
            else:
                msgs.append(f"⚠️ À améliorer ({score:.1f}/10)")
            if acrv == 4:
                msgs.append("A-C-R-V complet ✅")
            elif acrv >= 2:
                msgs.append(f"A-C-R-V : {acrv}/4")
            if step:
                msgs.append(f"Étape {step} détectée")
        elif lang == "en":
            if not conformite:
                msgs.append("🚨 FORBIDDEN WORD — avoid promises!")
            elif score >= 8.0:
                msgs.append(f"✅ Excellent response ({score:.1f}/10)")
            elif score >= 6.5:
                msgs.append(f"👍 Good response ({score:.1f}/10)")
            else:
                msgs.append(f"⚠️ Needs improvement ({score:.1f}/10)")
            if acrv == 4:
                msgs.append("A-C-R-V complete ✅")
            elif acrv >= 2:
                msgs.append(f"A-C-R-V: {acrv}/4")
            if step:
                msgs.append(f"Step {step} detected")
        elif lang == "es":
            if not conformite:
                msgs.append("🚨 PALABRA PROHIBIDA — ¡evite las promesas!")
            elif score >= 8.0:
                msgs.append(f"✅ Excelente respuesta ({score:.1f}/10)")
            elif score >= 6.5:
                msgs.append(f"👍 Buena respuesta ({score:.1f}/10)")
            else:
                msgs.append(f"⚠️ A mejorar ({score:.1f}/10)")
            if acrv == 4:
                msgs.append("A-C-R-V completo ✅")
            elif acrv >= 2:
                msgs.append(f"A-C-R-V: {acrv}/4")
            if step:
                msgs.append(f"Etapa {step} detectada")
        else:  # ar
            if not conformite:
                msgs.append("🚨 كلمة محظورة — تجنب الوعود!")
            elif score >= 8.0:
                msgs.append(f"✅ رد ممتاز ({score:.1f}/10)")
            elif score >= 6.5:
                msgs.append(f"👍 رد جيد ({score:.1f}/10)")
            else:
                msgs.append(f"⚠️ يحتاج تحسين ({score:.1f}/10)")
            if acrv == 4:
                msgs.append("A-C-R-V مكتمل ✅")
            elif acrv >= 2:
                msgs.append(f"A-C-R-V: {acrv}/4")
            if step:
                msgs.append(f"مرحلة {step} مكتشفة")
        if feedback:
            msgs.append(feedback[0])
        return " · ".join(msgs[:3])

    def _infer_niveau(self, avg: float) -> str:
        if avg >= 9.0: return "Expert"
        if avg >= 8.0: return "Confirmé"
        if avg >= 7.0: return "Junior"
        return "Débutant"

    def _static_fallback(self, closing: bool) -> str:
        import random
        lang = self.language or "fr"

        _CLOSE_PHARMA_POS = {
            "fr": "Bon, on va essayer avec une commande d'essai. Envoyez-moi votre bon de commande.",
            "en": "Alright, let's try with a trial order. Send me your order form.",
            "es": "Bien, vamos a probar con un pedido de prueba. Envíeme su formulario de pedido.",
            "ar": "حسناً، لنجرب بطلبية تجريبية. أرسل لي نموذج الطلب.",
        }
        _CLOSE_PHARMA_NEG = {
            "fr": "Je vais y réfléchir. Laissez-moi votre carte, je vous recontacterai.",
            "en": "I'll think about it. Leave me your card, I'll get back to you.",
            "es": "Lo pensaré. Déjeme su tarjeta, le contactaré.",
            "ar": "سأفكر في الأمر. اترك لي بطاقتك، سأتواصل معك.",
        }
        _CLOSE_DOC_POS = {
            "fr": "C'est intéressant. Laissez-moi votre documentation.",
            "en": "That's interesting. Leave me your documentation.",
            "es": "Es interesante. Déjeme su documentación.",
            "ar": "هذا مثير للاهتمام. اترك لي وثائقك.",
        }
        _CLOSE_DOC_NEG = {
            "fr": "Bien, merci pour votre passage. J'ai d'autres patients.",
            "en": "Well, thank you for coming. I have other patients.",
            "es": "Bien, gracias por su visita. Tengo otros pacientes.",
            "ar": "حسناً، شكراً لزيارتك. لدي مرضى آخرون.",
        }

        _OPT_PHARMA = {
            "fr": [
                "Et la marge sur ce produit, c'est combien exactement ?",
                "Vous avez des retours de ventes d'autres officines ?",
                "Le conditionnement permet combien de doses ou d'unités ?",
                "Vous proposez des PLV ou des présentoirs ?",
            ],
            "en": [
                "And the margin on this product, what is it exactly?",
                "Do you have sales feedback from other pharmacies?",
                "How many doses or units does the packaging allow?",
                "Do you offer POS displays or stands?",
            ],
            "es": [
                "Y el margen de este producto, ¿cuánto es exactamente?",
                "¿Tiene retornos de ventas de otras farmacias?",
                "¿Cuántas dosis o unidades permite el acondicionamiento?",
                "¿Ofrecen material de punto de venta?",
            ],
            "ar": [
                "وهامش الربح على هذا المنتج، كم تحديداً؟",
                "هل لديكم ملاحظات مبيعات من صيدليات أخرى؟",
                "كم جرعة أو وحدة يسمح بها التعبئة؟",
                "هل توفرون مواد عرض في نقطة البيع؟",
            ],
        }
        _OPT_DOC = {
            "fr": [
                "Qu'est-ce qui distingue vraiment votre produit ?",
                "Et sur le plan des données cliniques, qu'avez-vous ?",
                "Côté tolérance, des retours terrain ?",
                "Quel profil de patient ciblez-vous précisément ?",
                "Vous avez des études sur ce point ?",
            ],
            "en": [
                "What really distinguishes your product?",
                "What about clinical data, what do you have?",
                "In terms of tolerability, any field feedback?",
                "What patient profile are you targeting precisely?",
                "Do you have any studies on this?",
            ],
            "es": [
                "¿Qué distingue realmente su producto?",
                "En cuanto a datos clínicos, ¿qué tienen?",
                "En tolerancia, ¿algún retorno de campo?",
                "¿Qué perfil de paciente están enfocando precisamente?",
                "¿Tienen estudios sobre este punto?",
            ],
            "ar": [
                "ما الذي يميز منتجكم حقاً؟",
                "ماذا عن البيانات السريرية، ماذا لديكم؟",
                "من حيث التحمل، هل هناك ملاحظات ميدانية؟",
                "ما هو ملف المريض الذي تستهدفونه تحديداً؟",
                "هل لديكم دراسات حول هذه النقطة؟",
            ],
        }

        if self._is_pharmacist:
            if closing and self.openness >= 4:
                return _CLOSE_PHARMA_POS.get(lang, _CLOSE_PHARMA_POS["fr"])
            if closing:
                return _CLOSE_PHARMA_NEG.get(lang, _CLOSE_PHARMA_NEG["fr"])
            options = _OPT_PHARMA.get(lang, _OPT_PHARMA["fr"])
        else:
            if closing and self.openness >= 4:
                return _CLOSE_DOC_POS.get(lang, _CLOSE_DOC_POS["fr"])
            if closing:
                return _CLOSE_DOC_NEG.get(lang, _CLOSE_DOC_NEG["fr"])
            options = _OPT_DOC.get(lang, _OPT_DOC["fr"])
        return random.choice(options)

    # ── Sérialisation ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'doctor_id'          : self.interlocutor['id'],   # compat
            'interlocutor_id'    : self.interlocutor['id'],
            'product_id'         : self.product['id'],
            'niveau_alia'        : self.niveau_alia,
            'turn'               : self.turn,
            'openness'           : self.openness,
            'history'            : self.history,
            'scores_history'     : self.scores_history,
            'step_history'       : self.step_history,
            'is_finished'        : self.is_finished,
            'final_decision'     : self.final_decision,
            'suggestion_history' : self.suggestion_history,
            'language'           : self.language,
            'state'              : self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SimulationSession':
        # Compatibilité : ancien code utilise doctor_id
        iid = data.get('interlocutor_id') or data.get('doctor_id', 'sceptique')
        s = cls(iid, data['product_id'], data['niveau_alia'])
        s.turn           = data['turn']
        s.openness       = data['openness']
        s.history        = data['history']
        s.scores_history     = data['scores_history']
        s.step_history       = data['step_history']
        s.is_finished        = data['is_finished']
        s.final_decision     = data.get('final_decision', None)
        s.suggestion_history = data.get('suggestion_history', [])
        s.language           = data.get('language', 'fr')
        if 'state' in data:
            s.state = SessionState.from_dict(data['state'])
        return s
