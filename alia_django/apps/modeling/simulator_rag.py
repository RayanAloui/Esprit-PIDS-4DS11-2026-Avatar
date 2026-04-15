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
# PROMPT SYSTÈME ENRICHI PAR RAG
# ══════════════════════════════════════════════════════════════════════

def build_enriched_doctor_system(
    doctor: Dict,
    product: Dict,
    turn: int,
    openness: float,
    max_turns: int = 8,
) -> str:
    """
    Construit le prompt système du médecin enrichi par les données réelles
    de la base VITAL SA. Remplace _build_doctor_system() dans engine.py.
    """
    # Récupérer les données produit réelles depuis la KB
    kb_data = get_full_product_data(product["nom"])

    # Section données produit enrichies
    if kb_data:
        product_section = f"""
[DONNÉES RÉELLES DU PRODUIT — Base VITAL SA]
Nom : {kb_data['name']}
{kb_data['content']}
Forme : {kb_data['forme']} | Usage : {kb_data['usage']} | Cible : {kb_data['target']}

Tu connais ces données. Tes objections et questions doivent être cohérentes avec elles.
"""
    else:
        product_section = f"""
[PRODUIT PRÉSENTÉ]
Nom : {product['nom']} ({product['categorie']})
Indication : {product['indication']}
Argument clé : {product['argument_cle']}
"""

    return f"""Tu es {doctor['nom']}, {doctor['specialite']} à {doctor['ville']}.

PERSONNALITÉ :
{doctor['personnalite']}

CONTEXTE DE LA VISITE :
- Produit présenté : {product['nom']} ({product['categorie']})
- Tour actuel : {turn}/{max_turns}
- Ton niveau d'ouverture actuel : {openness:.1f}/5
{product_section}
RÈGLE ABSOLUE :
Réagis DIRECTEMENT au dernier message du délégué.
Ne répète jamais une formulation déjà utilisée.
Pose une seule objection ou question à la fois.

PROGRESSION ({_openness_label(openness)}) :
- Tour 1-2 : accueil réservé + première objection selon ta personnalité
- Tour 3-4 : objection précise sur le produit (données, effets secondaires, prix, concurrence)
- Tour 5-6 : position claire selon ton ouverture
- Tour 7-8 : conclusion (signal d'achat ou refus)

STYLE : 2-3 phrases max. Ton : {doctor['description_ui']}.
Réponds UNIQUEMENT avec tes paroles. Pas de guillemets, pas de préfixe.

OBJECTIONS TYPIQUES QUE TU UTILISES :
{chr(10).join(f'- {o}' for o in doctor['objections_favorites'])}
"""


def _openness_label(openness: float) -> str:
    if openness >= 4.2:
        return "Très ouvert — questions positives sur la mise en pratique"
    elif openness >= 3.5:
        return "Ouvert — demande des précisions sans s'engager"
    elif openness >= 2.5:
        return "Neutre — objection modérée, en attente d'arguments"
    elif openness >= 1.5:
        return "Sceptique — doute direct sur le dernier argument"
    else:
        return "Fermé — impatience, fin de visite imminente"


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
) -> Dict:
    """
    Calcule la décision finale du médecin en combinant :
    - Le score global (NLP 40% + LSTM 25% + conformité 20% + étapes VM 15%)
    - L'ouverture finale du médecin
    - Les étapes VM complétées
    - La difficulté du profil médecin

    Retourne :
    {
        "decision"     : "commande" | "conditions" | "refus",
        "label"        : str (texte affiché dans l'UI),
        "reason"       : str (explication interne),
        "commande_qty" : int (0 si refus),
        "color"        : str (css color token),
        "icon"         : str (emoji),
    }
    """
    # Malus selon difficulté médecin (profil très difficile = seuils plus exigeants)
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
        f"[SimRAG] Décision | global={global_score:.2f} | diff_malus={diff_malus} "
        f"| vm_bonus={vm_bonus:.2f} | effective={effective_score:.2f} "
        f"| openness={openness:.1f}"
    )

    # ── Arbre de décision ──────────────────────────────────────────
    if effective_score >= SEUIL_COMMANDE and openness >= 4.0:
        qty = _estimate_order_qty(effective_score, doctor)
        return {
            "decision"     : "commande",
            "label"        : f"Commande passée — {qty} unités",
            "reason"       : f"Score {effective_score:.1f}/10 + ouverture {openness:.1f}/5",
            "commande_qty" : qty,
            "color"        : "green",
            "icon"         : "🏆",
        }

    elif effective_score >= SEUIL_CONDITIONS and openness >= 3.0:
        return {
            "decision"     : "conditions",
            "label"        : "Intéressé sous conditions",
            "reason"       : f"Score {effective_score:.1f}/10 — suivi nécessaire",
            "commande_qty" : 0,
            "color"        : "gold",
            "icon"         : "⚠️",
        }

    elif effective_score < SEUIL_REFUS_DUR or openness < 1.5:
        return {
            "decision"     : "refus",
            "label"        : "Refus — médecin non convaincu",
            "reason"       : f"Score {effective_score:.1f}/10 — arguments insuffisants",
            "commande_qty" : 0,
            "color"        : "red",
            "icon"         : "❌",
        }

    else:
        return {
            "decision"     : "conditions",
            "label"        : "Engagement partiel — relance dans 2 semaines",
            "reason"       : f"Score {effective_score:.1f}/10 + ouverture {openness:.1f}/5",
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
    # Réduction si médecin difficile
    diff = doctor.get("difficulte_n", 2)
    factor = {1: 1.5, 2: 1.0, 3: 0.7, 4: 0.5}.get(diff, 1.0)
    return max(5, int(base * factor))


# ══════════════════════════════════════════════════════════════════════
# GÉNÉRATION TEXTE DÉCISION FINALE VIA OLLAMA
# ══════════════════════════════════════════════════════════════════════

def generate_closing_message(
    doctor: Dict,
    product: Dict,
    decision: Dict,
    global_score: float,
    openness: float,
    conversation_summary: str,
) -> str:
    """
    Génère le message de clôture du médecin via Ollama,
    cohérent avec la décision calculée et la personnalité du médecin.
    Fallback statique si Ollama indisponible.
    """
    decision_type = decision["decision"]

    if decision_type == "commande":
        instruction = (
            f"Tu viens de décider de passer une commande de {decision['commande_qty']} unités "
            f"de {product['nom']}. Conclus positivement en 2-3 phrases. "
            "Mentionne que tu vas tester sur quelques patients et que tu rappelleras le délégué."
        )
    elif decision_type == "conditions":
        instruction = (
            f"Tu es intéressé par {product['nom']} mais pas encore prêt à commander. "
            "Conclus en 2 phrases en précisant ce qu'il te faudrait encore "
            "(données cliniques, essai gratuit, ou information complémentaire)."
        )
    else:  # refus
        instruction = (
            f"Tu as décidé de ne pas commander {product['nom']} pour le moment. "
            "Conclus poliment mais fermement en 2 phrases. "
            "Reste dans ta personnalité de médecin exigeant."
        )

    system_prompt = (
        f"Tu es {doctor['nom']}, {doctor['specialite']}. "
        f"{doctor['personnalite']} "
        f"Réponds UNIQUEMENT avec tes paroles. Pas de guillemets ni de préfixe."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"[FIN DE VISITE]\n{instruction}\n\n"
        f"Médecin :"
    )

    try:
        payload = json.dumps({
            "model" : "llama3.2:latest",
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 120,
                "stop": ["\nDélégué", "\nMédecin", "Délégué :"],
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
            response = response.split("\nDélégué")[0].strip()
            response = response.split("\nMédecin")[0].strip()
            if response:
                return response
    except Exception as e:
        log.warning(f"[SimRAG] Ollama closing fallback : {e}")

    # Fallback statique selon décision
    fallbacks = {
        "commande"   : f"Bien. Je vais essayer {product['nom']} sur quelques-uns de mes patients. Passez me voir dans un mois avec les retours terrain.",
        "conditions" : f"C'est intéressant. Envoyez-moi les données cliniques complètes et nous en reparlerons.",
        "refus"      : "Merci pour votre présentation. Pour l'instant je reste sur mes prescriptions habituelles.",
    }
    return fallbacks.get(decision_type, "Merci pour votre visite.")
