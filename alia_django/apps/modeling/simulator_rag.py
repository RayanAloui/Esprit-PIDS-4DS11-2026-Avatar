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
# PROMPT PHARMACIEN ENRICHI PAR RAG
# ══════════════════════════════════════════════════════════════════════

def build_enriched_pharmacist_system(
    pharmacist: Dict,
    product: Dict,
    turn: int,
    openness: float,
    max_turns: int = 8,
) -> str:
    """
    Construit le prompt système du pharmacien enrichi par les données réelles
    de la base VITAL SA. Les données sont orientées officine :
    marge, rotation, conditionnement, PLV, DLC.
    """
    # Récupérer les données produit réelles depuis la KB
    kb_data = get_full_product_data(product["nom"])

    if kb_data:
        product_section = f"""
[DONNÉES RÉELLES DU PRODUIT — Base VITAL SA (vue officine)]
Nom : {kb_data['name']}
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
[PRODUIT PRÉSENTÉ]
Nom : {product['nom']} ({product['categorie']})
Indication : {product['indication']}
Argument clé : {product['argument_cle']}
"""

    return f"""Tu es {pharmacist['nom']}, {pharmacist['specialite']} à {pharmacist['ville']}.

PERSONNALITÉ :
{pharmacist['personnalite']}

CONTEXTE DE LA VISITE OFFICINE :
- Produit présenté : {product['nom']} ({product['categorie']})
- Tour actuel : {turn}/{max_turns}
- Ton niveau d'ouverture actuel : {openness:.1f}/5
{product_section}
CADRE DE LA VISITE OFFICINE (DIFFÉRENT D'UN MÉDECIN) :
- Tu penses en PHARMACIEN : marge brute, rotation de stock, DLC, conditionnement, PLV.
- Tu ne prescris pas — tu recommandes et tu vends au comptoir.
- Tes objections portent sur : le prix d'achat, la concurrence en rayon, le conditionnement,
  les conditions de retour, les remises de volume, la demande spontanée de tes clients.
- Tu n'es PAS intéressé par les mécanismes d'action cliniques (laisse ça au médecin).
- Tu veux savoir : est-ce que ça se vend bien ? Quelle marge ? Quel support terrain ?

RÈGLE ABSOLUE :
Réagis DIRECTEMENT au dernier message du délégué.
Ne répète jamais une formulation déjà utilisée.
Pose une seule objection commerciale à la fois.

PROGRESSION ({_openness_label(openness)}) :
- Tour 1-2 : accueil neutre + première objection commerciale.
- Tour 3-4 : creuser les conditions commerciales ou la demande marché.
- Tour 5-6 : position claire selon ton intérêt.
- Tour 7-8 : décision (commande d'essai ou refus).

STYLE : 2-3 phrases max. Ton direct, professionnel, commercial.
Réponds UNIQUEMENT avec tes paroles. Pas de guillemets, pas de préfixe.

OBJECTIONS TYPIQUES QUE TU UTILISES :
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
) -> str:
    """
    Construit le prompt système pour le mode généraliste.
    Contrairement au mode classique (un produit précis), le mode généraliste
    couvre toute la gamme VITAL SA, avec des questions plus ouvertes
    et un focus sur la capacité d'adaptation du délégué.
    """
    # Récupérer un contexte large depuis la KB (multi-produits)
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
                    "\n[GAMME VITAL SA — Données KB]\n"
                    + "\n".join(parts)
                    + "\n\nUtilise ces informations pour poser des questions variées "
                    "sur différents produits de la gamme.\n"
                )
        except Exception as e:
            log.warning(f"[SimRAG] Gamme retrieval error : {e}")

    is_pharm = interlocutor.get("type") == "pharmacist"
    role_context = (
        "Tu es un pharmacien qui reçoit un délégué pour une visite commerciale."
        if is_pharm else
        "Tu es un médecin qui reçoit un délégué pour une visite médicale."
    )

    return f"""Tu es {interlocutor['nom']}, {interlocutor['specialite']} à {interlocutor['ville']}.

PERSONNALITÉ :
{interlocutor['personnalite']}

MODE : ENTRAÎNEMENT GÉNÉRALISTE
{role_context}
Le délégué s'entraîne SANS produit spécifique. Il doit démontrer sa maîtrise
de la méthode de visite VITAL SA et sa capacité d'adaptation.

- Tour actuel : {turn}/{max_turns}
- Ton niveau d'ouverture actuel : {openness:.1f}/5
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
    label_type = "pharmacien" if is_pharm else "médecin"

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
        label = (
            f"Commande d'essai passée — {qty} unités" if is_pharm
            else f"Commande passée — {qty} unités"
        )
        return {
            "decision"     : "commande",
            "label"        : label,
            "reason"       : f"Score {effective_score:.1f}/10 + ouverture {openness:.1f}/5",
            "commande_qty" : qty,
            "color"        : "green",
            "icon"         : "🏆",
        }

    elif effective_score >= SEUIL_CONDITIONS and openness >= 3.0:
        label = (
            "Intéressé — demande des conditions commerciales" if is_pharm
            else "Intéressé sous conditions"
        )
        return {
            "decision"     : "conditions",
            "label"        : label,
            "reason"       : f"Score {effective_score:.1f}/10 — suivi nécessaire",
            "commande_qty" : 0,
            "color"        : "gold",
            "icon"         : "⚠️",
        }

    elif effective_score < SEUIL_REFUS_DUR or openness < 1.5:
        label = (
            f"Refus — {label_type} non convaincu"
        )
        return {
            "decision"     : "refus",
            "label"        : label,
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
    # Réduction si profil difficile
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

    system_prompt = (
        f"Tu es {doctor['nom']}, {doctor['specialite']}. "
        f"{doctor['personnalite']} "
        f"Réponds UNIQUEMENT avec tes paroles. Pas de guillemets ni de préfixe."
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
                "stop": ["\nDélégué", f"\n{label}", "Délégué :"],
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
            response = response.split(f"\n{label}")[0].strip()
            if response:
                return response
    except Exception as e:
        log.warning(f"[SimRAG] Ollama closing fallback : {e}")

    # Fallback statique selon décision + type
    if is_pharm:
        fallbacks = {
            "commande"   : f"Bon, on va essayer. Envoyez-moi {decision.get('commande_qty', 10)} unités de {product['nom']} avec les conditions qu'on a discutées.",
            "conditions" : f"C'est intéressant mais je voudrais de meilleures conditions. Revenez avec une offre réseau et on en reparle.",
            "refus"      : "Merci mais j'ai déjà un stock suffisant de produits sur ce segment. Peut-être la prochaine fois.",
        }
    else:
        fallbacks = {
            "commande"   : f"Bien. Je vais essayer {product['nom']} sur quelques-uns de mes patients. Passez me voir dans un mois avec les retours terrain.",
            "conditions" : f"C'est intéressant. Envoyez-moi les données cliniques complètes et nous en reparlerons.",
            "refus"      : "Merci pour votre présentation. Pour l'instant je reste sur mes prescriptions habituelles.",
        }
    return fallbacks.get(decision_type, "Merci pour votre visite.")
