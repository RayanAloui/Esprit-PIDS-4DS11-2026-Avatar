"""
engine.py
=========
Moteur de simulation — Version RAG + Ollama.

Architecture LLM :
    Ollama (llama3.2) joue le médecin virtuel.
    RAG (ChromaDB + BM25) enrichit chaque réponse avec les vraies
    données produit VITAL SA extraites de vital_products.csv.
    Claude API reste utilisé UNIQUEMENT pour le rapport final.
    Si Ollama est indisponible → fallback Claude pour le médecin aussi.

Flux par tour :
    1. NLPScoringModel évalue la réponse du délégué (T1→T6)
    2. RAG récupère le contexte produit pertinent (ChromaDB + BM25)
    3. Ollama génère la réaction du médecin (profil + historique + contexte RAG)
    4. Variation ouverture médecin selon score NLP
    5. Coaching temps réel retourné au frontend
"""

import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing  import Dict, List, Optional

from .profiles import get_product, get_doctor, VISIT_STEPS

log = logging.getLogger(__name__)

MAX_TURNS      = 8
MAX_TOKENS_RPT = 1200

# ══════════════════════════════════════════════════════════════════════
# NLP MODEL — singleton
# ══════════════════════════════════════════════════════════════════════

def _get_nlp_model():
    try:
        from django.conf import settings
        models_dir = str(settings.MODELS_AI_DIR)
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        import nlp_scoring_train_v2  # noqa — requis pour joblib
        from nlp_scoring_model_v2 import NLPScoringModel
        bundle = Path(settings.MODELS_AI_DIR) / 'nlp_scoring_bundle_v2.pkl'
        return NLPScoringModel.load(str(bundle))
    except Exception as e:
        log.warning(f"NLPScoringModel non disponible : {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# RAG — contexte produit depuis la KB VITAL SA
# ══════════════════════════════════════════════════════════════════════

def _get_rag_context(product_name: str, delegate_text: str) -> str:
    """
    Interroge la base RAG (ChromaDB + BM25) via le runtime ALIA.
    Retourne un contexte produit concis à injecter dans le prompt Ollama.

    On passe le nom du produit + le texte du délégué comme query
    pour récupérer les chunks les plus pertinents de vital_products.csv.
    """
    try:
        from apps.modeling.runtime import get_runtime
        rt    = get_runtime()
        query = f"{product_name} {delegate_text}"
        docs  = rt.kb.retriever.invoke(query)

        if not docs:
            return ""

        # Garder les 3 meilleurs documents, concis
        parts = []
        for doc in docs[:3]:
            name    = doc.metadata.get('name', '')
            # Tronquer à 300 chars pour ne pas surcharger le prompt
            preview = doc.page_content[:300].replace('\n', ' ').strip()
            parts.append(f"• {name} : {preview}")

        return "\n".join(parts)

    except Exception as e:
        log.warning(f"RAG context indisponible : {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════
# OLLAMA — LLM local pour le médecin virtuel
# ══════════════════════════════════════════════════════════════════════

def _call_ollama(system: str, history: List[Dict],
                 rag_context: str) -> str:
    """
    Appelle Ollama (llama3.2) pour générer la réponse du médecin.

    On reconstruit un prompt texte complet car Ollama /api/generate
    est plus simple que /api/chat pour injecter le contexte RAG.
    """
    # Reconstruire l'historique en texte
    history_text = ""
    for msg in history:
        role = "Délégué" if msg["role"] == "user" else "Médecin"
        history_text += f"\n{role} : {msg['content']}"

    # Section contexte RAG — injectée seulement si disponible
    rag_section = ""
    if rag_context:
        rag_section = (
            f"\n\n[CONTEXTE PRODUIT — base de connaissances VITAL SA]\n"
            f"{rag_context}\n"
            f"Utilise ces informations si le délégué en parle. "
            f"Ne les récite pas mot à mot — réagis naturellement.\n"
        )

    full_prompt = (
        f"{system}"
        f"{rag_section}"
        f"\n\n[CONVERSATION]\n"
        f"{history_text}"
        f"\nMédecin :"
    )

    payload = json.dumps({
        "model" : "llama3.2:latest",
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.75,
            "num_predict": 180,
            "stop": ["\nDélégué", "\nMédecin", "Délégué :"],
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data    = payload,
        headers = {'Content-Type': 'application/json'},
        method  = 'POST',
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data     = json.loads(resp.read())
        response = data.get('response', '').strip()
        # Nettoyer les artefacts éventuels
        response = response.split('\nDélégué')[0].strip()
        response = response.split('\nMédecin')[0].strip()
        return response


def _call_claude_doctor(system: str, history: List[Dict]) -> str:
    """Fallback Claude pour le médecin si Ollama est indisponible."""
    payload = json.dumps({
        "model"     : "claude-sonnet-4-20250514",
        "max_tokens": 400,
        "system"    : system,
        "messages"  : history,
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data    = payload,
        headers = {'Content-Type': 'application/json'},
        method  = 'POST',
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
        return data['content'][0]['text'].strip()


def _call_claude_report(messages: List[Dict], system: str) -> str:
    """Claude pour le rapport final uniquement."""
    payload = json.dumps({
        "model"     : "claude-sonnet-4-20250514",
        "max_tokens": MAX_TOKENS_RPT,
        "system"    : system,
        "messages"  : messages,
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data    = payload,
        headers = {'Content-Type': 'application/json'},
        method  = 'POST',
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return data['content'][0]['text'].strip()


def _generate_doctor_response(system: str, history: List[Dict],
                               rag_context: str,
                               fallback_fn) -> tuple:
    """
    Tente Ollama en premier.
    Si Ollama échoue → fallback Claude.
    Retourne (message, engine_used).
    """
    try:
        msg = _call_ollama(system, history, rag_context)
        if not msg:
            raise ValueError("Réponse Ollama vide")
        return msg, "ollama+rag"
    except Exception as e_ollama:
        log.info(f"Ollama indisponible ({e_ollama}) → fallback Claude")
        try:
            msg = _call_claude_doctor(system, history)
            return msg, "claude_fallback"
        except Exception:
            return fallback_fn(), "static_fallback"


# ══════════════════════════════════════════════════════════════════════
# PROMPT SYSTÈME DU MÉDECIN
# ══════════════════════════════════════════════════════════════════════

def _build_doctor_system(doctor: dict, product: dict,
                          turn: int, openness: float) -> str:
    return f"""Tu es {doctor['nom']}, {doctor['specialite']} à {doctor['ville']}.

PERSONNALITÉ :
{doctor['personnalite']}

CONTEXTE DE LA VISITE :
- Produit présenté : {product['nom']} ({product['categorie']})
- Tour actuel : {turn}/{MAX_TURNS}
- Ton niveau d'ouverture : {openness:.1f}/5

RÈGLE ABSOLUE N°1 — RÉAGIS AU DERNIER MESSAGE PRÉCISÉMENT :
Lis ce que vient de dire le délégué et réponds-y DIRECTEMENT.
Si le délégué mentionne une donnée (ex: "1200 patients") → réagis à cette donnée.
Si le délégué pose une question → réponds à CETTE question précise.
INTERDIT : répéter une phrase que tu as déjà dite dans cette conversation.

RÈGLE ABSOLUE N°2 — PROGRESSION NATURELLE :
Tour 1-2 : accueil + première question ou objection initiale.
Tour 3-4 : approfondissement ou objection plus précise.
Tour 5-6 : position claire selon ton ouverture.
Tour 7-8 : conclusion de la visite.

TON NIVEAU D'OUVERTURE {openness:.1f}/5 :
{_openness_behavior(openness)}

STYLE : 2-3 phrases maximum. Ton : {doctor['description_ui']}.
Varie tes formules — ne commence jamais par la même phrase deux fois.

OBJECTIONS QUE TU UTILISES TYPIQUEMENT :
{chr(10).join(f'- {o}' for o in doctor['objections_favorites'])}

Réponds UNIQUEMENT avec tes paroles. Pas de guillemets, pas de préfixe.
"""


def _build_report_system() -> str:
    return (
        "Tu es ALIA, le système d'évaluation IA de VITAL SA. "
        "Génère un rapport de visite concis et actionnable. "
        "Réponds UNIQUEMENT en JSON valide, sans markdown ni explication."
    )


def _openness_behavior(openness: float) -> str:
    if openness >= 4.2:
        return "Tu es clairement intéressé. Pose des questions positives sur la mise en pratique."
    elif openness >= 3.5:
        return "Tu commences à t'ouvrir. Demande des précisions sans t'engager encore."
    elif openness >= 2.5:
        return "Tu es neutre. Écoute mais reste prudent. Pose une objection modérée."
    elif openness >= 1.5:
        return "Tu es sceptique. Exprime un doute direct sur ce que vient de dire le délégué."
    else:
        return "Tu es très fermé. Montre de l'impatience. La visite touche à sa fin."


# ══════════════════════════════════════════════════════════════════════
# SIMULATION SESSION
# ══════════════════════════════════════════════════════════════════════

class SimulationSession:
    """
    Gère l'état complet d'une session de simulation.

    Médecin virtuel : Ollama + RAG (fallback Claude si Ollama down).
    Évaluation      : NLPScoringModel V2 à chaque tour.
    Rapport final   : Claude API.
    """

    def __init__(self, doctor_id: str, product_id: str, niveau_alia: str):
        self.doctor         = get_doctor(doctor_id)
        self.product        = get_product(product_id)
        self.niveau_alia    = niveau_alia
        self.turn           = 0
        self.openness       = float(self.doctor['ouverture_initiale'])
        self.history        = []
        self.scores_history = []
        self.step_history   = []
        self.is_finished    = False
        self.nlp_model      = _get_nlp_model()

    # ── Premier message du médecin ─────────────────────────────────────

    def first_message(self) -> Dict:
        system = _build_doctor_system(
            self.doctor, self.product, 0, self.openness)

        # RAG : récupérer infos produit pour enrichir l'accueil
        rag_ctx = _get_rag_context(
            self.product['nom'],
            f"présentation {self.product['categorie']} {self.product['indication']}"
        )

        opening_msg = {
            "role"   : "user",
            "content": (
                f"Un délégué médical de VITAL SA entre dans ton cabinet "
                f"pour te présenter {self.product['nom']} ({self.product['categorie']}). "
                f"Accueille-le en 1-2 phrases selon ta personnalité."
            )
        }

        msg, engine = _generate_doctor_response(
            system      = system,
            history     = [opening_msg],
            rag_context = rag_ctx,
            fallback_fn = lambda: "Oui, entrez. Vous avez quelques minutes, je vous écoute.",
        )

        self.history.append({"role": "assistant", "content": msg})

        log.info(f"[Simulator] First message — engine={engine}, rag={'oui' if rag_ctx else 'non'}")

        return {
            "message"  : msg,
            "turn"     : self.turn,
            "openness" : round(self.openness, 1),
            "step"     : None,
            "score"    : None,
            "coach"    : "La visite commence — gérez bien la permission (Étape 1).",
            "is_first" : True,
            "engine"   : engine,
            "rag_used" : bool(rag_ctx),
        }

    # ── Tour de conversation ───────────────────────────────────────────

    def process_delegate_response(self, delegate_text: str) -> Dict:
        self.turn += 1

        # ── 1. Évaluation NLP (T1→T6) ─────────────────────────────────
        nlp        = self._evaluate_nlp(delegate_text)
        score      = nlp.get('overall_score', 5.0)
        acrv       = nlp.get('acrv_score', 0)
        acrv_det   = nlp.get('acrv_detail', {})
        conformite = nlp.get('conformite', True)
        quality    = nlp.get('quality', 'Bon')
        scores_det = nlp.get('scores', {})
        feedback   = nlp.get('feedback_coaching', [])

        self.scores_history.append({
            'turn': self.turn, 'score': score,
            'quality': quality, 'acrv': acrv, 'conformite': conformite,
        })

        # ── 2. Variation ouverture du médecin ──────────────────────────
        delta        = self._delta_openness(score, conformite, quality)
        self.openness = max(1.0, min(5.0, self.openness + delta))

        # ── 3. Détection étape VM ──────────────────────────────────────
        step = self._detect_step(delegate_text)
        self.step_history.append(step)

        # ── 4. Coaching temps réel ─────────────────────────────────────
        coach = self._build_coaching(score, acrv, conformite, step, quality, feedback)

        # ── 5. Fin de visite ? ─────────────────────────────────────────
        should_close = (
            self.turn >= MAX_TURNS
            or (self.openness >= 4.2 and self.turn >= 4)
            or (self.openness <= 1.2 and self.turn >= 3)
        )

        # ── 6. Ajouter message délégué à l'historique ─────────────────
        self.history.append({"role": "user", "content": delegate_text})

        # ── 7. RAG — contexte produit lié au message du délégué ────────
        rag_ctx = _get_rag_context(self.product['nom'], delegate_text)

        # ── 8. Prompt système médecin ──────────────────────────────────
        system = _build_doctor_system(
            self.doctor, self.product, self.turn, self.openness)

        if should_close:
            if self.openness >= 4.0:
                system += (
                    "\n\nFIN DE VISITE. Tu es intéressé. "
                    "Conclus positivement en 2 phrases : "
                    "intérêt pour tester ou demande de documentation."
                )
            else:
                system += (
                    "\n\nFIN DE VISITE. "
                    "Conclus poliment en 1-2 phrases. "
                    "Tu as d'autres patients qui attendent."
                )

        # ── 9. Générer réponse médecin (Ollama + RAG → fallback Claude) ─
        msg, engine = _generate_doctor_response(
            system      = system,
            history     = self.history,
            rag_context = rag_ctx,
            fallback_fn = lambda: self._static_fallback(should_close),
        )

        self.history.append({"role": "assistant", "content": msg})

        if should_close:
            self.is_finished = True

        log.info(
            f"[Simulator] Tour {self.turn} — score={score:.2f} "
            f"openness={self.openness:.1f} engine={engine} "
            f"rag={'oui' if rag_ctx else 'non'}"
        )

        return {
            "message"    : msg,
            "turn"       : self.turn,
            "openness"   : round(self.openness, 1),
            "step"       : step,
            "score"      : round(score, 2),
            "quality"    : quality,
            "acrv"       : acrv,
            "acrv_detail": acrv_det,
            "conformite" : conformite,
            "scores"     : scores_det,
            "coach"      : coach,
            "is_finished": self.is_finished,
            "delta_open" : round(delta, 2),
            "engine"     : engine,
            "rag_used"   : bool(rag_ctx),
        }

    # ── Rapport final ──────────────────────────────────────────────────

    def generate_report(self) -> Dict:
        n = len(self.scores_history)
        if n == 0:
            return {"error": "Aucune donnée de visite."}

        scores    = [s['score'] for s in self.scores_history]
        avg_score = round(sum(scores) / n, 2)
        max_score = round(max(scores), 2)
        acrv_avg  = round(sum(s['acrv']  for s in self.scores_history) / n, 2)
        taux_conf = round(sum(1 for s in self.scores_history if s['conformite']) / n * 100)
        n_exc     = sum(1 for s in self.scores_history if s['quality'] == 'Excellent')

        if self.openness >= 4.2:
            resultat, ico, col = "Engagement confirmé — le médecin va tester le produit", "🏆", "green"
        elif self.openness >= 3.0:
            resultat, ico, col = "Engagement partiel — intérêt manifesté, suivi nécessaire", "⚠️", "gold"
        else:
            resultat, ico, col = "Visite non concluante — médecin non convaincu", "❌", "red"

        steps_done   = list(set(s for s in self.step_history if s))
        niveau_final = self._infer_niveau(avg_score)

        summary = {
            "produit"      : self.product['nom'],
            "medecin"      : self.doctor['nom'],
            "difficulte"   : self.doctor['difficulte'],
            "tours"        : n,
            "score_moyen"  : avg_score,
            "score_max"    : max_score,
            "acrv_moyen"   : acrv_avg,
            "conformite"   : f"{taux_conf}%",
            "ouverture_fin": round(self.openness, 1),
            "resultat"     : resultat,
        }

        # Rapport qualitatif via Claude API
        try:
            prompt = (
                f"Analyse cette visite médicale et génère un rapport.\n"
                f"Données : {json.dumps(summary, ensure_ascii=False)}\n\n"
                f"Retourne UNIQUEMENT ce JSON :\n"
                f'{{ "points_forts": ["...", "..."], '
                f'"axes_travail": ["...", "..."], '
                f'"conseil_final": "..." }}'
            )
            raw = _call_claude_report(
                messages = [{"role": "user", "content": prompt}],
                system   = _build_report_system(),
            )
            analysis = json.loads(raw.replace('```json','').replace('```','').strip())
        except Exception:
            analysis = {
                "points_forts" : ["Engagement dans la simulation"],
                "axes_travail" : ["Améliorer le closing BIP", "Renforcer l'A-C-R-V"],
                "conseil_final": "Continuez à pratiquer la méthode A-C-R-V sur ce profil médecin.",
            }

        return {
            "doctor"        : self.doctor,
            "product"       : self.product,
            "turns"         : n,
            "avg_score"     : avg_score,
            "max_score"     : max_score,
            "acrv_avg"      : acrv_avg,
            "taux_conforme" : taux_conf,
            "pct_excellent" : round(n_exc / n * 100),
            "ouverture_fin" : round(self.openness, 1),
            "niveau_final"  : niveau_final,
            "resultat"      : resultat,
            "resultat_ico"  : ico,
            "resultat_col"  : col,
            "steps_done"    : steps_done,
            "scores_history": self.scores_history,
            "points_forts"  : analysis.get('points_forts', []),
            "axes_travail"  : analysis.get('axes_travail', []),
            "conseil_final" : analysis.get('conseil_final', ''),
        }

    # ── Helpers privés ─────────────────────────────────────────────────

    def _evaluate_nlp(self, delegate_text: str) -> Dict:
        if self.nlp_model is None:
            return {
                'overall_score': 5.0, 'quality': 'Bon',
                'acrv_score': 2, 'acrv_detail': {},
                'conformite': True, 'feedback_coaching': [], 'scores': {},
            }
        last_doctor = next(
            (m["content"] for m in reversed(self.history)
             if m["role"] == "assistant"), ""
        )
        try:
            return self.nlp_model.predict(last_doctor, delegate_text)
        except Exception:
            return {
                'overall_score': 5.0, 'quality': 'Bon',
                'acrv_score': 2, 'acrv_detail': {},
                'conformite': True, 'feedback_coaching': [], 'scores': {},
            }

    def _delta_openness(self, score: float,
                        conformite: bool, quality: str) -> float:
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
        for step in VISIT_STEPS:
            if any(kw in tl for kw in step['keywords']):
                return step['num']
        return None

    def _build_coaching(self, score, acrv, conformite,
                         step, quality, feedback) -> str:
        msgs = []
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
        if closing and self.openness >= 4:
            return "C'est intéressant. Laissez-moi votre documentation, je vais y réfléchir."
        if closing:
            return "Bien, merci pour votre passage. J'ai d'autres patients."
        options = [
            "Qu'est-ce qui distingue vraiment votre produit de ce que j'utilise ?",
            "Et sur le plan des données cliniques, qu'avez-vous exactement ?",
            "Hmm. Et côté tolérance, vous avez des retours terrain ?",
            "Intéressant. Quel profil de patient ciblez-vous précisément ?",
            "Je vois. Vous avez des études sur ce point spécifique ?",
        ]
        return random.choice(options)

    # ── Sérialisation session Django ───────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'doctor_id'     : self.doctor['id'],
            'product_id'    : self.product['id'],
            'niveau_alia'   : self.niveau_alia,
            'turn'          : self.turn,
            'openness'      : self.openness,
            'history'       : self.history,
            'scores_history': self.scores_history,
            'step_history'  : self.step_history,
            'is_finished'   : self.is_finished,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SimulationSession':
        s = cls(data['doctor_id'], data['product_id'], data['niveau_alia'])
        s.turn          = data['turn']
        s.openness      = data['openness']
        s.history       = data['history']
        s.scores_history= data['scores_history']
        s.step_history  = data['step_history']
        s.is_finished   = data['is_finished']
        return s
