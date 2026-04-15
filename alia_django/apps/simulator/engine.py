"""
engine.py
=========
Moteur de simulation intégré — NLP + LSTM + RAG + SessionState.

Tous les modules poussent leurs données dans SessionState à chaque tour.
Fix répétition Ollama : historique des 4 dernières réponses médecin injecté
dans le prompt comme liste "INTERDIT de répéter".
"""

import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing  import Dict, List, Optional

from .profiles       import get_product, get_doctor, VISIT_STEPS
from .session_state  import SessionState
from apps.modeling.simulator_rag import (
    get_product_context,
    build_enriched_doctor_system,
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

def _call_ollama(system: str, history: List[Dict],
                 rag_context: str, recent_doctor_msgs: List[str]) -> str:
    """
    Appelle Ollama (llama3.2).
    Fix répétition : les 4 dernières réponses médecin sont injectées
    dans le prompt comme liste INTERDITE.
    """
    # Historique conversationnel
    history_text = ""
    for msg in history:
        role = "Délégué" if msg["role"] == "user" else "Médecin"
        history_text += f"\n{role} : {msg['content']}"

    # Section RAG
    rag_section = ""
    if rag_context:
        rag_section = (
            f"\n\n[DONNÉES PRODUIT — Base VITAL SA]\n{rag_context}\n"
            f"Utilise ces données si pertinent. Ne les récite pas.\n"
        )

    # FIX RÉPÉTITION — liste des phrases interdites
    forbidden_section = ""
    if recent_doctor_msgs:
        forbidden_list = "\n".join(f"  - \"{m[:80]}\"" for m in recent_doctor_msgs)
        forbidden_section = (
            f"\n\n[PHRASES QUE TU AS DÉJÀ DITES — INTERDITES]\n"
            f"{forbidden_list}\n"
            f"Ne commence PAS ta réponse par l'une de ces phrases. Varie obligatoirement.\n"
        )

    full_prompt = (
        f"{system}"
        f"{rag_section}"
        f"{forbidden_section}"
        f"\n\n[CONVERSATION]\n{history_text}\nMédecin :"
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
        data=payload, headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data     = json.loads(resp.read())
        response = data.get('response', '').strip()
        response = response.split('\nDélégué')[0].strip()
        response = response.split('\nMédecin')[0].strip()
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


def _generate_doctor_response(system: str, history: List[Dict],
                               rag_context: str,
                               recent_doctor_msgs: List[str],
                               fallback_fn) -> tuple:
    try:
        msg = _call_ollama(system, history, rag_context, recent_doctor_msgs)
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
# PROMPT SYSTÈME
# ══════════════════════════════════════════════════════════════════════

def _build_doctor_system(doctor: dict, product: dict,
                          turn: int, openness: float) -> str:
    return f"""Tu es {doctor['nom']}, {doctor['specialite']} à {doctor['ville']}.

PERSONNALITÉ :
{doctor['personnalite']}

CONTEXTE :
- Produit présenté : {product['nom']} ({product['categorie']})
- Tour actuel : {turn}/{MAX_TURNS}
- Ouverture : {openness:.1f}/5

RÈGLE ABSOLUE — RÉAGIS AU DERNIER MESSAGE PRÉCISÉMENT :
Lis ce que vient de dire le délégué et réponds-y DIRECTEMENT.
Ne répète jamais une formulation déjà utilisée dans cette conversation.

PROGRESSION ({_openness_behavior(openness)})

Tour 1-2 : accueil + première objection/question.
Tour 3-4 : approfondissement, objection précise.
Tour 5-6 : position claire selon ton ouverture.
Tour 7-8 : conclusion.

STYLE : 2-3 phrases max. Ton : {doctor['description_ui']}.
Réponds UNIQUEMENT avec tes paroles. Pas de guillemets, pas de préfixe.

OBJECTIONS TYPIQUES :
{chr(10).join(f'- {o}' for o in doctor['objections_favorites'])}
"""


def _openness_behavior(openness: float) -> str:
    if openness >= 4.2:
        return "Tu es clairement intéressé — questions positives sur la mise en pratique."
    elif openness >= 3.5:
        return "Tu t'ouvres — demande des précisions sans t'engager."
    elif openness >= 2.5:
        return "Tu es neutre — objection modérée."
    elif openness >= 1.5:
        return "Tu es sceptique — doute direct sur le dernier argument."
    else:
        return "Tu es très fermé — impatience, fin de visite imminente."


def _build_report_system() -> str:
    return (
        "Tu es ALIA, évaluateur IA de VITAL SA. "
        "Génère un rapport de visite concis et actionnable. "
        "Réponds UNIQUEMENT en JSON valide, sans markdown."
    )


# ══════════════════════════════════════════════════════════════════════
# SIMULATION SESSION
# ══════════════════════════════════════════════════════════════════════

class SimulationSession:
    """
    Session de simulation intégrée.
    Tous les modules (NLP, LSTM, RAG) alimentent SessionState
    à chaque tour pour un score global cohérent en temps réel.
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
        self.final_decision = None  # Décision finale (commande/refus/conditions)
        # ── État centralisé ───────────────────────────────────────────
        self.state          = SessionState()

    # ── Extraction des N dernières réponses du médecin ──────────────
    def _recent_doctor_msgs(self, n: int = 4) -> List[str]:
        """Retourne les N dernières réponses du médecin (fix répétition)."""
        msgs = [m["content"] for m in self.history if m["role"] == "assistant"]
        return msgs[-n:] if msgs else []

    # ── Premier message ─────────────────────────────────────────────

    def first_message(self) -> Dict:
        system  = build_enriched_doctor_system(self.doctor, self.product, 0, self.openness, MAX_TURNS)
        rag_ctx = _get_rag_context(
            self.product['nom'],
            f"présentation {self.product['categorie']} {self.product['indication']}"
        )
        opening = {
            "role": "user",
            "content": (
                f"Un délégué de VITAL SA entre pour présenter "
                f"{self.product['nom']} ({self.product['categorie']}). "
                f"Accueille-le en 1-2 phrases selon ta personnalité."
            )
        }
        msg, engine = _generate_doctor_response(
            system=system, history=[opening],
            rag_context=rag_ctx, recent_doctor_msgs=[],
            fallback_fn=lambda: "Oui, entrez. Je vous écoute.",
        )
        self.history.append({"role": "assistant", "content": msg})

        # Push RAG dans SessionState
        self.state.push_rag_hit(0, self.product['nom'], bool(rag_ctx), engine)

        log.info(f"[Simulator] Start — engine={engine} rag={'oui' if rag_ctx else 'non'}")
        return {
            "message" : msg, "turn": self.turn,
            "openness": round(self.openness, 1),
            "step": None, "score": None,
            "coach": "La visite commence — gérez bien la permission (Étape 1).",
            "is_first": True, "engine": engine, "rag_used": bool(rag_ctx),
        }

    # ── Tour de conversation ────────────────────────────────────────

    def process_delegate_response(self, delegate_text: str) -> Dict:
        self.turn += 1

        # ── 1. NLP évaluation ────────────────────────────────────────
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

        # ── 2. Push NLP → SessionState ────────────────────────────────
        # Variation ouverture médecin
        delta        = self._delta_openness(score, conformite, quality)
        self.openness = max(1.0, min(5.0, self.openness + delta))

        self.state.push_nlp_turn(
            turn=self.turn, score=score, quality=quality,
            acrv=acrv, conformite=conformite, scores_det=scores_det,
            feedback=feedback, openness=self.openness,
        )

        # ── 3. Détection étape VM + push SessionState ─────────────────
        step = self._detect_step(delegate_text)
        self.step_history.append(step)
        self.state.push_vm_step(step)

        # ── 4. Coaching NLP ──────────────────────────────────────────
        coach = self._build_coaching(score, acrv, conformite, step, quality, feedback)

        # ── 5. Fin de visite ? ────────────────────────────────────────
        should_close = (
            self.turn >= MAX_TURNS
            or (self.openness >= 4.2 and self.turn >= 4)
            or (self.openness <= 1.2 and self.turn >= 3)
        )

        # ── 6. Historique + RAG ───────────────────────────────────────
        self.history.append({"role": "user", "content": delegate_text})
        rag_ctx = _get_rag_context(self.product['nom'], delegate_text)

        # ── 7. Prompt médecin ─────────────────────────────────────────
        system = build_enriched_doctor_system(
            self.doctor, self.product, self.turn, self.openness, MAX_TURNS)
        # ── 8. Générer réponse médecin ────────────────────────────────
        recent_msgs = self._recent_doctor_msgs(n=4)

        if should_close:
            # ── Décision finale intégrée (score global + openness) ────
            global_score_now = self.state.global_score
            vm_steps_now     = list(self.state.vm_steps_done)
            conv_summary     = " | ".join(
                m["content"][:60] for m in self.history[-4:] if m["role"] == "user"
            )
            self.final_decision = compute_final_decision(
                global_score=global_score_now,
                openness=self.openness,
                doctor=self.doctor,
                product=self.product,
                vm_steps_done=vm_steps_now,
                turns=self.turn,
            )
            # Message de clôture cohérent avec la décision
            msg    = generate_closing_message(
                doctor=self.doctor, product=self.product,
                decision=self.final_decision,
                global_score=global_score_now,
                openness=self.openness,
                conversation_summary=conv_summary,
            )
            engine = "sim_rag_decision"
            log.info(f"[Sim] Décision finale : {self.final_decision['decision']} | {self.final_decision['label']}")
        else:
            if self.openness >= 4.0:
                system += "\n\nFIN IMMINENTE — Signal positif détecté. Conclus en orientant vers un accord."
            msg, engine = _generate_doctor_response(
                system=system, history=self.history,
                rag_context=rag_ctx, recent_doctor_msgs=recent_msgs,
                fallback_fn=lambda: self._static_fallback(False),
            )

        self.history.append({"role": "assistant", "content": msg})

        # ── 9. Push RAG → SessionState ────────────────────────────────
        self.state.push_rag_hit(self.turn, self.product['nom'],
                                bool(rag_ctx), engine)

        if should_close:
            self.is_finished = True

        # Score global depuis SessionState
        global_score  = self.state.global_score
        global_niveau = self.state.global_niveau

        log.info(
            f"[Sim] Tour {self.turn} | NLP={score:.2f} | open={self.openness:.1f} "
            f"| global={global_score:.2f} | engine={engine} | rag={'Y' if rag_ctx else 'N'}"
        )

        return {
            "message"      : msg,
            "turn"         : self.turn,
            "openness"     : round(self.openness, 1),
            "step"         : step,
            "score"        : round(score, 2),
            "quality"      : quality,
            "acrv"         : acrv,
            "acrv_detail"  : acrv_det,
            "conformite"   : conformite,
            "scores"       : scores_det,
            "coach"        : coach,
            "is_finished"  : self.is_finished,
            "delta_open"   : round(delta, 2),
            "engine"       : engine,
            "rag_used"     : bool(rag_ctx),
            # Données intégrées
            "global_score" : global_score,
            "global_niveau": global_niveau,
            # Décision finale (peuplée uniquement au dernier tour)
            "final_decision": self.final_decision,
        }

    # ── Rapport final ───────────────────────────────────────────────

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

        if self.openness >= 4.2:
            resultat, ico, col = "Engagement confirmé — le médecin va tester le produit", "🏆", "green"
        elif self.openness >= 3.0:
            resultat, ico, col = "Engagement partiel — intérêt manifesté, suivi nécessaire", "⚠️", "gold"
        else:
            resultat, ico, col = "Visite non concluante — médecin non convaincu", "❌", "red"

        steps_done   = list(set(s for s in self.step_history if s))
        niveau_final = self._infer_niveau(avg_score)

        # Score global final depuis SessionState
        global_score_final = self.state.global_score

        summary = {
            "produit": self.product['nom'], "medecin": self.doctor['nom'],
            "difficulte": self.doctor['difficulte'], "tours": n,
            "score_nlp": avg_score, "score_global": global_score_final,
            "acrv_moyen": acrv_avg, "conformite": f"{taux_conf}%",
            "ouverture_fin": round(self.openness, 1), "resultat": resultat,
            "lstm_score": self.state.lstm_posture_score,
            "vm_steps": sorted(list(self.state.vm_steps_done)),
            "rag_hits": len([h for h in self.state.rag_hits if h['context_used']]),
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
                system=_build_report_system(),
            )
            analysis = json.loads(raw.replace('```json','').replace('```','').strip())
        except Exception:
            analysis = {
                "points_forts" : ["Engagement dans la simulation intégrée"],
                "axes_travail" : ["Améliorer le closing BIP", "Renforcer l'A-C-R-V"],
                "conseil_final": "Continuez à pratiquer sur ce profil médecin.",
            }

        return {
            "doctor"         : self.doctor,
            "product"        : self.product,
            "turns"          : n,
            "avg_score"      : avg_score,
            "max_score"      : max_score,
            "global_score"   : global_score_final,
            "acrv_avg"       : acrv_avg,
            "taux_conforme"  : taux_conf,
            "pct_excellent"  : round(n_exc / n * 100),
            "ouverture_fin"  : round(self.openness, 1),
            "niveau_final"   : niveau_final,
            "resultat"       : resultat,
            "resultat_ico"   : ico,
            "resultat_col"   : col,
            "steps_done"     : steps_done,
            "scores_history" : self.scores_history,
            "points_forts"   : analysis.get('points_forts', []),
            "axes_travail"   : analysis.get('axes_travail', []),
            "conseil_final"  : analysis.get('conseil_final', ''),
            # Données intégrées dans le rapport
            "lstm_score"     : self.state.lstm_posture_score,
            "rag_hits"       : len([h for h in self.state.rag_hits if h['context_used']]),
            "vm_steps_done"  : sorted(list(self.state.vm_steps_done)),
            "state_snapshot" : self.state.dashboard_data(),
            "final_decision" : self.final_decision,
        }

    # ── Helpers ────────────────────────────────────────────────────

    def _evaluate_nlp(self, delegate_text: str) -> Dict:
        if self.nlp_model is None:
            return {
                'overall_score': 5.0, 'quality': 'Bon', 'acrv_score': 2,
                'acrv_detail': {}, 'conformite': True,
                'feedback_coaching': [], 'scores': {},
            }
        last_doctor = next(
            (m["content"] for m in reversed(self.history) if m["role"] == "assistant"), ""
        )
        try:
            return self.nlp_model.predict(last_doctor, delegate_text)
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
        for step in VISIT_STEPS:
            if any(kw in tl for kw in step['keywords']):
                return step['num']
        return None

    def _build_coaching(self, score, acrv, conformite, step, quality, feedback) -> str:
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
            return "C'est intéressant. Laissez-moi votre documentation."
        if closing:
            return "Bien, merci pour votre passage. J'ai d'autres patients."
        options = [
            "Qu'est-ce qui distingue vraiment votre produit ?",
            "Et sur le plan des données cliniques, qu'avez-vous ?",
            "Hmm. Et côté tolérance, des retours terrain ?",
            "Quel profil de patient ciblez-vous précisément ?",
            "Vous avez des études sur ce point ?",
        ]
        return random.choice(options)

    # ── Sérialisation ──────────────────────────────────────────────

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
            'is_finished'    : self.is_finished,
            'final_decision' : self.final_decision,
            'state'          : self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SimulationSession':
        s = cls(data['doctor_id'], data['product_id'], data['niveau_alia'])
        s.turn          = data['turn']
        s.openness      = data['openness']
        s.history       = data['history']
        s.scores_history= data['scores_history']
        s.step_history  = data['step_history']
        s.is_finished    = data['is_finished']
        s.final_decision = data.get('final_decision', None)
        if 'state' in data:
            s.state = SessionState.from_dict(data['state'])
        return s
