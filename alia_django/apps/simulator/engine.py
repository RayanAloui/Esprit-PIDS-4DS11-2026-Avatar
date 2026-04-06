"""
engine.py
=========
Moteur de simulation de visite médicale.
Gère l'état de la visite, les appels à Claude API pour le médecin virtuel,
et l'évaluation NLP à chaque tour de conversation.
"""

import json
import sys
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing  import Dict, List, Optional

from .profiles import get_product, get_doctor, VISIT_STEPS

log = logging.getLogger(__name__)

# ── Constantes ─────────────────────────────────────────────────────────
CLAUDE_MODEL    = "claude-sonnet-4-20250514"
MAX_TURNS       = 8     # nombre max de tours de conversation
MAX_TOKENS_DOC  = 800   # tokens pour la réponse du médecin
MAX_TOKENS_RPT  = 1200  # tokens pour le rapport final


def _call_claude(messages: List[Dict], system: str,
                 max_tokens: int = 800) -> str:
    """Appel à l'API Claude avec gestion d'erreur."""
    payload = json.dumps({
        "model"     : CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system"    : system,
        "messages"  : messages,
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data    = payload,
        headers = {'Content-Type': 'application/json'},
        method  = 'POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
            return data['content'][0]['text'].strip()
    except Exception as e:
        log.error(f"Claude API error: {e}")
        raise


def _get_nlp_model():
    """Charge NLPScoringModel (singleton)."""
    try:
        from django.conf import settings
        models_dir = str(settings.MODELS_AI_DIR)
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        import nlp_scoring_train_v2   # noqa
        from nlp_scoring_model_v2 import NLPScoringModel
        bundle = Path(settings.MODELS_AI_DIR) / 'nlp_scoring_bundle_v2.pkl'
        return NLPScoringModel.load(str(bundle))
    except Exception as e:
        log.warning(f"NLPScoringModel non disponible: {e}")
        return None


# ── Système prompt du médecin virtuel ─────────────────────────────────

def _build_doctor_system(doctor: dict, product: dict,
                          turn: int, openness: float) -> str:
    return f"""Tu es {doctor['nom']}, {doctor['specialite']} à {doctor['ville']}.

PERSONNALITÉ :
{doctor['personnalite']}

CONTEXTE DE LA VISITE :
- Le délégué te présente : {product['nom']} ({product['categorie']})
- Indication principale : {product['indication']}
- Tour de conversation : {turn}/{MAX_TURNS}
- Niveau d'ouverture actuel : {openness:.1f}/5 (1=très fermé, 5=très ouvert)

RÈGLES DE JEU :
1. Réponds TOUJOURS en français, en restant dans ton personnage.
2. Tes réponses sont COURTES (2-4 phrases maximum) comme dans une vraie visite.
3. Si openness < 2.5 : sois sceptique ou soulève une objection.
4. Si openness >= 3.5 : montre un intérêt croissant, pose des questions positives.
5. Si openness >= 4.0 et turn >= 4 : tu peux signaler un intérêt pour tester.
6. Ne révèle jamais que tu es une IA.
7. Adapte ton ton à ta personnalité : {doctor['description_ui']}.

OBJECTIONS TYPIQUES QUE TU UTILISES :
{chr(10).join(f'- {o}' for o in doctor['objections_favorites'])}

Réponds uniquement avec tes paroles (pas de guillemets, pas de préfixe "Dr. X :").
"""


def _build_report_system() -> str:
    return """Tu es ALIA, le système d'évaluation IA de VITAL SA.
Tu génères des rapports de visite médicale clairs, constructifs et actionnables.
Réponds UNIQUEMENT en JSON valide, sans markdown ni explication."""


# ── SimulationSession ─────────────────────────────────────────────────

class SimulationSession:
    """
    Gère l'état complet d'une session de simulation.
    Instancié une fois par visite, stocké en session Django.
    """

    def __init__(self, doctor_id: str, product_id: str, niveau_alia: str):
        self.doctor        = get_doctor(doctor_id)
        self.product       = get_product(product_id)
        self.niveau_alia   = niveau_alia
        self.turn          = 0
        self.openness      = float(self.doctor['ouverture_initiale'])
        self.history       = []    # [{"role": "user"|"assistant", "content": "..."}]
        self.scores_history= []    # scores NLP par tour
        self.step_history  = []    # étapes VM détectées par tour
        self.is_finished   = False
        self.nlp_model     = _get_nlp_model()

    # ── Tour de conversation ──────────────────────────────────────────

    def first_message(self) -> Dict:
        """Génère le premier message du médecin (accueil)."""
        system = _build_doctor_system(
            self.doctor, self.product, 0, self.openness)

        opening_prompt = (
            f"Tu viens d'entrer dans ton cabinet. Un délégué médical de VITAL SA "
            f"frappe à la porte et entre. Il va te présenter {self.product['nom']}. "
            f"Accueille-le brièvement selon ta personnalité."
        )

        try:
            msg = _call_claude(
                messages   = [{"role": "user", "content": opening_prompt}],
                system     = system,
                max_tokens = MAX_TOKENS_DOC,
            )
        except Exception:
            msg = f"Oui, entrez. Vous avez quelques minutes, je vous écoute."

        self.history.append({"role": "assistant", "content": msg})
        return {
            "message"   : msg,
            "turn"      : self.turn,
            "openness"  : round(self.openness, 1),
            "step"      : None,
            "score"     : None,
            "coach"     : "La visite commence — gérez bien la permission (Étape 1).",
            "is_first"  : True,
        }

    def process_delegate_response(self, delegate_text: str) -> Dict:
        """
        Traite la réponse du délégué :
        1. Évalue via NLP Scoring
        2. Met à jour l'ouverture du médecin
        3. Génère la réponse du médecin via Claude
        4. Détecte l'étape VM
        """
        self.turn += 1

        # ── 1. Évaluation NLP ──────────────────────────────────────────
        nlp_result  = self._evaluate_nlp(delegate_text)
        score       = nlp_result.get('overall_score', 5.0)
        acrv        = nlp_result.get('acrv_score', 0)
        conformite  = nlp_result.get('conformite', True)
        quality     = nlp_result.get('quality', 'Bon')
        feedback    = nlp_result.get('feedback_coaching', [])

        self.scores_history.append({
            'turn'      : self.turn,
            'score'     : score,
            'quality'   : quality,
            'acrv'      : acrv,
            'conformite': conformite,
        })

        # ── 2. Mettre à jour l'ouverture du médecin ───────────────────
        delta = self._compute_openness_delta(score, conformite, quality)
        self.openness = max(1.0, min(5.0, self.openness + delta))

        # ── 3. Détecter l'étape VM ────────────────────────────────────
        step = self._detect_step(delegate_text)
        self.step_history.append(step)

        # ── 4. Coaching temps réel ────────────────────────────────────
        coach = self._build_coaching(score, acrv, conformite,
                                     step, quality, feedback)

        # ── 5. Vérifier fin de visite ─────────────────────────────────
        should_close = (
            self.turn >= MAX_TURNS or
            (self.openness >= 4.2 and self.turn >= 4) or
            (self.openness <= 1.2 and self.turn >= 3)
        )

        # ── 6. Générer réponse du médecin ─────────────────────────────
        self.history.append({"role": "user", "content": delegate_text})
        system = _build_doctor_system(
            self.doctor, self.product, self.turn, self.openness)

        if should_close:
            closing_hint = (
                "\n\nC'est le moment de conclure la visite. "
                "Si openness >= 4 : montre un intérêt pour tester le produit. "
                "Si openness < 3 : mets fin poliment à la visite."
            )
            system += closing_hint

        try:
            doctor_msg = _call_claude(
                messages   = self.history[-6:],  # contexte glissant
                system     = system,
                max_tokens = MAX_TOKENS_DOC,
            )
        except Exception:
            doctor_msg = self._fallback_doctor_message(should_close)

        self.history.append({"role": "assistant", "content": doctor_msg})

        if should_close:
            self.is_finished = True

        return {
            "message"   : doctor_msg,
            "turn"      : self.turn,
            "openness"  : round(self.openness, 1),
            "step"      : step,
            "score"     : round(score, 2),
            "quality"   : quality,
            "acrv"      : acrv,
            "conformite": conformite,
            "coach"     : coach,
            "is_finished": self.is_finished,
            "delta_open": round(delta, 2),
        }

    # ── Rapport final ─────────────────────────────────────────────────

    def generate_report(self) -> Dict:
        """Génère le rapport complet de la visite."""
        n      = len(self.scores_history)
        if n == 0:
            return {"error": "Aucune donnée de visite."}

        scores      = [s['score']       for s in self.scores_history]
        avg_score   = round(sum(scores) / n, 2)
        max_score   = round(max(scores), 2)
        acrv_avg    = round(sum(s['acrv'] for s in self.scores_history) / n, 2)
        n_conforme  = sum(1 for s in self.scores_history if s['conformite'])
        taux_conf   = round(n_conforme / n * 100)
        n_excellent = sum(1 for s in self.scores_history if s['quality'] == 'Excellent')

        # Résultat visite selon l'ouverture finale
        if self.openness >= 4.2:
            resultat     = "Engagement confirmé — le médecin va tester le produit"
            resultat_ico = "🏆"
            resultat_col = "green"
        elif self.openness >= 3.0:
            resultat     = "Engagement partiel — intérêt manifesté, suivi nécessaire"
            resultat_ico = "⚠️"
            resultat_col = "gold"
        else:
            resultat     = "Visite non concluante — médecin non convaincu"
            resultat_ico = "❌"
            resultat_col = "red"

        # Étapes maîtrisées
        steps_done = list(set(self.step_history))

        # Niveau ALIA final
        niveau_final = self._infer_niveau(avg_score)

        # Appel Claude pour analyse qualitative
        summary_data = {
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
            "scores_detail": self.scores_history,
        }

        try:
            claude_analysis = self._claude_report(summary_data)
        except Exception:
            claude_analysis = {
                "points_forts"  : ["Engagement dans la visite", "Utilisation de la méthode A-C-R-V"],
                "axes_travail"  : ["Améliorer le closing BIP", "Renforcer les preuves scientifiques"],
                "conseil_final" : "Continuez à pratiquer la méthode A-C-R-V et travaillez le closing.",
            }

        return {
            "doctor"        : self.doctor,
            "product"       : self.product,
            "turns"         : n,
            "avg_score"     : avg_score,
            "max_score"     : max_score,
            "acrv_avg"      : acrv_avg,
            "taux_conforme" : taux_conf,
            "pct_excellent" : round(n_excellent / n * 100),
            "ouverture_fin" : round(self.openness, 1),
            "niveau_final"  : niveau_final,
            "resultat"      : resultat,
            "resultat_ico"  : resultat_ico,
            "resultat_col"  : resultat_col,
            "steps_done"    : steps_done,
            "scores_history": self.scores_history,
            "points_forts"  : claude_analysis.get('points_forts', []),
            "axes_travail"  : claude_analysis.get('axes_travail', []),
            "conseil_final" : claude_analysis.get('conseil_final', ''),
        }

    # ── Helpers privés ────────────────────────────────────────────────

    def _evaluate_nlp(self, delegate_text: str) -> Dict:
        """Évalue la réponse du délégué via NLPScoringModel."""
        if self.nlp_model is None:
            return {
                'overall_score': 5.0, 'quality': 'Bon',
                'acrv_score': 2, 'conformite': True,
                'feedback_coaching': [],
            }
        # On utilise le dernier message du médecin comme "objection"
        last_doctor = ""
        for msg in reversed(self.history):
            if msg["role"] == "assistant":
                last_doctor = msg["content"]
                break
        try:
            return self.nlp_model.predict(last_doctor, delegate_text)
        except Exception:
            return {'overall_score': 5.0, 'quality': 'Bon',
                    'acrv_score': 2, 'conformite': True, 'feedback_coaching': []}

    def _compute_openness_delta(self, score: float,
                                conformite: bool, quality: str) -> float:
        """Calcule la variation d'ouverture du médecin selon la qualité de réponse."""
        delta = 0.0
        if score >= 8.5:   delta += 0.8
        elif score >= 7.0: delta += 0.4
        elif score >= 5.5: delta += 0.1
        elif score >= 4.0: delta -= 0.2
        else:              delta -= 0.5
        if not conformite: delta -= 0.8  # mot tueur = fermeture immédiate
        if quality == 'Excellent': delta += 0.2
        if quality == 'Faible':    delta -= 0.3
        return delta

    def _detect_step(self, text: str) -> Optional[int]:
        """Détecte l'étape de visite médicale dans le texte du délégué."""
        text_lower = text.lower()
        for step in VISIT_STEPS:
            if any(kw in text_lower for kw in step['keywords']):
                return step['num']
        return None

    def _build_coaching(self, score: float, acrv: int, conformite: bool,
                         step: Optional[int], quality: str,
                         feedback: List[str]) -> str:
        """Construit le message coaching temps réel."""
        msgs = []

        if not conformite:
            msgs.append("🚨 MOT TUEUR détecté — évitez les promesses non vérifiables !")
        elif score >= 8.0:
            msgs.append(f"✅ Excellente réponse ({score:.1f}/10)")
        elif score >= 6.5:
            msgs.append(f"👍 Bonne réponse ({score:.1f}/10)")
        else:
            msgs.append(f"⚠️ Réponse à améliorer ({score:.1f}/10)")

        if acrv == 4:
            msgs.append("A-C-R-V complet ✅")
        elif acrv >= 2:
            msgs.append(f"A-C-R-V : {acrv}/4 — pensez à la Validation")

        if step:
            step_name = VISIT_STEPS[step-1]['nom']
            msgs.append(f"Étape {step} ({step_name}) détectée")

        if feedback and len(feedback) > 0:
            msgs.append(feedback[0])

        return " · ".join(msgs[:3])

    def _infer_niveau(self, avg_score: float) -> str:
        if avg_score >= 9.0: return "Expert"
        if avg_score >= 8.0: return "Confirmé"
        if avg_score >= 7.0: return "Junior"
        return "Débutant"

    def _claude_report(self, data: dict) -> dict:
        """Demande à Claude une analyse qualitative de la visite."""
        prompt = f"""Analyse cette visite médicale simulée et génère un rapport concis.

Données de la visite :
{json.dumps(data, ensure_ascii=False, indent=2)}

Retourne UNIQUEMENT ce JSON (sans markdown) :
{{
  "points_forts": ["point 1", "point 2"],
  "axes_travail": ["axe 1", "axe 2"],
  "conseil_final": "Un conseil actionnable en 1-2 phrases."
}}"""

        response = _call_claude(
            messages   = [{"role": "user", "content": prompt}],
            system     = _build_report_system(),
            max_tokens = MAX_TOKENS_RPT,
        )
        clean = response.replace('```json', '').replace('```', '').strip()
        return json.loads(clean)

    def _fallback_doctor_message(self, closing: bool) -> str:
        if closing and self.openness >= 4:
            return "C'est intéressant. Laissez-moi votre documentation, je vais regarder ça."
        elif closing:
            return "Bien, merci pour votre passage. J'ai d'autres patients."
        return "Je vois. Et qu'est-ce qui différencie vraiment votre produit ?"

    def to_dict(self) -> dict:
        """Sérialise la session pour stockage en session Django."""
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
        """Restaure une session depuis le stockage Django."""
        session = cls(data['doctor_id'], data['product_id'], data['niveau_alia'])
        session.turn          = data['turn']
        session.openness      = data['openness']
        session.history       = data['history']
        session.scores_history= data['scores_history']
        session.step_history  = data['step_history']
        session.is_finished   = data['is_finished']
        return session
