"""
session_state.py
================
Objet centralisé qui agrège tous les modules du Simulator en temps réel.

Architecture événementielle :
    NLP       → push_nlp_turn()     → met à jour scores NLP + score global
    LSTM      → push_lstm_frame()   → met à jour posture + pénalité stress
    RAG       → push_rag_hit()      → log du contexte RAG utilisé
    Simulator → push_turn_event()   → log événements visite

Score Global Composite (0-10) :
    0.40 × nlp_score_moyen
  + 0.25 × lstm_posture_score
  + 0.20 × conformite_score
  + 0.15 × vm_steps_score

Toutes les données sont sérialisables JSON pour stockage en session Django.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional

# ── Poids du score global ───────────────────────────────────────────
W_NLP       = 0.40
W_LSTM      = 0.25
W_CONFORMITE= 0.20
W_VM_STEPS  = 0.15

# ── Scores posture LSTM (confiance → score /10) ──────────────────────
POSTURE_BASE_SCORE = {
    'upright' : 8.5,
    'neutral' : 6.0,
    'slouched': 3.5,
}


class SessionState:
    """
    État centralisé de la session de simulation.
    Agrège NLP, LSTM, RAG et VM en un flux cohérent.

    Usage :
        state = SessionState()

        # À chaque tour NLP :
        state.push_nlp_turn(turn=1, score=7.8, quality='Bon',
                            acrv=3, conformite=True, scores_det={})

        # À chaque frame LSTM (polling) :
        state.push_lstm_frame(posture='neutral', confidence=65,
                              stress=35, arms_crossed=False,
                              face_touch=False)

        # Accéder au score global :
        score = state.global_score     # float 0-10
        data  = state.dashboard_data() # dict complet pour le frontend
    """

    def __init__(self):
        # ── NLP ───────────────────────────────────────────────────────
        self.nlp_turns: List[Dict]  = []
        self.nlp_score_mean: float  = 0.0
        self.nlp_conformite_rate: float = 1.0  # 1.0 = 100% conforme

        # ── LSTM ──────────────────────────────────────────────────────
        self.lstm_frames: List[Dict]= []
        self.all_lstm_frames: List[Dict] = []
        self.lstm_latest: Dict      = {
            'posture': 'neutral', 'confidence': 0.0,
            'stress': 0.0, 'arms_crossed': False,
            'face_touch': False, 'active': False,
        }
        self.lstm_posture_score: float = 6.0  # neutre par défaut
        self.lstm_stress_penalty: float= 0.0  # [0, 2] soustrait du score

        # ── RAG ───────────────────────────────────────────────────────
        self.rag_hits: List[Dict]   = []
        self.rag_active: bool       = False
        self.rag_last_product: str  = ''

        # ── VM Steps ──────────────────────────────────────────────────
        self.vm_steps_done: set     = set()
        self.vm_steps_score: float  = 0.0

        # ── Événements (feed temps réel) ──────────────────────────────
        self.events: List[Dict]     = []
        self.max_events: int        = 40

        # ── Score global ──────────────────────────────────────────────
        self.global_score: float    = 0.0
        self.global_niveau: str     = 'Débutant'

        # ── Openness timeline ─────────────────────────────────────────
        self.openness_timeline: List[float] = []

        # ── Timestamps ────────────────────────────────────────────────
        self.started_at: float      = time.time()
        self.last_updated: float    = time.time()

    # ══════════════════════════════════════════════════════════════════
    # PUSH — NLP
    # ══════════════════════════════════════════════════════════════════

    def push_nlp_turn(self, turn: int, score: float, quality: str,
                      acrv: int, conformite: bool,
                      scores_det: Dict, feedback: List[str],
                      openness: float) -> None:
        """Enregistre les résultats NLP d'un tour."""
        self.nlp_turns.append({
            'turn'      : turn,
            'score'     : round(score, 2),
            'quality'   : quality,
            'acrv'      : acrv,
            'conformite': conformite,
            'scores'    : scores_det,
            'ts'        : time.time(),
        })

        # Recalculer NLP moyen
        scores_list = [t['score'] for t in self.nlp_turns]
        self.nlp_score_mean = round(sum(scores_list) / len(scores_list), 2)

        # Taux conformité
        n_conf = sum(1 for t in self.nlp_turns if t['conformite'])
        self.nlp_conformite_rate = n_conf / len(self.nlp_turns)

        # Timeline ouverture
        self.openness_timeline.append(round(openness, 1))

        # Événements
        evt_type = 'success' if score >= 7.0 else ('warning' if score >= 5.0 else 'error')
        self._push_event(
            f"Tour {turn} — {quality} ({score:.1f}/10) · ACRV {acrv}/4",
            evt_type
        )
        if not conformite:
            self._push_event("🚨 MOT TUEUR détecté — conformité violée", 'critical')
        if feedback:
            self._push_event(feedback[0], 'info')

        # Recalculer score global
        self._recompute_global()
        self.last_updated = time.time()

    # ══════════════════════════════════════════════════════════════════
    # PUSH — LSTM
    # ══════════════════════════════════════════════════════════════════

    def push_lstm_frame(self, posture: str, confidence: float,
                        stress: float, arms_crossed: bool,
                        face_touch: bool) -> None:
        """
        Enregistre une frame LSTM.
        On garde uniquement les 60 dernières frames (fenêtre glissante ~30s).
        """
        frame = {
            'posture'     : posture,
            'confidence'  : round(confidence, 1),
            'stress'      : round(stress, 1),
            'arms_crossed': arms_crossed,
            'face_touch'  : face_touch,
            'ts'          : time.time(),
        }
        self.lstm_frames.append(frame)
        self.all_lstm_frames.append(frame)
        if len(self.lstm_frames) > 60:
            self.lstm_frames = self.lstm_frames[-60:]

        self.lstm_latest = {**frame, 'active': True}

        # Score posture moyen sur les 20 dernières frames
        recent = self.lstm_frames[-20:]
        posture_scores = [
            POSTURE_BASE_SCORE.get(f['posture'], 6.0) for f in recent
        ]
        self.lstm_posture_score = round(
            sum(posture_scores) / len(posture_scores), 2
        )

        # Pénalité stress : max 2 points si stress moyen > 70%
        avg_stress = sum(f['stress'] for f in recent) / len(recent)
        self.lstm_stress_penalty = round(min(2.0, avg_stress / 50), 2)

        # Événements posture (throttle — 1 event par changement de posture)
        if len(self.lstm_frames) >= 2:
            prev = self.lstm_frames[-2]['posture']
            curr = posture
            if prev != curr:
                msgs = {
                    'upright' : '🧍 Posture excellente détectée',
                    'neutral' : '🙆 Posture neutre',
                    'slouched': '⚠️ Posture voûtée — impact sur le score',
                }
                self._push_event(msgs.get(curr, f'Posture: {curr}'), 'lstm')

        self._recompute_global()
        self.last_updated = time.time()

    # ══════════════════════════════════════════════════════════════════
    # PUSH — RAG
    # ══════════════════════════════════════════════════════════════════

    def push_rag_hit(self, turn: int, product: str,
                     context_used: bool, engine: str) -> None:
        """Enregistre l'utilisation du RAG pour un tour."""
        self.rag_hits.append({
            'turn'        : turn,
            'product'     : product,
            'context_used': context_used,
            'engine'      : engine,
            'ts'          : time.time(),
        })
        self.rag_active       = context_used
        self.rag_last_product = product

        if context_used:
            self._push_event(
                f"🔍 RAG activé — {engine} · produit : {product}",
                'rag'
            )
        self.last_updated = time.time()

    # ══════════════════════════════════════════════════════════════════
    # PUSH — VM STEPS
    # ══════════════════════════════════════════════════════════════════

    def push_vm_step(self, step_num: Optional[int]) -> None:
        """Enregistre la détection d'une étape VM."""
        if step_num and step_num not in self.vm_steps_done:
            self.vm_steps_done.add(step_num)
            step_names = {
                1: 'Permission', 2: 'Sondage', 3: 'Synthèse',
                4: 'A-C-R-V', 5: 'Preuve', 6: 'BIP/Closing',
            }
            self._push_event(
                f"✅ Étape {step_num} ({step_names.get(step_num, '')}) validée",
                'vm'
            )
            # Score VM = étapes complétées / 6 × 10
            self.vm_steps_score = round(len(self.vm_steps_done) / 6 * 10, 2)
            self._recompute_global()
            self.last_updated = time.time()

    # ══════════════════════════════════════════════════════════════════
    # SCORE GLOBAL — calcul composite
    # ══════════════════════════════════════════════════════════════════

    def _recompute_global(self) -> None:
        """
        Recalcule le score global composite.
        Appelé automatiquement après chaque push.
        """
        # NLP (0-10)
        nlp_contrib = self.nlp_score_mean

        # LSTM (0-10) avec pénalité stress
        lstm_raw    = self.lstm_posture_score - self.lstm_stress_penalty
        lstm_contrib= max(0.0, min(10.0, lstm_raw))

        # Conformité (0-10) : 10 si 100% conforme, décroît avec violations
        conf_contrib = self.nlp_conformite_rate * 10.0

        # VM Steps (0-10) : étapes réalisées / 6
        vm_contrib  = self.vm_steps_score

        # Si aucune donnée NLP encore → score = 0
        if not self.nlp_turns:
            self.global_score  = 0.0
            self.global_niveau = 'En cours…'
            return

        self.global_score = round(
            W_NLP        * nlp_contrib   +
            W_LSTM       * lstm_contrib  +
            W_CONFORMITE * conf_contrib  +
            W_VM_STEPS   * vm_contrib,
            2
        )
        self.global_niveau = self._infer_niveau(self.global_score)

    def _infer_niveau(self, score: float) -> str:
        if score >= 9.0: return 'Expert'
        if score >= 8.0: return 'Confirmé'
        if score >= 7.0: return 'Junior'
        if score > 0:    return 'Débutant'
        return 'En cours…'

    # ══════════════════════════════════════════════════════════════════
    # ÉVÉNEMENTS
    # ══════════════════════════════════════════════════════════════════

    def _push_event(self, message: str, evt_type: str = 'info') -> None:
        """Ajoute un événement au feed. Conserve les N derniers."""
        self.events.append({
            'msg'  : message,
            'type' : evt_type,
            'ts'   : round(time.time() - self.started_at, 1),
        })
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    # ══════════════════════════════════════════════════════════════════
    # DASHBOARD DATA — données pour le frontend
    # ══════════════════════════════════════════════════════════════════

    def dashboard_data(self) -> Dict:
        """
        Retourne toutes les données nécessaires au dashboard temps réel.
        Appelé par GET /simulator/dashboard/ toutes les 500ms.
        """
        # Progression NLP tour par tour
        nlp_labels = [f"T{t['turn']}" for t in self.nlp_turns]
        nlp_scores = [t['score'] for t in self.nlp_turns]
        nlp_acrv   = [t['acrv']  for t in self.nlp_turns]

        # LSTM récent (20 dernières frames)
        recent_lstm = self.lstm_frames[-20:] if self.lstm_frames else []
        lstm_conf_hist  = [f['confidence'] for f in recent_lstm]
        lstm_stress_hist= [f['stress']     for f in recent_lstm]

        # Contribution de chaque module au score global
        nlp_raw   = self.nlp_score_mean
        lstm_raw  = max(0.0, self.lstm_posture_score - self.lstm_stress_penalty)
        conf_raw  = self.nlp_conformite_rate * 10.0
        vm_raw    = self.vm_steps_score

        contributions = {
            'nlp'       : round(W_NLP        * nlp_raw,  2),
            'lstm'      : round(W_LSTM       * lstm_raw, 2),
            'conformite': round(W_CONFORMITE * conf_raw, 2),
            'vm_steps'  : round(W_VM_STEPS   * vm_raw,  2),
        }

        return {
            # Score global
            'global_score'  : self.global_score,
            'global_niveau' : self.global_niveau,
            'contributions' : contributions,

            # NLP
            'nlp_score_mean': self.nlp_score_mean,
            'nlp_labels'    : nlp_labels,
            'nlp_scores'    : nlp_scores,
            'nlp_acrv'      : nlp_acrv,
            'conformite_rate': round(self.nlp_conformite_rate * 100, 1),

            # LSTM
            'lstm_latest'       : self.lstm_latest,
            'lstm_posture_score': self.lstm_posture_score,
            'lstm_stress_penalty': self.lstm_stress_penalty,
            'lstm_conf_hist'    : lstm_conf_hist,
            'lstm_stress_hist'  : lstm_stress_hist,

            # RAG
            'rag_active'    : self.rag_active,
            'rag_hits_count': len([h for h in self.rag_hits if h['context_used']]),
            'rag_last_engine': self.rag_hits[-1]['engine'] if self.rag_hits else '—',

            # VM Steps
            'vm_steps_done' : sorted(list(self.vm_steps_done)),
            'vm_steps_score': self.vm_steps_score,
            'openness_timeline': self.openness_timeline,

            # Feed événements (10 derniers)
            'events'        : self.events[-10:],

            # Meta
            'elapsed_s'     : round(time.time() - self.started_at),
            'last_updated'  : round(self.last_updated - self.started_at, 1),
        }

    # ══════════════════════════════════════════════════════════════════
    # SÉRIALISATION — session Django
    # ══════════════════════════════════════════════════════════════════

    def to_dict(self) -> Dict:
        """Sérialise pour stockage en session Django."""
        return {
            'nlp_turns'          : self.nlp_turns,
            'nlp_score_mean'     : self.nlp_score_mean,
            'nlp_conformite_rate': self.nlp_conformite_rate,
            'lstm_frames'        : self.lstm_frames[-30:],  # compact
            'all_lstm_frames'    : self.all_lstm_frames,
            'lstm_latest'        : self.lstm_latest,
            'lstm_posture_score' : self.lstm_posture_score,
            'lstm_stress_penalty': self.lstm_stress_penalty,
            'rag_hits'           : self.rag_hits,
            'rag_active'         : self.rag_active,
            'rag_last_product'   : self.rag_last_product,
            'vm_steps_done'      : sorted(list(self.vm_steps_done)),
            'vm_steps_score'     : self.vm_steps_score,
            'events'             : self.events,
            'global_score'       : self.global_score,
            'global_niveau'      : self.global_niveau,
            'openness_timeline'  : self.openness_timeline,
            'started_at'         : self.started_at,
            'last_updated'       : self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionState':
        """Restaure depuis la session Django."""
        s = cls()
        s.nlp_turns           = data.get('nlp_turns', [])
        s.nlp_score_mean      = data.get('nlp_score_mean', 0.0)
        s.nlp_conformite_rate = data.get('nlp_conformite_rate', 1.0)
        s.lstm_frames         = data.get('lstm_frames', [])
        s.all_lstm_frames     = data.get('all_lstm_frames', [])
        s.lstm_latest         = data.get('lstm_latest', s.lstm_latest)
        s.lstm_posture_score  = data.get('lstm_posture_score', 6.0)
        s.lstm_stress_penalty = data.get('lstm_stress_penalty', 0.0)
        s.rag_hits            = data.get('rag_hits', [])
        s.rag_active          = data.get('rag_active', False)
        s.rag_last_product    = data.get('rag_last_product', '')
        s.vm_steps_done       = set(data.get('vm_steps_done', []))
        s.vm_steps_score      = data.get('vm_steps_score', 0.0)
        s.events              = data.get('events', [])
        s.global_score        = data.get('global_score', 0.0)
        s.global_niveau       = data.get('global_niveau', 'En cours…')
        s.openness_timeline   = data.get('openness_timeline', [])
        s.started_at          = data.get('started_at', time.time())
        s.last_updated        = data.get('last_updated', time.time())
        return s
