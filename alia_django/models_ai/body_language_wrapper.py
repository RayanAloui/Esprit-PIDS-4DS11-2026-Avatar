"""
body_language_wrapper.py
========================
Wrapper professionnel pour la détection du langage corporel.

Remplace le bundle LSTM corrompu par une analyse géométrique robuste
basée sur les landmarks MediaPipe (33 points, coordonnées normalisées x,y).

Architecture de l'analyse :
    - 30 frames de landmarks sont accumulées dans un SequenceBuffer
    - Chaque frame est analysée géométriquement (posture, bras, mains, visage)
    - Les résultats sont agrégés et lissés temporellement sur les 30 frames
    - La classification finale combine plusieurs indicateurs pondérés

Indicateurs analysés :
    1. Posture (upright / neutral / slouched)
       - ratio épaules/nez  : détecte si la tête est rentrée dans les épaules
       - ratio épaules/hanches : détecte si le torse est droit
       - inclinaison des épaules : détecte l'asymétrie gauche/droite
       - angle de la colonne vertébrale : angle entre épaules et hanches

    2. Bras croisés
       - croisement des poignets par rapport au centre du corps
       - proximité des poignets l'un par rapport à l'autre
       - position des coudes par rapport au centre

    3. Mains près du visage
       - position verticale du poignet par rapport aux épaules
       - distance horizontale du poignet par rapport au nez
       - position verticale par rapport au menton estimé

    4. Stress / Confiance
       - combinaison pondérée de tous les indicateurs ci-dessus
       - lissage temporel sur un historique glissant

Author  : ALIA Project
Version : 3.0.0 — Geometric Analysis Engine
"""

from collections import deque
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════

POSTURE_COLORS = {
    'upright' : (0, 220,   0),   # vert   BGR
    'neutral' : (0, 200, 255),   # jaune  BGR
    'slouched': (0,  60, 255),   # rouge  BGR
}

# Indices MediaPipe des landmarks utilisés
LM = {
    'nose'      :  0,
    'l_eye'     :  2,
    'r_eye'     :  5,
    'l_ear'     :  7,
    'r_ear'     :  8,
    'l_shoulder': 11,
    'r_shoulder': 12,
    'l_elbow'   : 13,
    'r_elbow'   : 14,
    'l_wrist'   : 15,
    'r_wrist'   : 16,
    'l_hip'     : 23,
    'r_hip'     : 24,
    'l_knee'    : 25,
    'r_knee'    : 26,
}


# ══════════════════════════════════════════════════════════════════════
# SMOOTHER — lissage temporel
# ══════════════════════════════════════════════════════════════════════

class _Smoother:
    """Rolling average pour lisser les scores frame par frame."""

    def __init__(self, window: int = 15):
        self._bufs: dict = {}
        self._window     = window

    def smooth(self, key: str, value: float) -> float:
        if key not in self._bufs:
            self._bufs[key] = deque(maxlen=self._window)
        self._bufs[key].append(float(value))
        return float(np.mean(self._bufs[key]))

    def reset(self) -> None:
        self._bufs.clear()


# ══════════════════════════════════════════════════════════════════════
# ANALYSE D'UNE FRAME — fonctions utilitaires
# ══════════════════════════════════════════════════════════════════════

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle en degrés entre deux vecteurs 2D."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    cos   = np.dot(v1, v2) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _analyze_frame(lm_array: np.ndarray) -> dict:
    """
    Analyse une frame de landmarks (33, 2) et retourne un dict de features.

    Paramètre
    ---------
    lm_array : np.ndarray shape (33, 2)
        Coordonnées x,y normalisées [0,1] de chaque landmark MediaPipe.

    Retourne
    --------
    dict avec les features géométriques brutes de cette frame.
    """
    # ── Extraction des points clés ────────────────────────────────────
    nose    = lm_array[LM['nose']]
    l_sh    = lm_array[LM['l_shoulder']]
    r_sh    = lm_array[LM['r_shoulder']]
    l_hip   = lm_array[LM['l_hip']]
    r_hip   = lm_array[LM['r_hip']]
    l_wr    = lm_array[LM['l_wrist']]
    r_wr    = lm_array[LM['r_wrist']]
    l_el    = lm_array[LM['l_elbow']]
    r_el    = lm_array[LM['r_elbow']]
    l_eye   = lm_array[LM['l_eye']]
    r_eye   = lm_array[LM['r_eye']]

    # ── Points composites ─────────────────────────────────────────────
    mid_sh  = (l_sh + r_sh)   / 2.0
    mid_hip = (l_hip + r_hip) / 2.0
    mid_eye = (l_eye + r_eye) / 2.0

    # Largeur des épaules — sert d'unité de mesure normalisée
    sh_w = float(np.linalg.norm(l_sh - r_sh)) + 1e-9

    # ── 1. POSTURE — analyse de la courbure du dos ────────────────────

    # ratio_a : largeur épaules / distance nez→épaules
    #   petit = tête haute et éloignée = bonne posture
    #   grand = tête rentrée dans les épaules = voûté
    nose_sh_dist = abs(nose[1] - mid_sh[1]) + 1e-9
    ratio_a      = sh_w / nose_sh_dist

    # ratio_b : largeur épaules / hauteur torse
    #   grand = torse court/comprimé = voûté
    torso_h = abs(mid_hip[1] - mid_sh[1]) + 1e-9
    ratio_b = sh_w / torso_h

    # Inclinaison latérale des épaules
    sh_tilt = abs(l_sh[1] - r_sh[1]) / sh_w

    # Angle de la colonne vertébrale (épaules → hanches vs vertical)
    spine_vec   = mid_hip - mid_sh
    spine_angle = _angle_between(spine_vec, np.array([0.0, 1.0]))

    # Score de voûtement composite [0, 1]
    s_a         = float(np.clip((ratio_a - 0.55) / 1.1, 0.0, 1.0))
    s_b         = float(np.clip((ratio_b - 0.40) / 0.9, 0.0, 1.0))
    s_t         = float(np.clip(sh_tilt / 0.10,         0.0, 1.0))
    s_spine     = float(np.clip(spine_angle / 25.0,     0.0, 1.0))
    slouch      = s_a * 0.40 + s_b * 0.30 + s_t * 0.15 + s_spine * 0.15

    # Hunch (tête rentrée) — distance verticale épaules→nez normalisée
    hunch = float(max(0.0, 1.0 - nose_sh_dist / (sh_w * 0.55)))

    # Classification posture pour cette frame
    # Upright : épaules larges, mains au niveau des hanches
    l_hand_at_side = (abs(l_wr[1] - l_hip[1]) < sh_w * 0.9 and
                      abs(l_wr[0] - l_hip[0]) < sh_w * 1.1)
    r_hand_at_side = (abs(r_wr[1] - r_hip[1]) < sh_w * 0.9 and
                      abs(r_wr[0] - r_hip[0]) < sh_w * 1.1)
    is_upright_f   = (ratio_a < 0.95) and l_hand_at_side and r_hand_at_side

    # ── 2. BRAS CROISÉS — détection multi-critères ───────────────────

    mid_x = mid_sh[0]

    # Critère A : les poignets sont-ils du mauvais côté du centre ?
    # (poignet gauche à droite du centre ET poignet droit à gauche)
    # Note : en coordonnées normalisées MediaPipe, x augmente vers la droite
    # Mais l'image est flippée, donc gauche/droite sont inversés
    l_wr_crosses = l_wr[0] < (mid_x - sh_w * 0.08)
    r_wr_crosses = r_wr[0] > (mid_x + sh_w * 0.08)
    cross_A      = l_wr_crosses and r_wr_crosses

    # Critère B : les poignets sont très proches l'un de l'autre
    # devant le torse (entre épaules et hanches)
    wrist_dist = float(np.linalg.norm(l_wr - r_wr)) / sh_w
    torse_mid_y = (mid_sh[1] + mid_hip[1]) / 2.0
    wrists_in_front = (l_wr[1] > mid_sh[1] and l_wr[1] < mid_hip[1] and
                       r_wr[1] > mid_sh[1] and r_wr[1] < mid_hip[1])
    cross_B = wrist_dist < 0.45 and wrists_in_front

    # Critère C : les coudes pointent vers l'extérieur (bras croisés)
    l_el_out = l_el[0] < (mid_x - sh_w * 0.3)
    r_el_out = r_el[0] > (mid_x + sh_w * 0.3)
    cross_C  = l_el_out and r_el_out and wrist_dist < 0.7

    # Score croisement [0, 1]
    cross_score = float(cross_A) * 0.50 + float(cross_B) * 0.35 + float(cross_C) * 0.15
    arms_crossed_f = cross_score >= 0.50

    # ── 3. MAINS PRÈS DU VISAGE — détection précise ──────────────────

    # Estimation du niveau du menton (entre le nez et mi-épaules)
    chin_y    = nose[1] + (mid_sh[1] - nose[1]) * 0.35
    face_w    = sh_w * 0.65   # largeur approximative du visage

    # Un poignet est "près du visage" si :
    #   - verticalement : au-dessus du menton (y < chin_y)
    #   - horizontalement : dans la largeur du visage autour du nez
    l_face_touch = (l_wr[1] < chin_y and
                    abs(l_wr[0] - nose[0]) < face_w)
    r_face_touch = (r_wr[1] < chin_y and
                    abs(r_wr[0] - nose[0]) < face_w)

    # Tolérance étendue : poignet proche de l'oreille aussi
    l_ear_touch  = (l_wr[1] < mid_sh[1] and
                    abs(l_wr[0] - lm_array[LM['l_ear']][0]) < sh_w * 0.4)
    r_ear_touch  = (r_wr[1] < mid_sh[1] and
                    abs(r_wr[0] - lm_array[LM['r_ear']][0]) < sh_w * 0.4)

    face_touch_f = l_face_touch or r_face_touch or l_ear_touch or r_ear_touch
    face_score   = float(l_face_touch or l_ear_touch) * 0.5 + \
                   float(r_face_touch or r_ear_touch) * 0.5

    return {
        'slouch'       : slouch,
        'hunch'        : hunch,
        'spine_angle'  : spine_angle,
        'wrist_dist'   : wrist_dist,
        'cross_score'  : cross_score,
        'face_score'   : face_score,
        'arms_crossed' : arms_crossed_f,
        'face_touch'   : face_touch_f,
        'is_upright'   : is_upright_f,
        'ratio_a'      : ratio_a,
        'ratio_b'      : ratio_b,
    }


# ══════════════════════════════════════════════════════════════════════
# SimpleBodyLanguageModel — classe principale
# ══════════════════════════════════════════════════════════════════════

class SimpleBodyLanguageModel:
    """
    Modèle de détection du langage corporel basé sur l'analyse géométrique.

    Remplace le bundle LSTM corrompu par une analyse déterministe robuste
    sur 30 frames de landmarks MediaPipe.

    Usage
    -----
    model  = SimpleBodyLanguageModel(scaler)   # scaler depuis le bundle pkl
    buffer = SequenceBuffer(seq_len=30)        # buffer standard lstm_model_v2

    # Dans la boucle vidéo :
    buffer.add_frame(landmarks)
    if buffer.is_ready():
        cues = model.predict_from_buffer(buffer)
        # cues est compatible avec draw_hud() de Body-language-detection.py
    """

    VERSION = '3.0.0'

    def __init__(self, scaler=None):
        """
        Paramètre
        ---------
        scaler : sklearn StandardScaler ou None
            Accepté pour compatibilité mais non utilisé —
            l'analyse est purement géométrique.
        """
        self._scaler  = scaler   # conservé pour compatibilité
        self._smoother = _Smoother(window=18)

    # ── Prédiction depuis buffer ──────────────────────────────────────

    def predict_from_buffer(self, buffer, niveau_alia: str = 'Junior',
                            smooth: bool = True) -> dict:
        """
        Analyse un SequenceBuffer de 30 frames et retourne les cues HUD.

        Paramètre
        ---------
        buffer      : SequenceBuffer — doit être is_ready()
        niveau_alia : str — non utilisé, conservé pour compatibilité
        smooth      : bool — lissage temporel activé

        Retourne
        --------
        dict compatible avec draw_hud() de Body-language-detection.py
        """
        seq = buffer.get()   # (30, 66)
        return self._analyze_sequence(seq, smooth=smooth)

    # ── Analyse de la séquence de 30 frames ──────────────────────────

    def _analyze_sequence(self, seq: np.ndarray, smooth: bool = True) -> dict:
        """
        Analyse une séquence (30, 66) de landmarks et retourne les cues HUD.

        Stratégie d'agrégation :
        - Les features sont calculées sur chaque frame indépendamment
        - Les scores continus sont moyennés sur les 30 frames
        - Les booléens (bras croisés, mains au visage) sont décidés
          par vote majoritaire pondéré (seuil > 40% des frames)
        - La classification finale utilise les scores moyens agrégés
        - Un lissage temporel est appliqué sur l'historique des sessions
        """
        n_frames = seq.shape[0]   # 30

        # ── Accumulation des features sur toutes les frames ───────────
        slouch_acc    = 0.0
        hunch_acc     = 0.0
        spine_acc     = 0.0
        cross_acc     = 0.0
        face_acc      = 0.0
        upright_votes = 0
        cross_votes   = 0
        face_votes    = 0

        for f in range(n_frames):
            lm_flat  = seq[f]                    # (66,)
            lm_array = lm_flat.reshape(33, 2)    # (33, 2)
            feat     = _analyze_frame(lm_array)

            slouch_acc    += feat['slouch']
            hunch_acc     += feat['hunch']
            spine_acc     += feat['spine_angle']
            cross_acc     += feat['cross_score']
            face_acc      += feat['face_score']
            upright_votes += int(feat['is_upright'])
            cross_votes   += int(feat['arms_crossed'])
            face_votes    += int(feat['face_touch'])

        # ── Moyennes sur les 30 frames ────────────────────────────────
        slouch_mean = slouch_acc    / n_frames
        hunch_mean  = hunch_acc     / n_frames
        spine_mean  = spine_acc     / n_frames
        cross_mean  = cross_acc     / n_frames
        face_mean   = face_acc      / n_frames

        # Votes majoritaires (seuil 40% des frames pour éviter
        # les faux négatifs causés par des frames de transition)
        upright_ratio     = upright_votes / n_frames
        arms_crossed_final = cross_votes  / n_frames > 0.40
        face_touch_final   = face_votes   / n_frames > 0.40

        # ── Classification posture ────────────────────────────────────
        #
        # upright  : épaules larges, dos droit, mains au côté
        # slouched : voûtement important OU hunch important
        # neutral  : entre les deux
        #
        # Les seuils ont été calibrés empiriquement pour correspondre
        # à des comportements réels observés avec MediaPipe.

        if upright_ratio > 0.40 and slouch_mean < 0.42 and hunch_mean < 0.38:
            posture = 'upright'
        elif slouch_mean > 0.62 or hunch_mean > 0.55 or spine_mean > 22.0:
            posture = 'slouched'
        else:
            posture = 'neutral'

        # ── Score de stress composite ─────────────────────────────────
        #
        # Pondération :
        #   - voûtement (slouch)    : 30%  — indicateur postural principal
        #   - hunch (tête rentrée)  : 25%  — stress visible dans le cou
        #   - bras croisés (cross)  : 28%  — signal de fermeture/défense
        #   - mains au visage (face): 17%  — signal de stress/réflexion
        #
        raw_stress = (slouch_mean * 0.30 +
                      hunch_mean  * 0.25 +
                      cross_mean  * 0.28 +
                      face_mean   * 0.17) * 100.0

        raw_stress = float(np.clip(raw_stress, 0.0, 100.0))

        # ── Lissage temporel ──────────────────────────────────────────
        if smooth:
            stress     = self._smoother.smooth('stress',     raw_stress)
            spine_sm   = self._smoother.smooth('spine',      spine_mean)
            slouch_sm  = self._smoother.smooth('slouch',     slouch_mean)
        else:
            stress     = raw_stress
            spine_sm   = spine_mean
            slouch_sm  = slouch_mean

        stress     = float(np.clip(stress, 0.0, 100.0))
        confidence = round(100.0 - stress, 1)
        stress     = round(stress,         1)

        # ── Override posture basé sur la confidence ───────────────────
        #
        # Si la confidence est élevée (≥ 75%) et que la posture n'est pas
        # clairement voûtée, on force "upright" — car un score de confiance
        # élevé traduit un faible stress global et donc une posture ouverte.
        # Seuil : 75% (ajustable via CONFIDENCE_UPRIGHT_THRESHOLD)
        CONFIDENCE_UPRIGHT_THRESHOLD = 82.0
        if confidence >= CONFIDENCE_UPRIGHT_THRESHOLD and posture != 'slouched':
            posture = 'upright'

        # ── Retour compatible draw_hud() ──────────────────────────────
        return {
            'confidence'   : confidence,
            'stress'       : stress,
            'posture_label': posture,
            'posture_color': POSTURE_COLORS[posture],
            'arms_crossed' : arms_crossed_final,
            'face_touch'   : face_touch_final,
            'is_upright'   : posture == 'upright',
            'spine_angle'  : round(spine_sm, 1),
            'slouch_score' : round(slouch_sm, 3),
        }

    # ── Compatibilité avec l'interface lstm_model_v2 ─────────────────

    def get_hud_data(self, result: dict) -> dict:
        """Pass-through — résultat déjà au format HUD."""
        return result

    def reset_smoother(self) -> None:
        """Réinitialise le lissage — utile lors d'un changement de scène."""
        self._smoother.reset()

    def __repr__(self) -> str:
        return (f"SimpleBodyLanguageModel("
                f"version='{self.VERSION}', "
                f"engine='geometric_analysis')")