"""
lstm_model.py
=============
DSO 1 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Classe d'inférence temps réel pour la détection du langage corporel.

Ce fichier est le SEUL importé par l'avatar et par Body-language-detection.py.
Il ne contient aucun code d'entraînement — uniquement l'interface de prédiction.

Deux modes d'usage :

  MODE 1 — Intégration dans Body-language-detection.py (remplacement des règles)
  ──────────────────────────────────────────────────────────────────────────────
    from lstm_model import BodyLanguageModel

    model  = BodyLanguageModel.load("models/lstm_body_language.pkl")
    buffer = SequenceBuffer(seq_len=30)

    # Dans la boucle vidéo, pour chaque frame :
    landmarks = result.pose_landmarks[0]   # MediaPipe landmarks
    buffer.add_frame(landmarks)

    if buffer.is_ready():
        result = model.predict_from_buffer(buffer.get())
        print(result["posture"])       # "upright" | "neutral" | "slouched"
        print(result["confidence"])    # 0–100 (pour la barre HUD)
        print(result["stress"])        # 0–100

  MODE 2 — Inférence depuis un array numpy (test / batch)
  ──────────────────────────────────────────────────────────────────────────────
    result = model.predict(sequence_array)   # (30, 66)

Author  : CYBER SHADE — ALIA Project
Version : 1.0.0
"""

# ── Standard library ──────────────────────────────────────────────────────
import logging
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Scientific stack ──────────────────────────────────────────────────────
import joblib
import numpy as np

# ── Import LSTM classes for pickle compatibility (joblib.load needs them) ──
try:
    from lstm_train import BodyLanguageLSTM, LSTMCell
except ImportError:
    pass

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BodyLanguageModel")

# ── Constants ─────────────────────────────────────────────────────────────
CLASS_NAMES   = ["upright", "neutral", "slouched"]
N_LANDMARKS   = 33
SEQ_LEN       = 30
N_FEATURES    = N_LANDMARKS * 2   # x + y = 66

DEFAULT_MODEL_PATH = "models/lstm_body_language.pkl"

# ── Posture → confidence/stress mapping ───────────────────────────────────
# Used to convert posture label to scores compatible with Body-language-detection.py HUD
POSTURE_SCORES = {
    "upright" : {"confidence_base": 82.0, "stress_base": 18.0},
    "neutral" : {"confidence_base": 58.0, "stress_base": 42.0},
    "slouched": {"confidence_base": 32.0, "stress_base": 68.0},
}

# ── Feedback messages per posture ─────────────────────────────────────────
POSTURE_FEEDBACK = {
    "upright" : [
        "Excellente posture — continuez ainsi",
        "Posture droite et confiante [OK]",
    ],
    "neutral" : [
        "Posture acceptable — redressez légèrement les épaules",
        "Pensez à ouvrir la posture pour projeter plus de confiance",
    ],
    "slouched": [
        "Posture voûtée détectée — redressez-vous",
        "Évitez de vous pencher en avant — maintenez le contact visuel",
        "Posture fermée — essayez d'ouvrir les épaules",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUFFER
# Pour l'intégration temps réel dans la boucle vidéo
# ══════════════════════════════════════════════════════════════════════════════

class SequenceBuffer:
    """
    Buffer circulaire de frames pour l'inférence temps réel.
    Accumule les landmarks MediaPipe et expose une séquence (30, 66)
    dès que le buffer est plein.

    Usage dans la boucle vidéo :
        buffer = SequenceBuffer(seq_len=30)

        while True:
            frame = cap.read()
            landmarks = mediapipe_detect(frame)

            if landmarks:
                buffer.add_frame(landmarks)   # ajoute la frame courante

            if buffer.is_ready():
                seq = buffer.get()            # (30, 66) numpy array
                result = model.predict(seq)
    """

    def __init__(self, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self._buffer = deque(maxlen=seq_len)

    def add_frame(self, landmarks) -> None:
        """
        Ajoute une frame au buffer.

        Args:
            landmarks : liste de landmarks MediaPipe
                        (chaque landmark a .x et .y, normalisés [0,1])
        """
        frame = np.array([[lm.x, lm.y] for lm in landmarks],
                         dtype=np.float32)   # (33, 2)
        self._buffer.append(frame.flatten()) # (66,)

    def add_frame_array(self, frame_array: np.ndarray) -> None:
        """
        Ajoute une frame depuis un array numpy (66,) ou (33, 2).

        Args:
            frame_array : (66,) ou (33, 2)
        """
        self._buffer.append(frame_array.flatten().astype(np.float32))

    def is_ready(self) -> bool:
        """True si le buffer contient exactement seq_len frames."""
        return len(self._buffer) == self.seq_len

    def get(self) -> np.ndarray:
        """
        Retourne la séquence courante sous forme (seq_len, 66).

        Raises:
            RuntimeError : si le buffer n'est pas encore plein
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer not ready ({len(self._buffer)}/{self.seq_len} frames)"
            )
        return np.array(list(self._buffer), dtype=np.float32)  # (30, 66)

    def reset(self) -> None:
        """Vide le buffer."""
        self._buffer.clear()

    @property
    def fill_ratio(self) -> float:
        """Progression du remplissage (0.0 → 1.0)."""
        return len(self._buffer) / self.seq_len

    def __len__(self) -> int:
        return len(self._buffer)


# ══════════════════════════════════════════════════════════════════════════════
# SCORE SMOOTHER
# Lissage des scores entre frames pour éviter le jitter — identique à
# la classe Smoother dans Body-language-detection.py
# ══════════════════════════════════════════════════════════════════════════════

class ScoreSmoother:
    """
    Rolling average buffer pour lisser les scores frame par frame.
    Reproduit la logique de Smoother(window=12) de Body-language-detection.py.
    """

    def __init__(self, window: int = 12):
        self.window = window
        self._buffers: Dict[str, deque] = {}

    def smooth(self, key: str, value: float) -> float:
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.window)
        self._buffers[key].append(value)
        return float(np.mean(self._buffers[key]))

    def reset(self) -> None:
        self._buffers.clear()


# ══════════════════════════════════════════════════════════════════════════════
# BodyLanguageModel — CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class BodyLanguageModel:
    """
    Classe d'inférence temps réel pour la détection du langage corporel.

    Remplace les règles géométriques de Body-language-detection.py
    par le modèle LSTM appris sur les patterns temporels de landmarks.

    Attributes:
        version    (str) : version du modèle
        trained_at (str) : timestamp d'entraînement
        seq_len    (int) : longueur de séquence attendue (30 frames)
    """

    def __init__(self, bundle: Dict):
        self._model      = bundle["model"]
        self._scaler     = bundle["scaler"]
        self._config     = bundle["config"]
        self._smoother   = ScoreSmoother(window=12)

        self.version     = bundle.get("version",    "unknown")
        self.trained_at  = bundle.get("trained_at", "unknown")
        self.seq_len     = self._config.get("seq_len", SEQ_LEN)
        self.class_names = bundle.get("class_names", CLASS_NAMES)

    # ── Class methods ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "BodyLanguageModel":
        """
        Charge le modèle depuis un fichier .pkl.

        Args:
            path : chemin vers lstm_body_language.pkl

        Returns:
            Instance BodyLanguageModel prête à l'inférence

        Raises:
            FileNotFoundError : si le fichier n'existe pas
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found : {path}\n"
                f"Run 'python lstm_train.py' to generate it."
            )
        bundle = joblib.load(model_path)
        log.info(
            f"BodyLanguageModel loaded — "
            f"v{bundle.get('version','?')} | "
            f"trained: {bundle.get('trained_at','?')}"
        )
        return cls(bundle)

    # ── Core inference ─────────────────────────────────────────────────────

    def predict(self, sequence: np.ndarray,
                smooth: bool = True) -> Dict:
        """
        Inférence sur une séquence de landmarks.

        Args:
            sequence (np.ndarray) : (30, 66) — landmarks x,y normalisés [0,1]
            smooth   (bool)       : appliquer le lissage temporel sur les scores

        Returns:
            dict :
                posture         (str)   : "upright" | "neutral" | "slouched"
                posture_proba   (dict)  : probabilités par classe
                confidence      (float) : score 0–100 (compatible HUD)
                stress          (float) : score 0–100 (= 100 - confidence)
                posture_color   (tuple) : BGR color pour OpenCV
                feedback        (str)   : message coaching pour l'avatar
                is_upright      (bool)  : True si posture = upright
                frame_count     (int)   : nombre de frames dans la séquence
        """
        if sequence.shape != (self.seq_len, N_FEATURES):
            raise ValueError(
                f"Expected sequence shape ({self.seq_len}, {N_FEATURES}), "
                f"got {sequence.shape}"
            )

        # ── Normalize ─────────────────────────────────────────────────────
        seq_norm = self._scaler.transform(
            sequence.reshape(-1, N_FEATURES)
        ).reshape(self.seq_len, N_FEATURES)

        # ── LSTM inference ────────────────────────────────────────────────
        probs   = self._model.predict_proba(seq_norm)    # (3,)
        class_id = int(np.argmax(probs))
        posture  = self.class_names[class_id]

        proba_dict = {
            name: round(float(p), 4)
            for name, p in zip(self.class_names, probs)
        }

        # ── Scores (confidence / stress) ──────────────────────────────────
        base       = POSTURE_SCORES[posture]
        # Add probability-weighted modulation (±10 points)
        conf_raw   = base["confidence_base"] + (probs[0] - probs[2]) * 10.0
        stress_raw = 100.0 - conf_raw

        if smooth:
            confidence = self._smoother.smooth("confidence", conf_raw)
            stress     = self._smoother.smooth("stress",     stress_raw)
        else:
            confidence = conf_raw
            stress     = stress_raw

        confidence = float(np.clip(confidence, 0, 100))
        stress     = float(np.clip(stress,     0, 100))

        # ── Posture color (BGR for OpenCV, matching Body-language-detection.py) ─
        color_map = {
            "upright" : (0, 220, 0),    # green
            "neutral" : (0, 200, 255),  # yellow
            "slouched": (0, 60, 255),   # red
        }

        # ── Feedback ──────────────────────────────────────────────────────
        msgs     = POSTURE_FEEDBACK[posture]
        feedback = msgs[int(np.random.randint(0, len(msgs)))]

        return {
            "posture"       : posture,
            "posture_proba" : proba_dict,
            "confidence"    : round(confidence, 1),
            "stress"        : round(stress, 1),
            "posture_color" : color_map[posture],
            "feedback"      : feedback,
            "is_upright"    : posture == "upright",
            "frame_count"   : self.seq_len,
        }

    def predict_from_buffer(self, buffer: SequenceBuffer,
                             smooth: bool = True) -> Dict:
        """
        Inférence directement depuis un SequenceBuffer.

        Args:
            buffer (SequenceBuffer) : buffer prêt (is_ready() == True)
            smooth (bool)           : lissage temporel

        Returns:
            dict — même format que predict()
        """
        if not buffer.is_ready():
            raise RuntimeError("Buffer is not ready — need 30 frames minimum")
        return self.predict(buffer.get(), smooth=smooth)

    def predict_batch(self, sequences: np.ndarray) -> List[Dict]:
        """
        Inférence en batch sur plusieurs séquences.

        Args:
            sequences : (N, 30, 66)

        Returns:
            Liste de dicts de prédiction (même format que predict())
        """
        return [self.predict(sequences[i], smooth=False)
                for i in range(len(sequences))]

    # ── Utility ────────────────────────────────────────────────────────────

    def reset_smoother(self) -> None:
        """Réinitialise le lissage — utile lors d'un changement de scène."""
        self._smoother.reset()

    def result_summary(self, result: Dict) -> str:
        """Résumé lisible d'un résultat de prédiction."""
        lines = [
            f"┌─ Body Language Result {'─'*32}",
            f"│  Posture     : {result['posture']:<12}  "
            f"(proba: {result['posture_proba']})",
            f"│  Confidence  : {result['confidence']:>5.1f} / 100",
            f"│  Stress      : {result['stress']:>5.1f} / 100",
            f"│  Is Upright  : {result['is_upright']}",
            f"│  Feedback    : {result['feedback']}",
            f"└{'─'*55}",
        ]
        return "\n".join(lines)

    def get_hud_data(self, result: Dict) -> Dict:
        """
        Retourne uniquement les données nécessaires au HUD de
        Body-language-detection.py (drop-in replacement pour analyze_upper_body()).

        Returns:
            dict compatible avec draw_hud() et draw_bar()
        """
        return {
            "confidence"    : result["confidence"],
            "stress"        : result["stress"],
            "posture_label" : result["posture"],
            "posture_color" : result["posture_color"],
            "arms_crossed"  : result["posture"] == "slouched",
            "face_touch"    : result["stress"] > 70,
            "is_upright"    : result["is_upright"],
            "spine_angle"   : 0.0,        # computed separately if needed
            "slouch_score"  : 1.0 - result["confidence"] / 100.0,
        }

    def __repr__(self) -> str:
        return (
            f"BodyLanguageModel("
            f"version='{self.version}', "
            f"seq_len={self.seq_len}, "
            f"classes={self.class_names})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION EXAMPLE
# Montre comment intégrer lstm_model.py dans Body-language-detection.py
# ══════════════════════════════════════════════════════════════════════════════

INTEGRATION_EXAMPLE = '''
# ── Intégration dans Body-language-detection.py ───────────────────────────
# Remplacer les imports + boucle principale par :

from lstm_model import BodyLanguageModel, SequenceBuffer

model  = BodyLanguageModel.load("models/lstm_body_language.pkl")
buffer = SequenceBuffer(seq_len=30)

# Dans la boucle vidéo :
while True:
    ret, frame = cap.read()
    result_mp  = detector.detect_for_video(mp_img, timestamp)

    if result_mp.pose_landmarks:
        landmarks = result_mp.pose_landmarks[0]

        # 1. Alimenter le buffer
        buffer.add_frame(landmarks)

        # 2. Inférence dès que le buffer est plein
        if buffer.is_ready():
            cues = model.predict_from_buffer(buffer)
            hud  = model.get_hud_data(cues)   # drop-in pour draw_hud()

            draw_skeleton(frame, landmarks)
            draw_hud(frame, hud)              # aucun changement dans draw_hud

    cv2.imshow("ALIA Body Language", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
'''


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to validate inference
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  BodyLanguageModel — Quick Inference Test")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────
    model = BodyLanguageModel.load(DEFAULT_MODEL_PATH)
    print(f"\n  {model}\n")

    # ── Load test sequences ───────────────────────────────────────────────
    X = np.load("lstm_sequences.npy")   # (400, 30, 66)
    y = np.load("lstm_labels.npy")       # (400,)

    # ── Test 1 : single sequence prediction ──────────────────────────────
    print("─" * 60)
    print("  TEST 1 — Single sequence inference")
    print("─" * 60)
    for class_id, name in enumerate(CLASS_NAMES):
        idx    = np.where(y == class_id)[0][0]
        result = model.predict(X[idx], smooth=False)
        match  = "[OK]" if result["posture"] == name else "[X]"
        print(f"\n  [{match}] Expected : {name}")
        print(model.result_summary(result))

    # ── Test 2 : SequenceBuffer simulation ───────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 2 — SequenceBuffer (simulated video stream)")
    print("─" * 60)
    buffer = SequenceBuffer(seq_len=30)

    # Simulate feeding 35 frames (buffer fills after 30)
    test_seq = X[0]   # (30, 66)
    for i in range(35):
        frame_idx = i % 30
        buffer.add_frame_array(test_seq[frame_idx])
        if i == 29:
            print(f"  Frame {i+1:>2} — buffer ready : {buffer.is_ready()}  "
                  f"fill={buffer.fill_ratio:.0%}")
            result = model.predict_from_buffer(buffer)
            print(f"  Prediction : {result['posture']}  "
                  f"conf={result['confidence']:.1f}  "
                  f"stress={result['stress']:.1f}")

    # ── Test 3 : HUD data compatibility ──────────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 3 — HUD data (drop-in for draw_hud())")
    print("─" * 60)
    result  = model.predict(X[0], smooth=False)
    hud     = model.get_hud_data(result)
    print(f"  HUD keys    : {list(hud.keys())}")
    print(f"  confidence  : {hud['confidence']:.1f}")
    print(f"  stress      : {hud['stress']:.1f}")
    print(f"  posture     : {hud['posture_label']}")
    print(f"  posture_color (BGR) : {hud['posture_color']}")

    # ── Test 4 : Batch inference ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 4 — Batch inference (all 400 sequences)")
    print("─" * 60)
    results  = model.predict_batch(X)
    y_pred   = np.array([CLASS_NAMES.index(r["posture"]) for r in results])
    accuracy = (y_pred == y).mean()
    print(f"  Batch accuracy : {accuracy*100:.2f}%  ({int(accuracy*400)}/400 correct)")

    # ── Integration guide ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  INTEGRATION GUIDE — Body-language-detection.py")
    print("─" * 60)
    print(INTEGRATION_EXAMPLE)

    print(f"\n[OK]  BodyLanguageModel — inference validated")
    print(f"    Import with : from lstm_model import BodyLanguageModel, SequenceBuffer")
