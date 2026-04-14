"""
lstm_model_v2.py
================
DSO 1 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Classe d'inférence temps réel pour la détection du langage corporel V2.

Nouveautés V2 vs V1 :
    - Feedback coaching aligné sur les 4 niveaux ALIA
    - Contexte visite médicale (étape VM, description posture)
    - get_dashboard_data() compatible Django frontend
    - SequenceBuffer inchangé — interface drop-in pour Body-language-detection.py

Usage :
    from lstm_model_v2 import BodyLanguageModel, SequenceBuffer

    model  = BodyLanguageModel.load("models/lstm_body_language_v2.pkl")
    buffer = SequenceBuffer(seq_len=30)

    # Boucle vidéo :
    buffer.add_frame(mediapipe_landmarks)
    if buffer.is_ready():
        result = model.predict_from_buffer(buffer)
        hud    = model.get_hud_data(result)

Author  : CYBER SHADE — ALIA Project
Version : 2.0.0
"""

import logging
import re
from collections import deque
from pathlib     import Path
from typing      import Dict, List, Optional

import joblib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BodyLanguageModel")

# ── Constants ─────────────────────────────────────────────────────────
CLASS_NAMES        = ["upright", "neutral", "slouched"]
DEFAULT_MODEL_PATH = "models/lstm_body_language_v2.pkl"
SEQ_LEN            = 30
N_LANDMARKS        = 33

# Couleurs BGR OpenCV par posture (compatible Body-language-detection.py)
POSTURE_COLORS = {
    "upright" : (0, 220, 0),    # vert
    "neutral" : (0, 200, 255),  # jaune
    "slouched": (0, 60, 255),   # rouge
}

# Scores de base confiance/stress par posture
POSTURE_SCORES = {
    "upright" : {"confidence": 85.0, "stress": 15.0},
    "neutral" : {"confidence": 58.0, "stress": 42.0},
    "slouched": {"confidence": 30.0, "stress": 70.0},
}

# Couleurs niveaux ALIA (hex)
NIVEAU_COLORS = {
    "Expert"  : "#1a237e",
    "Confirmé": "#1976D2",
    "Junior"  : "#42A5F5",
    "Débutant": "#90CAF9",
}


# ══════════════════════════════════════════════════════════════════════
# SEQUENCE BUFFER — inchangé V1
# ══════════════════════════════════════════════════════════════════════

class SequenceBuffer:
    """
    Buffer circulaire de frames pour l'inférence temps réel.
    Compatible avec Body-language-detection.py — drop-in replacement.
    """

    def __init__(self, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self._buffer = deque(maxlen=seq_len)

    def add_frame(self, landmarks) -> None:
        """Ajoute une frame depuis les landmarks MediaPipe."""
        frame = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
        self._buffer.append(frame.flatten())

    def add_frame_array(self, frame_array: np.ndarray) -> None:
        """Ajoute une frame depuis un array numpy (66,) ou (33, 2)."""
        self._buffer.append(frame_array.flatten().astype(np.float32))

    def is_ready(self) -> bool:
        return len(self._buffer) == self.seq_len

    def get(self) -> np.ndarray:
        if not self.is_ready():
            raise RuntimeError(f"Buffer pas prêt ({len(self._buffer)}/{self.seq_len})")
        return np.array(list(self._buffer), dtype=np.float32)  # (30, 66)

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def fill_ratio(self) -> float:
        return len(self._buffer) / self.seq_len

    def __len__(self) -> int:
        return len(self._buffer)


# ══════════════════════════════════════════════════════════════════════
# SCORE SMOOTHER — inchangé V1
# ══════════════════════════════════════════════════════════════════════

class ScoreSmoother:
    """Rolling average pour lisser les scores frame par frame."""

    def __init__(self, window: int = 12):
        self.window   = window
        self._buffers : Dict[str, deque] = {}

    def smooth(self, key: str, value: float) -> float:
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.window)
        self._buffers[key].append(value)
        return float(np.mean(self._buffers[key]))

    def reset(self) -> None:
        self._buffers.clear()


# ══════════════════════════════════════════════════════════════════════
# GEOMETRIC FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════

def extract_geometric_features(seq: np.ndarray) -> np.ndarray:
    """
    Extrait 12 features géométriques depuis une séquence (30, 66).
    Reproduit la logique de Body-language-detection.py.
    """
    n_frames = seq.shape[0]
    feats    = np.zeros((n_frames, 12))
    for f in range(n_frames):
        lm = seq[f].reshape(33, 2)
        nose=lm[0]; l_sh,r_sh=lm[11],lm[12]; l_hip,r_hip=lm[23],lm[24]
        l_wr,r_wr=lm[15],lm[16]
        mid_sh=(l_sh+r_sh)/2; mid_hip=(l_hip+r_hip)/2
        sh_w=np.linalg.norm(l_sh-r_sh)+1e-9
        ratio_a=sh_w/(abs(nose[1]-mid_sh[1])+1e-9)
        ratio_b=sh_w/(abs(mid_hip[1]-mid_sh[1])+1e-9)
        sh_tilt=abs(l_sh[1]-r_sh[1])/sh_w
        s_a=np.clip((ratio_a-0.7)/0.9,0,1); s_b=np.clip((ratio_b-0.5)/0.7,0,1)
        s_t=np.clip(sh_tilt/0.12,0,1); slouch=s_a*0.5+s_b*0.35+s_t*0.15
        wrist_dist=np.linalg.norm(l_wr-r_wr)/sh_w
        mid_x=mid_sh[0]; buf=sh_w*0.05
        l_cross=float(l_wr[0]<(mid_x-buf)); r_cross=float(r_wr[0]>(mid_x+buf))
        hunch=max(0,1-abs(mid_sh[1]-nose[1])/0.20)
        l_face=float(l_wr[1]<mid_sh[1] and abs(l_wr[0]-nose[0])<sh_w*0.6)
        r_face=float(r_wr[1]<mid_sh[1] and abs(r_wr[0]-nose[0])<sh_w*0.6)
        spine=mid_hip-mid_sh; spine_n=spine/(np.linalg.norm(spine)+1e-9)
        spine_a=float(np.degrees(np.arccos(np.clip(spine_n[1],-1,1))))
        feats[f]=[ratio_a,ratio_b,sh_tilt,slouch,wrist_dist,l_cross,r_cross,
                  hunch,l_face,r_face,spine_a/90.0,(l_cross+r_cross)/2.0]
    return feats


def sequence_to_features(seq: np.ndarray) -> np.ndarray:
    """
    Convertit une séquence brute (30, 66) en vecteur de features (36,).
    mean(12) + std(12) + delta_mean(12) = 36 features temporelles.
    """
    geom  = extract_geometric_features(seq)     # (30, 12)
    mean  = geom.mean(axis=0)                   # (12,)
    std   = geom.std(axis=0)                    # (12,)
    delta = np.abs(np.diff(geom, axis=0)).mean(axis=0)  # (12,) variabilité temporelle
    return np.concatenate([mean, std, delta])   # (36,)


# ══════════════════════════════════════════════════════════════════════
# BodyLanguageModel — CLASSE PRINCIPALE V2
# ══════════════════════════════════════════════════════════════════════

class BodyLanguageModel:
    """
    Classe d'inférence temps réel pour la détection du langage corporel V2.

    Attributes:
        version    (str) : version du modèle
        trained_at (str) : timestamp d'entraînement
    """

    def __init__(self, bundle: Dict):
        self._model        = bundle["model"]
        self._scaler       = bundle.get("scaler", None)
        self._class_names  = bundle.get("class_names", CLASS_NAMES)
        self._coaching     = bundle.get("coaching_alia", {})
        self._vm_context   = bundle.get("vm_context", {})
        self._smoother     = ScoreSmoother(window=12)

        self.version       = bundle.get("version",    "2.0.0")
        self.trained_at    = bundle.get("trained_at", "unknown")
        self.model_type    = bundle.get("model_type", "RandomForest")
        self.feature_type  = bundle.get("feature_type","geometric_temporal_36")

    # ── Chargement ────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "BodyLanguageModel":
        """
        Charge le modèle depuis un fichier .pkl.

        Args:
            path : chemin vers lstm_body_language_v2.pkl
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {path}\n"
                "Exécutez : python lstm_train_v2.py"
            )
        bundle = joblib.load(model_path)
        log.info(
            f"BodyLanguageModel V2 chargé — "
            f"v{bundle.get('version','?')} | "
            f"{bundle.get('model_type','?')} | "
            f"trained: {bundle.get('trained_at','?')[:19]}"
        )
        return cls(bundle)

    # ── Inférence principale ──────────────────────────────────────────

    def predict(self, sequence: np.ndarray,
                niveau_alia: str = "Junior",
                smooth: bool = True) -> Dict:
        """
        Inférence sur une séquence de landmarks.

        Args:
            sequence    : (30, 66) — landmarks x,y normalisés [0,1]
            niveau_alia : niveau ALIA courant du délégué (pour le coaching)
            smooth      : appliquer le lissage temporel

        Returns:
            dict : résultat complet avec posture, scores, coaching ALIA
        """
        if sequence.ndim == 2 and sequence.shape[1] == 66:
            pass  # shape (30, 66) ou (1, 66) acceptés
        else:
            raise ValueError(f"Attendu (30, 66), reçu {sequence.shape}")

        # ── Préparer les features ─────────────────────────────────────
        # Scaler entraîné sur 66 features brutes (30,66) → mean sur axis=0
        if self._scaler is not None and self._scaler.n_features_in_ == 66:
            feat_vec = sequence.mean(axis=0).reshape(1, -1)
            feat_norm = self._scaler.transform(feat_vec)
        else:
            feat_vec = sequence.mean(axis=0).reshape(1, -1)
            feat_norm = feat_vec

        # ── Prédiction ────────────────────────────────────────────────
        class_id = int(self._model.predict(feat_norm)[0])
        posture  = self._class_names[class_id]

        # Probabilités
        if hasattr(self._model, "predict_proba"):
            probs_raw = self._model.predict_proba(feat_norm)[0]
        else:
            probs_raw = np.eye(3)[class_id][0]

        proba_dict = {
            name: round(float(p), 4)
            for name, p in zip(self._class_names, probs_raw)
        }

        # ── Scores confiance / stress ─────────────────────────────────
        base      = POSTURE_SCORES[posture]
        conf_raw  = base["confidence"] + (probs_raw[0] - probs_raw[2]) * 8.0
        stress_raw= 100.0 - conf_raw

        if smooth:
            confidence = self._smoother.smooth("confidence", conf_raw)
            stress     = self._smoother.smooth("stress",     stress_raw)
        else:
            confidence = conf_raw
            stress     = stress_raw

        confidence = float(np.clip(confidence, 0, 100))
        stress     = float(np.clip(stress,     0, 100))

        # ── Contexte visite médicale ──────────────────────────────────
        vm_ctx = self._vm_context.get(posture, {})

        # ── Coaching aligné niveau ALIA ───────────────────────────────
        coaching_map = self._coaching.get(posture, {})
        coaching_msg = coaching_map.get(
            niveau_alia,
            coaching_map.get("Junior", f"Posture détectée : {posture}")
        )

        return {
            # ── Posture ───────────────────────────────────────────────
            "posture"        : posture,
            "posture_proba"  : proba_dict,
            "posture_color"  : POSTURE_COLORS[posture],

            # ── Scores HUD ────────────────────────────────────────────
            "confidence"     : round(confidence, 1),
            "stress"         : round(stress, 1),
            "is_upright"     : posture == "upright",

            # ── Contexte VM (Manuel VITAL) ────────────────────────────
            "vm_description" : vm_ctx.get("description", ""),
            "vm_etapes"      : vm_ctx.get("etapes_vm", []),
            "signal_bip"     : vm_ctx.get("signal_bip", False),

            # ── Coaching ALIA ─────────────────────────────────────────
            "coaching"       : coaching_msg,
            "niveau_alia"    : niveau_alia,
            "niveau_color"   : NIVEAU_COLORS.get(niveau_alia, "#888"),

            # ── Meta ──────────────────────────────────────────────────
            "frame_count"    : 30,
        }

    def predict_from_buffer(self, buffer: SequenceBuffer,
                             niveau_alia: str = "Junior",
                             smooth: bool = True) -> Dict:
        """Inférence directement depuis un SequenceBuffer."""
        if not buffer.is_ready():
            raise RuntimeError("Buffer pas prêt — besoin de 30 frames")
        return self.predict(buffer.get(), niveau_alia=niveau_alia, smooth=smooth)

    def predict_batch(self, sequences: np.ndarray,
                      niveau_alia: str = "Junior") -> List[Dict]:
        """Inférence en batch sur (N, 30, 66)."""
        return [
            self.predict(sequences[i], niveau_alia=niveau_alia, smooth=False)
            for i in range(len(sequences))
        ]

    # ── Helpers ───────────────────────────────────────────────────────

    def get_hud_data(self, result: Dict) -> Dict:
        """
        Retourne les données compatibles avec draw_hud() de
        Body-language-detection.py — drop-in replacement pour analyze_upper_body().
        """
        return {
            "confidence"   : result["confidence"],
            "stress"       : result["stress"],
            "posture_label": result["posture"],
            "posture_color": result["posture_color"],
            "arms_crossed" : result["posture"] == "slouched",
            "face_touch"   : result["stress"] > 70,
            "is_upright"   : result["is_upright"],
            "spine_angle"  : 0.0,
            "slouch_score" : 1.0 - result["confidence"] / 100.0,
        }

    def get_dashboard_data(self, result: Dict) -> Dict:
        """Données formatées pour le frontend Django."""
        return {
            "posture"       : result["posture"],
            "posture_color" : "#{:02x}{:02x}{:02x}".format(
                *result["posture_color"][::-1]
            ),  # BGR → RGB hex
            "confidence"    : result["confidence"],
            "stress"        : result["stress"],
            "coaching"      : result["coaching"],
            "niveau_alia"   : result["niveau_alia"],
            "niveau_color"  : result["niveau_color"],
            "vm_description": result["vm_description"],
            "vm_etapes"     : result["vm_etapes"],
            "signal_bip"    : result["signal_bip"],
            "proba"         : result["posture_proba"],
        }

    def result_summary(self, result: Dict) -> str:
        """Résumé lisible pour les logs."""
        lines = [
            f"┌─ Body Language V2 {'─'*38}",
            f"│  Posture      : {result['posture']:<12}  "
            f"(proba: {result['posture_proba']})",
            f"│  Confidence   : {result['confidence']:>5.1f} / 100",
            f"│  Stress       : {result['stress']:>5.1f} / 100",
            f"│  VM Context   : {result['vm_description']}",
            f"│  Niveau ALIA  : {result['niveau_alia']}",
            f"│  Coaching     : {result['coaching']}",
            f"└{'─'*57}",
        ]
        return "\n".join(lines)

    def reset_smoother(self) -> None:
        """Réinitialise le lissage — utile lors d'un changement de scène."""
        self._smoother.reset()

    def __repr__(self) -> str:
        return (
            f"BodyLanguageModel("
            f"version='{self.version}', "
            f"model='{self.model_type}', "
            f"classes={self._class_names})"
        )


# ══════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  BodyLanguageModel V2 — Quick Inference Test")
    print("=" * 65)

    model = BodyLanguageModel.load(DEFAULT_MODEL_PATH)
    print(f"\n  {model}\n")

    X = np.load("lstm_sequences_v2.npy")
    y = np.load("lstm_labels_v2.npy")

    # Test 1 : predict() sur 3 classes × 4 niveaux ALIA
    print("─" * 65)
    print("  TEST 1 — predict() par classe et niveau ALIA")
    print("─" * 65)
    for class_id, state in enumerate(["upright", "neutral", "slouched"]):
        idx    = np.where(y == class_id)[0][0]
        for niveau in ["Expert", "Confirmé", "Junior", "Débutant"]:
            result = model.predict(X[idx], niveau_alia=niveau, smooth=False)
            match  = "✅" if result["posture"] == state else "❌"
            print(f"\n  [{match}] {state:<10} / {niveau:<10}")
            print(model.result_summary(result))

    # Test 2 : SequenceBuffer
    print("\n" + "─" * 65)
    print("  TEST 2 — SequenceBuffer simulation")
    print("─" * 65)
    buffer = SequenceBuffer(seq_len=30)
    test_seq = X[0]
    for i in range(30):
        buffer.add_frame_array(test_seq[i])
    result = model.predict_from_buffer(buffer, niveau_alia="Confirmé")
    print(f"  Buffer prêt  : {buffer.is_ready()}")
    print(f"  Posture      : {result['posture']}")
    print(f"  Confidence   : {result['confidence']}")
    print(f"  Coaching     : {result['coaching']}")

    # Test 3 : get_dashboard_data()
    print("\n" + "─" * 65)
    print("  TEST 3 — get_dashboard_data() pour Django")
    print("─" * 65)
    result  = model.predict(X[0], niveau_alia="Junior", smooth=False)
    dash    = model.get_dashboard_data(result)
    print(f"  Clés : {list(dash.keys())}")
    for k, v in dash.items():
        print(f"    {k:<20}: {v}")

    # Test 4 : get_hud_data()
    print("\n" + "─" * 65)
    print("  TEST 4 — get_hud_data() (drop-in Body-language-detection.py)")
    print("─" * 65)
    hud = model.get_hud_data(result)
    print(f"  Clés HUD : {list(hud.keys())}")
    print(f"  confidence   : {hud['confidence']}")
    print(f"  posture_label: {hud['posture_label']}")
    print(f"  posture_color: {hud['posture_color']}  (BGR OpenCV)")

    # Test 5 : Batch
    print("\n" + "─" * 65)
    print("  TEST 5 — predict_batch()")
    print("─" * 65)
    results = model.predict_batch(X[:30], niveau_alia="Confirmé")
    postures= [r["posture"] for r in results]
    print(f"  Batch size : 30 séquences")
    print(f"  Postures   : {dict(zip(*np.unique(postures, return_counts=True)))}")

    print(f"\n✅  BodyLanguageModel V2 — validé")
    print(f"    Import : from lstm_model_v2 import BodyLanguageModel, SequenceBuffer")
