"""
lstm_train.py
=============
DSO 1 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Script d'entraînement du modèle LSTM pour la détection du langage corporel.

Remplace les règles géométriques fixes de Body-language-detection.py
par un modèle séquentiel appris sur les patterns temporels de landmarks.

Architecture :
    Input  : (batch, 30 frames, 66 features) — landmarks x,y MediaPipe
    LSTM 1 : 64 unités, return_sequences=True
    Dropout: 0.30
    LSTM 2 : 32 unités, return_sequences=False
    Dense  : 16 unités, ReLU
    Output : 3 classes (upright / neutral / slouched), Softmax

Pipeline :
    lstm_sequences.npy  →  Preprocessing  →  Training Loop (NumPy)
        →  Evaluation  →  models/lstm_body_language.pkl

Usage :
    python lstm_train.py                        # standard training
    python lstm_train.py --epochs 100           # custom epochs
    python lstm_train.py --input lstm_X_geom.npy  # geometric features (12)
    python lstm_train.py --eval-only            # load & evaluate only

Output :
    models/lstm_body_language.pkl   ← artefact principal
    models/lstm_training_report.json
    models/lstm_training.log

Note architecture :
    Implémenté en NumPy pur pour compatibilité offline.
    Pour entraînement GPU, remplacer LSTMNumpy par torch.nn.LSTM
    dans lstm_model.py — l'interface predict() reste identique.

Author  : CYBER SHADE — ALIA Project
Version : 1.0.0
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Scientific stack ──────────────────────────────────────────────────────
import joblib
import numpy as np

# ── Scikit-learn ──────────────────────────────────────────────────────────
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     f1_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ─────────────────────────────────────────────────────────────
    "sequences_path" : "lstm_sequences.npy",   # (N, 30, 66) raw landmarks
    "labels_path"    : "lstm_labels.npy",       # (N,)
    "models_dir"     : "models",
    "test_size"      : 0.20,
    "val_size"       : 0.15,                    # from train set
    "random_seed"    : 42,

    # ── Classes ───────────────────────────────────────────────────────────
    "class_names"    : ["upright", "neutral", "slouched"],
    "n_classes"      : 3,

    # ── Sequence ──────────────────────────────────────────────────────────
    "seq_len"        : 30,
    "n_landmarks"    : 33,
    "input_size"     : 66,       # 33 landmarks × 2 (x, y)

    # ── Architecture ─────────────────────────────────────────────────────
    "hidden1"        : 64,
    "hidden2"        : 32,
    "dense_units"    : 16,
    "dropout_rate"   : 0.30,

    # ── Training ──────────────────────────────────────────────────────────
    "epochs"         : 80,
    "batch_size"     : 32,
    "learning_rate"  : 0.001,
    "lr_decay"       : 0.95,          # multiplicative LR decay per epoch
    "lr_decay_every" : 10,            # decay every N epochs
    "early_stop_patience": 15,        # stop if no improvement for N epochs
    "clip_grad_norm" : 5.0,           # gradient clipping threshold
}

CLASS_NAMES = CONFIG["class_names"]
SEED        = CONFIG["random_seed"]


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(CONFIG["models_dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(CONFIG["models_dir"]) / "lstm_training.log", mode="w"
        ),
    ],
)
log = logging.getLogger("LSTMTrain")


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC FEATURE EXTRACTOR
# (Reproduit la logique de Body-language-detection.py en vecteurs numpy)
# ══════════════════════════════════════════════════════════════════════════════

def extract_geometric_features(seq: np.ndarray) -> np.ndarray:
    """
    Extrait 12 features géométriques depuis une séquence de landmarks.
    Inspiré directement de analyze_upper_body() dans Body-language-detection.py.

    Args:
        seq : (n_frames, 66) — landmarks x,y aplatis

    Returns:
        feats : (n_frames, 12)
    """
    n_frames = seq.shape[0]
    feats    = np.zeros((n_frames, 12))

    for f in range(n_frames):
        lm = seq[f].reshape(33, 2)

        nose             = lm[0]
        l_sh, r_sh       = lm[11], lm[12]
        l_hip, r_hip     = lm[23], lm[24]
        l_wr, r_wr       = lm[15], lm[16]

        mid_sh   = (l_sh + r_sh)   / 2
        mid_hip  = (l_hip + r_hip) / 2
        sh_w     = np.linalg.norm(l_sh - r_sh) + 1e-9

        # 1–2. Slouch ratios (core signal)
        ratio_a = sh_w / (abs(nose[1] - mid_sh[1]) + 1e-9)
        ratio_b = sh_w / (abs(mid_hip[1] - mid_sh[1]) + 1e-9)

        # 3. Shoulder tilt
        sh_tilt = abs(l_sh[1] - r_sh[1]) / sh_w

        # 4. Composite slouch score
        s_a    = np.clip((ratio_a - 0.7) / 0.9, 0, 1)
        s_b    = np.clip((ratio_b - 0.5) / 0.7, 0, 1)
        s_t    = np.clip(sh_tilt / 0.12, 0, 1)
        slouch = s_a * 0.50 + s_b * 0.35 + s_t * 0.15

        # 5. Wrist distance (normalised)
        wrist_dist = np.linalg.norm(l_wr - r_wr) / sh_w

        # 6–7. Arms crossed (left / right)
        mid_x   = mid_sh[0]
        buf     = sh_w * 0.05
        l_cross = float(l_wr[0] < (mid_x - buf))
        r_cross = float(r_wr[0] > (mid_x + buf))

        # 8. Hunch score
        hunch = max(0.0, 1.0 - abs(mid_sh[1] - nose[1]) / 0.20)

        # 9–10. Face touch (left / right)
        l_face = float(l_wr[1] < mid_sh[1] and
                       abs(l_wr[0] - nose[0]) < sh_w * 0.6)
        r_face = float(r_wr[1] < mid_sh[1] and
                       abs(r_wr[0] - nose[0]) < sh_w * 0.6)

        # 11. Spine angle (normalised to [0,1])
        spine   = mid_hip - mid_sh
        spine_n = spine / (np.linalg.norm(spine) + 1e-9)
        spine_a = float(np.degrees(np.arccos(np.clip(spine_n[1], -1, 1))))

        # 12. Arms crossed composite score
        cross_score = (l_cross + r_cross) / 2.0

        feats[f] = [ratio_a, ratio_b, sh_tilt, slouch,
                    wrist_dist, l_cross, r_cross, hunch,
                    l_face, r_face, spine_a / 90.0, cross_score]

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# LSTM IMPLEMENTATION — NumPy
# ══════════════════════════════════════════════════════════════════════════════

class LSTMCell:
    """
    Single LSTM cell — NumPy implementation with Backpropagation Through Time.
    Équivalent à torch.nn.LSTMCell en forward pass.
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng   = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (input_size + hidden_size))   # Xavier init

        # Combined weight matrix for all 4 gates: [i, f, g, o]
        self.Wx = rng.randn(input_size,   4 * hidden_size) * scale
        self.Wh = rng.randn(hidden_size,  4 * hidden_size) * scale
        self.b  = np.zeros(4 * hidden_size)
        # Forget gate bias initialised to 1 (reduces vanishing gradient)
        self.b[hidden_size:2*hidden_size] = 1.0

        self.hidden_size = hidden_size

        # Adam optimizer states
        self.mWx = np.zeros_like(self.Wx)
        self.vWx = np.zeros_like(self.Wx)
        self.mWh = np.zeros_like(self.Wh)
        self.vWh = np.zeros_like(self.Wh)
        self.mb  = np.zeros_like(self.b)
        self.vb  = np.zeros_like(self.b)

    def forward(self, x: np.ndarray,
                h_prev: np.ndarray,
                c_prev: np.ndarray) -> Tuple:
        """
        Forward pass — single time step.

        Args:
            x      : (input_size,)
            h_prev : (hidden_size,)
            c_prev : (hidden_size,)

        Returns:
            h_next, c_next, cache (for backward pass)
        """
        H     = self.hidden_size
        gates = x @ self.Wx + h_prev @ self.Wh + self.b   # (4H,)

        i = self._sigmoid(gates[:H])        # input gate
        f = self._sigmoid(gates[H:2*H])     # forget gate
        g = np.tanh(gates[2*H:3*H])         # cell gate
        o = self._sigmoid(gates[3*H:])      # output gate

        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)

        cache = (x, h_prev, c_prev, i, f, g, o, c_next, gates)
        return h_next, c_next, cache

    def backward(self, dh: np.ndarray, dc: np.ndarray,
                 cache: tuple, clip: float = 5.0) -> Tuple:
        """
        Backward pass — BPTT single step.

        Returns:
            dx, dh_prev, dc_prev, dWx, dWh, db
        """
        x, h_prev, c_prev, i, f, g, o, c_next, gates = cache
        H = self.hidden_size

        tanh_c = np.tanh(c_next)
        do     = dh * tanh_c
        dc_    = dh * o * (1 - tanh_c ** 2) + dc

        di = dc_ * g
        df = dc_ * c_prev
        dg = dc_ * i
        dc_prev = dc_ * f

        # Gate gradients through activations
        di_raw = di * i * (1 - i)
        df_raw = df * f * (1 - f)
        dg_raw = dg * (1 - g ** 2)
        do_raw = do * o * (1 - o)

        dgates = np.concatenate([di_raw, df_raw, dg_raw, do_raw])  # (4H,)

        dWx    = np.outer(x, dgates)
        dWh    = np.outer(h_prev, dgates)
        db     = dgates
        dx     = self.Wx @ dgates
        dh_prev = self.Wh @ dgates

        # Gradient clipping
        for d in [dWx, dWh, db, dx, dh_prev, dc_prev]:
            np.clip(d, -clip, clip, out=d)

        return dx, dh_prev, dc_prev, dWx, dWh, db

    def adam_update(self, dWx, dWh, db,
                    lr: float, t: int,
                    beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8):
        """Adam optimizer update for this cell's parameters."""
        self.mWx = beta1 * self.mWx + (1 - beta1) * dWx
        self.vWx = beta2 * self.vWx + (1 - beta2) * dWx ** 2
        self.mWh = beta1 * self.mWh + (1 - beta1) * dWh
        self.vWh = beta2 * self.vWh + (1 - beta2) * dWh ** 2
        self.mb  = beta1 * self.mb  + (1 - beta1) * db
        self.vb  = beta2 * self.vb  + (1 - beta2) * db ** 2

        mWx_c = self.mWx / (1 - beta1 ** t)
        vWx_c = self.vWx / (1 - beta2 ** t)
        mWh_c = self.mWh / (1 - beta1 ** t)
        vWh_c = self.vWh / (1 - beta2 ** t)
        mb_c  = self.mb  / (1 - beta1 ** t)
        vb_c  = self.vb  / (1 - beta2 ** t)

        self.Wx -= lr * mWx_c / (np.sqrt(vWx_c) + eps)
        self.Wh -= lr * mWh_c / (np.sqrt(vWh_c) + eps)
        self.b  -= lr * mb_c  / (np.sqrt(vb_c)  + eps)

    @staticmethod
    def _sigmoid(x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))


class BodyLanguageLSTM:
    """
    2-layer LSTM classifier for body language posture detection.

    Architecture :
        LSTM Layer 1 : hidden1=64 units
        Dropout      : 0.30 (applied during training)
        LSTM Layer 2 : hidden2=32 units
        Dense Layer  : 16 units, ReLU
        Output Layer : n_classes=3, Softmax

    Equivalent PyTorch :
        self.lstm1  = nn.LSTM(input_size, 64,  batch_first=True)
        self.drop   = nn.Dropout(0.3)
        self.lstm2  = nn.LSTM(64, 32, batch_first=True)
        self.fc1    = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, n_classes)
    """

    def __init__(self, input_size: int = 66, hidden1: int = 64,
                 hidden2: int = 32, dense: int = 16,
                 n_classes: int = 3, seed: int = 42,
                 dropout: float = 0.30):

        self.hidden1   = hidden1
        self.hidden2   = hidden2
        self.n_classes = n_classes
        self.dropout   = dropout
        self.input_size = input_size

        rng   = np.random.RandomState(seed)
        s1    = np.sqrt(2.0 / (hidden1 + dense))
        s2    = np.sqrt(2.0 / (dense + n_classes))

        # LSTM layers
        self.lstm1 = LSTMCell(input_size, hidden1, seed)
        self.lstm2 = LSTMCell(hidden1,    hidden2, seed + 1)

        # Dense layers
        self.W_fc1    = rng.randn(hidden2, dense)     * s1
        self.b_fc1    = np.zeros(dense)
        self.W_out    = rng.randn(dense,   n_classes) * s2
        self.b_out    = np.zeros(n_classes)

        # Adam states for dense layers
        self.m_W_fc1 = np.zeros_like(self.W_fc1)
        self.v_W_fc1 = np.zeros_like(self.W_fc1)
        self.m_b_fc1 = np.zeros_like(self.b_fc1)
        self.v_b_fc1 = np.zeros_like(self.b_fc1)
        self.m_W_out = np.zeros_like(self.W_out)
        self.v_W_out = np.zeros_like(self.W_out)
        self.m_b_out = np.zeros_like(self.b_out)
        self.v_b_out = np.zeros_like(self.b_out)

    def forward(self, seq: np.ndarray,
                training: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Forward pass for one sequence.

        Args:
            seq      : (n_frames, input_size)
            training : if True, apply dropout

        Returns:
            probs  : (n_classes,) — softmax probabilities
            cache  : dict with all intermediate values for backward
        """
        n_frames = seq.shape[0]

        h1 = np.zeros(self.hidden1)
        c1 = np.zeros(self.hidden1)
        h2 = np.zeros(self.hidden2)
        c2 = np.zeros(self.hidden2)

        caches1, caches2 = [], []
        h1_seq = []

        # ── Layer 1 ───────────────────────────────────────────────────────
        for t in range(n_frames):
            h1, c1, cache1 = self.lstm1.forward(seq[t], h1, c1)
            caches1.append(cache1)
            h1_seq.append(h1.copy())

        # ── Dropout on layer 1 outputs ────────────────────────────────────
        if training and self.dropout > 0:
            mask = (np.random.rand(*h1.shape) > self.dropout).astype(float)
            mask /= (1.0 - self.dropout + 1e-8)   # inverted dropout
        else:
            mask = np.ones(self.hidden1)

        # ── Layer 2 ───────────────────────────────────────────────────────
        for t in range(n_frames):
            h1_dropped = h1_seq[t] * mask
            h2, c2, cache2 = self.lstm2.forward(h1_dropped, h2, c2)
            caches2.append(cache2)

        # ── Dense layers ──────────────────────────────────────────────────
        fc1_in  = h2                                   # (hidden2,)
        fc1_out = np.maximum(0, fc1_in @ self.W_fc1 + self.b_fc1)  # ReLU
        logits  = fc1_out @ self.W_out + self.b_out    # (n_classes,)
        probs   = self._softmax(logits)                # (n_classes,)

        cache = {
            "seq"     : seq,
            "caches1" : caches1,
            "caches2" : caches2,
            "mask"    : mask,
            "h1_seq"  : h1_seq,
            "h2"      : h2,
            "fc1_in"  : fc1_in,
            "fc1_out" : fc1_out,
            "logits"  : logits,
            "probs"   : probs,
        }
        return probs, cache

    def backward(self, probs: np.ndarray, y_true: int,
                 cache: dict, clip: float = 5.0) -> float:
        """
        Backward pass — cross-entropy loss + BPTT.

        Returns:
            loss (float) — cross-entropy for this sample
        """
        n_frames = len(cache["caches1"])

        # ── Cross-entropy loss + gradient ─────────────────────────────────
        loss    = -np.log(probs[y_true] + 1e-9)
        dlogits = probs.copy()
        dlogits[y_true] -= 1.0          # dL/dlogits (softmax + CE gradient)

        # ── Dense layer backward ──────────────────────────────────────────
        dW_out = np.outer(cache["fc1_out"], dlogits)
        db_out = dlogits
        dfc1   = self.W_out @ dlogits

        # ReLU backward
        dfc1_relu = dfc1 * (cache["fc1_out"] > 0)
        dW_fc1    = np.outer(cache["fc1_in"], dfc1_relu)
        db_fc1    = dfc1_relu
        dh2       = self.W_fc1 @ dfc1_relu

        # Clip dense gradients
        for d in [dW_out, db_out, dW_fc1, db_fc1, dh2]:
            np.clip(d, -clip, clip, out=d)

        # ── LSTM Layer 2 BPTT ─────────────────────────────────────────────
        dc2 = np.zeros(self.hidden2)
        dh1_total = np.zeros(self.hidden1)

        for t in reversed(range(n_frames)):
            dx2, dh2_prev, dc2, dWx2, dWh2, db2 = self.lstm2.backward(
                dh2, dc2, cache["caches2"][t], clip
            )
            dh2 = dh2_prev
            dh1_total += dx2 / n_frames   # accumulate gradient to layer 1

        # Dropout backward
        dh1_total *= cache["mask"]

        # ── LSTM Layer 1 BPTT ─────────────────────────────────────────────
        dc1 = np.zeros(self.hidden1)
        for t in reversed(range(n_frames)):
            _, dh1_prev, dc1, dWx1, dWh1, db1 = self.lstm1.backward(
                dh1_total, dc1, cache["caches1"][t], clip
            )
            dh1_total = dh1_prev

        return loss, dW_fc1, db_fc1, dW_out, db_out, dWx1, dWh1, db1, dWx2, dWh2, db2

    def predict_proba(self, seq: np.ndarray) -> np.ndarray:
        """Predict class probabilities for one sequence."""
        probs, _ = self.forward(seq, training=False)
        return probs

    def predict(self, seq: np.ndarray) -> int:
        """Predict class label for one sequence."""
        return int(np.argmax(self.predict_proba(seq)))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for a batch of sequences."""
        return np.array([self.predict(X[i]) for i in range(len(X))])

    def count_params(self) -> int:
        total  = self.lstm1.Wx.size + self.lstm1.Wh.size + self.lstm1.b.size
        total += self.lstm2.Wx.size + self.lstm2.Wh.size + self.lstm2.b.size
        total += self.W_fc1.size + self.b_fc1.size
        total += self.W_out.size + self.b_out.size
        return total

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Loads and preprocesses LSTM sequences."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.load(self.cfg["sequences_path"])    # (N, 30, 66)
        y = np.load(self.cfg["labels_path"])        # (N,)

        log.info(f"Data loaded : X={X.shape} | y={y.shape}")
        log.info(f"Classes     : { {i: (y==i).sum() for i in range(3)} }")
        log.info(f"Value range : [{X.min():.3f}, {X.max():.3f}]")
        return X, y

    def preprocess(self, X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """
        Normalize sequences using StandardScaler fitted on training data.
        Applied per-feature across all frames and sequences.
        """
        N, T, F = X.shape
        scaler  = StandardScaler()
        X_flat  = X.reshape(-1, F)
        X_norm  = scaler.fit_transform(X_flat).reshape(N, T, F)
        log.info(f"Normalized  : mean≈{X_norm.mean():.4f} std≈{X_norm.std():.4f}")
        return X_norm, scaler

    def split(self, X, y) -> Tuple:
        """Train / Val / Test split with stratification."""
        seed = self.cfg["random_seed"]
        ts   = self.cfg["test_size"]
        vs   = self.cfg["val_size"]

        # Test split
        X_tv, X_te, y_tv, y_te = train_test_split(
            X, y, test_size=ts, random_state=seed, stratify=y)

        # Val split from train
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tv, y_tv, test_size=vs, random_state=seed, stratify=y_tv)

        log.info(f"Split : train={len(X_tr)} | val={len(X_val)} | test={len(X_te)}")
        return X_tr, X_val, X_te, y_tr, y_val, y_te

    def make_batches(self, X, y, batch_size: int) -> List[Tuple]:
        """Yield (X_batch, y_batch) tuples with shuffling."""
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            b = idx[start:start + batch_size]
            yield X[b], y[b]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class LSTMTrainer:
    """
    Orchestrates LSTM training with:
    - Mini-batch gradient descent (Adam optimizer)
    - Learning rate decay
    - Early stopping
    - Train / Val / Test evaluation
    - Model checkpointing
    """

    def __init__(self, model: BodyLanguageLSTM, cfg: Dict):
        self.model     = model
        self.cfg       = cfg
        self.history   = {"train_loss": [], "val_loss": [],
                          "train_acc":  [], "val_acc":  []}
        self.best_val_loss = np.inf
        self.patience_counter = 0
        self.t_adam    = 0    # Adam step counter
        self.report    = {}

    # ── Epoch ─────────────────────────────────────────────────────────────
    def _run_epoch(self, X: np.ndarray, y: np.ndarray,
                   lr: float, training: bool = True) -> Tuple[float, float]:
        """Run one epoch — returns (mean_loss, accuracy)."""
        total_loss = 0.0
        correct    = 0
        n          = len(X)

        indices = np.random.permutation(n) if training else np.arange(n)

        for i in indices:
            probs, cache = self.model.forward(X[i], training=training)
            pred = int(np.argmax(probs))
            correct += int(pred == y[i])

            if training:
                self.t_adam += 1
                (loss, dW_fc1, db_fc1, dW_out, db_out,
                 dWx1, dWh1, db1, dWx2, dWh2, db2) = self.model.backward(
                    probs, int(y[i]), cache, self.cfg["clip_grad_norm"])

                total_loss += loss

                # ── Adam updates ──────────────────────────────────────────
                t = self.t_adam
                _b1, _b2 = 0.9, 0.999
                _eps, _lr = 1e-8, lr

                # LSTM 1
                self.model.lstm1.adam_update(dWx1, dWh1, db1, _lr, t)
                # LSTM 2
                self.model.lstm2.adam_update(dWx2, dWh2, db2, _lr, t)
                # Dense layers (inline Adam)
                self._adam_dense(dW_fc1, db_fc1, dW_out, db_out, _lr, t)
            else:
                total_loss += -np.log(probs[y[i]] + 1e-9)

        return total_loss / n, correct / n

    def _adam_dense(self, dW_fc1, db_fc1, dW_out, db_out,
                    lr: float, t: int,
                    b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        """Adam update for dense layer parameters."""
        m = self.model

        m.m_W_fc1 = b1*m.m_W_fc1 + (1-b1)*dW_fc1
        m.v_W_fc1 = b2*m.v_W_fc1 + (1-b2)*dW_fc1**2
        m.m_b_fc1 = b1*m.m_b_fc1 + (1-b1)*db_fc1
        m.v_b_fc1 = b2*m.v_b_fc1 + (1-b2)*db_fc1**2

        m.m_W_out = b1*m.m_W_out + (1-b1)*dW_out
        m.v_W_out = b2*m.v_W_out + (1-b2)*dW_out**2
        m.m_b_out = b1*m.m_b_out + (1-b1)*db_out
        m.v_b_out = b2*m.v_b_out + (1-b2)*db_out**2

        def _apply(mw, vw, w, mb, vb, b):
            mc = mw / (1-b1**t); vc = vw / (1-b2**t)
            w -= lr * mc / (np.sqrt(vc) + eps)
            mc2 = mb / (1-b1**t); vc2 = vb / (1-b2**t)
            b  -= lr * mc2 / (np.sqrt(vc2) + eps)

        _apply(m.m_W_fc1, m.v_W_fc1, m.W_fc1,
               m.m_b_fc1, m.v_b_fc1, m.b_fc1)
        _apply(m.m_W_out, m.v_W_out, m.W_out,
               m.m_b_out, m.v_b_out, m.b_out)

    # ── Full training loop ────────────────────────────────────────────────
    def train(self, X_tr, y_tr, X_val, y_val) -> Dict:
        """
        Full training loop with early stopping and LR decay.

        Returns:
            training history dict
        """
        cfg     = self.cfg
        lr      = cfg["learning_rate"]
        best_model_state = None

        log.info("=" * 60)
        log.info("  LSTM Training Loop — Adam optimizer")
        log.info(f"  Epochs={cfg['epochs']} | LR={lr} | "
                 f"EarlyStop={cfg['early_stop_patience']}")
        log.info("=" * 60)
        log.info(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  "
                 f"{'Train Acc':>10}  {'Val Acc':>9}  {'LR':>9}")
        log.info("  " + "-" * 62)

        t_start = time.time()

        for epoch in range(1, cfg["epochs"] + 1):

            # ── LR decay ──────────────────────────────────────────────────
            if epoch > 1 and (epoch - 1) % cfg["lr_decay_every"] == 0:
                lr *= cfg["lr_decay"]

            # ── Train ─────────────────────────────────────────────────────
            tr_loss, tr_acc = self._run_epoch(X_tr, y_tr, lr, training=True)

            # ── Validate ──────────────────────────────────────────────────
            va_loss, va_acc = self._run_epoch(X_val, y_val, lr, training=False)

            # ── Log ───────────────────────────────────────────────────────
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)

            if epoch % 5 == 0 or epoch == 1:
                log.info(f"  {epoch:>6}  {tr_loss:>11.4f}  {va_loss:>10.4f}  "
                         f"{tr_acc*100:>9.2f}%  {va_acc*100:>8.2f}%  {lr:>9.6f}")

            # ── Early stopping ────────────────────────────────────────────
            if va_loss < self.best_val_loss - 1e-4:
                self.best_val_loss    = va_loss
                self.patience_counter = 0
                best_model_state      = self._copy_weights()
            else:
                self.patience_counter += 1

            if self.patience_counter >= cfg["early_stop_patience"]:
                log.info(f"  Early stopping at epoch {epoch} "
                         f"(best val_loss={self.best_val_loss:.4f})")
                break

        elapsed = time.time() - t_start
        log.info(f"\n  Training complete in {elapsed:.1f}s")
        log.info(f"  Best val loss : {self.best_val_loss:.4f}")

        # Restore best weights
        if best_model_state is not None:
            self._load_weights(best_model_state)
            log.info("  [OK]  Best weights restored")

        return self.history

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(self, X_test, y_test, split_name: str = "Test") -> Dict:
        """Full evaluation on a split — returns metrics dict."""
        log.info("=" * 60)
        log.info(f"  Evaluation — {split_name} Set ({len(X_test)} samples)")
        log.info("=" * 60)

        y_pred = self.model.predict_batch(X_test)
        y_prob = np.array([self.model.predict_proba(X_test[i])
                           for i in range(len(X_test))])

        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        f1_wt  = f1_score(y_test, y_pred, average="weighted")
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

        log.info(f"  Accuracy        : {acc*100:.2f}%")
        log.info(f"  F1-Macro        : {f1_mac:.4f}")
        log.info(f"  F1-Weighted     : {f1_wt:.4f}")
        log.info(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)}")
        log.info(f"  Confusion matrix :\n{cm}")

        return {
            "accuracy"   : float(acc),
            "f1_macro"   : float(f1_mac),
            "f1_weighted": float(f1_wt),
            "per_class"  : {cls: report[cls] for cls in CLASS_NAMES},
            "confusion_matrix": cm.tolist(),
        }

    # ── Weight persistence ────────────────────────────────────────────────
    def _copy_weights(self) -> Dict:
        m = self.model
        return {
            "lstm1_Wx": m.lstm1.Wx.copy(), "lstm1_Wh": m.lstm1.Wh.copy(),
            "lstm1_b" : m.lstm1.b.copy(),
            "lstm2_Wx": m.lstm2.Wx.copy(), "lstm2_Wh": m.lstm2.Wh.copy(),
            "lstm2_b" : m.lstm2.b.copy(),
            "W_fc1"   : m.W_fc1.copy(),    "b_fc1"   : m.b_fc1.copy(),
            "W_out"   : m.W_out.copy(),    "b_out"   : m.b_out.copy(),
        }

    def _load_weights(self, state: Dict):
        m = self.model
        m.lstm1.Wx = state["lstm1_Wx"]; m.lstm1.Wh = state["lstm1_Wh"]
        m.lstm1.b  = state["lstm1_b"]
        m.lstm2.Wx = state["lstm2_Wx"]; m.lstm2.Wh = state["lstm2_Wh"]
        m.lstm2.b  = state["lstm2_b"]
        m.W_fc1    = state["W_fc1"];    m.b_fc1    = state["b_fc1"]
        m.W_out    = state["W_out"];    m.b_out    = state["b_out"]

    # ── Save ──────────────────────────────────────────────────────────────
    def save(self, scaler: StandardScaler, test_metrics: Dict):
        mdir = Path(self.cfg["models_dir"])

        bundle = {
            "model"        : self.model,
            "scaler"       : scaler,
            "config"       : self.cfg,
            "history"      : self.history,
            "class_names"  : CLASS_NAMES,
            "trained_at"   : datetime.now().isoformat(),
            "version"      : "1.0.0",
        }
        joblib.dump(bundle, mdir / "lstm_body_language.pkl")
        log.info(f"  [OK]  Model saved → {mdir}/lstm_body_language.pkl")

        # Training report
        report = {
            "test_metrics" : test_metrics,
            "best_val_loss": float(self.best_val_loss),
            "history"      : {k: [float(v) for v in vals]
                              for k, vals in self.history.items()},
            "model_params" : self.model.count_params(),
            "trained_at"   : datetime.now().isoformat(),
        }
        with open(mdir / "lstm_training_report.json", "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"  [OK]  Report saved → {mdir}/lstm_training_report.json")

        # Artifact summary
        log.info("\n  Saved artifacts :")
        for p in sorted(mdir.iterdir()):
            log.info(f"    {p.name:<45}  {p.stat().st_size/1024:>7.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_training(cfg: Dict, eval_only: bool = False):
    np.random.seed(cfg["random_seed"])
    t_start = time.time()

    log.info("=" * 70)
    log.info("  ALIA — LSTM Body Language Model Training Pipeline")
    log.info(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Data     : {cfg['sequences_path']}")
    log.info(f"  Epochs   : {cfg['epochs']} | LR : {cfg['learning_rate']}")
    log.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    loader          = DataLoader(cfg)
    X_raw, y        = loader.load()
    X_norm, scaler  = loader.preprocess(X_raw)
    X_tr, X_val, X_te, y_tr, y_val, y_te = loader.split(X_norm, y)

    if eval_only:
        log.info("Eval-only mode — loading saved model...")
        bundle  = joblib.load(Path(cfg["models_dir"]) / "lstm_body_language.pkl")
        model   = bundle["model"]
        trainer = LSTMTrainer(model, cfg)
        trainer.evaluate(X_te, y_te, "Test")
        _run_demo(model, scaler, X_te, y_te)
        return

    # ── Build model ───────────────────────────────────────────────────────
    model = BodyLanguageLSTM(
        input_size = cfg["input_size"],
        hidden1    = cfg["hidden1"],
        hidden2    = cfg["hidden2"],
        dense      = cfg["dense_units"],
        n_classes  = cfg["n_classes"],
        seed       = cfg["random_seed"],
        dropout    = cfg["dropout_rate"],
    )
    log.info(f"\n  Model architecture :")
    log.info(f"    Input      : (seq_len={cfg['seq_len']}, features={cfg['input_size']})")
    log.info(f"    LSTM 1     : {cfg['hidden1']} units")
    log.info(f"    Dropout    : {cfg['dropout_rate']}")
    log.info(f"    LSTM 2     : {cfg['hidden2']} units")
    log.info(f"    Dense      : {cfg['dense_units']} units (ReLU)")
    log.info(f"    Output     : {cfg['n_classes']} classes (Softmax)")
    log.info(f"    Parameters : {model.count_params():,}")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = LSTMTrainer(model, cfg)
    trainer.train(X_tr, y_tr, X_val, y_val)

    # ── Evaluate ──────────────────────────────────────────────────────────
    log.info("\n  Final evaluation :")
    train_metrics = trainer.evaluate(X_tr,  y_tr,  "Train")
    val_metrics   = trainer.evaluate(X_val, y_val, "Val")
    test_metrics  = trainer.evaluate(X_te,  y_te,  "Test")

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save(scaler, test_metrics)

    # ── Demo ──────────────────────────────────────────────────────────────
    _run_demo(model, scaler, X_te, y_te)

    log.info(f"\n[OK]  Pipeline complete in {time.time()-t_start:.1f}s")


def _run_demo(model: BodyLanguageLSTM, scaler: StandardScaler,
              X_te: np.ndarray, y_te: np.ndarray):
    """Quick inference demo on 3 test samples."""
    log.info("\n" + "=" * 60)
    log.info("  INFERENCE DEMO — 3 test samples")
    log.info("=" * 60)
    for i in range(min(3, len(X_te))):
        probs = model.predict_proba(X_te[i])
        pred  = int(np.argmax(probs))
        true  = int(y_te[i])
        match = "[OK]" if pred == true else "❌"
        log.info(f"  [{match}] true={CLASS_NAMES[true]:<10} "
                 f"pred={CLASS_NAMES[pred]:<10} "
                 f"probs={probs.round(3)}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="ALIA LSTM Body Language — Training Pipeline")
    p.add_argument("--input",      default=CONFIG["sequences_path"],
                   help="Path to sequences .npy file")
    p.add_argument("--labels",     default=CONFIG["labels_path"],
                   help="Path to labels .npy file")
    p.add_argument("--epochs",     type=int, default=CONFIG["epochs"])
    p.add_argument("--lr",         type=float, default=CONFIG["learning_rate"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--models-dir", default=CONFIG["models_dir"])
    p.add_argument("--eval-only",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["sequences_path"] = args.input
    CONFIG["labels_path"]    = args.labels
    CONFIG["epochs"]         = args.epochs
    CONFIG["learning_rate"]  = args.lr
    CONFIG["batch_size"]     = args.batch_size
    CONFIG["models_dir"]     = args.models_dir

    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    run_training(CONFIG, eval_only=args.eval_only)
