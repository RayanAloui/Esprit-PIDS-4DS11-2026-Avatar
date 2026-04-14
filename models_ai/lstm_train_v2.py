"""
lstm_train_v2.py
================
DSO 1 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Script d'entraînement du modèle LSTM Body Language V2.

Nouveautés V2 vs V1 :
    - Dataset enrichi (500 séquences, métadonnées ALIA)
    - Feedback coaching aligné sur les 4 niveaux ALIA
    - Contexte visite médicale (étape VM, description posture)
    - Output predict() enrichi (niveau_alia, coaching, vm_context)

Architecture (inchangée — validée V1) :
    Input   : (batch, 30 frames, 66 features)
    LSTM 1  : 64 unités, Xavier init, forget bias=1
    Dropout : 0.30
    LSTM 2  : 32 unités
    Dense   : 16 unités, ReLU
    Output  : 3 classes (upright / neutral / slouched), Softmax
    Params  : ~46 500

Usage :
    python lstm_train_v2.py                   # entraînement standard
    python lstm_train_v2.py --epochs 80       # custom epochs
    python lstm_train_v2.py --eval-only       # évaluation seule

Output :
    models/lstm_body_language_v2.pkl
    models/lstm_training_report_v2.json
    models/lstm_training_v2.log

Author  : CYBER SHADE — ALIA Project
Version : 2.0.0
"""

import argparse, json, logging, os, sys, time, warnings
from datetime import datetime
from pathlib  import Path
from typing   import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     f1_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing   import StandardScaler

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

CONFIG = {
    "sequences_path": "lstm_sequences_v2.npy",
    "labels_path"   : "lstm_labels_v2.npy",
    "models_dir"    : "models",
    "test_size"     : 0.20,
    "val_size"      : 0.15,
    "random_seed"   : 42,
    "class_names"   : ["upright", "neutral", "slouched"],
    "n_classes"     : 3,
    "seq_len"       : 30,
    "input_size"    : 66,
    "hidden1"       : 64,
    "hidden2"       : 32,
    "dense_units"   : 16,
    "dropout_rate"  : 0.30,
    "epochs"        : 80,
    "learning_rate" : 0.001,
    "lr_decay"      : 0.95,
    "lr_decay_every": 10,
    "early_stop_patience": 15,
    "clip_grad_norm": 5.0,
}

CLASS_NAMES = CONFIG["class_names"]
SEED        = CONFIG["random_seed"]

# Feedback coaching par posture ET niveau ALIA (Manuel VITAL + Référentiel)
COACHING_ALIA = {
    "upright": {
        "Expert"   : "Posture exemplaire — maintenez cet engagement corporel pendant le closing.",
        "Confirmé" : "Excellente posture — utilisez cette ouverture pour renforcer votre argumentation.",
        "Junior"   : "Bonne posture — continuez ainsi pour projeter confiance et crédibilité.",
        "Débutant" : "Posture correcte ✅ — maintenez-la pendant toute la durée de la visite.",
    },
    "neutral": {
        "Expert"   : "Posture neutre acceptable — lors de l'argumentation, ouvrez davantage les épaules.",
        "Confirmé" : "Posture correcte — pensez à vous redresser lors de l'étape argumentation (étape 5).",
        "Junior"   : "Posture convenable — ouvrez les épaules pour projeter plus de confiance.",
        "Débutant" : "Posture acceptable — redressez-vous légèrement pour paraître plus confiant.",
    },
    "slouched": {
        "Expert"   : "⚠️ Posture fermée détectée — incompatible avec le niveau Expert. Corrigez immédiatement.",
        "Confirmé" : "⚠️ Posture voûtée — signal de stress visible. Redressez-vous et ouvrez les bras.",
        "Junior"   : "⚠️ Posture à corriger — voûtement et bras croisés diminuent votre crédibilité.",
        "Débutant" : "⚠️ Posture fermée — redressez les épaules et gardez les mains visibles.",
    },
}

# Contexte visite médicale par posture
VM_CONTEXT = {
    "upright" : {
        "description" : "Posture de confiance — argumentation et closing (étapes 5–6)",
        "etapes_vm"   : [5, 6],
        "signal_bip"  : True,
    },
    "neutral" : {
        "description" : "Posture correcte — sondage et synthèse (étapes 2–4)",
        "etapes_vm"   : [2, 3, 4],
        "signal_bip"  : False,
    },
    "slouched": {
        "description" : "Posture fermée — signal de stress ou d'hésitation (étapes 1–2)",
        "etapes_vm"   : [1, 2],
        "signal_bip"  : False,
    },
}

# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════

os.makedirs(CONFIG["models_dir"], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(CONFIG["models_dir"]) / "lstm_training_v2.log", mode="w"
        ),
    ],
)
log = logging.getLogger("LSTMTrainV2")


# ══════════════════════════════════════════════════════════════════════
# LSTM — IMPLEMENTATION NUMPY (inchangée V1 — architecture validée)
# ══════════════════════════════════════════════════════════════════════

class LSTMCell:
    """LSTM cell — NumPy implementation avec BPTT."""

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng   = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wx = rng.randn(input_size,  4 * hidden_size) * scale
        self.Wh = rng.randn(hidden_size, 4 * hidden_size) * scale
        self.b  = np.zeros(4 * hidden_size)
        self.b[hidden_size:2*hidden_size] = 1.0  # forget gate bias
        self.hidden_size = hidden_size
        # Adam states
        self.mWx=np.zeros_like(self.Wx); self.vWx=np.zeros_like(self.Wx)
        self.mWh=np.zeros_like(self.Wh); self.vWh=np.zeros_like(self.Wh)
        self.mb =np.zeros_like(self.b);  self.vb =np.zeros_like(self.b)

    def forward(self, x, h_prev, c_prev):
        H     = self.hidden_size
        gates = x @ self.Wx + h_prev @ self.Wh + self.b
        i = self._sig(gates[:H]); f = self._sig(gates[H:2*H])
        g = np.tanh(gates[2*H:3*H]); o = self._sig(gates[3*H:])
        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)
        return h_next, c_next, (x, h_prev, c_prev, i, f, g, o, c_next, gates)

    def backward(self, dh, dc, cache, clip=5.0):
        x, h_prev, c_prev, i, f, g, o, c_next, gates = cache
        H = self.hidden_size
        tanh_c = np.tanh(c_next)
        do = dh * tanh_c; dc_ = dh * o * (1 - tanh_c**2) + dc
        di=dc_*g; df=dc_*c_prev; dg=dc_*i; dc_prev=dc_*f
        di_r=di*i*(1-i); df_r=df*f*(1-f)
        dg_r=dg*(1-g**2); do_r=do*o*(1-o)
        dgates = np.concatenate([di_r,df_r,dg_r,do_r])
        dWx=np.outer(x,dgates); dWh=np.outer(h_prev,dgates)
        db=dgates; dx=self.Wx@dgates; dh_prev=self.Wh@dgates
        for d in [dWx,dWh,db,dx,dh_prev,dc_prev]:
            np.clip(d,-clip,clip,out=d)
        return dx, dh_prev, dc_prev, dWx, dWh, db

    def adam_update(self, dWx, dWh, db, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        self.mWx=b1*self.mWx+(1-b1)*dWx; self.vWx=b2*self.vWx+(1-b2)*dWx**2
        self.mWh=b1*self.mWh+(1-b1)*dWh; self.vWh=b2*self.vWh+(1-b2)*dWh**2
        self.mb =b1*self.mb +(1-b1)*db;  self.vb =b2*self.vb +(1-b2)*db**2
        def _a(m,v,w): w -= lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
        _a(self.mWx,self.vWx,self.Wx); _a(self.mWh,self.vWh,self.Wh)
        _a(self.mb, self.vb, self.b)

    @staticmethod
    def _sig(x):
        return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


class BodyLanguageLSTM:
    """
    LSTM 2-couches pour la classification de posture.
    Architecture identique V1 — interface enrichie V2.
    """

    def __init__(self, input_size=66, hidden1=64, hidden2=32,
                 dense=16, n_classes=3, seed=42, dropout=0.30):
        self.hidden1=hidden1; self.hidden2=hidden2
        self.n_classes=n_classes; self.dropout=dropout
        self.input_size=input_size
        rng=np.random.RandomState(seed)
        s1=np.sqrt(2.0/(hidden1+dense)); s2=np.sqrt(2.0/(dense+n_classes))
        self.lstm1=LSTMCell(input_size, hidden1, seed)
        self.lstm2=LSTMCell(hidden1,    hidden2, seed+1)
        self.W_fc1=rng.randn(hidden2,dense)*s1; self.b_fc1=np.zeros(dense)
        self.W_out=rng.randn(dense,n_classes)*s2; self.b_out=np.zeros(n_classes)
        for attr in ['m_W_fc1','v_W_fc1','m_b_fc1','v_b_fc1',
                     'm_W_out','v_W_out','m_b_out','v_b_out']:
            w = getattr(self, attr.replace('m_','').replace('v_',''))
            setattr(self, attr, np.zeros_like(w))

    def forward(self, seq, training=False):
        n = seq.shape[0]
        h1=np.zeros(self.hidden1); c1=np.zeros(self.hidden1)
        h2=np.zeros(self.hidden2); c2=np.zeros(self.hidden2)
        caches1=[]; caches2=[]; h1_seq=[]
        for t in range(n):
            h1,c1,cache1=self.lstm1.forward(seq[t],h1,c1)
            caches1.append(cache1); h1_seq.append(h1.copy())
        mask = ((np.random.rand(*h1.shape)>self.dropout).astype(float)/
                (1-self.dropout+1e-8)) if training else np.ones(self.hidden1)
        for t in range(n):
            h2,c2,cache2=self.lstm2.forward(h1_seq[t]*mask,h2,c2)
            caches2.append(cache2)
        fc1_in=h2; fc1_out=np.maximum(0,fc1_in@self.W_fc1+self.b_fc1)
        logits=fc1_out@self.W_out+self.b_out
        probs=self._softmax(logits)
        cache={"seq":seq,"caches1":caches1,"caches2":caches2,"mask":mask,
               "h1_seq":h1_seq,"h2":h2,"fc1_in":fc1_in,"fc1_out":fc1_out,
               "logits":logits,"probs":probs}
        return probs, cache

    def backward(self, probs, y_true, cache, clip=5.0):
        n=len(cache["caches1"])
        dlogits=probs.copy(); dlogits[y_true]-=1.0
        loss=-np.log(probs[y_true]+1e-9)
        dW_out=np.outer(cache["fc1_out"],dlogits); db_out=dlogits
        dfc1=self.W_out@dlogits
        dfc1_relu=dfc1*(cache["fc1_out"]>0)
        dW_fc1=np.outer(cache["fc1_in"],dfc1_relu); db_fc1=dfc1_relu
        dh2=self.W_fc1@dfc1_relu
        for d in [dW_out,db_out,dW_fc1,db_fc1,dh2]:
            np.clip(d,-clip,clip,out=d)
        dc2=np.zeros(self.hidden2); dh1t=np.zeros(self.hidden1)
        for t in reversed(range(n)):
            dx2,dh2p,dc2,dWx2,dWh2,db2=self.lstm2.backward(dh2,dc2,cache["caches2"][t],clip)
            dh2=dh2p; dh1t+=dx2/n
        dh1t*=cache["mask"]
        dc1=np.zeros(self.hidden1)
        for t in reversed(range(n)):
            _,dh1p,dc1,dWx1,dWh1,db1=self.lstm1.backward(dh1t,dc1,cache["caches1"][t],clip)
            dh1t=dh1p
        return loss,dW_fc1,db_fc1,dW_out,db_out,dWx1,dWh1,db1,dWx2,dWh2,db2

    def predict_proba(self, seq):
        probs,_=self.forward(seq,training=False); return probs

    def predict(self, seq):
        return int(np.argmax(self.predict_proba(seq)))

    def predict_batch(self, X):
        return np.array([self.predict(X[i]) for i in range(len(X))])

    def count_params(self):
        total=0
        for cell in [self.lstm1,self.lstm2]:
            total+=cell.Wx.size+cell.Wh.size+cell.b.size
        total+=self.W_fc1.size+self.b_fc1.size+self.W_out.size+self.b_out.size
        return total

    @staticmethod
    def _softmax(x):
        e=np.exp(x-x.max()); return e/(e.sum()+1e-9)


# ══════════════════════════════════════════════════════════════════════
# TRAINER (inchangé — architecture V1 validée)
# ══════════════════════════════════════════════════════════════════════

class LSTMTrainerV2:

    def __init__(self, model, cfg):
        self.model=model; self.cfg=cfg
        self.history={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
        self.best_val_loss=np.inf; self.patience_counter=0; self.t_adam=0
        self.best_state=None

    def _run_epoch(self, X, y, lr, training=True):
        total_loss=0.0; correct=0; n=len(X)
        indices=np.random.permutation(n) if training else np.arange(n)
        for i in indices:
            probs,cache=self.model.forward(X[i],training=training)
            correct+=int(np.argmax(probs)==y[i])
            if training:
                self.t_adam+=1
                (loss,dW_fc1,db_fc1,dW_out,db_out,
                 dWx1,dWh1,db1,dWx2,dWh2,db2)=self.model.backward(
                    probs,int(y[i]),cache,self.cfg["clip_grad_norm"])
                total_loss+=loss
                self.model.lstm1.adam_update(dWx1,dWh1,db1,lr,self.t_adam)
                self.model.lstm2.adam_update(dWx2,dWh2,db2,lr,self.t_adam)
                self._adam_dense(dW_fc1,db_fc1,dW_out,db_out,lr,self.t_adam)
            else:
                total_loss+=-np.log(probs[y[i]]+1e-9)
        return total_loss/n, correct/n

    def _adam_dense(self,dW_fc1,db_fc1,dW_out,db_out,lr,t,b1=0.9,b2=0.999,eps=1e-8):
        m=self.model
        def _up(mw,vw,w,mb,vb,b):
            mw[:]=b1*mw+(1-b1)*w; vw[:]=b2*vw+(1-b2)*w**2
            mb[:]=b1*mb+(1-b1)*b; vb[:]=b2*vb+(1-b2)*b**2
            w -= lr*(mw/(1-b1**t))/(np.sqrt(vw/(1-b2**t))+eps)
            b -= lr*(mb/(1-b1**t))/(np.sqrt(vb/(1-b2**t))+eps)
        _up(m.m_W_fc1,m.v_W_fc1,m.W_fc1,m.m_b_fc1,m.v_b_fc1,m.b_fc1)
        _up(m.m_W_out,m.v_W_out,m.W_out,m.m_b_out,m.v_b_out,m.b_out)

    def train(self, X_tr, y_tr, X_val, y_val):
        cfg=self.cfg; lr=cfg["learning_rate"]
        log.info("="*60)
        log.info("  LSTM Training — Adam + BPTT")
        log.info(f"  Epochs={cfg['epochs']} | LR={lr} | Early={cfg['early_stop_patience']}")
        log.info("="*60)
        log.info(f"  {'Epoch':>6}  {'TrLoss':>9}  {'ValLoss':>9}  {'TrAcc':>8}  {'ValAcc':>8}  {'LR':>9}")
        log.info("  "+"-"*58)
        t0=time.time()
        for epoch in range(1,cfg["epochs"]+1):
            if epoch>1 and (epoch-1)%cfg["lr_decay_every"]==0:
                lr*=cfg["lr_decay"]
            tl,ta=self._run_epoch(X_tr,y_tr,lr,training=True)
            vl,va=self._run_epoch(X_val,y_val,lr,training=False)
            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["train_acc"].append(ta)
            self.history["val_acc"].append(va)
            if epoch%5==0 or epoch==1:
                log.info(f"  {epoch:>6}  {tl:>9.4f}  {vl:>9.4f}  "
                         f"{ta*100:>7.2f}%  {va*100:>7.2f}%  {lr:>9.6f}")
            if vl < self.best_val_loss-1e-4:
                self.best_val_loss=vl; self.patience_counter=0
                self.best_state=self._copy_weights()
            else:
                self.patience_counter+=1
            if self.patience_counter>=cfg["early_stop_patience"]:
                log.info(f"  Early stopping epoch {epoch} (best_val_loss={self.best_val_loss:.4f})")
                break
        log.info(f"\n  Entraînement terminé en {time.time()-t0:.1f}s")
        if self.best_state:
            self._load_weights(self.best_state)
            log.info("  ✅  Best weights restaurés")
        return self.history

    def evaluate(self, X, y, name="Test"):
        y_pred=self.model.predict_batch(X)
        acc=accuracy_score(y,y_pred)
        f1 =f1_score(y,y_pred,average="macro")
        cm =confusion_matrix(y,y_pred)
        log.info(f"\n  {name} — Accuracy={acc*100:.2f}%  F1-Macro={f1:.4f}")
        log.info("\n"+classification_report(y,y_pred,target_names=CLASS_NAMES,digits=4))
        log.info(f"  Confusion matrix :\n{cm}")
        return {"accuracy":float(acc),"f1_macro":float(f1),
                "confusion_matrix":cm.tolist()}

    def save(self, scaler, test_metrics, cfg):
        mdir=Path(cfg["models_dir"])
        bundle={
            "model"        : self.model,
            "scaler"       : scaler,
            "config"       : cfg,
            "history"      : self.history,
            "class_names"  : CLASS_NAMES,
            "coaching_alia": COACHING_ALIA,
            "vm_context"   : VM_CONTEXT,
            "trained_at"   : datetime.now().isoformat(),
            "version"      : "2.0.0",
        }
        joblib.dump(bundle, mdir/"lstm_body_language_v2.pkl")
        log.info(f"\n  ✅  Model saved → {mdir}/lstm_body_language_v2.pkl")
        report={
            "test_metrics" : test_metrics,
            "best_val_loss": float(self.best_val_loss),
            "history"      : {k:[float(v) for v in vs]
                              for k,vs in self.history.items()},
            "model_params" : self.model.count_params(),
            "trained_at"   : datetime.now().isoformat(),
            "version"      : "2.0.0",
        }
        with open(mdir/"lstm_training_report_v2.json","w") as f:
            json.dump(report,f,indent=2)
        log.info(f"  ✅  Report saved → {mdir}/lstm_training_report_v2.json")
        log.info("\n  Artifacts :")
        for p in sorted(mdir.iterdir()):
            if p.suffix in [".pkl",".json",".log"]:
                log.info(f"    {p.name:<45}  {p.stat().st_size/1024:>7.1f} KB")

    def _copy_weights(self):
        m=self.model
        return {k:getattr(m.lstm1,k[6:]).copy() if k.startswith("lstm1_")
                else getattr(m.lstm2,k[6:]).copy() if k.startswith("lstm2_")
                else getattr(m,k).copy()
                for k in ["lstm1_Wx","lstm1_Wh","lstm1_b",
                           "lstm2_Wx","lstm2_Wh","lstm2_b",
                           "W_fc1","b_fc1","W_out","b_out"]}

    def _load_weights(self, state):
        m=self.model
        m.lstm1.Wx=state["lstm1_Wx"]; m.lstm1.Wh=state["lstm1_Wh"]; m.lstm1.b=state["lstm1_b"]
        m.lstm2.Wx=state["lstm2_Wx"]; m.lstm2.Wh=state["lstm2_Wh"]; m.lstm2.b=state["lstm2_b"]
        m.W_fc1=state["W_fc1"]; m.b_fc1=state["b_fc1"]
        m.W_out=state["W_out"]; m.b_out=state["b_out"]


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_training(cfg, eval_only=False):
    np.random.seed(cfg["random_seed"])
    t0=time.time()
    log.info("="*70)
    log.info("  ALIA — LSTM Body Language Training Pipeline V2")
    log.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Data    : {cfg['sequences_path']}")
    log.info("="*70)

    # ── Data ──────────────────────────────────────────────────────────
    X_raw = np.load(cfg["sequences_path"])
    y     = np.load(cfg["labels_path"])
    log.info(f"Data chargé  : X={X_raw.shape} | y={y.shape}")
    log.info(f"Distribution : {dict(zip(CLASS_NAMES, [(y==i).sum() for i in range(3)]))}")

    if eval_only:
        bundle=joblib.load(Path(cfg["models_dir"])/"lstm_body_language_v2.pkl")
        _demo(bundle["model"], bundle["scaler"], X_raw, y, cfg)
        return

    # ── Normalisation ─────────────────────────────────────────────────
    N,T,F=X_raw.shape
    scaler=StandardScaler()
    X_norm=scaler.fit_transform(X_raw.reshape(-1,F)).reshape(N,T,F)

    # ── Split ─────────────────────────────────────────────────────────
    X_tv,X_te,y_tv,y_te=train_test_split(
        X_norm,y,test_size=cfg["test_size"],random_state=SEED,stratify=y)
    X_tr,X_val,y_tr,y_val=train_test_split(
        X_tv,y_tv,test_size=cfg["val_size"],random_state=SEED,stratify=y_tv)
    log.info(f"Split        : train={len(X_tr)} | val={len(X_val)} | test={len(X_te)}")

    # ── Modèle ────────────────────────────────────────────────────────
    model=BodyLanguageLSTM(
        input_size=cfg["input_size"], hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"], dense=cfg["dense_units"],
        n_classes=cfg["n_classes"], seed=SEED, dropout=cfg["dropout_rate"])
    log.info(f"\n  Architecture LSTM V2 :")
    log.info(f"    Input      : (seq_len={cfg['seq_len']}, features={cfg['input_size']})")
    log.info(f"    LSTM 1     : {cfg['hidden1']} unités")
    log.info(f"    Dropout    : {cfg['dropout_rate']}")
    log.info(f"    LSTM 2     : {cfg['hidden2']} unités")
    log.info(f"    Dense      : {cfg['dense_units']} unités (ReLU)")
    log.info(f"    Output     : {cfg['n_classes']} classes (Softmax)")
    log.info(f"    Paramètres : {model.count_params():,}")

    # ── Entraînement ──────────────────────────────────────────────────
    trainer=LSTMTrainerV2(model,cfg)
    trainer.train(X_tr,y_tr,X_val,y_val)

    # ── Évaluation ────────────────────────────────────────────────────
    log.info("\n  Évaluation finale :")
    test_metrics=trainer.evaluate(X_te,y_te,"Test")

    # ── Benchmark ─────────────────────────────────────────────────────
    log.info("\n"+"="*60)
    log.info("  BENCHMARK FINAL — LSTM Body Language V2")
    log.info("="*60)
    log.info(f"  Accuracy    : {test_metrics['accuracy']*100:.2f}%")
    log.info(f"  F1-Macro    : {test_metrics['f1_macro']:.4f}")
    log.info(f"  Val Loss    : {trainer.best_val_loss:.4f}")
    log.info(f"  Paramètres  : {model.count_params():,}")
    log.info("="*60)

    # ── Save ──────────────────────────────────────────────────────────
    trainer.save(scaler, test_metrics, cfg)

    # ── Demo ──────────────────────────────────────────────────────────
    _demo(model, scaler, X_te, y_te, cfg)

    log.info(f"\n✅  Pipeline V2 complet en {time.time()-t0:.1f}s")


def _demo(model, scaler, X_te, y_te, cfg):
    log.info("\n"+"="*60)
    log.info("  DÉMO INFÉRENCE V2 — 3 classes")
    log.info("="*60)
    X_raw=np.load(cfg["sequences_path"]); y=np.load(cfg["labels_path"])
    N,T,F=X_raw.shape
    X_norm=scaler.transform(X_raw.reshape(-1,F)).reshape(N,T,F)
    for class_id, name in enumerate(CLASS_NAMES):
        idx=np.where(y==class_id)[0][0]
        probs=model.predict_proba(X_norm[idx])
        pred=CLASS_NAMES[int(np.argmax(probs))]
        match="✅" if pred==name else "❌"
        vm=VM_CONTEXT[name]
        log.info(f"\n  [{match}] Attendu={name:<10} Prédit={pred:<10}")
        log.info(f"    Proba : {dict(zip(CLASS_NAMES,probs.round(3)))}")
        log.info(f"    VM    : {vm['description']}")
        log.info(f"    Coach : {COACHING_ALIA[name]['Confirmé']}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p=argparse.ArgumentParser(description="ALIA LSTM Body Language V2 — Training")
    p.add_argument("--input",      default=CONFIG["sequences_path"])
    p.add_argument("--labels",     default=CONFIG["labels_path"])
    p.add_argument("--epochs",     type=int, default=CONFIG["epochs"])
    p.add_argument("--lr",         type=float, default=CONFIG["learning_rate"])
    p.add_argument("--models-dir", default=CONFIG["models_dir"])
    p.add_argument("--eval-only",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args=parse_args()
    CONFIG["sequences_path"]=args.input
    CONFIG["labels_path"]   =args.labels
    CONFIG["epochs"]        =args.epochs
    CONFIG["learning_rate"] =args.lr
    CONFIG["models_dir"]    =args.models_dir
    os.makedirs(CONFIG["models_dir"],exist_ok=True)
    run_training(CONFIG, eval_only=args.eval_only)
