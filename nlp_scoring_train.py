"""
nlp_scoring_train.py
====================
DSO 1 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Script d'entraînement production-ready pour le moteur NLP de scoring des
conversations médicales.

Tâches :
    T1 — Classification qualité réponse  (Excellent / Bon / Faible)
    T2 — Régression multi-dimensionnelle (scientific / clarity / objection scores)
    T3 — Prédiction sentiment client     (Positif / Neutre / Négatif)

Usage :
    python nlp_scoring_train.py                        # full pipeline
    python nlp_scoring_train.py --data path/to/ds4.csv
    python nlp_scoring_train.py --tune                 # with GridSearchCV
    python nlp_scoring_train.py --eval-only            # load & evaluate only

Output :
    models/nlp_scoring_model.pkl   ← artefact principal (NLPScoringModel)
    models/t1_quality_classifier.pkl
    models/t2_score_regressor.pkl
    models/t3_sentiment_classifier.pkl
    models/training_report.json    ← métriques complètes

Architecture extensible :
    TF-IDF + sklearn (baseline)  ←  ce script
    CamemBERT / XLM-R (upgrade)  ←  remplacer _build_vectorizer() et _build_t1()

Author  : CYBER SHADE — ALIA Project
Version : 1.0.0
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Scientific stack ──────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd

# ── Scikit-learn ──────────────────────────────────────────────────────────
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression, Ridge
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     f1_score, mean_absolute_error, r2_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.multioutput     import MultiOutputRegressor
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import LabelEncoder
from sklearn.svm             import SVC

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ────────────────────────────────────────────────────────────────
    "data_path"      : "conversation_transcripts.csv",
    "models_dir"     : "models",
    "test_size"      : 0.20,
    "random_seed"    : 42,

    # ── ALIA scoring weights (must sum to 1.0) ───────────────────────────
    "score_weights"  : [0.25, 0.30, 0.45],   # scientific / clarity / objection
    "score_cols"     : [
        "scientific_accuracy_score",
        "communication_clarity_score",
        "objection_handling_score",
    ],

    # ── TF-IDF ───────────────────────────────────────────────────────────
    "tfidf" : {
        "ngram_range" : (1, 2),
        "max_features": 5000,
        "min_df"      : 2,
        "max_df"      : 0.95,
        "sublinear_tf": True,
    },

    # ── Models (baseline — no tuning) ────────────────────────────────────
    "t1_model" : "svm",     # "svm" | "lr" | "rf"
    "t2_model" : "ridge",   # "ridge" | "rf"
    "t3_model" : "lr",      # "lr"

    # ── GridSearchCV param grids ─────────────────────────────────────────
    "t1_param_grid" : {
        "clf__C"    : [0.1, 1.0, 5.0, 10.0],
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["rbf", "linear"],
    },
    "t2_param_grid" : {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },

    # ── Cross-validation ─────────────────────────────────────────────────
    "cv_folds"    : 5,
    "cv_scoring_t1": "f1_macro",
    "cv_scoring_t3": "f1_macro",
    "cv_scoring_t2": "neg_mean_absolute_error",
}

SCORE_WEIGHTS = np.array(CONFIG["score_weights"])
SCORE_COLS    = CONFIG["score_cols"]


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("models/training.log", mode="w"),
    ],
)
log = logging.getLogger("NLPScoring")


# ══════════════════════════════════════════════════════════════════════════════
# NLP FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class NLPFeatureExtractor:
    """
    Extrait des features linguistiques interprétables depuis les réponses texte.
    Basée sur des règles — aucun modèle pré-entraîné requis.
    Conçue pour être importée dans le moteur avatar DSO 1.
    """

    EMPATHY_KW   = ["comprends", "entends", "légitime", "inquiétude",
                    "préoccupation", "tout à fait", "bonne question"]
    DATA_PATTERN = re.compile(
        r"\d+\s*%|étude|clinique|données|résultat|prouve|littérature|publication",
        re.IGNORECASE
    )
    SOLUTION_KW  = ["vous propose", "permettez", "montrer",
                    "présenter", "solution", "répondre à"]
    VAGUE_KW     = ["bien", "fiable", "sérieux", "faites-moi confiance",
                    "bon produit", "faites confiance"]

    def extract(self, text: str) -> Dict:
        """Extrait 10 features depuis une réponse texte."""
        t = text.lower()

        has_empathy    = int(any(w in t for w in self.EMPATHY_KW))
        has_data_args  = int(bool(self.DATA_PATTERN.search(t)))
        has_followup_q = int("?" in text)
        has_solution   = int(any(w in t for w in self.SOLUTION_KW))
        is_vague       = int(any(w in t for w in self.VAGUE_KW))

        word_count      = len(text.split())
        response_length = len(text)
        sentence_count  = text.count(".") + text.count("?") + text.count("!")
        has_numbers     = int(bool(re.search(r"\d+", text)))
        arg_richness    = has_empathy + has_data_args + has_followup_q + has_solution

        return {
            "has_empathy"           : has_empathy,
            "has_data_args"         : has_data_args,
            "has_followup_q"        : has_followup_q,
            "has_solution"          : has_solution,
            "is_vague"              : is_vague,
            "word_count"            : word_count,
            "response_length"       : response_length,
            "sentence_count"        : sentence_count,
            "has_numbers"           : has_numbers,
            "argumentation_richness": arg_richness,
        }

    def generate_feedback(self, features: Dict) -> List[str]:
        """
        Génère des recommandations de feedback depuis les features NLP.
        Utilisé par l'avatar pour guider le délégué en temps réel.
        """
        hints = []
        if not features["has_empathy"]:
            hints.append("Ajouter une phrase d'empathie en début de réponse")
        if not features["has_data_args"]:
            hints.append("Inclure des données chiffrées ou références cliniques")
        if not features["has_followup_q"]:
            hints.append("Terminer par une question ouverte pour engager le client")
        if features["is_vague"]:
            hints.append("Éviter les formulations vagues — être plus précis et concret")
        if features["word_count"] < 30:
            hints.append("Développer davantage l'argumentation (réponse trop courte)")
        if features["argumentation_richness"] < 2:
            hints.append("Enrichir la structure : empathie + données + solution + question")
        return hints or ["Réponse de qualité — aucune amélioration critique détectée"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Charge et prépare DS4 pour l'entraînement."""

    def __init__(self, path: str):
        self.path      = path
        self.extractor = NLPFeatureExtractor()

    def load(self) -> pd.DataFrame:
        log.info(f"Loading dataset : {self.path}")
        df = pd.read_csv(self.path, encoding="utf-8-sig")
        log.info(f"  Shape         : {df.shape}")
        log.info(f"  Missing values: {df.isnull().sum().sum()}")
        log.info(f"  Duplicates    : {df.duplicated().sum()}")

        # ── Quality distribution ─────────────────────────────────────────
        q_dist = df["response_quality"].value_counts()
        log.info(f"  Quality dist  : {q_dist.to_dict()}")

        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit le texte d'entrée et les features NLP."""
        log.info("Building NLP features...")

        # ── Combined text (objection + response) — meilleure performance ─
        df = df.copy()
        df["text_input"] = (df["objection_text"].fillna("") +
                            " [SEP] " +
                            df["rep_response"].fillna(""))

        # ── NLP features ─────────────────────────────────────────────────
        nlp_feats = df["rep_response"].apply(self.extractor.extract).apply(pd.Series)
        df = pd.concat([df, nlp_feats], axis=1)

        log.info(f"  NLP features added : {nlp_feats.shape[1]} features")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(**CONFIG["tfidf"])


def _build_t1(model_type: str = "svm") -> Pipeline:
    """Builds T1 classification pipeline."""
    tfidf = _build_tfidf()
    if model_type == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                  random_state=CONFIG["random_seed"], probability=True)
    elif model_type == "lr":
        clf = LogisticRegression(C=1.0, max_iter=1000,
                                 random_state=CONFIG["random_seed"])
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                     random_state=CONFIG["random_seed"], n_jobs=-1)
    else:
        raise ValueError(f"Unknown T1 model type: {model_type}")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def _build_t2(model_type: str = "ridge") -> Tuple:
    """Builds T2 regression components (tfidf separate for reuse)."""
    tfidf = _build_tfidf()
    if model_type == "ridge":
        reg = MultiOutputRegressor(Ridge(alpha=1.0))
    elif model_type == "rf":
        reg = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=15,
                                  random_state=CONFIG["random_seed"], n_jobs=-1))
    else:
        raise ValueError(f"Unknown T2 model type: {model_type}")
    return tfidf, reg


def _build_t3() -> Pipeline:
    """Builds T3 sentiment pipeline."""
    return Pipeline([
        ("tfidf", _build_tfidf()),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000,
                                     random_state=CONFIG["random_seed"])),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class NLPScoringTrainer:
    """
    Orchestre l'entraînement complet des 3 tâches NLP.
    Gère le splitting, la cross-validation, l'évaluation et la sauvegarde.
    """

    def __init__(self, config: Dict = CONFIG):
        self.cfg       = config
        self.seed      = config["random_seed"]
        self.cv        = StratifiedKFold(n_splits=config["cv_folds"],
                                         shuffle=True, random_state=self.seed)
        self.report    = {}
        os.makedirs(config["models_dir"], exist_ok=True)

    # ── T1 : Quality Classification ──────────────────────────────────────
    def train_t1(self, X_train, X_test, y_train, y_test,
                 le: LabelEncoder, tune: bool = False) -> Pipeline:

        log.info("=" * 60)
        log.info("TASK 1 — Response Quality Classification")
        log.info("=" * 60)

        pipe = _build_t1(self.cfg["t1_model"])

        if tune:
            log.info("  Running GridSearchCV (T1)...")
            gs = GridSearchCV(
                pipe,
                param_grid=self.cfg["t1_param_grid"],
                cv=self.cv,
                scoring=self.cfg["cv_scoring_t1"],
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_train, y_train)
            pipe = gs.best_estimator_
            log.info(f"  Best params : {gs.best_params_}")
            log.info(f"  Best CV F1  : {gs.best_score_:.4f}")
        else:
            # ── Standard cross-validation ─────────────────────────────
            cv_scores = cross_val_score(pipe, X_train, y_train,
                                        cv=self.cv,
                                        scoring=self.cfg["cv_scoring_t1"])
            log.info(f"  CV F1-Macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            log.info(f"  CV range    : [{cv_scores.min():.4f} – {cv_scores.max():.4f}]")
            pipe.fit(X_train, y_train)

        # ── Test set evaluation ───────────────────────────────────────
        y_pred  = pipe.predict(X_test)
        f1_test = f1_score(y_test, y_pred, average="macro")
        report  = classification_report(y_test, y_pred,
                                         target_names=le.classes_,
                                         output_dict=True)

        log.info(f"  Test F1-Macro   : {f1_test:.4f}")
        log.info(f"  Test Accuracy   : {report['accuracy']:.4f}")
        for cls in le.classes_:
            log.info(f"    {cls:<12} precision={report[cls]['precision']:.3f}  "
                     f"recall={report[cls]['recall']:.3f}  "
                     f"f1={report[cls]['f1-score']:.3f}")

        # ── Confusion matrix ──────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        log.info(f"  Confusion matrix :\n{cm}")

        self.report["t1"] = {
            "cv_f1_mean"  : float(cv_scores.mean()) if not tune else float(gs.best_score_),
            "cv_f1_std"   : float(cv_scores.std())  if not tune else 0.0,
            "test_f1"     : float(f1_test),
            "test_accuracy": float(report["accuracy"]),
            "per_class"   : {cls: report[cls] for cls in le.classes_},
            "confusion_matrix": cm.tolist(),
        }
        return pipe

    # ── T2 : Score Regression ────────────────────────────────────────────
    def train_t2(self, X_train, X_test, y_train, y_test,
                 tune: bool = False) -> Dict:

        log.info("=" * 60)
        log.info("TASK 2 — Multi-Dimensional Score Regression")
        log.info("=" * 60)

        tfidf, reg = _build_t2(self.cfg["t2_model"])
        X_tr_vec   = tfidf.fit_transform(X_train)
        X_te_vec   = tfidf.transform(X_test)

        if tune:
            log.info("  Running GridSearchCV per target (T2)...")
            best_alphas = []
            for i, col in enumerate(SCORE_COLS):
                gs = GridSearchCV(
                    Ridge(),
                    param_grid=self.cfg["t2_param_grid"],
                    cv=5,
                    scoring=self.cfg["cv_scoring_t2"],
                    n_jobs=-1,
                )
                gs.fit(X_tr_vec, y_train[:, i])
                best_alphas.append(gs.best_params_["alpha"])
                log.info(f"  {col}: best alpha={gs.best_params_['alpha']}")
            # Refit with best alpha (use mean as single MultiOutput value)
            best_alpha = float(np.mean(best_alphas))
            reg = MultiOutputRegressor(Ridge(alpha=best_alpha))

        reg.fit(X_tr_vec, y_train)
        y_pred = np.clip(reg.predict(X_te_vec), 0, 10)

        # ── Per-score metrics ─────────────────────────────────────────
        t2_metrics = {}
        for i, col in enumerate(SCORE_COLS):
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2  = r2_score(y_test[:, i], y_pred[:, i])
            short = col.replace("_score", "").replace("_", " ").title()
            log.info(f"  {short:<30}  MAE={mae:.4f}  R²={r2:.4f}")
            t2_metrics[col] = {"mae": float(mae), "r2": float(r2)}

        # ── Overall ALIA score ────────────────────────────────────────
        mae_ov = mean_absolute_error(y_test @ SCORE_WEIGHTS, y_pred @ SCORE_WEIGHTS)
        r2_ov  = r2_score(y_test @ SCORE_WEIGHTS, y_pred @ SCORE_WEIGHTS)
        log.info(f"  {'Overall (ALIA formula)':<30}  MAE={mae_ov:.4f}  R²={r2_ov:.4f}")

        self.report["t2"] = {
            "per_score"     : t2_metrics,
            "overall_mae"   : float(mae_ov),
            "overall_r2"    : float(r2_ov),
            "score_weights" : SCORE_WEIGHTS.tolist(),
        }
        return {"tfidf": tfidf, "reg": reg}

    # ── T3 : Sentiment Prediction ────────────────────────────────────────
    def train_t3(self, X_train, X_test, y_train, y_test,
                 le: LabelEncoder) -> Pipeline:

        log.info("=" * 60)
        log.info("TASK 3 — Client Sentiment Prediction")
        log.info("=" * 60)

        pipe      = _build_t3()
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=self.cv,
                                    scoring=self.cfg["cv_scoring_t3"])
        log.info(f"  CV F1-Macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        f1_test = f1_score(y_test, y_pred, average="macro")
        report  = classification_report(y_test, y_pred,
                                         target_names=le.classes_,
                                         output_dict=True)

        log.info(f"  Test F1-Macro   : {f1_test:.4f}")
        log.info(f"  Test Accuracy   : {report['accuracy']:.4f}")
        log.info(f"  Note: F1~0.44 expected — sentiment ambiguity is natural")

        self.report["t3"] = {
            "cv_f1_mean"   : float(cv_scores.mean()),
            "cv_f1_std"    : float(cv_scores.std()),
            "test_f1"      : float(f1_test),
            "test_accuracy": float(report["accuracy"]),
            "note"         : "Moderate F1 expected — sentiment depends on full context",
        }
        return pipe

    # ── Save all artifacts ───────────────────────────────────────────────
    def save(self, t1_pipe, t2_artifact, t3_pipe,
             le_q: LabelEncoder, le_s: LabelEncoder) -> None:

        mdir = Path(self.cfg["models_dir"])

        joblib.dump(t1_pipe,     mdir / "t1_quality_classifier.pkl")
        joblib.dump(le_q,        mdir / "t1_label_encoder.pkl")
        joblib.dump(t2_artifact, mdir / "t2_score_regressor.pkl")
        joblib.dump(t3_pipe,     mdir / "t3_sentiment_classifier.pkl")
        joblib.dump(le_s,        mdir / "t3_label_encoder.pkl")

        # ── Save full NLPScoringModel bundle ─────────────────────────
        bundle = {
            "t1_pipeline"  : t1_pipe,
            "t2_artifact"  : t2_artifact,
            "t3_pipeline"  : t3_pipe,
            "le_quality"   : le_q,
            "le_sentiment" : le_s,
            "config"       : self.cfg,
            "trained_at"   : datetime.now().isoformat(),
            "version"      : "1.0.0",
        }
        joblib.dump(bundle, mdir / "nlp_scoring_model.pkl")
        log.info(f"  [OK]  NLP bundle saved → {mdir}/nlp_scoring_model.pkl")

        # ── Training report ───────────────────────────────────────────
        self.report["trained_at"] = datetime.now().isoformat()
        self.report["version"]    = "1.0.0"
        with open(mdir / "training_report.json", "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        log.info(f"  [OK]  Training report → {mdir}/training_report.json")

        # ── Artifact summary ──────────────────────────────────────────
        log.info("\n  Saved artifacts :")
        for p in sorted(mdir.iterdir()):
            size_kb = p.stat().st_size / 1024
            log.info(f"    {p.name:<45}  {size_kb:>7.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# NLPScoringModel — INFERENCE CLASS (importable by avatar)
# ══════════════════════════════════════════════════════════════════════════════

class NLPScoringModel:
    """
    Classe d'inférence production-ready.
    Chargée par le moteur avatar DSO 1 pour scorer les réponses en temps réel.

    Usage :
        model = NLPScoringModel.load("models/nlp_scoring_model.pkl")
        result = model.predict(objection="...", response="...")

    Returns :
        {
            "quality"       : "Excellent" | "Bon" | "Faible",
            "quality_proba" : {"Excellent": 0.82, "Bon": 0.15, "Faible": 0.03},
            "scores"        : {
                "scientific_accuracy"   : 8.4,
                "communication_clarity" : 8.7,
                "objection_handling"    : 7.9,
            },
            "overall_score" : 8.21,
            "sentiment"     : "Positif" | "Neutre" | "Négatif",
            "feedback"      : ["..."],       # coaching hints for the avatar
            "nlp_features"  : {...},         # interpretability
        }

    Extension vers transformers :
        Remplacer t1_pipeline et t3_pipeline par des pipelines HuggingFace,
        et t2_artifact par un RegressorHead sur embeddings CamemBERT.
        L'interface predict() reste identique — aucun changement côté avatar.
    """

    def __init__(self, bundle: Dict):
        self.t1          = bundle["t1_pipeline"]
        self.t2          = bundle["t2_artifact"]
        self.t3          = bundle["t3_pipeline"]
        self.le_q        = bundle["le_quality"]
        self.le_s        = bundle["le_sentiment"]
        self.extractor   = NLPFeatureExtractor()
        self.weights     = np.array(bundle["config"]["score_weights"])
        self.score_cols  = bundle["config"]["score_cols"]
        self.version     = bundle.get("version", "unknown")
        self.trained_at  = bundle.get("trained_at", "unknown")

    @classmethod
    def load(cls, path: str) -> "NLPScoringModel":
        """Load from saved bundle."""
        bundle = joblib.load(path)
        log.info(f"NLPScoringModel loaded — v{bundle.get('version','?')} "
                 f"trained at {bundle.get('trained_at','?')}")
        return cls(bundle)

    def predict(self, objection: str, response: str) -> Dict:
        """
        Pipeline d'inférence complet : texte → dict de scores et feedback.

        Args:
            objection : Texte de l'objection du professionnel de santé
            response  : Réponse du délégué médical

        Returns:
            dict avec quality, scores, overall_score, sentiment, feedback
        """
        text = objection + " [SEP] " + response

        # ── T1 : Quality classification ───────────────────────────────
        quality_enc   = self.t1.predict([text])[0]
        quality       = self.le_q.inverse_transform([quality_enc])[0]
        quality_proba = dict(zip(
            self.le_q.classes_,
            self.t1.predict_proba([text])[0].round(4)
        ))

        # ── T2 : Multi-score regression ───────────────────────────────
        vec    = self.t2["tfidf"].transform([text])
        raw    = self.t2["reg"].predict(vec)[0]
        scores = np.clip(raw, 0.0, 10.0)
        overall = float(scores @ self.weights)

        scores_dict = {
            col.replace("_score", "").replace("_", " ").strip(): round(float(s), 2)
            for col, s in zip(self.score_cols, scores)
        }

        # ── T3 : Sentiment prediction ─────────────────────────────────
        sentiment_enc = self.t3.predict([text])[0]
        sentiment     = self.le_s.inverse_transform([sentiment_enc])[0]

        # ── NLP features (interpretability + feedback) ────────────────
        nlp_features = self.extractor.extract(response)
        feedback     = self.extractor.generate_feedback(nlp_features)

        return {
            "quality"       : quality,
            "quality_proba" : quality_proba,
            "scores"        : scores_dict,
            "overall_score" : round(overall, 2),
            "sentiment"     : sentiment,
            "feedback"      : feedback,
            "nlp_features"  : nlp_features,
        }

    def predict_batch(self, conversations: List[Dict]) -> List[Dict]:
        """
        Inférence en batch pour plusieurs conversations.

        Args:
            conversations : [{"objection": "...", "response": "..."}, ...]

        Returns:
            List of prediction dicts
        """
        return [
            self.predict(c["objection"], c["response"])
            for c in conversations
        ]

    def score_summary(self, result: Dict) -> str:
        """Génère un résumé lisible d'un résultat de prédiction."""
        lines = [
            f"Quality       : {result['quality']}",
            f"Overall Score : {result['overall_score']:.1f} / 10",
            f"Sentiment     : {result['sentiment']}",
            f"Scores        : " + " | ".join(
                f"{k}={v}" for k, v in result["scores"].items()
            ),
            f"Feedback ({len(result['feedback'])} hints):",
        ]
        lines += [f"  • {h}" for h in result["feedback"]]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_training(data_path: str, tune: bool = False, eval_only: bool = False):

    np.random.seed(CONFIG["random_seed"])
    t_start = time.time()

    log.info("=" * 70)
    log.info("  ALIA — NLP Scoring Model Training Pipeline")
    log.info(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Data       : {data_path}")
    log.info(f"  Tune       : {tune}")
    log.info("=" * 70)

    # ── Load & prepare ────────────────────────────────────────────────────
    loader = DataLoader(data_path)
    df     = loader.load()
    df     = loader.build_features(df)

    X = df["text_input"]

    # ── Encode targets ───────────────────────────────────────────────────
    le_q = LabelEncoder()
    le_s = LabelEncoder()
    y_q  = le_q.fit_transform(df["response_quality"])
    y_s  = le_s.fit_transform(df["client_sentiment"])
    y_sc = df[SCORE_COLS].values

    log.info(f"Quality classes  : {le_q.classes_.tolist()}")
    log.info(f"Sentiment classes: {le_s.classes_.tolist()}")

    # ── Train/Test split ─────────────────────────────────────────────────
    seed = CONFIG["random_seed"]
    ts   = CONFIG["test_size"]

    # ── Use consistent indices across all tasks ──────────────────────────
    from sklearn.model_selection import StratifiedShuffleSplit
    sss   = StratifiedShuffleSplit(n_splits=1, test_size=ts, random_state=seed)
    tr_idx, te_idx = next(sss.split(X, y_q))

    X_arr  = X.values
    X_tr,  X_te   = X_arr[tr_idx],  X_arr[te_idx]
    y_tr_q, y_te_q = y_q[tr_idx],   y_q[te_idx]
    y_tr_sc, y_te_sc = y_sc[tr_idx], y_sc[te_idx]
    y_tr_s, y_te_s = y_s[tr_idx],   y_s[te_idx]

    log.info(f"Train : {len(X_tr)} | Test : {len(X_te)}")

    if eval_only:
        log.info("Eval-only mode — loading saved models...")
        model = NLPScoringModel.load(
            str(Path(CONFIG["models_dir"]) / "nlp_scoring_model.pkl"))
        _run_demo(model)
        return

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = NLPScoringTrainer(CONFIG)

    t1_pipe     = trainer.train_t1(X_tr, X_te, y_tr_q, y_te_q, le_q, tune=tune)
    t2_artifact = trainer.train_t2(X_tr, X_te, y_tr_sc, y_te_sc, tune=tune)
    t3_pipe     = trainer.train_t3(X_tr, X_te, y_tr_s, y_te_s, le_s)

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save(t1_pipe, t2_artifact, t3_pipe, le_q, le_s)

    # ── Demo inference ────────────────────────────────────────────────────
    model = NLPScoringModel.load(
        str(Path(CONFIG["models_dir"]) / "nlp_scoring_model.pkl"))
    _run_demo(model)

    elapsed = time.time() - t_start
    log.info(f"\n[OK]  Training complete in {elapsed:.1f}s")


def _run_demo(model: NLPScoringModel):
    """Lance une démo d'inférence end-to-end."""

    log.info("\n" + "=" * 60)
    log.info("  INFERENCE DEMO")
    log.info("=" * 60)

    test_cases = [
        {
            "label"    : "EXCELLENT (expected)",
            "objection": "J'ai des doutes sur l'efficacité. Avez-vous des preuves ?",
            "response" : (
                "Je comprends tout à fait votre préoccupation, c'est une question "
                "légitime. Des études cliniques publiées démontrent une efficacité "
                "de 82% après 6 semaines. Permettez-moi de vous montrer les données "
                "comparatives. Quels critères sont prioritaires pour vous ?"
            ),
        },
        {
            "label"    : "BON (expected)",
            "objection": "Le concurrent propose quelque chose de similaire.",
            "response" : (
                "Notre produit a eu de très bons retours de la part des pharmaciens "
                "de la région. C'est une solution que je recommande régulièrement."
            ),
        },
        {
            "label"    : "FAIBLE (expected)",
            "objection": "Le prix est trop élevé.",
            "response" : "Notre marque est fiable. Faites-moi confiance.",
        },
    ]

    for tc in test_cases:
        result = model.predict(tc["objection"], tc["response"])
        log.info(f"\n  [{tc['label']}]")
        log.info(model.score_summary(result))


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="ALIA NLP Scoring Model — Training Pipeline"
    )
    parser.add_argument(
        "--data", default=CONFIG["data_path"],
        help="Path to DS4 CSV file"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run GridSearchCV hyperparameter tuning"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, load saved model and run demo"
    )
    parser.add_argument(
        "--models-dir", default=CONFIG["models_dir"],
        help="Directory to save model artifacts"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["data_path"]  = args.data
    CONFIG["models_dir"] = args.models_dir

    os.makedirs(CONFIG["models_dir"], exist_ok=True)

    run_training(
        data_path  = args.data,
        tune       = args.tune,
        eval_only  = getattr(args, "eval_only", False),
    )
