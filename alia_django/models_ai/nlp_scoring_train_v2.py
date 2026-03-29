"""
nlp_scoring_train_v2.py
=======================
DSO 1 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Pipeline d'entraînement complet du NLP Scoring Model V2.

6 tâches multi-output sur le même texte d'entrée :
    T1  Classification   Qualité réponse       (Excellent / Bon / Faible)
    T2  Régression       Scores (×3)           (scientifique / clarté / objection)
    T3  Classification   Sentiment client       (Positif / Neutre / Négatif)
    T4  Classification   Niveau ALIA            (Débutant / Junior / Confirmé / Expert)
    T5  Classification   Format visite          (Flash / Standard / Approfondie)
    T6  Classification   Conformité             (conforme=1 / non-conforme=0)

Nouveautés V2 vs V1 :
    - 24 features linguistiques (vs 10) : A-C-R-V, BIP, conformité, 6 étapes VITAL
    - 16 produits VITAL réels (Annexe V2)
    - T4 niveau ALIA aligné sur le Référentiel officiel
    - T5 format visite (Manuel VITAL section 5)
    - T6 conformité / mots tueurs (Référentiel + Annexe V2)
    - TF-IDF 8000 features (vs 5000)
    - GridSearchCV sur T1 et T4

Usage :
    python nlp_scoring_train_v2.py                    # pipeline complet
    python nlp_scoring_train_v2.py --tune             # avec GridSearchCV
    python nlp_scoring_train_v2.py --eval-only        # évaluation seule
    python nlp_scoring_train_v2.py --data path/ds4.csv

Output :
    models/nlp_scoring_bundle_v2.pkl    ← bundle complet (import par nlp_scoring_model_v2.py)
    models/nlp_training_report_v2.json
    models/nlp_training_v2.log

Author  : CYBER SHADE — ALIA Project
Version : 2.0.0
"""

# ── Standard library ──────────────────────────────────────────────────
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

# ── Scientific stack ──────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import re

# ── Scikit-learn ──────────────────────────────────────────────────────
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model  import LogisticRegression, Ridge
from sklearn.metrics       import (accuracy_score, classification_report,
                                   confusion_matrix, f1_score,
                                   mean_absolute_error, r2_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.multioutput   import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm           import SVC

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────
    "data_path"   : "conversation_transcripts_v2.csv",
    "models_dir"  : "models",
    "test_size"   : 0.20,
    "random_seed" : 42,

    # ── Score weights ALIA (Manuel VITAL + Référentiel) ───────────────
    "score_weights": [0.25, 0.30, 0.45],  # sci / clarté / objection

    # ── TF-IDF ────────────────────────────────────────────────────────
    "tfidf": {
        "ngram_range" : (1, 2),
        "max_features": 8000,
        "min_df"      : 2,
        "sublinear_tf": True,
        "strip_accents": None,      # conserver accents français
    },

    # ── Modèles baseline ──────────────────────────────────────────────
    "t1_model": "svm",        # svm | lr | rf
    "t2_model": "ridge",
    "t3_model": "lr",
    "t4_model": "svm",        # GridSearch recommandé
    "t5_model": "lr",
    "t6_model": "lr",         # + règle déterministe regex

    # ── GridSearch param grids ────────────────────────────────────────
    "gs_t1": {"C": [0.1, 1.0, 5.0, 10.0], "gamma": ["scale", "auto"]},
    "gs_t4": {"C": [0.1, 0.5, 1.0, 5.0, 10.0], "gamma": ["scale", "auto"]},
    "gs_t3": {"C": [0.1, 0.5, 1.0, 5.0]},

    "cv_folds": 5,

    # ── Seuils ALIA (Référentiel officiel) ────────────────────────────
    "seuils_alia": {
        "Débutant" : 7.0,
        "Junior"   : 8.0,
        "Confirmé" : 9.0,
        "Expert"   : 10.0,
    },
    "seuil_competence"   : 7.0,
    "seuil_certification": 7.5,
    "seuil_coaching"     : 6.0,
}

CLASS_NAMES = {
    "t1": ["Excellent", "Bon", "Faible"],
    "t3": ["Positif", "Neutre", "Négatif"],
    "t4": ["Expert", "Confirmé", "Junior", "Débutant"],
    "t5": ["Flash", "Standard", "Approfondie"],
}

SCORE_NAMES = ["scientific_accuracy_score",
               "communication_clarity_score",
               "objection_handling_score"]


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
            Path(CONFIG["models_dir"]) / "nlp_training_v2.log", mode="w"
        ),
    ],
)
log = logging.getLogger("NLPTrainV2")


# ══════════════════════════════════════════════════════════════════════
# NLPFeatureExtractorV2
# 24 features linguistiques inspirées des PDFs VITAL
# ══════════════════════════════════════════════════════════════════════

class NLPFeatureExtractorV2:
    """
    Extracteur de 24 features linguistiques interprétables.

    Sources :
        Manuel VITAL SA V1   — 6 étapes de la visite médicale
        Méthode A-C-R-V      — Accueillir / Clarifier / Répondre / Valider
        Référentiel 4 niveaux ALIA — KPIs par niveau
        Annexe V2 Scripts    — Mots tueurs, scripts Top Sellers
    """

    # ── A-C-R-V (Manuel section 4.4) ─────────────────────────────────
    ACRV_A = [
        r"\bje comprends\b", r"c'est légitime", r"je vous remercie",
        r"\bpertinent\b", r"\bvalable\b", r"c'est normal",
        r"votre point", r"vous avez raison",
    ]
    ACRV_C = [
        r"quand vous dites", r"c'est plutôt", r"vous pensez à",
        r"pour bien", r"vous craignez", r"vous attendez",
        r"c'est plutôt quoi", r"pour préciser",
    ]
    ACRV_R = [
        r"\bselon\b", r"en pratique", r"\bprofil\b", r"\btest\b",
        r"\bdonnées\b", r"\bfiche\b", r"\brepère\b",
        r"\bconseils\b", r"plan b", r"positionnement",
    ]
    ACRV_V = [
        r"est-ce que", r"\brépond\b", r"d'accord",
        r"pensez-vous", r"seriez-vous", r"\bacceptable\b",
        r"ça vous semble", r"vous semble", r"pour vous",
    ]

    # ── Étape 1 — Permission (Manuel section 4.1) ─────────────────────
    PERMISSION = [
        r"c'est ok pour vous", r"\b2 minutes\b", r"très court",
        r"bonjour docteur", r"je fais court", r"je repasse",
        r"20 secondes",
    ]

    # ── Étape 3 — Synthèse (Manuel section 4.3) ───────────────────────
    SYNTHESE = [
        r"si je résume", r"votre priorité", r"votre attente",
        r"c'est bien ça", r"si je comprends bien", r"donc vous cherchez",
    ]

    # ── Étape 5 — Argumentation (Manuel section 4.5) ──────────────────
    PREUVE = [
        r"\bétude\b", r"\bclinique\b", r"\bdonnées\b",
        r"selon la notice", r"\brepère\b", r"\bpreuve\b",
        r"résultats", r"référence",
    ]
    PROFIL_PATIENT = [
        r"chez ce profil", r"pour ce type", r"ces patients",
        r"profil ciblé", r"chez vos patients", r"sur ces cas",
        r"ce type de patient",
    ]
    AVANTAGE = [
        r"\bbénéfice\b", r"\bavantage\b", r"\bsimple\b",
        r"\bobservance\b", r"\bcure\b", r"\broutine\b",
        r"facilite", r"pratique",
    ]

    # ── Étape 6 — Signaux BIP / Closing (Manuel section 4.6) ──────────
    BIP_SIGNAL = [
        r"\bessayer\b", r"2.?3 patients", r"je repasse",
        r"on est d'accord", r"\bengagement\b", r"test sur",
        r"je reviens", r"plan de test", r"micro-test",
        r"seriez-vous d'accord",
    ]

    # ── Conformité (Référentiel KPI Confirmé/Expert) ───────────────────
    CONFORMITE = [
        r"selon la notice", r"je vérifie", r"dans le cadre",
        r"selon les données", r"selon la fiche",
        r"je vous confirme par", r"je reviens avec une réponse",
        r"je note et je reviens",
    ]

    # ── Mots tueurs (Référentiel + Annexe V2 note de conformité) ──────
    MOTS_TUEURS = [
        r"\bgaranti\b", r"\b100%\b", r"\bguérit\b",
        r"toujours efficace", r"aucun effet secondaire",
        r"meilleur produit", r"\brévolutionnaire\b",
        r"\bmiracle\b", r"sans aucun risque",
        r"prouvé sans réserve", r"cure définitive",
        r"élimine complètement",
    ]

    # ── Questions de découverte (Manuel section 4.2) ───────────────────
    QUESTIONS = [
        r"\?", r"c'est plutôt", r"\bquel\b", r"\bcomment\b",
        r"\bpourquoi\b", r"\bsur quels\b", r"votre critère",
        r"vous voyez", r"le frein",
    ]

    # ── Empathie générale ──────────────────────────────────────────────
    EMPATHIE = [
        r"je comprends", r"vous avez raison",
        r"c'est une bonne question", r"je vous entends",
        r"c'est légitime",
    ]

    # ── Formulations vagues / non-structurées ─────────────────────────
    VAGUE = [
        r"c'est bien", r"pas de problème",
        r"ça marche", r"ne vous inquiétez pas",
        r"c'est très bien",
    ]

    def _match(self, text: str, patterns: List[str]) -> int:
        t = text.lower()
        return int(any(re.search(p, t) for p in patterns))

    def _count(self, text: str, patterns: List[str]) -> int:
        t = text.lower()
        return sum(1 for p in patterns if re.search(p, t))

    def extract(self, objection: str, response: str) -> Dict:
        """
        Extrait les 24 features depuis une paire (objection, réponse).

        Args:
            objection : texte de l'objection du médecin
            response  : texte de la réponse du délégué

        Returns:
            dict : 24 features nommées
        """
        r, o = response, objection

        # ── A-C-R-V (5 features) ──────────────────────────────────────
        acrv_a = self._match(r, self.ACRV_A)
        acrv_c = self._match(r, self.ACRV_C)
        acrv_r = self._match(r, self.ACRV_R)
        acrv_v = self._match(r, self.ACRV_V)
        acrv_completeness = acrv_a + acrv_c + acrv_r + acrv_v

        # ── Étapes visite (4 features) ────────────────────────────────
        has_permission = self._match(r + " " + o, self.PERMISSION)
        has_synthese   = self._match(r, self.SYNTHESE)
        has_preuve     = self._match(r, self.PREUVE)
        has_bip_signal = self._match(r, self.BIP_SIGNAL)

        # ── Argumentation (3 features) ────────────────────────────────
        has_profil_patient = self._match(r, self.PROFIL_PATIENT)
        has_avantage       = self._match(r, self.AVANTAGE)
        has_decouverte     = self._match(o, self.QUESTIONS)

        # ── Conformité (3 features) ───────────────────────────────────
        has_conformite = self._match(r, self.CONFORMITE)
        has_mot_tueur  = int(self._count(r, self.MOTS_TUEURS) > 0)
        n_mots_tueurs  = self._count(r, self.MOTS_TUEURS)

        # ── Structure générale (6 features) ──────────────────────────
        has_empathie   = self._match(r, self.EMPATHIE)
        is_vague       = self._match(r, self.VAGUE)
        question_count = len(re.findall(r"\?", r))
        word_count     = len(r.split())
        sentence_count = len(re.split(r"[.!?]+", r.strip()))
        has_numbers    = int(bool(re.search(r"\d+", r)))

        # ── Scores composites (4 features) ───────────────────────────
        argumentation_richness = (
            has_preuve + has_profil_patient +
            has_avantage + has_decouverte
        ) / 4.0

        conformite_score = max(0.0, has_conformite - n_mots_tueurs * 0.5)

        response_quality_proxy = (
            acrv_completeness / 4.0 * 0.40 +
            has_bip_signal            * 0.20 +
            has_preuve                * 0.20 +
            (1 - is_vague)            * 0.20
        )

        structure_score = (
            has_permission + has_synthese +
            has_preuve     + has_bip_signal
        ) / 4.0

        return {
            # A-C-R-V
            "has_acrv_a"              : acrv_a,
            "has_acrv_c"              : acrv_c,
            "has_acrv_r"              : acrv_r,
            "has_acrv_v"              : acrv_v,
            "acrv_completeness"       : acrv_completeness,
            # Étapes visite
            "has_permission"          : has_permission,
            "has_synthese"            : has_synthese,
            "has_preuve"              : has_preuve,
            "has_bip_signal"          : has_bip_signal,
            # Argumentation
            "has_profil_patient"      : has_profil_patient,
            "has_avantage"            : has_avantage,
            "has_decouverte"          : has_decouverte,
            # Conformité
            "has_conformite"          : has_conformite,
            "has_mot_tueur"           : has_mot_tueur,
            "n_mots_tueurs"           : n_mots_tueurs,
            # Structure
            "has_empathie"            : has_empathie,
            "is_vague"                : is_vague,
            "question_count"          : question_count,
            "word_count"              : word_count,
            "sentence_count"          : sentence_count,
            "has_numbers"             : has_numbers,
            # Scores composites
            "argumentation_richness"  : round(argumentation_richness, 3),
            "conformite_score"        : round(conformite_score, 3),
            "response_quality_proxy"  : round(response_quality_proxy, 3),
        }

    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Extrait les features sur tout un DataFrame. Retourne (N, 24)."""
        rows = [
            list(self.extract(row["objection_text"], row["rep_response"]).values())
            for _, row in df.iterrows()
        ]
        return np.array(rows, dtype=np.float32)

    def feature_names(self) -> List[str]:
        """Retourne les 24 noms de features dans l'ordre d'extraction."""
        return list(self.extract("dummy", "dummy").keys())


# ══════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════

class DataLoader:
    """Charge, prépare et découpe le dataset DS4 V2."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        path = Path(self.cfg["data_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {path}\n"
                "Générez d'abord conversation_transcripts_v2.csv"
            )
        df = pd.read_csv(path, encoding="utf-8-sig")
        log.info(f"Dataset chargé    : {df.shape[0]} conversations × {df.shape[1]} colonnes")
        log.info(f"Qualité           : {df['response_quality'].value_counts().to_dict()}")
        log.info(f"Niveau ALIA       : {df['niveau_alia'].value_counts().to_dict()}")
        log.info(f"Visit format      : {df['visit_format'].value_counts().to_dict()}")
        log.info(f"Conformité        : {df['conformite_flag'].value_counts().to_dict()}")
        return df

    def build_features(
        self, df: pd.DataFrame
    ) -> Tuple[object, np.ndarray, TfidfVectorizer,
               StandardScaler, NLPFeatureExtractorV2]:
        """
        Construit X_combined = TF-IDF (8000) ⊕ features linguistiques (24).

        Returns:
            X_combined, X_linguistic, tfidf, scaler, extractor
        """
        # Texte d'entrée
        df["input_text"] = (
            df["objection_text"] + " [SEP] " + df["rep_response"]
        )

        # TF-IDF
        tfidf_cfg = self.cfg["tfidf"]
        tfidf = TfidfVectorizer(**tfidf_cfg)
        X_tfidf = tfidf.fit_transform(df["input_text"])
        log.info(f"TF-IDF            : {X_tfidf.shape}  (ngram={tfidf_cfg['ngram_range']})")

        # Features linguistiques
        extractor = NLPFeatureExtractorV2()
        X_ling_raw = extractor.extract_batch(df)
        scaler = StandardScaler()
        X_ling_norm = scaler.fit_transform(X_ling_raw)
        X_ling_sparse = csr_matrix(X_ling_norm)
        log.info(f"Features ling.    : {X_ling_sparse.shape}  (24 features)")

        # Combinaison
        X_combined = hstack([X_tfidf, X_ling_sparse])
        log.info(f"X_combined        : {X_combined.shape}")

        return X_combined, X_ling_raw, tfidf, scaler, extractor

    def encode_labels(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, LabelEncoder], Dict[str, np.ndarray]]:
        """Encode toutes les variables cibles."""
        encoders = {}
        labels   = {}

        # T1 — Qualité
        le1 = LabelEncoder()
        labels["t1"] = le1.fit_transform(df["response_quality"])
        encoders["t1"] = le1

        # T2 — Scores (pas d'encodage — continu)
        labels["t2"] = df[SCORE_NAMES].values

        # T3 — Sentiment
        le3 = LabelEncoder()
        labels["t3"] = le3.fit_transform(df["client_sentiment"])
        encoders["t3"] = le3

        # T4 — Niveau ALIA
        le4 = LabelEncoder()
        labels["t4"] = le4.fit_transform(df["niveau_alia"])
        encoders["t4"] = le4

        # T5 — Format visite
        le5 = LabelEncoder()
        labels["t5"] = le5.fit_transform(df["visit_format"])
        encoders["t5"] = le5

        # T6 — Conformité (binaire — pas d'encodage nécessaire)
        labels["t6"] = df["conformite_flag"].values

        log.info(f"Labels encodés    :")
        for k, le in encoders.items():
            log.info(f"  {k} classes : {list(le.classes_)}")

        return encoders, labels

    def split(
        self, X, labels: Dict[str, np.ndarray]
    ) -> Tuple[object, object, Dict, Dict]:
        """Stratified split sur T1 (qualité) pour toutes les tâches."""
        seed = self.cfg["random_seed"]
        ts   = self.cfg["test_size"]

        _, _, idx_tr, idx_te = train_test_split(
            X, np.arange(X.shape[0]),
            test_size=ts, random_state=seed, stratify=labels["t1"]
        )

        X_tr = X[idx_tr]
        X_te = X[idx_te]

        labels_tr = {k: v[idx_tr] for k, v in labels.items()}
        labels_te = {k: v[idx_te] for k, v in labels.items()}

        log.info(f"Split             : train={X_tr.shape[0]} | test={X_te.shape[0]}")
        return X_tr, X_te, labels_tr, labels_te


# ══════════════════════════════════════════════════════════════════════
# NLP SCORING TRAINER
# ══════════════════════════════════════════════════════════════════════

class NLPScoringTrainerV2:
    """
    Orchestration de l'entraînement des 6 tâches NLP.
    Supporte le mode standard et le mode GridSearchCV (--tune).
    """

    def __init__(self, cfg: Dict):
        self.cfg    = cfg
        self.models = {}
        self.report = {}
        self.cv     = StratifiedKFold(
            n_splits=cfg["cv_folds"], shuffle=True,
            random_state=cfg["random_seed"]
        )

    # ── T1 — Classification qualité ───────────────────────────────────
    def train_t1(self, X_tr, y_tr, X_te, y_te,
                 encoders: Dict, tune: bool = False):
        log.info("\n" + "=" * 60)
        log.info("  T1 — Classification Qualité — SVM RBF")
        log.info("=" * 60)

        base = SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=self.cfg["random_seed"]
        )

        if tune:
            log.info("  GridSearchCV en cours...")
            gs = GridSearchCV(
                SVC(kernel="rbf", probability=True,
                    random_state=self.cfg["random_seed"]),
                self.cfg["gs_t1"], cv=self.cv,
                scoring="f1_macro", n_jobs=-1, verbose=0
            )
            gs.fit(X_tr, y_tr)
            model = gs.best_estimator_
            log.info(f"  Best params : {gs.best_params_}")
        else:
            cv_scores = cross_val_score(
                base, X_tr, y_tr, cv=self.cv, scoring="f1_macro"
            )
            log.info(f"  CV F1-Macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            model = base

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        f1     = f1_score(y_te, y_pred, average="macro")
        acc    = accuracy_score(y_te, y_pred)

        log.info(f"  Test F1-Macro : {f1:.4f}")
        log.info(f"  Test Accuracy : {acc:.4f}")
        log.info("\n" + classification_report(
            y_te, y_pred,
            target_names=encoders["t1"].classes_, digits=4
        ))

        self.models["t1"] = model
        self.report["t1"] = {
            "f1_macro": float(f1),
            "accuracy": float(acc),
            "per_class": {
                cls: {
                    "f1": float(f1_score(y_te == i, y_pred == i, average="binary"))
                }
                for i, cls in enumerate(encoders["t1"].classes_)
            },
        }

    # ── T2 — Régression scores ─────────────────────────────────────────
    def train_t2(self, X_tr, y_tr, X_te, y_te):
        log.info("\n" + "=" * 60)
        log.info("  T2 — Régression Scores — Ridge MultiOutput")
        log.info("=" * 60)

        model = MultiOutputRegressor(Ridge(alpha=1.0))
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        w      = np.array(self.cfg["score_weights"])
        ov_r2  = r2_score((y_te * w).sum(axis=1), (y_pred * w).sum(axis=1))
        ov_mae = mean_absolute_error(
            (y_te * w).sum(axis=1), (y_pred * w).sum(axis=1)
        )

        per_score = {}
        for i, name in enumerate(SCORE_NAMES):
            mae = mean_absolute_error(y_te[:, i], y_pred[:, i])
            r2  = r2_score(y_te[:, i], y_pred[:, i])
            log.info(f"  {name:<35} MAE={mae:.4f}  R²={r2:.4f}")
            per_score[name] = {"mae": float(mae), "r2": float(r2)}

        log.info(f"\n  Overall Score (ALIA)               MAE={ov_mae:.4f}  R²={ov_r2:.4f}")
        log.info(f"  Formule : {w[0]}×sci + {w[1]}×clar + {w[2]}×obj")

        self.models["t2"] = model
        self.report["t2"] = {
            "overall_r2"   : float(ov_r2),
            "overall_mae"  : float(ov_mae),
            "per_score"    : per_score,
            "score_weights": list(w),
        }

    # ── T3 — Sentiment ────────────────────────────────────────────────
    def train_t3(self, X_tr, y_tr, X_te, y_te,
                 encoders: Dict, tune: bool = False):
        log.info("\n" + "=" * 60)
        log.info("  T3 — Sentiment Client — Logistic Regression")
        log.info("=" * 60)

        base = LogisticRegression(
            max_iter=2000, random_state=self.cfg["random_seed"], C=1.0
        )

        if tune:
            gs = GridSearchCV(
                LogisticRegression(max_iter=2000,
                                   random_state=self.cfg["random_seed"]),
                self.cfg["gs_t3"], cv=self.cv,
                scoring="f1_macro", n_jobs=-1
            )
            gs.fit(X_tr, y_tr)
            model = gs.best_estimator_
            log.info(f"  Best params : {gs.best_params_}")
        else:
            cv_scores = cross_val_score(
                base, X_tr, y_tr, cv=self.cv, scoring="f1_macro"
            )
            log.info(f"  CV F1-Macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            model = base

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        f1     = f1_score(y_te, y_pred, average="macro")

        log.info(f"  Test F1-Macro : {f1:.4f}")
        log.info("\n" + classification_report(
            y_te, y_pred,
            target_names=encoders["t3"].classes_, digits=4
        ))

        self.models["t3"] = model
        self.report["t3"] = {"f1_macro": float(f1)}

    # ── T4 — Niveau ALIA ─────────────────────────────────────────────
    def train_t4(self, X_tr, y_tr, X_te, y_te,
                 encoders: Dict, tune: bool = False):
        log.info("\n" + "=" * 60)
        log.info("  T4 — Niveau ALIA — SVM RBF (GridSearch recommandé)")
        log.info("=" * 60)

        base = SVC(
            kernel="rbf", C=5.0, gamma="scale",
            probability=True, random_state=self.cfg["random_seed"]
        )

        if tune:
            log.info("  GridSearchCV en cours...")
            gs = GridSearchCV(
                SVC(kernel="rbf", probability=True,
                    random_state=self.cfg["random_seed"]),
                self.cfg["gs_t4"], cv=self.cv,
                scoring="f1_macro", n_jobs=-1, verbose=0
            )
            gs.fit(X_tr, y_tr)
            model = gs.best_estimator_
            log.info(f"  Best params : {gs.best_params_}")
            log.info(f"  Best CV F1  : {gs.best_score_:.4f}")
        else:
            cv_scores = cross_val_score(
                base, X_tr, y_tr, cv=self.cv, scoring="f1_macro"
            )
            log.info(f"  CV F1-Macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            model = base

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        f1     = f1_score(y_te, y_pred, average="macro")
        acc    = accuracy_score(y_te, y_pred)

        log.info(f"  Test F1-Macro : {f1:.4f}")
        log.info(f"  Test Accuracy : {acc:.4f}")
        log.info("\n" + classification_report(
            y_te, y_pred,
            target_names=encoders["t4"].classes_, digits=4
        ))

        self.models["t4"] = model
        self.report["t4"] = {
            "f1_macro": float(f1),
            "accuracy": float(acc),
        }

    # ── T5 — Format visite ────────────────────────────────────────────
    def train_t5(self, X_tr, y_tr, X_te, y_te, encoders: Dict):
        log.info("\n" + "=" * 60)
        log.info("  T5 — Format Visite — Logistic Regression")
        log.info("=" * 60)

        model = LogisticRegression(
            max_iter=2000, random_state=self.cfg["random_seed"], C=1.0
        )
        cv_scores = cross_val_score(
            model, X_tr, y_tr, cv=self.cv, scoring="f1_macro"
        )
        log.info(f"  CV F1-Macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        f1     = f1_score(y_te, y_pred, average="macro")

        log.info(f"  Test F1-Macro : {f1:.4f}")
        log.info("\n" + classification_report(
            y_te, y_pred,
            target_names=encoders["t5"].classes_, digits=4
        ))

        self.models["t5"] = model
        self.report["t5"] = {"f1_macro": float(f1)}

    # ── T6 — Conformité ───────────────────────────────────────────────
    def train_t6(self, X_tr, y_tr, X_te, y_te):
        log.info("\n" + "=" * 60)
        log.info("  T6 — Conformité — Règle regex + LogReg backup")
        log.info("=" * 60)

        # Modèle ML backup (utilisé si règle incertaine)
        model = LogisticRegression(
            max_iter=2000, random_state=self.cfg["random_seed"], C=1.0
        )
        cv_scores = cross_val_score(
            model, X_tr, y_tr, cv=self.cv, scoring="f1_macro"
        )
        log.info(f"  CV F1-Macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        f1     = f1_score(y_te, y_pred, average="macro")
        acc    = accuracy_score(y_te, y_pred)

        log.info(f"  Test F1-Macro : {f1:.4f}")
        log.info(f"  Test Accuracy : {acc:.4f}")
        log.info("  Note : la règle déterministe (regex mots tueurs) est")
        log.info("         prioritaire sur le modèle ML pour T6.")

        self.models["t6"] = model
        self.report["t6"] = {
            "f1_macro": float(f1),
            "accuracy": float(acc),
        }

    # ── Save ──────────────────────────────────────────────────────────
    def save(self, tfidf, scaler, extractor, encoders, cfg):
        mdir = Path(cfg["models_dir"])

        # Bundle principal
        bundle = {
            "tfidf"          : tfidf,
            "scaler"         : scaler,
            "extractor"      : extractor,
            "encoders"       : encoders,
            "models"         : self.models,
            "config"         : cfg,
            "feature_names"  : extractor.feature_names(),
            "class_names"    : CLASS_NAMES,
            "score_names"    : SCORE_NAMES,
            "seuils_alia"    : cfg["seuils_alia"],
            "trained_at"     : datetime.now().isoformat(),
            "version"        : "2.0.0",
        }
        bundle_path = mdir / "nlp_scoring_bundle_v2.pkl"
        joblib.dump(bundle, bundle_path)
        log.info(f"\n  ✅  Bundle sauvegardé → {bundle_path}")

        # Rapport
        report_path = mdir / "nlp_training_report_v2.json"
        full_report = {
            "tasks"     : self.report,
            "trained_at": datetime.now().isoformat(),
            "version"   : "2.0.0",
            "dataset"   : cfg["data_path"],
            "n_features": 8000 + 24,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        log.info(f"  ✅  Rapport sauvegardé → {report_path}")

        # Récap artifacts
        log.info("\n  Artifacts :")
        for p in sorted(mdir.iterdir()):
            if p.suffix in [".pkl", ".json", ".log"]:
                log.info(
                    f"    {p.name:<45}  {p.stat().st_size/1024:>7.1f} KB"
                )


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: Dict, tune: bool = False, eval_only: bool = False):
    np.random.seed(cfg["random_seed"])
    t_start = time.time()

    log.info("=" * 70)
    log.info("  ALIA — NLP Scoring Training Pipeline V2")
    log.info(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Dataset  : {cfg['data_path']}")
    log.info(f"  Tune     : {tune}")
    log.info(f"  6 tâches : T1 qualité | T2 scores | T3 sentiment |"
             f" T4 niveau | T5 format | T6 conformité")
    log.info("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    loader = DataLoader(cfg)
    df     = loader.load()

    if eval_only:
        bundle_path = Path(cfg["models_dir"]) / "nlp_scoring_bundle_v2.pkl"
        log.info(f"Eval-only — chargement depuis {bundle_path}")
        bundle = joblib.load(bundle_path)
        _run_demo(bundle)
        return

    X, X_ling, tfidf, scaler, extractor = loader.build_features(df)
    encoders, labels = loader.encode_labels(df)
    X_tr, X_te, labels_tr, labels_te    = loader.split(X, labels)

    # ── Train ─────────────────────────────────────────────────────────
    trainer = NLPScoringTrainerV2(cfg)
    trainer.train_t1(X_tr, labels_tr["t1"], X_te, labels_te["t1"], encoders, tune)
    trainer.train_t2(X_tr, labels_tr["t2"], X_te, labels_te["t2"])
    trainer.train_t3(X_tr, labels_tr["t3"], X_te, labels_te["t3"], encoders, tune)
    trainer.train_t4(X_tr, labels_tr["t4"], X_te, labels_te["t4"], encoders, tune)
    trainer.train_t5(X_tr, labels_tr["t5"], X_te, labels_te["t5"], encoders)
    trainer.train_t6(X_tr, labels_tr["t6"], X_te, labels_te["t6"])

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  BENCHMARK FINAL — 6 tâches NLP Scoring V2")
    log.info("=" * 70)
    log.info(f"  {'Tâche':<8}  {'Description':<20}  {'Métrique':>12}  Statut")
    log.info("  " + "-" * 58)
    summary = [
        ("T1", "Qualité",     trainer.report["t1"]["f1_macro"],  "F1-Macro"),
        ("T2", "Scores R²",   trainer.report["t2"]["overall_r2"],"R²"),
        ("T3", "Sentiment",   trainer.report["t3"]["f1_macro"],  "F1-Macro"),
        ("T4", "Niveau ALIA", trainer.report["t4"]["f1_macro"],  "F1-Macro"),
        ("T5", "Format",      trainer.report["t5"]["f1_macro"],  "F1-Macro"),
        ("T6", "Conformité",  trainer.report["t6"]["f1_macro"],  "F1-Macro"),
    ]
    for t, desc, val, metric in summary:
        st = "✅" if val >= 0.80 else "⚠️ " if val >= 0.65 else "📊"
        log.info(f"  {t:<8}  {desc:<20}  {val:>10.4f}    {st}")
    log.info("=" * 70)

    # ── Save ──────────────────────────────────────────────────────────
    trainer.save(tfidf, scaler, extractor, encoders, cfg)

    # ── Demo ──────────────────────────────────────────────────────────
    bundle = {
        "tfidf": tfidf, "scaler": scaler, "extractor": extractor,
        "encoders": encoders, "models": trainer.models,
        "config": cfg, "seuils_alia": cfg["seuils_alia"],
    }
    _run_demo(bundle)

    log.info(f"\n✅  Pipeline complet en {time.time()-t_start:.1f}s")


def _run_demo(bundle: Dict):
    """Démo rapide — 3 cas test représentatifs."""
    log.info("\n" + "=" * 60)
    log.info("  DÉMO INFÉRENCE — 3 cas test")
    log.info("=" * 60)

    test_cases = [
        (
            "Pas convaincu par ce produit",
            "Je comprends tout à fait votre préoccupation. Quand vous dites pas "
            "convaincu, c'est plutôt sur la composition ou la tolérance ? "
            "Selon les données disponibles, 1 repère ciblé et proposition de test "
            "sur 2 à 3 patients. Est-ce que ce repère suffit pour envisager un test ? "
            "(selon la notice)",
            "Excellent",
        ),
        (
            "J'ai mes habitudes avec un autre produit",
            "Je comprends. Sur ce type de patients, on peut positionner en plan B "
            "sur les cas moins satisfaits. Je vous laisse une fiche et je repasse.",
            "Bon",
        ),
        (
            "Pas le temps pour une présentation",
            "C'est le meilleur produit de notre gamme. Je vous garantis que vos "
            "patients seront toujours satisfaits. Aucun effet secondaire signalé.",
            "Faible",
        ),
    ]

    tfidf = bundle["tfidf"]; scaler = bundle["scaler"]
    ext   = bundle["extractor"]; enc = bundle["encoders"]
    m     = bundle["models"]

    for obj, resp, expected in test_cases:
        text   = obj + " [SEP] " + resp
        feats  = list(ext.extract(obj, resp).values())
        x_tf   = tfidf.transform([text])
        x_ling = csr_matrix(scaler.transform([feats]))
        x      = hstack([x_tf, x_ling])

        q     = enc["t1"].inverse_transform(m["t1"].predict(x))[0]
        niv   = enc["t4"].inverse_transform(m["t4"].predict(x))[0]
        fmt   = enc["t5"].inverse_transform(m["t5"].predict(x))[0]
        sent  = enc["t3"].inverse_transform(m["t3"].predict(x))[0]
        conf  = int(m["t6"].predict(x)[0])
        sc2   = m["t2"].predict(x)[0]
        w     = np.array(bundle["config"]["score_weights"])
        ov    = float((sc2 * w).sum())

        match = "✅" if q == expected else "❌"
        log.info(f"\n  {match}  Attendu={expected:<12} Prédit={q}")
        log.info(f"     Niveau ALIA  : {niv}  |  Format : {fmt}  |  Sentiment : {sent}")
        log.info(f"     Overall score: {ov:.2f}/10  |  Conformité : {'✅ OK' if conf else '🚨 MOT TUEUR'}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="ALIA NLP Scoring V2 — Training Pipeline"
    )
    p.add_argument("--data",       default=CONFIG["data_path"],
                   help="Chemin vers le dataset CSV")
    p.add_argument("--models-dir", default=CONFIG["models_dir"],
                   help="Répertoire de sauvegarde")
    p.add_argument("--tune",       action="store_true",
                   help="Activer GridSearchCV sur T1, T3, T4")
    p.add_argument("--eval-only",  action="store_true",
                   help="Évaluation uniquement (charge le bundle existant)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["data_path"]  = args.data
    CONFIG["models_dir"] = args.models_dir
    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    run_pipeline(CONFIG, tune=args.tune, eval_only=args.eval_only)
