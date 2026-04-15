"""
nlp_scoring_model.py
====================
DSO 1 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Classe d'inférence production-ready pour le moteur NLP de scoring.

Ce fichier est le SEUL importé par l'avatar en production.
Il ne contient aucun code d'entraînement — uniquement l'interface de prédiction.

Usage (depuis l'avatar DSO 1) :
    from nlp_scoring_model import NLPScoringModel

    model = NLPScoringModel.load("models/nlp_scoring_model.pkl")

    result = model.predict(
        objection="J'ai des doutes sur l'efficacité.",
        response="Je comprends. Des études cliniques montrent 82% d'efficacité..."
    )

    print(result["quality"])        # "Excellent"
    print(result["overall_score"])  # 8.42
    print(result["feedback"])       # ["Réponse de qualité..."]

Extension vers transformers (CamemBERT / XLM-R) :
    Remplacer _load_bundle() par un loader HuggingFace.
    L'interface predict() reste identique — aucun changement côté avatar.

Author  : CYBER SHADE — ALIA Project
Version : 1.0.0
"""

# ── Standard library ──────────────────────────────────────────────────────
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

# ── Scientific stack ──────────────────────────────────────────────────────
import joblib
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NLPScoringModel")

# ── ALIA scoring formula weights ──────────────────────────────────────────
SCORE_WEIGHTS = np.array([0.25, 0.30, 0.45])   # scientific / clarity / objection
SCORE_COLS    = [
    "scientific_accuracy_score",
    "communication_clarity_score",
    "objection_handling_score",
]
SCORE_LABELS  = [
    "scientific_accuracy",
    "communication_clarity",
    "objection_handling",
]

# ── Default model path ────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/nlp_scoring_model.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# NLP FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class NLPFeatureExtractor:
    """
    Extrait des features linguistiques interprétables depuis les réponses texte.
    Utilisée pour :
      1. Générer le feedback coaching de l'avatar en temps réel
      2. Fournir une couche d'interprétabilité sur les prédictions du modèle
    """

    # ── Keyword lists ──────────────────────────────────────────────────────
    EMPATHY_KW = [
        "comprends", "entends", "légitime", "inquiétude",
        "préoccupation", "tout à fait", "bonne question",
    ]
    DATA_PATTERN = re.compile(
        r"\d+\s*%|étude|clinique|données|résultat|prouve|littérature|publication",
        re.IGNORECASE,
    )
    SOLUTION_KW = [
        "vous propose", "permettez", "montrer",
        "présenter", "solution", "répondre à",
    ]
    VAGUE_KW = [
        "bien", "fiable", "sérieux", "faites-moi confiance",
        "bon produit", "faites confiance",
    ]

    def extract(self, text: str) -> Dict:
        """
        Extrait 10 features linguistiques depuis une réponse texte.

        Returns:
            dict avec features booléennes et quantitatives
        """
        t = text.lower()

        has_empathy    = int(any(w in t for w in self.EMPATHY_KW))
        has_data_args  = int(bool(self.DATA_PATTERN.search(t)))
        has_followup_q = int("?" in text)
        has_solution   = int(any(w in t for w in self.SOLUTION_KW))
        is_vague       = int(any(w in t for w in self.VAGUE_KW))

        return {
            "has_empathy"           : has_empathy,
            "has_data_args"         : has_data_args,
            "has_followup_q"        : has_followup_q,
            "has_solution"          : has_solution,
            "is_vague"              : is_vague,
            "word_count"            : len(text.split()),
            "response_length"       : len(text),
            "sentence_count"        : (text.count(".") +
                                       text.count("?") +
                                       text.count("!")),
            "has_numbers"           : int(bool(re.search(r"\d+", text))),
            "argumentation_richness": (has_empathy + has_data_args +
                                       has_followup_q + has_solution),
        }

    def generate_feedback(self, features: Dict) -> List[str]:
        """
        Génère des recommandations de coaching depuis les features NLP.
        Retourné à l'avatar pour guider le délégué après chaque réponse.

        Returns:
            Liste de hints textuels (vide si réponse de qualité)
        """
        hints = []

        if not features["has_empathy"]:
            hints.append(
                "Ajouter une phrase d'empathie en début de réponse "
                "(ex: 'Je comprends tout à fait votre préoccupation')"
            )
        if not features["has_data_args"]:
            hints.append(
                "Inclure des données chiffrées ou références cliniques "
                "(ex: 'Des études montrent X% d'efficacité après N semaines')"
            )
        if not features["has_followup_q"]:
            hints.append(
                "Terminer par une question ouverte pour engager le client "
                "(ex: 'Quels sont vos critères prioritaires ?')"
            )
        if features["is_vague"]:
            hints.append(
                "Éviter les formulations vagues — être plus précis et concret"
            )
        if features["word_count"] < 30:
            hints.append(
                "Développer davantage l'argumentation "
                f"({features['word_count']} mots — minimum recommandé : 30)"
            )
        if features["argumentation_richness"] < 2:
            hints.append(
                "Enrichir la structure de réponse : "
                "empathie → données → solution → question de suivi"
            )

        return hints if hints else [
            "Réponse de qualité — aucune amélioration critique détectée [OK]"
        ]


# ══════════════════════════════════════════════════════════════════════════════
# NLPScoringModel — CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class NLPScoringModel:
    """
    Classe d'inférence production-ready pour le scoring NLP des conversations.

    Encapsule les 3 tâches de scoring :
        T1 — Classification qualité réponse  (Excellent / Bon / Faible)
        T2 — Régression scores (scientific / clarity / objection)
        T3 — Prédiction sentiment client     (Positif / Neutre / Négatif)

    Attributes:
        version     (str)  : version du modèle
        trained_at  (str)  : timestamp d'entraînement
    """

    def __init__(self, bundle: Dict):
        # ── Composants T1 / T2 / T3 ──────────────────────────────────────
        self._t1       = bundle["t1_pipeline"]     # Pipeline TF-IDF + SVM
        self._t2       = bundle["t2_artifact"]     # {"tfidf": ..., "reg": ...}
        self._t3       = bundle["t3_pipeline"]     # Pipeline TF-IDF + LogReg
        self._le_q     = bundle["le_quality"]      # LabelEncoder qualité
        self._le_s     = bundle["le_sentiment"]    # LabelEncoder sentiment

        # ── Metadata ──────────────────────────────────────────────────────
        self.version    = bundle.get("version", "unknown")
        self.trained_at = bundle.get("trained_at", "unknown")
        self._weights   = SCORE_WEIGHTS
        self._extractor = NLPFeatureExtractor()

    # ── Class methods ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "NLPScoringModel":
        """
        Charge le modèle depuis un fichier .pkl.

        Args:
            path : chemin vers nlp_scoring_model.pkl

        Returns:
            Instance NLPScoringModel prête à l'inférence

        Raises:
            FileNotFoundError : si le fichier n'existe pas
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found : {path}\n"
                f"Run 'python nlp_scoring_train.py' to generate it."
            )

        bundle = joblib.load(model_path)
        log.info(
            f"NLPScoringModel loaded — "
            f"v{bundle.get('version', '?')} | "
            f"trained: {bundle.get('trained_at', '?')}"
        )
        return cls(bundle)

    # ── Core inference ─────────────────────────────────────────────────────

    def predict(self, objection: str, response: str) -> Dict:
        """
        Pipeline d'inférence complet pour une conversation.

        Args:
            objection (str) : Texte de l'objection du professionnel de santé
            response  (str) : Réponse du délégué médical

        Returns:
            dict :
                quality        (str)   : "Excellent" | "Bon" | "Faible"
                quality_proba  (dict)  : probabilités par classe
                scores         (dict)  : 3 scores dimensionnels (0–10)
                overall_score  (float) : score global ALIA (0–10)
                sentiment      (str)   : "Positif" | "Neutre" | "Négatif"
                feedback       (list)  : hints de coaching pour l'avatar
                nlp_features   (dict)  : features linguistiques (interprétabilité)
                competency_flag (bool) : True si overall_score >= 7.0
        """
        if not objection or not response:
            raise ValueError("objection and response must be non-empty strings")

        text = objection.strip() + " [SEP] " + response.strip()

        # ── T1 : Quality classification ───────────────────────────────────
        quality_enc   = self._t1.predict([text])[0]
        quality       = self._le_q.inverse_transform([quality_enc])[0]
        quality_proba = {
            cls: round(float(prob), 4)
            for cls, prob in zip(
                self._le_q.classes_,
                self._t1.predict_proba([text])[0]
            )
        }

        # ── T2 : Multi-dimensional score regression ───────────────────────
        vec        = self._t2["tfidf"].transform([text])
        raw_scores = self._t2["reg"].predict(vec)[0]
        scores     = np.clip(raw_scores, 0.0, 10.0)
        overall    = float(scores @ self._weights)

        scores_dict = {
            label: round(float(s), 2)
            for label, s in zip(SCORE_LABELS, scores)
        }

        # ── T3 : Client sentiment prediction ─────────────────────────────
        sentiment_enc = self._t3.predict([text])[0]
        sentiment     = self._le_s.inverse_transform([sentiment_enc])[0]

        # ── NLP features + feedback ───────────────────────────────────────
        nlp_features = self._extractor.extract(response)
        feedback     = self._extractor.generate_feedback(nlp_features)

        return {
            "quality"         : quality,
            "quality_proba"   : quality_proba,
            "scores"          : scores_dict,
            "overall_score"   : round(overall, 2),
            "sentiment"       : sentiment,
            "feedback"        : feedback,
            "nlp_features"    : nlp_features,
            "competency_flag" : overall >= 7.0,
        }

    def predict_batch(
        self,
        conversations: List[Dict[str, str]],
    ) -> List[Dict]:
        """
        Inférence en batch pour plusieurs conversations.

        Args:
            conversations : [
                {"objection": "...", "response": "..."},
                ...
            ]

        Returns:
            Liste de dicts de prédiction (même format que predict())
        """
        return [
            self.predict(c["objection"], c["response"])
            for c in conversations
        ]

    # ── Utility methods ────────────────────────────────────────────────────

    def score_summary(self, result: Dict, verbose: bool = False) -> str:
        """
        Génère un résumé lisible d'un résultat de prédiction.

        Args:
            result  : dict retourné par predict()
            verbose : inclure les features NLP si True

        Returns:
            Résumé formaté en string
        """
        competency = "[OK] COMPÉTENT" if result["competency_flag"] else "[!]  EN FORMATION"
        lines = [
            f"┌─ NLP Scoring Result {'─'*35}",
            f"│  Quality       : {result['quality']}  "
            f"(proba: {result['quality_proba']})",
            f"│  Overall Score : {result['overall_score']:.1f} / 10  [{competency}]",
            f"│  Sentiment     : {result['sentiment']}",
            f"│  Scores :",
            *[f"│    {k:<30}: {v}" for k, v in result["scores"].items()],
            f"│  Feedback ({len(result['feedback'])} hints) :",
            *[f"│    • {h}" for h in result["feedback"]],
        ]
        if verbose:
            lines += [
                f"│  NLP Features :",
                *[f"│    {k:<30}: {v}"
                  for k, v in result["nlp_features"].items()],
            ]
        lines.append(f"└{'─'*55}")
        return "\n".join(lines)

    def get_quality_threshold(self) -> Dict:
        """
        Retourne les seuils de compétence utilisés par le moteur adaptatif.
        Utilisé par l'Adaptive Learning Engine (DSO 1).
        """
        return {
            "competency"          : 7.0,   # overall_score >= 7.0
            "certification_ready" : 7.5,   # overall_score >= 7.5 on Advanced level
            "coaching_required"   : 6.0,   # overall_score < 6.0
        }

    def __repr__(self) -> str:
        return (
            f"NLPScoringModel("
            f"version='{self.version}', "
            f"trained_at='{self.trained_at}')"
        )


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to validate inference
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  NLPScoringModel — Quick Inference Test")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────
    model = NLPScoringModel.load(DEFAULT_MODEL_PATH)
    print(f"\n  {model}\n")

    # ── Test cases ────────────────────────────────────────────────────────
    test_cases = [
        {
            "label"    : "EXCELLENT",
            "objection": "J'ai des doutes sur l'efficacité. Avez-vous des preuves ?",
            "response" : (
                "Je comprends tout à fait votre préoccupation, c'est une question "
                "légitime. Des études cliniques publiées démontrent une efficacité "
                "de 82% après 6 semaines. Permettez-moi de vous montrer les données "
                "comparatives. Quels critères sont prioritaires pour vous ?"
            ),
        },
        {
            "label"    : "BON",
            "objection": "Le concurrent propose quelque chose de similaire moins cher.",
            "response" : (
                "Notre produit a eu de très bons retours de la part des pharmaciens "
                "de la région. C'est une solution que je recommande régulièrement."
            ),
        },
        {
            "label"    : "FAIBLE",
            "objection": "Le prix est trop élevé.",
            "response" : "Notre marque est fiable. Faites-moi confiance.",
        },
    ]

    for tc in test_cases:
        print(f"\n{'─'*60}")
        print(f"  Expected : {tc['label']}")
        print(f"  Response : \"{tc['response'][:60]}...\"")
        print()
        result = model.predict(tc["objection"], tc["response"])
        print(model.score_summary(result, verbose=False))

    # ── Batch inference demo ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Batch inference (3 conversations) :")
    batch = [{"objection": tc["objection"], "response": tc["response"]}
             for tc in test_cases]
    batch_results = model.predict_batch(batch)
    for i, r in enumerate(batch_results):
        flag = "[OK]" if r["competency_flag"] else "[!]"
        print(f"  [{i+1}] quality={r['quality']:<12} "
              f"overall={r['overall_score']:.1f}  "
              f"sentiment={r['sentiment']:<8}  {flag}")

    print(f"\n[OK]  NLPScoringModel — inference validated")
    print(f"    Import with : from nlp_scoring_model import NLPScoringModel")
