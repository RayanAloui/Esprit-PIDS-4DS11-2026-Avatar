"""
nlp_scoring_model_v2.py
=======================
DSO 1 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Classe d'inférence production pour le NLP Scoring Model V2.

Ce fichier est le SEUL importé par l'avatar Django et le dashboard.
Il ne contient aucun code d'entraînement.

Usage :
    from nlp_scoring_model_v2 import NLPScoringModel

    model  = NLPScoringModel.load("models/nlp_scoring_bundle_v2.pkl")
    result = model.predict(
        objection = "Pas convaincu par ce produit",
        response  = "Je comprends. Selon les données..."
    )

    # result = {
    #   "quality"          : "Excellent",
    #   "overall_score"    : 8.34,
    #   "niveau_alia"      : "Confirmé",
    #   "conformite"       : True,
    #   "acrv_score"       : 4,
    #   "visit_format"     : "Standard",
    #   "sentiment"        : "Positif",
    #   "feedback_coaching": [...],
    #   "next_level"       : {...},
    #   "competency_flag"  : True,
    #   ...
    # }

Author  : CYBER SHADE — ALIA Project
Version : 2.0.0
"""

# ── Standard library ──────────────────────────────────────────────────
import logging
import re
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

# ── Scientific stack ──────────────────────────────────────────────────
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NLPScoringModel")

# ── Constants ─────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/nlp_scoring_bundle_v2.pkl"

# Seuils officiels (Référentiel des 4 niveaux ALIA)
SEUILS_ALIA = {
    "Débutant" : 7.0,
    "Junior"   : 8.0,
    "Confirmé" : 9.0,
    "Expert"   : 10.0,
}

NIVEAU_ORDRE = ["Débutant", "Junior", "Confirmé", "Expert"]

PROCHAIN_NIVEAU = {
    "Débutant" : "Junior",
    "Junior"   : "Confirmé",
    "Confirmé" : "Expert",
    "Expert"   : None,
}

# Couleurs niveau pour le dashboard (hex)
NIVEAU_COLORS = {
    "Expert"   : "#1a237e",
    "Confirmé" : "#1976D2",
    "Junior"   : "#42A5F5",
    "Débutant" : "#90CAF9",
}

# Couleurs qualité
QUALITY_COLORS = {
    "Excellent": "#2ecc71",
    "Bon"      : "#f39c12",
    "Faible"   : "#e74c3c",
}

# ── Feedback coaching par niveau (Référentiel 4 niveaux) ──────────────
FEEDBACK_TEMPLATES = {
    "acrv_incomplet": {
        "Débutant" : "Structurez votre réponse en 4 étapes : Accueil → Clarification → Réponse → Validation.",
        "Junior"   : "Votre méthode A-C-R-V est partielle. Pensez à valider systématiquement avec une question de confirmation.",
        "Confirmé" : "Affinez la phase de Clarification : posez une question ouverte avant de répondre.",
        "Expert"   : "Méthode A-C-R-V maîtrisée — vérifiez la fluidité de la transition Réponse → Validation.",
    },
    "mot_tueur": {
        "Débutant" : "⚠️ CONFORMITÉ : Évitez les promesses ('garanti', 'toujours efficace'). Restez factuel.",
        "Junior"   : "⚠️ CONFORMITÉ : Supprimez les affirmations non vérifiées. Reformulez avec 'selon la notice'.",
        "Confirmé" : "⚠️ CONFORMITÉ CRITIQUE : Zéro surpromesse requis à votre niveau. Corrigez immédiatement.",
        "Expert"   : "⚠️ CONFORMITÉ CRITIQUE : Non acceptable au niveau Expert. Revoir le cadre réglementaire.",
    },
    "pas_de_bip": {
        "Débutant" : "Terminez chaque réponse par un micro-engagement simple ('Je vous laisse une fiche ?').",
        "Junior"   : "Détectez les signaux BIP pour conclure au bon moment (questions détaillées, intérêt manifesté).",
        "Confirmé" : "Proposez systématiquement un test sur 2-3 patients pour concrétiser l'engagement.",
        "Expert"   : "Orchestrez le closing en lien avec le cycle long : test → retour → deuxième engagement.",
    },
    "pas_de_preuve": {
        "Débutant" : "Appuyez votre réponse sur au moins 1 repère factuel (selon la notice / données disponibles).",
        "Junior"   : "Citez 1 à 2 preuves courtes pour renforcer votre argumentation.",
        "Confirmé" : "Résumez une preuve en 20 secondes max avec prudence : 'selon les données disponibles'.",
        "Expert"   : "Comparez les niveaux de preuve et citez la référence à partager si demandé.",
    },
    "vague": {
        "Débutant" : "Évitez les formules vagues ('c'est bien', 'pas de problème'). Soyez précis et structuré.",
        "Junior"   : "Remplacez les réponses génériques par une argumentation besoin → avantage → preuve → usage.",
        "Confirmé" : "Votre réponse manque de précision. Adaptez le discours au profil médecin (4 styles SONCAS).",
        "Expert"   : "Réponse insuffisamment précise pour votre niveau. Appliquez l'argumentation haute précision.",
    },
    "pas_de_profil": {
        "Junior"   : "Segmentez votre réponse par profil patient ('chez ce type de patient...').",
        "Confirmé" : "Différenciez l'argumentation par profil patient et spécialité médicale.",
        "Expert"   : "Maîtrisez la segmentation inter-gammes pour chaque profil de prescripteur.",
    },
    "score_faible": {
        "Débutant" : "Travaillez la structure des 6 étapes. Score cible : 7.0/10 pour passer au niveau Junior.",
        "Junior"   : "Renforcez la gestion des objections avec A-C-R-V. Score cible : 8.0/10 pour le niveau Confirmé.",
        "Confirmé" : "Améliorez la précision et la conformité. Score cible : 9.0/10 pour le niveau Expert.",
        "Expert"   : "Maintenez un score ≥ 9.0 sur toutes les dimensions pour rester au niveau Expert.",
    },
}

# Messages positifs par niveau
FEEDBACK_POSITIF = {
    "Expert"   : "Excellente réponse — niveau Expert maintenu ✅",
    "Confirmé" : "Bonne réponse — continuez à affiner pour progresser vers Expert.",
    "Junior"   : "Réponse correcte — travaillez la conformité et la segmentation pour atteindre Confirmé.",
    "Débutant" : "Bonne tentative — appliquez la méthode A-C-R-V pour progresser.",
}

# KPIs par niveau (Référentiel officiel)
KPIS_ALIA = {
    "Débutant": {
        "score_min"         : 0.0,
        "score_cible"       : 7.0,
        "structure_min"     : 0.90,
        "engagements_min"   : 0.50,
        "conformite_requise": True,
    },
    "Junior": {
        "score_min"         : 7.0,
        "score_cible"       : 8.0,
        "acrv_min"          : 0.70,
        "engagement_min"    : 0.60,
        "crm_min"           : 0.80,
        "conformite_requise": True,
    },
    "Confirmé": {
        "score_min"         : 8.0,
        "score_cible"       : 9.0,
        "adaptation_min"    : 0.80,
        "engagement_min"    : 0.70,
        "mots_tueurs_max"   : 0,
        "conformite_requise": True,
    },
    "Expert": {
        "score_min"         : 9.0,
        "score_cible"       : 10.0,
        "visites_difficiles": 0.70,
        "cycle_long_min"    : 0.60,
        "conformite_requise": True,
    },
}


# ══════════════════════════════════════════════════════════════════════
# NLPFeatureExtractorV2 — dupliquée ici pour autonomie du fichier
# ══════════════════════════════════════════════════════════════════════

class NLPFeatureExtractorV2:
    """
    Extracteur de 24 features linguistiques.
    Dupliqué ici pour que nlp_scoring_model_v2.py soit autonome
    (importable sans nlp_scoring_train_v2.py).
    """

    ACRV_A = [r"\bje comprends\b", r"c'est légitime", r"je vous remercie",
              r"\bpertinent\b", r"\bvalable\b", r"c'est normal",
              r"votre point", r"vous avez raison"]
    ACRV_C = [r"quand vous dites", r"c'est plutôt", r"vous pensez à",
              r"pour bien", r"vous craignez", r"vous attendez",
              r"c'est plutôt quoi", r"pour préciser"]
    ACRV_R = [r"\bselon\b", r"en pratique", r"\bprofil\b", r"\btest\b",
              r"\bdonnées\b", r"\bfiche\b", r"\brepère\b",
              r"\bconseils\b", r"plan b", r"positionnement"]
    ACRV_V = [r"est-ce que", r"\brépond\b", r"d'accord",
              r"pensez-vous", r"seriez-vous", r"\bacceptable\b",
              r"ça vous semble", r"vous semble", r"pour vous"]
    PERMISSION  = [r"c'est ok pour vous", r"\b2 minutes\b", r"très court",
                   r"bonjour docteur", r"je fais court", r"20 secondes"]
    SYNTHESE    = [r"si je résume", r"votre priorité", r"votre attente",
                   r"c'est bien ça", r"si je comprends bien", r"donc vous cherchez"]
    PREUVE      = [r"\bétude\b", r"\bclinique\b", r"\bdonnées\b",
                   r"selon la notice", r"\brepère\b", r"\bpreuve\b",
                   r"résultats", r"référence"]
    PROFIL_PATIENT = [r"chez ce profil", r"pour ce type", r"ces patients",
                      r"profil ciblé", r"chez vos patients", r"sur ces cas",
                      r"ce type de patient"]
    AVANTAGE    = [r"\bbénéfice\b", r"\bavantage\b", r"\bsimple\b",
                   r"\bobservance\b", r"\bcure\b", r"\broutine\b",
                   r"facilite", r"pratique"]
    BIP_SIGNAL  = [r"\bessayer\b", r"2.?3 patients", r"je repasse",
                   r"on est d'accord", r"\bengagement\b", r"test sur",
                   r"je reviens", r"plan de test", r"micro-test",
                   r"seriez-vous d'accord"]
    CONFORMITE  = [r"selon la notice", r"je vérifie", r"dans le cadre",
                   r"selon les données", r"selon la fiche",
                   r"je vous confirme par", r"je reviens avec une réponse",
                   r"je note et je reviens"]
    MOTS_TUEURS = [r"\bgaranti\b", r"\b100%\b", r"\bguérit\b",
                   r"toujours efficace", r"aucun effet secondaire",
                   r"meilleur produit", r"\brévolutionnaire\b",
                   r"\bmiracle\b", r"sans aucun risque",
                   r"prouvé sans réserve", r"cure définitive",
                   r"élimine complètement"]
    EMPATHIE    = [r"je comprends", r"vous avez raison",
                   r"c'est une bonne question", r"je vous entends",
                   r"c'est légitime"]
    VAGUE       = [r"c'est bien", r"pas de problème",
                   r"ça marche", r"ne vous inquiétez pas", r"c'est très bien"]
    QUESTIONS   = [r"\?", r"c'est plutôt", r"\bquel\b", r"\bcomment\b",
                   r"\bpourquoi\b", r"\bsur quels\b", r"votre critère",
                   r"vous voyez", r"le frein"]

    def _m(self, t, p): return int(any(re.search(x, t.lower()) for x in p))
    def _n(self, t, p): return sum(1 for x in p if re.search(x, t.lower()))

    def extract(self, objection: str, response: str) -> Dict:
        r, o = response, objection
        a  = self._m(r, self.ACRV_A)
        c  = self._m(r, self.ACRV_C)
        rv = self._m(r, self.ACRV_R)
        v  = self._m(r, self.ACRV_V)
        acrv = a + c + rv + v
        mt   = self._n(r, self.MOTS_TUEURS)
        wc   = len(r.split())
        sc   = len(re.split(r"[.!?]+", r.strip()))
        qc   = len(re.findall(r"\?", r))
        arg  = (self._m(r,self.PREUVE)+self._m(r,self.PROFIL_PATIENT)+
                self._m(r,self.AVANTAGE)+self._m(o,self.QUESTIONS)) / 4.0
        cs   = max(0.0, self._m(r,self.CONFORMITE) - mt*0.5)
        proxy= (acrv/4*0.4 + self._m(r,self.BIP_SIGNAL)*0.2 +
                self._m(r,self.PREUVE)*0.2 + (1-self._m(r,self.VAGUE))*0.2)
        struct=(self._m(r+o,self.PERMISSION)+self._m(r,self.SYNTHESE)+
                self._m(r,self.PREUVE)+self._m(r,self.BIP_SIGNAL)) / 4.0
        return {
            "has_acrv_a": a, "has_acrv_c": c, "has_acrv_r": rv, "has_acrv_v": v,
            "acrv_completeness": acrv,
            "has_permission": self._m(r+o,self.PERMISSION),
            "has_synthese"  : self._m(r,self.SYNTHESE),
            "has_preuve"    : self._m(r,self.PREUVE),
            "has_bip_signal": self._m(r,self.BIP_SIGNAL),
            "has_profil_patient": self._m(r,self.PROFIL_PATIENT),
            "has_avantage"      : self._m(r,self.AVANTAGE),
            "has_decouverte"    : self._m(o,self.QUESTIONS),
            "has_conformite": self._m(r,self.CONFORMITE),
            "has_mot_tueur" : int(mt > 0),
            "n_mots_tueurs" : mt,
            "has_empathie"  : self._m(r,self.EMPATHIE),
            "is_vague"      : self._m(r,self.VAGUE),
            "question_count": qc,
            "word_count"    : wc,
            "sentence_count": sc,
            "has_numbers"   : int(bool(re.search(r"\d+", r))),
            "argumentation_richness" : round(arg, 3),
            "conformite_score"       : round(cs,  3),
            "response_quality_proxy" : round(proxy, 3),
        }

    def extract_values(self, objection: str, response: str) -> List[float]:
        return list(self.extract(objection, response).values())


# ══════════════════════════════════════════════════════════════════════
# NLPScoringModel — CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════

class NLPScoringModel:
    """
    Classe d'inférence production pour le NLP Scoring Model V2.

    Évalue une réponse de délégué médical sur 6 dimensions :
        T1  Qualité réponse       (Excellent / Bon / Faible)
        T2  Scores détaillés      (scientifique / clarté / objection)
        T3  Sentiment client      (Positif / Neutre / Négatif)
        T4  Niveau ALIA           (Débutant / Junior / Confirmé / Expert)
        T5  Format visite         (Flash / Standard / Approfondie)
        T6  Conformité            (True = conforme / False = mot tueur détecté)

    Attributes:
        version    (str) : version du modèle
        trained_at (str) : timestamp d'entraînement
    """

    def __init__(self, bundle: Dict):
        self._tfidf     = bundle["tfidf"]
        self._scaler    = bundle["scaler"]
        self._extractor = NLPFeatureExtractorV2()   # toujours utiliser la version locale
        self._encoders  = bundle["encoders"]
        self._models    = bundle["models"]
        self._config    = bundle["config"]
        self._seuils    = bundle.get("seuils_alia", SEUILS_ALIA)

        self.version    = bundle.get("version",    "2.0.0")
        self.trained_at = bundle.get("trained_at", "unknown")

    # ── Class methods ──────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "NLPScoringModel":
        """
        Charge le modèle depuis un fichier .pkl.

        Args:
            path : chemin vers nlp_scoring_bundle_v2.pkl

        Returns:
            Instance NLPScoringModel prête à l'inférence
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Bundle introuvable : {path}\n"
                "Exécutez d'abord : python nlp_scoring_train_v2.py"
            )
        bundle = joblib.load(model_path)
        log.info(
            f"NLPScoringModel chargé — "
            f"v{bundle.get('version','?')} | "
            f"trained: {bundle.get('trained_at','?')[:19]}"
        )
        return cls(bundle)

    # ── Core inference ─────────────────────────────────────────────────

    def predict(self, objection: str, response: str) -> Dict:
        """
        Inférence complète sur une paire (objection, réponse).

        Args:
            objection (str) : texte de l'objection du médecin
            response  (str) : texte de la réponse du délégué

        Returns:
            dict : résultat complet avec toutes les dimensions ALIA
        """
        # ── Préparer les features ─────────────────────────────────────
        text       = objection + " [SEP] " + response
        feat_dict  = self._extractor.extract(objection, response)
        feat_vals  = list(feat_dict.values())

        x_tfidf = self._tfidf.transform([text])
        x_ling  = csr_matrix(
            self._scaler.transform([feat_vals])
        )
        x = hstack([x_tfidf, x_ling])

        # ── T1 — Qualité ──────────────────────────────────────────────
        quality      = self._encoders["t1"].inverse_transform(
            self._models["t1"].predict(x)
        )[0]
        quality_proba = dict(zip(
            self._encoders["t1"].classes_,
            self._models["t1"].predict_proba(x)[0].round(4)
        ))

        # ── T2 — Scores ───────────────────────────────────────────────
        scores_raw  = self._models["t2"].predict(x)[0]
        w           = np.array(self._config["score_weights"])
        scores = {
            "scientific_accuracy"   : round(float(np.clip(scores_raw[0], 0, 10)), 2),
            "communication_clarity" : round(float(np.clip(scores_raw[1], 0, 10)), 2),
            "objection_handling"    : round(float(np.clip(scores_raw[2], 0, 10)), 2),
        }
        overall_score = round(
            float(np.clip((scores_raw * w).sum(), 0, 10)), 2
        )

        # ── T3 — Sentiment ────────────────────────────────────────────
        sentiment = self._encoders["t3"].inverse_transform(
            self._models["t3"].predict(x)
        )[0]

        # ── T4 — Niveau ALIA ──────────────────────────────────────────
        niveau_alia      = self._encoders["t4"].inverse_transform(
            self._models["t4"].predict(x)
        )[0]
        niveau_proba = dict(zip(
            self._encoders["t4"].classes_,
            self._models["t4"].predict_proba(x)[0].round(4)
        ))

        # ── T5 — Format visite ────────────────────────────────────────
        visit_format = self._encoders["t5"].inverse_transform(
            self._models["t5"].predict(x)
        )[0]

        # ── T6 — Conformité (règle regex prioritaire) ─────────────────
        conformite = self._check_conformite(response)

        # ── Scores composites ─────────────────────────────────────────
        acrv_score = int(feat_dict["acrv_completeness"])

        # ── Seuils & progression ─────────────────────────────────────
        seuil_courant  = self._seuils.get(niveau_alia, 7.0)
        prochain       = PROCHAIN_NIVEAU.get(niveau_alia)
        seuil_prochain = self._seuils.get(prochain, None) if prochain else None
        competency_flag = overall_score >= self._config.get("seuil_competence", 7.0)
        ecart_seuil    = round(seuil_courant - overall_score, 2) if prochain else 0.0

        # ── Feedback coaching ─────────────────────────────────────────
        feedback = self._generate_feedback(
            feat_dict, overall_score, niveau_alia,
            conformite, quality
        )

        # ── Actions concrètes pour progresser ────────────────────────
        next_level_actions = self._next_level_actions(
            niveau_alia, overall_score, feat_dict, conformite
        )

        return {
            # ── Résultats principaux ──────────────────────────────────
            "quality"             : quality,
            "quality_proba"       : quality_proba,
            "overall_score"       : overall_score,
            "scores"              : scores,

            # ── Niveau ALIA (Référentiel officiel) ────────────────────
            "niveau_alia"         : niveau_alia,
            "niveau_proba"        : niveau_proba,
            "niveau_color"        : NIVEAU_COLORS.get(niveau_alia, "#888"),
            "seuil_courant"       : seuil_courant,
            "prochain_niveau"     : prochain,
            "seuil_prochain"      : seuil_prochain,
            "ecart_seuil"         : ecart_seuil,
            "competency_flag"     : competency_flag,

            # ── Analyse visite ────────────────────────────────────────
            "visit_format"        : visit_format,
            "acrv_score"          : acrv_score,
            "acrv_detail"         : {
                "accueil"    : int(feat_dict["has_acrv_a"]),
                "clarif"     : int(feat_dict["has_acrv_c"]),
                "reponse"    : int(feat_dict["has_acrv_r"]),
                "validation" : int(feat_dict["has_acrv_v"]),
            },

            # ── Conformité ────────────────────────────────────────────
            "conformite"          : conformite,
            "conformite_flag"     : int(conformite),
            "n_mots_tueurs"       : int(feat_dict["n_mots_tueurs"]),

            # ── Sentiment ─────────────────────────────────────────────
            "sentiment"           : sentiment,

            # ── Feedback coaching ─────────────────────────────────────
            "feedback_coaching"   : feedback,
            "next_level_actions"  : next_level_actions,

            # ── Features brutes (pour dashboard / debug) ──────────────
            "nlp_features"        : feat_dict,
        }

    def predict_batch(self, conversations: List[Dict]) -> List[Dict]:
        """
        Inférence en batch sur plusieurs conversations.

        Args:
            conversations : liste de dicts {"objection": ..., "response": ...}

        Returns:
            liste de résultats (même format que predict())
        """
        return [
            self.predict(c["objection"], c["response"])
            for c in conversations
        ]

    # ── Helpers internes ──────────────────────────────────────────────

    def _check_conformite(self, response: str) -> bool:
        """
        Détection déterministe des mots tueurs.
        Règle prioritaire sur le modèle ML (T6).
        """
        r = response.lower()
        mots_tueurs = [
            r"\bgaranti\b", r"\b100%\b", r"\bguérit\b",
            r"toujours efficace", r"aucun effet secondaire",
            r"meilleur produit", r"\brévolutionnaire\b",
            r"\bmiracle\b", r"sans aucun risque",
            r"prouvé sans réserve", r"cure définitive",
            r"élimine complètement",
        ]
        return not any(re.search(p, r) for p in mots_tueurs)

    def _generate_feedback(
        self,
        feats    : Dict,
        overall  : float,
        niveau   : str,
        conformite: bool,
        quality  : str,
    ) -> List[str]:
        """
        Génère une liste de feedbacks coaching ciblés.
        Basé sur le Référentiel des 4 niveaux ALIA.
        """
        msgs = []

        # Conformité — toujours en premier si violation
        if not conformite:
            msgs.append(
                FEEDBACK_TEMPLATES["mot_tueur"].get(
                    niveau, FEEDBACK_TEMPLATES["mot_tueur"]["Débutant"]
                )
            )

        # Score insuffisant pour le niveau courant
        seuil = SEUILS_ALIA.get(niveau, 7.0)
        if overall < seuil and PROCHAIN_NIVEAU.get(niveau):
            msgs.append(
                FEEDBACK_TEMPLATES["score_faible"].get(
                    niveau, FEEDBACK_TEMPLATES["score_faible"]["Junior"]
                )
            )

        # A-C-R-V incomplet
        if feats["acrv_completeness"] < 3 and niveau != "Débutant":
            msgs.append(
                FEEDBACK_TEMPLATES["acrv_incomplet"].get(
                    niveau, FEEDBACK_TEMPLATES["acrv_incomplet"]["Junior"]
                )
            )
        elif feats["acrv_completeness"] < 2 and niveau == "Débutant":
            msgs.append(FEEDBACK_TEMPLATES["acrv_incomplet"]["Débutant"])

        # Pas de signal BIP / closing
        if not feats["has_bip_signal"] and niveau in ["Junior", "Confirmé", "Expert"]:
            msgs.append(
                FEEDBACK_TEMPLATES["pas_de_bip"].get(
                    niveau, FEEDBACK_TEMPLATES["pas_de_bip"]["Junior"]
                )
            )
        elif not feats["has_bip_signal"] and niveau == "Débutant":
            msgs.append(FEEDBACK_TEMPLATES["pas_de_bip"]["Débutant"])

        # Pas de preuve
        if not feats["has_preuve"] and niveau in ["Confirmé", "Expert"]:
            msgs.append(
                FEEDBACK_TEMPLATES["pas_de_preuve"].get(
                    niveau, FEEDBACK_TEMPLATES["pas_de_preuve"]["Confirmé"]
                )
            )

        # Réponse vague
        if feats["is_vague"]:
            msgs.append(
                FEEDBACK_TEMPLATES["vague"].get(
                    niveau, FEEDBACK_TEMPLATES["vague"]["Junior"]
                )
            )

        # Pas de segmentation patient (Confirmé / Expert)
        if not feats["has_profil_patient"] and niveau in ["Confirmé", "Expert"]:
            msgs.append(
                FEEDBACK_TEMPLATES["pas_de_profil"].get(niveau, "")
            )

        # Si aucun feedback négatif — message positif
        if not msgs:
            msgs.append(
                FEEDBACK_POSITIF.get(niveau, "Bonne réponse — continuez ainsi ✅")
            )

        return msgs

    def _next_level_actions(
        self,
        niveau   : str,
        overall  : float,
        feats    : Dict,
        conformite: bool,
    ) -> Dict:
        """
        Retourne les actions concrètes pour passer au niveau suivant.
        Basé sur les seuils du Référentiel ALIA.
        """
        prochain = PROCHAIN_NIVEAU.get(niveau)
        if not prochain:
            return {
                "message": "Niveau Expert atteint — maintenez les standards.",
                "actions": [],
            }

        seuil_p = SEUILS_ALIA[prochain]
        manque  = max(0.0, round(seuil_p - overall, 2))
        actions = []

        # Score
        if manque > 0:
            actions.append(
                f"Améliorer le score global de +{manque:.1f} points "
                f"(cible : {seuil_p}/10)"
            )

        # Conformité bloquante pour tous les niveaux
        if not conformite:
            actions.append(
                "Supprimer toutes les surpromesses "
                "(conformité = condition sine qua non)"
            )

        # A-C-R-V
        kpi = KPIS_ALIA.get(prochain, {})
        if "acrv_min" in kpi and feats["acrv_completeness"] / 4.0 < kpi["acrv_min"]:
            actions.append(
                f"Maîtriser la méthode A-C-R-V ≥ {kpi['acrv_min']*100:.0f}% "
                f"(actuellement {feats['acrv_completeness']}/4)"
            )

        # BIP signal
        if not feats["has_bip_signal"] and prochain in ["Junior", "Confirmé", "Expert"]:
            actions.append("Détecter et utiliser les signaux BIP pour le closing")

        # Preuve
        if not feats["has_preuve"] and prochain in ["Confirmé", "Expert"]:
            actions.append("Inclure systématiquement 1 repère factuel ou preuve courte")

        # Segmentation patient
        if not feats["has_profil_patient"] and prochain in ["Confirmé", "Expert"]:
            actions.append("Segmenter l'argumentation par profil patient")

        return {
            "prochain_niveau": prochain,
            "seuil"          : seuil_p,
            "ecart"          : manque,
            "actions"        : actions,
            "message"        : (
                f"Pour passer {niveau} → {prochain} : "
                f"{len(actions)} point(s) à améliorer"
                if actions else
                f"Score suffisant pour {prochain} — vérifiez les autres KPIs"
            ),
        }

    # ── Résumé formaté ────────────────────────────────────────────────

    def result_summary(self, result: Dict, verbose: bool = False) -> str:
        """Résumé lisible d'un résultat de prédiction."""
        conf_str = "✅ Conforme" if result["conformite"] else "🚨 MOT TUEUR DÉTECTÉ"
        lines = [
            f"┌─ NLP Scoring Result {'─'*38}",
            f"│  Qualité      : {result['quality']:<12}  "
            f"(proba: {max(result['quality_proba'].values()):.2%})",
            f"│  Niveau ALIA  : {result['niveau_alia']:<12}  "
            f"→ Prochain : {result['prochain_niveau'] or '—'}",
            f"│  Score global : {result['overall_score']:>5.2f} / 10   "
            f"(seuil passage : {result['seuil_prochain'] or '—'})",
            f"│  Scores       : sci={result['scores']['scientific_accuracy']:.1f}  "
            f"clar={result['scores']['communication_clarity']:.1f}  "
            f"obj={result['scores']['objection_handling']:.1f}",
            f"│  A-C-R-V      : {result['acrv_score']}/4  "
            f"({result['acrv_detail']})",
            f"│  Conformité   : {conf_str}",
            f"│  Format       : {result['visit_format']}  |  "
            f"Sentiment : {result['sentiment']}",
        ]
        if verbose:
            lines.append(f"│  Feedback :")
            for fb in result["feedback_coaching"]:
                lines.append(f"│    • {fb}")
            if result["next_level_actions"]["actions"]:
                lines.append(f"│  Pour progresser :")
                for ac in result["next_level_actions"]["actions"]:
                    lines.append(f"│    → {ac}")
        lines.append(f"└{'─'*59}")
        return "\n".join(lines)

    def get_dashboard_data(self, result: Dict) -> Dict:
        """
        Retourne les données formatées pour le dashboard HTML / Django.
        Structure optimisée pour l'affichage frontend.
        """
        return {
            # Cartes principales
            "quality"       : result["quality"],
            "quality_color" : QUALITY_COLORS.get(result["quality"], "#888"),
            "overall_score" : result["overall_score"],
            "niveau_alia"   : result["niveau_alia"],
            "niveau_color"  : result["niveau_color"],
            "conformite"    : result["conformite"],

            # Jauges scores
            "scores"        : result["scores"],

            # Progression ALIA
            "progression"   : {
                "niveau_courant" : result["niveau_alia"],
                "prochain_niveau": result["prochain_niveau"],
                "seuil"          : result["seuil_prochain"],
                "score_actuel"   : result["overall_score"],
                "ecart"          : result["ecart_seuil"],
                "pct_progres"    : min(100, round(
                    result["overall_score"] / (result["seuil_prochain"] or 10) * 100
                )),
            },

            # ACRV radar
            "acrv"          : {
                "score"   : result["acrv_score"],
                "detail"  : result["acrv_detail"],
                "pct"     : result["acrv_score"] / 4 * 100,
            },

            # Feedback
            "feedback"      : result["feedback_coaching"],
            "actions"       : result["next_level_actions"]["actions"],

            # Meta
            "visit_format"  : result["visit_format"],
            "sentiment"     : result["sentiment"],
        }

    def __repr__(self) -> str:
        return (
            f"NLPScoringModel("
            f"version='{self.version}', "
            f"tasks=[T1,T2,T3,T4,T5,T6], "
            f"trained='{self.trained_at[:10]}')"
        )


# ══════════════════════════════════════════════════════════════════════
# QUICK TEST — validation rapide
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  NLPScoringModel V2 — Quick Inference Test")
    print("=" * 65)

    # ── Charger le modèle ─────────────────────────────────────────────
    model = NLPScoringModel.load(DEFAULT_MODEL_PATH)
    print(f"\n  {model}\n")

    # ── Cas de test ───────────────────────────────────────────────────
    test_cases = [
        {
            "label"    : "Expert / Excellent",
            "objection": "Pas convaincu par ce produit",
            "response" : (
                "Je comprends tout à fait votre préoccupation. "
                "Quand vous dites pas convaincu, c'est plutôt sur la composition "
                "ou la tolérance ? Selon les données disponibles, 1 repère ciblé "
                "et proposition de test sur 2 à 3 patients. "
                "Est-ce que ce repère suffit pour envisager un test ? "
                "(selon la notice)"
            ),
        },
        {
            "label"    : "Junior / Bon",
            "objection": "J'ai mes habitudes avec un autre produit",
            "response" : (
                "Je comprends. Sur ce type de patients, on peut positionner "
                "en plan B sur les cas moins satisfaits. "
                "Je vous laisse une fiche et je repasse."
            ),
        },
        {
            "label"    : "Débutant / Faible — MOT TUEUR",
            "objection": "Pas le temps pour une présentation",
            "response" : (
                "C'est le meilleur produit de notre gamme. "
                "Je vous garantis que vos patients seront toujours satisfaits. "
                "Aucun effet secondaire signalé."
            ),
        },
    ]

    # ── Test 1 : predict() ────────────────────────────────────────────
    print("─" * 65)
    print("  TEST 1 — predict() sur 3 cas")
    print("─" * 65)
    for tc in test_cases:
        result = model.predict(tc["objection"], tc["response"])
        print(f"\n  [{tc['label']}]")
        print(model.result_summary(result, verbose=True))

    # ── Test 2 : predict_batch() ──────────────────────────────────────
    print("\n" + "─" * 65)
    print("  TEST 2 — predict_batch()")
    print("─" * 65)
    batch = [{"objection": tc["objection"], "response": tc["response"]}
             for tc in test_cases]
    results = model.predict_batch(batch)
    print(f"  Batch size  : {len(results)}")
    print(f"  Qualités    : {[r['quality'] for r in results]}")
    print(f"  Niveaux     : {[r['niveau_alia'] for r in results]}")
    print(f"  Conformités : {[r['conformite'] for r in results]}")

    # ── Test 3 : get_dashboard_data() ─────────────────────────────────
    print("\n" + "─" * 65)
    print("  TEST 3 — get_dashboard_data() (pour le frontend Django)")
    print("─" * 65)
    dash = model.get_dashboard_data(results[0])
    print(f"  Clés dashboard : {list(dash.keys())}")
    print(f"  Qualité        : {dash['quality']}  ({dash['quality_color']})")
    print(f"  Niveau         : {dash['niveau_alia']}  ({dash['niveau_color']})")
    print(f"  Score          : {dash['overall_score']}/10")
    print(f"  Progression    : {dash['progression']}")
    print(f"  ACRV           : {dash['acrv']}")
    print(f"  Nb feedbacks   : {len(dash['feedback'])}")
    print(f"  Nb actions     : {len(dash['actions'])}")

    print(f"\n✅  NLPScoringModel V2 — inférence validée")
    print(f"    Import : from nlp_scoring_model_v2 import NLPScoringModel")
