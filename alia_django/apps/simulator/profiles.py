"""
profiles.py
===========
Profils médecins et produits VITAL SA pour le simulateur.
Chaque profil définit la personnalité, les objections favorites
et le comportement dynamique du médecin virtuel.
"""

# ── Produits VITAL SA (Annexe V2) ─────────────────────────────────────
VITAL_PRODUCTS = [
    {
        "id"         : "hexabrix",
        "nom"        : "Hexabrix",
        "categorie"  : "Produit de contraste",
        "indication" : "Imagerie médicale — IRM et scanner",
        "argument_cle": "Tolérance rénale supérieure aux produits concurrents",
    },
    {
        "id"         : "ferrimax",
        "nom"        : "Ferrimax",
        "categorie"  : "Fer injectable",
        "indication" : "Carence martiale sévère — patients intolérants per os",
        "argument_cle": "Administration IV rapide en 15 minutes, profil de tolérance favorable",
    },
    {
        "id"         : "calcium_vital",
        "nom"        : "Calcium Vital D3",
        "categorie"  : "Supplémentation osseuse",
        "indication" : "Ostéoporose, prévention fracturaire, patients âgés",
        "argument_cle": "Association calcium + vitamine D3 en dose optimale, observance facilitée",
    },
    {
        "id"         : "cardiozen",
        "nom"        : "Cardiozen",
        "categorie"  : "Cardioprotecteur",
        "indication" : "Insuffisance cardiaque légère à modérée",
        "argument_cle": "Réduction de 30% des hospitalisations selon l'étude CZEN-2024",
    },
    {
        "id"         : "neurovit",
        "nom"        : "Neurovit Complex",
        "categorie"  : "Neuroprotecteur vitamines B",
        "indication" : "Neuropathies périphériques, douleurs neuropathiques",
        "argument_cle": "Association B1-B6-B12 en doses thérapeutiques — effet synergique documenté",
    },
]

# ── Profils médecins ──────────────────────────────────────────────────
DOCTOR_PROFILES = [
    {
        "id"          : "sceptique",
        "nom"         : "Dr. Ben Salah",
        "specialite"  : "Médecin généraliste",
        "avatar"      : "👨‍⚕️",
        "difficulte"  : "Difficile",
        "difficulte_n": 3,
        "ville"       : "Tunis — Cabinet privé",
        "personnalite": (
            "Médecin expérimenté de 55 ans, très sceptique envers les délégués médicaux. "
            "Il a ses habitudes et prescrit depuis 15 ans les mêmes produits. "
            "Il coupe souvent les délégués et pose des questions techniques précises. "
            "Il accorde de la valeur uniquement aux données cliniques vérifiables. "
            "Il peut être brusque mais respecte les délégués qui connaissent leur sujet."
        ),
        "objections_favorites": [
            "J'ai mes habitudes, ça marche très bien pour mes patients.",
            "Vous avez des données cliniques tunisiennes sur ce produit ?",
            "Votre concurrent me propose exactement la même chose.",
            "Je n'ai pas le temps pour une présentation complète.",
        ],
        "ouverture_initiale": 2,   # 1-5, 2 = peu ouvert
        "signal_bip_seuil"  : 8.0, # score minimum pour déclencher un BIP
        "description_ui"    : "Sceptique, exigeant, basé sur les preuves",
        "couleur"           : "#8B1A1A",
    },
    {
        "id"          : "presse",
        "nom"         : "Dr. Miled",
        "specialite"  : "Cardiologue",
        "avatar"      : "👩‍⚕️",
        "difficulte"  : "Moyen",
        "difficulte_n": 2,
        "ville"       : "Sfax — Clinique",
        "personnalite": (
            "Cardiologue débordée de 42 ans, toujours pressée. "
            "Elle accorde maximum 3 minutes par délégué. "
            "Elle apprécie les délégués qui vont droit au but et maîtrisent leur produit. "
            "Si la permission est bien gérée, elle peut s'ouvrir. "
            "Elle pose des questions courtes et attend des réponses courtes."
        ),
        "objections_favorites": [
            "Je n'ai vraiment pas le temps là.",
            "Donnez-moi juste l'essentiel en 30 secondes.",
            "C'est remboursé par la CNAM ?",
            "Envoyez-moi la documentation par mail.",
        ],
        "ouverture_initiale": 3,
        "signal_bip_seuil"  : 7.0,
        "description_ui"    : "Pressée, efficace, apprécie la concision",
        "couleur"           : "#1A4A8B",
    },
    {
        "id"          : "curieux",
        "nom"         : "Dr. Trabelsi",
        "specialite"  : "Interniste",
        "avatar"      : "🧑‍⚕️",
        "difficulte"  : "Facile",
        "difficulte_n": 1,
        "ville"       : "Tunis — Hôpital universitaire",
        "personnalite": (
            "Interniste universitaire de 38 ans, ouvert et curieux. "
            "Il aime discuter des données scientifiques et poser des questions approfondies. "
            "Il compare les produits de manière rigoureuse. "
            "Il est favorable aux nouveautés si elles sont bien documentées. "
            "Il peut demander des précisions techniques pointues."
        ),
        "objections_favorites": [
            "Quelle est la différence précise avec votre concurrent ?",
            "Est-ce qu'il y a des études randomisées sur ce produit ?",
            "Sur quel profil de patient est-ce que vous recommandez ça ?",
            "Je suis intéressé — donnez-moi plus de détails.",
        ],
        "ouverture_initiale": 4,
        "signal_bip_seuil"  : 6.5,
        "description_ui"    : "Curieux, scientifique, ouvert à la discussion",
        "couleur"           : "#1A6B3A",
    },
    {
        "id"          : "concurrent",
        "nom"         : "Dr. Gharbi",
        "specialite"  : "Rhumatologue",
        "avatar"      : "👨‍⚕️",
        "difficulte"  : "Très difficile",
        "difficulte_n": 4,
        "ville"       : "Sousse — Cabinet",
        "personnalite": (
            "Rhumatologue de 48 ans, fidèle à un concurrent de longue date. "
            "Il a une relation commerciale établie avec un autre laboratoire. "
            "Il est poli mais fermé au changement. "
            "Il met souvent en avant la relation de confiance avec son fournisseur actuel. "
            "Seule une argumentation très solide sur un bénéfice différenciateur peut le faire bouger."
        ),
        "objections_favorites": [
            "Je suis très satisfait de ce que j'utilise actuellement.",
            "Pourquoi changerais-je si mes patients sont satisfaits ?",
            "Votre produit a quelque chose de vraiment différent ?",
            "J'ai 15 ans de recul sur mon traitement actuel.",
        ],
        "ouverture_initiale": 1,
        "signal_bip_seuil"  : 8.5,
        "description_ui"    : "Fidèle concurrent, très difficile à convaincre",
        "couleur"           : "#6B1A6B",
    },
]

# ── Étapes de la visite médicale (Manuel VITAL) ────────────────────────
VISIT_STEPS = [
    {
        "num"    : 1,
        "nom"    : "Permission",
        "desc"   : "Obtenir 2 minutes d'attention",
        "keywords": ["2 minutes", "court", "bonjour", "permission", "je fais court"],
        "coach"  : "Commencez par demander la permission et annoncez la durée.",
    },
    {
        "num"    : 2,
        "nom"    : "Sondage",
        "desc"   : "Questions de découverte",
        "keywords": ["comment", "quel", "vos patients", "sondage", "découverte", "profil"],
        "coach"  : "Posez des questions ouvertes pour découvrir les besoins.",
    },
    {
        "num"    : 3,
        "nom"    : "Synthèse",
        "desc"   : "Reformuler les besoins",
        "keywords": ["si je résume", "votre priorité", "si je comprends", "donc"],
        "coach"  : "Reformulez ce que vous avez compris avant d'argumenter.",
    },
    {
        "num"    : 4,
        "nom"    : "Argumentation A-C-R-V",
        "desc"   : "Répondre aux objections",
        "keywords": ["je comprends", "clarif", "selon", "données", "profil", "est-ce que"],
        "coach"  : "Appliquez A-C-R-V : Accueil → Clarification → Réponse → Validation.",
    },
    {
        "num"    : 5,
        "nom"    : "Preuve",
        "desc"   : "Appuyer sur des données",
        "keywords": ["étude", "données", "clinique", "selon la notice", "repère", "preuve"],
        "coach"  : "Citez une preuve courte et pertinente pour le profil médecin.",
    },
    {
        "num"    : 6,
        "nom"    : "BIP / Closing",
        "desc"   : "Détecter et conclure",
        "keywords": ["essayer", "test", "2-3 patients", "accord", "engagement", "je repasse"],
        "coach"  : "Proposez un micro-test sur 2-3 patients pour concrétiser.",
    },
]

def get_product(product_id: str) -> dict:
    for p in VITAL_PRODUCTS:
        if p["id"] == product_id:
            return p
    return VITAL_PRODUCTS[0]

def get_doctor(doctor_id: str) -> dict:
    for d in DOCTOR_PROFILES:
        if d["id"] == doctor_id:
            return d
    return DOCTOR_PROFILES[0]
