"""
profiles.py
===========
Profils médecins, pharmaciens et produits VITAL SA pour le simulateur.
Chaque profil définit la personnalité, les objections favorites
et le comportement dynamique du visiteur virtuel.

v2 — Ajouts :
  - PHARMACIST_PROFILES : profils pharmaciens avec logique officine
  - GENERIC_PRODUCT     : produit fictif pour l'entraînement généraliste
  - VISIT_STEPS_PHARMACIST : étapes adaptées à la visite officine
  - get_pharmacist() / is_pharmacist()
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
    # ── Produit généraliste (entraînement sans produit spécifique) ─────
    {
        "id"         : "generique",
        "nom"        : "Produit VITAL SA",
        "categorie"  : "Complément alimentaire",
        "indication" : "Indication selon le profil patient — à définir en visite",
        "argument_cle": "Gamme complète VITAL SA, bien documentée et disponible en officine",
        "_is_generic" : True,   # flag interne — ne pas afficher dans la liste produits normale
    },
]

# ── Profils médecins ──────────────────────────────────────────────────
DOCTOR_PROFILES = [
    {
        "id"          : "sceptique",
        "nom"         : "Dr. Ben Salah",
        "specialite"  : "Médecin généraliste",
        "type"        : "doctor",
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
        "ouverture_initiale": 2,
        "signal_bip_seuil"  : 8.0,
        "description_ui"    : "Sceptique, exigeant, basé sur les preuves",
        "couleur"           : "#8B1A1A",
    },
    {
        "id"          : "presse",
        "nom"         : "Dr. Miled",
        "specialite"  : "Cardiologue",
        "type"        : "doctor",
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
        "type"        : "doctor",
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
        "type"        : "doctor",
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

# ── Profils pharmaciens ───────────────────────────────────────────────
PHARMACIST_PROFILES = [
    {
        "id"          : "pharmacien_orient",
        "nom"         : "M. Hamdi",
        "specialite"  : "Pharmacien titulaire",
        "type"        : "pharmacist",
        "avatar"      : "👨‍💼",
        "difficulte"  : "Moyen",
        "difficulte_n": 2,
        "ville"       : "Tunis — Pharmacie El Amal",
        "personnalite": (
            "Pharmacien titulaire de 46 ans, pragmatique et orienté business. "
            "Il gère 3 employés et se préoccupe avant tout de la rotation de stock et de la marge. "
            "Il connaît bien ses clients habituels et leurs besoins récurrents. "
            "Il est réceptif si le délégué lui montre comment le produit va se vendre vite. "
            "Il pose des questions sur le conditionnement, la DLC et les conditions de retour."
        ),
        "objections_favorites": [
            "Quelle est la marge pharmacien sur ce produit ?",
            "J'ai déjà beaucoup de stock du concurrent, ça tourne bien.",
            "Quel est le conditionnement minimum de commande ?",
            "Est-ce que vous avez des présentoirs ou PLV ?",
            "La DLC est suffisante pour que ça ne périme pas en rayon ?",
        ],
        "ouverture_initiale": 3,
        "signal_bip_seuil"  : 6.5,
        "description_ui"    : "Orienté marge et rotation, pragmatique",
        "couleur"           : "#1A5A8B",
        # Étapes spécifiques à la visite officine
        "visit_context"     : "officine",
    },
    {
        "id"          : "pharmacien_sceptique",
        "nom"         : "Mme. Khedher",
        "specialite"  : "Pharmacien, adjointe principal",
        "type"        : "pharmacist",
        "avatar"      : "👩‍💼",
        "difficulte"  : "Difficile",
        "difficulte_n": 3,
        "ville"       : "Sousse — Grande Pharmacie Centrale",
        "personnalite": (
            "Pharmacienne adjointe de 39 ans, très rigoureuse sur la qualité. "
            "Elle refuse de référencer un produit sans avoir lu la fiche technique complète. "
            "Elle compare systématiquement avec les produits qu'elle connaît déjà. "
            "Elle est sensible aux arguments sur la tolérance, les contre-indications et les études. "
            "Elle peut devenir une excellente ambassadrice du produit si convaincue."
        ),
        "objections_favorites": [
            "Vous avez la monographie complète du produit ?",
            "Quelles sont les contre-indications chez la femme enceinte ?",
            "On a déjà un produit similaire qui marche bien — quelle est la différence ?",
            "Est-ce que c'est remboursé CNAM ou c'est OTC uniquement ?",
            "Quels sont les effets secondaires rapportés en pharmacovigilance ?",
        ],
        "ouverture_initiale": 2,
        "signal_bip_seuil"  : 7.5,
        "description_ui"    : "Rigoureuse, scientifique, exige la documentation",
        "couleur"           : "#6B1A3A",
        "visit_context"     : "officine",
    },
    {
        "id"          : "pharmacien_curieux",
        "nom"         : "M. Dridi",
        "specialite"  : "Pharmacien, nouvelle officine",
        "type"        : "pharmacist",
        "avatar"      : "🧑‍💼",
        "difficulte"  : "Facile",
        "difficulte_n": 1,
        "ville"       : "Ariana — Pharmacie Nouvelle",
        "personnalite": (
            "Jeune pharmacien de 31 ans qui vient d'ouvrir son officine il y a 8 mois. "
            "Il est très ouvert aux nouveaux produits pour diversifier son offre. "
            "Il cherche à fidéliser sa clientèle avec des produits différenciants. "
            "Il est particulièrement intéressé par les produits OTC avec une bonne communication patient. "
            "Il pose des questions sur le support marketing et les animations en officine."
        ),
        "objections_favorites": [
            "Est-ce que vous proposez des animations en officine ?",
            "Vous avez des supports de communication pour les patients ?",
            "Quelle est la politique de retour si le produit ne se vend pas ?",
            "Y a-t-il un minimum de commande pour avoir les tarifs préférentiels ?",
        ],
        "ouverture_initiale": 4,
        "signal_bip_seuil"  : 6.0,
        "description_ui"    : "Ouvert, cherche à diversifier, orienté patient",
        "couleur"           : "#1A6B4A",
        "visit_context"     : "officine",
    },
    {
        "id"          : "pharmacien_chain",
        "nom"         : "Mme. Ayari",
        "specialite"  : "Responsable achats — Réseau PharmaPlus",
        "type"        : "pharmacist",
        "avatar"      : "👩‍💼",
        "difficulte"  : "Très difficile",
        "difficulte_n": 4,
        "ville"       : "Tunis — Siège PharmaPlus (15 officines)",
        "personnalite": (
            "Responsable achats pour un réseau de 15 pharmacies, très professionnelle. "
            "Elle négocie des remises de volume et des conditions commerciales strictes. "
            "Elle a déjà des accords-cadres avec plusieurs laboratoires. "
            "Elle est intéressée par les exclusivités, les opérations promotionnelles groupées "
            "et les indicateurs de performance (sell-out, rotation, taux de retour). "
            "Seul un délégué très préparé, maîtrisant chiffres et conditions, peut la convaincre."
        ),
        "objections_favorites": [
            "Quel est le sell-out moyen constaté dans les officines de référence ?",
            "Vous proposez quelle remise pour un accord-cadre réseau ?",
            "On travaille déjà avec votre concurrent sur des conditions très avantageuses.",
            "Quel taux de retour constatez-vous sur ce produit ?",
            "On peut avoir une exclusivité promotionnelle sur votre nouvelle gamme ?",
        ],
        "ouverture_initiale": 1,
        "signal_bip_seuil"  : 9.0,
        "description_ui"    : "Acheteuse réseau, très exigeante, orientée volume",
        "couleur"           : "#6B4A1A",
        "visit_context"     : "officine",
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

# ── Étapes de la visite officine (pharmacien) ─────────────────────────
VISIT_STEPS_PHARMACIST = [
    {
        "num"    : 1,
        "nom"    : "Permission",
        "desc"   : "Obtenir 2 minutes",
        "keywords": ["2 minutes", "bonjour", "permission", "court", "rapide"],
        "coach"  : "Demandez 2 minutes et annoncez l'objet de votre visite.",
    },
    {
        "num"    : 2,
        "nom"    : "Découverte",
        "desc"   : "Explorer les besoins officine",
        "keywords": ["stock", "rotation", "client", "demande", "vente", "concurrent"],
        "coach"  : "Explorez le stock actuel, les produits concurrents et la demande client.",
    },
    {
        "num"    : 3,
        "nom"    : "Argumentation",
        "desc"   : "Valoriser le produit",
        "keywords": ["marge", "bénéfice", "rotation", "conditionnement", "avantage"],
        "coach"  : "Mettez en avant la marge, la rotation et les avantages différenciants.",
    },
    {
        "num"    : 4,
        "nom"    : "A-C-R-V",
        "desc"   : "Traiter les objections",
        "keywords": ["je comprends", "effectivement", "c'est vrai", "cependant", "selon"],
        "coach"  : "Accueillez l'objection, clarifiez, répondez et validez.",
    },
    {
        "num"    : 5,
        "nom"    : "Preuve",
        "desc"   : "Chiffres et données terrain",
        "keywords": ["sell-out", "étude", "données", "résultats", "retour terrain", "officine"],
        "coach"  : "Citez un chiffre concret : sell-out, taux de satisfaction, données officine.",
    },
    {
        "num"    : 6,
        "nom"    : "Commande / Closing",
        "desc"   : "Proposer une commande d'essai",
        "keywords": ["commande", "essai", "référencer", "intégrer", "essayer", "accord"],
        "coach"  : "Proposez une commande d'essai avec un minimum et conditions avantageuses.",
    },
]

# ── QCM pré-formation — Pool élargi de questions (tirage aléatoire) ────
QCM_QUESTIONS = [
    {
        "id"      : "q1",
        "question": "Quelle est la première étape de la visite médicale selon la méthode VITAL SA ?",
        "options" : [
            "Présenter directement le produit",
            "Demander la permission et annoncer la durée",
            "Poser une question sur les patients du médecin",
            "Distribuer la documentation produit",
        ],
        "correct" : 1,
        "explication": "La permission est fondamentale — elle crée un engagement de 2 minutes et réduit les interruptions.",
        "etape"   : 1,
    },
    {
        "id"      : "q2",
        "question": "Que signifie l'acronyme A-C-R-V dans la méthode VITAL SA ?",
        "options" : [
            "Argumenter, Convaincre, Répondre, Valider",
            "Accueil, Clarification, Réponse, Validation",
            "Analyser, Communiquer, Rectifier, Vendre",
            "Accueillir, Conclure, Relancer, Vérifier",
        ],
        "correct" : 1,
        "explication": "A-C-R-V : Accueil de l'objection → Clarification → Réponse argumentée → Validation de l'accord.",
        "etape"   : 4,
    },
    {
        "id"      : "q3",
        "question": "Qu'est-ce qu'un « mot tueur » dans la visite médicale ?",
        "options" : [
            "Un terme médical trop complexe pour le médecin",
            "Une promesse ou affirmation non prouvée qui viole la conformité réglementaire",
            "Un argument trop agressif qui ferme la conversation",
            "Un produit concurrent cité sans données",
        ],
        "correct" : 1,
        "explication": "Un mot tueur (ex : 'guérit', 'sans risque') est une affirmation non validée cliniquement — il viole la conformité et peut disqualifier la visite.",
        "etape"   : 4,
    },
    {
        "id"      : "q4",
        "question": "À quelle étape détecte-t-on le signal BIP (Besoin d'Intérêt Positif) ?",
        "options" : [
            "Étape 1 — Permission",
            "Étape 3 — Synthèse",
            "Étape 5 — Preuve",
            "Étape 6 — BIP/Closing",
        ],
        "correct" : 3,
        "explication": "Le BIP est un signal d'intérêt du médecin (question positive, prise de note…) qui indique le bon moment pour proposer un accord.",
        "etape"   : 6,
    },
    {
        "id"      : "q5",
        "question": "Lors d'une visite en officine, quelle objection est la plus fréquente du pharmacien ?",
        "options" : [
            "Je n'ai pas de données cliniques sur ce produit",
            "J'ai déjà un produit similaire avec une bonne rotation",
            "Mon patient préfère les génériques",
            "Ce n'est pas remboursé par l'assurance maladie",
        ],
        "correct" : 1,
        "explication": "Le pharmacien compare toujours avec son stock existant. La réponse doit montrer comment votre produit complémente ou surpasse en termes de marge et rotation.",
        "etape"   : 2,
    },
    {
        "id"      : "q6",
        "question": "Quelle est la durée idéale d'une visite médicale selon VITAL SA ?",
        "options" : [
            "30 secondes à 1 minute",
            "2 à 4 minutes",
            "10 à 15 minutes",
            "Aussi longtemps que le médecin le souhaite",
        ],
        "correct" : 1,
        "explication": "2 à 4 minutes est la durée cible. Au-delà, le médecin perd son attention. La permission demandée dès l'étape 1 cadre ce temps.",
        "etape"   : 1,
    },
    {
        "id"      : "q7",
        "question": "Lors de l'étape Sondage, quel type de question faut-il privilégier ?",
        "options" : [
            "Questions fermées (oui/non) pour gagner du temps",
            "Questions ouvertes pour explorer les besoins du médecin",
            "Questions sur le concurrent pour le déstabiliser",
            "Questions sur le remboursement CNAM en priorité",
        ],
        "correct" : 1,
        "explication": "Les questions ouvertes (comment, quel profil, qu'est-ce qui…) permettent au médecin de s'exprimer et révèlent ses véritables besoins.",
        "etape"   : 2,
    },
    # ── Questions supplémentaires pour le pool aléatoire ──────────────
    {
        "id"      : "q8",
        "question": "Que doit faire le délégué lorsqu'il reçoit une objection du médecin ?",
        "options" : [
            "Contredire immédiatement avec des données",
            "Ignorer l'objection et passer à la suite",
            "Accueillir l'objection, puis clarifier avant de répondre",
            "Reporter la discussion à la prochaine visite",
        ],
        "correct" : 2,
        "explication": "La méthode A-C-R-V impose d'ACCUEILLIR d'abord l'objection (« Je comprends votre point… ») puis de clarifier avant d'argumenter.",
        "etape"   : 4,
    },
    {
        "id"      : "q9",
        "question": "Quelle est la différence principale entre la visite chez un médecin et chez un pharmacien ?",
        "options" : [
            "Le pharmacien ne s'intéresse qu'aux données cliniques",
            "Le médecin décide du prix de vente",
            "Le pharmacien raisonne en termes de marge, rotation et stock",
            "Il n'y a aucune différence dans l'approche",
        ],
        "correct" : 2,
        "explication": "Le pharmacien est un commerçant : il pense marge brute, rotation de stock, DLC, conditionnement et PLV, pas mécanismes d'action.",
        "etape"   : 2,
    },
    {
        "id"      : "q10",
        "question": "Quel est le rôle de l'étape « Synthèse » dans la méthode de visite ?",
        "options" : [
            "Résumer toute la documentation produit",
            "Reformuler les besoins exprimés par le médecin avant d'argumenter",
            "Présenter un résumé chiffré de l'étude clinique",
            "Conclure la visite et prendre congé",
        ],
        "correct" : 1,
        "explication": "La synthèse consiste à reformuler ce que l'on a compris des besoins du médecin (« Si je résume, votre priorité est… »), avant de passer à l'argumentation.",
        "etape"   : 3,
    },
    {
        "id"      : "q11",
        "question": "Lequel de ces mots est considéré comme un « mot tueur » ?",
        "options" : [
            "« Selon les données disponibles »",
            "« Ce produit guérit la maladie »",
            "« Dans notre étude clinique »",
            "« Le profil de tolérance montre que… »",
        ],
        "correct" : 1,
        "explication": "« Guérit » est une promesse absolue non prouvée. Un délégué conforme dirait : « contribue à améliorer » ou « les données montrent une réduction significative ».",
        "etape"   : 4,
    },
    {
        "id"      : "q12",
        "question": "Comment un délégué doit-il proposer un micro-engagement au médecin ?",
        "options" : [
            "« Vous devez prescrire ce produit à tous vos patients »",
            "« Seriez-vous ouvert à tester sur 2-3 patients ciblés ? »",
            "« Je vous laisse 50 échantillons gratuits »",
            "« Signez ce bon de commande immédiatement »",
        ],
        "correct" : 1,
        "explication": "Le micro-engagement (2-3 patients) est peu risqué pour le médecin et crée un premier pas concret vers l'adoption du produit.",
        "etape"   : 6,
    },
    {
        "id"      : "q13",
        "question": "Quelle est la bonne attitude face à un médecin pressé ?",
        "options" : [
            "Présenter le maximum d'informations rapidement",
            "Aller droit au but : permission courte, argument clé, validation",
            "Reporter la visite à plus tard",
            "Insister pour obtenir plus de temps",
        ],
        "correct" : 1,
        "explication": "Avec un médecin pressé, le délégué doit être concis : permission rapide, un seul argument clé percutant, question de validation directe.",
        "etape"   : 1,
    },
    {
        "id"      : "q14",
        "question": "Qu'est-ce que le « sell-out » dans le contexte officinal ?",
        "options" : [
            "Le prix d'achat pharmacien",
            "Les ventes réelles aux clients en officine",
            "Le stock en réserve arrière",
            "Le chiffre d'affaires du laboratoire",
        ],
        "correct" : 1,
        "explication": "Le sell-out (ventes sorties de caisse) est l'indicateur clé pour le pharmacien : il prouve que le produit se vend vraiment au comptoir.",
        "etape"   : 5,
    },
    {
        "id"      : "q15",
        "question": "Comment gérer un médecin fidèle à un concurrent ?",
        "options" : [
            "Critiquer ouvertement le produit concurrent",
            "Proposer un bénéfice différenciateur précis sans dénigrer",
            "Offrir des cadeaux pour le faire changer",
            "Abandonner ce médecin et passer au suivant",
        ],
        "correct" : 1,
        "explication": "Face à la fidélité concurrentielle, il faut trouver UN bénéfice différenciateur objectif (tolérance, observance, coût…) sans jamais dénigrer.",
        "etape"   : 4,
    },
    {
        "id"      : "q16",
        "question": "Qu'est-ce que la « PLV » en visite officine ?",
        "options" : [
            "Le prix de lancement du produit",
            "La publicité sur le lieu de vente (présentoirs, affiches…)",
            "Le plan de livraison des volumes",
            "Le protocole de lancement des ventes",
        ],
        "correct" : 1,
        "explication": "PLV = Publicité sur le Lieu de Vente. Ce sont les supports visuels (présentoirs, affiches, stop-rayon) qui aident à la vente spontanée en officine.",
        "etape"   : 3,
    },
    {
        "id"      : "q17",
        "question": "Quel indicateur est le plus pertinent pour convaincre un pharmacien ?",
        "options" : [
            "Le mécanisme d'action moléculaire du produit",
            "La marge pharmacien et les données de rotation",
            "Les études cliniques randomisées en double aveugle",
            "Le nombre de publications dans les revues médicales",
        ],
        "correct" : 1,
        "explication": "Le pharmacien est un commerçant : la marge (PPA vs prix public) et la rotation (sell-out mensuel) sont ses critères de décision principaux.",
        "etape"   : 3,
    },
    {
        "id"      : "q18",
        "question": "Que signifie « DLC » dans le contexte pharmaceutique ?",
        "options" : [
            "Document de Liaison Commerciale",
            "Date Limite de Consommation (péremption)",
            "Dossier de Lancement Commercial",
            "Données de Lancement du Conditionnement",
        ],
        "correct" : 1,
        "explication": "DLC = Date Limite de Consommation. Le pharmacien s'assure que la DLC est suffisante pour que le produit ne périme pas en rayon.",
        "etape"   : 2,
    },
    {
        "id"      : "q19",
        "question": "Lors de l'étape Preuve, quelle source de preuve est la plus crédible ?",
        "options" : [
            "L'avis personnel du délégué",
            "Une étude clinique publiée avec des données chiffrées",
            "Le témoignage d'un seul patient",
            "Les ventes du produit dans un autre pays",
        ],
        "correct" : 1,
        "explication": "Les données cliniques chiffrées (études randomisées, méta-analyses) sont la preuve la plus crédible pour un médecin qui base ses décisions sur les preuves.",
        "etape"   : 5,
    },
    {
        "id"      : "q20",
        "question": "Comment un délégué doit-il terminer la visite si le médecin montre des signaux d'intérêt ?",
        "options" : [
            "Continuer à présenter tous les avantages du produit",
            "Proposer un engagement concret : test sur quelques patients ou commande d'essai",
            "Remercier et partir sans rien proposer",
            "Demander au médecin de rappeler le laboratoire",
        ],
        "correct" : 1,
        "explication": "Les signaux d'intérêt (BIP) indiquent le bon moment pour conclure avec un engagement concret et réalisable.",
        "etape"   : 6,
    },
    {
        "id"      : "q21",
        "question": "Quelle est la meilleure façon de présenter un argument clé produit ?",
        "options" : [
            "Lire la notice du produit à haute voix",
            "Relier l'argument directement aux besoins exprimés par le médecin",
            "Comparer avec tous les concurrents un par un",
            "Énumérer tous les avantages sans prioriser",
        ],
        "correct" : 1,
        "explication": "L'argument est percutant quand il répond directement au besoin identifié lors du sondage : « Vous parliez de [besoin], justement [argument clé]… ».",
        "etape"   : 4,
    },
    {
        "id"      : "q22",
        "question": "Quelle erreur est la plus grave lors de l'étape Permission ?",
        "options" : [
            "Demander 2 minutes au lieu de 3",
            "Commencer la présentation produit sans demander la permission",
            "Se présenter avec son nom complet",
            "Annoncer le nom du laboratoire",
        ],
        "correct" : 1,
        "explication": "Ne pas demander la permission = pas d'engagement du médecin. Il peut vous interrompre à tout moment. La permission cadre la visite et crée un contrat tacite.",
        "etape"   : 1,
    },
]

# Nombre de questions tirées aléatoirement pour chaque QCM
QCM_NB_QUESTIONS = 7

# Seuil minimum pour passer le QCM (pourcentage)
QCM_SEUIL_PASSAGE = 60   # 60%


def get_random_qcm(n: int = QCM_NB_QUESTIONS) -> list:
    """Tire n questions aléatoirement depuis le pool QCM_QUESTIONS."""
    import random
    pool = list(QCM_QUESTIONS)
    n = min(n, len(pool))
    return random.sample(pool, n)


# ── Fonctions helper ──────────────────────────────────────────────────

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


def get_pharmacist(pharmacist_id: str) -> dict:
    for p in PHARMACIST_PROFILES:
        if p["id"] == pharmacist_id:
            return p
    return PHARMACIST_PROFILES[0]


def get_interlocutor(interlocutor_id: str) -> dict:
    """Retourne un médecin ou pharmacien selon l'id."""
    for d in DOCTOR_PROFILES:
        if d["id"] == interlocutor_id:
            return d
    for p in PHARMACIST_PROFILES:
        if p["id"] == interlocutor_id:
            return p
    return DOCTOR_PROFILES[0]


def is_pharmacist(profile: dict) -> bool:
    """Retourne True si le profil est un pharmacien."""
    return profile.get("type") == "pharmacist"


def get_visit_steps(profile: dict) -> list:
    """Retourne les étapes de visite adaptées au type d'interlocuteur."""
    if is_pharmacist(profile):
        return VISIT_STEPS_PHARMACIST
    return VISIT_STEPS


def get_generic_product() -> dict:
    """Retourne le produit généraliste pour l'entraînement sans produit."""
    for p in VITAL_PRODUCTS:
        if p.get("_is_generic"):
            return p
    return VITAL_PRODUCTS[0]
