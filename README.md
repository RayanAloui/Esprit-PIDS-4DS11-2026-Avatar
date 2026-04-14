# ALIA — Avatar Visite Médicale VITAL SA
## Application Django — DSO 1 & DSO 4

---

## 📋 Prérequis

- Python 3.10+
- pip

---

## 🚀 Installation en 4 étapes

### Étape 1 — Cloner / extraire le projet

```bash
# Extraire le ZIP dans un dossier
unzip alia_django.zip -d alia_django
cd alia_django
```

### Étape 2 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 3 — Copier les modèles IA

Copiez les fichiers suivants dans le dossier `models_ai/` :

**Depuis votre dossier `outputs/` :**

| Fichier | Description |
|---------|-------------|
| `nlp_scoring_bundle_v2.pkl` | Bundle NLP complet (T1→T6) |
| `nlp_scoring_model_v2.py`   | Classe NLPScoringModel |
| `route_model.pkl`           | Bundle RouteOptimizer |
| `route_model.py`            | Classe RouteOptimizer |
| `lstm_body_language_v2.pkl` | Bundle LSTM |
| `lstm_model_v2.py`          | Classe BodyLanguageModel |
| `lstm_train_v2.py`          | (requis pour imports) |
| `pharmacies_grand_tunis.csv`| Dataset pharmacies |

**Commande automatique :**
```bash
python setup.py --models-src /chemin/vers/votre/dossier/outputs
```

### Étape 4 — Lancer le serveur

```bash
python manage.py migrate
python manage.py runserver
```

Ouvrez **http://127.0.0.1:8000** dans votre navigateur.

---

## 📁 Structure du projet

```
alia_django/
├── manage.py               ← Point d'entrée Django
├── requirements.txt        ← Dépendances
├── setup.py                ← Script de setup automatique
├── db.sqlite3              ← Base SQLite (créée automatiquement)
│
├── alia/                   ← Configuration Django
│   ├── settings.py
│   └── urls.py
│
├── apps/
│   ├── home/               ← Page d'accueil
│   ├── avatar/             ← NLP Scoring + Historique SQLite
│   └── routes/             ← Route Optimizer + Carte Leaflet
│
├── models_ai/              ← Modèles IA (à remplir)
│   ├── nlp_scoring_bundle_v2.pkl
│   ├── nlp_scoring_model_v2.py
│   ├── route_model.pkl
│   ├── route_model.py
│   └── ...
│
├── static/
│   └── css/style.css       ← Styles globaux ALIA
│
└── templates/
    ├── base.html            ← Layout principal
    ├── home/index.html      ← Page accueil
    ├── avatar/index.html    ← Page NLP Scoring
    └── routes/index.html    ← Page Route Optimizer
```

---

## 🖥️ Pages de l'application

### `/` — Accueil
- Présentation du projet ALIA
- Vue d'ensemble des 4 modules (DSO 1 NLP, DSO 1 LSTM, DSO 4 Routes, DSO 2 futur)
- Statistiques clés
- Pipeline TDSP

### `/avatar/` — Avatar NLP Scoring
- Saisie objection + réponse du délégué
- Sélection niveau ALIA (Débutant/Junior/Confirmé/Expert)
- 4 exemples rapides (Excellent/Bon/Faible/Mot tueur)
- Résultats temps réel : 6 dimensions T1→T6
- Méthode A-C-R-V (4 étapes)
- Conformité (détection mots tueurs)
- Progression niveau ALIA avec seuils officiels
- Feedback coaching personnalisé
- Section Body Language (simulation 3 postures)
- **Historique SQLite** des 10 dernières analyses

### `/routes/` — Route Optimizer
- Sélection jour / heure / nb stops / zone / dépôt
- Carte Leaflet interactive avec toutes les pharmacies
- Optimisation TSP 2-Opt en temps réel
- Affichage de la route optimisée sur la carte
- Liste ordonnée des stops avec scores et distances
- Métriques : distance totale, durée, amélioration 2-Opt

---

## 🔌 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/avatar/analyze/` | POST | Analyse NLP d'une réponse |
| `/avatar/history/` | GET | Historique des analyses |
| `/routes/optimize/` | POST | Optimisation de tournée |

---

## ⚙️ Variables de configuration

Dans `alia/settings.py` :

```python
MODELS_AI_DIR = BASE_DIR / 'models_ai'  # Chemin des modèles IA
```

Pour la génération de flyers Gemini, créez aussi un fichier `.env` à la racine :

```env
GEMINI_API_KEY=put_your_gemini_api_key_here
```

Commande de génération (module `modeling`) :

```bash
python manage.py generate_flyers --limit 5
```

---

## 🧪 Test rapide

Ouvrez `/avatar/`, cliquez sur **⭐ Excellent** puis **▶ Analyser**.
Vous devriez voir un score ≈ 8.3/10, niveau Expert, ACRV 4/4, Conforme.

---

## 📚 Technologies

- **Django 4.2+** — Framework web
- **SQLite** — Historique analyses
- **scikit-learn** — Modèles NLP + Route
- **Leaflet.js** — Carte interactive
- **Google Fonts** — Bebas Neue + Source Sans 3
- **CartoDB** — Tiles carte dark theme
