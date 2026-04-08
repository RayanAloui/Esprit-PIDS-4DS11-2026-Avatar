"""
setup.py — ALIA Django Project Setup
=====================================
Lance ce script une seule fois pour :
1. Copier les modèles IA depuis ton dossier outputs
2. Créer la base SQLite
3. Vérifier que tout est prêt

Usage :
    python setup.py
    python setup.py --models-src /chemin/vers/outputs
"""
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models_ai'

# Fichiers modèles requis
REQUIRED_FILES = [
    'nlp_scoring_bundle_v2.pkl',
    'nlp_scoring_model_v2.py',
    'route_model.pkl',
    'route_model.py',
    'lstm_body_language_v2.pkl',
    'lstm_model_v2.py',
    'lstm_train_v2.py',
]

# Fichier de données
DATA_FILES = [
    'pharmacies_grand_tunis.csv',
]


def copy_models(src_dir: Path):
    """Copie les modèles depuis le dossier source vers models_ai/."""
    print(f"\n📂  Copie des modèles depuis : {src_dir}")
    MODELS_DIR.mkdir(exist_ok=True)
    copied, missing = 0, []

    for fname in REQUIRED_FILES + DATA_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, MODELS_DIR / fname)
            print(f"  ✅  {fname}")
            copied += 1
        else:
            # Essayer aussi dans models/ sous-dossier
            alt = src_dir / 'models' / fname
            if alt.exists():
                shutil.copy2(alt, MODELS_DIR / fname)
                print(f"  ✅  {fname}  (depuis models/)")
                copied += 1
            else:
                print(f"  ❌  {fname}  — introuvable")
                missing.append(fname)

    print(f"\n  Copiés : {copied} | Manquants : {len(missing)}")
    return missing


def run_migrations():
    """Crée la base SQLite et applique les migrations."""
    print("\n🗄️  Création base SQLite...")
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alia.settings')
    result = subprocess.run(
        [sys.executable, 'manage.py', 'migrate', '--run-syncdb'],
        capture_output=True, text=True, cwd=BASE_DIR
    )
    if result.returncode == 0:
        print("  ✅  Base SQLite créée : db.sqlite3")
    else:
        print(f"  ❌  Erreur migrations :")
        print(result.stderr)
    return result.returncode == 0


def check_django():
    """Vérifie que Django est installé."""
    try:
        import django
        print(f"  ✅  Django {django.__version__}")
        return True
    except ImportError:
        print("  ❌  Django non installé — pip install django")
        return False


def check_dependencies():
    """Vérifie toutes les dépendances."""
    print("\n🔍  Vérification des dépendances...")
    deps = {
        'django'     : 'django',
        'numpy'      : 'numpy',
        'pandas'     : 'pandas',
        'sklearn'    : 'scikit-learn',
        'joblib'     : 'joblib',
        'scipy'      : 'scipy',
    }
    all_ok = True
    for mod, pkg in deps.items():
        try:
            __import__(mod)
            print(f"  ✅  {pkg}")
        except ImportError:
            print(f"  ❌  {pkg}  — pip install {pkg}")
            all_ok = False
    return all_ok


def check_models():
    """Vérifie que les fichiers modèles sont présents."""
    print("\n🤖  Vérification des modèles IA...")
    all_ok = True
    for fname in REQUIRED_FILES:
        path = MODELS_DIR / fname
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  ✅  {fname:<45}  {size:>7.1f} KB")
        else:
            print(f"  ❌  {fname}  — MANQUANT (lancez : python setup.py --models-src <dossier>)")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='ALIA Django Setup')
    parser.add_argument('--models-src', type=Path, default=None,
                        help='Dossier source des modèles (ex: /chemin/vers/outputs)')
    args = parser.parse_args()

    print("=" * 60)
    print("  ALIA — Django Project Setup")
    print("=" * 60)

    # 1. Dépendances
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n⚠️  Installez les dépendances manquantes puis relancez :")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # 2. Copier les modèles si source fournie
    if args.models_src:
        if not args.models_src.exists():
            print(f"\n❌  Dossier source introuvable : {args.models_src}")
            sys.exit(1)
        missing = copy_models(args.models_src)
        if missing:
            print(f"\n⚠️  {len(missing)} fichier(s) manquant(s) — le serveur fonctionnera partiellement.")

    # 3. Vérifier modèles
    models_ok = check_models()

    # 4. Migrations
    if deps_ok:
        run_migrations()

    # 5. Résumé
    print("\n" + "=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)
    if deps_ok and models_ok:
        print("  ✅  Tout est prêt ! Lancez :")
        print()
        print("       python manage.py runserver")
        print()
        print("  Puis ouvrez : http://127.0.0.1:8000")
    else:
        print("  ⚠️  Des éléments manquent — voir ci-dessus.")
        print()
        print("  Pour copier les modèles :")
        print("       python setup.py --models-src /chemin/vers/outputs")


if __name__ == '__main__':
    main()
