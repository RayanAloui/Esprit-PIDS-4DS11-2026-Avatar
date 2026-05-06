"""
ALIA Django Settings
====================
Projet ALIA — Avatar Visite Médicale VITAL SA
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'alia-vital-sa-dev-secret-key-2026-not-for-production'

DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

# ── Applications ──────────────────────────────────────────────────────
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # ALIA apps
    'apps.accounts',   # ← Auth ALIA
    'apps.home',
    'apps.avatar',
    'apps.routes',
    'apps.analytics',
    'apps.simulator',
    'apps.modeling',
    'apps.crm',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'apps.modeling.middleware.RequestMetricsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'alia.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'apps.modeling.context_processors.modeling_nav',
            ],
        },
    },
]

WSGI_APPLICATION = 'alia.wsgi.application'

# ── Database — SQLite ────────────────────────────────────────────────
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ── Static files ─────────────────────────────────────────────────────
STATIC_URL  = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL  = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# ── Internationalisation ──────────────────────────────────────────────
LANGUAGE_CODE = 'fr-fr'
TIME_ZONE     = 'Africa/Tunis'
USE_I18N      = True
USE_TZ        = True

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── Authentification ──────────────────────────────────────────────────
LOGIN_URL          = '/accounts/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL= '/accounts/login/'

# ── Chemins modèles IA ────────────────────────────────────────────────
MODELS_AI_DIR = BASE_DIR / 'models_ai'

# ── FastAPI ALIA RAG (apps.modeling) ─────────────────────────────────
MODELING_DIR          = BASE_DIR / 'apps' / 'modeling'
MODELING_DATA_DIR     = MODELING_DIR / 'data'
MODELING_CSV          = MODELING_DATA_DIR / 'vital_products.csv'
MODELING_KB_DIR       = MODELING_DATA_DIR / 'alia_knowledge_db'
MODELING_STATIC_DIR   = MODELING_DATA_DIR / 'static'
MODELING_AUDIO_DIR    = MODELING_STATIC_DIR / 'audio'
MODELING_API_MOUNT_PATH = '/alia-api'

# ── Logging ──────────────────────────────────────────────────────────
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {name} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'metrics_file': {
            'level': 'INFO',
            'class': 'apps.modeling.log_handlers.RealTimeFileHandler',
            'filename': LOGS_DIR / 'metrics.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'alia.metrics': {
            'handlers': ['metrics_file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
    '.ngrok-free.app',
    '.ngrok-free.dev',   # ← ajouter ça
]

CSRF_TRUSTED_ORIGINS = [
    'https://*.ngrok-free.app',
    'https://*.ngrok-free.dev',   # ← ajouter ça
]

MIDDLEWARE.append('whitenoise.middleware.WhiteNoiseMiddleware')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'