from django.apps import AppConfig

class ModelingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.modeling"
    label = "modeling"
    verbose_name = "ALIA Modeling (FastAPI RAG)"
    def ready(self):
        from .handlers import _ensure_whisper_loaded
        try:
            print("[modeling] Preloading Whisper model...")
            _ensure_whisper_loaded()
            print("[modeling] Whisper loaded successfully.")
        except Exception as e:
            print(f"[modeling] Whisper preload failed: {e}")

