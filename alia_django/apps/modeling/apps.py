from django.apps import AppConfig


class ModelingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.modeling"
    label = "modeling"
    verbose_name = "ALIA Modeling (FastAPI RAG)"
