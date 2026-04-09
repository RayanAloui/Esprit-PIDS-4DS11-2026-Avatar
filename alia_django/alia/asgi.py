"""
ASGI entrypoint: Django + FastAPI (ALIA RAG) on a sub-path.

Run with an ASGI server, for example:
  uvicorn alia.asgi:application --reload --host 127.0.0.1 --port 8000

``manage.py runserver`` uses WSGI by default and will not serve the mounted API.
"""
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "alia.settings")

from django.conf import settings
from django.core.asgi import get_asgi_application
from starlette.applications import Starlette
from starlette.routing import Mount

django_asgi_app = get_asgi_application()

_modeling_app = None


async def modeling_asgi(scope, receive, send):
    global _modeling_app
    if _modeling_app is None:
        from apps.modeling.fastapi_app import create_app

        _modeling_app = create_app()
    await _modeling_app(scope, receive, send)


_prefix = getattr(settings, "MODELING_API_MOUNT_PATH", "/alia-api").rstrip("/") or "/alia-api"

application = Starlette(
    routes=[
        Mount(_prefix, modeling_asgi),
        Mount("/", django_asgi_app),
    ],
)
