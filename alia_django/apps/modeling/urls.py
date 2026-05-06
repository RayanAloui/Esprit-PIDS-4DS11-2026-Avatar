"""
Same paths as the FastAPI mount (MODELING_API_MOUNT_PATH) so ``runserver`` (WSGI) works.

Under uvicorn + alia.asgi, Starlette handles these URLs first; these patterns are unused.
"""
from django.urls import path

from django.conf import settings

from . import views

_prefix = settings.MODELING_API_MOUNT_PATH.strip("/")

urlpatterns = [
    path(f"{_prefix}/static/<path:relpath>", views.modeling_static, name="modeling_static"),
    path(f"{_prefix}/listen", views.listen_view, name="modeling_listen"),
    path(f"{_prefix}/ask_alia", views.ask_alia_view, name="modeling_ask_alia"),
    path(f"{_prefix}/health", views.health_view, name="modeling_health"),
    path(f"{_prefix}/debug/search", views.debug_search_view, name="modeling_debug_search"),
    path(f"{_prefix}/debug/metrics", views.metrics_view, name="modeling_debug_metrics"),
    path(f"{_prefix}/", views.modeling_index, name="modeling_index"),
    path(f"{_prefix}/reset", views.reset_view, name="modeling_reset"),
    path(f"{_prefix}/set_mode", views.set_mode_view, name="modeling_set_mode"),
]
