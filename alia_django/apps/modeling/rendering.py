"""Shared HTML for the modeling UI (Django + FastAPI index)."""
from django.shortcuts import render
from django.test import RequestFactory
from django.urls import reverse

_rf = RequestFactory()


def modeling_page_context():
    return {
        "page": "modeling",
        "modeling_api_base": reverse("modeling_index").rstrip("/"),
    }


def render_modeling_index(request):
    return render(request, "modeling/index.html", modeling_page_context())


def render_modeling_index_sync():
    """For FastAPI index when Django templates must run outside a real request."""
    req = _rf.get("/alia-api/", HTTP_HOST="127.0.0.1")
    return render(req, "modeling/index.html", modeling_page_context())
