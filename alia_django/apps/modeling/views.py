import asyncio
import json
import traceback

from django.conf import settings
from django.http import HttpResponseNotAllowed, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from django.views.static import serve

from apps.modeling.handlers import (
    ask_alia_json,
    debug_search_json,
    health_json,
    listen_json,
)
from apps.modeling.metrics import get_system_metrics
from apps.modeling.rendering import render_modeling_index
from apps.modeling.runtime import get_runtime

@require_GET
def modeling_index(request):
    return render_modeling_index(request)


@require_GET
def modeling_static(request, relpath):
    return serve(request, relpath, document_root=str(settings.MODELING_STATIC_DIR))


@csrf_exempt
def ask_alia_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        body = json.loads(request.body.decode())
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)
    text = body.get("text", "")
    try:
        data = asyncio.run(ask_alia_json(text))
        return JsonResponse(data)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"detail": str(e) or type(e).__name__}, status=500)


@csrf_exempt
def debug_search_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        body = json.loads(request.body.decode())
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)
    text = body.get("text", "")
    return JsonResponse(debug_search_json(text))


@csrf_exempt
def listen_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    audio_file = request.FILES.get("audio")
    if not audio_file:
        return JsonResponse({"detail": "Missing audio file"}, status=400)
    try:
        audio_bytes = audio_file.read()
        data = asyncio.run(listen_json(audio_bytes))
        return JsonResponse(data)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"detail": str(e) or type(e).__name__}, status=500)


@csrf_exempt
def reset_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    rt = get_runtime()
    rt.alia.history = []
    return JsonResponse({"status": "ok"})


@csrf_exempt
def set_mode_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        body = json.loads(request.body.decode())
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)
    mode = body.get("mode", "commercial")
    if mode not in ("commercial", "training"):
        return JsonResponse({"detail": "Invalid mode"}, status=400)
    rt = get_runtime()
    rt.alia.set_mode(mode)
    return JsonResponse({"status": "ok", "mode": mode})

@require_GET
def metrics_view(request):
    return JsonResponse(get_system_metrics())


@require_GET
def health_view(request):
    return JsonResponse(health_json())
