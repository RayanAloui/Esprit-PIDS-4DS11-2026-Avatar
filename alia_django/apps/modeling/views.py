import json

from django.conf import settings
from django.http import HttpResponseNotAllowed, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from django.views.static import serve

from apps.modeling.handlers import ask_alia_json, debug_search_json, health_json
from apps.modeling.rendering import render_modeling_index


@require_GET
def modeling_index(request):
    return render_modeling_index(request)


@require_GET
def modeling_static(request, relpath):
    return serve(request, relpath, document_root=str(settings.MODELING_STATIC_DIR))


@csrf_exempt
async def ask_alia_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        body = json.loads(request.body.decode())
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)
    text = body.get("text", "")
    try:
        data = await ask_alia_json(text)
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({"detail": str(e)}, status=500)


@csrf_exempt
async def debug_search_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        body = json.loads(request.body.decode())
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)
    text = body.get("text", "")
    return JsonResponse(debug_search_json(text))


@require_GET
def health_view(request):
    return JsonResponse(health_json())
