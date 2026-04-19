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
    lang = body.get("lang", "fr")
    
    text_lower = text.lower()
    import re
    if re.search(r'[\u0600-\u06FF]', text_lower):
        lang = 'ar'
    else:
        words = set(re.findall(r'\b\w+\b', text_lower))
        en_words = {'the', 'is', 'are', 'you', 'and', 'to', 'of', 'in', 'hello', 'doctor', 'i', 'my', 'yes', 'no', 'what', 'how', 'good', 'morning'}
        es_words = {'el', 'la', 'los', 'las', 'de', 'que', 'en', 'un', 'una', 'hola', 'como', 'para', 'sí', 'no', 'bien', 'buenos', 'días', 'doctor', 'usted'}
        fr_words = {'le', 'la', 'les', 'des', 'un', 'une', 'bonjour', 'est', 'et', 'pour', 'oui', 'non', 'comment', 'bien', 'docteur', 'vous', 'avec'}
        
        score_en = len(words.intersection(en_words))
        score_es = len(words.intersection(es_words))
        score_fr = len(words.intersection(fr_words))
        
        if score_en > score_es and score_en > score_fr:
            lang = 'en'
        elif score_es > score_en and score_es > score_fr:
            lang = 'es'
        elif score_fr > score_en and score_fr > score_es:
            lang = 'fr'
    try:
        data = asyncio.run(ask_alia_json(text, lang=lang))
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
def health_view(request):
    return JsonResponse(health_json())
