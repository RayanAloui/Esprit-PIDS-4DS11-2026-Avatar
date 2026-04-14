import json
import asyncio
import concurrent.futures

from asgiref.sync import sync_to_async
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

def _run(coro):
    """Compatible Python 3.14 — crée toujours une nouvelle boucle propre."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()
def _run_sync(func, *args, **kwargs):
    """Run a synchronous function in a thread pool to avoid blocking."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *args, **kwargs)
        return future.result()

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
        data = _run(ask_alia_json(text))
        return JsonResponse(data)
    except Exception as e:
        print(f"[ERROR] ask_alia_view: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"detail": str(e)}, status=500)


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
        data = _run(listen_json(audio_bytes))
        return JsonResponse(data)
    except Exception as e:
        print(f"[ERROR] listen_view: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"detail": str(e)}, status=500)

@csrf_exempt
def reset_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        rt = get_runtime()
        if rt is None:
            raise RuntimeError("Runtime not initialized")
        if not hasattr(rt, 'alia') or rt.alia is None:
            raise RuntimeError("Alia orchestrator not initialized")
        
        # Call reset synchronously in thread pool
        _run_sync(rt.alia.reset)
        return JsonResponse({"status": "ok"})
    except Exception as e:
        print(f"[ERROR] reset_view: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"status": "error", "detail": str(e)}, status=500)

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
    try:
        rt = get_runtime()
        if rt is None:
            raise RuntimeError("Runtime not initialized")
        if not hasattr(rt, 'alia') or rt.alia is None:
            raise RuntimeError("Alia orchestrator not initialized")
        
        # Call set_mode synchronously in thread pool
        _run_sync(rt.alia.set_mode, mode)
        return JsonResponse({"status": "ok", "mode": mode})
    except Exception as e:
        print(f"[ERROR] set_mode_view: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"status": "error", "detail": str(e)}, status=500)

@csrf_exempt
def force_training_mode(request):
    """Force le mode training pour tester"""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        rt = get_runtime()
        rt.alia.set_mode("training")
        rt.alia.reset()
        return JsonResponse({"status": "ok", "mode": "training"})
    except Exception as e:
        return JsonResponse({"status": "error", "detail": str(e)}, status=500)
        
@require_GET
def health_view(request):
    return JsonResponse(health_json())
