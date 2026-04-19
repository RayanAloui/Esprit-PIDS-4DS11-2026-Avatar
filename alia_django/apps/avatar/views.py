"""
Avatar Views — NLP Scoring + Body Language Stream
"""
import json
from django.contrib.auth.decorators import login_required
from django.shortcuts    import render
from django.http         import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .services import analyze_response
from .models   import NLPAnalysis
from .camera   import CameraStream


@login_required
def avatar_index(request):
    history = NLPAnalysis.objects.filter(user=request.user).order_by('-created_at')[:10]
    stats   = _compute_stats(request.user)
    context = {'page':'avatar','history':history,'stats':stats}
    return render(request, 'avatar/index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def analyze_api(request):
    try:
        data        = json.loads(request.body)
        objection   = data.get('objection','').strip()
        response    = data.get('response','').strip()
        niveau_alia = data.get('niveau_alia','Junior')
        if not objection or not response:
            return JsonResponse({'error':True,'message':'Objection et réponse requises.'},status=400)
        result = analyze_response(objection, response, niveau_alia)
        if result.get('error'):
            return JsonResponse(result, status=500)
        _save_analysis(request.user, objection, response, niveau_alia, result)
        return JsonResponse(result)
    except json.JSONDecodeError:
        return JsonResponse({'error':True,'message':'JSON invalide.'},status=400)
    except Exception as e:
        return JsonResponse({'error':True,'message':str(e)},status=500)


@require_http_methods(["GET"])
@login_required
def history_api(request):
    analyses = NLPAnalysis.objects.filter(user=request.user).order_by('-created_at')[:20]
    data = [{'id':a.id,'quality':a.quality,'overall_score':a.overall_score,
             'niveau_alia':a.niveau_alia_pred,'conformite':a.conformite,
             'acrv_score':a.acrv_score,
             'created_at':a.created_at.strftime('%d/%m/%Y %H:%M'),
             'objection':a.objection[:60]+'…' if len(a.objection)>60 else a.objection}
            for a in analyses]
    return JsonResponse({'history':data})


def body_stream(request):
    """GET /avatar/stream/ — flux MJPEG pour <img src="/avatar/stream/">"""
    stream = CameraStream.get_instance()
    if not stream.is_running:
        stream.start()

    def generate():
        import time
        while True:
            frame = stream.get_latest_frame()
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + frame + b'\r\n')
            time.sleep(0.033)

    return StreamingHttpResponse(
        generate(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@require_http_methods(["GET"])
def body_status(request):
    """GET /avatar/body-status/ — scores JSON temps réel"""
    stream = CameraStream.get_instance()
    return JsonResponse(stream.get_latest_scores())


@csrf_exempt
@require_http_methods(["POST"])
def body_control(request):
    """POST /avatar/body-control/ — { action: 'start'|'stop' }"""
    try:
        data   = json.loads(request.body)
        action = data.get('action','start')
        stream = CameraStream.get_instance()
        if action == 'start':
            stream.start()
            return JsonResponse({'status':'started'})
        elif action == 'stop':
            stream.stop()
            return JsonResponse({'status':'stopped'})
        return JsonResponse({'error':'Action invalide'},status=400)
    except Exception as e:
        return JsonResponse({'error':str(e)},status=500)


def _save_analysis(user, objection, response, niveau_alia, result):
    try:
        scores = result.get('scores',{})
        NLPAnalysis.objects.create(
            user=user if user.is_authenticated else None,
            objection=objection, response=response,
            niveau_alia_input=niveau_alia,
            quality=result.get('quality','—'),
            overall_score=result.get('overall_score',0),
            score_scientific=scores.get('scientific_accuracy',0),
            score_clarity=scores.get('communication_clarity',0),
            score_objection=scores.get('objection_handling',0),
            sentiment=result.get('sentiment','—'),
            niveau_alia_pred=result.get('niveau_alia','—'),
            visit_format=result.get('visit_format','—'),
            conformite=result.get('conformite',True),
            acrv_score=result.get('acrv_score',0),
        )
    except Exception:
        pass


def _compute_stats(user=None):
    if user:
        qs = NLPAnalysis.objects.filter(user=user)
    else:
        qs = NLPAnalysis.objects.all()
    total = qs.count()
    if total == 0:
        return {'total':0,'avg_score':0,'pct_excellent':0,'pct_conforme':0}
    from django.db.models import Avg
    agg=qs.aggregate(avg_score=Avg('overall_score'))
    n_ex=qs.filter(quality='Excellent').count()
    n_co=qs.filter(conformite=True).count()
    return {'total':total,'avg_score':round(agg['avg_score'] or 0,2),
            'pct_excellent':round(n_ex/total*100),'pct_conforme':round(n_co/total*100)}
