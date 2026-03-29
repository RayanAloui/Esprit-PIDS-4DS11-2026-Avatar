"""
Avatar Views — Page NLP Scoring + Historique
"""
import json
from django.shortcuts   import render
from django.http        import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils       import timezone

from .services import analyze_response
from .models   import NLPAnalysis


def avatar_index(request):
    """Page principale Avatar — formulaire + résultats + historique."""
    history = NLPAnalysis.objects.order_by('-created_at')[:10]
    stats   = _compute_stats()
    context = {
        'page'   : 'avatar',
        'history': history,
        'stats'  : stats,
    }
    return render(request, 'avatar/index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def analyze_api(request):
    """
    API endpoint — Analyse NLP d'une réponse.
    POST /avatar/analyze/
    Body : { objection, response, niveau_alia }
    Returns : JSON résultat complet
    """
    try:
        data        = json.loads(request.body)
        objection   = data.get('objection', '').strip()
        response    = data.get('response',  '').strip()
        niveau_alia = data.get('niveau_alia', 'Junior')

        if not objection or not response:
            return JsonResponse(
                {'error': True, 'message': 'Objection et réponse requises.'},
                status=400
            )

        # ── Analyse ───────────────────────────────────────────────────
        result = analyze_response(objection, response, niveau_alia)

        if result.get('error'):
            return JsonResponse(result, status=500)

        # ── Sauvegarde SQLite ─────────────────────────────────────────
        _save_analysis(objection, response, niveau_alia, result)

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse(
            {'error': True, 'message': 'JSON invalide.'}, status=400
        )
    except Exception as e:
        return JsonResponse(
            {'error': True, 'message': str(e)}, status=500
        )


@require_http_methods(["GET"])
def history_api(request):
    """
    API endpoint — Historique des analyses.
    GET /avatar/history/
    Returns : JSON liste des 20 dernières analyses
    """
    analyses = NLPAnalysis.objects.order_by('-created_at')[:20]
    data = [
        {
            'id'           : a.id,
            'quality'      : a.quality,
            'overall_score': a.overall_score,
            'niveau_alia'  : a.niveau_alia_pred,
            'conformite'   : a.conformite,
            'acrv_score'   : a.acrv_score,
            'created_at'   : a.created_at.strftime('%d/%m/%Y %H:%M'),
            'objection'    : a.objection[:60] + '…' if len(a.objection) > 60 else a.objection,
        }
        for a in analyses
    ]
    return JsonResponse({'history': data})


# ── Helpers ───────────────────────────────────────────────────────────

def _save_analysis(objection, response, niveau_alia, result):
    """Sauvegarde une analyse dans SQLite."""
    try:
        scores = result.get('scores', {})
        NLPAnalysis.objects.create(
            objection         = objection,
            response          = response,
            niveau_alia_input = niveau_alia,
            quality           = result.get('quality', '—'),
            overall_score     = result.get('overall_score', 0),
            score_scientific  = scores.get('scientific_accuracy', 0),
            score_clarity     = scores.get('communication_clarity', 0),
            score_objection   = scores.get('objection_handling', 0),
            sentiment         = result.get('sentiment', '—'),
            niveau_alia_pred  = result.get('niveau_alia', '—'),
            visit_format      = result.get('visit_format', '—'),
            conformite        = result.get('conformite', True),
            acrv_score        = result.get('acrv_score', 0),
        )
    except Exception as e:
        pass  # Ne pas bloquer si la sauvegarde échoue


def _compute_stats():
    """Calcule les statistiques globales pour l'affichage."""
    total = NLPAnalysis.objects.count()
    if total == 0:
        return {'total': 0, 'avg_score': 0, 'pct_excellent': 0,
                'pct_conforme': 0}

    from django.db.models import Avg, Count
    agg = NLPAnalysis.objects.aggregate(avg_score=Avg('overall_score'))
    n_excellent = NLPAnalysis.objects.filter(quality='Excellent').count()
    n_conforme  = NLPAnalysis.objects.filter(conformite=True).count()

    return {
        'total'         : total,
        'avg_score'     : round(agg['avg_score'] or 0, 2),
        'pct_excellent' : round(n_excellent / total * 100),
        'pct_conforme'  : round(n_conforme  / total * 100),
    }
