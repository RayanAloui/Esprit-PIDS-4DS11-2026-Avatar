"""
Analytics Views — Tableau de bord de progression ALIA
"""
import json
import math
from collections import defaultdict
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http      import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db.models import Avg, Count, Min, Max, F
from django.utils     import timezone

from apps.avatar.models import NLPAnalysis


@login_required
def analytics_index(request):
    """Page principale Analytics."""
    total = NLPAnalysis.objects.filter(user=request.user).count()
    context = {
        'page' : 'analytics',
        'total': total,
        'data' : json.dumps(_build_dashboard_data(request.user)) if total > 0 else '{}',
        'has_data': total >= 2,
    }
    return render(request, 'analytics/index.html', context)


@login_required
@require_http_methods(["GET"])
def analytics_data(request):
    """GET /analytics/data/ — données JSON complètes pour les graphiques."""
    data = _build_dashboard_data(request.user)
    return JsonResponse(data)


@csrf_exempt
@require_http_methods(["POST"])
def action_plan_api(request):
    """
    POST /analytics/action-plan/
    Appelle l'API Claude pour générer un plan d'action personnalisé.
    """
    try:
        data    = json.loads(request.body)
        summary = data.get('summary', '')

        if not summary:
            return JsonResponse({'error': True, 'message': 'Résumé vide.'}, status=400)

        plan = _generate_action_plan(summary)
        return JsonResponse({'plan': plan})

    except Exception as e:
        return JsonResponse({'error': True, 'message': str(e)}, status=500)


# ══════════════════════════════════════════════════════════════════════
# AGGREGATIONS
# ══════════════════════════════════════════════════════════════════════

def _build_dashboard_data(user) -> dict:
    """Construit toutes les données analytiques depuis SQLite."""
    qs = NLPAnalysis.objects.filter(user=user).order_by('created_at')
    if not qs.exists():
        return {'username': user.get_full_name() or user.username if user else 'Inconnu'}

    analyses = list(qs.values(
        'id', 'created_at', 'overall_score',
        'score_scientific', 'score_clarity', 'score_objection',
        'quality', 'niveau_alia_pred', 'conformite', 'acrv_score',
        'sentiment', 'visit_format', 'objection',
    ))

    return {
        'progression'  : _progression(analyses),
        'radar'        : _radar(analyses),
        'niveau_hist'  : _niveau_histogram(analyses),
        'quality_dist' : _quality_distribution(analyses),
        'conformite'   : _conformite_stats(analyses),
        'acrv_trend'   : _acrv_trend(analyses),
        'format_dist'  : _format_distribution(analyses),
        'summary_stats': _summary_stats(analyses),
        'objection_clusters': _objection_clusters(analyses),
        'username': user.get_full_name() or user.username if user else 'Inconnu',
    }


def _progression(analyses: list) -> dict:
    """Courbe de progression du score global dans le temps."""
    labels = []
    scores = []
    sci    = []
    clar   = []
    obj    = []

    for a in analyses:
        dt = a['created_at']
        if hasattr(dt, 'strftime'):
            labels.append(dt.strftime('%d/%m %H:%M'))
        else:
            labels.append(str(dt)[:16])

        scores.append(round(float(a['overall_score'] or 0), 2))
        sci.append(round(float(a['score_scientific'] or 0), 2))
        clar.append(round(float(a['score_clarity'] or 0), 2))
        obj.append(round(float(a['score_objection'] or 0), 2))

    # Moyenne mobile sur 3 sessions
    moving_avg = []
    for i in range(len(scores)):
        window = scores[max(0, i-2):i+1]
        moving_avg.append(round(sum(window)/len(window), 2))

    return {
        'labels'    : labels,
        'scores'    : scores,
        'moving_avg': moving_avg,
        'sci'       : sci,
        'clar'      : clar,
        'obj'       : obj,
    }


def _radar(analyses: list) -> dict:
    """Radar des 5 compétences clés moyennées."""
    n = len(analyses)
    if n == 0:
        return {}

    avg_sci   = sum(float(a['score_scientific'] or 0) for a in analyses) / n
    avg_clar  = sum(float(a['score_clarity']    or 0) for a in analyses) / n
    avg_obj   = sum(float(a['score_objection']  or 0) for a in analyses) / n
    avg_acrv  = sum(float(a['acrv_score']       or 0) for a in analyses) / n * 2.5  # ramené sur 10
    avg_conf  = sum(1 for a in analyses if a['conformite']) / n * 10

    return {
        'labels': [
            'Précision\nscientifique',
            'Clarté\ncommunication',
            'Gestion\nobjections',
            'Méthode\nA-C-R-V',
            'Conformité\nréglementaire',
        ],
        'values': [
            round(avg_sci,  2),
            round(avg_clar, 2),
            round(avg_obj,  2),
            round(avg_acrv, 2),
            round(avg_conf, 2),
        ],
    }


def _niveau_histogram(analyses: list) -> dict:
    """Distribution et évolution des niveaux ALIA prédits."""
    ordre  = ['Débutant', 'Junior', 'Confirmé', 'Expert']
    counts = defaultdict(int)
    for a in analyses:
        n = a['niveau_alia_pred'] or 'Junior'
        counts[n] += 1

    # Timeline des niveaux
    timeline = [a['niveau_alia_pred'] or 'Junior' for a in analyses]
    encoded  = [ordre.index(n) if n in ordre else 1 for n in timeline]

    labels = []
    for a in analyses:
        dt = a['created_at']
        if hasattr(dt, 'strftime'):
            labels.append(dt.strftime('%d/%m'))
        else:
            labels.append(str(dt)[:10])

    return {
        'distribution': {n: counts[n] for n in ordre},
        'timeline'    : encoded,
        'labels'      : labels,
        'ordre'       : ordre,
    }


def _quality_distribution(analyses: list) -> dict:
    """Distribution Excellent / Bon / Faible."""
    counts = defaultdict(int)
    for a in analyses:
        counts[a['quality'] or 'Faible'] += 1
    return {
        'Excellent': counts['Excellent'],
        'Bon'      : counts['Bon'],
        'Faible'   : counts['Faible'],
    }


def _conformite_stats(analyses: list) -> dict:
    """Taux de conformité et évolution."""
    n          = len(analyses)
    n_ok       = sum(1 for a in analyses if a['conformite'])
    taux       = round(n_ok / n * 100, 1) if n > 0 else 0

    # Par tranche de 5 sessions
    tranches   = []
    labels     = []
    chunk_size = max(1, n // 5 if n >= 5 else 1)
    for i in range(0, n, chunk_size):
        chunk  = analyses[i:i+chunk_size]
        t_ok   = sum(1 for a in chunk if a['conformite'])
        tranches.append(round(t_ok / len(chunk) * 100, 1))
        labels.append(f"S{i//chunk_size + 1}")

    return {
        'taux'     : taux,
        'n_ok'     : n_ok,
        'n_fail'   : n - n_ok,
        'tranches' : tranches,
        'labels'   : labels,
    }


def _acrv_trend(analyses: list) -> dict:
    """Évolution du score A-C-R-V dans le temps."""
    labels = []
    values = []
    for a in analyses:
        dt = a['created_at']
        if hasattr(dt, 'strftime'):
            labels.append(dt.strftime('%d/%m'))
        else:
            labels.append(str(dt)[:10])
        values.append(int(a['acrv_score'] or 0))

    avg = round(sum(values) / len(values), 2) if values else 0
    return {'labels': labels, 'values': values, 'avg': avg}


def _format_distribution(analyses: list) -> dict:
    """Distribution des formats de visite."""
    counts = defaultdict(int)
    for a in analyses:
        counts[a['visit_format'] or 'Standard'] += 1
    return dict(counts)


def _summary_stats(analyses: list) -> dict:
    """Statistiques résumées pour les cartes KPI."""
    n = len(analyses)
    if n == 0:
        return {}

    scores    = [float(a['overall_score'] or 0) for a in analyses]
    avg_score = round(sum(scores) / n, 2)
    max_score = round(max(scores), 2)
    min_score = round(min(scores), 2)

    # Tendance : dernières 3 sessions vs premières 3
    trend = 0
    if n >= 6:
        first3 = sum(scores[:3]) / 3
        last3  = sum(scores[-3:]) / 3
        trend  = round(last3 - first3, 2)

    # Niveau dominant
    niveaux   = [a['niveau_alia_pred'] or 'Junior' for a in analyses]
    n_counts  = defaultdict(int)
    for nv in niveaux:
        n_counts[nv] += 1
    niveau_dominant = max(n_counts, key=n_counts.get)

    # Dernier niveau
    dernier_niveau = analyses[-1]['niveau_alia_pred'] or 'Junior'

    # Taux excellent
    n_exc = sum(1 for a in analyses if a['quality'] == 'Excellent')

    return {
        'total'           : n,
        'avg_score'       : avg_score,
        'max_score'       : max_score,
        'min_score'       : min_score,
        'trend'           : trend,
        'trend_positive'  : trend >= 0,
        'niveau_dominant' : niveau_dominant,
        'dernier_niveau'  : dernier_niveau,
        'taux_excellent'  : round(n_exc / n * 100, 1),
        'taux_conforme'   : round(sum(1 for a in analyses if a['conformite']) / n * 100, 1),
    }


def _objection_clusters(analyses: list) -> dict:
    """Clustering simple des objections par mots-clés."""
    categories = {
        'Efficacité'  : ['convaincu', 'efficac', 'résultat', 'marche', 'preuve'],
        'Temps'       : ['temps', 'pressé', 'rapide', 'court', 'minute'],
        'Habitudes'   : ['habitude', 'autre produit', 'fidèle', 'prescri'],
        'Prix/Budget' : ['prix', 'cher', 'budget', 'coût', 'remboursé'],
        'Sécurité'    : ['effet', 'secondaire', 'risque', 'tolérance', 'sûr'],
        'Autre'       : [],
    }

    counts     = defaultdict(int)
    avg_scores = defaultdict(list)

    for a in analyses:
        objection = (a['objection'] or '').lower()
        matched   = False
        for cat, keywords in categories.items():
            if cat == 'Autre':
                continue
            if any(kw in objection for kw in keywords):
                counts[cat] += 1
                avg_scores[cat].append(float(a['overall_score'] or 0))
                matched = True
                break
        if not matched:
            counts['Autre'] += 1
            avg_scores['Autre'].append(float(a['overall_score'] or 0))

    result = {}
    for cat in categories:
        if counts[cat] > 0:
            result[cat] = {
                'count'    : counts[cat],
                'avg_score': round(sum(avg_scores[cat]) / len(avg_scores[cat]), 2),
            }

    return result


def _generate_action_plan(summary: str) -> list:
    """Génère un plan d'action via l'API locale Ollama."""
    import urllib.request
    import json

    prompt = f"""Tu es ALIA, le coach IA expert de VITAL SA pour les délégués médicaux.
Ton objectif est de fournir 3 conseils pratiques et concrets pour aider le délégué.

Voici le résumé analytique des performances d'un délégué :
{summary}

Génère exactement 3 axes de progression prioritaires et actionnables.
Chaque axe doit être très concret, mesurable, et aligné sur la méthode VITAL SA (A-C-R-V, seuils ALIA, conformité).

Réponds UNIQUEMENT en JSON valide contenant un tableau de 3 objets avec les clés exactes suivantes : "axe", "description", "priorite", "seuil_cible".
La clé "priorite" doit valoir "haute", "moyenne" ou "faible".

Exemple de format attendu exact:
[
  {{"axe": "Titre court", "description": "Explication concrète.", "priorite": "haute", "seuil_cible": "Score > 7"}},
  ...
]"""

    payload = json.dumps({
        "model": "llama3.2:latest",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 800
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            raw = result.get("response", "").strip()
            # Clean possible markdown wrap
            if "```" in raw:
                import re
                m = re.search(r'```(?:json)?(.*?)```', raw, re.DOTALL)
                if m:
                    raw = m.group(1).strip()
            
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Parfois Ollama force un objet si 'format':'json'
                if len(parsed) == 1 and isinstance(list(parsed.values())[0], list):
                    return list(parsed.values())[0]
                return [parsed]
            elif isinstance(parsed, list):
                return parsed
            else:
                return _fallback_plan()
    except Exception as e:
        import logging
        logging.error(f"[ALIA] Ollama API error in _generate_action_plan: {e}")
        return _fallback_plan()

def _fallback_plan() -> list:
    return [
        {
            "axe"         : "Améliorer la gestion des objections",
            "description" : "Appliquez systématiquement les 4 étapes A-C-R-V.",
            "priorite"    : "haute",
            "seuil_cible" : "Score objection > 7.0"
        },
        {
            "axe"         : "Conformité réglementaire",
            "description" : "Éliminez tout mot tueur de vos réponses.",
            "priorite"    : "haute",
            "seuil_cible" : "Taux conformité = 100%"
        },
        {
            "axe"         : "Signal BIP et closing",
            "description" : "Détectez les signaux d'intérêt très rapidement.",
            "priorite"    : "moyenne",
            "seuil_cible" : "BIP signal dans 80%"
        },
    ]


@csrf_exempt
@login_required
@require_http_methods(["POST"])
def export_pdf_view(request):
    """
    Exporte le Dashboard de l'utilisateur en PDF.
    Si un plan d'action (JSON string) est envoyé depuis le front, il sera intégré au tableau.
    """
    from django.http import HttpResponse
    from .pdf_service import generate_analytics_pdf

    try:
        # 1. Obtenir les KPI actuels
        data = _build_dashboard_data(request.user)
        
        # 2. Récupérer le plan d'action et les images passés depuis le front
        req_data = json.loads(request.body)
        action_plan = req_data.get('action_plan', None)
        images = req_data.get('images', {})
        
        # 3. Générer le PDF
        pdf_buffer = generate_analytics_pdf(data, action_plan, images)
        
        # 4. Retourner le document
        response = HttpResponse(pdf_buffer.getvalue(), content_type='application/pdf')
        filename = f"ALIA_Analytics_{request.user.username}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
        import logging
        logging.error(f"[ALIA] Export PDF Analytics Failed : {e}")
        return JsonResponse({"error": True, "message": str(e)}, status=500)
