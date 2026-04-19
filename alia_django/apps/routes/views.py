"""
Routes Views — Page Route Optimizer
"""
import json
from django.contrib.auth.decorators import login_required
from django.shortcuts  import render
from django.http       import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .services import optimize_route, get_all_pharmacies

DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
DAYS_FR = {
    'Monday':'Lundi','Tuesday':'Mardi','Wednesday':'Mercredi',
    'Thursday':'Jeudi','Friday':'Vendredi',
    'Saturday':'Samedi','Sunday':'Dimanche',
}
ZONES = {
    -1: 'Toutes les zones',
     0: 'Zone A — Nord-Est',
     1: 'Zone B — Centre',
     2: 'Zone C — Centre-Ouest',
     3: 'Zone D — Sud',
}


@login_required
def routes_index(request):
    """Page principale Route Optimizer."""
    pharmacies = get_all_pharmacies()
    context = {
        'page'      : 'routes',
        'days'      : [(d, DAYS_FR[d]) for d in DAYS],
        'zones'     : list(ZONES.items()),
        'hours'     : list(range(8, 19)),
        'pharmacies': json.dumps(pharmacies),
        'depot'     : {'lat': 36.8190, 'lon': 10.1660,
                       'name': 'VITAL SA — Siège (Tunis)'},
    }
    return render(request, 'routes/index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def optimize_api(request):
    """
    API endpoint — Optimisation de tournée.
    POST /routes/optimize/
    Body : { target_day, target_hour, n_stops, depot_lat, depot_lon, cluster_id }
    Returns : JSON résultat complet
    """
    try:
        data        = json.loads(request.body)
        target_day  = data.get('target_day',  'Monday')
        target_hour = int(data.get('target_hour', 10))
        n_stops     = int(data.get('n_stops', 8))
        depot_lat   = float(data.get('depot_lat', 36.8190))
        depot_lon   = float(data.get('depot_lon', 10.1660))
        cluster_id  = data.get('cluster_id', None)
        if cluster_id is not None:
            cluster_id = int(cluster_id)

        # Validation
        n_stops    = max(4, min(n_stops, 10))
        target_hour= max(7, min(target_hour, 18))

        result = optimize_route(
            target_day, target_hour, n_stops,
            depot_lat, depot_lon, cluster_id
        )

        if result.get('error'):
            return JsonResponse(result, status=500)

        # Gère les NaN → null pour JSON valide
        import math
        def clean_nan(obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean_nan(i) for i in obj]
            return obj

        clean_result = clean_nan(result)
        return JsonResponse(clean_result)

    except json.JSONDecodeError:
        return JsonResponse({'error': True, 'message': 'JSON invalide.'}, status=400)
    except Exception as e:
        return JsonResponse({'error': True, 'message': str(e)}, status=500)
