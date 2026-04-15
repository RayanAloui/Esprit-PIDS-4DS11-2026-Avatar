import logging
import sys
from pathlib import Path
from django.conf import settings

log = logging.getLogger(__name__)
_route_optimizer = None

def get_route_optimizer():
    global _route_optimizer
    if _route_optimizer is not None:
        return _route_optimizer

    models_dir = str(settings.MODELS_AI_DIR)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    try:
        # Import des classes nécessaires pour que joblib puisse désérialiser
        import route_optimizer   # noqa — requis pour joblib
        from route_model import RouteOptimizer
        bundle_path      = Path(settings.MODELS_AI_DIR) / 'route_model.pkl'
        _route_optimizer = RouteOptimizer.load(str(bundle_path))
        log.info("RouteOptimizer chargé avec succès")
    except Exception as e:
        log.error(f"Erreur chargement RouteOptimizer : {e}")
        _route_optimizer = None

    return _route_optimizer


def optimize_route(target_day, target_hour, n_stops,
                   depot_lat, depot_lon, cluster_id=None) -> dict:
    optimizer = get_route_optimizer()

    if optimizer is None:
        return {
            'error'  : True,
            'message': 'Optimiseur non disponible. Vérifiez models_ai/route_model.pkl',
        }

    try:
        result = optimizer.optimize(
            target_day  = target_day,
            target_hour = target_hour,
            n_stops     = n_stops,
            depot_lat   = depot_lat,
            depot_lon   = depot_lon,
            cluster_id  = cluster_id if cluster_id != -1 else None,
        )
        return result
    except Exception as e:
        log.error(f"Erreur optimisation route : {e}")
        return {'error': True, 'message': str(e)}


def get_all_pharmacies() -> list:
    optimizer = get_route_optimizer()
    if optimizer is None:
        return []
    try:
        df = optimizer.get_pharmacies_df()
        return df[['venue_name','venue_address','latitude','longitude',
                   'priority_score','day_mean','cluster','data_source']].to_dict('records')
    except Exception:
        return []
