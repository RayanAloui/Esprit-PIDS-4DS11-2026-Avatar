import logging
import sys
from pathlib import Path
from django.conf import settings

log = logging.getLogger(__name__)
_nlp_model = None

def get_nlp_model():
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    models_dir = str(settings.MODELS_AI_DIR)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    try:
        # Import des classes nécessaires pour que joblib puisse désérialiser
        import nlp_scoring_train_v2   # noqa — requis pour joblib
        from nlp_scoring_model_v2 import NLPScoringModel
        bundle_path = Path(settings.MODELS_AI_DIR) / 'nlp_scoring_bundle_v2.pkl'
        _nlp_model  = NLPScoringModel.load(str(bundle_path))
        log.info("NLPScoringModel V2 chargé avec succès")
    except Exception as e:
        log.error(f"Erreur chargement NLPScoringModel : {e}")
        _nlp_model = None

    return _nlp_model


def analyze_response(objection: str, response: str,
                     niveau_alia: str = 'Junior') -> dict:
    model = get_nlp_model()

    if model is None:
        return {
            'error'  : True,
            'message': 'Modèle NLP non disponible. Vérifiez models_ai/nlp_scoring_bundle_v2.pkl',
        }

    try:
        result = model.predict(objection.strip(), response.strip())
        result['niveau_alia_input'] = niveau_alia
        return result
    except Exception as e:
        log.error(f"Erreur prédiction NLP : {e}")
        return {'error': True, 'message': str(e)}