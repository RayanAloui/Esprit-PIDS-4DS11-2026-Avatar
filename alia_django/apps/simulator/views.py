"""
Simulator Views
"""
import json
from django.shortcuts  import render
from django.http       import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .profiles import DOCTOR_PROFILES, VITAL_PRODUCTS
from .engine   import SimulationSession

SESSION_KEY = 'alia_sim_session'


def simulator_index(request):
    """Page setup — choix médecin, produit, niveau."""
    context = {
        'page'    : 'simulator',
        'doctors' : DOCTOR_PROFILES,
        'products': VITAL_PRODUCTS,
        'niveaux' : ['Débutant', 'Junior', 'Confirmé', 'Expert'],
    }
    return render(request, 'simulator/index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def start_simulation(request):
    """POST /simulator/start/ — démarre une nouvelle simulation."""
    try:
        data       = json.loads(request.body)
        doctor_id  = data.get('doctor_id',  'sceptique')
        product_id = data.get('product_id', 'ferrimax')
        niveau     = data.get('niveau_alia','Junior')

        session = SimulationSession(doctor_id, product_id, niveau)
        first   = session.first_message()

        # Stocker en session Django
        request.session[SESSION_KEY] = session.to_dict()
        request.session.modified     = True

        return JsonResponse({
            'ok'     : True,
            'first'  : first,
            'doctor' : session.doctor,
            'product': session.product,
        })
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    """POST /simulator/message/ — envoie la réponse du délégué."""
    try:
        data    = json.loads(request.body)
        text    = data.get('text', '').strip()

        if not text:
            return JsonResponse({'ok': False, 'error': 'Message vide.'}, status=400)

        # Restaurer la session
        session_data = request.session.get(SESSION_KEY)
        if not session_data:
            return JsonResponse({'ok': False, 'error': 'Session expirée.'}, status=400)

        session  = SimulationSession.from_dict(session_data)
        result   = session.process_delegate_response(text)

        # Sauvegarder session mise à jour
        request.session[SESSION_KEY] = session.to_dict()
        request.session.modified     = True

        result['ok'] = True
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_report(request):
    """GET /simulator/report/ — génère le rapport final."""
    try:
        session_data = request.session.get(SESSION_KEY)
        if not session_data:
            return JsonResponse({'ok': False, 'error': 'Aucune session.'}, status=400)

        session = SimulationSession.from_dict(session_data)
        report  = session.generate_report()
        report['ok'] = True
        return JsonResponse(report)

    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def reset_simulation(request):
    """POST /simulator/reset/ — remet à zéro."""
    if SESSION_KEY in request.session:
        del request.session[SESSION_KEY]
    return JsonResponse({'ok': True})


# ══════════════════════════════════════════════════════════════════════
# BODY LANGUAGE — réutilise CameraStream de apps.avatar
# ══════════════════════════════════════════════════════════════════════

from django.http import StreamingHttpResponse
import time as _time

def sim_body_stream(request):
    """GET /simulator/stream/ — flux MJPEG webcam pour le simulator."""
    from apps.avatar.camera import CameraStream
    stream = CameraStream.get_instance()
    if not stream.is_running:
        stream.start()

    def generate():
        while True:
            frame = stream.get_latest_frame()
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + frame + b'\r\n')
            _time.sleep(0.033)

    return StreamingHttpResponse(
        generate(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@require_http_methods(["GET"])
def sim_body_status(request):
    """GET /simulator/body-status/ — scores LSTM JSON."""
    from apps.avatar.camera import CameraStream
    return JsonResponse(CameraStream.get_instance().get_latest_scores())


@csrf_exempt
@require_http_methods(["POST"])
def sim_body_control(request):
    """POST /simulator/body-control/ — {action: 'start'|'stop'}."""
    from apps.avatar.camera import CameraStream
    try:
        data   = json.loads(request.body)
        action = data.get('action', 'start')
        stream = CameraStream.get_instance()
        if action == 'start':
            stream.start()
            return JsonResponse({'status': 'started'})
        stream.stop()
        return JsonResponse({'status': 'stopped'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
