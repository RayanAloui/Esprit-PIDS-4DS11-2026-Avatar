"""
Simulator Views — Intégrées avec SessionState.
"""
import json
import time as _time

from django.shortcuts   import render
from django.http        import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .profiles import DOCTOR_PROFILES, VITAL_PRODUCTS
from .engine   import SimulationSession

SESSION_KEY = 'alia_sim_session'


# ══════════════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════

def simulator_index(request):
    context = {
        'page'    : 'simulator',
        'doctors' : DOCTOR_PROFILES,
        'products': VITAL_PRODUCTS,
        'niveaux' : ['Débutant', 'Junior', 'Confirmé', 'Expert'],
    }
    return render(request, 'simulator/index.html', context)


# ══════════════════════════════════════════════════════════════════════
# SIMULATION — START / MESSAGE / REPORT / RESET
# ══════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def start_simulation(request):
    try:
        data       = json.loads(request.body)
        session    = SimulationSession(
            data.get('doctor_id',   'sceptique'),
            data.get('product_id',  'ferrimax'),
            data.get('niveau_alia', 'Junior'),
        )
        first = session.first_message()
        request.session[SESSION_KEY] = session.to_dict()
        request.session.modified     = True
        return JsonResponse({'ok': True, 'first': first,
                             'doctor': session.doctor,
                             'product': session.product})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        if not text:
            return JsonResponse({'ok': False, 'error': 'Message vide.'}, status=400)

        session_data = request.session.get(SESSION_KEY)
        if not session_data:
            return JsonResponse({'ok': False, 'error': 'Session expirée.'}, status=400)

        session = SimulationSession.from_dict(session_data)
        result  = session.process_delegate_response(text)

        request.session[SESSION_KEY] = session.to_dict()
        request.session.modified     = True

        result['ok'] = True
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_report(request):
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
    if SESSION_KEY in request.session:
        del request.session[SESSION_KEY]
    return JsonResponse({'ok': True})


# ══════════════════════════════════════════════════════════════════════
# DASHBOARD — données temps réel (polling 500ms)
# ══════════════════════════════════════════════════════════════════════

@require_http_methods(["GET"])
def dashboard_data(request):
    """
    GET /simulator/dashboard/
    Retourne toutes les données du SessionState pour le dashboard unifié.
    Appelé toutes les 500ms par le frontend pendant la simulation.
    """
    session_data = request.session.get(SESSION_KEY)
    if not session_data:
        return JsonResponse({'ok': False, 'active': False})

    try:
        session = SimulationSession.from_dict(session_data)
        data    = session.state.dashboard_data()
        data['ok']     = True
        data['active'] = True
        data['turn']   = session.turn
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


# ══════════════════════════════════════════════════════════════════════
# BODY LANGUAGE — LSTM push vers SessionState
# ══════════════════════════════════════════════════════════════════════

def sim_body_stream(request):
    """GET /simulator/stream/ — flux MJPEG webcam."""
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
        generate(), content_type='multipart/x-mixed-replace; boundary=frame'
    )


@require_http_methods(["GET"])
def sim_body_status(request):
    """
    GET /simulator/body-status/
    Retourne les scores LSTM ET les pousse dans SessionState.
    C'est le point d'intégration LSTM → SessionState.
    Expose également les erreurs caméra pour le heartbeat frontend.
    """
    from apps.avatar.camera import CameraStream
    stream = CameraStream.get_instance()
    scores = stream.get_latest_scores()

    # ── Si le stream s'est arrêté, indiquer l'erreur au frontend ────
    if not stream.is_running and not scores.get('error'):
        scores['error'] = 'Stream webcam arrêté.'

    # ── Push LSTM → SessionState si session active ──────────────────
    session_data = request.session.get(SESSION_KEY)
    if session_data and scores.get('active'):
        try:
            session = SimulationSession.from_dict(session_data)
            session.state.push_lstm_frame(
                posture     = scores.get('posture', 'neutral'),
                confidence  = scores.get('confidence', 0.0),
                stress      = scores.get('stress', 0.0),
                arms_crossed= scores.get('arms_crossed', False),
                face_touch  = scores.get('face_touch', False),
            )
            request.session[SESSION_KEY] = session.to_dict()
            request.session.modified     = True
        except Exception:
            pass  # Ne jamais bloquer le stream LSTM

    return JsonResponse(scores)


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


# ══════════════════════════════════════════════════════════════════════
# REPLAY — données complètes pour la timeline interactive
# ══════════════════════════════════════════════════════════════════════

@require_http_methods(["GET"])
def replay_data(request):
    """
    GET /simulator/replay/
    Retourne toutes les données nécessaires pour reconstruire
    la timeline interactive de la visite tour par tour.
    """
    session_data = request.session.get(SESSION_KEY)
    if not session_data:
        return JsonResponse({'ok': False, 'error': 'Aucune session à rejouer.'}, status=400)

    try:
        session = SimulationSession.from_dict(session_data)
        state   = session.state

        # ── Reconstruire la timeline tour par tour ─────────────────
        # history = alternance assistant / user / assistant / user ...
        # On aligne : history[0] = message d'ouverture médecin (turn 0)
        # puis par paires : history[2i-1] = délégué turn i
        #                   history[2i]   = médecin turn i

        history      = session.history         # [{"role","content"}]
        nlp_turns    = state.nlp_turns         # [{turn,score,quality,acrv,conformite}]
        rag_hits     = state.rag_hits           # [{turn,product,context_used,engine}]
        openness_tl  = state.openness_timeline  # [float] par tour
        events_all   = state.events             # [{msg,type,ts}]

        # Construire un index RAG par tour
        rag_by_turn = {}
        for h in rag_hits:
            rag_by_turn[h['turn']] = h

        # Construire un index NLP par tour
        nlp_by_turn = {t['turn']: t for t in nlp_turns}

        # Reconstruire les moments de la timeline
        turns_data = []

        # Tour 0 — accueil médecin (pas d'évaluation NLP)
        if history:
            turns_data.append({
                'turn'       : 0,
                'type'       : 'opening',
                'doctor_msg' : history[0]['content'],
                'delegate_msg': None,
                'score'      : None,
                'quality'    : None,
                'acrv'       : None,
                'conformite' : None,
                'openness'   : session.doctor.get('ouverture_initiale', 2),
                'delta_open' : 0,
                'step'       : None,
                'engine'     : rag_by_turn.get(0, {}).get('engine', '—'),
                'rag_used'   : rag_by_turn.get(0, {}).get('context_used', False),
            })

        # Tours 1..N
        for turn_num in range(1, session.turn + 1):
            # Index dans history : délégué à 2*turn_num - 1, médecin à 2*turn_num
            del_idx = 2 * turn_num - 1
            doc_idx = 2 * turn_num

            delegate_msg = history[del_idx]['content'] if del_idx < len(history) else ''
            doctor_msg   = history[doc_idx]['content'] if doc_idx < len(history) else ''

            nlp  = nlp_by_turn.get(turn_num, {})
            rag  = rag_by_turn.get(turn_num, {})
            open_val = openness_tl[turn_num - 1] if turn_num - 1 < len(openness_tl) else None
            open_prev= openness_tl[turn_num - 2] if turn_num >= 2 and turn_num - 2 < len(openness_tl) else None
            delta    = round(open_val - open_prev, 2) if open_val is not None and open_prev is not None else 0

            # Étape VM détectée à ce tour
            step = session.step_history[turn_num - 1] if turn_num - 1 < len(session.step_history) else None

            turns_data.append({
                'turn'        : turn_num,
                'type'        : 'turn',
                'doctor_msg'  : doctor_msg,
                'delegate_msg': delegate_msg,
                'score'       : nlp.get('score'),
                'quality'     : nlp.get('quality'),
                'acrv'        : nlp.get('acrv'),
                'conformite'  : nlp.get('conformite'),
                'openness'    : open_val,
                'delta_open'  : delta,
                'step'        : step,
                'engine'      : rag.get('engine', '—'),
                'rag_used'    : rag.get('context_used', False),
            })

        # Moments clés : score le plus haut, le plus bas, mot tueur
        scores_only = [t['score'] for t in turns_data if t['score'] is not None]
        best_turn   = max(turns_data, key=lambda t: t['score'] or 0, default=None)
        worst_turn  = min(turns_data, key=lambda t: t['score'] or 10, default=None)
        tueur_turns = [t['turn'] for t in turns_data if t.get('conformite') is False]

        return JsonResponse({
            'ok'          : True,
            'turns'       : turns_data,
            'total_turns' : session.turn,
            'doctor'      : session.doctor,
            'product'     : session.product,
            'global_score': state.global_score,
            'global_niveau': state.global_niveau,
            'openness_final': round(session.openness, 1),
            'openness_timeline': openness_tl,
            'events'      : events_all,
            'highlights'  : {
                'best_turn'  : best_turn['turn'] if best_turn else None,
                'worst_turn' : worst_turn['turn'] if worst_turn else None,
                'tueur_turns': tueur_turns,
                'max_score'  : max(scores_only) if scores_only else 0,
                'min_score'  : min(scores_only) if scores_only else 0,
            },
        })

    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


def replay_page(request):
    """GET /simulator/replay/view — page HTML du replay."""
    return render(request, 'simulator/replay.html', {'page': 'simulator'})
