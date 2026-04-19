"""
Simulator Views — Intégrées avec SessionState.
v2 — Bloc 1 : pharmaciens, QCM, suggestions, mode généraliste
"""
import json
import time as _time

from django.shortcuts   import render
from django.http        import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from .profiles import (
    DOCTOR_PROFILES, PHARMACIST_PROFILES, VITAL_PRODUCTS,
    QCM_QUESTIONS, QCM_SEUIL_PASSAGE, get_generic_product,
    get_random_qcm, QCM_NB_QUESTIONS,
)
from .engine   import SimulationSession

SESSION_KEY = 'alia_sim_session'


# ══════════════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════

@login_required
def simulator_index(request):
    # Produits sans le produit généraliste (affiché séparément)
    products_display = [p for p in VITAL_PRODUCTS if not p.get('_is_generic')]
    context = {
        'page'        : 'simulator',
        'doctors'     : DOCTOR_PROFILES,
        'pharmacists' : PHARMACIST_PROFILES,
        'products'    : products_display,
        'niveaux'     : ['Débutant', 'Junior', 'Confirmé', 'Expert'],
        'qcm_seuil'   : QCM_SEUIL_PASSAGE,
        'qcm_nb'      : QCM_NB_QUESTIONS,
    }
    return render(request, 'simulator/index.html', context)


# ══════════════════════════════════════════════════════════════════════
# QCM — Validation du niveau pré-formation
# ══════════════════════════════════════════════════════════════════════

@require_http_methods(["GET"])
def get_qcm_questions(request):
    """
    GET /simulator/qcm/questions/
    Retourne un jeu de questions aléatoires pour le QCM.
    Chaque rechargement donne des questions différentes.
    """
    questions = get_random_qcm()
    return JsonResponse({
        'ok': True,
        'questions': questions,
        'total': len(questions),
        'seuil': QCM_SEUIL_PASSAGE,
    })


@csrf_exempt
@require_http_methods(["POST"])
def submit_qcm(request):
    """
    POST /simulator/qcm/
    Body : { "answers": { "q1": 1, "q2": 0, ... } }
    Retourne : { ok, score_pct, passed, details, niveau_recommande }
    """
    try:
        data    = json.loads(request.body)
        answers = data.get('answers', {})
        qids    = data.get('question_ids', [])  # IDs des questions présentées

        if not answers:
            return JsonResponse({'ok': False, 'error': 'Aucune réponse soumise.'}, status=400)

        # Construire la map de questions par id
        q_map = {q['id']: q for q in QCM_QUESTIONS}

        # Utiliser les question_ids envoyés par le frontend, sinon les clés de answers
        presented_ids = qids if qids else list(answers.keys())
        presented_questions = [q_map[qid] for qid in presented_ids if qid in q_map]

        if not presented_questions:
            return JsonResponse({'ok': False, 'error': 'Questions invalides.'}, status=400)

        total   = len(presented_questions)
        correct = 0
        details = []

        for q in presented_questions:
            qid         = q['id']
            user_answer = answers.get(qid)
            is_correct  = (user_answer == q['correct'])
            if is_correct:
                correct += 1
            details.append({
                'id'          : qid,
                'question'    : q['question'],
                'user_answer' : user_answer,
                'correct'     : q['correct'],
                'is_correct'  : is_correct,
                'explication' : q['explication'],
            })

        score_pct = round(correct / total * 100)
        passed    = score_pct >= QCM_SEUIL_PASSAGE

        # Niveau recommandé selon le score
        if score_pct >= 85:
            niveau_recommande = 'Expert'
        elif score_pct >= 70:
            niveau_recommande = 'Confirmé'
        elif score_pct >= QCM_SEUIL_PASSAGE:
            niveau_recommande = 'Junior'
        else:
            niveau_recommande = 'Débutant'

        # Sauvegarder le résultat QCM en session
        request.session['qcm_result'] = {
            'score_pct'        : score_pct,
            'passed'           : passed,
            'niveau_recommande': niveau_recommande,
            'correct'          : correct,
            'total'            : total,
        }
        request.session.modified = True

        return JsonResponse({
            'ok'               : True,
            'score_pct'        : score_pct,
            'correct'          : correct,
            'total'            : total,
            'passed'           : passed,
            'niveau_recommande': niveau_recommande,
            'details'          : details,
            'seuil'            : QCM_SEUIL_PASSAGE,
        })

    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


# ══════════════════════════════════════════════════════════════════════
# SIMULATION — START / MESSAGE / REPORT / RESET
# ══════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def start_simulation(request):
    try:
        data             = json.loads(request.body)
        interlocutor_id  = data.get('doctor_id') or data.get('interlocutor_id', 'sceptique')
        product_id       = data.get('product_id', 'ferrimax')
        niveau_alia      = data.get('niveau_alia', 'Junior')
        mode_generaliste = data.get('mode_generaliste', False)

        # Mode généraliste : utiliser le produit fictif générique
        if mode_generaliste:
            product_id = 'generique'

        session = SimulationSession(interlocutor_id, product_id, niveau_alia)
        first   = session.first_message()

        request.session[SESSION_KEY] = session.to_dict()
        request.session.modified     = True

        return JsonResponse({
            'ok'              : True,
            'first'           : first,
            'doctor'          : session.interlocutor,   # compat frontend
            'interlocutor'    : session.interlocutor,
            'interlocutor_type': session._label,
            'product'         : session.product,
        })
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
        lang    = data.get('lang', None)
        
        # Heuristique de détection de langue si le texte est tapé manuellement
        # (car côté JS, detectedLang reste souvent "fr" par défaut sans STT)
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

        user_id = request.user.id if hasattr(request, 'user') and request.user.is_authenticated else None
        result  = session.process_delegate_response(text, lang=lang, user_id=user_id)

        request.session[SESSION_KEY] = session.to_dict()
        request.session.modified     = True

        result['ok'] = True
        result['lang'] = lang
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
    from apps.avatar.camera import CameraStream
    stream = CameraStream.get_instance()
    scores = stream.get_latest_scores()

    if not stream.is_running and not scores.get('error'):
        scores['error'] = 'Stream webcam arrêté.'

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
            pass

    return JsonResponse(scores)


@csrf_exempt
@require_http_methods(["POST"])
def sim_body_control(request):
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
# REPLAY
# ══════════════════════════════════════════════════════════════════════

@require_http_methods(["GET"])
def replay_data(request):
    session_data = request.session.get(SESSION_KEY)
    if not session_data:
        return JsonResponse({'ok': False, 'error': 'Aucune session à rejouer.'}, status=400)
    try:
        session = SimulationSession.from_dict(session_data)
        state   = session.state
        history     = session.history
        nlp_turns   = state.nlp_turns
        rag_hits    = state.rag_hits
        openness_tl = state.openness_timeline
        events_all  = state.events

        rag_by_turn = {h['turn']: h for h in rag_hits}
        nlp_by_turn = {t['turn']: t for t in nlp_turns}
        turns_data  = []

        if history:
            turns_data.append({
                'turn': 0, 'type': 'opening',
                'doctor_msg': history[0]['content'], 'delegate_msg': None,
                'score': None, 'quality': None, 'acrv': None, 'conformite': None,
                'openness': session.interlocutor.get('ouverture_initiale', 2),
                'delta_open': 0, 'step': None,
                'engine': rag_by_turn.get(0, {}).get('engine', '—'),
                'rag_used': rag_by_turn.get(0, {}).get('context_used', False),
            })

        for turn_num in range(1, session.turn + 1):
            del_idx = 2 * turn_num - 1
            doc_idx = 2 * turn_num
            delegate_msg = history[del_idx]['content'] if del_idx < len(history) else ''
            doctor_msg   = history[doc_idx]['content'] if doc_idx < len(history) else ''
            nlp  = nlp_by_turn.get(turn_num, {})
            rag  = rag_by_turn.get(turn_num, {})
            open_val  = openness_tl[turn_num - 1] if turn_num - 1 < len(openness_tl) else None
            open_prev = openness_tl[turn_num - 2] if turn_num >= 2 and turn_num - 2 < len(openness_tl) else None
            delta = round(open_val - open_prev, 2) if open_val is not None and open_prev is not None else 0
            step  = session.step_history[turn_num - 1] if turn_num - 1 < len(session.step_history) else None

            turns_data.append({
                'turn': turn_num, 'type': 'turn',
                'doctor_msg': doctor_msg, 'delegate_msg': delegate_msg,
                'score': nlp.get('score'), 'quality': nlp.get('quality'),
                'acrv': nlp.get('acrv'), 'conformite': nlp.get('conformite'),
                'openness': open_val, 'delta_open': delta, 'step': step,
                'engine': rag.get('engine', '—'),
                'rag_used': rag.get('context_used', False),
            })

        scores_only = [t['score'] for t in turns_data if t['score'] is not None]
        best_turn   = max(turns_data, key=lambda t: t['score'] or 0, default=None)
        worst_turn  = min(turns_data, key=lambda t: t['score'] or 10, default=None)
        tueur_turns = [t['turn'] for t in turns_data if t.get('conformite') is False]

        return JsonResponse({
            'ok': True, 'turns': turns_data,
            'total_turns': session.turn,
            'doctor': session.interlocutor,
            'interlocutor': session.interlocutor,
            'interlocutor_type': session._label,
            'product': session.product,
            'global_score': state.global_score,
            'global_niveau': state.global_niveau,
            'openness_final': round(session.openness, 1),
            'openness_timeline': openness_tl,
            'events': events_all,
            'highlights': {
                'best_turn': best_turn['turn'] if best_turn else None,
                'worst_turn': worst_turn['turn'] if worst_turn else None,
                'tueur_turns': tueur_turns,
                'max_score': max(scores_only) if scores_only else 0,
                'min_score': min(scores_only) if scores_only else 0,
            },
        })
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


def replay_page(request):
    return render(request, 'simulator/replay.html', {'page': 'simulator'})


# ══════════════════════════════════════════════════════════════════════
# STT / TTS — Speech (partagé avec modeling)
# ══════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def sim_stt(request):
    import asyncio
    from apps.modeling.handlers import listen_json
    audio_file = request.FILES.get("audio")
    if not audio_file:
        return JsonResponse({"error": "Fichier audio manquant."}, status=400)
    try:
        data = asyncio.run(listen_json(audio_file.read()))
        # data = {"text": "...", "detected_lang": "fr"|"en"|"es"|"ar"}
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def sim_tts(request):
    import asyncio, uuid
    from pathlib import Path
    from django.conf import settings
    from apps.modeling.handlers import _synthesize_to_file, clean_for_tts, api_prefix
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "JSON invalide."}, status=400)
    text = body.get("text", "").strip()
    lang = body.get("lang", "fr")
    if not text:
        return JsonResponse({"error": "Texte vide."}, status=400)
    try:
        filename = f"sim_{uuid.uuid4()}.mp3"
        path     = Path(settings.MODELING_AUDIO_DIR) / filename
        asyncio.run(_synthesize_to_file(clean_for_tts(text), path, lang=lang))
        return JsonResponse({"audio_url": f"{api_prefix()}/static/audio/{filename}"})
    except Exception as e:
        return JsonResponse({"error": str(e), "audio_url": None}, status=500)
