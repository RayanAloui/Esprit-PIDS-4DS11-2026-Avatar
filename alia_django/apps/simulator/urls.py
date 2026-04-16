from django.urls import path
from . import views

app_name = 'simulator'

urlpatterns = [
    path('',              views.simulator_index,   name='index'),
    path('start/',        views.start_simulation,  name='start'),
    path('message/',      views.send_message,       name='message'),
    path('report/',       views.get_report,         name='report'),
    path('reset/',        views.reset_simulation,   name='reset'),
    path('dashboard/',    views.dashboard_data,     name='dashboard'),
    path('stream/',       views.sim_body_stream,    name='stream'),
    path('body-status/',  views.sim_body_status,    name='body_status'),
    path('body-control/', views.sim_body_control,   name='body_control'),
    # QCM pré-formation (Bloc 1)
    path('qcm/',          views.submit_qcm,         name='qcm'),
    path('qcm/questions/',views.get_qcm_questions,   name='qcm_questions'),
    # Replay
    path('replay/',       views.replay_data,        name='replay_data'),
    path('replay/view/',  views.replay_page,        name='replay_page'),
    # STT / TTS — Speech
    path('stt/',          views.sim_stt,            name='stt'),
    path('tts/',          views.sim_tts,            name='tts'),
]
