from django.urls import path
from . import views

app_name = 'simulator'

urlpatterns = [
    path('',        views.simulator_index,  name='index'),
    path('start/',  views.start_simulation, name='start'),
    path('message/',views.send_message,     name='message'),
    path('report/', views.get_report,       name='report'),
    path('reset/',  views.reset_simulation, name='reset'),
]
