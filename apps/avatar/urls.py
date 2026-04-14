from django.urls import path
from . import views

app_name = 'avatar'

urlpatterns = [
    path('',              views.avatar_index,  name='index'),
    path('analyze/',      views.analyze_api,   name='analyze'),
    path('history/',      views.history_api,   name='history'),
    path('stream/',       views.body_stream,   name='stream'),
    path('body-status/',  views.body_status,   name='body_status'),
    path('body-control/', views.body_control,  name='body_control'),
]
