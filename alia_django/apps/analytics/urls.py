from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('',            views.analytics_index,  name='index'),
    path('data/',       views.analytics_data,   name='data'),
    path('action-plan/',views.action_plan_api,  name='action_plan'),
]
