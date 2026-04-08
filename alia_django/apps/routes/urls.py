from django.urls import path
from . import views

app_name = 'routes'

urlpatterns = [
    path('',         views.routes_index, name='index'),
    path('optimize/',views.optimize_api, name='optimize'),
]
