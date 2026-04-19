from django.urls import path
from . import views

app_name = 'crm'

urlpatterns = [
    path('',              views.overview,     name='overview'),
    path('zones/',        views.zones,        name='zones'),
    path('delegates/',    views.delegates,    name='delegates'),
    path('pharmacies/',   views.pharmacies,   name='pharmacies'),
    path('predictions/',  views.predictions,  name='predictions'),
    path('overview/pdf/', views.overview_pdf, name='overview_pdf'),
]
