from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.prediction, name = 'prediction'),
    path('modelprediction', views.modelprediction, name = 'modelprediction'),

]