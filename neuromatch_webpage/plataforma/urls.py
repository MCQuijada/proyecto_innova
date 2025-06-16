from django.urls import path
from .views import plataforma_view

urlpatterns = [
    path('plataforma/', plataforma_view, name='plataforma'),
]
