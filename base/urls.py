from django.urls import path
from . import views

urlpatterns = [
    path('', views.check, name='check'),
    # path('check/', views.check, name='check'),
    path('predict/', views.predict, name='predict'),
    # path('about/', views.about, name='about'),
]