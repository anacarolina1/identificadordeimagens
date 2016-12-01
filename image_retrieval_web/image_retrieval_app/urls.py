from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^$', views.image_app, name='image_app'),
    url(r'^$', views.list, name='image_list'),
]
