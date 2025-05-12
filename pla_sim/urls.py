
# pla_sim/urls.py
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path("graphql", views.GraphQLViewCustom.as_view(), name="graphql"),
    path("login", views.login_view, name="login-view"),
    path("", views.serve_react_app, name='react-app'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
