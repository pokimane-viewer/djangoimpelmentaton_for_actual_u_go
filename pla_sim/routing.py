
# pla_sim/routing.py
from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/subscriptions/', consumers.GraphQLSubscriptionConsumer.as_asgi()),
]
