# pla_sim/asgi.py
import os
import django
from channels.routing import get_default_application
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pla_sim.settings')
django.setup()

from channels.routing import ProtocolTypeRouter, URLRouter
import pla_sim.routing

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(pla_sim.routing.websocket_urlpatterns),
})
