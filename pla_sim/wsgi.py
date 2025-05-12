
# -*- coding: utf-8 -*- # pla_sim/views.py# pla_sim/wsgi.py
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pla_sim.settings')
application = get_wsgi_application()