from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/stocks/$', consumers.StockConsumer.as_asgi()),
    re_path(r'ws/news/$', consumers.NewsConsumer.as_asgi()),
]