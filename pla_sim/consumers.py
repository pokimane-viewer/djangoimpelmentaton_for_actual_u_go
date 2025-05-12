
# pla_sim/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from graphql.execution.executors.asyncio import AsyncioExecutor
from strawberry.django.views import AsyncGraphQLView
from strawberry.subscriptions import SUBSCRIPTION_PROTOCOLS

class GraphQLSubscriptionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(json.dumps({"message": "GraphQL Subscriptions connected"}))

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            data = json.loads(text_data)
            if data.get("type") == "subscribe":
                # Example push of subscription data
                await self.send(json.dumps({
                    "type": "next",
                    "payload": {
                        "data": {"echo": "Subscription Data!"}
                    }
                }))
            elif data.get("type") == "stop":
                await self.send(json.dumps({"type": "complete"}))

    async def disconnect(self, close_code):
        pass