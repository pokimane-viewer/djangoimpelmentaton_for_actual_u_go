# pla_sim/schema.py
import strawberry
from typing import AsyncGenerator

@strawberry.type
class SimulationData:
    status: str
    detail: str

@strawberry.type
class Query:
    hello: str = "Welcome to the PLA SSR GraphQL system."

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def watch_simulation(self, interval: float = 1.0) -> AsyncGenerator[SimulationData, None]:
        import asyncio
        import time
        start = time.time()
        while time.time() - start < 10:
            await asyncio.sleep(interval)
            yield SimulationData(
                status="running",
                detail=f"Time: {time.time() - start:.1f}s"
            )

schema = strawberry.Schema(Query, subscription=Subscription)