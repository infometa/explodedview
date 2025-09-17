import asyncio
from typing import Awaitable


def schedule_async(awaitable: Awaitable) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(awaitable)
    else:
        loop.create_task(awaitable)
