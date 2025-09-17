import asyncio
from typing import Set

from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        self.connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.connections:
                self.connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            targets = list(self.connections)
        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception:
                await self.disconnect(connection)
