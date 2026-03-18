"""WebSocket manager for real-time updates."""

import json
import asyncio
from typing import Dict, Any, List, Set
from dataclasses import dataclass, field

from fastapi import WebSocket

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Maximum allowed message size: 1 MB
MAX_MESSAGE_SIZE = 1_048_576


class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize manager."""
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def send_personal(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to a specific client.

        Args:
            message: Message to send
            websocket: Target WebSocket
        """
        try:
            data = json.dumps(message)
            if len(data) > MAX_MESSAGE_SIZE:
                logger.warning("Skipping oversized personal message (%d bytes)", len(data))
                return
            await websocket.send_text(data)
        except Exception:
            await self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients.

        Applies backpressure: if a client send fails, it is disconnected.
        Messages larger than 1 MB are silently skipped.

        Args:
            message: Message to broadcast
        """
        data = json.dumps(message)
        if len(data) > MAX_MESSAGE_SIZE:
            logger.warning("Skipping oversized broadcast message (%d bytes)", len(data))
            return

        disconnected: List[WebSocket] = []

        async with self._lock:
            for connection in self.active_connections:
                try:
                    await asyncio.wait_for(connection.send_text(data), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    # Client is too slow or disconnected – mark for removal
                    disconnected.append(connection)

            # Clean up slow / disconnected clients
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
                    logger.info("Disconnected slow/broken WebSocket client")

    async def broadcast_text(self, text: str):
        """Broadcast raw text to all clients.

        Args:
            text: Text to broadcast
        """
        if len(text) > MAX_MESSAGE_SIZE:
            logger.warning("Skipping oversized broadcast_text (%d bytes)", len(text))
            return

        disconnected: List[WebSocket] = []

        async with self._lock:
            for connection in self.active_connections:
                try:
                    await asyncio.wait_for(connection.send_text(text), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    disconnected.append(connection)

            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
                    logger.info("Disconnected slow/broken WebSocket client")

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


class StreamingBuffer:
    """Buffer for streaming model generation tokens."""

    def __init__(self, ws_manager: WebSocketManager, pr_id: str, turn: int):
        """Initialize buffer.

        Args:
            ws_manager: WebSocket manager for broadcasting
            pr_id: Current PR ID
            turn: Current turn number
        """
        self.ws_manager = ws_manager
        self.pr_id = pr_id
        self.turn = turn
        self.tokens: List[str] = []
        self.full_text = ""

    async def add_token(self, token: str):
        """Add a token and broadcast update.

        Args:
            token: Generated token
        """
        self.tokens.append(token)
        self.full_text += token

        await self.ws_manager.broadcast({
            'type': 'generation_token',
            'pr_id': self.pr_id,
            'turn': self.turn,
            'token': token,
            'full_text': self.full_text
        })

    async def finish(self):
        """Signal generation complete."""
        await self.ws_manager.broadcast({
            'type': 'generation_complete',
            'pr_id': self.pr_id,
            'turn': self.turn,
            'full_text': self.full_text
        })
