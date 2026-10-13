from __future__ import annotations

from typing import Any

import websocket


class WebSocketClientTransport:
    """Production transport using websocket-client.

    Authentication/subscription payloads remain provider-specific and are supplied
    by the market-data adapter; this class only owns socket I/O and timeouts.
    """

    def __init__(self, url: str, *, headers: list[str] | None = None, timeout: float = 10.0):
        if not url:
            raise ValueError("url is required")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self.url = url
        self.headers = headers or []
        self.timeout = timeout
        self._socket: Any | None = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        self._socket = websocket.create_connection(self.url, timeout=self.timeout, header=self.headers)

    def send(self, message: str | bytes) -> None:
        self.connect()
        assert self._socket is not None
        if isinstance(message, bytes):
            self._socket.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
        else:
            self._socket.send(message)

    def recv(self) -> str | bytes:
        self.connect()
        assert self._socket is not None
        return self._socket.recv()

    def close(self) -> None:
        sock, self._socket = self._socket, None
        if sock is not None:
            sock.close()
