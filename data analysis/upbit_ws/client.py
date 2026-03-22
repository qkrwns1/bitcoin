"""WebSocket client abstraction."""

from .config import AppConfig


class UpbitWebSocketClient:
    """Placeholder client for connection lifecycle management."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def connect(self) -> None:
        raise NotImplementedError("Connection logic is not implemented yet.")

    def send_subscription(self) -> None:
        raise NotImplementedError("Subscription logic is not implemented yet.")

    def receive_forever(self) -> None:
        raise NotImplementedError("Receive loop is not implemented yet.")
