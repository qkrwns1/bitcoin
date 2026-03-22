"""Configuration models for the WebSocket app."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration scaffold."""

    endpoint: str = "wss://api.upbit.com/websocket/v1"
    reconnect_seconds: int = 5
    ping_interval_seconds: int = 30
    output_dir: str = "output"
    default_codes: list[str] = field(default_factory=lambda: ["KRW-BTC"])
