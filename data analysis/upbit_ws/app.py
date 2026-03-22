"""Application entrypoint and wiring."""

from .config import AppConfig
from .subscription import build_default_subscription


def run() -> None:
    """Boot the app with placeholder wiring only."""
    config = AppConfig()
    subscription = build_default_subscription()

    print("Upbit WebSocket scaffold")
    print(f"Endpoint: {config.endpoint}")
    print(f"Markets: {subscription.codes}")
    print("Implementation is intentionally pending.")
