"""Subscription request models and builders."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class SubscriptionRequest:
    """Represents one logical subscription setup."""

    ticket: str = "analysis-session"
    type: str = "ticker"
    codes: list[str] = field(default_factory=lambda: ["KRW-BTC"])
    format: str = "DEFAULT"


def build_default_subscription() -> SubscriptionRequest:
    """Return a minimal default subscription scaffold."""
    return SubscriptionRequest()
