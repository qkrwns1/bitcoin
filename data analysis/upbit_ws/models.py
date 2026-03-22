"""Data models for normalized market events."""

from dataclasses import dataclass


@dataclass(slots=True)
class TickerEvent:
    """Normalized ticker event scaffold."""

    market: str
    trade_price: float
    timestamp: int
