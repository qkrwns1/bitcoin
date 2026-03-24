"""Data loading and feature preparation scaffold.

Planned responsibilities:
- load processed 1-minute Bitcoin data
- define sequence/context feature groups
- split data by time into train/validation/test
- provide RL-ready market frames
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


DEFAULT_SEQUENCE_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_value",
    "return_5m",
    "log_return_5m",
    "return_15m",
    "return_1h",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_ratio",
    "volume_zscore_1h",
    "volume_zscore_2h",
    "volume_zscore_4h",
    "volatility_15m",
    "volatility_1h",
    "dist_to_15m_high",
    "dist_to_15m_low",
    "dist_to_1h_high",
    "dist_to_1h_low",
    "dist_to_4h_high",
    "dist_to_4h_low",
    "dist_to_1d_high",
    "dist_to_1d_low",
]

DEFAULT_CONTEXT_FEATURES = [
    "dist_to_daily_high",
    "dist_to_daily_low",
    "dist_to_52w_high",
    "dist_to_52w_low",
    "days_since_52w_high",
    "days_since_52w_low",
    "position_in_52w_range",
    "acc_trade_volume_24h",
    "acc_trade_price_24h",
    "change_rate",
    "signed_change_rate",
    "market_state",
    "market_warning",
    "is_trading_suspended",
    "ask_bid",
]

DEFAULT_PORTFOLIO_FEATURES = [
    "cash",
    "btc_holding",
    "avg_entry_price",
    "position_ratio",
    "total_equity",
    "unrealized_pnl",
    "realized_pnl",
]


@dataclass(slots=True)
class DataConfig:
    data_root: Path = Path(__file__).resolve().parent.parent / "data analysis" / "data"
    market: str = "KRW-BTC"
    step_minutes: int = 5
    sequence_length: int = 60   # 60봉 × 5분 = 5시간 컨텍스트
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    context_feature_names: list[str] = field(default_factory=lambda: DEFAULT_CONTEXT_FEATURES.copy())
    sequence_feature_names: list[str] = field(default_factory=lambda: DEFAULT_SEQUENCE_FEATURES.copy())
    portfolio_feature_names: list[str] = field(default_factory=lambda: DEFAULT_PORTFOLIO_FEATURES.copy())


class BitcoinDataHandler:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.frame_path = self.config.data_root / "processed" / "rl" / "rl_market_frame_5m.parquet"
        self.market_frame: pd.DataFrame | None = None

    def load_processed_frames(self) -> pd.DataFrame:
        if not self.frame_path.exists():
            raise FileNotFoundError(f"Processed frame not found: {self.frame_path}")

        frame = pd.read_parquet(self.frame_path)
        frame["ts"] = pd.to_datetime(frame["ts"])
        frame = frame.sort_values("ts").reset_index(drop=True)
        self.market_frame = frame
        return frame

    def build_feature_layout(self) -> dict[str, list[str]]:
        return {
            "sequence": self.config.sequence_feature_names,
            "context": self.config.context_feature_names,
            "portfolio": self.config.portfolio_feature_names,
        }

    def split_by_time(self) -> dict[str, pd.DataFrame]:
        if self.market_frame is None:
            self.load_processed_frames()

        assert self.market_frame is not None
        n_rows = len(self.market_frame)
        train_end = int(n_rows * self.config.train_ratio)
        val_end = int(n_rows * (self.config.train_ratio + self.config.val_ratio))

        return {
            "train": self.market_frame.iloc[:train_end].reset_index(drop=True),
            "val": self.market_frame.iloc[train_end:val_end].reset_index(drop=True),
            "test": self.market_frame.iloc[val_end:].reset_index(drop=True),
        }

    def summary(self) -> dict[str, object]:
        if self.market_frame is None:
            self.load_processed_frames()

        assert self.market_frame is not None
        splits = self.split_by_time()
        return {
            "frame_path": str(self.frame_path),
            "rows": len(self.market_frame),
            "columns": len(self.market_frame.columns),
            "ts_start": str(self.market_frame["ts"].min()),
            "ts_end": str(self.market_frame["ts"].max()),
            "train_rows": len(splits["train"]),
            "val_rows": len(splits["val"]),
            "test_rows": len(splits["test"]),
        }
