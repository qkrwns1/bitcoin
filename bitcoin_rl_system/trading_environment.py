"""Custom Gymnasium environment scaffold for Bitcoin RL."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

_PRICE_COLS: frozenset[str] = frozenset({"open", "high", "low", "close"})
_LOG_COLS: frozenset[str] = frozenset({"volume", "trade_value"})


@dataclass(slots=True)
class EnvironmentConfig:
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0005
    step_minutes: int = 1
    action_low: float = 0.0
    action_high: float = 1.0
    sequence_length: int = 360


class BitcoinTradingEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        market_frame: pd.DataFrame,
        sequence_features: list[str],
        context_features: list[str],
        portfolio_features: list[str],
        config: EnvironmentConfig,
    ) -> None:
        super().__init__()
        self.market_frame = market_frame.reset_index(drop=True).copy()
        self.sequence_features = sequence_features
        self.context_features = context_features
        self.portfolio_features = portfolio_features
        self.config = config
        self.sequence_length = config.sequence_length

        self._prepare_frame()

        self.sequence_dim = len(self.sequence_features)
        self.context_dim = len(self.context_features)
        self.portfolio_dim = len(self.portfolio_features)
        self.obs_dim = self.sequence_length * self.sequence_dim + self.context_dim + self.portfolio_dim

        # 정규화 마스크 사전계산 — _normalize_sequence 호출마다 재계산하지 않도록
        self._price_mask = np.array([col in _PRICE_COLS for col in sequence_features])
        self._log_mask = np.array([col in _LOG_COLS for col in sequence_features])

        self.action_space = spaces.Box(
            low=np.array([self.config.action_low], dtype=np.float32),
            high=np.array([self.config.action_high], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.current_step = self.sequence_length
        self.cash = self.config.initial_cash
        self.btc_holding = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0

    def _prepare_frame(self) -> None:
        market_map = {"ACTIVE": 1.0}
        warning_map = {"NONE": 0.0}
        ask_bid_map = {"ASK": -1.0, "BID": 1.0}

        self.market_frame["market_state"] = self.market_frame["market_state"].map(market_map).fillna(0.0)
        self.market_frame["market_warning"] = self.market_frame["market_warning"].map(warning_map).fillna(0.0)
        self.market_frame["ask_bid"] = self.market_frame["ask_bid"].map(ask_bid_map).fillna(0.0)
        self.market_frame["is_trading_suspended"] = self.market_frame["is_trading_suspended"].astype(float)

        for col in self.sequence_features + self.context_features:
            self.market_frame[col] = pd.to_numeric(self.market_frame[col], errors="coerce").fillna(0.0)

    def _current_close(self) -> float:
        return float(self.market_frame.loc[self.current_step, "close"])

    def _next_open(self) -> float:
        return float(self.market_frame.loc[self.current_step + 1, "open"])

    def _next_close(self) -> float:
        return float(self.market_frame.loc[self.current_step + 1, "close"])

    def _total_equity(self, mark_price: float) -> float:
        return float(self.cash + self.btc_holding * mark_price)

    def _normalize_sequence(self, seq: np.ndarray, current_close: float) -> np.ndarray:
        """가격 컬럼은 현재 close로 나누고, 거래량 컬럼은 log1p 변환."""
        if current_close <= 0:
            return seq
        seq[:, self._price_mask] /= current_close
        seq[:, self._log_mask] = np.log1p(np.maximum(seq[:, self._log_mask], 0.0))
        return seq

    def _portfolio_vector(self, mark_price: float) -> np.ndarray:
        """포트폴리오 상태를 initial_cash 기준 비율로 정규화."""
        ic = self.config.initial_cash
        total_equity = self._total_equity(mark_price)
        position_value = self.btc_holding * mark_price
        position_ratio = 0.0 if total_equity <= 0 else position_value / total_equity
        has_position = self.btc_holding > 0
        unrealized_pnl = self.btc_holding * (mark_price - self.avg_entry_price) if has_position else 0.0
        entry_price_ratio = (self.avg_entry_price / mark_price) if has_position else 0.0
        return np.array(
            [
                self.cash / ic,
                position_value / ic,
                entry_price_ratio,
                position_ratio,
                total_equity / ic,
                unrealized_pnl / ic,
                self.realized_pnl / ic,
            ],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        start = self.current_step - self.sequence_length + 1
        end = self.current_step + 1
        current_close = self._current_close()
        seq = self.market_frame.loc[start:end - 1, self.sequence_features].to_numpy(dtype=np.float32)
        seq = self._normalize_sequence(seq, current_close)
        ctx = self.market_frame.loc[self.current_step, self.context_features].to_numpy(dtype=np.float32)
        port = self._portfolio_vector(current_close)
        return np.concatenate([seq.reshape(-1), ctx, port]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.cash = self.config.initial_cash
        self.btc_holding = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        return self._observation(), {}

    def _rebalance_to_target(self, target_ratio: float, execution_price: float) -> None:
        total_equity = self._total_equity(execution_price)
        target_value = total_equity * target_ratio
        current_value = self.btc_holding * execution_price
        diff_value = target_value - current_value

        if abs(diff_value) < 1.0:
            return

        if diff_value > 0:
            affordable_value = min(diff_value, self.cash / (1.0 + self.config.fee_rate))
            if affordable_value <= 0:
                return
            fee = affordable_value * self.config.fee_rate
            btc_bought = affordable_value / execution_price
            total_cost = affordable_value + fee
            previous_cost_basis = self.avg_entry_price * self.btc_holding
            self.cash -= total_cost
            self.btc_holding += btc_bought
            self.avg_entry_price = (previous_cost_basis + affordable_value) / self.btc_holding
        else:
            sell_value = min(-diff_value, current_value)
            if sell_value <= 0:
                return
            btc_sold = sell_value / execution_price
            fee = sell_value * self.config.fee_rate
            proceeds = sell_value - fee
            realized = btc_sold * (execution_price - self.avg_entry_price) - fee
            self.cash += proceeds
            self.btc_holding -= btc_sold
            self.realized_pnl += realized
            if self.btc_holding <= 1e-12:
                self.btc_holding = 0.0
                self.avg_entry_price = 0.0

    def step(self, action: np.ndarray):
        target_ratio = float(np.clip(action[0], self.config.action_low, self.config.action_high))

        execution_price = self._next_open()
        # prev_equity: open[t+1]에서 리밸런스 직전 자산가치 — close[t]→open[t+1] 갭 노이즈 제거
        prev_equity = self._total_equity(execution_price)
        self._rebalance_to_target(target_ratio, execution_price)

        next_close = self._next_close()
        next_equity = self._total_equity(next_close)
        # 매 스텝 무조건 부과 — 연 -20% 기준 분당 차감률
        time_penalty = prev_equity * (0.20 / 525_600) * self.config.step_minutes
        reward = next_equity - prev_equity - time_penalty

        self.current_step += 1
        terminated = self.current_step >= len(self.market_frame) - 2
        truncated = False
        row = self.market_frame.loc[self.current_step]
        info = {
            "target_ratio": target_ratio,
            "prev_equity": prev_equity,
            "next_equity": next_equity,
            "time_penalty": time_penalty,
            "cash": self.cash,
            "btc_holding": self.btc_holding,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(next_close),
            "ts": str(row["ts"]),
        }
        return self._observation(), float(reward), terminated, truncated, info
