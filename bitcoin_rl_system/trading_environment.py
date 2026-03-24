"""Custom Gymnasium environment — 5-level discrete target ratio trading."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

_PRICE_COLS: frozenset[str] = frozenset({"open", "high", "low", "close"})
_LOG_COLS: frozenset[str] = frozenset({"volume", "trade_value"})

# 목표 비중 레벨: 0% / 25% / 50% / 75% / 100%
TARGET_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(slots=True)
class EnvironmentConfig:
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0005
    step_minutes: int = 1
    sequence_length: int = 60
    # 너무 소액 리밸런싱 방지 (수수료 낭비 차단)
    min_rebalance_value: float = 5_000.0


class BitcoinTradingEnvironment(gym.Env):
    """
    Action space: Discrete(5)
        0 = BTC 목표 비중  0%  (전량 현금)
        1 = BTC 목표 비중 25%
        2 = BTC 목표 비중 50%
        3 = BTC 목표 비중 75%
        4 = BTC 목표 비중 100% (전량 BTC)

    모든 액션이 항상 유효 — 현재 포지션에서 목표 비중으로 리밸런싱.
    동일 비중 유지 = HOLD, 수수료가 자연스럽게 잦은 매매를 억제.

    Reward: (next_equity - prev_equity) / prev_equity
        포지션 크기에 비례한 순수 자산 변화율.
    """

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
        self.obs_dim = (
            self.sequence_length * self.sequence_dim
            + self.context_dim
            + self.portfolio_dim
        )

        self._price_mask = np.array([col in _PRICE_COLS for col in sequence_features])
        self._log_mask = np.array([col in _LOG_COLS for col in sequence_features])

        # pandas 슬라이싱 대신 numpy 배열로 사전 캐싱 — 스텝 속도 10x 향상
        self._seq_arr  = self.market_frame[self.sequence_features].to_numpy(dtype=np.float32)
        self._ctx_arr  = self.market_frame[self.context_features].to_numpy(dtype=np.float32)
        self._close_arr = self.market_frame["close"].to_numpy(dtype=np.float32)

        # Discrete(5): 0~4 → 0%/25%/50%/75%/100%
        self.action_space = spaces.Discrete(len(TARGET_LEVELS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.current_step: int = self.sequence_length
        self.cash: float = self.config.initial_cash
        self.btc_holding: float = 0.0
        self.avg_entry_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.steps_since_rebalance: int = 0
        self.last_target_ratio: float = 0.0

    # ── 전처리 ────────────────────────────────────────────────

    def _prepare_frame(self) -> None:
        self.market_frame["market_state"] = (
            self.market_frame["market_state"].map({"ACTIVE": 1.0}).fillna(0.0)
        )
        self.market_frame["market_warning"] = (
            self.market_frame["market_warning"].map({"NONE": 0.0}).fillna(1.0)
        )
        self.market_frame["ask_bid"] = (
            self.market_frame["ask_bid"].map({"ASK": -1.0, "BID": 1.0}).fillna(0.0)
        )
        self.market_frame["is_trading_suspended"] = (
            self.market_frame["is_trading_suspended"].astype(float)
        )
        for col in self.sequence_features + self.context_features:
            self.market_frame[col] = (
                pd.to_numeric(self.market_frame[col], errors="coerce").fillna(0.0)
            )

    # ── 가격 헬퍼 ─────────────────────────────────────────────

    def _current_close(self) -> float:
        return float(self._close_arr[self.current_step])

    def _next_open(self) -> float:
        return float(self.market_frame.loc[self.current_step + 1, "open"])

    def _next_close(self) -> float:
        return float(self._close_arr[self.current_step + 1])

    def _total_equity(self, mark_price: float) -> float:
        return float(self.cash + self.btc_holding * mark_price)

    # ── 관측값 ────────────────────────────────────────────────

    def _normalize_sequence(self, seq: np.ndarray, current_close: float) -> np.ndarray:
        if current_close > 0:
            seq[:, self._price_mask] /= current_close
        seq[:, self._log_mask] = np.log1p(np.maximum(seq[:, self._log_mask], 0.0))
        return seq

    def _portfolio_vector(self, mark_price: float) -> np.ndarray:
        """7-dim 포트폴리오 벡터."""
        ic = self.config.initial_cash
        total_equity = self._total_equity(mark_price)
        btc_value = self.btc_holding * mark_price
        position_ratio = btc_value / total_equity if total_equity > 0 else 0.0

        unrealized_pnl_pct = 0.0
        if self.btc_holding > 0 and self.avg_entry_price > 0:
            unrealized_pnl_pct = (mark_price - self.avg_entry_price) / self.avg_entry_price

        return np.array(
            [
                self.cash / ic,                                        # 현금 비율
                btc_value / ic,                                        # BTC 평가액 비율
                position_ratio,                                        # 현재 BTC 비중
                unrealized_pnl_pct,                                    # 미실현 수익률
                min(self.steps_since_rebalance / 60.0, 2.0),          # 마지막 거래 후 경과 (정규화)
                total_equity / ic,                                     # 총자산 비율
                self.realized_pnl / ic,                                # 실현손익 비율
            ],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        start = self.current_step - self.sequence_length + 1
        end = self.current_step + 1
        current_close = self._current_close()
        seq = self._seq_arr[start:end].copy()
        seq = self._normalize_sequence(seq, current_close)
        ctx = self._ctx_arr[self.current_step]
        port = self._portfolio_vector(current_close)
        return np.concatenate([seq.reshape(-1), ctx, port]).astype(np.float32)

    # ── 리밸런싱 ──────────────────────────────────────────────

    def _rebalance_to_target(self, target_ratio: float, execution_price: float) -> bool:
        """현재 비중 → target_ratio 로 리밸런싱. 실제 거래 발생 시 True 반환."""
        total_equity = self._total_equity(execution_price)
        if total_equity <= 0 or execution_price <= 0:
            return False

        target_value = total_equity * target_ratio
        current_value = self.btc_holding * execution_price
        diff_value = target_value - current_value

        if abs(diff_value) < self.config.min_rebalance_value:
            return False  # 변화가 너무 작으면 거래 안 함

        if diff_value > 0:
            # 매수
            affordable = min(diff_value, self.cash / (1.0 + self.config.fee_rate))
            if affordable <= 0:
                return False
            fee = affordable * self.config.fee_rate
            btc_bought = affordable / execution_price
            prev_cost = self.avg_entry_price * self.btc_holding
            self.cash -= affordable + fee
            self.btc_holding += btc_bought
            self.avg_entry_price = (prev_cost + affordable) / self.btc_holding
            return True

        # 매도
        sell_value = min(-diff_value, current_value)
        if sell_value <= 0:
            return False
        btc_sold = sell_value / execution_price
        fee = sell_value * self.config.fee_rate
        proceeds = sell_value - fee
        realized = btc_sold * (execution_price - self.avg_entry_price) - fee
        self.cash += proceeds
        self.btc_holding = max(0.0, self.btc_holding - btc_sold)
        self.realized_pnl += realized
        if self.btc_holding < 1e-12:
            self.btc_holding = 0.0
            self.avg_entry_price = 0.0
        return True

    # ── 환경 루프 ─────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.cash = self.config.initial_cash
        self.btc_holding = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.steps_since_rebalance = 0
        self.last_target_ratio = 0.0
        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        target_ratio = TARGET_LEVELS[action]

        execution_price = self._next_open()
        prev_equity = self._total_equity(execution_price)

        trade_executed = self._rebalance_to_target(target_ratio, execution_price)

        if trade_executed:
            self.steps_since_rebalance = 0
            self.last_target_ratio = target_ratio
        else:
            self.steps_since_rebalance += 1

        next_close = self._next_close()
        next_equity = self._total_equity(next_close)

        # 순수 자산 변화율 — 포지션 크기에 자연스럽게 비례
        reward = (next_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

        btc_value = self.btc_holding * next_close
        position_ratio = btc_value / next_equity if next_equity > 0 else 0.0

        self.current_step += 1
        terminated = self.current_step >= len(self.market_frame) - 2
        truncated = False
        row = self.market_frame.loc[self.current_step]

        info = {
            "action": action,
            "target_ratio": target_ratio,
            "position_ratio": position_ratio,
            "trade_executed": trade_executed,
            "prev_equity": prev_equity,
            "next_equity": next_equity,
            "cash": self.cash,
            "btc_holding": self.btc_holding,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
            "steps_since_rebalance": self.steps_since_rebalance,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(next_close),
            "ts": str(row["ts"]),
        }
        return self._observation(), float(reward), terminated, truncated, info
