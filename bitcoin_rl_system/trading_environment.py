"""Custom Gymnasium environment — discrete action trading (HOLD/BUY/SELL)."""

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
    sequence_length: int = 60
    # 매수 시 총 자본 대비 BTC 투자 비율
    btc_target_ratio: float = 0.9
    # 최소 보유 스텝 (1분봉 × 5 = 5분) — 손절 허용, 즉각 매매만 방지
    min_hold_steps: int = 5
    # 보유 중 미실현 손익 신호 가중치 (value fn 학습 보조)
    hold_signal_scale: float = 0.01
    # 포지션 없을 때 기회비용 가중치 (진입 유도)
    opportunity_scale: float = 0.001


class BitcoinTradingEnvironment(gym.Env):
    """
    Action space: Discrete(3)
        0 = HOLD
        1 = BUY  (btc_target_ratio 만큼 전량 매수)
        2 = SELL (전량 매도)

    Reward:
        SELL → realized_pnl / prev_equity          (주 신호)
        BUY  → -total_fee / prev_equity             (진입 비용)
        HOLD, 포지션 있음 → unrealized_pnl_pct * hold_signal_scale
        HOLD, 포지션 없음 → -max(price_return, 0) * opportunity_scale

    Constraint:
        min_hold_steps 미만이면 SELL 액션을 HOLD 로 강제 변환.
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

        # 정규화 마스크 사전계산
        self._price_mask = np.array([col in _PRICE_COLS for col in sequence_features])
        self._log_mask = np.array([col in _LOG_COLS for col in sequence_features])

        # Discrete(3): 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # 상태 초기화
        self.current_step: int = self.sequence_length
        self.cash: float = self.config.initial_cash
        self.btc_holding: float = 0.0
        self.avg_entry_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.in_position: bool = False
        self.steps_in_position: int = 0

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
        return float(self.market_frame.loc[self.current_step, "close"])

    def _next_open(self) -> float:
        return float(self.market_frame.loc[self.current_step + 1, "open"])

    def _next_close(self) -> float:
        return float(self.market_frame.loc[self.current_step + 1, "close"])

    def _total_equity(self, mark_price: float) -> float:
        return float(self.cash + self.btc_holding * mark_price)

    # ── 관측값 ────────────────────────────────────────────────

    def _normalize_sequence(self, seq: np.ndarray, current_close: float) -> np.ndarray:
        if current_close > 0:
            seq[:, self._price_mask] /= current_close
        seq[:, self._log_mask] = np.log1p(np.maximum(seq[:, self._log_mask], 0.0))
        return seq

    def _portfolio_vector(self, mark_price: float) -> np.ndarray:
        """7-dim 포트폴리오 벡터 (portfolio_features 길이와 일치)."""
        ic = self.config.initial_cash
        total_equity = self._total_equity(mark_price)
        position_value = self.btc_holding * mark_price

        unrealized_pnl_pct = 0.0
        if self.in_position and self.avg_entry_price > 0:
            unrealized_pnl_pct = (mark_price - self.avg_entry_price) / self.avg_entry_price

        steps_held_norm = min(self.steps_in_position / max(self.config.min_hold_steps, 1), 2.0)

        return np.array(
            [
                self.cash / ic,                           # 현금 비율
                position_value / ic,                      # BTC 포지션 비율
                float(self.in_position),                  # 포지션 보유 여부
                unrealized_pnl_pct,                       # 미실현 수익률
                steps_held_norm,                          # 보유 스텝 (정규화)
                total_equity / ic,                        # 총자산 비율
                self.realized_pnl / ic,                   # 실현손익 비율
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

    # ── 거래 실행 ─────────────────────────────────────────────

    def _execute_buy(self, execution_price: float) -> float:
        """btc_target_ratio 만큼 전량 매수. 반환값: 납부한 수수료."""
        if self.in_position or execution_price <= 0:
            return 0.0
        invest_cash = self.cash * self.config.btc_target_ratio
        fee = invest_cash * self.config.fee_rate
        net_invest = invest_cash - fee
        if net_invest <= 0:
            return 0.0
        btc_bought = net_invest / execution_price
        self.cash -= invest_cash
        self.btc_holding += btc_bought
        self.avg_entry_price = execution_price
        self.in_position = True
        self.steps_in_position = 0
        return fee

    def _execute_sell_all(self, execution_price: float) -> float:
        """전량 매도. 반환값: 실현 손익 (수수료 차감 후)."""
        if not self.in_position or self.btc_holding <= 0 or execution_price <= 0:
            return 0.0
        gross_proceeds = self.btc_holding * execution_price
        fee = gross_proceeds * self.config.fee_rate
        net_proceeds = gross_proceeds - fee
        realized = net_proceeds - (self.btc_holding * self.avg_entry_price)
        self.cash += net_proceeds
        self.realized_pnl += realized
        self.btc_holding = 0.0
        self.avg_entry_price = 0.0
        self.in_position = False
        self.steps_in_position = 0
        return realized

    # ── 환경 루프 ─────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.cash = self.config.initial_cash
        self.btc_holding = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.in_position = False
        self.steps_in_position = 0
        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        execution_price = self._next_open()
        prev_equity = self._total_equity(execution_price)

        # min_hold_steps 미만이면 SELL → HOLD 강제 변환
        if action == 2 and self.in_position and self.steps_in_position < self.config.min_hold_steps:
            action = 0

        reward = 0.0
        trade_executed = False
        realized_pnl_step = 0.0
        fee_paid = 0.0

        if action == 1:  # BUY
            fee_paid = self._execute_buy(execution_price)
            if fee_paid > 0:
                reward = -(fee_paid / prev_equity) if prev_equity > 0 else 0.0
                trade_executed = True

        elif action == 2:  # SELL
            realized_pnl_step = self._execute_sell_all(execution_price)
            reward = (realized_pnl_step / prev_equity) if prev_equity > 0 else 0.0
            trade_executed = True

        else:  # HOLD (action == 0)
            next_close_for_hold = self._next_close()
            price_return = (
                (next_close_for_hold - execution_price) / execution_price
                if execution_price > 0
                else 0.0
            )
            if self.in_position:
                # 이번 스텝에서 가격이 오르면 +, 내리면 -
                # → 에이전트가 "지금 들고 있는 게 이득인가?" 를 즉각 학습
                reward = price_return * self.config.hold_signal_scale
            else:
                # 포지션 없을 때 가격이 오르면 소폭 패널티 (진입 유도)
                reward = -max(price_return, 0.0) * self.config.opportunity_scale

        # 보유 스텝 카운터 증가
        if self.in_position:
            self.steps_in_position += 1

        next_close = self._next_close()
        next_equity = self._total_equity(next_close)
        position_value = self.btc_holding * next_close
        position_ratio = position_value / next_equity if next_equity > 0 else 0.0

        self.current_step += 1
        terminated = self.current_step >= len(self.market_frame) - 2
        truncated = False
        row = self.market_frame.loc[self.current_step]

        info = {
            "action": action,
            "in_position": self.in_position,
            "steps_in_position": self.steps_in_position,
            "position_ratio": position_ratio,
            "realized_pnl_step": realized_pnl_step,
            "fee_paid": fee_paid,
            "prev_equity": prev_equity,
            "next_equity": next_equity,
            "trade_executed": trade_executed,
            "cash": self.cash,
            "btc_holding": self.btc_holding,
            "avg_entry_price": self.avg_entry_price,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(next_close),
            "ts": str(row["ts"]),
        }
        return self._observation(), float(reward), terminated, truncated, info
