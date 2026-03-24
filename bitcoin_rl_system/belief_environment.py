"""Phase 2: Belief-Augmented Environment.

BitcoinTradingEnvironment를 래핑하여 FAISS 유사 패턴 검색 결과를
belief state로 observation에 추가한다.

추가되는 3개 feature (portfolio_dim 7 → 10):
  belief_expected_return : 유사 패턴 이후 평균 수익률 (1h 기준, 스케일 ×10)
  belief_confidence      : 유사도 평균 (cosine, 0~1)
  belief_divergence      : (실제 진입 후 수익률) - (진입 시 예상 수익률)
"""

from __future__ import annotations

from pathlib import Path

import faiss
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

_FAISS_DIR = Path(__file__).parent / "faiss"


class BeliefAugmentedEnv(gym.Wrapper):
    """
    래핑 대상: BitcoinTradingEnvironment
    FAISS 검색으로 belief state 3개를 obs 끝에 추가.
    obs_dim: 1642 → 1645
    portfolio_dim: 7 → 10
    """

    N_BELIEF = 3
    TOP_K    = 20
    HORIZON  = "return_1h"   # return_1h / return_5h / return_1d

    def __init__(
        self,
        env: gym.Env,
        embeddings: np.ndarray,    # (N, D) float32
        metadata: pd.DataFrame,    # bar_idx, return_1h, return_5h, return_1d
        index: faiss.Index,
        seq_len: int,
    ) -> None:
        super().__init__(env)

        self.embeddings = embeddings.astype(np.float32)
        self.metadata   = metadata.reset_index(drop=True)
        self.index      = index
        self.seq_len    = seq_len

        # obs space 확장
        old_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(old_dim + self.N_BELIEF,),
            dtype=np.float32,
        )
        # portfolio_dim 갱신 (rl_agent.py build_model 에서 참조)
        self.portfolio_dim = env.portfolio_dim + self.N_BELIEF
        self.sequence_dim  = env.sequence_dim
        self.context_dim   = env.context_dim
        self.sequence_length = env.sequence_length

        # 진입 시점 예상 수익률 추적
        self._entry_expected_return: float = 0.0
        self._belief: np.ndarray = np.zeros(self.N_BELIEF, dtype=np.float32)

    # ── FAISS 검색 ──────────────────────────────────────────────────

    def _query_belief(self, current_step: int) -> np.ndarray:
        embed_idx = current_step - self.seq_len
        if embed_idx < 0 or embed_idx >= len(self.embeddings):
            return np.zeros(self.N_BELIEF, dtype=np.float32)

        query = self.embeddings[embed_idx : embed_idx + 1].copy()
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, self.TOP_K + 1)
        scores  = scores[0]
        indices = indices[0]

        # 자기 자신 제외
        mask    = indices != embed_idx
        scores  = scores[mask][: self.TOP_K]
        indices = indices[mask][: self.TOP_K]
        valid   = indices[indices >= 0]

        if len(valid) == 0:
            return np.zeros(self.N_BELIEF, dtype=np.float32)

        future_rets = self.metadata[self.HORIZON].iloc[valid].values
        expected    = float(np.mean(future_rets))
        confidence  = float(np.mean(scores[: len(valid)]))

        # divergence: 실제 보유 수익률 - 진입 시 예상 수익률
        divergence = 0.0
        inner = self.unwrapped
        if hasattr(inner, "btc_holding") and inner.btc_holding > 0:
            cp = float(inner._close_arr[inner.current_step])
            if inner.avg_entry_price > 0:
                actual = (cp - inner.avg_entry_price) / inner.avg_entry_price
                divergence = actual - self._entry_expected_return

        return np.array([
            np.clip(expected  * 10, -3.0, 3.0),
            np.clip(confidence,      0.0, 1.0),
            np.clip(divergence * 10, -3.0, 3.0),
        ], dtype=np.float32)

    # ── Gym API ─────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._entry_expected_return = 0.0
        self._belief = self._query_belief(self.unwrapped.current_step)
        return np.concatenate([obs, self._belief]).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        step = self.unwrapped.current_step
        self._belief = self._query_belief(step)

        # 매수 시점에 예상 수익률 기록
        if info.get("trade_executed") and self.unwrapped.btc_holding > 0:
            self._entry_expected_return = float(self._belief[0]) / 10.0

        return (
            np.concatenate([obs, self._belief]).astype(np.float32),
            reward, terminated, truncated, info,
        )


# ── Factory ─────────────────────────────────────────────────────────

def make_belief_env(
    market_frame: pd.DataFrame,
    sequence_features: list[str],
    context_features: list[str],
    portfolio_features: list[str],
    env_config,
    seq_len: int,
) -> BeliefAugmentedEnv:
    """BitcoinTradingEnvironment를 BeliefAugmentedEnv로 래핑."""
    from bitcoin_rl_system.trading_environment import BitcoinTradingEnvironment

    base = BitcoinTradingEnvironment(
        market_frame=market_frame,
        sequence_features=sequence_features,
        context_features=context_features,
        portfolio_features=portfolio_features,
        config=env_config,
    )

    print("[belief] Loading FAISS index & embeddings...")
    embeddings = np.load(str(_FAISS_DIR / "embeddings.npy")).astype(np.float32)
    metadata   = pd.read_parquet(_FAISS_DIR / "metadata.parquet")
    index      = faiss.read_index(str(_FAISS_DIR / "index.faiss"))
    index.nprobe = 64
    print(f"[belief] index loaded: {index.ntotal:,} vectors")

    return BeliefAugmentedEnv(base, embeddings, metadata, index, seq_len)
