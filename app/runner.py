"""Agent runner — loads trained PPO model and runs online learning on test split."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from bitcoin_rl_system.data_handler import BitcoinDataHandler, DataConfig
from bitcoin_rl_system.trading_environment import BitcoinTradingEnvironment, EnvironmentConfig

BroadcastFn = Callable[[dict], Awaitable[None]]

# 브로드캐스트 최대 주기 (초) — GPU에서 너무 빠를 경우 클라이언트 부하 방지
_MIN_BROADCAST_INTERVAL = 0.05  # 20 updates/sec max


class _StepCallback(BaseCallback):
    def __init__(
        self,
        broadcast: BroadcastFn,
        loop: asyncio.AbstractEventLoop,
        model_path: Path,
        vecnorm_path: Path,
        save_freq: int = 5_000,
    ) -> None:
        super().__init__()
        self._broadcast = broadcast
        self._loop = loop
        self._model_path = model_path
        self._vecnorm_path = vecnorm_path
        self._save_freq = save_freq
        self._last_sent = 0.0

    def _emit(self, data: dict) -> None:
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)

    def _on_step(self) -> bool:
        now = time.monotonic()
        if now - self._last_sent < _MIN_BROADCAST_INTERVAL:
            return True
        self._last_sent = now

        info = (self.locals.get("infos") or [{}])[0]
        rewards = self.locals.get("rewards") or [0.0]

        self._emit({
            "type": "step",
            "step": self.num_timesteps,
            "n_updates": self.model.n_updates,
            "reward": float(rewards[0]),
            "target_ratio": float(info.get("target_ratio", 0)),
            "equity": float(info.get("next_equity", 0)),
            "cash": float(info.get("cash", 0)),
            "btc_holding": float(info.get("btc_holding", 0)),
            "open": float(info.get("open", 0)),
            "high": float(info.get("high", 0)),
            "low": float(info.get("low", 0)),
            "close": float(info.get("close", 0)),
            "ts": str(info.get("ts", "")),
        })

        if self.num_timesteps % self._save_freq == 0:
            self.model.save(str(self._model_path))
            self.training_env.save(str(self._vecnorm_path))

        return True

    def _on_rollout_end(self) -> None:
        self._emit({"type": "update", "n_updates": self.model.n_updates})


class AgentRunner:
    def __init__(self, model_path: Path, vecnorm_path: Path) -> None:
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path
        self.is_running = False

    async def start(
        self,
        broadcast: BroadcastFn,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        thread = threading.Thread(
            target=self._run_loop,
            args=(broadcast, loop),
            daemon=True,
        )
        thread.start()
        self.is_running = True

    def _run_loop(
        self,
        broadcast: BroadcastFn,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        data_handler = BitcoinDataHandler(DataConfig())
        data_handler.load_processed_frames()
        splits = data_handler.split_by_time()
        layout = data_handler.build_feature_layout()
        seq_len = data_handler.config.sequence_length

        env_kwargs = {
            "market_frame": splits["test"],
            "sequence_features": layout["sequence"],
            "context_features": layout["context"],
            "portfolio_features": layout["portfolio"],
            "config": EnvironmentConfig(sequence_length=seq_len),
        }

        vec_env = make_vec_env(BitcoinTradingEnvironment, n_envs=1, env_kwargs=env_kwargs)
        vec_env = VecNormalize.load(str(self.vecnorm_path), vec_env)
        vec_env.training = True   # 온라인 학습: VecNormalize 통계도 계속 갱신
        vec_env.norm_reward = True

        model = PPO.load(str(self.model_path), env=vec_env)

        cb = _StepCallback(
            broadcast, loop,
            self.model_path, self.vecnorm_path,
            save_freq=5_000,
        )

        # 사실상 무한 학습 — 에피소드 끝나면 자동 reset 후 재시작
        model.learn(
            total_timesteps=100_000_000,
            callback=cb,
            reset_num_timesteps=False,
            progress_bar=False,
        )
