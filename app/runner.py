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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from bitcoin_rl_system.data_handler import BitcoinDataHandler, DataConfig
from bitcoin_rl_system.trading_environment import BitcoinTradingEnvironment, EnvironmentConfig

BroadcastFn = Callable[[dict], Awaitable[None]]

# 브로드캐스트 최대 주기 (초) — GPU에서 너무 빠를 경우 클라이언트 부하 방지
_MIN_BROADCAST_INTERVAL = 0.05  # 20 updates/sec max


class _InferenceStreamer:
    def __init__(
        self,
        broadcast: BroadcastFn,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._broadcast = broadcast
        self._loop = loop
        self._last_sent = 0.0

    def emit(self, data: dict) -> None:
        now = time.monotonic()
        if now - self._last_sent < _MIN_BROADCAST_INTERVAL:
            return
        self._last_sent = now
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)


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

        try:
            vec_env = VecNormalize.load(str(self.vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            model = PPO.load(str(self.model_path), env=vec_env)
        except (AssertionError, Exception) as e:
            print(f"[runner] Model load failed: {e}")
            print("[runner] Starting demo mode (price replay only)")
            vec_env.close()
            self._demo_loop(broadcast, loop, splits["test"])
            return

        streamer = _InferenceStreamer(broadcast, loop)
        obs = vec_env.reset()
        step = 0

        while True:
            t0 = time.monotonic()
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            info = infos[0]
            reward = rewards[0]
            step += 1

            streamer.emit({
                "type": "step",
                "step": step,
                "n_updates": 0,
                "reward": float(reward),
                "action": int(info.get("action", 0)),
                "target_ratio": float(info.get("target_ratio", 0.0)),
                "position_ratio": float(info.get("position_ratio", 0.0)),
                "equity": float(info.get("next_equity", 0.0)),
                "cash": float(info.get("cash", 0.0)),
                "btc_holding": float(info.get("btc_holding", 0.0)),
                "avg_entry_price": float(info.get("avg_entry_price", 0.0)),
                "realized_pnl": float(info.get("realized_pnl", 0.0)),
                "open": float(info.get("open", 0.0)),
                "high": float(info.get("high", 0.0)),
                "low": float(info.get("low", 0.0)),
                "close": float(info.get("close", 0.0)),
                "ts": str(info.get("ts", "")),
                "trade_executed": bool(info.get("trade_executed", False)),
            })

            elapsed = time.monotonic() - t0
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

            if dones[0]:
                streamer.emit({"type": "update", "n_updates": 0})
                obs = vec_env.reset()
                step = 0

    def _demo_loop(
        self,
        broadcast: BroadcastFn,
        loop: asyncio.AbstractEventLoop,
        test_df,
    ) -> None:
        import pandas as pd
        streamer = _InferenceStreamer(broadcast, loop)
        step = 0
        df = test_df.reset_index(drop=True)
        while True:
            for idx in range(len(df)):
                row = df.iloc[idx]
                ts = str(row.get("ts", ""))
                streamer.emit({
                    "type": "step",
                    "step": step,
                    "n_updates": 0,
                    "reward": 0.0,
                    "action": 2,
                    "target_ratio": 0.0,
                    "position_ratio": 0.0,
                    "equity": 1_000_000.0,
                    "cash": 1_000_000.0,
                    "btc_holding": 0.0,
                    "avg_entry_price": 0.0,
                    "realized_pnl": 0.0,
                    "open":  float(row.get("open",  0)),
                    "high":  float(row.get("high",  0)),
                    "low":   float(row.get("low",   0)),
                    "close": float(row.get("close", 0)),
                    "ts": ts,
                    "trade_executed": False,
                })
                step += 1
                time.sleep(0.05)
            streamer.emit({"type": "update", "n_updates": 0})
            step = 0
