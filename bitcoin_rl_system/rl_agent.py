"""PPO agent scaffold with explicit Transformer attachment point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv


@dataclass(slots=True)
class AgentConfig:
    algorithm: str = "PPO"
    transformer_layers: int = 1
    transformer_hidden_dim: int = 64
    context_hidden_dim: int = 32
    portfolio_hidden_dim: int = 32
    fused_hidden_dim: int = 128
    learning_rate: float = 3e-5
    n_steps: int = 4096
    batch_size: int = 128
    n_epochs: int = 5
    clip_range: float = 0.1
    total_timesteps: int = 5_000_000
    save_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    tensorboard_log: Path = field(default_factory=lambda: Path("runs"))
    checkpoint_freq: int = 100_000


class SequenceContextFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        sequence_length: int,
        sequence_dim: int,
        context_dim: int,
        portfolio_dim: int,
        transformer_hidden_dim: int = 128,
        transformer_layers: int = 2,
        context_hidden_dim: int = 64,
        portfolio_hidden_dim: int = 64,
        fused_hidden_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim=fused_hidden_dim)
        self.sequence_length = sequence_length
        self.sequence_dim = sequence_dim
        self.context_dim = context_dim
        self.portfolio_dim = portfolio_dim

        self.sequence_projection = nn.Linear(sequence_dim, transformer_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=4,
            dim_feedforward=transformer_hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.GELU(),
        )
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(portfolio_dim, portfolio_hidden_dim),
            nn.LayerNorm(portfolio_hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(transformer_hidden_dim + context_hidden_dim + portfolio_hidden_dim, fused_hidden_dim),
            nn.LayerNorm(fused_hidden_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        seq_end = self.sequence_length * self.sequence_dim
        ctx_end = seq_end + self.context_dim

        seq_flat = observations[:, :seq_end]
        ctx = observations[:, seq_end:ctx_end]
        port = observations[:, ctx_end:]

        seq = seq_flat.view(batch_size, self.sequence_length, self.sequence_dim)
        seq = self.sequence_projection(seq)
        seq_encoded = self.transformer(seq)
        seq_embedding = seq_encoded[:, -1, :]
        ctx_embedding = self.context_mlp(ctx)
        port_embedding = self.portfolio_mlp(port)
        fused = torch.cat([seq_embedding, ctx_embedding, port_embedding], dim=1)
        return self.fusion(fused)


class BitcoinRLAgent:
    def __init__(self, env: VecEnv, config: AgentConfig) -> None:
        self.env = env
        self.config = config
        self.model: PPO | None = None

    def _unwrap_env(self) -> gym.Env:
        """VecNormalize/DummyVecEnv/Monitor 래퍼를 벗겨서 원본 환경을 반환."""
        env = self.env
        while hasattr(env, "venv"):
            env = env.venv
        if hasattr(env, "envs"):
            env = env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env

    def build_model(self) -> PPO:
        raw_env = self._unwrap_env()
        policy_kwargs = {
            "features_extractor_class": SequenceContextFeatureExtractor,
            "features_extractor_kwargs": {
                "sequence_length": raw_env.sequence_length,
                "sequence_dim": raw_env.sequence_dim,
                "context_dim": raw_env.context_dim,
                "portfolio_dim": raw_env.portfolio_dim,
                "transformer_hidden_dim": self.config.transformer_hidden_dim,
                "transformer_layers": self.config.transformer_layers,
                "context_hidden_dim": self.config.context_hidden_dim,
                "portfolio_hidden_dim": self.config.portfolio_hidden_dim,
                "fused_hidden_dim": self.config.fused_hidden_dim,
            },
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        }
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            clip_range=self.config.clip_range,
            ent_coef=0.01,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.config.tensorboard_log),
        )
        return self.model

    def train(self, eval_env: VecEnv | None = None) -> PPO:
        if self.model is None:
            self.build_model()
        assert self.model is not None

        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            CheckpointCallback(
                save_freq=self.config.checkpoint_freq,
                save_path=str(self.config.save_dir),
                save_vecnormalize=True,
                verbose=1,
            )
        ]

        if eval_env is not None:
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(self.config.save_dir / "best"),
                    log_path=str(self.config.save_dir / "eval_logs"),
                    eval_freq=self.config.checkpoint_freq,
                    n_eval_episodes=3,
                    deterministic=True,
                    verbose=1,
                )
            )

        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
        return self.model

    def save(self, path: Path) -> None:
        assert self.model is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Path) -> PPO:
        self.model = PPO.load(str(path), env=self.env)
        return self.model

    def evaluate(self, env: VecEnv, n_episodes: int = 5) -> dict[str, float]:
        assert self.model is not None
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += float(reward[0])
                steps += 1
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "n_episodes": n_episodes,
        }
