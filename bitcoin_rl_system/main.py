"""Pipeline entry point for the Bitcoin RL system."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT = PROJECT_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from bitcoin_rl_system.data_handler import BitcoinDataHandler, DataConfig
from bitcoin_rl_system.rl_agent import AgentConfig, BitcoinRLAgent
from bitcoin_rl_system.trading_environment import BitcoinTradingEnvironment, EnvironmentConfig

VECNORM_PATH = PROJECT_ROOT / "vec_normalize.pkl"


def build_training_components():
    data_handler = BitcoinDataHandler(DataConfig())
    data_handler.load_processed_frames()
    splits = data_handler.split_by_time()
    layout = data_handler.build_feature_layout()
    seq_len = data_handler.config.sequence_length

    env_kwargs = {
        "market_frame": splits["train"],
        "sequence_features": layout["sequence"],
        "context_features": layout["context"],
        "portfolio_features": layout["portfolio"],
        "config": EnvironmentConfig(sequence_length=seq_len),
    }
    vec_env = make_vec_env(BitcoinTradingEnvironment, n_envs=1, env_kwargs=env_kwargs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    agent = BitcoinRLAgent(env=vec_env, config=AgentConfig())
    return data_handler, vec_env, agent


def build_eval_env(market_frame, layout, seq_len, vecnorm_path: Path = VECNORM_PATH):
    """평가용 환경: 훈련 때 기록한 VecNormalize 통계를 freeze해서 로드."""
    env_kwargs = {
        "market_frame": market_frame,
        "sequence_features": layout["sequence"],
        "context_features": layout["context"],
        "portfolio_features": layout["portfolio"],
        "config": EnvironmentConfig(sequence_length=seq_len),
    }
    vec_env = make_vec_env(BitcoinTradingEnvironment, n_envs=1, env_kwargs=env_kwargs)
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def main() -> None:
    data_handler, vec_env, agent = build_training_components()
    raw_env = agent._unwrap_env()
    print("Bitcoin RL system")
    print(f"Root: {PROJECT_ROOT}")
    print(data_handler.summary())
    print(f"Observation dim: {raw_env.obs_dim}")
    print(f"Action space: {raw_env.action_space}")
    agent.build_model()
    print("PPO model build complete (with VecNormalize).")
    agent.train()
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
