"""Pipeline entry point for the Bitcoin RL system."""

from __future__ import annotations

import argparse
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

VECNORM_PATH = PROJECT_ROOT / "checkpoints" / "vec_normalize.pkl"


def _make_vec_env(market_frame, layout, seq_len) -> make_vec_env:
    env_kwargs = {
        "market_frame": market_frame,
        "sequence_features": layout["sequence"],
        "context_features": layout["context"],
        "portfolio_features": layout["portfolio"],
        "config": EnvironmentConfig(sequence_length=seq_len),
    }
    return make_vec_env(BitcoinTradingEnvironment, n_envs=4, env_kwargs=env_kwargs)


def build_eval_env(market_frame, layout, seq_len, vecnorm_path: Path = VECNORM_PATH):
    """평가용 환경: 훈련 때 기록한 VecNormalize 통계를 freeze해서 로드."""
    vec_env = _make_vec_env(market_frame, layout, seq_len)
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def cmd_train(args) -> None:
    data_handler = BitcoinDataHandler(DataConfig())
    data_handler.load_processed_frames()
    splits = data_handler.split_by_time()
    layout = data_handler.build_feature_layout()
    seq_len = data_handler.config.sequence_length

    print("Bitcoin RL: Train")
    print(data_handler.summary())

    train_env = _make_vec_env(splits["train"], layout, seq_len)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    val_env = build_eval_env(splits["val"], layout, seq_len, VECNORM_PATH) if VECNORM_PATH.exists() else None

    agent_config = AgentConfig(
        total_timesteps=args.timesteps,
        save_dir=PROJECT_ROOT / "checkpoints",
        tensorboard_log=PROJECT_ROOT / "runs",
        checkpoint_freq=args.checkpoint_freq,
    )
    agent = BitcoinRLAgent(env=train_env, config=agent_config)
    agent.build_model()

    raw_env = agent._unwrap_env()
    print(f"Observation dim: {raw_env.obs_dim}")
    print(f"Action space: {raw_env.action_space}")
    print(f"Total timesteps: {args.timesteps:,}")

    agent.train(eval_env=val_env)

    # 최종 모델 + VecNormalize 저장
    save_path = PROJECT_ROOT / "checkpoints" / "final_model"
    agent.save(save_path)
    train_env.save(str(VECNORM_PATH))
    print(f"모델 저장: {save_path}.zip")
    print(f"VecNormalize 저장: {VECNORM_PATH}")


def cmd_eval(args) -> None:
    data_handler = BitcoinDataHandler(DataConfig())
    data_handler.load_processed_frames()
    splits = data_handler.split_by_time()
    layout = data_handler.build_feature_layout()
    seq_len = data_handler.config.sequence_length

    split_name = args.split
    eval_env = build_eval_env(splits[split_name], layout, seq_len, Path(args.vecnorm))

    agent_config = AgentConfig()
    agent = BitcoinRLAgent(env=eval_env, config=agent_config)
    agent.load(Path(args.model))

    print(f"평가 중: {split_name} split ({args.episodes} episodes)")
    results = agent.evaluate(eval_env, n_episodes=args.episodes)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def cmd_smoke(args) -> None:
    data_handler = BitcoinDataHandler(DataConfig())
    data_handler.load_processed_frames()
    splits = data_handler.split_by_time()
    layout = data_handler.build_feature_layout()
    seq_len = data_handler.config.sequence_length

    train_env = _make_vec_env(splits["train"], layout, seq_len)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    agent_config = AgentConfig(total_timesteps=1024, n_steps=256, batch_size=64)
    agent = BitcoinRLAgent(env=train_env, config=agent_config)
    agent.build_model()

    raw_env = agent._unwrap_env()
    print("Bitcoin RL: Smoke Test")
    print(data_handler.summary())
    print(f"Observation dim: {raw_env.obs_dim}")
    print(f"Action space: {raw_env.action_space}")

    agent.train()
    print("Smoke test complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitcoin RL System")
    sub = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train", help="모델 학습")
    p_train.add_argument("--timesteps", type=int, default=15_000_000)
    p_train.add_argument("--checkpoint-freq", type=int, default=100_000)

    # eval
    p_eval = sub.add_parser("eval", help="모델 평가")
    p_eval.add_argument("--model", default=str(PROJECT_ROOT / "checkpoints" / "final_model"))
    p_eval.add_argument("--vecnorm", default=str(VECNORM_PATH))
    p_eval.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_eval.add_argument("--episodes", type=int, default=5)

    # smoke
    sub.add_parser("smoke", help="파이프라인 검증 (1024 스텝)")

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
