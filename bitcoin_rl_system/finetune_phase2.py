"""Phase 2 Fine-tuning: Belief-Augmented RL.

Phase 1 체크포인트(portfolio_dim=7, obs=1642)에서 가중치를 이어받아
BeliefAugmentedEnv(portfolio_dim=10, obs=1645)로 PPO fine-tuning.

실행: python -m bitcoin_rl_system.finetune_phase2 [--timesteps N]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent))

from bitcoin_rl_system.belief_environment import make_belief_env
from bitcoin_rl_system.data_handler import BitcoinDataHandler, DataConfig
from bitcoin_rl_system.rl_agent import AgentConfig, SequenceContextFeatureExtractor
from bitcoin_rl_system.trading_environment import EnvironmentConfig

_HERE = Path(__file__).parent
PHASE1_MODEL   = _HERE / "checkpoints" / "final_model"        # .zip 자동
PHASE1_VECNORM = _HERE / "checkpoints" / "vec_normalize.pkl"
PHASE2_CKPT    = _HERE / "checkpoints" / "phase2_model"
PHASE2_VECNORM = _HERE / "checkpoints" / "phase2_vec_normalize.pkl"
RUNS_DIR       = _HERE / "runs"

N_ENVS = 4   # SubprocVecEnv로 CPU 코어 병렬 사용
N_BELIEF = 3


# ── VecNormalize 확장 ────────────────────────────────────────────────

def _extend_vecnorm(vn_path: Path, new_dim: int) -> VecNormalize:
    """Phase 1 VecNormalize의 obs_rms를 1642→1645로 확장."""
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)

    old_mean = vn.obs_rms.mean       # (1642,)
    old_var  = vn.obs_rms.var        # (1642,)
    pad      = N_BELIEF

    vn.obs_rms.mean  = np.concatenate([old_mean, np.zeros(pad, dtype=old_mean.dtype)])
    vn.obs_rms.var   = np.concatenate([old_var,  np.ones( pad, dtype=old_var.dtype)])
    vn.obs_rms.count = float(vn.obs_rms.count)

    # observation_space 교체
    import gymnasium as gym
    vn.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
    )
    return vn


# ── 가중치 이전 ──────────────────────────────────────────────────────

def _transfer_weights(phase1_model: PPO, phase2_model: PPO) -> None:
    """Phase 1 → Phase 2 가중치 복사.

    portfolio_mlp[0] (Linear 7→32) 만 shape 불일치 → 첫 7열 복사, 나머지 0 초기화.
    나머지 모든 레이어는 그대로 복사.
    """
    src = phase1_model.policy.state_dict()
    dst = phase2_model.policy.state_dict()

    copied = skipped = 0
    for name, dst_param in dst.items():
        if name not in src:
            skipped += 1
            continue

        src_param = src[name]
        if src_param.shape == dst_param.shape:
            dst_param.copy_(src_param)
            copied += 1
        else:
            # portfolio_mlp[0].weight: (32, 7) → (32, 10)
            if "portfolio_mlp" in name and "weight" in name:
                old_cols = src_param.shape[1]
                dst_param.zero_()
                dst_param[:, :old_cols].copy_(src_param)
                copied += 1
                print(f"  [partial] {name}: {src_param.shape} → {dst_param.shape} "
                      f"(cols 0-{old_cols-1} 복사, 나머지 0)")
            else:
                print(f"  [skip] {name}: shape {src_param.shape} ≠ {dst_param.shape}")
                skipped += 1

    phase2_model.policy.load_state_dict(dst)
    print(f"  가중치 이전 완료: {copied} 레이어 복사, {skipped} 건 스킵")


# ── 환경 구성 ─────────────────────────────────────────────────────────

def _make_phase2_vec_env(market_frame, layout, seq_len):
    env_kwargs = {
        "market_frame": market_frame,
        "sequence_features": layout["sequence"],
        "context_features": layout["context"],
        "portfolio_features": layout["portfolio"],
        "env_config": EnvironmentConfig(sequence_length=seq_len),
        "seq_len": seq_len,
    }
    return make_vec_env(make_belief_env, n_envs=N_ENVS, env_kwargs=env_kwargs,
                        vec_env_cls=SubprocVecEnv)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Fine-tuning")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    args = parser.parse_args()

    # ── 데이터 로드 ───────────────────────────────────────────────
    print("데이터 로드...")
    cfg     = DataConfig()
    handler = BitcoinDataHandler(cfg)
    handler.load_processed_frames()
    splits  = handler.split_by_time()
    layout  = handler.build_feature_layout()
    seq_len = cfg.sequence_length
    print(handler.summary())

    # ── Phase 2 환경 생성 ─────────────────────────────────────────
    print(f"\nPhase 2 환경 생성 (BeliefAugmentedEnv × {N_ENVS}, SubprocVecEnv)...")
    train_vec = _make_phase2_vec_env(splits["train"], layout, seq_len)

    # ── Phase 1 VecNormalize → 확장 후 적용 ───────────────────────
    print("VecNormalize 확장 (1642 → 1645)...")
    vn = _extend_vecnorm(PHASE1_VECNORM, new_dim=1642 + N_BELIEF)

    # SB3 VecNormalize를 새 환경으로 재래핑
    train_env = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # obs_rms 를 Phase 1 통계로 초기화
    train_env.obs_rms  = vn.obs_rms
    train_env.ret_rms  = vn.ret_rms

    # ── Phase 2 PPO 모델 신규 빌드 ────────────────────────────────
    print("Phase 2 PPO 모델 빌드...")
    # SubprocVecEnv는 envs 직접 접근 불가 → observation_space로 dim 직접 지정
    # BeliefAugmentedEnv: portfolio_dim=10, sequence_length/dim/context_dim은 base env와 동일
    _base_cfg = DataConfig()
    _base_layout = layout
    _portfolio_dim = len(_base_layout["portfolio"]) + N_BELIEF   # 7 + 3 = 10
    _seq_len  = seq_len
    _seq_dim  = len(_base_layout["sequence"])
    _ctx_dim  = len(_base_layout["context"])

    policy_kwargs = {
        "features_extractor_class": SequenceContextFeatureExtractor,
        "features_extractor_kwargs": {
            "sequence_length":      _seq_len,
            "sequence_dim":         _seq_dim,
            "context_dim":          _ctx_dim,
            "portfolio_dim":        _portfolio_dim,   # 10
            "transformer_hidden_dim": 128,
            "transformer_layers":   2,
            "context_hidden_dim":   64,
            "portfolio_hidden_dim": 32,
            "fused_hidden_dim":     256,
        },
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),
    }

    phase2_model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=4096,    # 4096 × 4 envs = 16384 스텝/업데이트 → GPU 사용률 증가
        batch_size=512,
        n_epochs=5,
        clip_range=0.1,
        ent_coef=0.001,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(RUNS_DIR),
    )
    print(f"  portfolio_dim: {_portfolio_dim}")
    print(f"  obs_dim:       {_seq_len * _seq_dim + _ctx_dim + _portfolio_dim}")

    # ── Phase 1 → Phase 2 가중치 이전 ────────────────────────────
    print("\nPhase 1 체크포인트 로드: ", PHASE1_MODEL)
    phase1_model = PPO.load(str(PHASE1_MODEL))

    print("가중치 이전...")
    _transfer_weights(phase1_model, phase2_model)
    del phase1_model

    # ── Fine-tuning ───────────────────────────────────────────────
    from stable_baselines3.common.callbacks import CheckpointCallback

    PHASE2_CKPT.parent.mkdir(parents=True, exist_ok=True)
    cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(PHASE2_CKPT.parent),
        name_prefix="phase2_ckpt",
        save_vecnormalize=True,
        verbose=1,
    )

    print(f"\nFine-tuning 시작: {args.timesteps:,} steps  lr={args.lr}")
    phase2_model.learn(
        total_timesteps=args.timesteps,
        callback=cb,
        progress_bar=False,
        reset_num_timesteps=True,
    )

    # ── 저장 ──────────────────────────────────────────────────────
    phase2_model.save(str(PHASE2_CKPT))
    train_env.save(str(PHASE2_VECNORM))
    print(f"\n완료!")
    print(f"  모델     : {PHASE2_CKPT}.zip")
    print(f"  VecNorm  : {PHASE2_VECNORM}")


if __name__ == "__main__":
    main()
