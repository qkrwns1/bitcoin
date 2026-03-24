"""Step 1 of Phase 2: Transformer embedding extraction.

전체 5m 데이터셋의 각 bar에 대해 256-dim encoder embedding을 추출하고
미래 수익률(1h/5h/1d)을 메타데이터로 저장.

실행: python -m bitcoin_rl_system.extract_embeddings
출력:
  bitcoin_rl_system/faiss/embeddings.npy     - (N, 256) float32
  bitcoin_rl_system/faiss/metadata.parquet   - ts, close, return_1h/5h/1d, bar_idx
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from bitcoin_rl_system.data_handler import BitcoinDataHandler, DataConfig
from bitcoin_rl_system.trading_environment import _PRICE_COLS, _LOG_COLS

_HERE       = Path(__file__).parent
CHECKPOINT  = _HERE / "checkpoints" / "final_model"   # SB3가 .zip 자동 추가
VECNORM     = _HERE / "checkpoints" / "vec_normalize.pkl"
OUT_DIR     = _HERE / "faiss"
BATCH_SIZE  = 512

# 미래 수익률 윈도우 (bar 단위, 5m봉)
FUTURE_WINDOWS = {"return_1h": 12, "return_5h": 60, "return_1d": 288}


# ── Normalization ───────────────────────────────────────────────────

def load_normalizer(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def normalize(obs: np.ndarray, vn) -> np.ndarray:
    return np.clip(
        (obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8),
        -10.0, 10.0,
    ).astype(np.float32)


# ── Observation builder ─────────────────────────────────────────────

def build_obs_batch(
    seq_arr: np.ndarray,   # (N, seq_dim)
    ctx_arr: np.ndarray,   # (N, ctx_dim)
    close_arr: np.ndarray, # (N,)
    price_mask: np.ndarray,
    log_mask: np.ndarray,
    seq_len: int,
    portfolio_dim: int,
    indices: np.ndarray,   # bar indices to process
) -> np.ndarray:
    B = len(indices)
    seq_dim = seq_arr.shape[1]
    ctx_dim = ctx_arr.shape[1]
    obs_dim = seq_len * seq_dim + ctx_dim + portfolio_dim

    out = np.zeros((B, obs_dim), dtype=np.float32)

    for k, i in enumerate(indices):
        start = i - seq_len + 1
        seq   = seq_arr[start:i + 1].copy()   # (seq_len, seq_dim)
        cp    = float(close_arr[i])

        if cp > 0:
            seq[:, price_mask] /= cp
        seq[:, log_mask] = np.log1p(np.maximum(seq[:, log_mask], 0.0))

        ctx = ctx_arr[i]
        # portfolio = zeros (neutral state)

        offset = 0
        out[k, offset:offset + seq_len * seq_dim] = seq.reshape(-1)
        offset += seq_len * seq_dim
        out[k, offset:offset + ctx_dim] = ctx
        # portfolio part stays zero

    return out


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ────────────────────────────────────────────────
    print(f"모델 로드: {CHECKPOINT}")
    model = PPO.load(str(CHECKPOINT))
    extractor = model.policy.features_extractor
    extractor.eval()
    device = next(extractor.parameters()).device
    print(f"  device: {device}")

    vn = load_normalizer(VECNORM)
    print("  VecNormalize 로드 완료")

    # ── 데이터 로드 ──────────────────────────────────────────────
    print("데이터 로드...")
    cfg     = DataConfig()
    handler = BitcoinDataHandler(cfg)
    frame   = handler.load_processed_frames()
    layout  = handler.build_feature_layout()
    seq_len = cfg.sequence_length

    seq_features = layout["sequence"]
    ctx_features = layout["context"]
    portfolio_dim = len(layout["portfolio"])  # 7

    # 문자열 컬럼 → float (trading_environment._prepare_frame 동일)
    frame["market_state"]   = frame["market_state"].map({"ACTIVE": 1.0}).fillna(0.0)
    frame["market_warning"] = frame["market_warning"].map({"NONE": 0.0}).fillna(1.0)
    frame["ask_bid"]        = frame["ask_bid"].map({"ASK": -1.0, "BID": 1.0}).fillna(0.0)
    frame["is_trading_suspended"] = frame["is_trading_suspended"].astype(float)
    for col in seq_features + ctx_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    seq_arr   = frame[seq_features].to_numpy(dtype=np.float32)
    ctx_arr   = frame[ctx_features].to_numpy(dtype=np.float32)
    close_arr = frame["close"].to_numpy(dtype=np.float32)
    ts_arr    = frame["ts"].values

    price_mask = np.array([col in _PRICE_COLS for col in seq_features])
    log_mask   = np.array([col in _LOG_COLS   for col in seq_features])

    # ── 유효 bar 범위 ────────────────────────────────────────────
    max_future = max(FUTURE_WINDOWS.values())
    valid_start = seq_len          # 최소 seq_len 이전 bar 필요
    valid_end   = len(frame) - max_future - 1
    valid_idx   = np.arange(valid_start, valid_end)
    N = len(valid_idx)
    print(f"  유효 bar 수: {N:,}  ({frame['ts'].iloc[valid_start]} ~ {frame['ts'].iloc[valid_end]})")

    # ── embedding 추출 ───────────────────────────────────────────
    emb_dim    = extractor.fusion[-1].normalized_shape[0] if hasattr(extractor.fusion[-1], "normalized_shape") else 256
    # 실제 출력 차원 확인
    with torch.no_grad():
        dummy = torch.zeros(1, seq_len * len(seq_features) + len(ctx_features) + portfolio_dim, device=device)
        emb_dim = extractor(dummy).shape[1]
    print(f"  embedding 차원: {emb_dim}")

    embeddings = np.zeros((N, emb_dim), dtype=np.float32)
    n_batches  = (N + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"embedding 추출 중 (batch={BATCH_SIZE}, total={n_batches} batches)...")
    for b in range(n_batches):
        batch_idx = valid_idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        obs = build_obs_batch(
            seq_arr, ctx_arr, close_arr,
            price_mask, log_mask,
            seq_len, portfolio_dim, batch_idx,
        )
        obs_norm = normalize(obs, vn)
        obs_t    = torch.tensor(obs_norm, dtype=torch.float32, device=device)

        with torch.no_grad():
            emb = extractor(obs_t).cpu().numpy()

        start = b * BATCH_SIZE
        embeddings[start:start + len(batch_idx)] = emb

        if (b + 1) % 100 == 0 or b == n_batches - 1:
            pct = (b + 1) / n_batches * 100
            print(f"  {b+1}/{n_batches} ({pct:.1f}%)  bar {batch_idx[-1]:,}")

    # ── 메타데이터 (미래 수익률) ─────────────────────────────────
    print("메타데이터 생성...")
    meta = {
        "bar_idx": valid_idx,
        "ts":      ts_arr[valid_idx],
        "close":   close_arr[valid_idx],
    }
    for name, w in FUTURE_WINDOWS.items():
        future_close = close_arr[valid_idx + w]
        meta[name]   = (future_close - close_arr[valid_idx]) / close_arr[valid_idx]

    meta_df = pd.DataFrame(meta)

    # ── 저장 ────────────────────────────────────────────────────
    emb_path  = OUT_DIR / "embeddings.npy"
    meta_path = OUT_DIR / "metadata.parquet"

    np.save(str(emb_path), embeddings)
    meta_df.to_parquet(str(meta_path), index=False)

    print(f"\n완료!")
    print(f"  embeddings : {emb_path}  {embeddings.shape}  {embeddings.nbytes/1e6:.1f} MB")
    print(f"  metadata   : {meta_path}  {len(meta_df):,} rows")
    print(f"\n다음 단계: python -m bitcoin_rl_system.build_faiss")


if __name__ == "__main__":
    main()
