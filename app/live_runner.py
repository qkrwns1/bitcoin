"""Live inference runner — real-time Upbit 5m candles + PPO agent."""

from __future__ import annotations

import asyncio
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from bitcoin_rl_system.data_handler import BitcoinDataHandler, DataConfig
from bitcoin_rl_system.trading_environment import _PRICE_COLS, _LOG_COLS, TARGET_LEVELS

_UPBIT_5M = "https://api.upbit.com/v1/candles/minutes/5"
_INITIAL_CASH = 1_000_000.0
_FEE_RATE = 0.0005
_MIN_REBALANCE = 5_000.0
# 52주 rolling에 필요한 최소 raw bar 수 (365 * 288 = 105,120)
_FEATURE_BUFFER = 105_120 + 500


# ── Upbit fetch ─────────────────────────────────────────────────────

def _fetch_new_raw(since_ts: pd.Timestamp) -> pd.DataFrame:
    """since_ts 이후의 모든 5m 캔들을 Upbit에서 가져옴."""
    rows: list[dict] = []
    to_param: str | None = None

    while True:
        params: dict = {"market": "KRW-BTC", "count": 200}
        if to_param:
            params["to"] = to_param
        try:
            resp = requests.get(_UPBIT_5M, params=params, timeout=10)
            data = resp.json()
        except Exception as e:
            print(f"[live] Upbit fetch error: {e}")
            break

        if not data:
            break

        done = False
        for c in data:
            ts = pd.Timestamp(c["candle_date_time_utc"])
            if ts <= since_ts:
                done = True
                break
            rows.append({
                "ts": ts,
                "market": "KRW-BTC",
                "opening_price":           float(c["opening_price"]),
                "high_price":              float(c["high_price"]),
                "low_price":               float(c["low_price"]),
                "trade_price":             float(c["trade_price"]),
                "candle_acc_trade_volume": float(c["candle_acc_trade_volume"]),
                "candle_acc_trade_price":  float(c["candle_acc_trade_price"]),
            })
        if done or len(data) < 200:
            break
        to_param = data[-1]["candle_date_time_utc"]
        time.sleep(0.12)  # Upbit rate limit

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


# ── Feature computation ─────────────────────────────────────────────

def _compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """prepare_5m_data.build_features 와 동일한 로직."""
    from bitcoin_rl_system.prepare_5m_data import build_features
    featured = build_features(raw_df)
    # 컨텍스트 문자열 → float (trading_environment._prepare_frame 동일)
    featured["market_state"]   = featured["market_state"].map({"ACTIVE": 1.0}).fillna(0.0)
    featured["market_warning"] = featured["market_warning"].map({"NONE": 0.0}).fillna(1.0)
    featured["ask_bid"]        = featured["ask_bid"].map({"ASK": -1.0, "BID": 1.0}).fillna(0.0)
    featured["is_trading_suspended"] = featured["is_trading_suspended"].astype(float)
    return featured.reset_index(drop=True)


# ── Observation builder ─────────────────────────────────────────────

def _build_obs(
    featured: pd.DataFrame,
    seq_features: list[str],
    ctx_features: list[str],
    seq_len: int,
    cash: float,
    btc: float,
    avg_entry: float,
    realized_pnl: float,
    steps_since_rebalance: int,
    current_price: float,
) -> np.ndarray:
    price_mask = np.array([col in _PRICE_COLS for col in seq_features])
    log_mask   = np.array([col in _LOG_COLS   for col in seq_features])

    seq = featured[seq_features].tail(seq_len).to_numpy(dtype=np.float32)
    ctx = featured[ctx_features].iloc[-1].to_numpy(dtype=np.float32)

    if current_price > 0:
        seq[:, price_mask] /= current_price
    seq[:, log_mask] = np.log1p(np.maximum(seq[:, log_mask], 0.0))

    total_equity  = cash + btc * current_price
    btc_value     = btc * current_price
    position_ratio = btc_value / total_equity if total_equity > 0 else 0.0
    unrealized_pct = 0.0
    if btc > 0 and avg_entry > 0:
        unrealized_pct = (current_price - avg_entry) / avg_entry

    port = np.array([
        cash / _INITIAL_CASH,
        btc_value / _INITIAL_CASH,
        position_ratio,
        unrealized_pct,
        min(steps_since_rebalance / 60.0, 2.0),
        total_equity / _INITIAL_CASH,
        realized_pnl / _INITIAL_CASH,
    ], dtype=np.float32)

    return np.concatenate([seq.reshape(-1), ctx, port]).astype(np.float32)


# ── VecNormalize manual normalization ──────────────────────────────

def _normalize_obs(obs: np.ndarray, vn) -> np.ndarray:
    clipped = np.clip(
        (obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8),
        -10.0, 10.0,
    )
    return clipped.astype(np.float32)


# ── Portfolio: rebalance ───────────────────────────────────────────

def _rebalance(
    target_ratio: float,
    price: float,
    cash: float,
    btc: float,
    avg_entry: float,
    realized_pnl: float,
) -> tuple[float, float, float, float, bool]:
    total = cash + btc * price
    if total <= 0 or price <= 0:
        return cash, btc, avg_entry, realized_pnl, False

    target_val  = total * target_ratio
    current_val = btc * price
    diff        = target_val - current_val

    if abs(diff) < _MIN_REBALANCE:
        return cash, btc, avg_entry, realized_pnl, False

    if diff > 0:  # 매수
        affordable = min(diff, cash / (1.0 + _FEE_RATE))
        if affordable <= 0:
            return cash, btc, avg_entry, realized_pnl, False
        fee       = affordable * _FEE_RATE
        bought    = affordable / price
        prev_cost = avg_entry * btc
        cash     -= affordable + fee
        btc      += bought
        avg_entry = (prev_cost + affordable) / btc
        return cash, btc, avg_entry, realized_pnl, True

    # 매도
    sell_val = min(-diff, current_val)
    if sell_val <= 0:
        return cash, btc, avg_entry, realized_pnl, False
    sold      = sell_val / price
    fee       = sell_val * _FEE_RATE
    proceeds  = sell_val - fee
    realized  = sold * (price - avg_entry) - fee
    cash     += proceeds
    btc       = max(0.0, btc - sold)
    realized_pnl += realized
    if btc < 1e-12:
        btc = 0.0
        avg_entry = 0.0
    return cash, btc, avg_entry, realized_pnl, True


# ── Timing helper ──────────────────────────────────────────────────

def _seconds_to_next_bar(interval: int = 300) -> float:
    """다음 5분봉 마감까지 남은 초 (+10초 여유)."""
    now = time.time()
    return interval - (now % interval) + 10.0


# ── LiveAgentRunner ────────────────────────────────────────────────

class LiveAgentRunner:
    def __init__(self, model_path: Path, vecnorm_path: Path) -> None:
        self.model_path   = model_path
        self.vecnorm_path = vecnorm_path
        self.is_running   = False

    async def start(
        self,
        broadcast,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        thread = threading.Thread(
            target=self._run_loop,
            args=(broadcast, loop),
            daemon=True,
        )
        thread.start()
        self.is_running = True

    def _run_loop(self, broadcast, loop: asyncio.AbstractEventLoop) -> None:
        from app.runner import _InferenceStreamer
        streamer = _InferenceStreamer(broadcast, loop)

        # ── 1. 모델 로드 ──────────────────────────────────────────
        print("[live] Loading model...")
        model = PPO.load(str(self.model_path))
        with open(self.vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        print("[live] Model loaded.")

        # ── 2. 역사 데이터 로드 ───────────────────────────────────
        print("[live] Loading historical raw candles...")
        cfg      = DataConfig()
        raw_path = cfg.data_root / "raw" / "candles_5m" / "KRW-BTC_5m.parquet"
        raw_hist = pd.read_parquet(raw_path)
        raw_hist["ts"] = pd.to_datetime(raw_hist["candle_date_time_utc"])
        raw_hist = raw_hist.sort_values("ts").reset_index(drop=True)

        layout  = BitcoinDataHandler(cfg).build_feature_layout()
        seq_len = cfg.sequence_length
        seq_feats = layout["sequence"]
        ctx_feats = layout["context"]

        # rolling window용 buffer (최근 _FEATURE_BUFFER bars)
        raw_buf = raw_hist.tail(_FEATURE_BUFFER).copy().reset_index(drop=True)
        last_ts = raw_buf["ts"].max()
        print(f"[live] Historical buffer up to: {last_ts}")

        # ── 3. 최신 캔들 업데이트 ────────────────────────────────
        print("[live] Fetching new candles since last timestamp...")
        new_raw = _fetch_new_raw(last_ts)
        if not new_raw.empty:
            print(f"[live] {len(new_raw)} new candles fetched.")
            raw_buf = pd.concat([raw_buf, new_raw], ignore_index=True)
            raw_buf = raw_buf.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
            raw_buf = raw_buf.tail(_FEATURE_BUFFER).reset_index(drop=True)
            last_ts = raw_buf["ts"].max()

        # ── 4. feature 계산 ───────────────────────────────────────
        print("[live] Computing features...")
        featured = _compute_features(raw_buf)
        print(f"[live] Features ready. Latest: {featured.iloc[-1]['ts']}")

        # ── 5. 포트폴리오 초기화 ─────────────────────────────────
        cash                 = _INITIAL_CASH
        btc                  = 0.0
        avg_entry            = 0.0
        realized_pnl         = 0.0
        steps_since_rebalance = 0
        step                 = 0
        last_target_ratio    = 0.0

        print("[live] Starting live inference loop.")

        while True:
            # 현재 가격 = featured 마지막 close
            current_price = float(featured["close"].iloc[-1])
            current_ts    = str(featured["ts"].iloc[-1])

            # obs 생성 및 정규화
            obs = _build_obs(
                featured, seq_feats, ctx_feats, seq_len,
                cash, btc, avg_entry, realized_pnl,
                steps_since_rebalance, current_price,
            )
            obs_norm = _normalize_obs(obs, vn)

            # 예측
            action, _ = model.predict(obs_norm[None], deterministic=True)
            action_int    = int(action[0])
            target_ratio  = TARGET_LEVELS[action_int]

            # 체결
            cash, btc, avg_entry, realized_pnl, traded = _rebalance(
                target_ratio, current_price,
                cash, btc, avg_entry, realized_pnl,
            )
            if traded:
                steps_since_rebalance = 0
                last_target_ratio = target_ratio
            else:
                steps_since_rebalance += 1

            total_equity  = cash + btc * current_price
            btc_value     = btc * current_price
            position_ratio = btc_value / total_equity if total_equity > 0 else 0.0
            step += 1

            streamer.emit({
                "type":            "step",
                "step":            step,
                "n_updates":       0,
                "reward":          0.0,
                "action":          action_int,
                "target_ratio":    target_ratio,
                "position_ratio":  position_ratio,
                "equity":          total_equity,
                "cash":            cash,
                "btc_holding":     btc,
                "avg_entry_price": avg_entry,
                "realized_pnl":    realized_pnl,
                "open":            float(featured["open"].iloc[-1]),
                "high":            float(featured["high"].iloc[-1]),
                "low":             float(featured["low"].iloc[-1]),
                "close":           current_price,
                "ts":              current_ts,
                "trade_executed":  traded,
            })

            # ── 다음 5분봉까지 대기 ───────────────────────────────
            wait = _seconds_to_next_bar()
            print(f"[live] step={step} action={action_int}({target_ratio:.0%}) "
                  f"price={current_price:,.0f} equity={total_equity:,.0f} "
                  f"→ sleep {wait:.0f}s")
            time.sleep(wait)

            # ── 새 캔들 수신 및 feature 갱신 ─────────────────────
            new_raw = _fetch_new_raw(last_ts)
            if not new_raw.empty:
                raw_buf = pd.concat([raw_buf, new_raw], ignore_index=True)
                raw_buf = raw_buf.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
                raw_buf = raw_buf.tail(_FEATURE_BUFFER).reset_index(drop=True)
                last_ts = raw_buf["ts"].max()
                featured = _compute_features(raw_buf)
            else:
                print("[live] No new candle yet, retrying next cycle.")
