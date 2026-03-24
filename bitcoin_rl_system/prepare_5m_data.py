"""5분봉 RL 학습 데이터 전처리 스크립트.

raw 5m 캔들 → 피처 계산 → rl_market_frame_5m.parquet 저장
실행: python bitcoin_rl_system/prepare_5m_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data analysis" / "data"
RAW_PATH  = DATA_ROOT / "raw" / "candles_5m" / "KRW-BTC_5m.parquet"
OUT_PATH  = DATA_ROOT / "processed" / "rl" / "rl_market_frame_5m.parquet"

# 1 bar = 5분
BAR_3   = 3     # 15분
BAR_12  = 12    # 1시간
BAR_24  = 24    # 2시간
BAR_48  = 48    # 4시간
BAR_288 = 288   # 24시간


def _zscore(s: pd.Series, window: int) -> pd.Series:
    m   = s.rolling(window, min_periods=1).mean()
    std = s.rolling(window, min_periods=1).std().replace(0, np.nan)
    return ((s - m) / std).fillna(0.0)


def _dist_to_high(c: pd.Series, window: int) -> pd.Series:
    h = c.rolling(window, min_periods=1).max()
    return ((c - h) / h.replace(0, np.nan)).fillna(0.0)


def _dist_to_low(c: pd.Series, window: int) -> pd.Series:
    lo = c.rolling(window, min_periods=1).min()
    return ((c - lo) / lo.replace(0, np.nan)).fillna(0.0)


def _bars_since_new_extreme(c: pd.Series, window: int, is_max: bool) -> pd.Series:
    """O(n) — 마지막으로 rolling 극값을 갱신한 이후 bar 수."""
    roll = (c.rolling(window, min_periods=1).max() if is_max
            else c.rolling(window, min_periods=1).min())
    # 극값 갱신 시점: 현재 close == rolling 극값
    is_extreme = (c >= roll) if is_max else (c <= roll)
    vals = is_extreme.values
    result = np.zeros(len(vals), dtype=np.float32)
    cnt = 0
    for i, v in enumerate(vals):
        cnt = 0 if v else cnt + 1
        result[i] = cnt
    return pd.Series(result, index=c.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["ts"]          = df["ts"]
    out["market"]      = df["market"]
    out["open"]        = df["opening_price"]
    out["high"]        = df["high_price"]
    out["low"]         = df["low_price"]
    out["close"]       = df["trade_price"]
    out["volume"]      = df["candle_acc_trade_volume"]
    out["trade_value"] = df["candle_acc_trade_price"]

    c = out["close"]
    o = out["open"]

    print("  수익률 계산...")
    out["return_5m"]     = c.pct_change(1).fillna(0.0)
    out["log_return_5m"] = np.log(c / c.shift(1)).fillna(0.0)
    out["return_15m"]    = c.pct_change(BAR_3).fillna(0.0)
    out["return_1h"]     = c.pct_change(BAR_12).fillna(0.0)

    print("  캔들 형태...")
    body   = (c - o).abs()
    cr     = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_ratio"]       = (body / cr).fillna(0.0)
    out["upper_wick_ratio"] = ((out["high"] - c.clip(lower=o)) / cr).fillna(0.0)
    out["lower_wick_ratio"] = ((c.clip(upper=o) - out["low"]) / cr).fillna(0.0)
    out["range_ratio"]      = (cr / c.replace(0, np.nan)).fillna(0.0)

    print("  거래량 z-score...")
    out["volume_zscore_1h"] = _zscore(out["volume"], BAR_12)
    out["volume_zscore_2h"] = _zscore(out["volume"], BAR_24)
    out["volume_zscore_4h"] = _zscore(out["volume"], BAR_48)

    print("  변동성...")
    ret = out["return_5m"]
    out["volatility_15m"] = ret.rolling(BAR_3,  min_periods=1).std().fillna(0.0)
    out["volatility_1h"]  = ret.rolling(BAR_12, min_periods=1).std().fillna(0.0)

    print("  단기 고/저가 거리...")
    out["dist_to_15m_high"] = _dist_to_high(c, BAR_3)
    out["dist_to_15m_low"]  = _dist_to_low(c,  BAR_3)
    out["dist_to_1h_high"]  = _dist_to_high(c, BAR_12)
    out["dist_to_1h_low"]   = _dist_to_low(c,  BAR_12)
    out["dist_to_4h_high"]  = _dist_to_high(c, BAR_48)
    out["dist_to_4h_low"]   = _dist_to_low(c,  BAR_48)
    out["dist_to_1d_high"]  = _dist_to_high(c, BAR_288)
    out["dist_to_1d_low"]   = _dist_to_low(c,  BAR_288)

    # ── 컨텍스트 피처 ─────────────────────────────────────────
    print("  일간 컨텍스트...")
    out["dist_to_daily_high"] = _dist_to_high(c, BAR_288)
    out["dist_to_daily_low"]  = _dist_to_low(c,  BAR_288)

    # 52주 = 365일 = 365 * BAR_288 bar
    BAR_52W = 365 * BAR_288
    print(f"  52주 rolling (window={BAR_52W:,})...")
    high_52w = c.rolling(BAR_52W, min_periods=1).max()
    low_52w  = c.rolling(BAR_52W, min_periods=1).min()

    out["dist_to_52w_high"]      = ((c - high_52w) / high_52w.replace(0, np.nan)).fillna(0.0)
    out["dist_to_52w_low"]       = ((c - low_52w)  / low_52w.replace(0, np.nan)).fillna(0.0)
    out["position_in_52w_range"] = (
        (c - low_52w) / (high_52w - low_52w).replace(0, np.nan)
    ).fillna(0.5)

    print("  52주 고/저 경과일...")
    out["days_since_52w_high"] = _bars_since_new_extreme(c, BAR_52W, is_max=True)  / BAR_288
    out["days_since_52w_low"]  = _bars_since_new_extreme(c, BAR_52W, is_max=False) / BAR_288

    print("  24h 누적 거래량...")
    out["acc_trade_volume_24h"] = out["volume"].rolling(BAR_288, min_periods=1).sum()
    out["acc_trade_price_24h"]  = out["trade_value"].rolling(BAR_288, min_periods=1).sum()

    print("  일간 변화율...")
    # 하루 288봉 중 첫 봉의 시가를 일간 기준으로 사용
    daily_open = out["open"].rolling(BAR_288, min_periods=1).apply(
        lambda x: x[0], raw=True
    )
    out["change_rate"]        = ((c - daily_open) / daily_open.replace(0, np.nan)).fillna(0.0)
    out["signed_change_rate"] = out["change_rate"]

    # 마켓 상태 (역사 데이터엔 없으므로 기본값)
    out["market_state"]         = "ACTIVE"
    out["market_warning"]       = "NONE"
    out["is_trading_suspended"] = 0
    out["ask_bid"]              = "BID"

    return out


def main() -> None:
    print(f"원본 로드: {RAW_PATH}")
    raw = pd.read_parquet(RAW_PATH)
    raw["ts"] = pd.to_datetime(raw["candle_date_time_utc"])
    raw = raw.sort_values("ts").reset_index(drop=True)

    start_date = pd.Timestamp("2020-01-01")
    raw = raw[raw["ts"] >= start_date].reset_index(drop=True)
    print(f"2020-01-01 이후: {len(raw):,}행  ({raw['ts'].iloc[0]} ~ {raw['ts'].iloc[-1]})")

    print("피처 계산 시작...")
    df = build_features(raw)
    df = df.dropna().reset_index(drop=True)
    print(f"NaN 제거 후: {len(df):,}행")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n저장 완료: {OUT_PATH}")
    print(f"컬럼 {len(df.columns)}개:")
    for col in df.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()
