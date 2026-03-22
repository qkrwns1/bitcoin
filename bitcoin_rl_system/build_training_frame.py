from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


DATA_ROOT = Path(__file__).resolve().parent.parent / "data analysis" / "data"
MINUTE_PATH = DATA_ROOT / "raw" / "candles_1m" / "KRW-BTC_minutes_1m.parquet"
DAY_PATH = DATA_ROOT / "raw" / "candles_days" / "KRW-BTC_days.parquet"
OUTPUT_PATH = DATA_ROOT / "processed" / "rl" / "rl_market_frame_1m.parquet"
DB_PATH = DATA_ROOT / "db" / "research.duckdb"


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(2, window // 4)).mean()
    std = series.rolling(window, min_periods=max(2, window // 4)).std()
    return ((series - mean) / std.replace(0, np.nan)).fillna(0.0)


def _days_since(current_dates: pd.Series, event_dates: pd.Series) -> pd.Series:
    delta = current_dates - pd.to_datetime(event_dates)
    return delta.dt.days.fillna(0).clip(lower=0)


def build_frame() -> pd.DataFrame:
    minute_df = pd.read_parquet(MINUTE_PATH).copy()
    day_df = pd.read_parquet(DAY_PATH).copy()

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(minute_df["ts"]),
            "market": minute_df["market"],
            "open": minute_df["opening_price"].astype(float),
            "high": minute_df["high_price"].astype(float),
            "low": minute_df["low_price"].astype(float),
            "close": minute_df["trade_price"].astype(float),
            "volume": minute_df["candle_acc_trade_volume"].astype(float),
            "trade_value": minute_df["candle_acc_trade_price"].astype(float),
        }
    )

    df["date_kst"] = df["ts"].dt.normalize()
    prev_daily_close = day_df[["date_kst", "prev_closing_price"]].copy()
    prev_daily_close["date_kst"] = pd.to_datetime(prev_daily_close["date_kst"])
    df = df.merge(prev_daily_close, on="date_kst", how="left")

    df["return_1m"] = df["close"].pct_change().fillna(0.0)
    df["log_return_1m"] = np.log(df["close"] / df["close"].shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["return_5m"] = df["close"].pct_change(5).fillna(0.0)
    df["return_15m"] = df["close"].pct_change(15).fillna(0.0)

    df["body_ratio"] = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
    df["range_ratio"] = (df["high"] - df["low"]) / df["open"]

    df["volume_zscore_30m"] = _zscore(df["volume"], 30)
    df["volume_zscore_60m"] = _zscore(df["volume"], 60)
    df["volume_zscore_120m"] = _zscore(df["volume"], 120)

    df["volatility_15m"] = df["return_1m"].rolling(15, min_periods=5).std().fillna(0.0)
    df["volatility_60m"] = df["return_1m"].rolling(60, min_periods=15).std().fillna(0.0)

    for window, label in [(15, "15m"), (30, "30m"), (60, "60m"), (240, "240m")]:
        rolling_high = df["high"].rolling(window, min_periods=max(2, window // 3)).max()
        rolling_low = df["low"].rolling(window, min_periods=max(2, window // 3)).min()
        df[f"dist_to_{label}_high"] = (df["close"] / rolling_high - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df[f"dist_to_{label}_low"] = (df["close"] / rolling_low - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    daily_high = df.groupby("date_kst")["high"].cummax()
    daily_low = df.groupby("date_kst")["low"].cummin()
    df["dist_to_daily_high"] = (df["close"] / daily_high - 1.0).fillna(0.0)
    df["dist_to_daily_low"] = (df["close"] / daily_low - 1.0).fillna(0.0)

    df["acc_trade_volume_24h"] = df["volume"].rolling(1440, min_periods=60).sum().bfill().fillna(df["volume"])
    df["acc_trade_price_24h"] = df["trade_value"].rolling(1440, min_periods=60).sum().bfill().fillna(df["trade_value"])
    df["change_rate"] = ((df["close"] / df["prev_closing_price"]) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["signed_change_rate"] = df["change_rate"]

    day_features = day_df[[
        "date_kst",
        "high_price",
        "low_price",
        "trade_price",
        "candle_acc_trade_volume",
        "candle_acc_trade_price",
    ]].copy()
    day_features["date_kst"] = pd.to_datetime(day_features["date_kst"])
    day_features = day_features.sort_values("date_kst").reset_index(drop=True)

    # shift(1): 당일 고/저가 제외 — 장 중 관측 시점에 당일 최종값은 미래 데이터
    day_features["highest_52_week_price"] = day_features["high_price"].shift(1).rolling(365, min_periods=30).max()
    day_features["lowest_52_week_price"] = day_features["low_price"].shift(1).rolling(365, min_periods=30).min()
    day_features["highest_52_week_date"] = pd.NaT
    day_features["lowest_52_week_date"] = pd.NaT

    for idx in range(len(day_features)):
        # idx 행(당일)은 제외 — 당일 고/저가는 장 중에 미래 데이터를 포함하므로 leakage 방지
        start = max(0, idx - 364)
        window = day_features.iloc[start:idx]
        if window.empty:
            continue
        day_features.at[idx, "highest_52_week_date"] = window.loc[window["high_price"].idxmax(), "date_kst"]
        day_features.at[idx, "lowest_52_week_date"] = window.loc[window["low_price"].idxmin(), "date_kst"]

    df = df.merge(
        day_features[[
            "date_kst",
            "highest_52_week_price",
            "lowest_52_week_price",
            "highest_52_week_date",
            "lowest_52_week_date",
        ]],
        on="date_kst",
        how="left",
    )

    df["dist_to_52w_high"] = (df["close"] / df["highest_52_week_price"] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["dist_to_52w_low"] = (df["close"] / df["lowest_52_week_price"] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["days_since_52w_high"] = _days_since(df["date_kst"], df["highest_52_week_date"])
    df["days_since_52w_low"] = _days_since(df["date_kst"], df["lowest_52_week_date"])
    range_den = (df["highest_52_week_price"] - df["lowest_52_week_price"]).replace(0, np.nan)
    df["position_in_52w_range"] = ((df["close"] - df["lowest_52_week_price"]) / range_den).clip(0, 1).fillna(0.5)

    df["market_state"] = "ACTIVE"
    df["market_warning"] = "NONE"
    df["is_trading_suspended"] = 0
    df["ask_bid"] = np.where(df["close"] >= df["open"], "BID", "ASK")

    keep_columns = [
        "ts",
        "market",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_value",
        "return_1m",
        "log_return_1m",
        "return_5m",
        "return_15m",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "range_ratio",
        "volume_zscore_30m",
        "volume_zscore_60m",
        "volume_zscore_120m",
        "volatility_15m",
        "volatility_60m",
        "dist_to_15m_high",
        "dist_to_15m_low",
        "dist_to_30m_high",
        "dist_to_30m_low",
        "dist_to_60m_high",
        "dist_to_60m_low",
        "dist_to_240m_high",
        "dist_to_240m_low",
        "dist_to_daily_high",
        "dist_to_daily_low",
        "dist_to_52w_high",
        "dist_to_52w_low",
        "days_since_52w_high",
        "days_since_52w_low",
        "position_in_52w_range",
        "acc_trade_volume_24h",
        "acc_trade_price_24h",
        "change_rate",
        "signed_change_rate",
        "market_state",
        "market_warning",
        "is_trading_suspended",
        "ask_bid",
    ]
    return df[keep_columns].copy()


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = build_frame()
    frame.to_parquet(OUTPUT_PATH, index=False)

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE rl_market_frame_1m AS SELECT * FROM read_parquet(?)", [str(OUTPUT_PATH)])
    finally:
        con.close()

    print(f"Saved training frame: {OUTPUT_PATH}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")
    print(f"Range: {frame['ts'].min()} -> {frame['ts'].max()}")
    print(frame.head(3).to_string())


if __name__ == "__main__":
    main()
