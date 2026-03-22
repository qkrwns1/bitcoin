"""Multi-timeframe candle data manager.

앱 시작 시 각 타임프레임 parquet를 로드하고, 최신 데이터만 incremental 업데이트.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.upbit.com/v1"
DATA_ROOT = Path(__file__).parent.parent / "data analysis" / "data"

# 타임프레임 설정
TF = {
    "weekly": {
        "path": DATA_ROOT / "raw" / "candles_1w"  / "KRW-BTC_1w.parquet",
        "endpoint": "/candles/weeks",
        "interval_min": 7 * 24 * 60,
        "target": 260,          # ~5년치 주봉
    },
    "daily": {
        "path": DATA_ROOT / "raw" / "candles_days" / "KRW-BTC_days.parquet",
        "endpoint": "/candles/days",
        "interval_min": 24 * 60,
        "target": 365 * 5,      # 5년치 일봉
    },
    "1h": {
        "path": DATA_ROOT / "raw" / "candles_1h"  / "KRW-BTC_1h.parquet",
        "endpoint": "/candles/minutes/60",
        "interval_min": 60,
        "target": 24 * 365 * 3, # 3년치 1시간봉
    },
    "5m": {
        "path": DATA_ROOT / "raw" / "candles_5m"  / "KRW-BTC_5m.parquet",
        "endpoint": "/candles/minutes/5",
        "interval_min": 5,
        "target": 12 * 24 * 365 * 3,  # 3년치 5분봉
    },
}


def _fetch_candles(endpoint: str, count: int, to_utc: str | None = None) -> list[dict]:
    params: dict = {"market": "KRW-BTC", "count": min(count, 200)}
    if to_utc:
        params["to"] = to_utc
    resp = requests.get(BASE_URL + endpoint, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["candle_date_time_utc"] = pd.to_datetime(df["candle_date_time_utc"])
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def update_tf(tf_key: str) -> pd.DataFrame:
    """해당 타임프레임 parquet를 로드하고 최신 데이터만 Upbit에서 받아 업데이트."""
    cfg = TF[tf_key]
    path: Path = cfg["path"]
    endpoint: str = cfg["endpoint"]
    interval_min: int = cfg["interval_min"]
    target: int = cfg["target"]

    existing = _load(path)

    if existing is None:
        # 처음 다운로드
        print(f"[{tf_key}] 최초 수집 시작 (목표: {target:,}개)")
        chunks = []
        remaining = target
        to_utc = None
        req = 0
        while remaining > 0:
            batch = _fetch_candles(endpoint, min(200, remaining), to_utc)
            if not batch:
                break
            chunks.append(pd.DataFrame(batch))
            to_utc = batch[-1]["candle_date_time_utc"]
            remaining -= len(batch)
            req += 1
            if req % 50 == 0:
                print(f"  [{tf_key}] {target - remaining:,} / {target:,}")
            time.sleep(0.12)
            if len(batch) < 200:
                break
        df = pd.concat(chunks, ignore_index=True)
    else:
        # Incremental: 최신 ts 이후만 수집
        latest_utc = existing["candle_date_time_utc"].max()
        missing = int((pd.Timestamp.utcnow().tz_localize(None) - latest_utc).total_seconds() / 60 / interval_min)
        if missing <= 0:
            print(f"[{tf_key}] 이미 최신 상태")
            return existing
        print(f"[{tf_key}] {missing}개 누락 → 업데이트")
        batch = _fetch_candles(endpoint, min(missing + 5, 200))
        df = pd.concat([existing, pd.DataFrame(batch)], ignore_index=True)

    # 중복 제거 + 정렬
    df["candle_date_time_utc"] = pd.to_datetime(df["candle_date_time_utc"])
    df = df.drop_duplicates(subset=["candle_date_time_utc"]).sort_values("candle_date_time_utc").reset_index(drop=True)
    _save(df, path)
    print(f"[{tf_key}] 저장 완료: {len(df):,}행")
    return df


def to_chart_candles(df: pd.DataFrame) -> list[dict]:
    """parquet DataFrame → lightweight-charts용 KST OHLCV 리스트."""
    KST = 9 * 3600
    return [
        {
            "t": int(row.candle_date_time_utc.timestamp()) + KST,
            "o": int(row.opening_price),
            "h": int(row.high_price),
            "l": int(row.low_price),
            "c": int(row.trade_price),
        }
        for row in df.itertuples()
    ]


def update_all() -> dict[str, pd.DataFrame]:
    """앱 시작 시 모든 타임프레임 업데이트."""
    result = {}
    for key in TF:
        try:
            result[key] = update_tf(key)
        except Exception as e:
            print(f"[{key}] 업데이트 실패: {e}")
            existing = _load(TF[key]["path"])
            if existing is not None:
                result[key] = existing
    return result
