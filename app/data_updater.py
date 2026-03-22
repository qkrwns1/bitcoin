"""Multi-timeframe candle data manager.

앱 시작 시 각 타임프레임 parquet를 로드하고,
2020-01-01 이전 누락분(backfill) + 최신 누락분(forward) 모두 업데이트.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL  = "https://api.upbit.com/v1"
DATA_ROOT = Path(__file__).parent.parent / "data analysis" / "data"

# 수집 시작 기준일 (UTC)
START_DATE = pd.Timestamp("2020-01-01")

# 타임프레임 설정
TF = {
    "weekly": {
        "path":         DATA_ROOT / "raw" / "candles_1w"   / "KRW-BTC_1w.parquet",
        "endpoint":     "/candles/weeks",
        "interval_min": 7 * 24 * 60,
    },
    "daily": {
        "path":         DATA_ROOT / "raw" / "candles_days"  / "KRW-BTC_days.parquet",
        "endpoint":     "/candles/days",
        "interval_min": 24 * 60,
    },
    "1h": {
        "path":         DATA_ROOT / "raw" / "candles_1h"   / "KRW-BTC_1h.parquet",
        "endpoint":     "/candles/minutes/60",
        "interval_min": 60,
    },
    "5m": {
        "path":         DATA_ROOT / "raw" / "candles_5m"   / "KRW-BTC_5m.parquet",
        "endpoint":     "/candles/minutes/5",
        "interval_min": 5,
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


def _fetch_backwards(endpoint: str, from_utc: str | None, label: str) -> list[pd.DataFrame]:
    """START_DATE까지 과거 방향으로 페이지네이션."""
    chunks: list[pd.DataFrame] = []
    to_utc = from_utc
    req = 0
    while True:
        batch = _fetch_candles(endpoint, 200, to_utc)
        if not batch:
            break
        df_b = pd.DataFrame(batch)
        df_b["candle_date_time_utc"] = pd.to_datetime(df_b["candle_date_time_utc"])
        chunks.append(df_b)

        oldest_in_batch = df_b["candle_date_time_utc"].min()
        req += 1
        if req % 50 == 0:
            print(f"  [{label}] 백필 중... 현재 {oldest_in_batch.date()}")

        if oldest_in_batch <= START_DATE:
            break

        to_utc = batch[-1]["candle_date_time_utc"]
        time.sleep(0.12)

        if len(batch) < 200:
            break

    return chunks


def update_tf(tf_key: str) -> pd.DataFrame:
    """2020-01-01 ~ 현재까지 해당 타임프레임 parquet를 완전히 채움.

    - 최초 실행: 현재 → 2020-01-01 방향으로 전체 다운로드
    - 재실행:
        * backfill  : 기존 최솟값 > 2020-01-01 이면 과거 방향으로 추가 수집
        * forward   : 기존 최댓값 < 현재 이면 최신 방향으로 추가 수집
    """
    cfg      = TF[tf_key]
    path: Path       = cfg["path"]
    endpoint: str    = cfg["endpoint"]
    interval_min: int = cfg["interval_min"]

    existing = _load(path)
    chunks: list[pd.DataFrame] = []

    if existing is not None:
        chunks.append(existing)

    # ── 1. Backfill: 2020-01-01보다 오래된 데이터가 없으면 과거로 수집 ──
    oldest = existing["candle_date_time_utc"].min() if existing is not None else None
    if oldest is None or oldest > START_DATE:
        from_utc = oldest.strftime("%Y-%m-%dT%H:%M:%S") if oldest is not None else None
        print(f"[{tf_key}] 백필 시작: {from_utc or '현재'} → {START_DATE.date()}")
        back_chunks = _fetch_backwards(endpoint, from_utc, tf_key)
        chunks.extend(back_chunks)

    # ── 2. Forward: 최신 데이터가 빠져 있으면 앞으로 수집 ──
    if existing is not None:
        latest_utc = existing["candle_date_time_utc"].max()
        missing = int(
            (pd.Timestamp.utcnow().tz_localize(None) - latest_utc).total_seconds()
            / 60 / interval_min
        )
        if missing > 0:
            print(f"[{tf_key}] 최신 {missing}개 누락 → forward 수집")
            batch = _fetch_candles(endpoint, min(missing + 5, 200))
            if batch:
                chunks.append(pd.DataFrame(batch))

    if not chunks:
        raise RuntimeError(f"[{tf_key}] 수집된 데이터가 없습니다")

    df = pd.concat(chunks, ignore_index=True)
    df["candle_date_time_utc"] = pd.to_datetime(df["candle_date_time_utc"])
    df = (
        df.drop_duplicates(subset=["candle_date_time_utc"])
          .sort_values("candle_date_time_utc")
          .reset_index(drop=True)
    )
    _save(df, path)
    print(f"[{tf_key}] 저장 완료: {len(df):,}행  "
          f"({df['candle_date_time_utc'].min().date()} ~ {df['candle_date_time_utc'].max().date()})")
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
