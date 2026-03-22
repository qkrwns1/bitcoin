from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


BASE_URL = "https://api.upbit.com/v1"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data analysis" / "data"
MINUTE_PATH = DATA_ROOT / "raw" / "candles_1m" / "KRW-BTC_minutes_1m.parquet"
DAY_PATH = DATA_ROOT / "raw" / "candles_days" / "KRW-BTC_days.parquet"

# 1년치 1분봉 = 365 * 24 * 60 = 525,600
MINUTE_TARGET = 365 * 24 * 60
DAY_TARGET = 365 * 3  # 3년치 일봉 (52주 피처용)


def _request_json(url: str, params: dict) -> list[dict]:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _load_existing_minute() -> Optional[pd.DataFrame]:
    if not MINUTE_PATH.exists():
        return None
    df = pd.read_parquet(MINUTE_PATH)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def _load_existing_day() -> Optional[pd.DataFrame]:
    if not DAY_PATH.exists():
        return None
    df = pd.read_parquet(DAY_PATH)
    df["date_kst"] = pd.to_datetime(df["date_kst"])
    return df


def fetch_minute_candles(
    market: str = "KRW-BTC",
    total_count: int = MINUTE_TARGET,
    to_value: Optional[str] = None,
) -> pd.DataFrame:
    """과거 방향으로 total_count개 1분봉을 수집한다. to_value 지정 시 그 시점부터 이전으로 진행."""
    url = f"{BASE_URL}/candles/minutes/1"
    remaining = total_count
    chunks: list[pd.DataFrame] = []
    req_count = 0

    while remaining > 0:
        batch = min(200, remaining)
        params = {"market": market, "count": batch}
        if to_value:
            params["to"] = to_value

        payload = _request_json(url, params)
        if not payload:
            break

        frame = pd.DataFrame(payload)
        chunks.append(frame)

        to_value = frame["candle_date_time_utc"].iloc[-1]
        remaining -= len(frame)
        req_count += 1

        if req_count % 100 == 0:
            print(f"  1m 수집 중: {total_count - remaining:,} / {total_count:,}")

        time.sleep(0.12)

        if len(frame) < batch:
            print("  API에서 더 이상 데이터 없음 (1m)")
            break

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["candle_date_time_kst"]).copy()
    df["ts"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def fetch_day_candles(
    market: str = "KRW-BTC",
    total_count: int = DAY_TARGET,
    to_value: Optional[str] = None,
) -> pd.DataFrame:
    url = f"{BASE_URL}/candles/days"
    remaining = total_count
    chunks: list[pd.DataFrame] = []

    while remaining > 0:
        batch = min(200, remaining)
        params = {"market": market, "count": batch}
        if to_value:
            params["to"] = to_value

        payload = _request_json(url, params)
        if not payload:
            break

        frame = pd.DataFrame(payload)
        chunks.append(frame)

        to_value = frame["candle_date_time_utc"].iloc[-1]
        remaining -= len(frame)
        time.sleep(0.12)

        if len(frame) < batch:
            print("  API에서 더 이상 데이터 없음 (day)")
            break

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["candle_date_time_kst"]).copy()
    df["date_kst"] = pd.to_datetime(df["candle_date_time_kst"]).dt.normalize()
    df = df.sort_values("date_kst").reset_index(drop=True)
    return df


def update_minute_candles(market: str = "KRW-BTC") -> pd.DataFrame:
    """기존 파일이 있으면 앞뒤로 채우고, 없으면 전체 수집."""
    existing = _load_existing_minute()

    if existing is None:
        print(f"기존 1분봉 없음 → 전체 {MINUTE_TARGET:,}행 수집 시작")
        return fetch_minute_candles(total_count=MINUTE_TARGET)

    chunks = [existing]

    # 1) 최신 방향 이어받기
    latest_ts = existing["ts"].max()
    minutes_missing = int((pd.Timestamp.now() - latest_ts).total_seconds() / 60)
    if minutes_missing > 1:
        print(f"1분봉 최신 이어받기: {minutes_missing:,}개 누락 ({latest_ts} 이후)")
        new_df = fetch_minute_candles(total_count=minutes_missing + 10)
        chunks.append(new_df)

    # 2) 과거 방향 채우기 (목표치에 부족할 경우)
    combined = pd.concat(chunks, ignore_index=True)
    combined["ts"] = pd.to_datetime(combined["ts"])
    combined = combined.drop_duplicates(subset=["candle_date_time_kst"])
    combined = combined.sort_values("ts").reset_index(drop=True)

    if len(combined) < MINUTE_TARGET:
        oldest_ts = combined["ts"].min()
        need = MINUTE_TARGET - len(combined)
        oldest_utc = oldest_ts.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"1분봉 과거 채우기: {need:,}행 필요 ({oldest_ts} 이전)")
        hist_df = fetch_minute_candles(total_count=need + 10, to_value=oldest_utc)
        combined = pd.concat([combined, hist_df], ignore_index=True)
        combined["ts"] = pd.to_datetime(combined["ts"])
        combined = combined.drop_duplicates(subset=["candle_date_time_kst"])
        combined = combined.sort_values("ts").reset_index(drop=True)

    return combined


def update_day_candles(market: str = "KRW-BTC") -> pd.DataFrame:
    """기존 일봉 파일이 있으면 최신 데이터만 이어받고, 없으면 전체 수집."""
    existing = _load_existing_day()

    if existing is None:
        print(f"기존 일봉 없음 → 전체 {DAY_TARGET}일 수집 시작")
        return fetch_day_candles(total_count=DAY_TARGET)

    latest_date = existing["date_kst"].max()
    days_missing = int((pd.Timestamp.now().normalize() - latest_date).days)

    if days_missing <= 0:
        print(f"일봉 이미 최신 상태 ({latest_date.date()})")
        return existing

    print(f"일봉 이어받기: {days_missing}일 누락")
    new_df = fetch_day_candles(total_count=days_missing + 5)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined["date_kst"] = pd.to_datetime(combined["date_kst"])
    combined = combined.drop_duplicates(subset=["candle_date_time_kst"]).copy()
    combined = combined.sort_values("date_kst").reset_index(drop=True)
    return combined


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MINUTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DAY_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== 일봉 수집 ===")
    day_df = update_day_candles()
    day_df.to_parquet(DAY_PATH, index=False)
    print(f"저장 완료: {DAY_PATH}")
    print(f"일봉 {len(day_df)}행 | {day_df['date_kst'].min().date()} ~ {day_df['date_kst'].max().date()}")

    print("\n=== 1분봉 수집 ===")
    minute_df = update_minute_candles()
    minute_df.to_parquet(MINUTE_PATH, index=False)
    print(f"저장 완료: {MINUTE_PATH}")
    print(f"1분봉 {len(minute_df):,}행 | {minute_df['ts'].min()} ~ {minute_df['ts'].max()}")


if __name__ == "__main__":
    main()
