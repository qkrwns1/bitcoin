"""FastAPI dashboard server for Bitcoin RL system."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "final_model"
VECNORM_PATH = PROJECT_ROOT / "checkpoints" / "vec_normalize.pkl"

app = FastAPI(title="Bitcoin RL Dashboard")

_clients: list[WebSocket] = []
_runner = None


async def _broadcast(data: dict[str, Any]) -> None:
    dead = []
    for ws in _clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _clients:
            _clients.remove(ws)


@app.on_event("startup")
async def _startup() -> None:
    global _runner

    # Background: update all TF parquets from Upbit
    async def _update_data() -> None:
        from app.data_updater import update_all
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, update_all)

    asyncio.create_task(_update_data())

    from app.runner import AgentRunner

    if MODEL_PATH.with_suffix(".zip").exists() and VECNORM_PATH.exists():
        _runner = AgentRunner(MODEL_PATH, VECNORM_PATH)
        loop = asyncio.get_running_loop()
        asyncio.create_task(_runner.start(_broadcast, loop))
        print(f"[dashboard] Model loaded: {MODEL_PATH}.zip")
    else:
        print(f"[dashboard] Model not found at {MODEL_PATH}.zip")
        print(f"[dashboard] Run training first, then restart the server.")


@app.get("/api/history")
async def api_history(tf: str = "") -> dict:
    """로컬 parquet → OHLCV 반환.

    tf=weekly|daily|1h|5m : 해당 타임프레임 parquet (data_updater)
    tf 없음                : RL processed parquet (멀티 해상도)
    """
    import pandas as pd
    from app.data_updater import TF, _load, to_chart_candles

    if tf in TF:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, _load, TF[tf]["path"])
        if df is None or df.empty:
            return {"candles": []}
        candles = to_chart_candles(df)
        return {"candles": candles}

    # --- RL market frame fallback (멀티 해상도) ---
    parquet_path = PROJECT_ROOT / "data analysis" / "data" / "processed" / "rl" / "rl_market_frame_1m.parquet"
    if not parquet_path.exists():
        return {"candles": []}

    df = pd.read_parquet(parquet_path, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    cutoff = df["ts"].max() - pd.Timedelta(days=3)

    old = df[df["ts"] < cutoff].copy()
    if not old.empty:
        old = old.set_index("ts").resample("1h").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
        ).dropna().reset_index()

    recent = df[df["ts"] >= cutoff].tail(5_000)
    combined = pd.concat([old, recent], ignore_index=True).sort_values("ts")

    kst_offset = 9 * 3600
    candles = [
        {
            "t": int(row.ts.timestamp()) + kst_offset,
            "o": round(float(row.open), 0),
            "h": round(float(row.high), 0),
            "l": round(float(row.low), 0),
            "c": round(float(row.close), 0),
        }
        for row in combined.itertuples()
    ]
    return {"candles": candles}


@app.get("/api/candles")
async def api_candles(pages: int = 1) -> list:
    """실시간 최신 캔들 (1분봉, 최대 200개)"""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://api.upbit.com/v1/candles/minutes/1",
            params={"market": "KRW-BTC", "count": 200},
        )
    return r.json()


@app.get("/api/status")
async def api_status() -> dict:
    return {
        "model_loaded": _runner is not None,
        "is_running": _runner.is_running if _runner else False,
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _clients:
            _clients.remove(websocket)


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
