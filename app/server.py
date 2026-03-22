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
    from app.runner import AgentRunner

    if MODEL_PATH.with_suffix(".zip").exists() and VECNORM_PATH.exists():
        _runner = AgentRunner(MODEL_PATH, VECNORM_PATH)
        loop = asyncio.get_running_loop()
        asyncio.create_task(_runner.start(_broadcast, loop))
        print(f"[dashboard] Model loaded: {MODEL_PATH}.zip")
    else:
        print(f"[dashboard] Model not found at {MODEL_PATH}.zip")
        print(f"[dashboard] Run training first, then restart the server.")


@app.get("/api/candles")
async def api_candles(count: int = 200) -> list:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://api.upbit.com/v1/candles/minutes/1",
            params={"market": "KRW-BTC", "count": min(count, 200)},
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
