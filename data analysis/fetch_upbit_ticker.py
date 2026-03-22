from __future__ import annotations

import json
from pathlib import Path

import duckdb
from websocket import create_connection


WS_URL = "wss://api.upbit.com/websocket/v1"
DB_PATH = Path(__file__).resolve().parent / "data" / "db" / "research.duckdb"
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw" / "ticker"


def ensure_storage(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS upbit_ticker_raw (
            type VARCHAR,
            code VARCHAR,
            opening_price DOUBLE,
            high_price DOUBLE,
            low_price DOUBLE,
            trade_price DOUBLE,
            prev_closing_price DOUBLE,
            acc_trade_price DOUBLE,
            change VARCHAR,
            change_price DOUBLE,
            signed_change_price DOUBLE,
            change_rate DOUBLE,
            signed_change_rate DOUBLE,
            ask_bid VARCHAR,
            trade_volume DOUBLE,
            acc_trade_volume DOUBLE,
            trade_date VARCHAR,
            trade_time VARCHAR,
            trade_timestamp BIGINT,
            acc_ask_volume DOUBLE,
            acc_bid_volume DOUBLE,
            highest_52_week_price DOUBLE,
            highest_52_week_date VARCHAR,
            lowest_52_week_price DOUBLE,
            lowest_52_week_date VARCHAR,
            market_state VARCHAR,
            is_trading_suspended BOOLEAN,
            delisting_date VARCHAR,
            market_warning VARCHAR,
            timestamp BIGINT,
            acc_trade_price_24h DOUBLE,
            acc_trade_volume_24h DOUBLE,
            stream_type VARCHAR
        )
        """
    )


def fetch_one_ticker(code: str = "KRW-BTC") -> dict:
    ws = create_connection(WS_URL, timeout=10)
    try:
        request = [
            {"ticket": "codex-upbit-ticker"},
            {"type": "ticker", "codes": [code]},
            {"format": "DEFAULT"},
        ]
        ws.send(json.dumps(request))
        message = ws.recv()
    finally:
        ws.close()

    if isinstance(message, bytes):
        return json.loads(message.decode("utf-8"))
    return json.loads(message)


def save_raw_json(payload: dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = payload["timestamp"]
    path = RAW_DIR / f"{payload['code']}_{suffix}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def insert_payload(con: duckdb.DuckDBPyConnection, payload: dict) -> None:
    con.execute(
        """
        INSERT INTO upbit_ticker_raw VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            payload.get("type"),
            payload.get("code"),
            payload.get("opening_price"),
            payload.get("high_price"),
            payload.get("low_price"),
            payload.get("trade_price"),
            payload.get("prev_closing_price"),
            payload.get("acc_trade_price"),
            payload.get("change"),
            payload.get("change_price"),
            payload.get("signed_change_price"),
            payload.get("change_rate"),
            payload.get("signed_change_rate"),
            payload.get("ask_bid"),
            payload.get("trade_volume"),
            payload.get("acc_trade_volume"),
            payload.get("trade_date"),
            payload.get("trade_time"),
            payload.get("trade_timestamp"),
            payload.get("acc_ask_volume"),
            payload.get("acc_bid_volume"),
            payload.get("highest_52_week_price"),
            payload.get("highest_52_week_date"),
            payload.get("lowest_52_week_price"),
            payload.get("lowest_52_week_date"),
            payload.get("market_state"),
            payload.get("is_trading_suspended"),
            payload.get("delisting_date"),
            payload.get("market_warning"),
            payload.get("timestamp"),
            payload.get("acc_trade_price_24h"),
            payload.get("acc_trade_volume_24h"),
            payload.get("stream_type"),
        ],
    )


def main() -> None:
    payload = fetch_one_ticker()
    json_path = save_raw_json(payload)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        ensure_storage(con)
        insert_payload(con, payload)
    finally:
        con.close()

    print(f"Saved JSON: {json_path}")
    print(f"Saved DB row in: {DB_PATH}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
