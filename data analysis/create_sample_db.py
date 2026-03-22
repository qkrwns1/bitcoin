from pathlib import Path

import duckdb


def main() -> None:
    db_path = Path(__file__).resolve().parent / "data" / "db" / "research.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))

    con.execute("DROP TABLE IF EXISTS raw_ticker")
    con.execute("DROP TABLE IF EXISTS raw_candles_1s")
    con.execute("DROP TABLE IF EXISTS raw_trades")
    con.execute("DROP TABLE IF EXISTS features_1s")

    con.execute(
        """
        CREATE TABLE raw_ticker (
            type VARCHAR,
            code VARCHAR,
            opening_price DOUBLE,
            high_price DOUBLE,
            low_price DOUBLE,
            trade_price DOUBLE,
            prev_closing_price DOUBLE,
            change VARCHAR,
            change_price DOUBLE,
            signed_change_price DOUBLE,
            change_rate DOUBLE,
            signed_change_rate DOUBLE,
            trade_volume DOUBLE,
            acc_trade_volume DOUBLE,
            acc_trade_volume_24h DOUBLE,
            acc_trade_price DOUBLE,
            acc_trade_price_24h DOUBLE,
            trade_date VARCHAR,
            trade_time VARCHAR,
            trade_timestamp BIGINT,
            ask_bid VARCHAR,
            acc_ask_volume DOUBLE,
            acc_bid_volume DOUBLE,
            highest_52_week_price DOUBLE,
            highest_52_week_date DATE,
            lowest_52_week_price DOUBLE,
            lowest_52_week_date DATE,
            market_state VARCHAR,
            is_trading_suspended BOOLEAN,
            delisting_date DATE,
            market_warning VARCHAR,
            timestamp BIGINT,
            stream_type VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE raw_candles_1s (
            type VARCHAR,
            code VARCHAR,
            candle_date_time_utc TIMESTAMP,
            candle_date_time_kst TIMESTAMP,
            opening_price DOUBLE,
            high_price DOUBLE,
            low_price DOUBLE,
            trade_price DOUBLE,
            candle_acc_trade_volume DOUBLE,
            candle_acc_trade_price DOUBLE,
            timestamp BIGINT,
            stream_type VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE raw_trades (
            type VARCHAR,
            code VARCHAR,
            trade_price DOUBLE,
            trade_volume DOUBLE,
            ask_bid VARCHAR,
            prev_closing_price DOUBLE,
            change VARCHAR,
            change_price DOUBLE,
            trade_date DATE,
            trade_time TIME,
            trade_timestamp BIGINT,
            timestamp BIGINT,
            sequential_id BIGINT,
            best_ask_price DOUBLE,
            best_ask_size DOUBLE,
            best_bid_price DOUBLE,
            best_bid_size DOUBLE,
            stream_type VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE features_1s (
            ts TIMESTAMP,
            market VARCHAR,
            close DOUBLE,
            return_1s DOUBLE,
            return_3s DOUBLE,
            volume DOUBLE,
            trade_count_1s INTEGER,
            buy_volume_1s DOUBLE,
            sell_volume_1s DOUBLE,
            order_flow_imbalance DOUBLE,
            volatility_3s DOUBLE
        )
        """
    )

    con.execute(
        """
        INSERT INTO raw_ticker VALUES
        (
            'ticker', 'KRW-BTC', 128500000, 128540000, 128480000, 128530000,
            128100000, 'RISE', 430000, 430000, 0.0033567525, 0.0033567525,
            0.021, 245.37, 1312.82, 31500299120.0, 168420001211.0,
            '20260322', '000001', 1774137601000, 'BID', 120.11, 125.26,
            163325000, '2025-01-20', 82000000, '2025-09-10',
            'ACTIVE', false, null, 'NONE', 1774137601021, 'REALTIME'
        ),
        (
            'ticker', 'KRW-BTC', 128500000, 128550000, 128490000, 128495000,
            128100000, 'RISE', 395000, 395000, 0.0030835285, 0.0030835285,
            0.011, 245.38, 1312.83, 31501712570.0, 168421414920.0,
            '20260322', '000002', 1774137602000, 'ASK', 120.12, 125.27,
            163325000, '2025-01-20', 82000000, '2025-09-10',
            'ACTIVE', false, null, 'NONE', 1774137602028, 'REALTIME'
        )
        """
    )
    con.execute(
        """
        INSERT INTO raw_candles_1s VALUES
        (
            'candle.1s', 'KRW-BTC', '2026-03-22 00:00:00', '2026-03-22 09:00:00',
            128500000, 128520000, 128480000, 128510000, 1.42, 182484200, 1774137600000, 'REALTIME'
        ),
        (
            'candle.1s', 'KRW-BTC', '2026-03-22 00:00:01', '2026-03-22 09:00:01',
            128510000, 128540000, 128500000, 128530000, 0.91, 116962300, 1774137601000, 'REALTIME'
        ),
        (
            'candle.1s', 'KRW-BTC', '2026-03-22 00:00:02', '2026-03-22 09:00:02',
            128530000, 128530000, 128490000, 128495000, 0.67, 86091650, 1774137602000, 'REALTIME'
        )
        """
    )
    con.execute(
        """
        INSERT INTO raw_trades VALUES
        (
            'trade', 'KRW-BTC', 128510000, 0.015, 'BID', 128100000, 'RISE', 410000,
            '2026-03-22', '00:00:00', 1774137600120, 1774137600123, 17741376001200001,
            128510000, 0.4313, 128505000, 0.0199, 'REALTIME'
        ),
        (
            'trade', 'KRW-BTC', 128505000, 0.008, 'ASK', 128100000, 'RISE', 405000,
            '2026-03-22', '00:00:00', 1774137600540, 1774137600543, 17741376005400002,
            128510000, 0.4200, 128505000, 0.0311, 'REALTIME'
        ),
        (
            'trade', 'KRW-BTC', 128530000, 0.021, 'BID', 128100000, 'RISE', 430000,
            '2026-03-22', '00:00:01', 1774137601010, 1774137601015, 17741376010100003,
            128530000, 0.5121, 128525000, 0.0275, 'REALTIME'
        ),
        (
            'trade', 'KRW-BTC', 128525000, 0.011, 'ASK', 128100000, 'RISE', 425000,
            '2026-03-22', '00:00:01', 1774137601420, 1774137601424, 17741376014200004,
            128530000, 0.4980, 128525000, 0.0222, 'REALTIME'
        )
        """
    )
    con.execute(
        """
        INSERT INTO features_1s VALUES
        (
            '2026-03-22 09:00:00', 'KRW-BTC', 128510000,
            null, null, 1.42, 2, 0.015, 0.008, 0.3043478261, null
        ),
        (
            '2026-03-22 09:00:01', 'KRW-BTC', 128530000,
            0.0001556307, null, 0.91, 2, 0.021, 0.011, 0.3125, null
        ),
        (
            '2026-03-22 09:00:02', 'KRW-BTC', 128495000,
            -0.0002723092, -0.0000389088, 0.67, 0, 0.0, 0.0, 0.0, 0.000214
        )
        """
    )

    con.close()
    print(db_path)


if __name__ == "__main__":
    main()
