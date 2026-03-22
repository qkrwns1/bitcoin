from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "db" / "research.duckdb"


def print_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    print("Tables:")
    for table in tables:
        print(f"- {table}")
    print()
    return tables


def print_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    print(f"Schema: {table}")
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    for name, dtype, nullable, default, key, extra in rows:
        print(
            f"  {name:<16} {dtype:<12} nullable={nullable} default={default} key={key} extra={extra}"
        )
    print()


def print_sample(con: duckdb.DuckDBPyConnection, table: str, limit: int) -> None:
    print(f"Sample rows: {table} (limit {limit})")
    rows = con.execute(f"SELECT * FROM {table} LIMIT {limit}").fetchall()
    if not rows:
        print("  <empty>")
        print()
        return

    for row in rows:
        print(f"  {row}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="View local DuckDB tables and sample rows.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to DuckDB file.")
    parser.add_argument("--table", help="Only show one table.")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to show.")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Database file not found: {args.db}")

    con = duckdb.connect(str(args.db), read_only=True)

    try:
        tables = print_tables(con)

        if args.table:
            target_tables = [args.table]
        else:
            target_tables = tables

        for table in target_tables:
            if table not in tables:
                print(f"Skipping missing table: {table}")
                print()
                continue
            print_schema(con, table)
            print_sample(con, table, args.limit)
    finally:
        con.close()


if __name__ == "__main__":
    main()
