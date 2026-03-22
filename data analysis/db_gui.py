from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import duckdb


DB_PATH = Path(__file__).resolve().parent / "data" / "db" / "research.duckdb"


class DuckDBViewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DuckDB Viewer")
        self.root.geometry("1100x700")

        self.conn = duckdb.connect(str(DB_PATH), read_only=True)

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text=f"DB: {DB_PATH}").pack(anchor="w")

        controls = ttk.Frame(root, padding=(10, 0, 10, 10))
        controls.pack(fill="x")

        ttk.Label(controls, text="Table").pack(side="left")
        self.table_var = tk.StringVar()
        self.table_box = ttk.Combobox(controls, textvariable=self.table_var, state="readonly", width=40)
        self.table_box.pack(side="left", padx=(8, 12))
        self.table_box.bind("<<ComboboxSelected>>", self.on_table_selected)

        ttk.Label(controls, text="Limit").pack(side="left")
        self.limit_var = tk.StringVar(value="50")
        self.limit_entry = ttk.Entry(controls, textvariable=self.limit_var, width=8)
        self.limit_entry.pack(side="left", padx=(8, 12))

        ttk.Button(controls, text="Refresh", command=self.refresh_current_table).pack(side="left")

        self.schema_text = tk.Text(root, height=10, wrap="none")
        self.schema_text.pack(fill="x", padx=10, pady=(0, 10))

        table_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        table_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(table_frame, show="headings")
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.load_tables()

    def load_tables(self) -> None:
        tables = [row[0] for row in self.conn.execute("SHOW TABLES").fetchall()]
        self.table_box["values"] = tables
        if tables:
            self.table_var.set(tables[0])
            self.show_table(tables[0])

    def on_table_selected(self, _event: object) -> None:
        self.refresh_current_table()

    def refresh_current_table(self) -> None:
        table = self.table_var.get()
        if table:
            self.show_table(table)

    def show_table(self, table: str) -> None:
        try:
            limit = int(self.limit_var.get())
        except ValueError:
            messagebox.showerror("Invalid limit", "Limit must be an integer.")
            return

        schema_rows = self.conn.execute(f"DESCRIBE {table}").fetchall()
        self.schema_text.delete("1.0", tk.END)
        self.schema_text.insert(tk.END, f"Schema: {table}\n")
        for name, dtype, nullable, default, key, extra in schema_rows:
            self.schema_text.insert(
                tk.END,
                f"{name:<18} {dtype:<12} nullable={nullable} default={default} key={key} extra={extra}\n",
            )

        columns = [row[0] for row in schema_rows]
        rows = self.conn.execute(f"SELECT * FROM {table} LIMIT {limit}").fetchall()

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = columns

        for column in columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=150, anchor="w")

        for row in rows:
            self.tree.insert("", tk.END, values=[str(value) for value in row])


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Database file not found: {DB_PATH}")

    root = tk.Tk()
    app = DuckDBViewer(root)

    def on_close() -> None:
        app.conn.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
