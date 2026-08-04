"""Live Yahoo Finance table for the S&P 500 and Nasdaq-100 universe."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Callable

from market_scanner import chunks, load_universe_details


REFRESH_MS = 60_000


@dataclass(frozen=True)
class Quote:
    price: float
    previous_close: float

    @property
    def change_pct(self) -> float:
        return (self.price / self.previous_close - 1) * 100 if self.previous_close else 0


def _download_quotes(
    symbols: list[str], on_batch: Callable[[dict[str, Quote]], None] | None = None
) -> dict[str, Quote]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("חסרה yfinance. יש להתקין את requirements.txt") from exc

    quotes: dict[str, Quote] = {}
    for batch in chunks(symbols, 50):
        batch_quotes: dict[str, Quote] = {}
        data = yf.download(
            batch,
            period="5d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            prepost=False,
            threads=True,
            progress=False,
            timeout=25,
            multi_level_index=True,
        )
        if data.empty:
            continue
        for symbol in batch:
            try:
                frame = data[symbol][["Close"]].dropna()
            except (KeyError, TypeError):
                continue
            if frame.empty:
                continue
            close = frame["Close"]
            latest = float(close.iloc[-1])
            dates = close.index.tz_convert("America/New_York").date
            latest_date = dates[-1]
            earlier = close[dates < latest_date]
            if earlier.empty:
                continue
            quote = Quote(latest, float(earlier.iloc[-1]))
            quotes[symbol] = quote
            batch_quotes[symbol] = quote
        if on_batch and batch_quotes:
            on_batch(batch_quotes)
    return quotes


class StockWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("S&P 500 + QQQ — מחירים בזמן אמת")
        self.root.geometry("1050x720")
        self.root.minsize(760, 480)
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.symbols: list[str] = []
        self.names: dict[str, str] = {}
        self.memberships: dict[str, str] = {}
        self.quotes: dict[str, Quote] = {}
        self.loading = False
        self._build()
        self.root.after(100, self._drain_messages)
        self.root.after(200, self._initialize)

    def _initialize(self) -> None:
        try:
            self.symbols, self.memberships, self.names = load_universe_details()
            self.status.configure(text=f"נטענו {len(self.symbols)} מניות — מתחיל הורדת מחירים")
            self._render()
            self.refresh()
        except Exception as exc:
            self.status.configure(text="טעינת רשימת המניות נכשלה")
            messagebox.showerror("שגיאה", str(exc))

    def _build(self) -> None:
        self.root.configure(background="#f3f5f7")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dark.TFrame", background="#f3f5f7")
        style.configure("Dark.TLabel", background="#f3f5f7", foreground="#17212b")
        style.configure(
            "Dark.Treeview", background="#ffffff", fieldbackground="#ffffff",
            foreground="#17212b", rowheight=28, borderwidth=1,
        )
        style.map("Dark.Treeview", background=[("selected", "#cde8ff")],
                  foreground=[("selected", "#17212b")])
        style.configure(
            "Dark.Treeview.Heading", background="#dce3e8", foreground="#17212b",
            relief="flat", font=("Arial", 12, "bold"), padding=8,
        )
        style.map("Dark.Treeview.Heading", background=[("active", "#c8d2d9")])

        top = ttk.Frame(self.root, padding=10, style="Dark.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="חיפוש:", style="Dark.TLabel").pack(side="right", padx=(8, 0))
        self.search = tk.StringVar()
        entry = ttk.Entry(top, textvariable=self.search, width=30, justify="right")
        entry.pack(side="right")
        self.search.trace_add("write", lambda *_: self._render())
        self.refresh_button = ttk.Button(top, text="רענון עכשיו", command=self.refresh)
        self.refresh_button.pack(side="left")
        self.status = ttk.Label(top, text="מתחיל…", style="Dark.TLabel")
        self.status.pack(side="left", padx=12)

        container = ttk.Frame(self.root, padding=(10, 0, 10, 10), style="Dark.TFrame")
        container.pack(fill="both", expand=True)
        columns = ("symbol", "company", "price", "change", "index")
        self.tree = ttk.Treeview(
            container, columns=columns, show="headings", style="Dark.Treeview"
        )
        headings = {
            "symbol": "סימול", "company": "שם החברה", "price": "מחיר נוכחי",
            "change": "שינוי יומי", "index": "מדד",
        }
        widths = {"symbol": 90, "company": 330, "price": 125, "change": 120, "index": 150}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="center")
        self.tree.column("company", anchor="e")
        self.tree.tag_configure("up", foreground="#07883b")
        self.tree.tag_configure("down", foreground="#d22121")
        self.tree.tag_configure("flat", foreground="#555555")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def refresh(self) -> None:
        if self.loading:
            return
        self.loading = True
        self.refresh_button.configure(state="disabled")
        self.status.configure(text="מוריד נתונים…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            if not self.symbols:
                symbols, memberships, names = load_universe_details()
                self.messages.put(("universe", (symbols, memberships, names)))
            else:
                symbols = self.symbols
            self.messages.put(("quotes", _download_quotes(
                symbols, lambda batch: self.messages.put(("quote_batch", batch))
            )))
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "universe":
                    self.symbols, self.memberships, self.names = payload  # type: ignore[misc]
                    self.status.configure(text=f"נטענו {len(self.symbols)} מניות — מוריד מחירים…")
                    self._render()
                elif kind == "quote_batch":
                    self.quotes.update(payload)  # type: ignore[arg-type]
                    self.status.configure(
                        text=f"מוריד מחירים… {len(self.quotes)} מתוך {len(self.symbols)}"
                    )
                    self._render()
                elif kind == "quotes":
                    self.quotes = payload  # type: ignore[assignment]
                    self.loading = False
                    self.refresh_button.configure(state="normal")
                    now = datetime.now().strftime("%H:%M:%S")
                    self.status.configure(text=f"עודכן: {now} | {len(self.quotes)} מניות | רענון כל דקה")
                    self._render()
                    self.root.after(REFRESH_MS, self.refresh)
                elif kind == "error":
                    self.loading = False
                    self.refresh_button.configure(state="normal")
                    self.status.configure(text="הרענון נכשל")
                    messagebox.showerror("שגיאה", str(payload))
        except queue.Empty:
            pass
        self.root.after(150, self._drain_messages)

    def _render(self) -> None:
        query = self.search.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        for symbol in self.symbols:
            company = self.names.get(symbol, symbol)
            if query and query not in symbol.lower() and query not in company.lower():
                continue
            quote = self.quotes.get(symbol)
            if quote is None:
                self.tree.insert(
                    "", "end", values=(symbol, company, "טוען…", "—",
                    self.memberships.get(symbol, "")), tags=("flat",)
                )
                continue
            change = quote.change_pct
            tag = "up" if change > 0 else "down" if change < 0 else "flat"
            self.tree.insert(
                "", "end", values=(symbol, company, f"${quote.price:,.2f}",
                f"{change:+.2f}%", self.memberships.get(symbol, "")), tags=(tag,)
            )


def run_gui() -> None:
    root = tk.Tk()
    StockWindow(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
