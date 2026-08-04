#!/usr/bin/env python3
"""Scan S&P 500 and Nasdaq-100 constituents for opening-session moves."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from io import StringIO
from typing import Iterable
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
CACHE_FILE = Path(__file__).with_name("constituents_cache.json")
WATCHLIST = {
    "AAOI": "Applied Optoelectronics, Inc.",
    "COHR": "Coherent Corp.",
}


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    close: float


@dataclass(frozen=True)
class ScanResult:
    symbol: str
    indexes: str
    first_10m_pct: float | None
    next_50m_pct: float | None
    after_30m_pct: float | None


def normalize_symbol(value: object) -> str:
    """Convert index notation such as BRK.B to Alpaca's BRK-B notation."""
    return str(value).strip().upper().replace(".", "-")


def _download_table(pd: object, url: str, required_columns: set[str]):
    """Download Wikipedia HTML with browser-like headers, then parse locally."""
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/124 Safari/537.36 stock-research-tool/1.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    tables = pd.read_html(StringIO(html), flavor="lxml")
    for table in tables:
        columns = {str(column).strip() for column in table.columns}
        if required_columns.issubset(columns):
            return table
    raise RuntimeError(f"No table with columns {sorted(required_columns)} found at {url}")


def _load_cached_universe() -> tuple[list[str], dict[str, str], dict[str, str]] | None:
    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        symbols = [normalize_symbol(value) for value in payload["symbols"]]
        if not symbols:
            return None
        return symbols, payload["memberships"], payload["names"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_cached_universe(
    symbols: list[str], memberships: dict[str, str], names: dict[str, str]
) -> None:
    payload = {"symbols": symbols, "memberships": memberships, "names": names}
    CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _add_watchlist(
    symbols: list[str], memberships: dict[str, str], names: dict[str, str]
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Add explicitly tracked stocks without duplicating index constituents."""
    labels, company_names = dict(memberships), dict(names)
    for raw_symbol, company_name in WATCHLIST.items():
        symbol = normalize_symbol(raw_symbol)
        groups = set(filter(None, labels.get(symbol, "").split("+")))
        groups.add("Watchlist")
        labels[symbol] = "+".join(sorted(groups))
        company_names.setdefault(symbol, company_name)
    return sorted(set(symbols) | set(WATCHLIST)), labels, company_names


def load_universe_details() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Load symbols, index memberships and company names."""
    cached = _load_cached_universe()
    if cached is not None:
        return _add_watchlist(*cached)
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Missing dependencies. Run: pip install -r requirements.txt") from exc
    try:
        sp_table = _download_table(pd, SP500_URL, {"Symbol", "Security"})
        ndx_table = _download_table(pd, NASDAQ100_URL, {"Ticker", "Company"})
    except Exception as exc:  # pandas/lxml errors vary between versions
        raise RuntimeError(
            "Could not download index constituents. Check the internet connection. "
            f"Technical detail: {type(exc).__name__}: {exc}"
        ) from exc

    sp_col = next((c for c in ("Symbol", "Ticker") if c in sp_table.columns), None)
    ndx_col = next((c for c in ("Ticker", "Symbol") if c in ndx_table.columns), None)
    if sp_col is None or ndx_col is None:
        raise RuntimeError("The constituent table format changed.")

    memberships: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    sp_name_col = next((c for c in ("Security", "Company") if c in sp_table.columns), None)
    ndx_name_col = next((c for c in ("Company", "Security") if c in ndx_table.columns), None)
    for _, row in sp_table.dropna(subset=[sp_col]).iterrows():
        symbol = normalize_symbol(row[sp_col])
        memberships.setdefault(symbol, set()).add("S&P 500")
        if sp_name_col:
            names[symbol] = str(row[sp_name_col]).strip()
    for _, row in ndx_table.dropna(subset=[ndx_col]).iterrows():
        symbol = normalize_symbol(row[ndx_col])
        memberships.setdefault(symbol, set()).add("QQQ")
        if ndx_name_col:
            names[symbol] = str(row[ndx_name_col]).strip()

    labels = {symbol: "+".join(sorted(groups)) for symbol, groups in memberships.items()}
    symbols = sorted(labels)
    symbols, labels, names = _add_watchlist(symbols, labels, names)
    _save_cached_universe(symbols, labels, names)
    return symbols, labels, names


def load_universe() -> tuple[list[str], dict[str, str]]:
    symbols, labels, _ = load_universe_details()
    return symbols, labels


def load_symbols_file(path: Path) -> tuple[list[str], dict[str, str]]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split(",", 1)[0].strip()
        if value and not value.startswith("#") and value.lower() not in {"symbol", "ticker"}:
            symbols.append(normalize_symbol(value))
    symbols = sorted(set(symbols))
    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    return symbols, {symbol: "custom" for symbol in symbols}


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def fetch_bars(symbols: list[str], trading_date: date) -> dict[str, list[Bar]]:
    """Fetch regular-session minute bars from Yahoo Finance in small batches."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Missing dependencies. Run: pip install -r requirements.txt") from exc

    start = datetime.combine(trading_date, dt_time(0, 0), NY)
    end = start + timedelta(days=1)
    regular_start = datetime.combine(trading_date, dt_time(9, 30), NY)
    regular_end = datetime.combine(trading_date, dt_time(16, 0), NY)
    output: dict[str, list[Bar]] = {symbol: [] for symbol in symbols}

    for number, batch in enumerate(chunks(symbols, 50), start=1):
        print(f"Downloading Yahoo batch {number}...", file=sys.stderr)
        data = yf.download(
            batch,
            start=start,
            end=end,
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            prepost=False,
            threads=True,
            progress=False,
            timeout=30,
            multi_level_index=True,
        )
        if data.empty:
            continue
        for symbol in batch:
            try:
                frame = data[symbol]
            except KeyError:
                continue
            for timestamp, raw in frame.iterrows():
                open_price, close_price = raw.get("Open"), raw.get("Close")
                if pd_is_missing(open_price) or pd_is_missing(close_price):
                    continue
                stamp = timestamp.to_pydatetime()
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=NY)
                else:
                    stamp = stamp.astimezone(NY)
                if regular_start <= stamp < regular_end:
                    output[symbol].append(Bar(stamp, float(open_price), float(close_price)))
        time.sleep(0.5)
    return output


def pd_is_missing(value: object) -> bool:
    """Avoid importing pandas globally just for its missing-value helper."""
    try:
        return value != value  # NaN is the only normal price value unequal to itself.
    except Exception:
        return True


def window_change(
    bars: list[Bar], start: dt_time, end: dt_time, require_complete: bool = True
) -> float | None:
    selected = [bar for bar in bars if start <= bar.timestamp.time().replace(tzinfo=None) < end]
    if not selected or selected[0].open <= 0:
        return None
    if require_complete:
        expected_last = (datetime.combine(date.min, end) - timedelta(minutes=1)).time()
        first_time = selected[0].timestamp.time().replace(tzinfo=None)
        last_time = selected[-1].timestamp.time().replace(tzinfo=None)
        if first_time != start or last_time != expected_last:
            return None
    return (selected[-1].close / selected[0].open - 1.0) * 100.0


def analyze(symbol: str, indexes: str, bars: list[Bar]) -> ScanResult:
    ordered = sorted(bars, key=lambda bar: bar.timestamp)
    return ScanResult(
        symbol=symbol,
        indexes=indexes,
        first_10m_pct=window_change(ordered, dt_time(9, 30), dt_time(9, 40)),
        next_50m_pct=window_change(ordered, dt_time(9, 40), dt_time(10, 30)),
        after_30m_pct=window_change(
            ordered, dt_time(10, 0), dt_time(16, 0), require_complete=False
        ),
    )


def status(value: float | None, threshold: float) -> str:
    if value is None:
        return "NO_DATA"
    if value > threshold:
        return f"UP>{threshold:g}%"
    if value < -threshold:
        return f"DOWN>{threshold:g}%"
    return "WITHIN_RANGE"


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def write_csv(path: Path, results: list[ScanResult]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["symbol", "indexes", "first_10m_pct", "first_10m_status", "next_50m_pct",
             "next_50m_status", "after_30m_pct", "after_30m_up_over_5pct"]
        )
        for row in results:
            writer.writerow(
                [row.symbol, row.indexes, fmt(row.first_10m_pct), status(row.first_10m_pct, 8),
                 fmt(row.next_50m_pct), status(row.next_50m_pct, 8), fmt(row.after_30m_pct),
                 row.after_30m_pct is not None and row.after_30m_pct > 5]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", type=date.fromisoformat, default=datetime.now(NY).date(), help="YYYY-MM-DD"
    )
    parser.add_argument("--cli", action="store_true", help="Run the CSV scanner instead of the GUI")
    parser.add_argument("--output", type=Path, default=Path("scan_results.csv"))
    parser.add_argument("--symbols-file", type=Path, help="Optional one-symbol-per-line file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cli:
        from web_gui import run_web_gui

        run_web_gui()
        return 0
    symbols, memberships = (
        load_symbols_file(args.symbols_file) if args.symbols_file else load_universe()
    )
    bars = fetch_bars(symbols, args.date)
    results = [analyze(symbol, memberships[symbol], bars[symbol]) for symbol in symbols]
    write_csv(args.output, results)

    candidates = sorted(
        (row for row in results if row.after_30m_pct is not None and row.after_30m_pct > 5),
        key=lambda row: row.after_30m_pct or 0,
        reverse=True,
    )
    print(f"\nStocks up more than 5% from 10:00 ET ({args.date}):")
    if not candidates:
        print("  None")
    for row in candidates:
        print(f"  {row.symbol:6} {row.after_30m_pct:+.2f}%  [{row.indexes}]")
    print(f"\nFull report: {args.output} ({len(results)} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
