"""Incremental local Parquet store for Yahoo Finance market bars.

This module is intentionally independent from the dashboard calculations.
Phase 2 builds and validates the local source before existing readers migrate.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from market_scanner import normalize_symbol

NY = ZoneInfo("America/New_York")
CONFIG_FILE = Path(__file__).with_name("market_data_config.json")
STANDARD_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume",
                    "Dividends", "Stock Splits"]


@dataclass
class UpdateMetadata:
    ticker: str
    interval: str
    updated_at: str
    first_timestamp: str | None
    last_timestamp: str | None
    rows: int
    status: str
    requested_start: str | None
    requested_end: str | None
    rows_downloaded: int
    rows_added: int
    duplicates_removed: int
    invalid_rows_removed: int
    source: str = "Yahoo Finance via yfinance"
    error: str | None = None


def load_config(path: Path = CONFIG_FILE) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class MarketDataStore:
    def __init__(self, root: Path | None = None, config: dict[str, object] | None = None):
        self.config = config or load_config()
        configured = Path(str(self.config.get("data_root", "data")))
        self.root = root or (CONFIG_FILE.parent / configured)
        for relative in ("benchmarks", "sectors", "metadata", "cache"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def _interval_config(self, interval: str) -> dict[str, object]:
        intervals = self.config.get("intervals", {})
        if interval not in intervals:
            raise ValueError(f"Unsupported interval: {interval}")
        return intervals[interval]

    def data_path(self, ticker: str, interval: str) -> Path:
        directory = str(self._interval_config(interval)["directory"])
        return self.root / directory / f"{normalize_symbol(ticker)}.parquet"

    def metadata_path(self, ticker: str, interval: str) -> Path:
        safe = normalize_symbol(ticker)
        return self.root / "metadata" / interval / f"{safe}.json"

    def read(self, ticker: str, interval: str):
        import pandas as pd
        path = self.data_path(ticker, interval)
        if not path.exists():
            return pd.DataFrame(columns=STANDARD_COLUMNS,
                                index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = "timestamp"
        return frame.sort_index()

    @staticmethod
    def normalize(frame, interval: str):
        import pandas as pd
        if frame is None or frame.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS,
                                index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
        result = frame.copy()
        if isinstance(result.columns, pd.MultiIndex):
            if len(set(result.columns.get_level_values(-1))) == 1:
                result.columns = result.columns.get_level_values(0)
            else:
                raise ValueError("Expected data for one ticker, received multi-ticker columns")
        index = pd.to_datetime(result.index)
        if index.tz is None:
            index = index.tz_localize(NY)
        result.index = index.tz_convert("UTC")
        result.index.name = "timestamp"
        for column in STANDARD_COLUMNS:
            if column not in result:
                result[column] = 0.0 if column in {"Dividends", "Stock Splits"} else float("nan")
        return result[STANDARD_COLUMNS].sort_index()

    @staticmethod
    def validate(frame) -> tuple[object, int]:
        import pandas as pd
        if frame.empty:
            return frame, 0
        numeric = frame.copy()
        for column in STANDARD_COLUMNS:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        required = numeric[["Open", "High", "Low", "Close"]]
        valid = required.notna().all(axis=1) & (required > 0).all(axis=1)
        valid &= numeric["High"] + 1e-9 >= required[["Open", "Close", "Low"]].max(axis=1)
        valid &= numeric["Low"] - 1e-9 <= required[["Open", "Close", "High"]].min(axis=1)
        valid &= numeric["Volume"].fillna(0) >= 0
        removed = int((~valid).sum())
        return numeric.loc[valid], removed

    @staticmethod
    def merge(existing, downloaded) -> tuple[object, int, int]:
        import pandas as pd
        before = len(existing)
        if existing.empty:
            combined = downloaded.copy().sort_index()
        elif downloaded.empty:
            combined = existing.copy().sort_index()
        else:
            combined = pd.concat([existing, downloaded]).sort_index()
        duplicate_count = int(combined.index.duplicated(keep="last").sum())
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined, duplicate_count, max(0, len(combined) - before)

    def _request_range(self, existing, interval: str, now: datetime) -> tuple[datetime | None, datetime]:
        end = now.astimezone(timezone.utc) + timedelta(days=1)
        if existing.empty:
            lookback = self._interval_config(interval).get("lookback_days")
            return (end - timedelta(days=int(lookback))) if lookback else None, end
        overlap = int(self.config.get("overlap_bars", 3))
        if interval == "1d":
            return existing.index[-1].to_pydatetime() - timedelta(days=max(7, overlap * 3)), end
        minutes = {"60m": 60, "30m": 30, "15m": 15, "5m": 5}[interval]
        start = existing.index[-1].to_pydatetime() - timedelta(minutes=minutes * overlap)
        lookback = int(self._interval_config(interval).get("lookback_days") or 60)
        return max(start, end - timedelta(days=lookback)), end

    @staticmethod
    def yahoo_download(ticker: str, interval: str, start: datetime | None,
                       end: datetime, initial_daily_period: str = "max"):
        import yfinance as yf
        arguments = dict(interval=interval, auto_adjust=False, actions=True, prepost=False,
                         progress=False, threads=False, timeout=30, repair=True,
                         multi_level_index=False)
        if start is None and interval == "1d":
            arguments["period"] = initial_daily_period
        else:
            arguments.update(start=start, end=end)
        return yf.download(ticker, **arguments)

    def update(self, ticker: str, interval: str,
               downloader: Callable | None = None, now: datetime | None = None) -> UpdateMetadata:
        ticker = normalize_symbol(ticker)
        now = now or datetime.now(timezone.utc)
        existing = self.read(ticker, interval)
        start, end = self._request_range(existing, interval, now)
        download = downloader or self.yahoo_download
        try:
            raw = download(ticker, interval, start, end,
                           str(self.config.get("daily_initial_period", "max")))
            normalized = self.normalize(raw, interval)
            normalized, invalid = self.validate(normalized)
            merged, duplicates, added = self.merge(existing, normalized)
            if merged.empty:
                status = "empty"
            else:
                path = self.data_path(ticker, interval)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".parquet.tmp")
                merged.to_parquet(temporary, engine="pyarrow", compression="zstd")
                temporary.replace(path)
                status = "updated" if added else "current"
            error = None
        except Exception as exc:
            merged, normalized = existing, existing.iloc[0:0]
            duplicates = added = invalid = 0
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        metadata = UpdateMetadata(
            ticker=ticker, interval=interval, updated_at=now.astimezone(timezone.utc).isoformat(),
            first_timestamp=merged.index[0].isoformat() if len(merged) else None,
            last_timestamp=merged.index[-1].isoformat() if len(merged) else None,
            rows=len(merged), status=status,
            requested_start=start.isoformat() if start else None, requested_end=end.isoformat(),
            rows_downloaded=len(normalized), rows_added=added,
            duplicates_removed=duplicates, invalid_rows_removed=invalid, error=error)
        _atomic_json(self.metadata_path(ticker, interval), asdict(metadata))
        return metadata


def update_many(tickers: list[str], intervals: list[str], root: Path | None = None) -> list[UpdateMetadata]:
    store = MarketDataStore(root=root)
    results = []
    pause = float(store.config.get("request_pause_seconds", 0.15))
    for ticker in tickers:
        for interval in intervals:
            result = store.update(ticker, interval)
            results.append(result)
            print(f"{ticker:6} {interval:3} {result.status:7} rows={result.rows} +{result.rows_added}")
            time.sleep(pause)
    return results
