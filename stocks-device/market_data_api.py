"""Read-only API for Phase 3A local Parquet inputs."""

from __future__ import annotations

from pathlib import Path

from market_data_store import MarketDataStore


class MarketDataUnavailable(FileNotFoundError):
    pass


def load_market_data(symbol: str, interval: str = "1d", start: str | None = None,
                     end: str | None = None, data_root: Path | None = None):
    """Return a defensive, sorted UTC copy without modifying its Parquet source."""
    import pandas as pd
    store = MarketDataStore(root=data_root)
    path = store.data_path(symbol, interval)
    if not path.exists():
        raise MarketDataUnavailable(
            f"Local {interval} Parquet data is missing for {symbol}: {path}. "
            "Phase 3A never downloads data; run update_market_data.py first.")
    frame = pd.read_parquet(path).copy(deep=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.index.name = "timestamp"
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start, tz="UTC")
                          if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")]
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end, tz="UTC")
                          if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")]
    if frame.empty:
        raise MarketDataUnavailable(f"No local {interval} observations for {symbol} in requested range")
    return frame.copy(deep=True)


def load_aligned_market_data(symbols: list[str], interval: str = "1d",
                             start: str | None = None, end: str | None = None,
                             data_root: Path | None = None) -> dict[str, object]:
    """Load inputs independently; alignment is performed on returns, never filled."""
    return {symbol: load_market_data(symbol, interval, start, end, data_root)
            for symbol in symbols}
