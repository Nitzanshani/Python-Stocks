"""Run a second full daily refresh and record Parquet idempotency evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from data_quality import build_daily_quality_report
from market_data_store import MarketDataStore, load_config, update_many
from market_scanner import load_universe_details


def _snapshot(store: MarketDataStore, symbols: list[str]) -> dict[str, dict[str, object]]:
    import pandas as pd
    result = {}
    for symbol in symbols:
        path = store.data_path(symbol, "1d")
        frame = store.read(symbol, "1d")
        result[symbol] = {"rows": len(frame), "sorted": bool(frame.index.is_monotonic_increasing),
            "duplicates": int(frame.index.duplicated().sum()),
            # Semantic hash: Parquet container metadata is not market data and
            # may differ after an otherwise identical rewrite.
            "content_hash": str(pd.util.hash_pandas_object(frame, index=True).sum())
            if path.exists() else None}
    return result


def main() -> int:
    config = load_config(); store = MarketDataStore()
    symbols, _, _ = load_universe_details()
    symbols = sorted(set(symbols) | set(config.get("benchmarks", [])) |
                     set(config.get("sector_etfs", [])))
    before = _snapshot(store, symbols)
    results = update_many(symbols, ["1d"], max_workers=1, batch_size=20)
    after = _snapshot(store, symbols)
    shrunk = [s for s in symbols if after[s]["rows"] < before[s]["rows"]]
    duplicate_files = [s for s in symbols if after[s]["duplicates"]]
    unsorted = [s for s in symbols if not after[s]["sorted"]]
    changed = [s for s in symbols if before[s]["content_hash"] != after[s]["content_hash"]]
    payload = {"verified_at": datetime.now(timezone.utc).isoformat(), "tickers": len(symbols),
        "failed": [r.ticker for r in results if r.status == "failed"],
        "rows_added": sum(r.rows_added for r in results), "shrunk": shrunk,
        "files_with_duplicates": duplicate_files, "unsorted": unsorted,
        "changed_content": changed,
        "note": "A live current-day bar may legitimately change intraday; row-count idempotency still holds."}
    reports = Path(__file__).with_name("reports"); reports.mkdir(exist_ok=True)
    (reports / "daily_idempotency.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    build_daily_quality_report(store, symbols, reports)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["failed"] or shrunk or duplicate_files or unsorted else 0


if __name__ == "__main__":
    raise SystemExit(main())
