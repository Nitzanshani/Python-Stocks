"""Build or incrementally refresh the local Parquet market-data store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_data_store import MarketDataStore, load_config, update_many
from market_scanner import load_symbols_file, load_universe_details, normalize_symbol


def select_failed_symbols(store: MarketDataStore, symbols: list[str],
                          intervals: list[str]) -> list[str]:
    """Return only symbols whose latest checkpoint contains a failed status."""
    failed_symbols = []
    for symbol in symbols:
        for interval in intervals:
            try:
                metadata = json.loads(store.metadata_path(symbol, interval).read_text())
            except (OSError, ValueError):
                continue
            if metadata.get("status") == "failed":
                failed_symbols.append(symbol)
                break
    return sorted(set(failed_symbols))


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", help="Comma-separated tickers; default is full research universe")
    parser.add_argument("--intervals", default="1d", help="Comma-separated intervals")
    parser.add_argument("--data-root", type=Path, help="Override configured data directory")
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--quality-report", action="store_true")
    args = parser.parse_args()
    if args.symbols_file:
        symbols, _ = load_symbols_file(args.symbols_file)
    elif args.symbols:
        symbols = sorted({normalize_symbol(item) for item in args.symbols.split(",") if item.strip()})
    else:
        symbols, _, _ = load_universe_details()
        symbols = sorted(set(symbols) | set(config.get("benchmarks", [])) |
                         set(config.get("sector_etfs", [])))
    intervals = [item.strip() for item in args.intervals.split(",") if item.strip()]
    report_symbols = list(symbols)
    unknown = sorted(set(intervals) - set(config["intervals"]))
    if unknown:
        parser.error(f"unsupported intervals: {', '.join(unknown)}")
    store = MarketDataStore(root=args.data_root)
    if args.retry_failed:
        symbols = select_failed_symbols(store, symbols, intervals)
        print(f"Retrying {len(symbols)} failed tickers")
    results = update_many(symbols, intervals, root=args.data_root,
                          max_workers=args.max_workers, force_refresh=args.force_refresh,
                          batch_size=args.batch_size)
    failed = sum(result.status == "failed" for result in results)
    if args.quality_report:
        from data_quality import build_daily_quality_report
        from universe_metadata import build_universe_snapshot
        build_universe_snapshot(store)
        build_daily_quality_report(store, report_symbols)
    print(f"Completed {len(results)} updates; failures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
