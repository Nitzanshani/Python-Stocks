"""Build or incrementally refresh the local Parquet market-data store."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_data_store import load_config, update_many
from market_scanner import load_universe_details, normalize_symbol


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", help="Comma-separated tickers; default is full research universe")
    parser.add_argument("--intervals", default="1d", help="Comma-separated intervals")
    parser.add_argument("--data-root", type=Path, help="Override configured data directory")
    args = parser.parse_args()
    if args.symbols:
        symbols = sorted({normalize_symbol(item) for item in args.symbols.split(",") if item.strip()})
    else:
        symbols, _, _ = load_universe_details()
    intervals = [item.strip() for item in args.intervals.split(",") if item.strip()]
    unknown = sorted(set(intervals) - set(config["intervals"]))
    if unknown:
        parser.error(f"unsupported intervals: {', '.join(unknown)}")
    results = update_many(symbols, intervals, root=args.data_root)
    failed = sum(result.status == "failed" for result in results)
    print(f"Completed {len(results)} updates; failures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
