"""Manually accumulate Phase 3B intraday bars in the local differential store."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from market_data_store import update_many

BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=BASE/"intraday_influence_config.json")
    parser.add_argument("--symbols"); parser.add_argument("--intervals")
    parser.add_argument("--max-workers",type=int,default=1); parser.add_argument("--batch-size",type=int,default=10)
    parser.add_argument("--force-refresh",action="store_true")
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    symbols=[x.strip().upper() for x in args.symbols.split(",")] if args.symbols else config["symbols"]+config["controls"]
    intervals=[x.strip() for x in args.intervals.split(",")] if args.intervals else config["intervals"]
    invalid=set(intervals)-{"5m","15m","30m","60m"}
    if invalid: parser.error(f"unsupported intraday intervals: {sorted(invalid)}")
    results=update_many(list(dict.fromkeys(symbols)),intervals,max_workers=args.max_workers,
                        batch_size=args.batch_size,force_refresh=args.force_refresh)
    summary={"updates":len(results),"failed":sum(x.status=="failed" for x in results),
             "rows_added":sum(x.rows_added for x in results),
             "overlap_rows_replaced":sum(x.duplicates_removed for x in results)}
    print(json.dumps(summary,indent=2)); return 1 if summary["failed"] else 0

if __name__=="__main__": raise SystemExit(main())
