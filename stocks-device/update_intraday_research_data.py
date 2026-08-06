"""Manually accumulate Phase 3B intraday bars in the local differential store."""

from __future__ import annotations
import argparse, json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from market_data_store import MarketDataStore,update_many
from intraday_quality_monitor import build_quality_monitor

BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=BASE/"intraday_influence_config.json")
    parser.add_argument("--symbols"); parser.add_argument("--intervals")
    parser.add_argument("--max-workers",type=int,default=1); parser.add_argument("--batch-size",type=int,default=10)
    parser.add_argument("--force-refresh",action="store_true")
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--quality-report",action="store_true")
    parser.add_argument("--verify-sessions",action="store_true")
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    symbols=[x.strip().upper() for x in args.symbols.split(",")] if args.symbols else config["symbols"]+config["controls"]
    intervals=[x.strip() for x in args.intervals.split(",")] if args.intervals else config["intervals"]
    invalid=set(intervals)-{"5m","15m","30m","60m"}
    if invalid: parser.error(f"unsupported intraday intervals: {sorted(invalid)}")
    store=MarketDataStore(); jobs=[(s,i) for s in list(dict.fromkeys(symbols)) for i in intervals]
    if args.retry_failed:
        jobs=[(s,i) for s,i in jobs if store.metadata_path(s,i).exists() and json.loads(store.metadata_path(s,i).read_text()).get("status")=="failed"]
    checkpoint=BASE/"research_intraday/update_checkpoint.json"
    done=set()
    if args.resume and checkpoint.exists():
        done={tuple(x) for x in json.loads(checkpoint.read_text()).get("completed",[])};jobs=[x for x in jobs if x not in done]
    before={(s,i):store.read(s,i) for s,i in jobs};old_audit_path=BASE/"research_intraday/reports/intraday_data_quality.csv"
    previous_quality=pd.read_csv(old_audit_path) if old_audit_path.exists() else pd.DataFrame()
    results=[]
    for s,i in jobs:
        result=update_many([s],[i],max_workers=args.max_workers,batch_size=args.batch_size,force_refresh=args.force_refresh)[0];results.append(result)
        checkpoint.parent.mkdir(parents=True,exist_ok=True);completed=[list(x) for x in sorted(done|{(x.ticker,x.interval) for x in results if x.status!="failed"})]
        checkpoint.write_text(json.dumps({"run_date":datetime.now(timezone.utc).isoformat(),"completed":completed},indent=2))
    quality=build_quality_monitor(symbols,intervals,BASE/"research_intraday",previous_quality) if (args.quality_report or args.verify_sessions) else None
    quality_map={(r.ticker,r.interval):r for r in quality.itertuples()} if quality is not None else {}
    log_rows=[]
    for result in results:
        old=before[(result.ticker,result.interval)];q=quality_map.get((result.ticker,result.interval))
        log_rows.append({"run_date":result.updated_at,"ticker":result.ticker,"interval":result.interval,
          "old_last_timestamp":old.index.max().isoformat() if len(old) else None,"new_last_timestamp":result.last_timestamp,
          "new_rows":result.rows_added,"replaced_overlap_rows":result.duplicates_removed,
          "complete_sessions_added":getattr(q,"sessions_added",None),"partial_sessions":getattr(q,"incomplete_sessions",None),
          "failed_sessions":getattr(q,"missing_sessions",None),"status":result.status})
    log_path=BASE/"research_intraday/update_runs.csv";new=pd.DataFrame(log_rows)
    if log_path.exists() and not new.empty:new=pd.concat([pd.read_csv(log_path),new],ignore_index=True)
    if not new.empty:new.to_csv(log_path,index=False)
    summary={"updates":len(results),"failed":sum(x.status=="failed" for x in results),
             "rows_added":sum(x.rows_added for x in results),
             "overlap_rows_replaced":sum(x.duplicates_removed for x in results)}
    print(json.dumps(summary,indent=2)); return 1 if summary["failed"] else 0

if __name__=="__main__": raise SystemExit(main())
