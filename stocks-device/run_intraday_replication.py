"""Phase 3C out-of-time replication CLI; holdout is locked by default."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess
from datetime import datetime,timezone
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
import pandas as pd
from intraday_quality_monitor import build_quality_monitor,require_quality_gate
from phase3c_periods import load_frozen_spec,period_bounds

BASE=Path(__file__).resolve().parent
EXPECTED_HASH=(BASE/"PHASE3B_FROZEN_SPEC.sha256").read_text().strip()

def input_hashes(spec):
 result={}
 directories={"5m":"5m","15m":"15m","30m":"30m","60m":"hourly"}
 for ticker in spec["symbols"]+spec["controls"]:
  for interval in spec["intervals"]:
   p=BASE/"data/market"/directories[interval]/f"{ticker}.parquet"
   if p.exists():result[f"{ticker}:{interval}"]=hashlib.sha256(p.read_bytes()).hexdigest()
 return result

def period_session_count(spec,period,audit):
 start,end=period_bounds(spec,period,period=="holdout")
 # Counts are verified from the benchmark, then quality must pass for all inputs.
 from market_data_api import load_market_data
 from intraday_sessions import annotate_intraday_bars
 frame=annotate_intraday_bars(load_market_data("SPY","5m"),"5m");dates=pd.to_datetime(frame.session_date)
 mask=dates>=pd.Timestamp(start)
 if end:mask&=dates<=pd.Timestamp(end)
 return int(frame.loc[mask&frame.is_regular_session].groupby("session_date").size().ge(78).sum())

def write_manifest(root,spec,period,unlocked,warnings,test_status):
 try:commit=subprocess.run(["git","rev-parse","HEAD"],cwd=BASE,capture_output=True,text=True,check=True).stdout.strip();dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=BASE,capture_output=True,text=True).stdout)
 except Exception:commit="uncommitted-workspace";dirty=True
 deps={}
 for name in ["pandas","numpy","scipy","pyarrow","pandas-market-calendars"]:
  try:deps[name]=version(name)
  except PackageNotFoundError:deps[name]=None
 outputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file() and "manifests" not in p.parts}
 manifest={"run_id":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),"run_type":"phase3c_replication","period":period,
  "frozen_spec_hash":spec["_frozen_spec_hash"],"git_commit":commit,"dirty_worktree":dirty,"input_hashes":input_hashes(spec),
  "session_range":period_bounds(spec,period,unlocked),"symbols":spec["symbols"]+spec["controls"],"intervals":spec["intervals"],
  "random_seed":spec["random_seed"],"dependency_versions":deps,"holdout_unlocked":unlocked,"output_files":outputs,"warnings":warnings,"test_status":test_status}
 target=root/"manifests";target.mkdir(parents=True,exist_ok=True);(target/f"{manifest['run_id']}_{period}.json").write_text(json.dumps(manifest,indent=2));return manifest

def accumulation_report(root,spec,period,sessions,required,quality):
 status="ready" if sessions>=required else "accumulating"
 rows=["# Phase 3C Replication Accumulation","",f"- period: {period}",f"- status: {status}",f"- complete sessions: {sessions}",f"- required sessions: {required}",f"- frozen spec hash: {spec['_frozen_spec_hash']}","","No inference is produced until the predeclared minimum is reached."]
 reports=root/"reports";reports.mkdir(parents=True,exist_ok=True);(reports/"phase3c_replication_summary.md").write_text("\n".join(rows));(reports/"phase3c_replication_summary.html").write_text("<!doctype html><meta charset='utf-8'><pre>"+"\n".join(rows)+"</pre>")
 return status

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--frozen-spec",type=Path,default=BASE/"PHASE3B_FROZEN_SPEC.json");p.add_argument("--period",choices=["discovery","confirmation","holdout"],default="confirmation")
 p.add_argument("--relationship");p.add_argument("--all-relationships",action="store_true");p.add_argument("--rolling-stability",action="store_true");p.add_argument("--placebo",action="store_true");p.add_argument("--report-only",action="store_true");p.add_argument("--force",action="store_true");p.add_argument("--unlock-holdout",action="store_true");args=p.parse_args()
 if args.period=="holdout" and not args.unlock_holdout:p.error("holdout is locked; pass --unlock-holdout to record and open it")
 spec=load_frozen_spec(args.frozen_spec,EXPECTED_HASH);root=BASE/"research_replication"
 old=root/"data_quality/reports/intraday_data_quality.csv";previous=pd.read_csv(old) if old.exists() else pd.DataFrame();quality=build_quality_monitor(spec["symbols"]+spec["controls"],spec["intervals"],root/"data_quality",previous)
 old.parent.mkdir(parents=True,exist_ok=True);quality.to_csv(old,index=False);require_quality_gate(quality)
 sessions=period_session_count(spec,args.period,quality);required=spec["periods"]["confirmation_min_sessions" if args.period=="confirmation" else "future_holdout_min_sessions"] if args.period!="discovery" else 1
 status=accumulation_report(root,spec,args.period,sessions,required,quality);warnings=[] if status=="ready" else ["Minimum sessions not reached; accumulation report only"]
 write_manifest(root,spec,args.period,args.unlock_holdout,warnings,{"quality_gate":"passed","period_status":status})
 if status!="ready":print(json.dumps({"status":status,"sessions":sessions,"required":required},indent=2));return 0
 print(json.dumps({"status":"ready_for_replication","period":args.period,"sessions":sessions},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
