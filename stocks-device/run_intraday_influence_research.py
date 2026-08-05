"""Phase 3B local-only intraday research pipeline."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess
from datetime import datetime,timezone
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
import pandas as pd
from intraday_audit import audit_intraday
from intraday_event_engine import detect_intraday_events,cluster_intraday_events
from intraday_features import build_intraday_panel
from intraday_residuals import build_intraday_residuals
from intraday_response_engine import measure_intraday_responses
from market_data_api import load_aligned_market_data

BASE=Path(__file__).resolve().parent
def load_config(path):return json.loads(Path(path).read_text())
def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run_engines(config_path):
 c=load_config(config_path);symbols=c["symbols"]+c["controls"];root=BASE/c["output_root"]
 audit=audit_intraday(symbols,c["intervals"],root); all_features=[];all_residuals=[];all_events=[];all_clusters=[];all_reps=[];all_responses=[]
 for interval in c["analysis_intervals"]:
  print(f"Phase 3B engines: {interval}",flush=True);frames=load_aligned_market_data(symbols,interval)
  features=build_intraday_panel(frames,interval,c["history_sessions"])
  residuals=build_intraday_residuals(features,c["market_benchmark"],c["sector_benchmark"],c["history_sessions"],c["minimum_history_sessions"])
  research=residuals[residuals.ticker.isin(c["symbols"])]
  events=detect_intraday_events(research,c);clusters,reps=cluster_intraday_events(events,c["event_thresholds"]["cluster_minutes"])
  responses=measure_intraday_responses(reps,residuals,c["symbols"],c["response_horizons_minutes"])
  if not responses.empty and not reps.empty:
   target_events=reps[["ticker","session_date","event_timestamp"]].rename(columns={"ticker":"target_ticker","event_timestamp":"target_event_timestamp"})
   responses=responses.merge(target_events,on=["target_ticker","session_date"],how="left")
   source_time=responses.event_available_at;target_time=responses.target_event_timestamp
   responses["lead_lag_classification"]="insufficient_resolution"
   responses.loc[target_time==responses.event_available_at-pd.to_timedelta({"5m":5,"15m":15,"30m":30,"60m":60}[interval],unit="m"),"lead_lag_classification"]="same_bar"
   responses.loc[target_time>=responses.event_available_at,"lead_lag_classification"]="source_before_target"
   responses.loc[target_time<responses.event_available_at-pd.to_timedelta({"5m":5,"15m":15,"30m":30,"60m":60}[interval],unit="m"),"lead_lag_classification"]="target_before_source"
   responses=responses.sort_values("target_event_timestamp").drop_duplicates(["event_id","target_ticker","horizon_minutes"],keep="first")
  all_features.append(features);all_residuals.append(residuals);all_events.append(events);all_clusters.append(clusters);all_reps.append(reps);all_responses.append(responses)
 values={"features":pd.concat(all_features,ignore_index=True),"residuals":pd.concat(all_residuals,ignore_index=True),
  "events":pd.concat(all_events,ignore_index=True),"clusters":pd.concat(all_clusters,ignore_index=True),
  "representatives":pd.concat(all_reps,ignore_index=True),"responses":pd.concat(all_responses,ignore_index=True)}
 paths={"features":"features/intraday_features.parquet","residuals":"residuals/intraday_residuals.parquet",
  "events":"events/intraday_events.parquet","clusters":"events/intraday_event_clusters.parquet",
  "representatives":"events/intraday_representative_events.parquet","responses":"responses/intraday_responses.parquet"}
 for name,relative in paths.items():p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);values[name].to_parquet(p,index=False,compression="zstd")
 report=build_interim(root,audit,values);build_manifest(root,config_path,c,symbols,report)
 return values,report

def build_interim(root,audit,values):
 r=values["responses"];summary={"coverage_rows":len(audit),"minimum_complete_sessions":int(audit.complete_sessions.min()),
  "features":len(values["features"]),"residuals_ok":int((values["residuals"].residual_status=="ok").sum()),
  "residuals_rejected":int((values["residuals"].residual_status!="ok").sum()),"events":len(values["events"]),
  "clusters":len(values["clusters"]),"representatives":len(values["representatives"]),
  "responses":int(r.response_available.sum()),"responses_rejected":int((~r.response_available).sum()),
  "same_bar_rate":float((r.lead_lag_classification=="same_bar").mean()),
  "near_close_events":int((values["representatives"].minutes_from_open>=330).sum())}
 counts=r.groupby(["horizon_minutes","response_available"]).size().unstack(fill_value=0)
 reports=root/"reports";reports.mkdir(parents=True,exist_ok=True)
 lines=["# Phase 3B Interim Report","",*(f"- {k}: {v}" for k,v in summary.items()),"","## Responses by horizon","",counts.to_csv()]
 (reports/"phase3b_interim.md").write_text("\n".join(lines));(reports/"phase3b_interim.json").write_text(json.dumps(summary,indent=2))
 return summary

def build_manifest(root,config_path,c,symbols,test_status):
 try:commit=subprocess.run(["git","rev-parse","HEAD"],cwd=BASE,capture_output=True,text=True,check=True).stdout.strip();dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=BASE,capture_output=True,text=True).stdout)
 except Exception:commit="uncommitted-workspace";dirty=True
 deps={};
 for name in ["pandas","numpy","scipy","pyarrow","pandas-market-calendars"]:
  try:deps[name]=version(name)
  except PackageNotFoundError:deps[name]=None
 files={f"{s}:{i}":_hash(BASE/"data"/{"5m":"market/5m","15m":"market/15m","30m":"market/30m","60m":"market/hourly"}[i]/f"{s}.parquet") for s in symbols for i in c["intervals"]}
 manifest={"run_id":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),"git_commit":commit,"dirty_worktree":dirty,
  "configuration_hash":_hash(config_path),"engine_version":c["engine_version"],"input_file_hashes":files,
  "intervals":c["intervals"],"symbols":symbols,"timezone":"America/New_York","random_seed":c["random_seed"],
  "dependency_versions":deps,"test_status":test_status,"warnings":["Yahoo intraday retention limits history to observed local range"]}
 p=root/"manifests";p.mkdir(parents=True,exist_ok=True);(p/"latest_manifest.json").write_text(json.dumps(manifest,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=BASE/"intraday_influence_config.json");p.add_argument("--symbols");args=p.parse_args()
 if args.symbols:
  c=load_config(args.config);requested=[x.strip().upper() for x in args.symbols.split(",")]
  if not set(requested)<=set(c["symbols"]):p.error("subset must use configured research symbols")
  c["symbols"]=requested;runtime=BASE/"research_intraday/.runtime_config.json";runtime.parent.mkdir(exist_ok=True);runtime.write_text(json.dumps(c));args.config=runtime
 _,summary=run_engines(args.config);print(json.dumps(summary,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
