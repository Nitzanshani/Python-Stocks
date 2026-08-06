"""Phase 3B local-only intraday research pipeline."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess
from datetime import datetime,timezone
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
import numpy as np,pandas as pd
from intraday_audit import audit_intraday
from intraday_event_engine import detect_intraday_events,cluster_intraday_events
from intraday_features import build_intraday_panel
from intraday_residuals import build_intraday_residuals
from intraday_response_engine import measure_intraday_responses
from intraday_event_study import select_intraday_controls,summarize_intraday_event_study
from intraday_predictive import run_intraday_walk_forward
from intraday_relationships import response_curves,build_intraday_relationships,compare_with_phase3a
from intraday_report import build_intraday_report
from market_data_api import load_aligned_market_data

BASE=Path(__file__).resolve().parent
def load_config(path):return json.loads(Path(path).read_text())
def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def classify_lead_lag(responses,representatives,interval):
 """Classify using event timestamps only; same-bar has precedence over later events."""
 if responses.empty:return responses
 minutes={"5m":5,"15m":15,"30m":30,"60m":60}[interval]
 grouped={(ticker,session):np.sort(g.event_timestamp.dropna().astype("int64").to_numpy())
          for (ticker,session),g in representatives.groupby(["ticker","session_date"])}
 source=representatives.set_index("event_id")[["event_timestamp","available_at"]]
 keys=responses[["event_id","target_ticker","session_date"]].drop_duplicates();labels={}
 for row in keys.itertuples(index=False):
  times=grouped.get((row.target_ticker,row.session_date),np.array([],dtype="int64"))
  if row.event_id not in source.index or not len(times):label="insufficient_resolution"
  else:
   event=source.loc[row.event_id];start=pd.Timestamp(event.event_timestamp).value;available=pd.Timestamp(event.available_at).value
   if np.any((times>=start)&(times<available)):label="same_bar"
   elif np.any(times>=available):label="source_before_target"
   elif np.any(times<start):label="target_before_source"
   else:label="insufficient_resolution"
  labels[(row.event_id,row.target_ticker,row.session_date)]=label
 responses=responses.copy();responses["lead_lag_classification"]=[labels[(r.event_id,r.target_ticker,r.session_date)] for r in responses.itertuples()]
 return responses

def run_engines(config_path):
 c=load_config(config_path);symbols=c["symbols"]+c["controls"];root=BASE/c["output_root"]
 audit=audit_intraday(symbols,c["intervals"],root); all_features=[];all_residuals=[];all_events=[];all_clusters=[];all_reps=[];all_responses=[]
 for interval in c["analysis_intervals"]:
  print(f"Phase 3B engines: {interval}",flush=True);frames=load_aligned_market_data(symbols,interval)
  features=build_intraday_panel(frames,interval,c["history_sessions"])
  residuals=build_intraday_residuals(features,c["market_benchmark"],c["sector_benchmark"],c["history_sessions"],c["minimum_history_sessions"])
  research=residuals[residuals.ticker.isin(c["symbols"])]
  events=detect_intraday_events(research,c);clusters,reps=cluster_intraday_events(events,c["event_thresholds"]["cluster_minutes"])
  responses=measure_intraday_responses(reps,residuals,c["symbols"],c["response_horizons_minutes"],c["response_threshold"])
  if not responses.empty and not reps.empty:responses=classify_lead_lag(responses,reps,interval)
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
 rejected_responses=r.loc[~r.response_available].groupby(["interval","horizon_name","missing_reason"]).size().rename("count").reset_index()
 rejected_event_inputs=values["residuals"].loc[values["residuals"].residual_status!="ok"].groupby(["ticker","interval","residual_status"]).size().rename("count").reset_index()
 reports=root/"reports";reports.mkdir(parents=True,exist_ok=True)
 rejected_responses.to_csv(reports/"phase3b_rejected_responses.csv",index=False);rejected_event_inputs.to_csv(reports/"phase3b_rejected_event_inputs.csv",index=False)
 lines=["# Phase 3B Interim Report","",*(f"- {k}: {v}" for k,v in summary.items()),"","## Responses by horizon","",counts.to_csv(),"","## Rejected responses","",rejected_responses.to_csv(index=False),"","## Event inputs rejected for insufficient history/data","",rejected_event_inputs.to_csv(index=False)]
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
 audit_path=root/"reports/intraday_data_audit.parquet";audit=pd.read_parquet(audit_path) if audit_path.exists() else pd.DataFrame()
 outputs={str(p.relative_to(root)):_hash(p) for p in root.rglob("*") if p.is_file() and "manifests" not in p.parts}
 manifest={"run_id":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),"git_commit":commit,"dirty_worktree":dirty,
  "configuration_hash":_hash(config_path),"engine_version":c["engine_version"],"input_file_hashes":files,
  "intervals":c["intervals"],"symbols":symbols,"timezone":"America/New_York","random_seed":c["random_seed"],
  "session_range":{"first":str(audit.first_timestamp.min()) if not audit.empty else None,"last":str(audit.last_timestamp.max()) if not audit.empty else None},
  "dependency_versions":deps,"output_file_hashes":outputs,"test_status":test_status,
  "warnings":["Yahoo intraday retention limits history to observed local range"]}
 p=root/"manifests";p.mkdir(parents=True,exist_ok=True);(p/"latest_manifest.json").write_text(json.dumps(manifest,indent=2))

def run_validation(config_path,force=False):
 c=load_config(config_path);root=BASE/c["output_root"];read=lambda p:pd.read_parquet(root/p)
 audit=read("reports/intraday_data_audit.parquet");residuals=read("residuals/intraday_residuals.parquet");events=read("events/intraday_representative_events.parquet");responses=read("responses/intraday_responses.parquet")
 if "interval" not in responses:responses=responses.merge(events[["event_id","interval"]],on="event_id",how="left")
 print("Phase 3B: matched controls",flush=True);p=root/"controls/intraday_control_matches.parquet";controls=pd.read_parquet(p) if p.exists() and not force else select_intraday_controls(events,responses,residuals,3);p.parent.mkdir(parents=True,exist_ok=True);controls.to_parquet(p,index=False,compression="zstd")
 study=summarize_intraday_event_study(events,responses,controls,c["fdr_alpha"],300,c["random_seed"]);study.to_parquet(root/"controls/intraday_event_study.parquet",index=False,compression="zstd")
 print("Phase 3B: session walk-forward",flush=True);p=root/"influence/intraday_predictive_folds.parquet";folds=pd.read_parquet(p) if p.exists() and not force else run_intraday_walk_forward(residuals,events,c["symbols"],c["walk_forward"]);p.parent.mkdir(parents=True,exist_ok=True);folds.to_parquet(p,index=False,compression="zstd")
 relationships=build_intraday_relationships(study,folds,c);relationships.to_parquet(root/"influence/intraday_relationships.parquet",index=False,compression="zstd")
 curves,decay=response_curves(responses);curves.to_parquet(root/"influence/intraday_response_curves.parquet",index=False);decay.to_parquet(root/"influence/intraday_decay_models.parquet",index=False)
 comparison=compare_with_phase3a(relationships,BASE/"research/influence/influence_relationships.parquet");comparison.to_parquet(root/"influence/phase3a_phase3b_comparison.parquet",index=False)
 build_intraday_report(root,c,audit,events,responses,study,folds,relationships,comparison,curves,decay)
 result={"controls":len(controls),"event_studies":len(study),"folds":len(folds),"relationships":len(relationships),"statuses":relationships.intraday_status.value_counts().to_dict(),"fdr_significant":int(study.fdr_significant.sum())}
 build_manifest(root,config_path,c,c["symbols"]+c["controls"],{"pipeline":"passed","validation":result})
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=BASE/"intraday_influence_config.json");p.add_argument("--symbols");args=p.parse_args()
 if args.symbols:
  c=load_config(args.config);requested=[x.strip().upper() for x in args.symbols.split(",")]
  if not set(requested)<=set(c["symbols"]):p.error("subset must use configured research symbols")
  c["symbols"]=requested;runtime=BASE/"research_intraday/.runtime_config.json";runtime.parent.mkdir(exist_ok=True);runtime.write_text(json.dumps(c));args.config=runtime
 _,summary=run_engines(args.config);validation=run_validation(args.config,force=True);print(json.dumps({"engines":summary,"validation":validation},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
