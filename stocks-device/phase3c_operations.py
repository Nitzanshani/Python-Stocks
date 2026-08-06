"""Blind operational accumulation, session gate, and weekly integrity audit."""
from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from intraday_sessions import INTERVAL_MINUTES,annotate_intraday_bars
from market_data_api import load_market_data

def complete_session_sets(spec):
 sets={};partial={};reasons=[]
 for ticker in spec["symbols"]+spec["controls"]:
  for interval in spec["intervals"]:
   try:frame=annotate_intraday_bars(load_market_data(ticker,interval),interval)
   except Exception as exc:sets[(ticker,interval)]=set();reasons.append({"ticker":ticker,"interval":interval,"session_date":None,"reason":f"missing_series:{type(exc).__name__}"});continue
   regular=frame[frame.is_regular_session];complete=set();partials=set()
   for date,g in regular.groupby("session_date"):
    expected=int(np.ceil((g.session_close.iloc[0]-g.session_open.iloc[0]).total_seconds()/60/INTERVAL_MINUTES[interval]))
    why=[]
    if len(g)!=expected:why.append(f"missing_bars:{expected-len(g)}")
    if not bool(g.bar_closed.all()):why.append("session_not_closed")
    if g.index.duplicated().any():why.append("duplicate_timestamps")
    if why:partials.add(date);reasons.append({"ticker":ticker,"interval":interval,"session_date":str(date),"reason":"|".join(why)})
    else:complete.add(date)
   sets[(ticker,interval)]=complete;partial[(ticker,interval)]=partials
 common=set.intersection(*sets.values()) if sets else set()
 return sets,partial,common,pd.DataFrame(reasons)

def operational_status(spec,quality,git_root):
 sets,partials,common,reasons=complete_session_sets(spec);p=spec["periods"];start=pd.Timestamp(p["confirmation_start"]).date();end=pd.Timestamp(p["confirmation_end"]).date()
 confirmation=sorted(d for d in common if start<=d<=end);required=int(p["confirmation_min_sessions"])
 failed=quality[~quality.quality_gate_passed] if "quality_gate_passed" in quality else quality
 try:commit=subprocess.run(["git","rev-parse","HEAD"],cwd=git_root,capture_output=True,text=True,check=True).stdout.strip()
 except Exception:commit="uncommitted-workspace"
 return {"current_date":datetime.now(timezone.utc).date().isoformat(),"confirmation_sessions_complete":len(confirmation),"confirmation_sessions_required":required,"sessions_remaining":max(0,required-len(confirmation)),"latest_complete_session":str(max(confirmation)) if confirmation else None,"partial_sessions":int(len(set().union(*partials.values()))) if partials else 0,"failed_symbols":sorted(failed.ticker.unique().tolist()) if not failed.empty else [],"failed_intervals":sorted(failed.interval.unique().tolist()) if not failed.empty else [],"quality_gate_status":"passed" if failed.empty else "failed","holdout_locked":True,"frozen_spec_hash":spec["_frozen_spec_hash"],"git_commit":commit},reasons

def write_operational_status(root,status,reasons):
 root=Path(root);reports=root/"reports";reports.mkdir(parents=True,exist_ok=True);target=reports/"daily_accumulation_status.json";previous=json.loads(target.read_text()).get("confirmation_sessions_complete",0) if target.exists() else 0;target.write_text(json.dumps(status,indent=2));reasons.to_csv(reports/"rejected_sessions.csv",index=False)
 rows=["# Phase 3C Blind Accumulation Status","",*(f"- {k}: {json.dumps(v)}" for k,v in status.items()),"","No relationship metrics were calculated or displayed."]
 (reports/"daily_accumulation_status.md").write_text("\n".join(rows));(reports/"daily_accumulation_status.html").write_text("<!doctype html><meta charset='utf-8'><pre>"+"\n".join(rows)+"</pre>")
 return [m for m in (10,20,30,40,50,60) if previous<m<=status["confirmation_sessions_complete"]]

def write_milestone_report(root,milestone,status,event_count,response_count,quality):
 reports=Path(root)/"reports";payload={"milestone_sessions":milestone,"data_quality_rows":len(quality),"coverage_min":float(quality.coverage_ratio.min()),"event_count_available":event_count,"response_count_available":response_count,"practical_effect_technically_calculable":bool(milestone>=60),"research_conclusion_generated":False,"holdout_locked":True,"warnings":quality.loc[quality.warnings!="","warnings"].tolist()}
 (reports/f"milestone_{milestone:02d}_sessions.json").write_text(json.dumps(payload,indent=2));(reports/f"milestone_{milestone:02d}_sessions.html").write_text("<!doctype html><meta charset='utf-8'><h1>Blind accumulation milestone</h1><pre>"+json.dumps(payload,indent=2)+"</pre>")

def weekly_integrity_audit(spec,root,data_root):
 root=Path(root);data_root=Path(data_root);snapshot_path=root/"operations/integrity_snapshot.json";previous=json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
 hashes={};metadata_mismatches=[]
 directories={"5m":"5m","15m":"15m","30m":"30m","60m":"hourly"}
 for ticker in spec["symbols"]+spec["controls"]:
  for interval in spec["intervals"]:
   path=data_root/directories[interval]/f"{ticker}.parquet";key=f"{ticker}:{interval}"
   if path.exists():hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
   meta=data_root.parent/"metadata"/interval/f"{ticker}.json"
   if not meta.exists():metadata_mismatches.append(f"{key}:metadata_missing")
 changed=sorted(k for k,v in hashes.items() if previous.get("input_hashes",{}).get(k) not in (None,v));payload={"audit_timestamp":datetime.now(timezone.utc).isoformat(),"frozen_spec_hash":spec["_frozen_spec_hash"],"input_hashes":hashes,"changed_since_previous_audit":changed,"metadata_mismatches":metadata_mismatches,"holdout_locked":True}
 snapshot_path.parent.mkdir(parents=True,exist_ok=True);snapshot_path.write_text(json.dumps(payload,indent=2));reports=root/"reports";reports.mkdir(parents=True,exist_ok=True);(reports/"weekly_integrity_audit.html").write_text("<!doctype html><meta charset='utf-8'><h1>Weekly Integrity Audit</h1><pre>"+json.dumps(payload,indent=2)+"</pre>");return payload
