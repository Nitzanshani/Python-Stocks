"""Static Phase 3C replication and accumulation reports."""
from pathlib import Path
import pandas as pd

def build_pending_reports(root,relationships,sessions,required):
 root=Path(root);reports=root/"reports";reports.mkdir(parents=True,exist_ok=True);relationships.to_csv(reports/"phase3c_replication_relationships.csv",index=False)
 nvda=relationships[(relationships.source_ticker=="NVDA")&(relationships.target_ticker=="ANET")&(relationships.interval=="5m")&(relationships.horizon_minutes==5)]
 candidates=relationships[relationships.discovery_status=="candidate"]
 intro=f"Confirmation is accumulating: {sessions}/{required} complete sessions. No replication conclusion was computed."
 tables={"nvda_anet_replication.html":nvda,"candidate_replication.html":candidates,"rolling_stability.html":pd.DataFrame([{"status":"insufficient_data","reason":"confirmation minimum not reached"}]),"placebo_results.html":pd.DataFrame([{"status":"not_run","reason":"confirmation minimum not reached"}])}
 for name,frame in tables.items():(reports/name).write_text("<!doctype html><meta charset='utf-8'><p>"+intro+"</p>"+frame.to_html(index=False))
 summary=["# Phase 3C Out-of-Time Replication","",intro,"",f"- relationships awaiting confirmation: {len(relationships)}",f"- discovery candidates awaiting confirmation: {len(candidates)}","- holdout: locked","- thresholds changed after discovery: no"]
 (reports/"phase3c_replication_summary.md").write_text("\n".join(summary));(reports/"phase3c_replication_summary.html").write_text("<!doctype html><meta charset='utf-8'><pre>"+"\n".join(summary)+"</pre>")

def build_final_reports(root,relationships,stats):
 root=Path(root);reports=root/"reports";reports.mkdir(parents=True,exist_ok=True);relationships.to_csv(reports/"phase3c_replication_relationships.csv",index=False)
 nvda=relationships[(relationships.source_ticker=="NVDA")&(relationships.target_ticker=="ANET")&(relationships.interval=="5m")&(relationships.horizon_minutes==5)]
 candidates=relationships[relationships.discovery_status=="candidate"]
 counts=relationships.replication_result.value_counts().to_dict();summary=["# Phase 3C Out-of-Time Replication","",*(f"- {k}: {v}" for k,v in stats.items()),"",*(f"- {k}: {v}" for k,v in counts.items()),"","Results are associations, not causal findings or trading signals."]
 (reports/"phase3c_replication_summary.md").write_text("\n".join(summary));(reports/"phase3c_replication_summary.html").write_text("<!doctype html><meta charset='utf-8'><pre>"+"\n".join(summary)+"</pre>"+relationships.to_html(index=False))
 (reports/"nvda_anet_replication.html").write_text("<!doctype html><meta charset='utf-8'><h1>NVDA → ANET frozen replication</h1>"+nvda.to_html(index=False))
 (reports/"candidate_replication.html").write_text("<!doctype html><meta charset='utf-8'><h1>Frozen discovery candidates</h1>"+candidates.to_html(index=False))
 for source,name in [(root/"rolling_stability/rolling_stability.parquet","rolling_stability.html"),(root/"placebo/placebo_results.parquet","placebo_results.html")]:
  frame=pd.read_parquet(source) if source.exists() else pd.DataFrame();(reports/name).write_text("<!doctype html><meta charset='utf-8'>"+frame.to_html(index=False))
