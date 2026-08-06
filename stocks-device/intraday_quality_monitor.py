"""Persistent Phase 3C intraday quality monitor and replication gate."""
from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
from intraday_audit import audit_intraday

MATERIAL_FIELDS=("duplicate_timestamps","out_of_session_bars","invalid_ohlc_rows","negative_volume_rows")

def build_quality_monitor(symbols,intervals,output_root,previous=None):
 root=Path(output_root);report=audit_intraday(symbols,intervals,root)
 previous=previous if previous is not None else pd.DataFrame()
 prev={(r.ticker,r.interval):r for r in previous.itertuples()} if not previous.empty else {}
 report["sessions_added"]=[max(0,int(r.complete_sessions)-int(getattr(prev.get((r.ticker,r.interval)),"complete_sessions",0))) for r in report.itertuples()]
 penalties=(report.incomplete_sessions*8+report.missing_sessions*10+report.duplicate_timestamps*10+
            report.out_of_session_bars*2+report.invalid_ohlc_rows*10+report.negative_volume_rows*10)
 report["quality_score"]=(100-penalties).clip(0,100)*report.coverage_ratio.fillna(0)
 report["warnings"]=report.apply(lambda r:", ".join([name for name in MATERIAL_FIELDS if int(r.get(name,0))>0]+(["incomplete_sessions"] if int(r.get("incomplete_sessions",0)) else [])+(["low_coverage"] if float(r.get("coverage_ratio",0))<.98 else [])),axis=1)
 report["quality_gate_passed"]=(report.status=="available")&(report.quality_score>=90)&(report[list(MATERIAL_FIELDS)].sum(axis=1)==0)
 target=root/"reports/intraday_data_quality.html";target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text("<!doctype html><meta charset='utf-8'><h1>Phase 3C Intraday Data Quality</h1>"+report.to_html(index=False))
 report.to_csv(root/"reports/intraday_data_quality.csv",index=False)
 return report

def require_quality_gate(report):
 failed=report[~report.quality_gate_passed]
 if not failed.empty:raise RuntimeError("Material intraday data quality failure: "+", ".join(f"{r.ticker}:{r.interval}" for r in failed.itertuples()))
 return True
