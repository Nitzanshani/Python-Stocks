"""Quality audit for Phase 3B local intraday coverage."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from intraday_sessions import INTERVAL_MINUTES,annotate_intraday_bars,nyse_schedule
from market_data_api import MarketDataUnavailable,load_market_data

def audit_intraday(symbols,intervals,output_root=Path("research_intraday")):
    rows=[]
    for interval in intervals:
      for ticker in symbols:
        try: raw=load_market_data(ticker,interval)
        except MarketDataUnavailable:
            rows.append({"ticker":ticker,"interval":interval,"status":"missing","row_count":0});continue
        frame=annotate_intraday_bars(raw,interval); regular=frame[frame.is_regular_session]
        counts=regular.groupby("session_date").size(); expected={}
        for date,group in regular.groupby("session_date"):
            expected[date]=int(np.ceil((group.session_close.iloc[0]-group.session_open.iloc[0]).total_seconds()/60/INTERVAL_MINUTES[interval]))
        complete=sum(counts.get(d,0)>=n for d,n in expected.items()); current_partial=int((~regular.bar_closed).any())
        required=raw[["Open","High","Low","Close"]].apply(pd.to_numeric,errors="coerce")
        invalid=(required.isna().any(axis=1)|(required<=0).any(axis=1)|
            (required.High<required[["Open","Close","Low"]].max(axis=1))|
            (required.Low>required[["Open","Close","High"]].min(axis=1))).sum()
        schedule=nyse_schedule(min(counts.index),max(counts.index)) if len(counts) else pd.DataFrame()
        rows.append({"ticker":ticker,"interval":interval,"status":"available",
          "first_timestamp":raw.index.min(),"last_timestamp":raw.index.max(),"row_count":len(raw),
          "number_of_sessions":len(counts),"complete_sessions":complete,
          "incomplete_sessions":len(counts)-complete,"current_incomplete_session":current_partial,
          "bars_per_session_median":float(counts.median()),"missing_sessions":max(0,len(schedule)-len(counts)),
          "duplicate_timestamps":int(raw.index.duplicated().sum()),
          "out_of_session_bars":int((~frame.is_regular_session).sum()),"invalid_ohlc_rows":int(invalid),
          "negative_volume_rows":int((pd.to_numeric(raw.Volume,errors="coerce")<0).sum()),
          "timezone_status":"UTC" if str(raw.index.tz)=="UTC" else str(raw.index.tz),
          "coverage_ratio":complete/max(1,len(schedule))})
    report=pd.DataFrame(rows); output_root=Path(output_root); (output_root/"reports").mkdir(parents=True,exist_ok=True)
    report.to_parquet(output_root/"reports/intraday_data_audit.parquet",index=False)
    report.to_csv(output_root/"reports/intraday_data_audit.csv",index=False)
    return report
