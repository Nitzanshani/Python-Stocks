"""Same-session future-bar responses measured in trading minutes."""
from __future__ import annotations
import numpy as np,pandas as pd
from intraday_sessions import INTERVAL_MINUTES

def measure_intraday_responses(events,residuals,targets,horizons):
    panels={(t,d):g.sort_values("bar_start") for (t,d),g in residuals.groupby(["ticker","session_date"])};rows=[]
    for _,e in events.iterrows():
      for target in targets:
       if target==e.ticker:continue
       panel=panels.get((target,e.session_date))
       for horizon in horizons:
        base={"event_id":e.event_id,"source_ticker":e.ticker,"target_ticker":target,"session_date":e.session_date,
              "event_available_at":e.available_at,"horizon_minutes":horizon,"source_direction":e.direction,
              "source_amplitude":e.residual_return,"response_available":False}
        interval=INTERVAL_MINUTES[e.interval]
        if horizon<interval or horizon%interval:
          rows.append({**base,"missing_reason":"insufficient_resolution"});continue
        if panel is None:
          rows.append({**base,"missing_reason":"missing_target_session"});continue
        future=panel[(panel.bar_start>=e.available_at)&(panel.bar_end<=e.available_at+pd.Timedelta(minutes=horizon))]
        expected=horizon//interval
        if len(future)!=expected:
          reason="session_boundary" if e.available_at+pd.Timedelta(minutes=horizon)>panel.session_close.iloc[0] else "missing_bar"
          rows.append({**base,"missing_reason":reason});continue
        raw=future.raw_bar_return;res=future.market_sector_adjusted_bar_residual
        raw_path=np.expm1(raw.cumsum());res_path=np.expm1(res.cumsum());target_amp=float(res_path.iloc[-1]);den=e.residual_return
        rows.append({**base,"response_start":future.bar_start.iloc[0],"response_end":future.bar_end.iloc[-1],
          "point_raw_return":float(np.expm1(raw.iloc[-1])),"point_residual_return":float(np.expm1(res.iloc[-1])),
          "cumulative_raw_return":float(raw_path.iloc[-1]),"cumulative_residual_return":target_amp,
          "maximum_positive_excursion":float(res_path.max()),"maximum_negative_excursion":float(res_path.min()),
          "time_to_max_positive":int((res_path.to_numpy().argmax()+1)*interval),"time_to_max_negative":int((res_path.to_numpy().argmin()+1)*interval),
          "same_direction_as_source":bool(np.sign(target_amp)==(1 if e.direction=="positive" else -1)),
          "reversal_before_horizon":bool((np.sign(res.to_numpy())*(1 if e.direction=="positive" else -1)<0).any()),
          "target_amplitude":target_amp,"amplitude_ratio":target_amp/den if abs(den)>1e-6 else np.nan,
          "absolute_transfer":abs(target_amp/den) if abs(den)>1e-6 else np.nan,
          "signed_transfer":target_amp/den if abs(den)>1e-6 else np.nan,
          "response_available":True,"missing_reason":None})
    return pd.DataFrame(rows)
