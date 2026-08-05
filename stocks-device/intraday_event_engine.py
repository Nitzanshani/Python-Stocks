"""Causal intraday events and non-retrospective representatives."""
from __future__ import annotations
import hashlib,numpy as np,pandas as pd

def _id(row,event_type,name):
    return hashlib.sha256(f"{row.ticker}|{row.interval}|{row.timestamp}|{event_type}|{name}".encode()).hexdigest()[:24]

def detect_intraday_events(residuals,config):
    t=config["event_thresholds"];records=[]
    def add(row,event_type,direction,name,value,start=None):
      records.append({"event_id":_id(row,event_type,name),"ticker":row.ticker,"session_date":row.session_date,
       "event_timestamp":row.timestamp,"available_at":row.bar_end,"interval":row.interval,"event_type":event_type,
       "direction":direction,"start_timestamp":start or row.bar_start,"end_timestamp":row.bar_end,
       "start_price":row.Open,"end_price":row.Close,"raw_return":row.log_bar_return,
       "residual_return":row.market_sector_adjusted_bar_residual,"return_z_score":row.residual_z_score,
       "relative_volume":row.relative_bar_volume,"return_from_open":row.return_from_open,
       "return_from_previous_close":row.return_from_previous_close,"market_return":row.market_return,
       "sector_return":row.sector_return,"minutes_from_open":row.minutes_from_open,
       "threshold_name":name,"threshold_value":value,"engine_version":config["engine_version"]})
    for _,row in residuals[residuals.residual_status=="ok"].iterrows():
      direction="positive" if row.market_sector_adjusted_bar_residual>=0 else "negative"
      z=abs(row.residual_z_score);rv=row.relative_bar_volume
      if z>=t["residual_z"]:add(row,"bar_shock",direction,"residual_z",t["residual_z"])
      if rv>=t["relative_volume"]:add(row,"volume_shock",direction,"relative_volume",t["relative_volume"])
      if z>=t["residual_z"] and rv>=t["relative_volume"]:add(row,"price_volume_shock",direction,"combined",1)
      if abs(row.return_from_open)>=t["cumulative_return"]:add(row,"cumulative_shock","positive" if row.return_from_open>=0 else "negative","return_from_open",t["cumulative_return"])
    # First threshold crossing per ticker/session/reference/direction/level.
    for (ticker,session),group in residuals.groupby(["ticker","session_date"]):
      group=group.sort_values("timestamp")
      for reference in ["return_from_open","return_from_previous_close"]:
       for level in t["crossings"]:
        for sign,label in [(1,"positive"),(-1,"negative")]:
          hit=group[sign*group[reference]>=level]
          if not hit.empty:add(hit.iloc[0],"threshold_crossing",label,f"{reference}_{sign*level:+.2f}",sign*level,group.iloc[0].bar_start)
    return pd.DataFrame(records).sort_values(["ticker","session_date","event_timestamp","event_type"]).reset_index(drop=True)

def cluster_intraday_events(events,cluster_minutes=15):
    clusters=[]
    for _,group in events.groupby(["ticker","session_date","direction"]):
      current=[]
      for _,row in group.sort_values("available_at").iterrows():
        if current and row.available_at-current[-1].available_at>pd.Timedelta(minutes=cluster_minutes):clusters.append(current);current=[]
        current.append(row)
      if current:clusters.append(current)
    records=[];representatives=[]
    for rows in clusters:
      f=pd.DataFrame(rows);first=f.sort_values("available_at").iloc[0];magnitude=f.return_z_score.abs().fillna(0)+f.raw_return.abs().fillna(0);maximum=f.loc[magnitude.idxmax()]
      cid=hashlib.sha256("|".join(sorted(f.event_id)).encode()).hexdigest()[:24]
      records.append({"cluster_id":cid,"ticker":first.ticker,"session_date":first.session_date,"direction":first.direction,
       "first_detected_event_id":first.event_id,"maximum_magnitude_event_id":maximum.event_id,
       "cluster_start":f.available_at.min(),"cluster_end":f.available_at.max(),"cluster_event_count":len(f)})
      rep=first.copy();rep["cluster_id"]=cid;rep["maximum_magnitude_event_id"]=maximum.event_id;representatives.append(rep)
    return pd.DataFrame(records),pd.DataFrame(representatives)
