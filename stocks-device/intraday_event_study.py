"""Intraday matched controls and session-level resampling."""
from __future__ import annotations
import hashlib,numpy as np,pandas as pd
from influence_statistics import apply_fdr_families

def select_intraday_controls(events,responses,residuals,matches=3):
 valid=responses[responses.response_available].copy();event_lookup=events.set_index("event_id");rows=[]
 factor=residuals.groupby(["interval","timestamp"])[["market_return","sector_return"]].first().reset_index()
 candidates=residuals.merge(factor,on=["interval","timestamp"],suffixes=("","_factor"))
 event_times={ticker:set(group.event_timestamp) for ticker,group in events.groupby("ticker")}
 panels={(t,i):g.sort_values("timestamp") for (t,i),g in residuals.groupby(["ticker","interval"])}
 pool_cache={}
 for (event_id,target_ticker),response_group in valid.groupby(["event_id","target_ticker"]):
  event=event_lookup.loc[event_id];first_response=response_group.iloc[0];source=panels[(first_response.source_ticker,event.interval)]
  if event_id not in pool_cache:
   event_weekday=pd.Timestamp(event.event_timestamp).tz_convert("America/New_York").dayofweek
   unconditional=source[(source.slot==int(event.minutes_from_open)) &
       (source.timestamp.dt.tz_convert("America/New_York").dt.dayofweek==event_weekday) &
       (~source.timestamp.isin(event_times.get(first_response.source_ticker,set())))].copy()
   pool_cache[event_id]=(unconditional,unconditional[
       (np.sign(unconditional.market_return)==np.sign(event.market_return)) &
       (np.sign(unconditional.sector_return)==np.sign(event.sector_return))].copy())
  unconditional,pool=pool_cache[event_id];unconditional=unconditional.copy();pool=pool.copy()
  target=panels[(target_ticker,event.interval)];target_index=target.set_index("timestamp")
  target_event=target_index.reindex([event.event_timestamp])
  target_return=float(target_event.return_from_open.iloc[0]) if not target_event.empty else 0.0
  target_volume=float(target_event.volume_share_of_session.iloc[0]) if not target_event.empty else 0.0
  pool["target_return_state"]=pool.timestamp.map(target_index.return_from_open)
  pool["target_volume_state"]=pool.timestamp.map(target_index.volume_share_of_session)
  pool["distance"]=((pool.market_return-event.market_return).abs()+
      (pool.sector_return-event.sector_return).abs()+
      (pool.return_from_open-event.return_from_open).abs()+
      (pool.relative_bar_volume-event.relative_volume).abs()*.01+
      (pool.target_return_state-target_return).abs()+
      (pool.target_volume_state-target_volume).abs()*.01)
  pool=pool.sort_values("distance").head(50)
  for response in response_group.itertuples():
   accepted=0
   for _,control in pool.iterrows():
    limit=target[target.session_date==control.session_date].session_close.iloc[0] if response.horizon_minutes==-1 else control.bar_end+pd.Timedelta(minutes=response.horizon_minutes)
    future=target[(target.session_date==control.session_date)&(target.bar_start>=control.bar_end)&(target.bar_end<=limit)]
    expected=int((limit-control.bar_end).total_seconds()/60)//({"5m":5,"15m":15,"30m":30,"60m":60}[event.interval])
    if len(future)!=expected:continue
    accepted+=1;value=float(np.expm1(future.market_sector_adjusted_bar_residual.sum()))
    rows.append({"control_id":hashlib.sha256(f"{response.event_id}|{response.target_ticker}|{response.horizon_minutes}|{control.timestamp}".encode()).hexdigest()[:24],
     "event_id":response.event_id,"source_ticker":response.source_ticker,"target_ticker":response.target_ticker,
     "interval":event.interval,"event_type":event.event_type,"horizon_minutes":response.horizon_minutes,
     "event_session":response.session_date,"control_session":control.session_date,"control_timestamp":control.timestamp,
     "match_rank":accepted,"distance":control.distance,"control_residual_response":value,"control_type":"matched",
     "matching_features":"slot,weekday,SPY_direction,SMH_direction,SPY_magnitude,SMH_magnitude,target_return_to_time,target_volume_to_time",
     "target_future_used_for_matching":False})
    if accepted>=matches:break
   # Unconditional baseline: same clock slot and weekday, but deliberately no
   # state matching. Selection uses only timestamps known before the response.
   accepted=0
   for _,control in unconditional.sort_values("timestamp").iterrows():
    limit=target[target.session_date==control.session_date].session_close.iloc[0] if response.horizon_minutes==-1 else control.bar_end+pd.Timedelta(minutes=response.horizon_minutes)
    future=target[(target.session_date==control.session_date)&(target.bar_start>=control.bar_end)&(target.bar_end<=limit)]
    expected=int((limit-control.bar_end).total_seconds()/60)//({"5m":5,"15m":15,"30m":30,"60m":60}[event.interval])
    if len(future)!=expected:continue
    accepted+=1;value=float(np.expm1(future.market_sector_adjusted_bar_residual.sum()))
    rows.append({"control_id":hashlib.sha256(f"unconditional|{response.event_id}|{response.target_ticker}|{response.horizon_minutes}|{control.timestamp}".encode()).hexdigest()[:24],
     "event_id":response.event_id,"source_ticker":response.source_ticker,"target_ticker":response.target_ticker,
     "interval":event.interval,"event_type":event.event_type,"horizon_minutes":response.horizon_minutes,
     "event_session":response.session_date,"control_session":control.session_date,"control_timestamp":control.timestamp,
     "match_rank":accepted,"distance":np.nan,"control_residual_response":value,"control_type":"unconditional",
     "matching_features":"slot,weekday only","target_future_used_for_matching":False})
    if accepted>=matches:break
 return pd.DataFrame(rows)

def summarize_intraday_event_study(events,responses,controls,alpha=.05,iterations=300,seed=0):
 lookup_columns=["event_id","event_type"]+([] if "interval" in responses else ["interval"])
 lookup=events[lookup_columns];valid=responses[responses.response_available].merge(lookup,on="event_id")
 groups=["source_ticker","target_ticker","interval","event_type","horizon_minutes"];rows=[]
 control_groups={key:g for key,g in controls.groupby(groups,sort=False)}
 for key,g in valid.groupby(groups):
  base_controls=control_groups.get(key,pd.DataFrame())
  if base_controls.empty:continue
  for control_type,c in base_controls.groupby("control_type"):
   ev=g.cumulative_residual_return.dropna().to_numpy();cv=c.control_residual_response.dropna().to_numpy()
   if len(ev)<3 or len(cv)<3:continue
   rng=np.random.default_rng(seed+int(hashlib.sha256(f"{key}|{control_type}".encode()).hexdigest()[:8],16));effect=ev.mean()-cv.mean();pooled=np.sqrt((ev.var(ddof=1)+cv.var(ddof=1))/2);perm=[]
   sessions=g.session_date.unique()
   for _ in range(iterations):
    shuffled=rng.permutation(np.r_[ev,cv]);perm.append(shuffled[:len(ev)].mean()-shuffled[len(ev):].mean())
   # Session-block bootstrap resamples whole event sessions.
   session_means=g.groupby("session_date").cumulative_residual_return.mean().to_numpy()
   event_boot=rng.choice(session_means,size=(iterations,len(session_means)),replace=True).mean(axis=1)
   control_boot=rng.choice(cv,size=(iterations,len(cv)),replace=True).mean(axis=1)
   block=event_boot-control_boot
   simple=(rng.choice(ev,size=(iterations,len(ev)),replace=True).mean(axis=1)-
           rng.choice(cv,size=(iterations,len(cv)),replace=True).mean(axis=1))
   rows.append(dict(zip(groups,key),control_type=control_type,event_count=len(ev),session_count=len(sessions),mean_response=float(ev.mean()),
     baseline_mean=float(cv.mean()),abnormal_effect=float(effect),effect_size=float(effect/pooled) if pooled else np.nan,
     response_consistency=float((np.sign(ev)==np.sign(ev.mean())).mean()),reversal_probability=float(g.reversal_before_horizon.mean()),
     simple_ci_low=float(np.percentile(simple,2.5)),simple_ci_high=float(np.percentile(simple,97.5)),
     block_ci_low=float(np.percentile(block,2.5)),block_ci_high=float(np.percentile(block,97.5)),
     raw_p_value=(1+sum(abs(x)>=abs(effect) for x in perm))/(iterations+1),residual_model="market_sector_historical",
     test_period=f"{g.session_date.min()}:{g.session_date.max()}"))
 result=pd.DataFrame(rows)
 if not result.empty:result["analysis_type"]="intraday_event_study"
 return apply_fdr_families(result,"raw_p_value",["analysis_type","control_type","interval","horizon_minutes","event_type","residual_model","test_period"],alpha) if not result.empty else result
