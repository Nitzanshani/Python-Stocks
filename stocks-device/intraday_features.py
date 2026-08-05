"""Causal time-of-day-normalized intraday features."""
from __future__ import annotations
import numpy as np,pandas as pd
from intraday_sessions import annotate_intraday_bars

def build_intraday_features(frame:pd.DataFrame,ticker:str,interval:str,history_sessions=20):
    x=annotate_intraday_bars(frame,interval); x=x[x.is_regular_session & x.bar_closed].copy()
    x["ticker"]=ticker;x["interval"]=interval;x["slot"]=x.minutes_from_open.astype(int)
    close=pd.to_numeric(x.Close,errors="coerce");volume=pd.to_numeric(x.Volume,errors="coerce")
    x["log_bar_return"]=np.log(close/close.shift(1));x["bar_return"]=close.pct_change(fill_method=None)
    first_open=x.groupby("session_date").Open.transform("first");x["return_from_open"]=close/first_open-1
    prior_close=x.groupby("session_date").Close.last().shift(1);x["previous_close"]=x.session_date.map(prior_close)
    x["return_from_previous_close"]=close/x.previous_close-1
    by_slot=x.groupby("slot",group_keys=False)
    mean=by_slot.log_bar_return.transform(lambda s:s.shift(1).rolling(history_sessions,min_periods=10).median())
    vol=by_slot.log_bar_return.transform(lambda s:s.shift(1).rolling(history_sessions,min_periods=10).std())
    median_volume=by_slot.Volume.transform(lambda s:s.shift(1).rolling(history_sessions,min_periods=10).median())
    x["rolling_intraday_volatility"]=vol;x["bar_return_z_score"]=(x.log_bar_return-mean)/vol.replace(0,np.nan)
    x["relative_bar_volume"]=volume/median_volume.replace(0,np.nan)
    x["cumulative_session_volume"]=x.groupby("session_date").Volume.cumsum()
    totals=x.groupby("session_date").Volume.sum();historical_total=totals.shift(1).rolling(history_sessions,min_periods=10).median()
    x["volume_share_of_session"]=x.cumulative_session_volume/x.session_date.map(historical_total).replace(0,np.nan)
    typical=(x.High+x.Low+x.Close)/3;cum_value=(typical*volume).groupby(x.session_date).cumsum()
    vwap=cum_value/x.cumulative_session_volume.replace(0,np.nan);x["distance_from_vwap"]=close/vwap-1
    x["time_of_day_bucket"]=(x.minutes_from_open//30).astype(int);x["available_at"]=x.bar_end
    return x.reset_index(names="timestamp")

def build_intraday_panel(frames,interval,history_sessions=20):
    return pd.concat([build_intraday_features(f,s,interval,history_sessions) for s,f in frames.items()],ignore_index=True).sort_values(["timestamp","ticker"])
