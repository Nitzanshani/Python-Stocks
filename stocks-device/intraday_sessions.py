"""Exchange-session annotation for closed intraday bars."""

from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pandas as pd

INTERVAL_MINUTES={"5m":5,"15m":15,"30m":30,"60m":60}

def nyse_schedule(start,end):
    import pandas_market_calendars as mcal
    schedule=mcal.get_calendar("NYSE").schedule(start_date=start,end_date=end)
    schedule["market_open"]=pd.to_datetime(schedule.market_open,utc=True)
    schedule["market_close"]=pd.to_datetime(schedule.market_close,utc=True)
    return schedule

def annotate_intraday_bars(frame:pd.DataFrame,interval:str,now=None)->pd.DataFrame:
    if frame.empty:return frame.copy()
    minutes=INTERVAL_MINUTES[interval]; result=frame.copy(); result.index=pd.to_datetime(result.index,utc=True)
    result=result[~result.index.duplicated(keep="last")].sort_index(); local=result.index.tz_convert("America/New_York")
    schedule=nyse_schedule(local.min().date(),local.max().date()); mapping={d.date():(r.market_open,r.market_close) for d,r in schedule.iterrows()}
    session_dates=[]; opens=[]; closes=[]
    for timestamp in result.index:
        day=timestamp.tz_convert("America/New_York").date(); pair=mapping.get(day,(pd.NaT,pd.NaT)); session_dates.append(day); opens.append(pair[0]); closes.append(pair[1])
    result["session_date"]=session_dates; result["session_open"]=opens; result["session_close"]=closes
    result["bar_start"]=result.index; result["bar_end"]=result.index+pd.Timedelta(minutes=minutes)
    result["is_regular_session"]=(result.bar_start>=result.session_open)&(result.bar_start<result.session_close)
    result["is_short_session"]=(result.session_close-result.session_open)<pd.Timedelta(hours=6,minutes=30)
    result["minutes_from_open"]=(result.bar_start-result.session_open).dt.total_seconds()/60
    result["minutes_to_close"]=(result.session_close-result.bar_end).dt.total_seconds()/60
    now=pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC")
    result["bar_closed"]=result.bar_end<=now
    return result
