"""Session-purged intraday baseline versus source models."""
from __future__ import annotations
import numpy as np,pandas as pd
from predictive_influence import _ridge_fit,_ridge_predict,_scale_fit,_select_alpha

def session_walk_forward_splits(sessions,train_sessions=20,test_sessions=5,step_sessions=5,embargo_sessions=1):
 sessions=np.array(sorted(sessions));embargoed=set();fold=0;start=train_sessions
 while start+test_sessions<=len(sessions):
  train=list(sessions[max(0,start-train_sessions):start]);train=[d for d in train if d not in embargoed]
  test=list(sessions[start:start+test_sessions]);yield fold,train,test
  for d in sessions[start+test_sessions:start+test_sessions+embargo_sessions]:embargoed.add(d)
  fold+=1;start+=step_sessions

def _pair_dataset(residuals,events,source,target,horizon,interval):
 s=residuals[(residuals.ticker==source)&(residuals.interval==interval)].set_index("timestamp");t=residuals[(residuals.ticker==target)&(residuals.interval==interval)].set_index("timestamp")
 idx=s.index.intersection(t.index);s=s.loc[idx];t=t.loc[idx];data=pd.DataFrame(index=idx);base=[]
 for lag in range(1,4):
  for name,series in [(f"target_residual_lag{lag}",t.market_sector_adjusted_bar_residual.shift(lag)),(f"target_raw_lag{lag}",t.raw_bar_return.shift(lag)),(f"market_lag{lag}",t.market_return.shift(lag)),(f"sector_lag{lag}",t.sector_return.shift(lag))]:data[name]=series;base.append(name)
 for name,series in [("target_volatility",t.rolling_intraday_volatility),("target_relative_volume",t.relative_bar_volume),("time_of_day",t.minutes_from_open),("target_return_open",t.return_from_open)]:data[name]=series;base.append(name)
 extended=list(base)
 for lag in range(1,4):
  data[f"source_residual_lag{lag}"]=s.market_sector_adjusted_bar_residual.shift(lag);data[f"source_raw_lag{lag}"]=s.raw_bar_return.shift(lag);extended +=[f"source_residual_lag{lag}",f"source_raw_lag{lag}"]
 source_events=events[(events.ticker==source)&(events.interval==interval)].set_index("event_timestamp")
 data["source_event_flag"]=data.index.isin(source_events.index).astype(float);data["source_event_magnitude"]=source_events.return_z_score.abs().groupby(level=0).max().reindex(data.index).fillna(0)
 data["source_relative_volume"]=s.relative_bar_volume;data["source_crossing_flag"]=data.index.isin(source_events[source_events.event_type=="threshold_crossing"].index).astype(float)
 pos=np.where(data.source_event_flag.to_numpy()>0,np.arange(len(data)),np.nan);last=pd.Series(pos,index=data.index).ffill();data["minutes_since_source_event"]=(np.arange(len(data))-last).fillna(999)*{"5m":5,"15m":15,"30m":30,"60m":60}[interval]
 extended +=["source_event_flag","source_event_magnitude","source_relative_volume","source_crossing_flag","minutes_since_source_event"]
 bars=horizon//{"5m":5,"15m":15,"30m":30,"60m":60}[interval];series=t.market_sector_adjusted_bar_residual
 data["label"]=series.shift(-1).rolling(bars,min_periods=bars).sum().shift(-(bars-1));data["session_date"]=t.session_date;data["minutes_from_open"]=t.minutes_from_open
 # Labels are invalid when they leave the source session.
 end_session=t.session_date.shift(-bars);data.loc[end_session!=data.session_date,"label"]=np.nan
 return data.replace([np.inf,-np.inf],np.nan).dropna(),base,extended

def _logistic_fit(x,y,alpha,iterations=150,rate=.05):
 w=np.zeros(x.shape[1]+1);X=np.column_stack([np.ones(len(x)),np.clip(x,-20,20)]);target=(y>0).astype(float)
 for _ in range(iterations):
  with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
   z=np.clip(X@w,-30,30);p=1/(1+np.exp(-z));gradient=X.T@(p-target)/len(X)
  gradient[1:]+=alpha*w[1:]/len(X);w-=rate*gradient
 return w
def _logistic_predict(x,w):
 with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
  return 1/(1+np.exp(-np.clip(np.column_stack([np.ones(len(x)),np.clip(x,-20,20)])@w,-30,30)))>=.5

def run_intraday_walk_forward(residuals,events,symbols,settings):
 rows=[]
 for interval in sorted(residuals.interval.unique()):
  minutes={"5m":5,"15m":15,"30m":30,"60m":60}[interval]
  for horizon in [5,15,30]:
   if horizon<minutes or horizon%minutes:continue
   for source in symbols:
    for target in symbols:
     if source==target:continue
     data,base,extended=_pair_dataset(residuals,events,source,target,horizon,interval)
     if data.empty:continue
     for fold,train_sessions,test_sessions in session_walk_forward_splits(data.session_date.unique(),settings["train_sessions"],settings["test_sessions"],settings["step_sessions"],settings["embargo_sessions"]):
      train_mask=data.session_date.isin(train_sessions);final_train=max(train_sessions);train_mask &= ~((data.session_date==final_train)&(data.minutes_from_open>=(390-settings["purge_minutes"])))
      test_mask=data.session_date.isin(test_sessions);y=data.label.to_numpy(float);pred={};logistic={};alphas={}
      train=np.where(train_mask)[0];test=np.where(test_mask)[0]
      if len(train)<100 or len(test)<20:continue
      for name,cols in [("baseline",base),("extended",extended)]:
       x=data[cols].to_numpy(float);alpha=_select_alpha(x[train],y[train],settings["ridge_alphas"],1);mean,scale=_scale_fit(x[train]);ztrain=(x[train]-mean)/scale;ztest=(x[test]-mean)/scale
       pred[name]=_ridge_predict(ztest,_ridge_fit(ztrain,y[train],alpha));logistic[name]=_logistic_predict(ztest,_logistic_fit(ztrain,y[train],alpha));alphas[name]=alpha
      actual=y[test]
      def metric(p):return np.mean(abs(p-actual)),np.sqrt(np.mean((p-actual)**2)),np.mean(np.sign(p)==np.sign(actual))
      bm,br,bd=metric(pred["baseline"]);em,er,ed=metric(pred["extended"]);ols=_ridge_predict((data[extended].to_numpy(float)[test]-_scale_fit(data[extended].to_numpy(float)[train])[0])/_scale_fit(data[extended].to_numpy(float)[train])[1],_ridge_fit((data[extended].to_numpy(float)[train]-_scale_fit(data[extended].to_numpy(float)[train])[0])/_scale_fit(data[extended].to_numpy(float)[train])[1],y[train],0))
      rows.append({"source_ticker":source,"target_ticker":target,"interval":interval,"horizon_minutes":horizon,"fold":fold,"train_start":min(train_sessions),"train_end":max(train_sessions),"test_start":min(test_sessions),"test_end":max(test_sessions),"train_sessions":len(train_sessions),"test_sessions":len(test_sessions),"purge_minutes":settings["purge_minutes"],"embargo_sessions":settings["embargo_sessions"],"preprocessing_scope":"inside_fold_only","baseline_alpha":alphas["baseline"],"extended_alpha":alphas["extended"],"baseline_mae":bm,"extended_mae":em,"mae_improvement":bm-em,"baseline_rmse":br,"extended_rmse":er,"rmse_improvement":br-er,"baseline_direction_accuracy":bd,"extended_direction_accuracy":ed,"direction_improvement":ed-bd,"baseline_logistic_accuracy":float(np.mean(logistic["baseline"]==(actual>0))),"extended_logistic_accuracy":float(np.mean(logistic["extended"]==(actual>0))),"lagged_ols_rmse":float(np.sqrt(np.mean((ols-actual)**2))),"zero_rmse":float(np.sqrt(np.mean(actual**2))),"sample_size":len(test)})
 return pd.DataFrame(rows)
