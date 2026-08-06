"""Predeclared rolling, regime, event concentration and placebo diagnostics."""
from __future__ import annotations
import numpy as np,pandas as pd
from phase3c_analysis import leave_one_event_out,rolling_stability,generate_placebos

KEYS=["source_ticker","target_ticker","interval","horizon_minutes"]

def leave_one_out_table(responses):
 valid=responses[responses.response_available];rows=[]
 for key,g in valid.groupby(KEYS):rows.append(dict(zip(KEYS,key),event_count=len(g),**leave_one_event_out(g.cumulative_residual_return.dropna())))
 return pd.DataFrame(rows)

def rolling_table(responses,window=20,step=5):
 valid=responses[responses.response_available];rows=[]
 for key,g in valid.groupby(KEYS):
  renamed=g.rename(columns={"cumulative_residual_return":"effect"})
  roll=rolling_stability(renamed,"effect",window,step)
  for row in roll.to_dict("records"):rows.append({**dict(zip(KEYS,key)),**row})
 return pd.DataFrame(rows)

def regime_table(responses,residuals):
 valid=responses[responses.response_available].copy();state=residuals[["ticker","interval","timestamp","market_return","sector_return","rolling_intraday_volatility","minutes_from_open"]].rename(columns={"ticker":"source_ticker","timestamp":"event_available_at"})
 valid=valid.merge(state,on=["source_ticker","interval","event_available_at"],how="left");valid["spy_regime"]=np.where(valid.market_return>=0,"positive","negative");valid["smh_regime"]=np.where(valid.sector_return>=0,"positive","negative");valid["time_regime"]=pd.cut(valid.minutes_from_open,[-1,59,299,390],labels=["open","middle","close"])
 rows=[]
 for dimensions in [["spy_regime"],["smh_regime"],["time_regime"]]:
  for key,g in valid.groupby(KEYS+dimensions,observed=True):rows.append(dict(zip(KEYS+dimensions,key),regime_dimension=dimensions[0],event_count=len(g),effect_size=g.cumulative_residual_return.mean(),sign_consistency=(np.sign(g.cumulative_residual_return)==np.sign(g.cumulative_residual_return.mean())).mean()))
 return pd.DataFrame(rows)

def placebo_table(responses,seed=0):
 valid=responses[responses.response_available];rows=[]
 for key,g in valid.groupby(KEYS):
  x=g.source_amplitude.to_numpy(float);y=g.cumulative_residual_return.to_numpy(float);rng=np.random.default_rng(seed+len(rows));real=float(np.corrcoef(x,y)[0,1]) if len(g)>2 and np.std(x)>0 and np.std(y)>0 else np.nan
  variants={"shuffled_source_sessions":rng.permutation(x),"random_lag_4_to_12_bars":np.roll(x,int(rng.integers(4,13))),"wrong_day_source":np.roll(x,max(1,len(x)//max(1,g.session_date.nunique()))),"unrelated_pair":rng.permutation(x),"future_to_past_diagnostic":x[::-1]}
  for name,px in variants.items():
   effect=float(np.corrcoef(px,y)[0,1]) if len(g)>2 and np.std(px)>0 and np.std(y)>0 else np.nan
   rows.append({**dict(zip(KEYS,key)),"placebo_type":name,"event_count":len(y),"real_association":real,"placebo_association":effect,"real_exceeds_placebo":bool(abs(real)>abs(effect)) if np.isfinite(real) and np.isfinite(effect) else False,"future_to_past_legitimate":False if name=="future_to_past_diagnostic" else pd.NA})
 return pd.DataFrame(rows)
