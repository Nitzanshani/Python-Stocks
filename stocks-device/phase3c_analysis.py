"""Frozen Phase 3C replication diagnostics; no parameter search is performed."""
from __future__ import annotations
import hashlib,numpy as np,pandas as pd

def relative_rmse_improvement(baseline,extended):
 return (baseline-extended)/baseline if baseline and np.isfinite(baseline) else np.nan

def spread_proxy_bps(bars):
 """Documented cost proxy, not a historical bid/ask spread."""
 close=bars.Close.replace(0,np.nan);range_bps=((bars.High-bars.Low)/close*10000).replace([np.inf,-np.inf],np.nan)
 abs_return_bps=np.log(close).diff().abs()*10000
 volume_adjusted=range_bps/np.sqrt(pd.to_numeric(bars.Volume,errors="coerce").clip(lower=1))
 return {"range_proxy_bps":float(range_bps.median()),"median_absolute_return_bps":float(abs_return_bps.median()),"volume_adjusted_range":float(volume_adjusted.median()),"spread_proxy_bps":float(np.nanmedian([range_bps.median(),abs_return_bps.median()]))}

def practical_effect_classification(row,thresholds):
 required=[row.get("relative_rmse_improvement"),row.get("absolute_rmse_improvement"),row.get("effect_bps"),row.get("effect_to_cost_proxy_ratio")]
 if any(pd.isna(x) for x in required):return "insufficient_precision"
 statistical=bool(row.get("fdr_significant",False));practical=(row["relative_rmse_improvement"]>=thresholds["relative_rmse_improvement"] and row["absolute_rmse_improvement"]>=thresholds["absolute_rmse_improvement"] and abs(row["effect_bps"])>=thresholds["effect_bps"] and abs(row["effect_to_cost_proxy_ratio"])>=thresholds["effect_to_cost_proxy_ratio"])
 if practical:return "potentially_meaningful"
 if statistical:return "statistically_detectable_but_negligible"
 return "practically_small"

def replication_result(discovery,confirmation,thresholds):
 if confirmation.get("sessions",0)<60 or confirmation.get("event_count",0)<thresholds["minimum_events"]:return "insufficient_data"
 d=np.sign(discovery.get("effect",np.nan));c=np.sign(confirmation.get("effect",np.nan))
 if not np.isfinite(d) or not np.isfinite(c):return "insufficient_data"
 if d!=c:return "reversed"
 practical=practical_effect_classification(confirmation,thresholds)=="potentially_meaningful"
 predictive=confirmation.get("relative_rmse_improvement",0)>0 and confirmation.get("fold_consistency",0)>=thresholds["fold_consistency"]
 if practical and predictive and confirmation.get("fdr_significant",False):return "replicated"
 if predictive:return "direction_only"
 return "weakened" if abs(confirmation.get("effect",0))<abs(discovery.get("effect",0)) else "not_replicated"

def leave_one_event_out(values):
 values=np.asarray(values,float);full=float(np.mean(values)) if len(values) else np.nan
 if len(values)<2:return {"max_single_event_contribution":np.nan,"leave_one_out_min_effect":np.nan,"leave_one_out_max_effect":np.nan,"classification_changes":0,"event_concentrated":True}
 loo=np.array([np.mean(np.delete(values,i)) for i in range(len(values))]);contribution=np.max(np.abs(loo-full))
 changes=int(np.sum(np.sign(loo)!=np.sign(full)))
 return {"max_single_event_contribution":float(contribution),"leave_one_out_min_effect":float(loo.min()),"leave_one_out_max_effect":float(loo.max()),"classification_changes":changes,"event_concentrated":bool(changes or contribution>abs(full)*.5)}

def rolling_stability(frame,value="effect",window=20,step=5):
 rows=[];sessions=sorted(frame.session_date.unique())
 for start in range(0,len(sessions)-window+1,step):
  chosen=sessions[start:start+window];g=frame[frame.session_date.isin(chosen)][value].dropna();mean=g.mean()
  rows.append({"start":chosen[0],"end":chosen[-1],"sessions":window,"event_count":len(g),"effect":mean,"sign":int(np.sign(mean)),"ci_low":g.quantile(.025),"ci_high":g.quantile(.975)})
 return pd.DataFrame(rows)

def generate_placebos(frame,seed=0):
 """Deterministic null datasets preserving intraday rows and session structure."""
 rng=np.random.default_rng(seed);sessions=np.array(sorted(frame.session_date.unique()));result={}
 shuffled=frame.copy();mapping=dict(zip(sessions,rng.permutation(sessions)));shuffled["session_date"]=shuffled.session_date.map(mapping);result["shuffled_source"]=shuffled
 random_lag=frame.copy();random_lag["placebo_lag"]=rng.integers(4,13,len(frame));result["random_lag"]=random_lag
 wrong=frame.copy();mapping=dict(zip(sessions,np.roll(sessions,1)));wrong["session_date"]=wrong.session_date.map(mapping);result["wrong_day_source"]=wrong
 future=frame.copy();future["future_to_past_only_diagnostic"]=True;result["future_to_past"]=future
 return result

def pending_relationships(discovery,confirmation_sessions):
 result=discovery.copy();result=result.rename(columns={"intraday_status":"discovery_status","effect_size":"discovery_effect","prediction_improvement":"discovery_prediction_improvement"})
 result["confirmation_status"]="insufficient_data";result["confirmation_effect"]=np.nan;result["confirmation_prediction_improvement"]=np.nan
 result["sign_preserved"]=pd.NA;result["lag_preserved"]=pd.NA;result["classification_preserved"]=False;result["replication_result"]="insufficient_data";result["confirmation_sessions"]=confirmation_sessions
 return result

def deterministic_id(*parts):return hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:24]
