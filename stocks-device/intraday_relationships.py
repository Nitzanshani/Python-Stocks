"""Response curves, statuses and Phase 3A comparison."""
from __future__ import annotations
import numpy as np,pandas as pd,warnings
from scipy.optimize import curve_fit

def response_curves(responses):
 # Session-close is retained in event studies, but it has no fixed minute coordinate
 # and therefore must not be forced into a decay curve.
 valid=responses[responses.response_available&(responses.horizon_minutes>0)];rows=[]
 for key,g in valid.groupby(["source_ticker","target_ticker","interval","horizon_minutes"]):
  values=g.cumulative_residual_return.dropna();rows.append({"source_ticker":key[0],"target_ticker":key[1],"interval":key[2],"minute":key[3],"mean":values.mean(),"median":values.median(),"ci_low":values.quantile(.025),"ci_high":values.quantile(.975),"events":len(values)})
 curves=pd.DataFrame(rows);summaries=[]
 for key,g in curves.groupby(["source_ticker","target_ticker","interval"]):
  g=g.sort_values("minute");x=g.minute.to_numpy(float);y=g["mean"].to_numpy(float);models=[]
  def score(name,pred,k,params):
   rss=max(1e-16,float(np.sum((y-pred)**2)));models.append((len(y)*np.log(rss/len(y))+2*k,name,params))
  score("no_decay",np.repeat(y.mean(),len(y)),1,{"level":float(y.mean())})
  if len(y)>=3:
   try:
    f=lambda z,a,k:a*np.exp(-k*z)
    with warnings.catch_warnings():warnings.simplefilter("ignore");p,_=curve_fit(f,x,y,p0=[y[0] or .001,.02],maxfev=5000)
    score("exponential_decay",f(x,*p),2,{"amplitude":float(p[0]),"decay_rate":float(p[1])})
   except Exception:pass
  if len(y)>=5:
   try:
    f=lambda z,a,k,w,phase:a*np.exp(-k*z)*np.cos(w*z+phase)
    with warnings.catch_warnings():warnings.simplefilter("ignore");p,_=curve_fit(f,x,y,p0=[y[0] or .001,.02,.05,0],maxfev=10000)
    score("damped_oscillation",f(x,*p),4,{"amplitude":float(p[0]),"decay_rate":float(p[1]),"omega":float(p[2]),"phase":float(p[3])})
   except Exception:pass
  best=min(models,key=lambda z:z[0]);peak=g.loc[g["mean"].abs().idxmax()]
  half=np.log(2)/best[2].get("decay_rate",np.nan) if best[1]=="exponential_decay" and best[2].get("decay_rate",0)>0 else np.nan
  summaries.append({"source_ticker":key[0],"target_ticker":key[1],"interval":key[2],"peak_amplitude":peak["mean"],"time_to_peak":peak.minute,"area_under_response_curve":float(np.trapz(y,x)),"selected_decay_model":best[1],"decay_aic":best[0],"half_life_minutes":half,"decay_parameters":str(best[2]),"reversal_probability":float(valid[(valid.source_ticker==key[0])&(valid.target_ticker==key[1])].reversal_before_horizon.mean())})
 return curves,pd.DataFrame(summaries)

def build_intraday_relationships(event_study,folds,config):
 pred=folds.groupby(["source_ticker","target_ticker","interval","horizon_minutes"]).agg(
  baseline_rmse=("baseline_rmse","mean"),extended_rmse=("extended_rmse","mean"),prediction_improvement=("rmse_improvement","mean"),
  direction_improvement=("direction_improvement","mean"),folds_improved=("rmse_improvement",lambda x:int((x>0).sum())),total_folds=("fold","count"),sample_size=("sample_size","sum"),improvement_std=("rmse_improvement","std")).reset_index()
 pred["fold_consistency"]=pred.folds_improved/pred.total_folds
 primary=event_study[event_study.control_type=="matched"] if "control_type" in event_study else event_study
 best=primary.assign(abs_effect=primary.effect_size.abs()).sort_values("abs_effect",ascending=False).drop_duplicates(["source_ticker","target_ticker","interval","horizon_minutes"])
 result=pred.merge(best[["source_ticker","target_ticker","interval","horizon_minutes","event_type","event_count","effect_size","abnormal_effect","adjusted_p_value","fdr_significant","response_consistency"]],on=["source_ticker","target_ticker","interval","horizon_minutes"],how="left")
 def status(r):
  if r.total_folds<2 or pd.isna(r.event_count) or r.event_count<5:return "insufficient_data"
  relative=r.prediction_improvement/r.baseline_rmse if r.baseline_rmse else 0
  if abs(r.effect_size)>=.3 and relative<=0:return "rejected"
  if relative>0 and r.fold_consistency>=.6 and bool(r.fdr_significant):return "validated"
  if relative>0 or bool(r.fdr_significant):return "candidate" if r.fold_consistency>=.5 else "unstable"
  return "no_evidence"
 result["intraday_status"]=result.apply(status,axis=1);result["sessions_improved"]=result.folds_improved*config["walk_forward"]["test_sessions"]
 return result.sort_values("prediction_improvement",ascending=False)

def compare_with_phase3a(intraday,daily_path):
 daily=pd.read_parquet(daily_path);best=intraday.sort_values("prediction_improvement",ascending=False).drop_duplicates(["source_ticker","target_ticker"])
 daily=daily.rename(columns={"relationship_status":"daily_status","rmse_improvement":"daily_prediction_improvement","horizon":"daily_best_horizon","adjusted_p_value_granger":"daily_granger_evidence"})
 result=best.merge(daily[["source_ticker","target_ticker","daily_status","daily_prediction_improvement","daily_best_horizon","daily_granger_evidence"]],on=["source_ticker","target_ticker"],how="left")
 return result.rename(columns={"intraday_status":"intraday_status","prediction_improvement":"intraday_prediction_improvement","horizon_minutes":"intraday_best_horizon","adjusted_p_value":"intraday_lead_lag_evidence"})
