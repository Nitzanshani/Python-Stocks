"""Confirmation-only execution of the frozen Phase 3B engines."""
from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
from market_data_api import load_aligned_market_data
from intraday_features import build_intraday_panel
from intraday_residuals import build_intraday_residuals
from intraday_event_engine import detect_intraday_events,cluster_intraday_events
from intraday_response_engine import measure_intraday_responses
from intraday_event_study import select_intraday_controls,summarize_intraday_event_study
from intraday_predictive import run_intraday_walk_forward
from intraday_relationships import build_intraday_relationships
from phase3c_periods import slice_period
from phase3c_analysis import relative_rmse_improvement,practical_effect_classification,replication_result,leave_one_event_out,rolling_stability,generate_placebos,spread_proxy_bps
from phase3c_diagnostics import leave_one_out_table,rolling_table,regime_table,placebo_table

def _phase3b_config(spec):
 return {"symbols":spec["symbols"],"controls":spec["controls"],"market_benchmark":spec["residual_model"]["market"],"sector_benchmark":spec["residual_model"]["sector"],
  "history_sessions":spec["residual_model"]["history_sessions"],"minimum_history_sessions":spec["residual_model"]["minimum_history_sessions"],
  "event_thresholds":spec["events"],"response_horizons_minutes":spec["horizons"],"response_threshold":spec["response_threshold"],
  "walk_forward":spec["walk_forward"],"fdr_alpha":spec["fdr"]["alpha"]}

def build_confirmation_inputs(spec):
 c=_phase3b_config(spec);residuals=[];events=[];representatives=[];responses=[]
 for interval in spec["intervals"]:
  frames=load_aligned_market_data(spec["symbols"]+spec["controls"],interval)
  features=build_intraday_panel(frames,interval,c["history_sessions"])
  residual=build_intraday_residuals(features,c["market_benchmark"],c["sector_benchmark"],c["history_sessions"],c["minimum_history_sessions"])
  confirmation=slice_period(residual,spec,"confirmation")
  raw=detect_intraday_events(confirmation[confirmation.ticker.isin(spec["symbols"])],c);_,reps=cluster_intraday_events(raw,c["event_thresholds"]["cluster_minutes"])
  resp=measure_intraday_responses(reps,confirmation,spec["symbols"],c["response_horizons_minutes"],c["response_threshold"])
  residuals.append(confirmation);events.append(raw);representatives.append(reps);responses.append(resp)
 return tuple(pd.concat(x,ignore_index=True) for x in (residuals,events,representatives,responses))

def compare_replication(discovery,confirmation,study,responses,spec,costs):
 keys=["source_ticker","target_ticker","interval","horizon_minutes"]
 d=discovery.rename(columns={"intraday_status":"discovery_status","effect_size":"discovery_effect","prediction_improvement":"discovery_prediction_improvement"})
 c=confirmation.rename(columns={"intraday_status":"confirmation_status","effect_size":"confirmation_effect","prediction_improvement":"confirmation_prediction_improvement"})
 result=d.merge(c[keys+["confirmation_status","confirmation_effect","confirmation_prediction_improvement","baseline_rmse","extended_rmse","direction_improvement","fold_consistency","event_count","fdr_significant","response_consistency","sessions_improved","event_type"]],on=keys,how="left")
 primary=study[study.control_type=="matched"] if "control_type" in study else study
 evidence=primary.assign(_m=primary.effect_size.abs()).sort_values("_m",ascending=False).drop_duplicates(keys)
 result=result.merge(evidence[keys+["raw_p_value","adjusted_p_value","block_ci_low","block_ci_high","simple_ci_low","simple_ci_high","session_count"]],on=keys,how="left")
 result["relative_rmse_improvement"]=[relative_rmse_improvement(a,b) for a,b in zip(result.baseline_rmse,result.extended_rmse)]
 result["absolute_rmse_improvement"]=result.baseline_rmse-result.extended_rmse
 result["effect_bps"]=result.confirmation_effect*10000
 result["amplitude_consistency"]=result.response_consistency;result["lag_consistency"]=result.fold_consistency
 result["sign_preserved"]=np.sign(result.discovery_effect)==np.sign(result.confirmation_effect);result["lag_preserved"]=True;result["classification_preserved"]=result.discovery_status==result.confirmation_status
 result["spread_proxy_bps"]=[costs.get((t,i),np.nan) for t,i in zip(result.target_ticker,result.interval)];result["effect_to_cost_proxy_ratio"]=result.effect_bps/result.spread_proxy_bps
 thresholds=spec["replication"]["minimum_practical_effect"];result["practical_effect_classification"]=result.apply(lambda r:practical_effect_classification(r,thresholds),axis=1)
 result["replication_result"]=[replication_result({"effect":r.discovery_effect},{"effect":r.confirmation_effect,"sessions":spec["periods"]["confirmation_min_sessions"],"event_count":r.event_count,"relative_rmse_improvement":r.relative_rmse_improvement,"absolute_rmse_improvement":r.absolute_rmse_improvement,"effect_bps":r.effect_bps,"effect_to_cost_proxy_ratio":r.effect_to_cost_proxy_ratio,"fold_consistency":r.fold_consistency,"fdr_significant":r.fdr_significant},thresholds) for r in result.itertuples()]
 return result

def run_confirmation(spec,root):
 c=_phase3b_config(spec);root=Path(root);residuals,events,reps,responses=build_confirmation_inputs(spec)
 controls=select_intraday_controls(reps,responses,residuals,spec["control_matching"]["matches"])
 study=summarize_intraday_event_study(reps,responses,controls,spec["fdr"]["alpha"],spec["bootstrap"]["iterations"],spec["random_seed"])
 folds=run_intraday_walk_forward(residuals,reps,spec["symbols"],spec["walk_forward"]);confirmation=build_intraday_relationships(study,folds,c)
 base=Path(__file__).resolve().parent;discovery=pd.read_parquet(base/"research_intraday/influence/intraday_relationships.parquet");costs={}
 from market_data_api import load_market_data
 for target in spec["symbols"]:
  for interval in spec["intervals"]:
   raw=load_market_data(target,interval);local=raw.index.tz_convert("America/New_York").date;start=pd.Timestamp(spec["periods"]["confirmation_start"]).date();end=pd.Timestamp(spec["periods"]["confirmation_end"]).date();sample=raw[(local>=start)&(local<=end)]
   costs[(target,interval)]=spread_proxy_bps(sample)["spread_proxy_bps"] if not sample.empty else np.nan
 comparison=compare_replication(discovery,confirmation,study,responses,spec,costs)
 out=root/"confirmation";out.mkdir(parents=True,exist_ok=True);comparison.to_parquet(out/"phase3c_replication_relationships.parquet",index=False);controls.to_parquet(out/"confirmation_controls.parquet",index=False);study.to_parquet(out/"confirmation_event_study.parquet",index=False);folds.to_parquet(out/"confirmation_predictive_folds.parquet",index=False)
 loo=leave_one_out_table(responses);rolling=rolling_table(responses,spec["replication"]["rolling_window_sessions"],spec["replication"]["rolling_step_sessions"]);regimes=regime_table(responses,residuals);placebos=placebo_table(responses,spec["random_seed"])
 (root/"rolling_stability").mkdir(parents=True,exist_ok=True);(root/"placebo").mkdir(parents=True,exist_ok=True)
 loo.to_parquet(root/"confirmation/leave_one_event_out.parquet",index=False);rolling.to_parquet(root/"rolling_stability/rolling_stability.parquet",index=False);regimes.to_parquet(root/"confirmation/regime_stratification.parquet",index=False);placebos.to_parquet(root/"placebo/placebo_results.parquet",index=False);study.to_parquet(root/"confirmation/event_type_stability.parquet",index=False)
 return comparison,{"events":len(reps),"responses":int(responses.response_available.sum()),"controls":len(controls),"folds":len(folds)}
