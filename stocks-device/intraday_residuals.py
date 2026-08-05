"""Historical same-slot OLS using previous sessions only."""
from __future__ import annotations
import numpy as np,pandas as pd

def build_intraday_residuals(features,market="SPY",sector="SMH",history_sessions=20,min_sessions=10):
    keys=["timestamp","session_date","slot"]
    factor=features[features.ticker==market][keys+["log_bar_return"]].rename(columns={"log_bar_return":"market_return"})
    factor=factor.merge(features[features.ticker==sector][keys+["log_bar_return"]].rename(columns={"log_bar_return":"sector_return"}),on=keys)
    pieces=[]
    for ticker,base in features.groupby("ticker"):
      if ticker in {market,sector}:continue
      data=base.merge(factor,on=keys,how="inner").sort_values(["slot","timestamp"]);rows=[]
      for slot,group in data.groupby("slot"):
        group=group.sort_values("timestamp")
        for position,(_,row) in enumerate(group.iterrows()):
          history=group.iloc[max(0,position-history_sessions):position].dropna(subset=["log_bar_return","market_return","sector_return"])
          record=row.to_dict();record.update(raw_bar_return=row.log_bar_return,observations_used=len(history),residual_status="insufficient_history")
          if len(history)>=min_sessions:
            X=np.column_stack([np.ones(len(history)),history.market_return,history.sector_return]);y=history.log_bar_return.to_numpy(float)
            beta,*_=np.linalg.lstsq(X,y,rcond=None);pred=float(beta[0]+beta[1]*row.market_return+beta[2]*row.sector_return)
            market_pred=float(beta[0]+beta[1]*row.market_return)
            record.update(alpha=float(beta[0]),market_beta=float(beta[1]),sector_beta=float(beta[2]),
              market_adjusted_bar_residual=float(row.log_bar_return-market_pred),
              market_sector_adjusted_bar_residual=float(row.log_bar_return-pred),predicted_common_return=pred,residual_status="ok")
          rows.append(record)
      pieces.append(pd.DataFrame(rows))
    result=pd.concat(pieces,ignore_index=True)
    result=result.sort_values(["ticker","slot","timestamp"])
    result["residual_historical_mean"]=result.groupby(["ticker","slot"])["market_sector_adjusted_bar_residual"].transform(lambda s:s.shift(1).rolling(history_sessions,min_periods=min_sessions).mean())
    result["residual_historical_std"]=result.groupby(["ticker","slot"])["market_sector_adjusted_bar_residual"].transform(lambda s:s.shift(1).rolling(history_sessions,min_periods=min_sessions).std())
    result["residual_z_score"]=(result.market_sector_adjusted_bar_residual-result.residual_historical_mean)/result.residual_historical_std.replace(0,np.nan)
    return result.sort_values(["timestamp","ticker"])
