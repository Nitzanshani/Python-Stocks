"""Causal rolling market and sector residual models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _causal_rolling_ols(y: pd.Series, factors: pd.DataFrame, window: int,
                        min_observations: int) -> pd.DataFrame:
    aligned = pd.concat([y.rename("y"), factors], axis=1, join="inner").dropna()
    names = list(factors.columns)
    rows = []
    for position, (date, current) in enumerate(aligned.iterrows()):
        history = aligned.iloc[max(0, position - window):position]
        row = {"timestamp": date, "observations_used": len(history),
               "estimation_window": window, "residual_status": "insufficient_history"}
        if len(history) < min_observations:
            rows.append(row); continue
        x = np.column_stack([np.ones(len(history)), history[names].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(x, history["y"].to_numpy(float), rcond=None)
        prediction = float(beta[0] + np.dot(beta[1:], current[names].to_numpy(float)))
        row.update(alpha=float(beta[0]), predicted_common_return=prediction,
                   residual_return=float(current["y"] - prediction), residual_status="ok")
        for name, value in zip(names, beta[1:]):
            row[f"{name}_beta"] = float(value)
        rows.append(row)
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def build_residual_returns(feature_panel: pd.DataFrame, market: str = "SPY",
                           sector: str = "SMH", window: int = 126,
                           min_observations: int = 63) -> pd.DataFrame:
    indexed = feature_panel.set_index(["timestamp", "ticker"])["log_return"].unstack()
    if market not in indexed or sector not in indexed:
        raise ValueError(f"Required control series missing: {market}, {sector}")
    output = []
    for ticker in indexed.columns:
        if ticker in {market, sector}:
            continue
        common = pd.concat([indexed[ticker].rename("raw_return"),
                            indexed[market].rename("market_return"),
                            indexed[sector].rename("sector_return")], axis=1).dropna()
        market_fit = _causal_rolling_ols(common.raw_return,
            common[["market_return"]], window, min_observations).add_prefix("market_")
        sector_fit = _causal_rolling_ols(common.raw_return,
            common[["market_return", "sector_return"]], window, min_observations)
        result = common.join(sector_fit, how="left").join(market_fit, how="left")
        result.insert(0, "ticker", ticker)
        result["residual_model"] = "market_and_sector"
        own_features = feature_panel.loc[feature_panel.ticker == ticker,
            ["timestamp", "relative_volume", "rolling_volatility", "rolling_z_score"]
        ].set_index("timestamp")
        result = result.join(own_features, how="left")
        output.append(result.reset_index())
    return pd.concat(output, ignore_index=True).sort_values(["ticker", "timestamp"])
