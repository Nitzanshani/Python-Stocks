"""Causal daily features for Phase 3A predictive-influence research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_return_features(frame: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Build returns and causal context; event-day data is excluded from baselines."""
    required = {"Close", "Adj Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing market-data columns: {sorted(missing)}")
    adjusted = pd.to_numeric(frame["Adj Close"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")
    out = pd.DataFrame(index=frame.index.copy())
    out.index.name = "timestamp"
    out["simple_return"] = close.pct_change(fill_method=None)
    out["log_return"] = np.log(adjusted / adjusted.shift(1))
    out["adjusted_close_return"] = adjusted.pct_change(fill_method=None)
    history = out["log_return"].shift(1)
    out["rolling_mean"] = history.rolling(window, min_periods=window).mean()
    out["rolling_volatility"] = history.rolling(window, min_periods=window).std(ddof=1)
    out["rolling_z_score"] = ((out["log_return"] - out["rolling_mean"]) /
                              out["rolling_volatility"].replace(0, np.nan))
    historical_volume = volume.shift(1).rolling(window, min_periods=window).mean()
    out["volume"] = volume
    out["relative_volume"] = volume / historical_volume.replace(0, np.nan)
    return out


def build_feature_panel(frames: dict[str, pd.DataFrame], window: int = 63) -> pd.DataFrame:
    pieces = []
    for ticker, frame in frames.items():
        features = build_return_features(frame, window).reset_index()
        features.insert(0, "ticker", ticker)
        pieces.append(features)
    return pd.concat(pieces, ignore_index=True).sort_values(["ticker", "timestamp"])
