"""Matched-control daily event study with reproducible resampling."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import stats


def select_control_days(events: pd.DataFrame, responses: pd.DataFrame,
                        residuals: pd.DataFrame, matches_per_event: int = 10) -> pd.DataFrame:
    """Match on source-event absence and contemporaneous factor state only."""
    factor = residuals.groupby("timestamp")[["market_return", "sector_return"]].first().sort_index()
    factor["market_volatility_20"] = factor.market_return.shift(1).rolling(20, min_periods=20).std()
    factor["market_direction"] = np.sign(factor.market_return)
    factor["sector_direction"] = np.sign(factor.sector_return)
    event_dates = events.groupby("ticker").event_date.apply(set).to_dict()
    target_panels = {ticker: group.set_index("timestamp").sort_index()
                     for ticker, group in residuals.groupby("ticker")}
    horizons = sorted(responses.horizon.unique())
    forward_returns = {}
    for ticker, panel in target_panels.items():
        for horizon in horizons:
            future_sum = panel.residual_return.shift(-1).rolling(
                int(horizon), min_periods=int(horizon)).sum().shift(-(int(horizon)-1))
            forward_returns[(ticker, int(horizon))] = np.expm1(future_sum)
    event_lookup = events.set_index("event_id")
    rows = []
    valid_responses = responses.loc[responses.response_status == "ok"]
    for event_id, event_responses in valid_responses.groupby("event_id"):
        source_event = event_lookup.loc[event_id]
        event_date = source_event.event_date
        if event_date not in factor.index: continue
        state = factor.loc[event_date]
        candidates = factor.loc[(factor.index != event_date) &
            (~factor.index.isin(event_dates.get(source_event.ticker, set()))) &
            (factor.market_direction == state.market_direction) &
            (factor.sector_direction == state.sector_direction)].dropna()
        candidates = candidates.assign(distance=(candidates.market_volatility_20 -
                                                   state.market_volatility_20).abs())
        # Keep a wider candidate pool because some target securities did not
        # exist on early control dates. Availability may filter a date; the
        # future return's value is never used for matching or ranking.
        candidates = candidates.sort_values(["distance"], kind="stable").head(
            max(100, matches_per_event * 20))
        for response in event_responses.itertuples():
            target_returns = forward_returns[(response.target_ticker, int(response.horizon))]
            accepted = 0
            for _, (date, candidate) in enumerate(candidates.iterrows(), 1):
                value = target_returns.get(date, np.nan)
                if np.isnan(value): continue
                accepted += 1
                control_id = hashlib.sha256(
                    f"{response.event_id}|{response.target_ticker}|{response.horizon}|{date}".encode()
                ).hexdigest()[:24]
                rows.append({"control_id": control_id, "event_id": response.event_id,
                    "source_ticker": response.source_ticker, "target_ticker": response.target_ticker,
                    "event_type": source_event.event_type, "event_date": response.event_date,
                    "control_date": date, "horizon": response.horizon, "match_rank": accepted,
                    "market_direction": int(candidate.market_direction),
                    "sector_direction": int(candidate.sector_direction),
                    "event_market_volatility": state.market_volatility_20,
                    "control_market_volatility": candidate.market_volatility_20,
                    "volatility_distance": candidate.distance,
                    "control_target_residual_return": value,
                    "matching_features": "source_no_event,market_direction,sector_direction,market_volatility_20",
                    "target_future_used_for_matching": False})
                if accepted >= matches_per_event: break
    return pd.DataFrame(rows)


def _bootstrap_effect(event_values, control_values, iterations, rng, block_length=1):
    event_values = np.asarray(event_values, float); control_values = np.asarray(control_values, float)
    effects = []
    if block_length <= 1:
        for _ in range(iterations):
            e = rng.choice(event_values, len(event_values), replace=True)
            c = rng.choice(control_values, len(control_values), replace=True)
            effects.append(e.mean() - c.mean())
    else:
        def sample_blocks(values):
            starts = rng.integers(0, len(values), int(np.ceil(len(values) / block_length)))
            return np.concatenate([values[np.arange(s, s + block_length) % len(values)]
                                   for s in starts])[:len(values)]
        for _ in range(iterations):
            effects.append(sample_blocks(event_values).mean() - sample_blocks(control_values).mean())
    return np.percentile(effects, [2.5, 97.5])


def summarize_event_study(events: pd.DataFrame, responses: pd.DataFrame,
                          controls: pd.DataFrame, iterations: int = 500,
                          block_length: int = 5, seed: int = 0) -> pd.DataFrame:
    event_lookup = events[["event_id", "event_type"]]
    valid = responses.loc[responses.response_status == "ok"].merge(event_lookup, on="event_id")
    groups = ["source_ticker", "target_ticker", "event_type", "horizon"]
    output = []
    for key, group in valid.groupby(groups):
        control = controls.loc[controls.event_id.isin(group.event_id) &
            controls.target_ticker.eq(key[1]) & controls.horizon.eq(key[3])]
        event_values = group.target_residual_return.dropna().to_numpy()
        control_values = control.control_target_residual_return.dropna().to_numpy()
        if len(event_values) < 2 or len(control_values) < 2: continue
        rng = np.random.default_rng(seed + int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16))
        effect = event_values.mean() - control_values.mean()
        pooled = np.sqrt(((len(event_values)-1)*event_values.var(ddof=1) +
                          (len(control_values)-1)*control_values.var(ddof=1)) /
                         max(1, len(event_values)+len(control_values)-2))
        simple_ci = _bootstrap_effect(event_values, control_values, iterations, rng)
        block_ci = _bootstrap_effect(event_values, control_values, iterations, rng, block_length)
        combined = np.concatenate([event_values, control_values]); count = len(event_values)
        permuted = []
        for _ in range(iterations):
            shuffled = rng.permutation(combined)
            permuted.append(shuffled[:count].mean() - shuffled[count:].mean())
        p_value = (1 + sum(abs(x) >= abs(effect) for x in permuted)) / (iterations + 1)
        output.append(dict(zip(groups, key), event_count=len(event_values),
            mean_response=float(event_values.mean()), median_response=float(np.median(event_values)),
            standard_deviation=float(event_values.std(ddof=1)),
            standard_error=float(stats.sem(event_values)),
            positive_response_rate=float((event_values > 0).mean()),
            same_direction_rate=float(group.same_direction_as_source.mean()),
            confidence_interval_low=float(simple_ci[0]), confidence_interval_high=float(simple_ci[1]),
            block_confidence_interval_low=float(block_ci[0]),
            block_confidence_interval_high=float(block_ci[1]),
            baseline_mean_response=float(control_values.mean()), abnormal_effect=float(effect),
            effect_size=float(effect / pooled) if pooled > 0 else np.nan,
            permutation_p_value=float(p_value), control_observations=len(control_values)))
    return pd.DataFrame(output)
