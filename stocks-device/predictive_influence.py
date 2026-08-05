"""Purged walk-forward baseline-versus-source predictive comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def purged_walk_forward_splits(n: int, train_days: int, test_days: int, step_days: int,
                               purge_days: int, embargo_days: int):
    """Yield chronological indices; training never reaches the purged test boundary."""
    previous_embargoes = []
    test_start = train_days + purge_days
    fold = 0
    while test_start + test_days <= n:
        train_end = test_start - purge_days
        train_start = max(0, train_end - train_days)
        train = np.arange(train_start, train_end)
        for start, end in previous_embargoes:
            train = train[(train < start) | (train > end)]
        test = np.arange(test_start, test_start + test_days)
        yield fold, train, test
        previous_embargoes.append((test[-1] + 1, test[-1] + embargo_days))
        test_start += step_days; fold += 1


def build_pair_dataset(residuals: pd.DataFrame, events: pd.DataFrame,
                       source: str, target: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    source_frame = residuals.loc[residuals.ticker == source].set_index("timestamp").sort_index()
    target_frame = residuals.loc[residuals.ticker == target].set_index("timestamp").sort_index()
    common = target_frame.index.intersection(source_frame.index)
    s, t = source_frame.loc[common], target_frame.loc[common]
    data = pd.DataFrame(index=common)
    baseline = []
    for lag in range(1, 6):
        data[f"target_residual_lag_{lag}"] = t.residual_return.shift(lag)
        data[f"market_lag_{lag}"] = t.market_return.shift(lag)
        data[f"sector_lag_{lag}"] = t.sector_return.shift(lag)
        baseline += [f"target_residual_lag_{lag}", f"market_lag_{lag}", f"sector_lag_{lag}"]
    data["target_rolling_volatility"] = t.residual_return.shift(1).rolling(20).std()
    data["target_relative_volume"] = t.relative_volume
    data["day_of_week"] = data.index.tz_convert("America/New_York").dayofweek
    baseline += ["target_rolling_volatility", "target_relative_volume", "day_of_week"]
    extended = list(baseline)
    for lag in range(1, 6):
        data[f"source_residual_lag_{lag}"] = s.residual_return.shift(lag)
        data[f"source_raw_lag_{lag}"] = s.raw_return.shift(lag)
        extended += [f"source_residual_lag_{lag}", f"source_raw_lag_{lag}"]
    source_events = events.loc[events.ticker == source].set_index("event_date")
    flag = pd.Series(0.0, index=common)
    magnitude = pd.Series(0.0, index=common)
    if not source_events.empty:
        dates = common.intersection(source_events.index)
        flag.loc[dates] = 1.0
        magnitude.loc[dates] = source_events.groupby(level=0).return_z_score.apply(
            lambda x: x.abs().max()).reindex(dates).fillna(0)
    data["source_event_flag"] = flag
    data["source_event_magnitude"] = magnitude
    event_positions = np.where(flag.to_numpy() > 0, np.arange(len(flag)), np.nan)
    data["days_since_source_event"] = pd.Series(event_positions, index=common).ffill()
    data["days_since_source_event"] = np.arange(len(data)) - data.days_since_source_event
    data["days_since_source_event"] = data.days_since_source_event.fillna(len(data)).clip(upper=252)
    data["source_relative_volume"] = s.relative_volume
    extended += ["source_event_flag", "source_event_magnitude",
                 "days_since_source_event", "source_relative_volume"]
    data["target_next_residual_return"] = t.residual_return.shift(-1)
    return data.replace([np.inf, -np.inf], np.nan).dropna(), baseline, extended


def _scale_fit(x):
    mean = x.mean(axis=0); scale = x.std(axis=0, ddof=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _ridge_fit(x, y, alpha):
    design = np.column_stack([np.ones(len(x)), np.clip(x, -20, 20)])
    penalty = np.sqrt(alpha) * np.eye(design.shape[1]); penalty[0, 0] = 0
    augmented_x = np.vstack([design, penalty])
    augmented_y = np.concatenate([y, np.zeros(design.shape[1])])
    coefficients, *_ = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)
    return coefficients


def _ridge_predict(x, coefficients):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prediction = np.column_stack([np.ones(len(x)), np.clip(x, -20, 20)]) @ coefficients
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite Ridge prediction after fold-local scaling")
    return prediction


def _select_alpha(x, y, alphas, purge_days):
    validation = max(20, int(len(x) * .2)); split = len(x) - validation
    fit_end = max(1, split - purge_days)
    if fit_end < 20: return float(alphas[0])
    mean, scale = _scale_fit(x[:fit_end])
    train_x = (x[:fit_end] - mean) / scale; validation_x = (x[split:] - mean) / scale
    scores = []
    for alpha in alphas:
        coefficients = _ridge_fit(train_x, y[:fit_end], float(alpha))
        scores.append(np.mean((_ridge_predict(validation_x, coefficients) - y[split:]) ** 2))
    return float(alphas[int(np.argmin(scores))])


def run_pair_walk_forward(residuals: pd.DataFrame, events: pd.DataFrame,
                          source: str, target: str, settings: dict) -> pd.DataFrame:
    data, baseline_columns, extended_columns = build_pair_dataset(residuals, events, source, target)
    y = data.target_next_residual_return.to_numpy(float)
    rows = []
    splits = purged_walk_forward_splits(len(data), int(settings["train_days"]),
        int(settings["test_days"]), int(settings["step_days"]),
        int(settings["purge_days"]), int(settings["embargo_days"]))
    for fold, train, test in splits:
        predictions = {}
        selected = {}
        for name, columns in (("baseline", baseline_columns), ("extended", extended_columns)):
            x = data[columns].to_numpy(float)
            alpha = _select_alpha(x[train], y[train], settings["ridge_alphas"],
                                  int(settings["purge_days"]))
            mean, scale = _scale_fit(x[train])
            coefficients = _ridge_fit((x[train] - mean) / scale, y[train], alpha)
            predictions[name] = _ridge_predict((x[test] - mean) / scale, coefficients)
            selected[name] = alpha
        actual = y[test]; naive = np.zeros(len(test))
        def metrics(prediction):
            return (float(np.mean(np.abs(prediction-actual))),
                    float(np.sqrt(np.mean((prediction-actual)**2))),
                    float((np.sign(prediction) == np.sign(actual)).mean()))
        bm, br, bd = metrics(predictions["baseline"]); em, er, ed = metrics(predictions["extended"])
        nm, nr, nd = metrics(naive)
        rows.append({"source_ticker": source, "target_ticker": target, "fold": fold,
            "train_start": data.index[train[0]], "train_end": data.index[train[-1]],
            "test_start": data.index[test[0]], "test_end": data.index[test[-1]],
            "train_observations": len(train), "test_observations": len(test),
            "purge_days": settings["purge_days"], "embargo_days": settings["embargo_days"],
            "baseline_alpha": selected["baseline"], "extended_alpha": selected["extended"],
            "preprocessing_scope": "fit inside fold only", "baseline_mae": bm,
            "extended_mae": em, "mae_improvement": bm-em, "baseline_rmse": br,
            "extended_rmse": er, "rmse_improvement": br-er,
            "baseline_direction_accuracy": bd, "extended_direction_accuracy": ed,
            "direction_improvement": ed-bd, "naive_mae": nm, "naive_rmse": nr,
            "naive_direction_accuracy": nd})
    return pd.DataFrame(rows)


def run_all_pairs_walk_forward(residuals: pd.DataFrame, events: pd.DataFrame,
                               research_symbols: list[str], settings: dict) -> pd.DataFrame:
    pieces = []
    for source in research_symbols:
        for target in research_symbols:
            if source == target: continue
            print(f"Walk-forward {source} -> {target}", flush=True)
            result = run_pair_walk_forward(residuals, events, source, target, settings)
            if not result.empty: pieces.append(result)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
