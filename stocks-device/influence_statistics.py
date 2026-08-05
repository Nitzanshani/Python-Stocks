"""Granger tests, FDR families and experimental relationship summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution


def benjamini_hochberg(p_values) -> np.ndarray:
    values = np.asarray(p_values, float); n = len(values)
    order = np.argsort(values); ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    result = np.empty(n); result[order] = np.clip(adjusted, 0, 1)
    return result


def apply_fdr_families(frame: pd.DataFrame, p_column: str, family_columns: list[str],
                       alpha: float, output_column: str = "adjusted_p_value") -> pd.DataFrame:
    result = frame.copy(); result[output_column] = np.nan; result["number_of_tests"] = 0
    result["fdr_alpha"] = alpha; result["fdr_significant"] = False
    for _, indices in result.groupby(family_columns, dropna=False).groups.items():
        index = list(indices); valid = result.loc[index, p_column].notna()
        valid_index = list(result.loc[index].index[valid])
        if not valid_index: continue
        result.loc[valid_index, output_column] = benjamini_hochberg(
            result.loc[valid_index, p_column].to_numpy())
        result.loc[index, "number_of_tests"] = len(valid_index)
        result.loc[valid_index, "fdr_significant"] = (
            result.loc[valid_index, output_column] <= alpha)
    result["fdr_family"] = result[family_columns].astype(str).agg("|".join, axis=1)
    return result


def _granger_one_direction(source: pd.Series, target: pd.Series, lag: int):
    data = pd.concat([source.rename("source"), target.rename("target")], axis=1).dropna()
    columns = {"target": data.target}
    for shift in range(1, lag + 1):
        columns[f"target_lag_{shift}"] = data.target.shift(shift)
        columns[f"source_lag_{shift}"] = data.source.shift(shift)
    design = pd.DataFrame(columns).dropna(); y = design.pop("target").to_numpy(float)
    restricted_names = [c for c in design if c.startswith("target_lag")]
    unrestricted_names = restricted_names + [c for c in design if c.startswith("source_lag")]
    if len(y) <= len(unrestricted_names) + 5: return np.nan, np.nan, len(y)
    def rss(names):
        values = design[names].to_numpy(float)
        scale = values.std(axis=0); scale[scale == 0] = 1.0
        values = np.clip((values - values.mean(axis=0)) / scale, -20, 20)
        centered_y = y - y.mean()
        x = np.column_stack([np.ones(len(y)), values])
        beta, *_ = np.linalg.lstsq(x, centered_y, rcond=None)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fitted = x @ beta
        if not np.isfinite(fitted).all():
            return np.nan
        return float(np.sum((centered_y - fitted) ** 2))
    restricted, unrestricted = rss(restricted_names), rss(unrestricted_names)
    df_denominator = len(y) - len(unrestricted_names) - 1
    if not np.isfinite(restricted) or not np.isfinite(unrestricted) or unrestricted <= np.finfo(float).eps:
        return np.nan, np.nan, len(y)
    statistic = max(0.0, ((restricted-unrestricted)/lag) / (unrestricted/df_denominator))
    return statistic, float(f_distribution.sf(statistic, lag, df_denominator)), len(y)


def run_granger_all_directions(residuals: pd.DataFrame, symbols: list[str],
                               lags: list[int], alpha: float) -> pd.DataFrame:
    panel = residuals.pivot(index="timestamp", columns="ticker", values="residual_return")
    rows = []
    for source in symbols:
        for target in symbols:
            if source == target: continue
            for lag in lags:
                statistic, p_value, observations = _granger_one_direction(
                    panel[source], panel[target], int(lag))
                rows.append({"source_ticker": source, "target_ticker": target,
                    "window_start": panel.index.min(), "window_end": panel.index.max(),
                    "lag": int(lag), "test_statistic": statistic, "raw_p_value": p_value,
                    "observations": observations, "analysis_type": "granger_residual",
                    "window_family": "full_phase3a_history"})
    raw = pd.DataFrame(rows)
    # A family contains every directed pair for one pre-declared lag and window.
    return apply_fdr_families(raw, "raw_p_value",
        ["analysis_type", "window_family", "lag"], alpha)


def build_relationships(event_summary: pd.DataFrame, folds: pd.DataFrame,
                        granger: pd.DataFrame, config: dict) -> pd.DataFrame:
    prediction = folds.groupby(["source_ticker", "target_ticker"]).agg(
        baseline_rmse=("baseline_rmse", "mean"), extended_rmse=("extended_rmse", "mean"),
        rmse_improvement=("rmse_improvement", "mean"),
        direction_improvement=("direction_improvement", "mean"),
        folds_improved=("rmse_improvement", lambda x: int((x > 0).sum())),
        total_folds=("fold", "count"), improvement_std=("rmse_improvement", "std")
    ).reset_index()
    prediction["improvement_consistency"] = prediction.folds_improved / prediction.total_folds
    event_best = event_summary.assign(abs_effect=event_summary.effect_size.abs()).sort_values(
        "abs_effect", ascending=False).drop_duplicates(["source_ticker", "target_ticker"])
    granger_best = granger.sort_values("adjusted_p_value").drop_duplicates(
        ["source_ticker", "target_ticker"])
    result = prediction.merge(event_best[["source_ticker", "target_ticker", "event_count",
        "event_type", "horizon", "abnormal_effect", "effect_size", "adjusted_p_value"]],
        on=["source_ticker", "target_ticker"], how="left", suffixes=("", "_event"))
    result = result.merge(granger_best[["source_ticker", "target_ticker", "lag",
        "adjusted_p_value", "fdr_significant"]], on=["source_ticker", "target_ticker"],
        how="left", suffixes=("_event", "_granger"))
    result["q_prediction"] = (result.rmse_improvement /
        result.baseline_rmse.replace(0, np.nan)).clip(lower=0, upper=.20).fillna(0) / .20
    result["q_event"] = result.effect_size.abs().clip(upper=1).fillna(0)
    result["q_consistency"] = result.improvement_consistency.clip(0, 1)
    significant_lags = granger.groupby(["source_ticker", "target_ticker"]).fdr_significant.mean()
    result["q_lag"] = [significant_lags.get((s, t), 0) for s, t in
                       zip(result.source_ticker, result.target_ticker)]
    result["q_statistics"] = np.where(result.fdr_significant.fillna(False),
        1-result.adjusted_p_value_granger.fillna(1), 0)
    result["q_sample"] = np.sqrt((result.event_count.fillna(0)/20).clip(upper=1) *
                                  (result.total_folds/10).clip(upper=1))
    components = ["q_prediction", "q_event", "q_consistency", "q_lag", "q_statistics", "q_sample"]
    schemes = {"balanced": [.35,.25,.15,.10,.10,.05],
               "prediction_heavy": [.50,.15,.15,.05,.10,.05],
               "statistics_heavy": [.25,.20,.10,.10,.30,.05]}
    for name, weights in schemes.items():
        result[f"experimental_score_{name}"] = sum(result[c]*w for c,w in zip(components,weights))
    minimum_events = int(config["minimum_event_count"]); minimum_folds = int(config["minimum_folds"])
    def status(row):
        if row.total_folds < minimum_folds or pd.isna(row.event_count) or row.event_count < minimum_events:
            return "insufficient_data"
        relative = row.rmse_improvement / row.baseline_rmse if row.baseline_rmse else 0
        if abs(row.effect_size) >= .3 and relative <= 0: return "rejected"
        if relative > 0 and row.improvement_consistency >= .6 and abs(row.effect_size) >= .2:
            return "validated"
        has_candidate_signal = (abs(row.effect_size) >= .2 or bool(row.fdr_significant) or relative > 0)
        if (has_candidate_signal and row.improvement_consistency < .4 and
                row.improvement_std > abs(row.rmse_improvement)):
            return "unstable"
        if abs(row.effect_size) >= .2 or bool(row.fdr_significant): return "candidate"
        return "no_evidence"
    result["relationship_status"] = result.apply(status, axis=1)
    result["score_status"] = "experimental_not_production"
    return result.sort_values("experimental_score_balanced", ascending=False)
