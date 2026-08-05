"""Point and cumulative target responses beginning after a daily event."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _return_from_logs(values: pd.Series) -> float:
    return float(np.expm1(values.sum()))


def measure_daily_responses(events: pd.DataFrame, residuals: pd.DataFrame,
                            target_symbols: list[str], horizons: list[int],
                            include_self: bool = False) -> pd.DataFrame:
    records = []
    panels = {ticker: group.set_index("timestamp").sort_index()
              for ticker, group in residuals.groupby("ticker")}
    for _, event in events.iterrows():
        for target in target_symbols:
            if not include_self and target == event.ticker:
                continue
            target_data = panels.get(target)
            if target_data is None:
                for horizon in horizons:
                    records.append(_missing_record(event, target, horizon, "missing_target"))
                continue
            future = target_data.loc[target_data.index > pd.Timestamp(event.event_date)]
            for horizon in horizons:
                if len(future) < horizon:
                    records.append(_missing_record(event, target, horizon, "insufficient_future_data",
                                                   len(future)))
                    continue
                window = future.iloc[:horizon]
                raw_logs = window["raw_return"]
                market_logs = window["market_residual_return"]
                residual_logs = window["residual_return"]
                raw_path = np.expm1(raw_logs.cumsum())
                result = {"event_id": event.event_id, "source_ticker": event.ticker,
                    "target_ticker": target, "event_date": event.event_date,
                    "response_start_date": window.index[0], "response_end_date": window.index[-1],
                    "horizon": horizon, "source_event_direction": event.direction,
                    "source_raw_return": event.raw_return,
                    "source_residual_return": event.residual_return,
                    "point_raw_return": float(np.expm1(raw_logs.iloc[-1])),
                    "point_market_adjusted_return": float(np.expm1(market_logs.iloc[-1])),
                    "point_residual_return": float(np.expm1(residual_logs.iloc[-1])),
                    "target_raw_return": _return_from_logs(raw_logs),
                    "target_market_adjusted_return": _return_from_logs(market_logs),
                    "target_residual_return": _return_from_logs(residual_logs),
                    "target_log_return_sum": float(raw_logs.sum()),
                    "target_max_positive_excursion": float(raw_path.max()),
                    "target_max_negative_excursion": float(raw_path.min()),
                    "time_to_max_positive": int(np.argmax(raw_path.to_numpy()) + 1),
                    "time_to_max_negative": int(np.argmin(raw_path.to_numpy()) + 1),
                    "available_observations": len(window), "response_status": "ok"}
                result["response_direction"] = "positive" if result["target_raw_return"] >= 0 else "negative"
                result["same_direction_as_source"] = result["response_direction"] == event.direction
                daily_signs = np.sign(raw_logs.to_numpy())
                source_sign = 1 if event.direction == "positive" else -1
                result["reversal_within_horizon"] = bool((daily_signs * source_sign < 0).any())
                records.append(result)
    return pd.DataFrame(records).sort_values(
        ["event_date", "source_ticker", "target_ticker", "horizon"]).reset_index(drop=True)


def _missing_record(event, target: str, horizon: int, status: str,
                    available: int = 0) -> dict:
    return {"event_id": event.event_id, "source_ticker": event.ticker,
        "target_ticker": target, "event_date": event.event_date,
        "horizon": horizon, "source_event_direction": event.direction,
        "source_raw_return": event.raw_return, "source_residual_return": event.residual_return,
        "available_observations": available, "response_status": status}
