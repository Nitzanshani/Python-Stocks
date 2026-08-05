"""Immutable daily raw events, clusters and representative events."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")


def _event_id(ticker: str, timestamp, event_type: str, direction: str,
              threshold_version: str) -> str:
    key = f"{ticker}|{pd.Timestamp(timestamp).date()}|{event_type}|{direction}|{threshold_version}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _available_at(timestamp) -> str:
    day = pd.Timestamp(timestamp).tz_convert(NY).date()
    return pd.Timestamp(datetime.combine(day, datetime.min.time()).replace(
        hour=16, tzinfo=NY)).tz_convert("UTC").isoformat()


def detect_daily_events(features: pd.DataFrame, residuals: pd.DataFrame,
                        config: dict) -> pd.DataFrame:
    settings = config["events"]; version = settings["threshold_version"]
    merged = residuals.merge(features, on=["ticker", "timestamp"], how="left",
                             suffixes=("", "_feature"))
    merged = merged.sort_values(["ticker", "timestamp"])
    historical = merged.groupby("ticker")["residual_return"].transform(
        lambda s: s.shift(1).rolling(int(config["feature_window"]),
                                     min_periods=int(config["feature_window"])).std(ddof=1))
    merged["residual_z_score"] = merged["residual_return"] / historical.replace(0, np.nan)
    definitions = [
        ("raw_return", merged["raw_return"].abs() >= float(settings["raw_return_threshold"]),
         "raw_return", float(settings["raw_return_threshold"])),
        ("residual_return", merged["residual_z_score"].abs() >= float(settings["residual_z_threshold"]),
         "residual_z_score", float(settings["residual_z_threshold"])),
        ("relative_volume", merged["relative_volume"] >= float(settings["relative_volume_threshold"]),
         "relative_volume", float(settings["relative_volume_threshold"])),
    ]
    return_shock = definitions[0][1] | definitions[1][1]
    definitions.append(("combined_shock", return_shock & definitions[1][1] & definitions[2][1],
                        "combined", 1.0))
    records = []
    created = datetime.now(timezone.utc).isoformat()
    for event_type, mask, threshold_name, threshold in definitions:
        for _, row in merged.loc[mask.fillna(False)].iterrows():
            direction_value = row["residual_return"] if event_type != "raw_return" else row["raw_return"]
            direction = "positive" if direction_value >= 0 else "negative"
            records.append({
                "event_id": _event_id(row.ticker, row.timestamp, event_type, direction, version),
                "ticker": row.ticker, "event_date": row.timestamp, "event_type": event_type,
                "direction": direction, "raw_return": row.raw_return,
                "log_return": row.log_return, "residual_return": row.residual_return,
                "return_z_score": row.residual_z_score, "volume": row.volume,
                "relative_volume": row.relative_volume, "market_return": row.market_return,
                "sector_return": row.sector_return,
                "rolling_volatility": row.rolling_volatility,
                "threshold_name": threshold_name, "threshold_value": threshold,
                "residual_model": row.residual_model,
                "residual_window": row.estimation_window,
                "data_available_at": _available_at(row.timestamp), "created_at": created,
                "engine_version": config["engine_version"]})
    columns = ["event_id", "ticker", "event_date", "event_type", "direction", "raw_return",
        "log_return", "residual_return", "return_z_score", "volume", "relative_volume",
        "market_return", "sector_return", "rolling_volatility", "threshold_name",
        "threshold_value", "residual_model", "residual_window", "data_available_at",
        "created_at", "engine_version"]
    return pd.DataFrame(records, columns=columns).sort_values(
        ["ticker", "event_date", "event_type"]).reset_index(drop=True)


def cluster_daily_events(events: pd.DataFrame, trading_dates: list,
                         cooldown: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cluster records and one strongest representative raw event per cluster."""
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    positions = {pd.Timestamp(date): i for i, date in enumerate(sorted(pd.to_datetime(trading_dates, utc=True)))}
    event_rows = events.sort_values(["ticker", "event_type", "direction", "event_date"])
    clusters = []
    for key, group in event_rows.groupby(["ticker", "event_type", "direction"], sort=True):
        current = []
        for _, event in group.iterrows():
            if current and positions.get(pd.Timestamp(event.event_date), 10**9) - positions.get(
                    pd.Timestamp(current[-1].event_date), -10**9) > cooldown:
                clusters.append(current); current = []
            current.append(event)
        if current: clusters.append(current)
    cluster_records, representatives = [], []
    for group in clusters:
        frame = pd.DataFrame(group)
        magnitude = frame["return_z_score"].abs().fillna(
            frame["residual_return"].abs().fillna(frame["raw_return"].abs()))
        representative = frame.loc[magnitude.idxmax()].copy()
        cluster_id = hashlib.sha256("|".join(sorted(frame.event_id)).encode()).hexdigest()[:24]
        cluster_records.append({"cluster_id": cluster_id, "ticker": representative.ticker,
            "event_type": representative.event_type, "direction": representative.direction,
            "cluster_start": frame.event_date.min(), "cluster_end": frame.event_date.max(),
            "peak_event_date": representative.event_date,
            "peak_magnitude": float(magnitude.loc[representative.name]),
            "number_of_event_days": int(frame.event_date.nunique())})
        representative["cluster_id"] = cluster_id
        representative["representative_reason"] = "largest absolute standardized magnitude"
        representatives.append(representative)
    return (pd.DataFrame(cluster_records).sort_values(["ticker", "cluster_start"]),
            pd.DataFrame(representatives).sort_values(["ticker", "event_date"]).reset_index(drop=True))
