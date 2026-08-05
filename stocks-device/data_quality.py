"""Daily Parquet quality audit and CSV/HTML/Parquet report generation."""

from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from market_data_store import MarketDataStore


def quality_score(coverage: float, duplicates: int, invalid_ohlc: int,
                  completeness: float, stale_days: int, update_ok: bool,
                  rows: int) -> float:
    """Transparent source-quality score; never used as a trading feature."""
    denominator = max(rows, 1)
    duplicate_q = max(0.0, 1 - duplicates / denominator)
    ohlc_q = max(0.0, 1 - invalid_ohlc / denominator)
    freshness_q = 1.0 if stale_days <= 2 else math.exp(-(stale_days - 2) / 5)
    score = (.45 * max(0.0, min(1.0, coverage)) + .15 * duplicate_q +
             .15 * ohlc_q + .10 * max(0.0, min(1.0, completeness)) +
             .10 * freshness_q + .05 * float(update_ok))
    return round(score, 4)


def _metadata(store: MarketDataStore, ticker: str) -> dict[str, object]:
    try:
        return json.loads(store.metadata_path(ticker, "1d").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "missing", "error_type": "missing_metadata",
                "error": "No daily update metadata"}


def _latest_completed_sessions(calendar, now: datetime):
    schedule = calendar.schedule(start_date="1960-01-01", end_date=now.date())
    closes = schedule["market_close"]
    return schedule.loc[closes <= now.astimezone(timezone.utc)]


def inspect_ticker(store: MarketDataStore, ticker: str, schedule,
                   spy_dates: set | None = None) -> dict[str, object]:
    import pandas as pd

    metadata = _metadata(store, ticker)
    frame = store.read(ticker, "1d")
    status = str(metadata.get("status", "missing"))
    if frame.empty:
        return {"ticker": ticker, "status": status, "first_timestamp": None,
            "last_timestamp": None, "row_count": 0, "expected_trading_days": 0,
            "missing_trading_days": 0, "duplicate_count": 0,
            "overlap_rows_replaced": int(metadata.get("duplicates_removed", 0) or 0),
            "invalid_ohlc_rows": int(metadata.get(
                "invalid_rows_removed_total", metadata.get("invalid_rows_removed", 0)) or 0),
            "negative_volume_rows": 0, "null_open": 0, "null_high": 0,
            "null_low": 0, "null_close": 0, "null_adjusted_close": 0,
            "null_volume": 0, "split_count": 0, "dividend_count": 0,
            "largest_abs_daily_return": None, "largest_volume": None,
            "stale_days": len(schedule), "coverage_ratio": 0.0, "quality_score": 0.0,
            "adjustment_warning_count": 0, "spy_overlap_ratio": 0.0,
            "error_type": metadata.get("error_type"), "error_message": metadata.get("error"),
            "last_updated": metadata.get("updated_at"), "manual_review": True}
    local_dates = set(frame.index.tz_convert("America/New_York").date)
    first, last = min(local_dates), max(local_dates)
    expected_index = schedule.loc[str(first):str(last)].index
    expected_dates = set(expected_index.date)
    missing = expected_dates - local_dates
    later_dates = set(schedule.loc[str(last):].index.date) - {last}
    duplicate_count = int(frame.index.duplicated().sum())
    overlap_replaced = int(metadata.get("duplicates_removed", 0) or 0)
    required = ["Open", "High", "Low", "Close"]
    nulls = {column: int(frame[column].isna().sum()) for column in
             [*required, "Adj Close", "Volume"]}
    valid_values = frame[required].notna().all(axis=1)
    invalid = (~valid_values |
        (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)) |
        (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)))
    invalid_count = int(invalid.sum()) + int(metadata.get(
        "invalid_rows_removed_total", metadata.get("invalid_rows_removed", 0)) or 0)
    negative_volume = int((frame["Volume"] < 0).sum())
    returns = frame["Adj Close"].pct_change(fill_method=None).replace([math.inf, -math.inf], float("nan"))
    actions = (frame["Stock Splits"].fillna(0) != 0) | (frame["Dividends"].fillna(0) != 0)
    factor = frame["Close"] / frame["Adj Close"]
    factor_jump = factor.pct_change(fill_method=None).abs() > .05
    split_warning = (frame["Stock Splits"].fillna(0) != 0) & (returns.abs() > .20)
    adjustment_warnings = int((factor_jump & ~actions).sum() + split_warning.sum())
    total_fields = max(1, len(frame) * 6)
    completeness = 1 - sum(nulls.values()) / total_fields
    coverage = len(local_dates & expected_dates) / max(1, len(expected_dates))
    update_ok = status in {"updated", "current"}
    score = quality_score(coverage, duplicate_count, invalid_count + negative_volume,
                          completeness, len(later_dates), update_ok, len(frame))
    overlap = len(local_dates & spy_dates) / max(1, len(local_dates)) if spy_dates else None
    manual = bool(status == "failed" or score < .90 or len(missing) > 3 or
                  len(later_dates) > 3 or adjustment_warnings or invalid_count or
                  negative_volume)
    return {"ticker": ticker, "status": status,
        "first_timestamp": frame.index[0].isoformat(), "last_timestamp": frame.index[-1].isoformat(),
        "row_count": len(frame), "expected_trading_days": len(expected_dates),
        "missing_trading_days": len(missing), "duplicate_count": duplicate_count,
        "overlap_rows_replaced": overlap_replaced,
        "invalid_ohlc_rows": invalid_count, "negative_volume_rows": negative_volume,
        "null_open": nulls["Open"], "null_high": nulls["High"], "null_low": nulls["Low"],
        "null_close": nulls["Close"], "null_adjusted_close": nulls["Adj Close"],
        "null_volume": nulls["Volume"],
        "split_count": int((frame["Stock Splits"].fillna(0) != 0).sum()),
        "dividend_count": int((frame["Dividends"].fillna(0) != 0).sum()),
        "largest_abs_daily_return": round(float(returns.abs().max() * 100), 4),
        "largest_volume": int(frame["Volume"].max()) if frame["Volume"].notna().any() else None,
        "stale_days": len(later_dates), "coverage_ratio": round(coverage, 6),
        "quality_score": score, "adjustment_warning_count": adjustment_warnings,
        "spy_overlap_ratio": round(overlap, 6) if overlap is not None else None,
        "error_type": metadata.get("error_type"), "error_message": metadata.get("error"),
        "last_updated": metadata.get("updated_at"), "manual_review": manual}


def build_daily_quality_report(store: MarketDataStore, tickers: list[str],
                               reports_dir: Path | None = None):
    import pandas as pd
    import pandas_market_calendars as mcal

    now = datetime.now(timezone.utc)
    schedule = _latest_completed_sessions(mcal.get_calendar("NYSE"), now)
    spy = store.read("SPY", "1d")
    spy_dates = set(spy.index.tz_convert("America/New_York").date) if not spy.empty else None
    rows = [inspect_ticker(store, ticker, schedule, spy_dates) for ticker in sorted(set(tickers))]
    report = pd.DataFrame(rows).sort_values(["quality_score", "ticker"])
    directory = reports_dir or (store.root.parent / "reports")
    directory.mkdir(parents=True, exist_ok=True)
    report.to_parquet(directory / "data_quality_daily.parquet", index=False, compression="zstd")
    report.to_csv(directory / "data_quality_daily.csv", index=False)
    storage_bytes = sum(path.stat().st_size for path in store.root.rglob("*.parquet"))
    successful = int(report["status"].isin(["updated", "current"]).sum())
    failed = int((report["status"] == "failed").sum())
    stale = int((report["stale_days"] > 2).sum())
    missing = int((report["missing_trading_days"] > 0).sum())
    invalid = int(((report["invalid_ohlc_rows"] + report["negative_volume_rows"]) > 0).sum())
    years = ((pd.to_datetime(report["last_timestamp"], utc=True) -
              pd.to_datetime(report["first_timestamp"], utc=True)).dt.days / 365.25).dropna()
    buckets = pd.cut(years, [-1, 1, 3, 5, 10, float("inf")],
                     labels=["<1y", "1–3y", "3–5y", "5–10y", "10y+"]).value_counts().sort_index()
    versions = {name: version(name) for name in ("yfinance", "pandas", "pyarrow")}
    summary = {"generated_at": now.isoformat(), "tickers": len(report),
        "successful": successful, "failed": failed, "stale": stale,
        "with_missing_sessions": missing, "with_invalid_rows": invalid,
        "manual_review": int(report["manual_review"].sum()), "storage_bytes": storage_bytes,
        "python": platform.python_version(), **versions}
    cards = "".join(f"<div><b>{value}</b><span>{key.replace('_',' ')}</span></div>"
                    for key, value in summary.items())
    worst = report.head(20).to_html(index=False, escape=True)
    review = report[report["manual_review"]].to_html(index=False, escape=True)
    html = f'''<!doctype html><meta charset="utf-8"><meta name="robots" content="noindex">
<title>Daily Market Data Quality</title><style>body{{font:14px system-ui;background:#0b1117;color:#e9eef4;padding:28px}}h1,h2{{color:#fff}}.cards{{display:flex;flex-wrap:wrap;gap:10px}}.cards div{{background:#172433;border:1px solid #31506c;border-radius:10px;padding:12px;min-width:130px}}.cards b,.cards span{{display:block}}.cards b{{font-size:20px;color:#35d07f}}.cards span{{color:#91a1b3}}table{{border-collapse:collapse;width:100%;background:#121a23;font-size:11px}}th,td{{padding:7px;border:1px solid #29394a;text-align:left}}th{{position:sticky;top:0;background:#192a3b}}.box{{overflow:auto;max-height:520px;margin-bottom:28px}}</style>
<h1>Daily Market Data Quality</h1><p>Missing sessions may reflect IPOs, halts, delistings or source gaps; they are not automatically data errors. Calendar: NYSE via pandas-market-calendars.</p><div class="cards">{cards}</div><h2>History distribution</h2>{buckets.rename("tickers").to_frame().to_html()}<h2>20 lowest quality</h2><div class="box">{worst}</div><h2>Manual review</h2><div class="box">{review}</div>'''
    (directory / "data_quality_daily.html").write_text(html, encoding="utf-8")
    (directory / "data_quality_daily_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Quality report: {directory / 'data_quality_daily.html'}")
    return report, summary
