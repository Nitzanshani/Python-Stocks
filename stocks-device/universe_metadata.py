"""Build a dated local universe/company/sector/industry snapshot."""

from __future__ import annotations

from datetime import datetime, timezone

from market_data_store import MarketDataStore, load_config
from market_scanner import (NASDAQ100_URL, SP500_URL, WATCHLIST, _download_table,
                            load_universe_details, normalize_symbol)


def _column(table, candidates):
    return next((name for name in candidates if name in table.columns), None)


def build_universe_snapshot(store: MarketDataStore):
    import pandas as pd

    symbols, memberships, names = load_universe_details()
    profiles: dict[str, dict[str, str]] = {}
    warnings = []
    existing_path = store.root / "metadata" / "universe.parquet"
    if existing_path.exists():
        try:
            existing = pd.read_parquet(existing_path)
            for _, row in existing.iterrows():
                profiles[str(row["ticker"])] = {
                    "sector": row.get("sector") if pd.notna(row.get("sector")) else None,
                    "industry": row.get("industry") if pd.notna(row.get("industry")) else None,
                }
        except Exception as exc:
            warnings.append(f"Existing universe snapshot unreadable: {exc}")
    try:
        sources = [
            (_download_table(pd, SP500_URL, {"Symbol", "Security"}), ("Symbol", "Ticker")),
            (_download_table(pd, NASDAQ100_URL, {"Ticker", "Company"}), ("Ticker", "Symbol")),
        ]
        for table, symbol_candidates in sources:
            symbol_column = _column(table, symbol_candidates)
            sector_column = _column(table, ("GICS Sector", "Sector"))
            industry_column = _column(table, ("GICS Sub-Industry", "Sub-Industry", "Industry"))
            if not symbol_column:
                continue
            for _, row in table.iterrows():
                symbol = normalize_symbol(row[symbol_column])
                profile = profiles.setdefault(symbol, {})
                if sector_column and pd.notna(row[sector_column]):
                    profile["sector"] = str(row[sector_column]).strip()
                if industry_column and pd.notna(row[industry_column]):
                    profile["industry"] = str(row[industry_column]).strip()
    except Exception as exc:
        warnings.append(f"Wikipedia profile refresh failed: {type(exc).__name__}: {exc}")
    manual = load_config().get("manual_profiles", {})
    missing_profiles = [symbol for symbol in symbols
        if not profiles.get(symbol, {}).get("sector") or not profiles.get(symbol, {}).get("industry")]
    if missing_profiles:
        try:
            import yfinance as yf
            for symbol in missing_profiles:
                try:
                    info = yf.Ticker(symbol).get_info()
                    profile = profiles.setdefault(symbol, {})
                    profile["sector"] = profile.get("sector") or info.get("sector")
                    profile["industry"] = profile.get("industry") or info.get("industry")
                except Exception as exc:
                    warnings.append(f"{symbol} profile unavailable: {type(exc).__name__}")
        except ImportError:
            warnings.append("yfinance unavailable for missing sector enrichment")
    rows = []
    for symbol in symbols:
        groups = set(filter(None, memberships.get(symbol, "").split("+")))
        profile = {**profiles.get(symbol, {}), **manual.get(symbol, {})}
        rows.append({
            "ticker": symbol, "company_name": names.get(symbol, symbol),
            "sector": profile.get("sector"), "industry": profile.get("industry"),
            "in_sp500": "S&P 500" in groups, "in_nasdaq100": "QQQ" in groups,
            "in_manual_watchlist": symbol in WATCHLIST,
            "snapshot_at": datetime.now(timezone.utc).isoformat(),
        })
    frame = pd.DataFrame(rows).sort_values("ticker")
    directory = store.root / "metadata"; directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "universe.parquet", index=False, compression="zstd")
    frame.to_csv(directory / "universe.csv", index=False)
    return frame, warnings
