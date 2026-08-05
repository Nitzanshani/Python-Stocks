from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pandas as pd

from data_quality import build_daily_quality_report, quality_score
from market_data_store import MarketDataStore
from update_market_data import select_failed_symbols


CONFIG = {
    "data_root": "data", "daily_initial_period": "max", "overlap_bars": 3,
    "request_pause_seconds": 0,
    "intervals": {
        "1d": {"directory": "market/daily", "lookback_days": None},
        "60m": {"directory": "market/hourly", "lookback_days": 60},
        "30m": {"directory": "market/30m", "lookback_days": 60},
        "15m": {"directory": "market/15m", "lookback_days": 60},
        "5m": {"directory": "market/5m", "lookback_days": 60},
    },
}


def bars(dates, closes, split=None):
    split = split or [0] * len(dates)
    return pd.DataFrame({
        "Open": closes, "High": [value + 1 for value in closes],
        "Low": [value - 1 for value in closes], "Close": closes,
        "Adj Close": closes, "Volume": [1000] * len(dates),
        "Dividends": [0] * len(dates), "Stock Splits": split,
    }, index=pd.DatetimeIndex(dates))


class MarketDataStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = MarketDataStore(Path(self.temp.name), CONFIG)

    def tearDown(self):
        self.temp.cleanup()

    def test_normalize_converts_new_york_to_utc_and_keeps_splits(self):
        frame = bars(["2026-08-03 09:30"], [100], [2])
        normalized = self.store.normalize(frame, "60m")
        self.assertEqual(normalized.index[0].isoformat(), "2026-08-03T13:30:00+00:00")
        self.assertEqual(normalized.iloc[0]["Stock Splits"], 2)

    def test_missing_trading_day_is_preserved_as_a_gap(self):
        frame = bars(["2026-08-03", "2026-08-05"], [100, 102])
        normalized, removed = self.store.validate(self.store.normalize(frame, "1d"))
        self.assertEqual(removed, 0)
        self.assertEqual(len(normalized), 2)
        self.assertEqual([stamp.astimezone(timezone.utc).day for stamp in normalized.index], [3, 5])

    def test_merge_sorts_and_new_download_replaces_duplicate(self):
        old = self.store.normalize(bars(["2026-08-03", "2026-08-04"], [100, 101]), "1d")
        new = self.store.normalize(bars(["2026-08-04", "2026-08-05"], [102, 103]), "1d")
        merged, duplicates, added = self.store.merge(old, new)
        self.assertEqual(duplicates, 1)
        self.assertEqual(added, 1)
        self.assertEqual(merged["Close"].tolist(), [100, 102, 103])
        self.assertTrue(merged.index.is_monotonic_increasing)

    def test_validation_removes_invalid_ohlc_and_negative_volume(self):
        frame = bars(["2026-08-03", "2026-08-04", "2026-08-05"], [100, 101, 102])
        frame.iloc[1, frame.columns.get_loc("High")] = 90
        frame.iloc[2, frame.columns.get_loc("Volume")] = -1
        valid, removed = self.store.validate(self.store.normalize(frame, "1d"))
        self.assertEqual(removed, 2)
        self.assertEqual(len(valid), 1)

    def test_invalid_row_audit_is_cumulative_across_clean_refresh(self):
        dirty = bars(["2026-08-03", "2026-08-04"], [100, 101])
        dirty.iloc[1, dirty.columns.get_loc("High")] = 50
        clean = bars(["2026-08-03", "2026-08-04"], [100, 101])
        calls = []
        def downloader(*_):
            calls.append(1); return dirty if len(calls) == 1 else clean
        now = datetime(2026, 8, 5, tzinfo=timezone.utc)
        self.store.update("AUDIT", "1d", downloader, now, max_attempts=1)
        result = self.store.update("AUDIT", "1d", downloader, now, max_attempts=1)
        self.assertEqual(result.invalid_rows_removed, 0)
        self.assertEqual(result.invalid_rows_removed_total, 1)

    def test_full_then_incremental_matches_one_full_merge_and_writes_metadata(self):
        calls = []
        first = bars(["2026-08-03", "2026-08-04"], [100, 101])
        second = bars(["2026-08-04", "2026-08-05"], [102, 103])

        def downloader(ticker, interval, start, end, period):
            calls.append((start, end))
            return first if len(calls) == 1 else second

        now = datetime(2026, 8, 5, tzinfo=timezone.utc)
        initial = self.store.update("AAOI", "1d", downloader, now)
        refreshed = self.store.update("AAOI", "1d", downloader, now)
        stored = self.store.read("AAOI", "1d")
        expected, _, _ = self.store.merge(
            self.store.normalize(first, "1d"), self.store.normalize(second, "1d"))
        pd.testing.assert_frame_equal(stored, expected, check_freq=False)
        self.assertIsNone(calls[0][0])
        self.assertIsNotNone(calls[1][0])
        self.assertEqual(initial.status, "updated")
        self.assertEqual(refreshed.rows_added, 1)
        metadata = json.loads(self.store.metadata_path("AAOI", "1d").read_text())
        self.assertEqual(metadata["rows"], 3)
        self.assertEqual(metadata["duplicates_removed"], 1)

    def test_failed_refresh_preserves_existing_parquet(self):
        def good(*_):
            return bars(["2026-08-03"], [100])

        def bad(*_):
            raise TimeoutError("source unavailable")

        now = datetime(2026, 8, 5, tzinfo=timezone.utc)
        self.store.update("AAOI", "1d", good, now)
        failed = self.store.update("AAOI", "1d", bad, now, max_attempts=1)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(len(self.store.read("AAOI", "1d")), 1)
        self.assertIn("TimeoutError", failed.error)

    def test_temporary_failure_is_retried_then_succeeds(self):
        calls = []
        def flaky(*_):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("temporary")
            return bars(["2026-08-03"], [100])
        result = self.store.update("NEW", "1d", flaky,
            datetime(2026, 8, 5, tzinfo=timezone.utc), max_attempts=2, backoff_seconds=0)
        self.assertEqual(result.status, "updated")
        self.assertEqual(result.attempts, 2)

    def test_identical_refresh_does_not_rewrite_parquet(self):
        frame = bars(["2026-08-03"], [100])
        now = datetime(2026, 8, 5, tzinfo=timezone.utc)
        self.store.update("AAOI", "1d", lambda *_: frame, now)
        path = self.store.data_path("AAOI", "1d")
        before = path.read_bytes()
        result = self.store.update("AAOI", "1d", lambda *_: frame, now)
        self.assertEqual(result.status, "current")
        self.assertEqual(path.read_bytes(), before)

    def test_nonexistent_ticker_records_failure_checkpoint(self):
        def missing(*_):
            raise ValueError("no price data returned; possibly delisted")
        result = self.store.update("NONE", "1d", missing,
            datetime(2026, 8, 5, tzinfo=timezone.utc), max_attempts=1)
        self.assertEqual(result.error_type, "not_found")
        self.assertTrue(self.store.metadata_path("NONE", "1d").exists())

    def test_retry_selection_uses_only_failed_checkpoints(self):
        now = datetime(2026, 8, 5, tzinfo=timezone.utc)
        self.store.update("GOOD", "1d", lambda *_: bars(["2026-08-03"], [100]), now)
        self.store.update("BAD", "1d", lambda *_: (_ for _ in ()).throw(
            TimeoutError("temporary")), now, max_attempts=1)
        self.assertEqual(select_failed_symbols(
            self.store, ["GOOD", "BAD", "UNKNOWN"], ["1d"]), ["BAD"])

    def test_quality_score_rewards_complete_current_data(self):
        good = quality_score(1, 0, 0, 1, 0, True, 100)
        poor = quality_score(.7, 4, 3, .8, 10, False, 100)
        self.assertEqual(good, 1.0)
        self.assertLess(poor, good)

    def test_report_is_created_with_partial_failure_and_benchmark_sync(self):
        now = datetime.now(timezone.utc)
        dates = pd.bdate_range(end=now.date(), periods=10)
        def good(*_): return bars(dates, list(range(100, 110)))
        def bad(*_): raise TimeoutError("temporary")
        self.store.update("SPY", "1d", good, now, max_attempts=1)
        self.store.update("AAOI", "1d", good, now, max_attempts=1)
        self.store.update("FAIL", "1d", bad, now, max_attempts=1)
        report_dir = Path(self.temp.name) / "reports"
        report, summary = build_daily_quality_report(
            self.store, ["SPY", "AAOI", "FAIL"], report_dir)
        self.assertEqual(summary["tickers"], 3)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(float(report.loc[report.ticker == "AAOI", "spy_overlap_ratio"].iloc[0]), 1.0)
        for suffix in ("parquet", "csv", "html"):
            self.assertTrue((report_dir / f"data_quality_daily.{suffix}").exists())


if __name__ == "__main__":
    unittest.main()
