from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pandas as pd

from market_data_store import MarketDataStore


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
        failed = self.store.update("AAOI", "1d", bad, now)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(len(self.store.read("AAOI", "1d")), 1)
        self.assertIn("TimeoutError", failed.error)


if __name__ == "__main__":
    unittest.main()
