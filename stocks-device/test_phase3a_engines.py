from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from event_engine import cluster_daily_events, detect_daily_events
from influence_features import build_feature_panel, build_return_features
from market_data_api import MarketDataUnavailable, load_market_data
from residual_returns import build_residual_returns
from response_engine import measure_daily_responses


def prices(returns, start="2024-01-02"):
    dates = pd.bdate_range(start, periods=len(returns), tz="UTC")
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Adj Close": close, "Volume": np.arange(len(close)) + 1000,
        "Dividends": 0.0, "Stock Splits": 0.0}, index=dates)


CONFIG = {"engine_version": "test", "feature_window": 5,
    "events": {"raw_return_threshold": .10, "residual_z_threshold": 3,
               "relative_volume_threshold": 2, "threshold_version": "v1"}}


class Phase3AEngineTests(unittest.TestCase):
    def test_local_api_parity_sorted_utc_and_defensive_copy(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / "market/daily/AAOI.parquet"
            path.parent.mkdir(parents=True)
            source = prices([0, .01, -.01]).iloc[::-1]
            source.to_parquet(path)
            loaded = load_market_data("AAOI", data_root=root)
            direct = pd.read_parquet(path).sort_index()
            direct.index = pd.to_datetime(direct.index, utc=True); direct.index.name = "timestamp"
            pd.testing.assert_frame_equal(loaded, direct)
            loaded.iloc[0, 0] = -999
            self.assertNotEqual(pd.read_parquet(path).iloc[-1, 0], -999)
            with self.assertRaises(MarketDataUnavailable):
                load_market_data("NONE", data_root=root)

    def test_log_return_zscore_and_volume_baselines_exclude_current_day(self):
        frame = prices([0, .01, .02, .03, .04, .50, .01])
        features = build_return_features(frame, 5)
        self.assertAlmostEqual(features.log_return.iloc[2], .02, places=10)
        expected_mean = features.log_return.iloc[1:6].mean()
        self.assertAlmostEqual(features.rolling_mean.iloc[6], expected_mean)
        original_relative = features.relative_volume.iloc[5]
        changed = frame.copy(); changed.iloc[5, changed.columns.get_loc("Volume")] *= 100
        changed_features = build_return_features(changed, 5)
        self.assertAlmostEqual(changed_features.relative_volume.iloc[5], original_relative * 100)
        self.assertEqual(changed_features.relative_volume.iloc[6],
                         changed.Volume.iloc[6] / changed.Volume.iloc[1:6].mean())

    def _panel(self):
        rng = np.random.default_rng(7); n = 90
        market = rng.normal(0, .01, n); sector = rng.normal(0, .008, n)
        target = .6 * market + .4 * sector + rng.normal(0, .003, n)
        frames = {"SPY": prices(market), "SMH": prices(sector), "A": prices(target)}
        return build_feature_panel(frames, 5)

    def test_residual_for_t_uses_coefficients_only_through_t_minus_one(self):
        panel = self._panel()
        first = build_residual_returns(panel, "SPY", "SMH", 30, 20)
        cutoff = first.loc[(first.ticker == "A") & first.residual_status.eq("ok"), "timestamp"].iloc[10]
        changed = panel.copy()
        changed.loc[(changed.ticker == "A") & (changed.timestamp > cutoff), "log_return"] += 10
        second = build_residual_returns(changed, "SPY", "SMH", 30, 20)
        cols = ["alpha", "market_return_beta", "sector_return_beta", "residual_return"]
        a = first.loc[(first.ticker == "A") & (first.timestamp <= cutoff)].set_index("timestamp")[cols]
        b = second.loc[(second.ticker == "A") & (second.timestamp <= cutoff)].set_index("timestamp")[cols]
        pd.testing.assert_frame_equal(a, b)

    def test_event_availability_clustering_and_representative(self):
        panel = self._panel(); residual = build_residual_returns(panel, "SPY", "SMH", 20, 10)
        dates = residual.loc[residual.ticker == "A", "timestamp"].iloc[-5:-2].tolist()
        residual.loc[(residual.ticker == "A") & residual.timestamp.isin(dates), "raw_return"] = .20
        residual.loc[(residual.ticker == "A") & residual.timestamp.isin(dates), "residual_return"] = .20
        residual.loc[(residual.ticker == "A") & residual.timestamp.isin(dates), "residual_status"] = "ok"
        events = detect_daily_events(panel, residual, CONFIG)
        raw = events.loc[events.event_type == "raw_return"]
        self.assertGreaterEqual(len(raw), 3)
        for _, event in raw.iterrows():
            available = pd.Timestamp(event.data_available_at)
            self.assertEqual(available.tz_convert("America/New_York").hour, 16)
            self.assertEqual(available.date(), event.event_date.tz_convert("America/New_York").date())
        clusters, representatives = cluster_daily_events(raw, panel.timestamp.unique(), 3)
        self.assertLess(len(clusters), len(raw))
        self.assertEqual(len(representatives), len(clusters))
        self.assertTrue(set(representatives.event_id) <= set(raw.event_id))

    def test_responses_start_next_session_separate_point_and_cumulative(self):
        dates = pd.bdate_range("2026-01-02", periods=5, tz="UTC")
        residuals = pd.DataFrame({"ticker": "B", "timestamp": dates,
            "raw_return": [0, .1, -.05, .02, .01],
            "market_residual_return": [0, .08, -.04, .01, .01],
            "residual_return": [0, .07, -.03, .01, .01]})
        event = pd.DataFrame([{"event_id": "e", "ticker": "A", "event_date": dates[0],
            "direction": "positive", "raw_return": .2, "residual_return": .15}])
        result = measure_daily_responses(event, residuals, ["A", "B", "MISSING"], [1, 2])
        self.assertFalse((result.target_ticker == "A").any())
        b2 = result.loc[(result.target_ticker == "B") & (result.horizon == 2)].iloc[0]
        self.assertEqual(b2.response_start_date, dates[1])
        self.assertAlmostEqual(b2.point_raw_return, np.expm1(-.05))
        self.assertAlmostEqual(b2.target_raw_return, np.expm1(.1 - .05))
        self.assertEqual(result.loc[result.target_ticker == "MISSING", "response_status"].iloc[0],
                         "missing_target")


if __name__ == "__main__":
    unittest.main()
