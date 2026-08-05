from datetime import date, datetime, timezone
import unittest

from stock_news_session_chart import _parse_historical_feed, _session_news_window


class StockNewsSessionChartTests(unittest.TestCase):
    def test_news_window_ends_at_market_open(self):
        start, end = _session_news_window(date(2026, 8, 3), date(2026, 8, 4))
        self.assertLess(start, end)
        self.assertEqual(end, datetime(2026, 8, 4, 13, 30, tzinfo=timezone.utc))

    def test_feed_counts_only_relevant_pre_open_unique_articles(self):
        xml = b'''<rss><channel>
      <item><title>AAOI stock rises on earnings - Reuters</title><link>one</link><source url="https://reuters.com">Reuters</source><pubDate>Tue, 04 Aug 2026 12:00:00 GMT</pubDate></item>
      <item><title>AAOI stock rises on earnings - Reuters</title><link>two</link><source url="https://reuters.com">Reuters</source><pubDate>Tue, 04 Aug 2026 12:10:00 GMT</pubDate></item>
      <item><title>AAOI stock falls later - Reuters</title><link>late</link><source url="https://reuters.com">Reuters</source><pubDate>Tue, 04 Aug 2026 15:00:00 GMT</pubDate></item>
    </channel></rss>'''
        result = _parse_historical_feed(
            xml, datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc),
            datetime(2026, 8, 4, 13, 30, tzinfo=timezone.utc),
            "AAOI", "Applied Optoelectronics Inc.")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "Reuters")
