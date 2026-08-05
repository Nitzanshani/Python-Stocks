# Local Market Data Storage

## Purpose

Phase 2/2.5 adds an optional local Parquet source. It does not yet replace the
dashboard's existing downloads. The goals are reproducibility, differential
updates, fewer repeated requests and an auditable data-availability record.

## Layout

```text
data/
  market/daily/{TICKER}.parquet
  market/hourly/{TICKER}.parquet
  market/30m/{TICKER}.parquet
  market/15m/{TICKER}.parquet
  market/5m/{TICKER}.parquet
  benchmarks/
  sectors/
  metadata/{INTERVAL}/{TICKER}.json
  cache/
```

The root can be overridden with `--data-root`. Parquet is the calculation
source; CSV/Excel are exports only.

## Schema and time

The index is a timezone-aware UTC `timestamp`. Naive Yahoo timestamps are
interpreted as America/New_York before conversion. Columns are raw `Open`,
`High`, `Low`, `Close`, `Adj Close`, `Volume`, `Dividends`, and `Stock Splits`.
Raw and adjusted values are retained so future event research can explicitly
choose its adjustment policy.

## Incremental algorithm

1. Read existing Parquet when present.
2. Initial daily request uses `period=max`; intraday uses the configured source
   lookback.
3. A refresh begins several bars before the last stored timestamp.
4. Normalize timezone/schema, reject invalid OHLC/volume rows, concatenate,
   sort, and keep the newly downloaded version of duplicate timestamps.
5. Write compressed Parquet and metadata through temporary files followed by
   atomic replacement. An identical refresh does not rewrite the Parquet file.

The overlap is intentional: Yahoo may revise recent bars or adjustments.

## Metadata

Every attempt records ticker, interval, UTC update time, first/last stored
timestamp, row count, requested range, downloaded/added row counts, duplicates,
invalid rows, status (`updated`, `current`, `failed`), retry count and classified
error (`rate_limit`, `timeout`, `not_found`, `parsing`, `source_error`). A failed
request does not destroy existing data and its checkpoint can be retried alone.

## Daily quality audit

Daily completeness uses the NYSE calendar, not weekdays. Each ticker reports
history range, expected/actual sessions, gaps, duplicates, invalid OHLC,
negative volume, update status, staleness, split/dividend warnings and overlap
with SPY. New listings are evaluated only from their first observed session.

The transparent quality score is:

```text
0.45 × coverage + 0.15 × duplicate quality + 0.15 × OHLC quality
+ 0.10 × completeness + 0.10 × freshness + 0.05 × update success
```

Reports are written to `reports/data_quality_daily.{parquet,csv,html}`. The
universe mapping (ticker, company, sector, industry, membership flags) is stored
at `data/metadata/universe.{parquet,csv}`.

## Yahoo limitations

The current yfinance API documents that intraday requests cannot extend beyond
the latest 60 days. Actual availability can differ by interval/ticker and may
change. The store therefore records what was actually returned and never labels
an intraday series “complete history.” Local accumulation can preserve bars
after Yahoo's rolling window has expired, provided updates run regularly.

## Commands

```bash
python3 -m pip install -r requirements-research.txt
python3 update_market_data.py --symbols AAOI,COHR --intervals 1d,60m
python3 update_market_data.py --intervals 1d --quality-report
python3 update_market_data.py --intervals 1d --retry-failed --quality-report
python3 update_market_data.py --symbols-file symbols.txt --batch-size 20 --max-workers 1
python3 update_market_data.py --symbols AAOI --intervals 1d --force-refresh
python3 verify_daily_idempotency.py
```

The second command updates the full current universe. Heavy multi-interval
updates are manual and must not run automatically when the web server starts.
PyArrow is kept in `requirements-research.txt`, not Render's normal
`requirements.txt`, so Phase 2 does not enlarge or destabilize the existing web
deployment before the dashboard migrates to Parquet.

## Known limitations

- The current universe is not point-in-time and causes survivorship bias.
- Render storage may be ephemeral; local Mac storage is the initial authority.
- Yahoo is an unofficial source and may revise history or return isolated bad
  rows; these are retained in audit metadata and flagged for manual review.
- Daily updates default to one worker because concurrent yfinance downloads can
  corrupt shared internal state in some versions.
