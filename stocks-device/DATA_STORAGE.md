# Local Market Data Storage

## Purpose

Phase 2 adds an optional local Parquet source. It does not yet replace the
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
   atomic replacement.

The overlap is intentional: Yahoo may revise recent bars or adjustments.

## Metadata

Every attempt records ticker, interval, UTC update time, first/last stored
timestamp, row count, requested range, downloaded/added row counts, duplicates,
invalid rows, status and error. A failed request does not destroy existing data.

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
python3 update_market_data.py --intervals 1d
```

The second command updates the full current universe. Heavy multi-interval
updates are manual and must not run automatically when the web server starts.
PyArrow is kept in `requirements-research.txt`, not Render's normal
`requirements.txt`, so Phase 2 does not enlarge or destabilize the existing web
deployment before the dashboard migrates to Parquet.

## Known limitations

- The current universe is not point-in-time and causes survivorship bias.
- Render storage may be ephemeral; local Mac storage is the initial authority.
- Exchange calendar completeness and split parity checks are scheduled before
  existing analytics migrate to this source.
