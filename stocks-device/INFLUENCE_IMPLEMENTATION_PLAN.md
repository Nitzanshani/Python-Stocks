# Dynamic Influence Research — Implementation Plan

## Existing architecture audit

The existing application remains the production baseline. `market_scanner.py`
loads the S&P 500/Nasdaq-100 universe, normalizes symbols and starts the current
CLI or browser UI. `stock_gui.py` downloads one-minute quotes. `web_gui.py`
downloads five-minute opening data and daily/hourly reference data, then owns
the Oscillation, ZigZag, Fourier, Curve Models, API and dashboard rendering.
News caches are independent JSON stores. Backtests currently call yfinance
directly and deliberately remain unchanged during Phase 2.

Reusable components:

- `load_universe_details`, `normalize_symbol`, and `chunks`.
- `_reference_daily_frame` for Daily Close/First-hour selection.
- Existing America/New_York conversion and regular-session boundaries.
- Existing manual-job pattern in `StockState`; heavy influence jobs must copy
  this opt-in behavior rather than run at server startup.
- Existing no-look-ahead conventions in pivot confirmation and walk-forward
  validation.

## Minimal new files

- `market_data_config.json`: intervals, paths and resource limits.
- `market_data_store.py`: Parquet IO, validation, merge and incremental update.
- `update_market_data.py`: independently runnable Phase 2 CLI.
- `universe_metadata.py`: Phase 2.5 company/sector/index membership snapshot.
- `data_quality.py`: NYSE-calendar audit and Parquet/CSV/HTML reports.
- `verify_daily_idempotency.py`: repeat-run validation for the daily store.
- `event_engine.py` and `response_engine.py`: Phase 3 only.
- `pairwise_plugins.py`, `influence_matrix.py`: Phase 4–6 only.
- `influence_jobs.py`: resumable job/checkpoint layer before dashboard wiring.
- The documentation files required by the research specification.

## Minimal changes to existing files

Phase 2.5 remains additive. No current Yahoo
reader is replaced yet. Migration will happen reader by reader after parity
tests: daily analytics first, hourly reference series second, five-minute
opening analytics third, and live one-minute quotes last (or never, because
live quotes have different freshness requirements).

`market_scanner.py` remains the normal entry point. The storage CLI is additive:

```bash
python3 update_market_data.py --symbols AAOI,COHR --intervals 1d,60m,5m
```

## Milestones

1. Phase 2: validated Parquet storage, metadata, overlap refresh and tests.
2. Phase 2.5: full daily universe, resilient checkpoints, universe mapping,
   benchmark/sector ETF synchronization and quality audit. Completed locally;
   no analytical formula or dashboard reader was changed.
3. Phase 3A: completed small-universe daily Event/Response Engines, causal
   residuals, matched controls, purged walk-forward, Granger/FDR and static
   research reports. Controls remain outside the central target ranking.
4. Phase 4: candidate pair screening, baseline-vs-extended walk-forward and FDR.
5. Phase 5: intraday events/responses after data-availability validation.
6. Phase 6: Influence Matrix, network snapshots and evolution.
7. Phase 7: separate dashboard page, animation and reports.
8. Phase 8: DMD/Prony/Matrix Pencil/state-space research, isolated from Freq Q.

## Technical risks

- Yahoo intraday retention and rate limits; metadata must report observed data.
- Adjusted historical prices can be revised after splits/dividends, so updates
  intentionally overlap and replace recent timestamps.
- Current-index constituents introduce survivorship bias in old research.
- DST, early closes, missing bars and exchange holidays can mimic missing data.
- More than 250,000 directed pairs per window require staged screening and disk
  batches rather than one all-pairs DataFrame.
- Render's ephemeral filesystem cannot be the authoritative research store.
- Parquet schema/version migrations need explicit versioning before Phase 3.
- Statistical association is not causality; shared market/sector/news factors
  must be controlled and negative results retained.

## Migration safety

Every existing formula and column remains untouched. A future reader switches
to local data only after a full-vs-incremental parity test and a fallback to the
current Yahoo path. Each phase receives its own tests, documentation, example
result and commit.
