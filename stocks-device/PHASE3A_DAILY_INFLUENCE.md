# Phase 3A Daily Predictive Influence

Phase 3A is an isolated research pipeline. It does not change the dashboard or
the existing Fourier, Freq Q, Phase, ZigZag, Oscillation or Curve Model logic.
It reads only the validated local daily Parquet store and never calls Yahoo.

## Scope and configuration

`influence_config.json` limits the ranked universe to NVDA, MRVL, AMD, AVGO,
COHR, AAOI, QCOM, ANET and MU. SPY, QQQ, SMH and SOXX are controls, not ranked
targets. Every threshold, horizon, window and random seed is configurable.

## Data and causal features

The default return is `log(Adj Close[t] / Adj Close[t-1])`. Prices are aligned
only on observed sessions and never forward-filled. Rolling mean, volatility,
Z-score and relative-volume baselines are shifted by one session, so an event
day is not included in its own threshold. An incomplete current daily bar is
excluded before analysis.

For a response on day `t`, rolling OLS coefficients are estimated only through
`t-1`. Both market-only and market-plus-sector models are retained. The default
sector control is SMH; SOXX is an alternative and is not included alongside SMH
by default because the two can be highly collinear.

## Events

Raw events include ±10% raw returns, ±3 historical residual standard deviations,
relative volume above 2, and combined shocks. IDs are deterministic over ticker,
date, type, direction and threshold version. `data_available_at` is 16:00 New
York time, so a daily response cannot begin before the next observed session.

The pipeline preserves three layers: all raw events, cooldown-based clusters,
and one representative event per cluster. The representative is the existing
raw event with the largest absolute standardized magnitude; source events are
never deleted.

## Responses

For each directed source-target pair, point and cumulative responses are stored
separately for 1, 2, 3, 5 and 10 observed sessions. Raw, market-adjusted and
market-plus-sector-adjusted responses are retained. Excursions are descriptive
response fields only and must never be predictive features. Self-response is
excluded from the main table.

## Reproducibility and output

`research/run_manifest.json` records configuration and input SHA-256 hashes,
engine/Python/dependency versions and Git commit. Generated research Parquet and
reports are ignored by Git. Run the first four stages with:

```bash
python3 run_daily_influence_research.py --config influence_config.json
```

The remaining sections—saved control-day event studies, purged walk-forward,
Granger/FDR, relationship status and experimental score sensitivity—are added
in the second Phase 3A implementation segment.
