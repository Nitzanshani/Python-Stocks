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

## Matched-control event study

Control days require no source event, matching SPY and SMH directions, and the
nearest trailing 20-day market volatility. The selected dates and distances are
stored in `event_study_controls.parquet`. Target future availability may reject
a candidate, but the magnitude or sign of its future response is never used for
selection. Event effects are compared with simple bootstrap, moving-block
bootstrap (default block length 5) and a permutation test.

Benjamini-Hochberg correction is applied within a declared event-study family:
one event type × horizon × response metric × analysis window.

## Purged walk-forward prediction

The target is next-session target residual return. Baseline features contain
target lags 1–5, lagged SPY/SMH, trailing target volatility, relative volume and
day of week. The extended model adds source raw/residual lags, event flag and
magnitude, days since event and source relative volume. Excursions are excluded.

Each chronological fold defaults to 504 train, 63 test and step 21, with 10
purged sessions and 5 embargoed sessions. Scaling and Ridge-alpha selection use
only the fold's training segment; the alpha uses an inner chronological holdout
with its own purge. Results are compared with the identical baseline and a zero
return naive forecast. Overlapping test folds are retained as repeated regime
tests, not treated as independent observations.

## Granger and multiple testing

Granger F-tests use residual returns at the four pre-declared lags 1, 2, 3 and
5. Every lag and both A → B and B → A are stored. A Granger family is one lag ×
analysis window across all directed pairs. Granger evidence is complementary
and does not demonstrate economic causality.

## Relationship status and experimental scores

Statuses are `insufficient_data`, `no_evidence`, `candidate`, `validated`,
`unstable` and `rejected`. A relationship cannot be validated without minimum
events/folds, positive out-of-sample improvement, consistency and a meaningful
event effect. Strong in-sample effects that fail out of sample are `rejected`.

The six exposed 0–1 components are prediction improvement, event effect,
fold consistency, significant-lag share, FDR evidence and sample confidence.
Three sensitivity schemes are reported: balanced, prediction-heavy and
statistics-heavy. These scores are explicitly experimental and are not a
production ranking or trading signal; missing evidence is not imputed as a
positive score.

## Generated files

```text
research/events/daily_events.parquet
research/events/daily_event_clusters.parquet
research/events/daily_representative_events.parquet
research/residuals/daily_residual_returns.parquet
research/responses/daily_responses.parquet
research/responses/event_study_controls.parquet
research/responses/event_study_summary.parquet
research/influence/predictive_model_folds.parquet
research/influence/granger_results.parquet
research/influence/influence_relationships.parquet
research/reports/phase3a_daily_influence.{html,csv,md}
research/run_manifest.json
```

Use `--resume` to reuse completed stages and `--report-only` to rebuild static
reports. `run_manifest.json` records the exact configuration and input hashes,
dependency versions and commit for reproducibility.
