# Phase 3B Intraday Predictive Influence

Phase 3B is isolated from the daily Phase 3A pipeline and all dashboard logic.
Research reads only local Parquet bars. Downloads occur only through the manual
`update_intraday_research_data.py` command.

## Frozen pre-analysis choices

The universe is nine research stocks with SPY, QQQ, SMH and SOXX controls.
Observed Yahoo retention produced 41 complete sessions for every symbol at 5m,
15m, 30m and 60m. Before viewing influence results, validation was fixed at 20
train sessions, 5 test sessions, step 5, purge 30 minutes and embargo 1 session.
Fewer than 30 complete sessions produces `insufficient_data`.

## Causal bar semantics

All storage is UTC; sessions are interpreted in America/New_York using the NYSE
calendar. A bar is available only at `bar_end`. Events may affect only bars
whose start is at or after that time and never cross the session close by
default. Missing bars are not filled.

Volume, volatility and return baselines compare the same time-of-day slot using
previous sessions only. Historical same-slot OLS removes SPY and SMH using only
prior sessions. Raw, market-adjusted and market-plus-sector residuals remain
separate.

## Events and responses

The engine stores bar, cumulative, volume, combined and first threshold-crossing
events. Raw events, clusters and representatives are separate. The representative
is always the earliest valid detection; the later maximum-magnitude event is
stored only as cluster metadata. Responses expose point and cumulative returns,
excursions, reversal, amplitude ratio and explicit missing reasons.

Generated files live under `research_intraday/` and are ignored by Git.

## Validation and inference

Each directed pair is evaluated in non-overlapping session folds: 20 training
sessions, 5 test sessions and a 5-session step. The last 30 minutes of the final
training session are purged and one session is embargoed. Scaling, ridge-alpha
selection and every fitted coefficient are learned inside the training fold.
The baseline contains only lagged target/market/sector state; the extended model
adds lagged source residuals, source volume and causally available source-event
state. Excursions and other future outcomes are never features.

Event studies preserve two baselines. `unconditional` controls share only clock
slot and weekday. `matched` controls additionally use contemporaneously known
SPY/SMH direction and magnitude plus source/target state up to the event time.
Every selected control-bar identifier is stored and
`target_future_used_for_matching` is always false. Simple permutation and
session-block bootstrap estimates are retained. Benjamini-Hochberg correction
uses declared families split by analysis type, interval, horizon, event type,
residual model and test period.

Responses use fixed trading-minute horizons 5, 10, 15, 30, 60 and 120, plus an
explicit `session_close` horizon. The latter participates in event studies but
is excluded from fixed-minute decay fitting. `same_bar` is ambiguity metadata,
not evidence of direction. Relationship statuses remain experimental.

## Reproduction

```bash
python3 update_intraday_research_data.py --config intraday_influence_config.json
python3 run_intraday_influence_research.py --config intraday_influence_config.json
python3 -m unittest test_phase3b_intraday.py
```

The manifest records code/configuration/input hashes and dependency versions.
Reports are static HTML, CSV and Markdown and compare Phase 3A with Phase 3B.

## Frozen replication baseline

Phase 3B was frozen at commit `420fbd9` and tag
`phase3b-intraday-baseline`. `PHASE3B_FROZEN_SPEC.json` and its canonical SHA-256
lock every analytical choice. Discovery ends on 2026-08-05; confirmation is the
next 60 scheduled NYSE sessions, and the subsequent holdout is locked by default.
