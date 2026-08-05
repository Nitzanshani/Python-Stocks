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
