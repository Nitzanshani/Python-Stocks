# Phase 3C — Out-of-Time Intraday Replication

Phase 3C never tunes the Phase 3B model. The immutable Phase 3B specification is
`PHASE3B_FROZEN_SPEC.json`; its canonical SHA-256 must equal the adjacent lock
file. Any mismatch stops a replication run and becomes a new experiment.

## Fixed periods

- Discovery: 2026-06-08 through 2026-08-05.
- Confirmation: 2026-08-06 through 2026-10-29, exactly 60 scheduled sessions.
- Holdout: begins 2026-10-30 and remains locked until explicitly opened after
  at least 60 complete sessions.

The boundaries were declared with zero confirmation sessions available. The
holdout CLI requires `--unlock-holdout`, and every opening is recorded in the
run manifest.

## Practical significance

FDR alone is insufficient. The predeclared minimum practical effect requires at
least 0.5% relative RMSE improvement, 0.00001 absolute RMSE improvement, one
basis point of response, an effect-to-cost-proxy ratio of one, 60% fold
consistency and ten events. High-low range and median absolute bar return are
cost proxies only; they are never described as a real bid/ask spread.

## Diagnostics

The confirmation engine preserves all 432 directed Phase 3B definitions and
the fixed NVDA → ANET 5m/5m definition. It writes matched controls, fold-local
walk-forward results, event-type results, rolling 20-session/5-session-step
stability, leave-one-event-out concentration, predefined regimes, deterministic
placebos and Phase 3A/3B discovery comparisons. Negative and insufficient-data
results remain in the outputs.

Until 60 complete confirmation sessions exist, the CLI produces accumulation
reports only and labels all 432 relationships `insufficient_data`.

```bash
python3 update_intraday_research_data.py --symbols NVDA,MRVL,AMD,AVGO,COHR,AAOI,QCOM,ANET,MU,SPY,QQQ,SMH,SOXX --intervals 5m,15m,30m,60m --resume --quality-report --verify-sessions
python3 run_intraday_replication.py --frozen-spec PHASE3B_FROZEN_SPEC.json --period confirmation --all-relationships --rolling-stability --placebo
```

Data update and research execution are deliberately separate commands.
