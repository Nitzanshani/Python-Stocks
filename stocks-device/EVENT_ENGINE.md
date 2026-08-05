# Event Engine (Phase 3 Design)

`event_engine.py` converts abnormal daily moves into immutable, uniquely
identified research events. Phase 3A preserves raw events, cooldown clusters
and representative events as separate datasets.

Inputs will come from the validated local store. Thresholds, horizons and
minimum completeness will live in configuration rather than source code.
Every event will contain raw and residual returns, volume/volatility context,
sector/industry, `event_timestamp`, and the causal `data_available_at` time.

Initial daily events: ±10% return, rolling 3-sigma return, abnormal opening gap
and abnormal relative volume. Intraday events are postponed until Phase 5.
Existing opening-session scores, ZigZag, Oscillation, Fourier and Curve Models
will not be reused as event labels unless a later experiment explicitly opts in.

Tests must cover threshold boundaries, missing bars, splits, early closes,
timezone conversion and events whose detection becomes possible only at a bar's
close. Full formulas and dashboard hover text will be added with Phase 3.
