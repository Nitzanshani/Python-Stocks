# Response Engine (Phase 3 Design)

`response_engine.py` measures target-stock behavior after a source event.
Point and cumulative responses are explicit separate fields.

Daily horizons begin with 1, 2, 3, 5 and 10 trading sessions. Intraday horizons
of 5, 15, 30 and 60 minutes are Phase 5. Every response begins at the event's
`data_available_at`, never at an unknowable retrospective extreme.

Outputs will include raw/market-adjusted/sector-adjusted return, positive and
negative extrema, time to response, response area, duration, decay and reversal.
The initial residual model will be causal rolling OLS using market and sector
returns. Raw, market-only and market+sector results will all be retained.

Synthetic tests will include A leading B, no reverse link, a false common-market
link that disappears after controls, a true residual link, and known decay.

For horizon `h`, the cumulative response is `exp(sum(log_return[t+1:t+h]))-1`;
the point response is the return at exactly the h-th observed target session.
Missing target sessions are not filled. Maximum/minimum excursions are stored
only as descriptive outcomes and are prohibited from predictive features.
Self-responses are excluded from the main response table.

Phase 3B implements fixed same-session horizons 5/10/15/30/60/120 minutes and
`session_close`. Missing bars are rejected, never filled. Point and cumulative
responses remain separate; same-bar cases are retained as ambiguity metadata.
