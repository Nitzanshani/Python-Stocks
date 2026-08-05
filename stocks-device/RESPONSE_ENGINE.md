# Response Engine (Phase 3 Design)

`response_engine.py` will measure target-stock behavior after a source event.
It is not implemented in Phase 2.

Daily horizons begin with 1, 2, 3, 5 and 10 trading sessions. Intraday horizons
of 5, 15, 30 and 60 minutes are Phase 5. Every response begins at the event's
`data_available_at`, never at an unknowable retrospective extreme.

Outputs will include raw/market-adjusted/sector-adjusted return, positive and
negative extrema, time to response, response area, duration, decay and reversal.
The initial residual model will be causal rolling OLS using market and sector
returns. Raw, market-only and market+sector results will all be retained.

Synthetic tests will include A leading B, no reverse link, a false common-market
link that disappears after controls, a true residual link, and known decay.
