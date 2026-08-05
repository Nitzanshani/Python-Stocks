"""Static Phase 3A research report (not a dashboard integration)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_phase3a_report(output_root: Path, config: dict, residuals: pd.DataFrame,
                         events: pd.DataFrame, responses: pd.DataFrame,
                         folds: pd.DataFrame, granger: pd.DataFrame,
                         relationships: pd.DataFrame) -> None:
    reports = output_root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    relationships.to_csv(reports / "phase3a_daily_influence.csv", index=False)
    dates = pd.to_datetime(residuals.timestamp, utc=True)
    nvda = relationships.loc[(relationships.source_ticker == "NVDA") |
                             (relationships.target_ticker == "NVDA")]
    columns = ["source_ticker", "target_ticker", "event_count", "event_type", "horizon",
        "abnormal_effect", "effect_size", "rmse_improvement", "improvement_consistency",
        "lag", "adjusted_p_value_granger", "experimental_score_balanced",
        "experimental_score_prediction_heavy", "experimental_score_statistics_heavy",
        "relationship_status"]
    pairwise = relationships[[c for c in columns if c in relationships]].copy()
    negative = relationships.loc[relationships.relationship_status.isin(
        ["rejected", "unstable", "insufficient_data", "no_evidence"])]
    event_counts = events.groupby(["ticker", "event_type", "direction"]).size().rename("events").reset_index()
    summary = {"date_start": dates.min(), "date_end": dates.max(),
        "research_symbols": ", ".join(config["research_symbols"]),
        "control_symbols": ", ".join(config["control_symbols"]),
        "events": len(events), "positive_events": int((events.direction == "positive").sum()),
        "negative_events": int((events.direction == "negative").sum()),
        "valid_responses": int((responses.response_status == "ok").sum()),
        "directed_pairs": len(relationships), "walk_forward_folds": len(folds)}
    lines = ["# Phase 3A Daily Predictive Influence", "",
        "> Experimental research only. Scores are sensitivity analyses, not trading signals.", "",
        "## Data Summary", "", *(f"- **{k}:** {v}" for k,v in summary.items()), "",
        "## Method", "",
        "Adjusted-close log returns; coefficients for t are estimated through t-1. Events are known after close and responses start next session. Controls are selected without target future returns. Walk-forward preprocessing and Ridge selection occur inside each purged fold.", "",
        "## FDR families", "",
        "Event-study tests are corrected separately for each event type, horizon, response metric and analysis window. Granger tests are corrected separately for each declared lag and analysis window; every lag is retained.", "",
        "## Relationship statuses", "",
        *[f"- {status}: {count}" for status,count in relationships.relationship_status.value_counts().items()], "",
        "## NVDA directed experiment", "",
        "Every NVDA → peer and peer → NVDA direction is reported independently.", "",
        nvda.to_csv(index=False), "", "## Negative findings", "",
        negative.to_csv(index=False), "", "## Limitations", "",
        "Survivorship bias; unofficial Yahoo source; daily resolution; event detection only after close; no intraday ordering; multiple testing; regime changes; statistical association is not economic causality."]
    (reports / "phase3a_daily_influence.md").write_text("\n".join(lines), encoding="utf-8")
    cards = "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k,v in summary.items())
    html = f'''<!doctype html><meta charset="utf-8"><meta name="robots" content="noindex">
<title>Phase 3A Daily Predictive Influence</title><style>body{{font:14px system-ui;background:#0b1117;color:#dce7f2;padding:26px}}h1,h2{{color:white}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}.cards div{{padding:12px;background:#172433;border:1px solid #31506c;border-radius:9px}}.cards b,.cards span{{display:block}}.cards b{{color:#35d07f}}table{{border-collapse:collapse;width:100%;font-size:11px}}th,td{{padding:6px;border:1px solid #29394a}}th{{background:#192a3b;position:sticky;top:0}}.box{{overflow:auto;max-height:600px}}</style>
<h1>Phase 3A Daily Predictive Influence</h1><p><b>Experimental research only — not a trading signal.</b></p><div class="cards">{cards}</div>
<h2>Event summary</h2><div class="box">{event_counts.to_html(index=False)}</div>
<h2>Directed pairwise results</h2><div class="box">{pairwise.to_html(index=False)}</div>
<h2>NVDA ↔ peers</h2><div class="box">{nvda.to_html(index=False)}</div>
<h2>Negative findings</h2><div class="box">{negative.to_html(index=False)}</div>
<h2>Limitations</h2><p>Survivorship bias; Yahoo Finance; daily resolution; events known only after close; no intraday ordering; multiple testing; regime changes; statistical association is not economic causality.</p>'''
    (reports / "phase3a_daily_influence.html").write_text(html, encoding="utf-8")
