"""Compare curve models on the same 20-stock research universe."""

from __future__ import annotations

import csv
from pathlib import Path

from curve_models import fit_curve_models
from experiment_hitting_wave_starts import SYMBOLS

OUTPUT = Path(__file__).with_name("curve_model_experiment.csv")


def main() -> int:
    import yfinance as yf

    raw = yf.download(SYMBOLS, period="1y", interval="1d", auto_adjust=True,
                      actions=False, progress=False, threads=True, timeout=30,
                      group_by="ticker", multi_level_index=True)
    rows = []
    for symbol in SYMBOLS:
        try:
            frame = raw[symbol][["Close"]].dropna()
        except (KeyError, TypeError):
            continue
        result = fit_curve_models(frame)
        if not result:
            continue
        for model in result["models"]:
            rows.append({"symbol": symbol, "model": model["name"],
                "window": model["window"], "quality": model["quality"],
                "validation_nrmse": model["validation_nrmse"],
                "direction_accuracy": model["direction_accuracy"],
                "turn_accuracy": model["turn_accuracy"],
                "stability": model["stability"],
                "forecast_5d_pct": model["forecast_5d_pct"],
                "selected_best": model["name"] == result["best"]["name"]})
        print(symbol, result["best"]["name"], result["best"]["window"],
              result["best"]["quality"])
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} model results to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
