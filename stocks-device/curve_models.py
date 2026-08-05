"""Causal curve fitting and walk-forward comparison for stock price series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class CurveModelResult:
    name: str
    window: int
    quality: float
    validation_nrmse: float
    direction_accuracy: float
    turn_accuracy: float
    stability: float
    horizon_skill: dict[str, float]
    forecast_5d_pct: float
    phase: str
    next_turn: float | None
    next_turn_type: str | None
    chart: list[dict[str, float | int | bool | None]]

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _design(x, degree: int):
    import numpy as np
    return np.vander(np.asarray(x, dtype=float), degree + 1, increasing=True)


def _poly_predict(train, future_x, degree: int):
    import numpy as np
    y = np.log(np.asarray(train, dtype=float))
    n = len(y); center = (n - 1) / 2; scale = max(center, 1)
    x = (np.arange(n) - center) / scale
    matrix = _design(x, degree)
    penalty = np.eye(degree + 1) * 1e-5
    penalty[0, 0] = 0
    beta = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ y)
    query = (np.asarray(future_x, dtype=float) - center) / scale
    return np.exp(_design(query, degree) @ beta)


def _linear_predict(train, future_x):
    return _poly_predict(train, future_x, 1)


def _spline_predict(train, future_x):
    """Cubic smoothing spline; lambda is scaled to the series' noise."""
    import numpy as np
    from scipy.interpolate import make_smoothing_spline
    y = np.log(np.asarray(train, dtype=float))
    x = np.arange(len(y), dtype=float)
    noise = float(np.median(np.abs(np.diff(y) - np.median(np.diff(y)))))
    lam = max(1e-6, len(y) * noise * noise * 8)
    spline = make_smoothing_spline(x, y, lam=lam)
    return np.exp(spline(np.asarray(future_x, dtype=float)))


def _predict(name: str, train, future_x):
    if name == "Linear / drift":
        return _linear_predict(train, future_x)
    if name.startswith("Polynomial"):
        return _poly_predict(train, future_x, int(name.rsplit(" ", 1)[1]))
    return _spline_predict(train, future_x)


def _turn(values) -> tuple[float | None, str | None]:
    import numpy as np
    values = np.asarray(values, dtype=float)
    slopes = np.diff(values)
    for index in range(1, len(slopes)):
        if slopes[index - 1] > 0 >= slopes[index]:
            return float(index), "peak"
        if slopes[index - 1] < 0 <= slopes[index]:
            return float(index), "trough"
    return None, None


def _phase(fitted) -> str:
    import numpy as np
    slopes = np.diff(np.asarray(fitted, dtype=float))
    current = slopes[-1]
    scale = max(float(np.std(slopes)), 1e-9)
    if abs(current) <= 0.25 * scale:
        return "near peak" if len(slopes) > 1 and slopes[-2] > 0 else "near trough"
    return "rising" if current > 0 else "falling"


def _evaluate(name: str, values, window: int) -> CurveModelResult | None:
    import numpy as np
    values = np.asarray(values, dtype=float)
    horizon = 10
    origins = list(range(max(window, len(values) - 80), len(values) - horizon, 3))
    if len(origins) < 3:
        return None
    errors, naive_errors, directions, turn_hits, predictions = [], [], [], [], []
    horizon_errors = {1: [[], []], 5: [[], []], 10: [[], []]}
    for origin in origins:
        train = values[origin - window:origin]
        actual = values[origin:origin + horizon]
        try:
            predicted = _predict(name, train, np.arange(window, window + horizon))
        except (ValueError, ArithmeticError, ImportError):
            return None
        errors.extend((np.log(predicted) - np.log(actual)).tolist())
        naive_errors.extend((np.log(train[-1]) - np.log(actual)).tolist())
        for step in horizon_errors:
            horizon_errors[step][0].append(float(np.log(predicted[step - 1]) - np.log(actual[step - 1])))
            horizon_errors[step][1].append(float(np.log(train[-1]) - np.log(actual[step - 1])))
        directions.append(float(np.sign(predicted[4] - train[-1]) == np.sign(actual[4] - train[-1])))
        predicted_turn, predicted_kind = _turn(np.r_[train[-1], predicted])
        actual_turn, actual_kind = _turn(np.r_[train[-1], actual])
        if predicted_turn is not None:
            turn_hits.append(float(actual_turn is not None and actual_kind == predicted_kind
                                   and abs(actual_turn - predicted_turn) <= 2))
        predictions.append(float((predicted[4] / train[-1] - 1) * 100))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    naive = max(float(np.sqrt(np.mean(np.square(naive_errors)))), 1e-9)
    forecast_q = naive / (naive + rmse)
    direction = float(np.mean(directions))
    turn_accuracy = float(np.mean(turn_hits)) if turn_hits else 0.5
    stability = math.exp(-float(np.std(predictions)) / max(abs(float(np.mean(predictions))), 2.0))
    coverage = math.exp(-abs(float(np.mean(errors))) / max(rmse, 1e-9))
    quality = .35 * forecast_q + .20 * direction + .15 * turn_accuracy + .15 * stability + .15 * coverage
    train = values[-window:]
    future_x = np.arange(window, window + 11)
    fitted = _predict(name, train, np.arange(window))
    forecast = _predict(name, train, future_x)
    next_turn, turn_type = _turn(np.r_[train[-1], forecast])
    chart = ([{"x": i - window + 1, "actual": float(v), "fitted": float(f)}
              for i, (v, f) in enumerate(zip(train, fitted))] +
             [{"x": i + 1, "actual": None, "fitted": float(v), "forecast": True}
              for i, v in enumerate(forecast)])
    horizon_skill = {}
    for step, (model_errors, base_errors) in horizon_errors.items():
        model_rmse = float(np.sqrt(np.mean(np.square(model_errors))))
        base_rmse = max(float(np.sqrt(np.mean(np.square(base_errors)))), 1e-9)
        horizon_skill[f"{step}d"] = round(base_rmse / (base_rmse + model_rmse), 4)
    return CurveModelResult(name, window, round(quality, 4), round(rmse, 6),
        round(direction, 4), round(turn_accuracy, 4), round(stability, 4), horizon_skill,
        round(float((forecast[4] / train[-1] - 1) * 100), 3), _phase(fitted),
        next_turn, turn_type, chart)


def fit_curve_models(frame) -> dict[str, object] | None:
    """Compare all requested models/windows using causal walk-forward validation."""
    import numpy as np
    values = frame["Close"].dropna().to_numpy(dtype=float)[-252:]
    if len(values) < 40 or np.any(values <= 0):
        return None
    models = ("Linear / drift", "Polynomial 3", "Polynomial 4", "Polynomial 5",
              "Cubic smoothing spline")
    results = []
    for name in models:
        candidates = [_evaluate(name, values, window) for window in (14, 21, 42, 63)
                      if len(values) >= window + 20]
        results.extend(result for result in candidates if result is not None)
    if not results:
        return None
    best_per_model = [max((r for r in results if r.name == name), key=lambda r: r.quality)
                      for name in models if any(r.name == name for r in results)]
    best = max(best_per_model, key=lambda result: result.quality)
    signs = [math.copysign(1, result.forecast_5d_pct) for result in best_per_model
             if abs(result.forecast_5d_pct) >= 0.1]
    agreement = (max(signs.count(1.0), signs.count(-1.0)) / len(signs)) if signs else 0.0
    return {"best": best.payload(), "models": [result.payload() for result in best_per_model],
            "agreement": round(agreement, 3)}
