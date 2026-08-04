"""Local browser UI for the live stock table."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.etree import ElementTree
from concurrent.futures import ThreadPoolExecutor, as_completed

from market_scanner import chunks, load_universe_details
from stock_gui import Quote, _download_quotes

TALK_CACHE_FILE = Path(__file__).with_name("talk_of_day_cache_v2.json")
FINANCE_SOURCES = {
    "benzinga", "bloomberg", "cnbc", "finance.yahoo.com", "fool.com", "forbes",
    "gurufocus", "investing.com", "investorplace", "marketbeat", "marketwatch",
    "morningstar", "nasdaq.com", "reuters", "seekingalpha.com", "thestreet",
    "tipranks", "zacks",
}


def _company_search_name(company: str) -> str:
    return re.sub(
        r"\s+(incorporated|inc\.?|corporation|corp\.?|company|co\.?|limited|ltd\.?|plc)$",
        "", company.strip(), flags=re.IGNORECASE,
    ).strip()


def _is_relevant_talk(title: str, source: str, domain: str, symbol: str, company: str) -> bool:
    company_name = _company_search_name(company)
    if len(company_name) >= 4 and company_name.casefold() in title.casefold():
        return True
    ticker_match = re.search(rf"(?<![A-Z0-9])\$?{re.escape(symbol)}(?![A-Z0-9])", title, re.I)
    finance_source = any(name in f"{source} {domain}".casefold() for name in FINANCE_SOURCES)
    finance_context = re.search(r"\b(stock|shares?|nasdaq|nyse|earnings|investors?|price target)\b", title, re.I)
    return bool(ticker_match and finance_source and finance_context)


def _parse_talk_feed(
    xml: bytes, cutoff: datetime, symbol: str = "", company: str = ""
) -> dict[str, object]:
    """Count distinct publishers in a Google News RSS response."""
    articles, domains = [], set()
    root = ElementTree.fromstring(xml)
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source_node = item.find("source")
        source = (source_node.text or "Unknown").strip() if source_node is not None else "Unknown"
        source_url = source_node.get("url", "") if source_node is not None else ""
        # Google News commonly appends " - Publisher" to the headline even
        # though the RSS item already has a separate <source> field.
        title = re.sub(rf"\s+[-–—]\s*{re.escape(source)}\s*$", "", title,
                       flags=re.IGNORECASE).strip()
        published_text = (item.findtext("pubDate") or "").strip()
        try:
            published = parsedate_to_datetime(published_text).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        if published < cutoff:
            continue
        domain = urllib.parse.urlparse(source_url).netloc.lower().removeprefix("www.")
        if symbol and company and not _is_relevant_talk(title, source, domain, symbol, company):
            continue
        source_key = domain or source.casefold()
        domains.add(source_key)
        articles.append({"source": source, "domain": domain, "title": title,
                         "link": link, "published": published.isoformat()})
    articles.sort(key=lambda item: item["published"], reverse=True)
    return {"score": len(domains), "articles": articles[:20], "hours": 24}


def _download_talk_score(symbol: str, company: str) -> dict[str, object]:
    company_name = _company_search_name(company)
    terms = f'("{company_name}" OR ("{symbol}" (stock OR shares OR Nasdaq OR NYSE))) when:1d'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": terms, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 stock-research-tool/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return _parse_talk_feed(response.read(), datetime.now(timezone.utc) - timedelta(hours=24),
                                symbol, company)


@dataclass(frozen=True)
class FourierMetrics:
    period: float
    quality: float
    phase: str
    next_turn: float
    active_since: str
    cycles_seen: float
    regime: str
    period_std: float
    candidates: list[dict[str, float]]
    frequencies_used: int
    next_turn_type: str
    chart: dict[str, object]


@dataclass(frozen=True)
class ZigZagMetrics:
    cycles: int
    average_up: float | None
    average_down: float | None
    average_days: float | None
    last_pivot: str | None
    last_pivot_date: str | None
    confirmation_date: str | None
    move_since_pivot: float | None
    score: float
    possible_entry: str
    chart: list[dict[str, object]]


@dataclass(frozen=True)
class Analytics:
    first_10m_pct: float | None
    next_50m_pct: float | None
    after_30m_pct: float | None
    averages: dict[str, tuple[float | None, int]]
    oscillations: dict[str, tuple[float | None, float | None, int]]
    charts: dict[str, list[dict[str, object]]]
    fourier: FourierMetrics | None
    fourier_models: list[FourierMetrics]
    zigzag: ZigZagMetrics | None


def _pct(frame, start: str, end: str) -> float | None:
    selected = frame.between_time(start, end, inclusive="left")
    if selected.empty or float(selected["Open"].iloc[0]) <= 0:
        return None
    return (float(selected["Close"].iloc[-1]) / float(selected["Open"].iloc[0]) - 1) * 100


def _zigzag_metrics(daily_frame, threshold: float = 7.0) -> ZigZagMetrics | None:
    """Measure alternating close-to-close swings of at least ``threshold`` percent."""
    import numpy as np
    close = daily_frame["Close"].dropna().iloc[-252:]
    if len(close) < 20:
        return None
    values, dates = close.to_numpy(dtype=float), list(close.index)
    direction, low_i, high_i = 0, 0, 0
    # Each pivot stores its extreme and the later session on which a 7% move
    # made that extreme knowable without look-ahead.
    pivots: list[tuple[int, str, float, int]] = []
    for index in range(1, len(values)):
        if values[index] < values[low_i]:
            low_i = index
        if values[index] > values[high_i]:
            high_i = index
        if direction == 0:
            if (values[index] / values[low_i] - 1) * 100 >= threshold:
                pivots.append((low_i, "trough", values[low_i], index))
                direction, high_i = 1, index
            elif (values[index] / values[high_i] - 1) * 100 <= -threshold:
                pivots.append((high_i, "peak", values[high_i], index))
                direction, low_i = -1, index
        elif direction == 1:
            if values[index] > values[high_i]:
                high_i = index
            if (values[index] / values[high_i] - 1) * 100 <= -threshold:
                pivots.append((high_i, "peak", values[high_i], index))
                direction, low_i = -1, index
        else:
            if values[index] < values[low_i]:
                low_i = index
            if (values[index] / values[low_i] - 1) * 100 >= threshold:
                pivots.append((low_i, "trough", values[low_i], index))
                direction, high_i = 1, index
    legs = []
    for left, right in zip(pivots, pivots[1:]):
        move = (right[2] / left[2] - 1) * 100
        legs.append((left[1], move, right[0] - left[0]))
    ups = [move for kind, move, _ in legs if kind == "trough" and move >= threshold]
    downs = [-move for kind, move, _ in legs if kind == "peak" and move <= -threshold]
    cycle_days = [pivots[i + 2][0] - pivots[i][0] for i in range(len(pivots) - 2)]
    cycles = min(len(ups), len(downs))
    avg_up = float(np.mean(ups)) if ups else None
    avg_down = float(np.mean(downs)) if downs else None
    avg_days = float(np.mean(cycle_days)) if cycle_days else None
    if cycles and avg_up and avg_down:
        cvs = [float(np.std(group) / max(np.mean(group), 1e-9))
               for group in (ups, downs, cycle_days) if len(group) > 1]
        consistency = math.exp(-sum(cvs) / max(len(cvs), 1))
        amplitude = 2 * avg_up * avg_down / (avg_up + avg_down)
        age = len(values) - 1 - pivots[-1][0]
        score = cycles * (amplitude / threshold) * consistency * math.exp(-age / 126)
    else:
        score = 0.0
    if pivots:
        pivot_i, pivot_type, pivot_price, confirmation_i = pivots[-1]
        move_since = (values[-1] / pivot_price - 1) * 100
        pivot_date = dates[pivot_i].date().isoformat()
        confirmation_date = dates[confirmation_i].date().isoformat()
        if direction == -1:
            rebound = (values[-1] / values[low_i] - 1) * 100
            entry = "watch possible trough" if 1 <= rebound < threshold else "falling / new low"
        elif pivot_type == "trough" and move_since >= threshold:
            entry = "confirmed rise"
        else:
            entry = "wait"
    else:
        pivot_type = pivot_date = confirmation_date = None
        move_since, entry = None, "insufficient cycles"
    base = values[0]
    chart = [{"date": stamp.date().isoformat(), "value": round(float(value / base * 100), 4),
              "pivot": bool(pivots and index == pivots[-1][0]),
              "confirmation": bool(pivots and index == pivots[-1][3])}
             for index, (stamp, value) in enumerate(zip(dates, values))]
    return ZigZagMetrics(cycles, round(avg_up, 2) if avg_up is not None else None,
        round(avg_down, 2) if avg_down is not None else None,
        round(avg_days, 1) if avg_days is not None else None, pivot_type, pivot_date,
        confirmation_date,
        round(float(move_since), 2) if move_since is not None else None,
        round(float(score), 3), entry, chart)


def _spectral_candidates(values, limit: int = 5) -> list[dict[str, float]]:
    import numpy as np
    values = np.asarray(values, dtype=float)
    if len(values) < 32:
        return []
    x = np.arange(len(values), dtype=float)
    residual = values - np.polyval(np.polyfit(x, values, 1), x)
    fft = np.fft.rfft(residual * np.hanning(len(residual)))
    frequencies = np.fft.rfftfreq(len(residual), d=1.0)
    band = (frequencies >= 1 / min(63.0, len(values) / 3)) & (frequencies <= 1 / 4)
    indexes = np.flatnonzero(band)
    if not len(indexes):
        return []
    power = np.abs(fft) ** 2
    band_power = power[band]
    total, floor, spread = float(band_power.sum()), float(np.median(band_power)), max(float(np.std(band_power)), 1e-12)
    local_peaks = [index for index in indexes
                   if power[index] >= power[index - 1] and power[index] >= power[index + 1]]
    regions = []
    for index in local_peaks:
        left = index
        while left > indexes[0] and power[left - 1] <= power[left]:
            left -= 1
        right = index
        while right < indexes[-1] and power[right + 1] <= power[right]:
            right += 1
        regions.append((index, float(power[left:right + 1].sum())))
    candidates = []
    for index, region_power in sorted(regions, key=lambda item: item[1], reverse=True):
        period = float(1 / frequencies[index])
        if any(abs(math.log(period / item["period"])) < math.log(1.12) for item in candidates):
            continue
        harmonic_of = 0.0
        for item in candidates:
            ratio = frequencies[index] / item["frequency"]
            nearest = round(ratio)
            if nearest >= 2 and abs(ratio - nearest) / nearest <= 0.06:
                harmonic_of = item["period"]
                break
        candidates.append({"frequency": float(frequencies[index]), "period": period,
            "power_share": float(region_power / max(total, 1e-12)),
            "z_score": float((power[index] - floor) / spread), "harmonic_of": harmonic_of})
        if len(candidates) == limit:
            break
    return candidates


def _design(x, frequencies):
    import numpy as np
    return np.column_stack([
        function(2 * np.pi * frequency * x)
        for frequency in frequencies for function in (np.sin, np.cos)
    ])


def _fit_frequency_model(values, exact_k: int | None = None) -> dict[str, object] | None:
    import numpy as np
    values = np.asarray(values, dtype=float)[-252:]
    if len(values) < 60:
        return None
    split = max(40, int(len(values) * 0.8))
    train_x = np.arange(split, dtype=float)
    full_x = np.arange(len(values), dtype=float)
    trend_coefficients = np.polyfit(train_x, values[:split], 1)
    detrended = values - np.polyval(trend_coefficients, full_x)
    candidates = _spectral_candidates(values[:split], 5)
    if not candidates:
        return None
    best = None
    for count in range(1, min(3, len(candidates)) + 1):
        if exact_k is not None and count != exact_k:
            continue
        selected = candidates[:count]
        frequencies = [item["frequency"] for item in selected]
        train_design = _design(train_x, frequencies)
        if not np.all(np.isfinite(train_design)) or np.linalg.cond(train_design) > 1e8:
            continue
        joint_train = np.column_stack((np.ones(split), train_x, train_design))
        joint_coefficients = np.linalg.lstsq(joint_train, values[:split], rcond=None)[0]
        coefficients = joint_coefficients[2:]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            train_residual = values[:split] - joint_train @ joint_coefficients
        if not np.all(np.isfinite(train_residual)):
            continue
        variance_y = max(float(np.var(detrended[:split])), 1e-12)
        residual_ratio = math.sqrt(float(np.var(train_residual)) / variance_y)
        reconstruction = 1 / (1 + residual_ratio)
        parameters = 2 + 2 * count
        # Penalise additional sine/cosine parameters without allowing one weak
        # component to collapse the complete geometric quality score to zero.
        adjusted = reconstruction * math.exp(-parameters / max(2 * split, 1))
        validation_design = _design(full_x[split:], frequencies)
        validation_joint = np.column_stack((np.ones(len(validation_design)),
                                            full_x[split:], validation_design))
        validation_residual = values[split:] - validation_joint @ joint_coefficients
        validation_variance = max(float(np.var(detrended[split:])), 1e-12)
        validation_ratio = math.sqrt(float(np.var(validation_residual)) / validation_variance)
        out_sample = 1 / (1 + validation_ratio)
        amplitudes = np.asarray([
            math.hypot(float(coefficients[2 * index]), float(coefficients[2 * index + 1]))
            for index in range(count)
        ])
        weights = amplitudes ** 2
        weights = weights / weights.sum() if weights.sum() else np.ones(count) / count
        spectral = float(sum(weight * min(1.0, max(0.0, item["z_score"] / 3))
                             for weight, item in zip(weights, selected)))
        period_samples = [[] for _ in range(count)]
        hop = max(2, int(round(selected[0]["period"] / 4)))
        for shift in (hop, 2 * hop, 3 * hop, 4 * hop, 5 * hop):
            if split - shift < 32:
                continue
            prior = _spectral_candidates(values[:split - shift], 5)
            for index, item in enumerate(selected):
                matches = [candidate for candidate in prior
                           if abs(candidate["period"] - item["period"]) / item["period"] <= 0.35]
                if matches:
                    period_samples[index].append(min(matches,
                        key=lambda candidate: abs(candidate["period"] - item["period"]))["period"])
        stabilities, deviations = [], []
        for item, samples in zip(selected, period_samples):
            deviation = statistics.pstdev([item["period"]] + samples) if samples else item["period"]
            deviations.append(deviation)
            stabilities.append(math.exp(-0.5 * (deviation / (0.15 * item["period"])) ** 2))
        stability = float(sum(weight * value for weight, value in zip(weights, stabilities)))
        quality = adjusted ** 0.30 * out_sample ** 0.40 * stability ** 0.20 * spectral ** 0.10
        model = {"quality": quality, "adjusted": adjusted, "out_sample": out_sample,
                 "stability": stability, "spectral": spectral, "selected": selected,
                 "period_std": float(sum(weight * value for weight, value in zip(weights, deviations)))}
        if best is None or exact_k is not None or quality >= float(best["quality"]) + 0.03:
            best = model
    if not best:
        return None
    frequencies = [item["frequency"] for item in best["selected"]]
    final_x = np.arange(len(values), dtype=float)
    final_design = _design(final_x, frequencies)
    final_joint = np.column_stack((np.ones(len(values)), final_x, final_design))
    final_solution = np.linalg.lstsq(final_joint, values, rcond=None)[0]
    final_trend = np.asarray([final_solution[1], final_solution[0]])
    final_coefficients = final_solution[2:]
    best.update({"values": values, "frequencies": frequencies,
                 "coefficients": final_coefficients, "window_length": len(values),
                 "trend_coefficients": final_trend})
    return best


def _fourier_metrics(daily_frame, exact_k: int | None = None) -> FourierMetrics | None:
    import numpy as np
    close = daily_frame["Close"].dropna()
    model = _fit_frequency_model(np.log(close.to_numpy(dtype=float)), exact_k=exact_k)
    if not model:
        return None
    frequencies, coefficients = model["frequencies"], model["coefficients"]
    period = 1 / frequencies[0]
    now = len(model["values"]) - 1
    grid = np.arange(now, now + 64.05, 0.05)
    derivative = np.zeros(len(grid))
    for index, frequency in enumerate(frequencies):
        omega = 2 * np.pi * frequency
        a, b = coefficients[2 * index], coefficients[2 * index + 1]
        derivative += a * omega * np.cos(omega * grid) - b * omega * np.sin(omega * grid)
    current_sign = 1 if derivative[0] >= 0 else -1
    turning_index = next((index for index in range(1, len(grid))
                          if (1 if derivative[index] >= 0 else -1) != current_sign), len(grid) - 1)
    next_turn = float(grid[turning_index] - now)
    next_type = "peak" if current_sign > 0 else "trough"
    phase = f"near {next_type}" if next_turn <= max(2, period * 0.15) else ("rising" if current_sign > 0 else "falling")
    quality = float(model["quality"])
    regime = "active" if quality >= 0.60 else "weakening" if quality >= 0.35 else "broken"
    selected_frequencies = set(model["frequencies"])
    candidates = [{"period": round(item["period"], 2), "power_share": round(item["power_share"], 4),
                   "z_score": round(item["z_score"], 2),
                   "harmonic_of": round(item.get("harmonic_of", 0), 2),
                   "selected": item["frequency"] in selected_frequencies}
                  for item in _spectral_candidates(model["values"][:max(40, int(len(model["values"]) * 0.8))], 5)]
    values = model["values"]
    confirmation, failures, active_since_index = max(2, math.ceil(period / 4)), 0, len(values) - 1
    for end in range(len(values), 59, -1):
        historical = _spectral_candidates(values[:end], 3)
        match = next((item for item in historical
                      if abs(item["period"] - period) / period <= 0.20), None)
        if match and min(1.0, max(0.0, match["z_score"] / 3)) >= 0.35:
            failures = 0
            active_since_index = end - 1
        else:
            failures += 1
            if failures >= confirmation:
                break
    close_segment = close.iloc[-len(values):]
    active_cycles = (len(values) - active_since_index) / period
    display_start = max(0, len(values) - 126)
    actual_residual = values - np.polyval(model["trend_coefficients"], np.arange(len(values)))
    future_grid = np.arange(now + 0.25, now + next_turn, 0.25)
    fit_grid = np.concatenate((np.arange(display_start, now + 1, dtype=float),
                               future_grid, np.asarray([now + next_turn])))
    fitted_wave = np.zeros(len(fit_grid))
    for index, frequency in enumerate(frequencies):
        omega = 2 * np.pi * frequency
        a, b = coefficients[2 * index], coefficients[2 * index + 1]
        fitted_wave += a * np.sin(omega * fit_grid) + b * np.cos(omega * fit_grid)
    chart = {
        "actual": [{"x": index - display_start, "value": round(float(value * 100), 5)}
                   for index, value in enumerate(actual_residual[display_start:], start=display_start)],
        "fitted": [{"x": round(float(x - display_start), 3),
                    "value": round(float(value * 100), 5), "forecast": bool(x > now)}
                   for x, value in zip(fit_grid, fitted_wave)],
        "today_x": now - display_start,
        "turn_x": round(float(now + next_turn - display_start), 3),
        "turn_type": next_type,
    }
    return FourierMetrics(round(period, 2), round(quality, 3), phase, round(next_turn, 1),
        close_segment.index[active_since_index].date().isoformat(), round(active_cycles, 1), regime,
        round(float(model["period_std"]), 2), candidates, len(frequencies), next_type, chart)


def _detrended_score(days: list[object]) -> float:
    """Return 0..1 residual movement around an OLS trend over three first hours."""
    path = [0.0]
    cumulative = 0.0
    for day in days:
        hour = day.between_time("09:30", "10:30", inclusive="left")
        prices = [float(value) for value in hour["Close"] if float(value) > 0]
        if not prices:
            return 0.0
        base = prices[0]
        for price in prices[1:]:
            path.append(cumulative + math.log(price / base))
        cumulative = path[-1]
    if len(path) < 12:
        return 0.0
    n = len(path)
    x_mean = (n - 1) / 2
    y_mean = sum(path) / n
    denominator = sum((x - x_mean) ** 2 for x in range(n))
    slope = sum((x - x_mean) * (y - y_mean) for x, y in enumerate(path)) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in enumerate(path)]
    residual_variation = sum(abs(b - a) for a, b in zip(residuals, residuals[1:]))
    raw_variation = sum(abs(b - a) for a, b in zip(path, path[1:]))
    if raw_variation <= 1e-12:
        return 0.0
    # Hysteresis ignores tiny zero crossings caused by quote noise.
    sigma = statistics.pstdev(residuals)
    band = sigma * 0.35
    states = [1 if value > band else -1 if value < -band else 0 for value in residuals]
    meaningful = [state for state in states if state]
    crossings = sum(a != b for a, b in zip(meaningful, meaningful[1:]))
    crossing_factor = min(1.0, crossings / 2) if meaningful else 0.0
    return round(min(1.0, residual_variation / raw_variation) * crossing_factor, 6)


def _oscillation_events(
    sessions: list[tuple[object, float, object]]
) -> list[tuple[object, float, str]]:
    moves = [move for _, move, _ in sessions]
    median_amplitude = max(0.01, statistics.median(abs(move) for move in moves))
    noise = max(0.25, median_amplitude * 0.5)
    directions = [1 if move > noise else -1 if move < -noise else 0 for move in moves]
    events: list[tuple[object, float]] = []
    run = 0
    for index, direction in enumerate(directions):
        if not direction:
            run = 0
            continue
        run = run + 1 if index and directions[index - 1] == direction else 1
        previous = next(
            ((j, directions[j]) for j in range(index - 1, max(-1, index - 3), -1)
             if directions[j] != 0), None
        )
        base_score = None
        amplitude_pair = None
        event_type = None
        if previous and previous[1] == -direction:
            base_score = 1.0
            event_type = "reversal"
            amplitude_pair = (abs(sessions[previous[0]][1]), abs(sessions[index][1]))
        elif run == 2:
            base_score = 0.5
            event_type = "continuation"
            amplitude_pair = (abs(sessions[index - 1][1]), abs(sessions[index][1]))
        elif run >= 3:
            base_score = _detrended_score(
                [sessions[index - 2][2], sessions[index - 1][2], sessions[index][2]]
            )
            event_type = "detrended"
            amplitude_pair = (abs(sessions[index - 1][1]), abs(sessions[index][1]))
        if base_score is not None and amplitude_pair:
            left, right = amplitude_pair
            harmonic = 0.0 if left <= 0 or right <= 0 else 2 * left * right / (left + right)
            normalized = min(3.0, harmonic / median_amplitude)
            final_score = round(base_score * normalized, 6)
            if final_score > 0:
                events.append((sessions[index][0], final_score, event_type))
    return events


def _download_analytics(symbols: list[str], on_batch=None) -> dict[str, Analytics]:
    import yfinance as yf
    results: dict[str, Analytics] = {}
    for batch in chunks(symbols, 25):
        data = yf.download(
            batch, period="1mo", interval="5m", group_by="ticker", auto_adjust=False,
            actions=False, prepost=False, threads=True, progress=False, timeout=30,
            multi_level_index=True,
        )
        daily_data = yf.download(
            batch, period="1y", interval="1d", group_by="ticker", auto_adjust=True,
            actions=False, threads=True, progress=False, timeout=30, multi_level_index=True,
        )
        batch_results: dict[str, Analytics] = {}
        if data.empty:
            continue
        for symbol in batch:
            try:
                frame = data[symbol][["Open", "Close"]].dropna()
            except (KeyError, TypeError):
                continue
            if frame.empty:
                continue
            frame.index = (frame.index.tz_localize("America/New_York") if frame.index.tz is None
                           else frame.index.tz_convert("America/New_York"))
            sessions = []
            for session_date, day in frame.groupby(frame.index.date):
                first_hour = day.between_time("09:30", "10:30", inclusive="left")
                times = {stamp.strftime("%H:%M") for stamp in first_hour.index}
                if "09:30" not in times or "10:25" not in times:
                    continue
                move = _pct(day, "09:30", "10:30")
                if move is not None:
                    sessions.append((session_date, move, day))
            if not sessions:
                continue
            latest_date, _, latest_day = sessions[-1]
            point_sessions = [
                (session_date, 1 if move > 7 else -1 if move < -7 else 0)
                for session_date, move, _ in sessions
            ]
            oscillation_events = _oscillation_events(sessions)
            averages = {}
            oscillations = {}
            charts = {}
            for label, days in (("d5", 5), ("d10", 10), ("w2", 14), ("m1", 30)):
                cutoff = latest_date - timedelta(days=days - 1)
                scores = [score for d, score in point_sessions if d >= cutoff]
                window_events = [(d, score, kind) for d, score, kind in oscillation_events if d >= cutoff]
                osc_scores = [score for _, score, _ in window_events]
                averages[label] = (sum(scores) / len(scores) if scores else None, len(scores))
                osc_mean = sum(osc_scores) / len(osc_scores) if osc_scores else None
                osc_rank = osc_mean * len(osc_scores) / (len(osc_scores) + 5) if osc_mean is not None else None
                oscillations[label] = (osc_mean, osc_rank, len(osc_scores))
                event_types = {d.isoformat(): kind for d, _, kind in window_events}
                chart_sessions = [(d, day) for d, _, day in sessions if d >= cutoff]
                closes = [float(day["Close"].iloc[-1]) for _, day in chart_sessions]
                base_close = closes[0] if closes else 0
                charts[label] = [
                    {"date": d.isoformat(), "value": close / base_close * 100,
                     "event": event_types.get(d.isoformat())}
                    for (d, _), close in zip(chart_sessions, closes) if base_close > 0
                ]
            daily_frame = daily_data[symbol] if not daily_data.empty and symbol in daily_data else None
            zigzag = _zigzag_metrics(daily_frame) if daily_frame is not None else None
            fourier_models = ([metric for metric in
                (_fourier_metrics(daily_frame, exact_k=k) for k in (1, 2, 3)) if metric]
                if daily_frame is not None else [])
            best_fourier = max(fourier_models, key=lambda metric: metric.quality, default=None)
            item = Analytics(
                _pct(latest_day, "09:30", "09:40"), _pct(latest_day, "09:40", "10:30"),
                _pct(latest_day, "10:00", "16:00"), averages, oscillations, charts,
                best_fourier, fourier_models, zigzag,
            )
            results[symbol] = item
            batch_results[symbol] = item
        if on_batch and batch_results:
            on_batch(batch_results)
    return results


HTML = r"""<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow,noarchive">
  <title>S&P 500 + QQQ — מחירים חיים</title>
  <style>
    :root{color-scheme:dark;--bg:#0b1117;--panel:#121a23;--line:#243140;--text:#e9eef4;--muted:#91a1b3;--green:#35d07f;--red:#ff6673;--blue:#4da3ff}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
    .wrap{max-width:1400px;margin:auto;padding:28px}.top{display:flex;gap:18px;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap}
    h1{font-size:25px;margin:0}.sub{color:var(--muted);font-size:14px;margin-top:7px}.controls{display:flex;gap:10px}
    input,button{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:10px;padding:11px 14px;font-size:15px}
    input{width:280px}button{cursor:pointer;background:#173659;border-color:#265a91}button:hover{background:#204a79}
    .status{padding:13px 16px;border:1px solid var(--line);background:var(--panel);border-radius:12px;margin-bottom:14px;color:var(--muted)}
    .fftfilters{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:0 0 14px;direction:ltr}.fftfilters label{font-size:11px;color:var(--muted)}
    .fftfilters select,.fftfilters input{width:auto;min-width:125px;padding:8px 10px;font-size:12px}.fftfilters input{max-width:115px}
    .tablebox{border:1px solid var(--line);border-radius:14px;overflow:auto;background:var(--panel);height:calc(100vh - 190px);min-height:420px}
    .tablebox>table{border-collapse:collapse;min-width:2750px;width:100%;direction:ltr}.tablebox>table>thead{position:sticky;top:0;background:#192430;z-index:2}
    .tablebox th,.tablebox td{padding:13px 16px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.tablebox th{font-size:13px;color:#b9c7d5;text-transform:uppercase;letter-spacing:.4px}
    .tablebox th.sortable{cursor:pointer;user-select:none}.tablebox th.sortable:hover{color:#fff;background:#223242}.tablebox th.sortable.active{color:var(--blue)}
    .tablebox th{cursor:help}.tablebox th.sortable{cursor:pointer}
    .tablebox tbody tr:hover{background:#17212c}.company{width:100%;white-space:normal}.symbol{font-weight:700;color:#d9e9fa}.price,.change{font-variant-numeric:tabular-nums;font-weight:650}
    .up{color:var(--green)}.down{color:var(--red)}.flat,.loading{color:var(--muted)}
    .osc-cell,.phase-cell{cursor:crosshair}.phase-cell{border-bottom:1px dotted #91a1b3}.osc-cell small{display:block;color:var(--muted);font-size:10px;margin-top:3px}
    .regime-active{color:var(--green)}.regime-weakening{color:#ffd43b}.regime-broken{color:var(--red)}
    #chartTip{display:none;position:fixed;z-index:20;width:360px;max-width:calc(100vw - 16px);padding:12px;background:#132335;border:1px solid #31506c;border-radius:12px;box-shadow:0 14px 40px #0009;pointer-events:none;overflow:hidden}
    #chartTip .tip-title{font-size:13px;color:#c8d8e8;margin-bottom:7px;direction:ltr}
    #chartTip.freq-tip,#chartTip.pivot-tip,#chartTip.talk-tip,#chartTip.fourier-tip{width:520px}.freq-grid{direction:ltr;font-size:11px}.freq-row{display:grid;grid-template-columns:100px minmax(170px,1fr) 70px 85px;align-items:center;gap:8px;padding:9px 4px;border-bottom:1px solid #294055}.freq-row.header{padding-top:4px;color:#91a1b3;font-weight:700;text-transform:uppercase}.freq-row.selected{color:#35d07f}.freq-period{white-space:normal;line-height:1.35}.pivot-cell,.talk-cell{cursor:crosshair;border-bottom:1px dotted #91a1b3}.talk-list{direction:ltr;max-height:330px;overflow:hidden}.talk-item{display:grid;grid-template-columns:105px minmax(0,1fr) 105px;align-items:start;gap:10px;padding:9px 2px;border-bottom:1px solid #294055;font-size:11px;line-height:1.4}.talk-item.header{padding-top:3px;color:#91a1b3;font-weight:700;text-transform:uppercase}.talk-source{color:#4da3ff;font-weight:700;overflow-wrap:anywhere}.talk-title{white-space:normal}.talk-time{color:#91a1b3;text-align:right;white-space:normal}
    @media(max-width:700px){.wrap{padding:14px}.controls,input{width:100%}.controls{flex:1}.index{display:none}.tablebox{height:calc(100vh - 230px)}.tablebox th,.tablebox td{padding:11px 10px}#chartTip.freq-tip,#chartTip.pivot-tip,#chartTip.talk-tip,#chartTip.fourier-tip{width:calc(100vw - 16px)}.freq-row{grid-template-columns:82px minmax(120px,1fr) 58px 72px;gap:5px}}
  </style>
</head>
<body><main class="wrap">
  <div class="top"><div><h1>מחירי S&P 500 + QQQ</h1><div class="sub">השינוי מחושב מול סגירת יום המסחר הקודם</div></div>
    <div class="controls"><input id="search" placeholder="חיפוש לפי סימול או חברה…"><button id="scanTalk">Scan all Talk</button><button id="refresh">חשב את כל המניות</button></div></div>
  <div id="status" class="status">מוכן לסריקה — לחץ על “חשב את כל המניות”</div>
  <div id="talkStatus" class="status" style="display:none"></div>
  <div id="talkStatus" class="status" style="display:none"></div>
  <div class="fftfilters"><label>Oscillation / Fourier filters</label>
    <input id="zigzagFilter" type="number" min="0" step="1" placeholder="Min 7% cycles">
    <select id="periodFilter"><option value="">All periods</option><option value="4-7">4–7d</option><option value="8-14">8–14d</option><option value="15-25">15–25d</option><option value="26-45">26–45d</option><option value="46-63">46–63d</option></select>
    <input id="qualityFilter" type="number" min="0" max="1" step="0.05" placeholder="Min quality">
    <select id="phaseFilter"><option value="">All phases</option><option>rising</option><option>near peak</option><option>falling</option><option>near trough</option></select>
    <input id="turnFilter" type="number" min="0" step="1" placeholder="Max days to turn">
    <input id="cyclesFilter" type="number" min="0" step="0.5" placeholder="Min cycles">
    <select id="regimeFilter"><option value="">All regimes</option><option>active</option><option>weakening</option><option>broken</option></select>
    <button id="clearFilters">Clear filters</button>
  </div>
  <div class="tablebox"><table><thead><tr><th>Symbol</th><th>Company</th><th>Price</th><th>Daily</th><th class="sortable" data-sort="zz_score">7% Score</th><th class="sortable" data-sort="zz_cycles">7% Cycles</th><th class="sortable" data-sort="zz_avg_up">Avg Up</th><th class="sortable" data-sort="zz_avg_down">Avg Down</th><th class="sortable" data-sort="zz_avg_days">Cycle Days</th><th>Last Pivot</th><th class="sortable" data-sort="zz_move">Since Pivot</th><th>Possible Entry</th><th>First 10m</th><th>Next 50m</th><th>From 10:00</th><th class="sortable" data-sort="avg_d5">Avg 5d</th><th class="sortable" data-sort="orank_d5">Osc 5d</th><th class="sortable" data-sort="avg_d10">Avg 10d</th><th class="sortable" data-sort="orank_d10">Osc 10d</th><th class="sortable" data-sort="avg_w2">Avg 2w</th><th class="sortable" data-sort="orank_w2">Osc 2w</th><th class="sortable" data-sort="avg_m1">Avg month</th><th class="sortable" data-sort="orank_m1">Osc month</th><th>Periods K1</th><th class="sortable" data-sort="fft_quality_1">Freq Q1</th><th class="sortable" data-sort="fft_phase_1">Phase 1</th><th class="sortable" data-sort="fft_next_turn_1">Next Turn 1</th><th>Periods K2</th><th class="sortable" data-sort="fft_quality_2">Freq Q2</th><th class="sortable" data-sort="fft_phase_2">Phase 2</th><th class="sortable" data-sort="fft_next_turn_2">Next Turn 2</th><th>Periods K3</th><th class="sortable" data-sort="fft_quality_3">Freq Q3</th><th class="sortable" data-sort="fft_phase_3">Phase 3</th><th class="sortable" data-sort="fft_next_turn_3">Next Turn 3</th><th class="sortable" data-sort="fft_active_since">Active since</th><th class="sortable" data-sort="fft_cycles">Active cycles</th><th class="sortable" data-sort="fft_regime">Regime</th><th class="index">Index</th></tr></thead><tbody id="rows"></tbody></table></div>
</main><div id="chartTip"></div><script>
const API_BASE=location.pathname.startsWith('/stocks-device')?'/stocks-device':'';
let stocks=[],sortKey=null,sortDirection='desc',talkCache={},talkScores={},talkLoading=new Set();
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const q=document.getElementById('search').value.trim().toLowerCase(),minZigzag=parseFloat(document.getElementById('zigzagFilter').value),period=document.getElementById('periodFilter').value,minQ=parseFloat(document.getElementById('qualityFilter').value),phase=document.getElementById('phaseFilter').value,maxTurn=parseFloat(document.getElementById('turnFilter').value),minCycles=parseFloat(document.getElementById('cyclesFilter').value),regime=document.getElementById('regimeFilter').value;
 let shown=stocks.filter(x=>{if(q&&!x.symbol.toLowerCase().includes(q)&&!x.company.toLowerCase().includes(q))return false;if(!Number.isNaN(minZigzag)&&(x.zz_cycles===null||x.zz_cycles<minZigzag))return false;if(period){const [lo,hi]=period.split('-').map(Number);if(x.fft_period===null||x.fft_period<lo||x.fft_period>hi)return false}if(!Number.isNaN(minQ)&&(x.fft_quality===null||x.fft_quality<minQ))return false;if(phase&&x.fft_phase!==phase)return false;if(!Number.isNaN(maxTurn)&&(x.fft_next_turn===null||x.fft_next_turn>maxTurn))return false;if(!Number.isNaN(minCycles)&&(x.fft_cycles===null||x.fft_cycles<minCycles))return false;if(regime&&x.fft_regime!==regime)return false;return true});
 if(sortKey){shown=[...shown].sort((a,b)=>{const av=a[sortKey],bv=b[sortKey];if(av===null&&bv===null)return a.symbol.localeCompare(b.symbol);if(av===null)return 1;if(bv===null)return -1;let diff=typeof av==='number'&&typeof bv==='number'?av-bv:String(av).localeCompare(String(bv));return sortDirection==='desc'?-diff:diff})}
 const metric=(v,limit)=>v===null?'—':`<span class="${v>limit?'up':v < -limit?'down':'flat'}">${v>=0?'+':''}${v.toFixed(2)}%</span>`;
 const avg=(v,n)=>v===null?'—':`<span class="${v>0?'up':v<0?'down':'flat'}">${v>=0?'+':''}${v.toFixed(3)} <small>(${n})</small></span>`;
 const osc=(mean,rank,n,key,symbol)=>rank===null?'—':`<span class="osc-cell up" data-symbol="${symbol}" data-range="${key}">${rank>=0?'+':''}${rank.toFixed(3)}<small>mean ${mean.toFixed(3)} · ${n} events</small></span>`;
 const zig=x=>`<td class="up">${x.zz_score===null?'—':x.zz_score.toFixed(3)}</td><td>${x.zz_cycles??'—'}</td><td class="up">${x.zz_avg_up===null?'—':'+'+x.zz_avg_up.toFixed(1)+'%'}</td><td class="down">${x.zz_avg_down===null?'—':'−'+x.zz_avg_down.toFixed(1)+'%'}</td><td>${x.zz_avg_days===null?'—':x.zz_avg_days.toFixed(1)+'d'}</td><td>${x.zz_last_pivot?'<span class="pivot-cell" data-symbol="'+esc(x.symbol)+'">'+esc(x.zz_last_pivot)+' · '+esc(x.zz_last_pivot_date)+'</span>':'—'}</td><td class="${x.zz_move>0?'up':x.zz_move<0?'down':'flat'}">${x.zz_move===null?'—':(x.zz_move>=0?'+':'')+x.zz_move.toFixed(1)+'%'}</td><td class="${x.zz_entry==='watch possible trough'?'up':'flat'}">${esc(x.zz_entry||'—')}</td>`;
 const fft=(x,k)=>{const q=x[`fft_quality_${k}`],phase=x[`fft_phase_${k}`];return `<td>${esc(x[`fft_periods_${k}`]||'—')}</td><td><span class="freq-cell ${q>=.6?'up':q>=.35?'flat':'down'}" data-symbol="${x.symbol}" data-k="${k}">${q===null?'—':q.toFixed(3)}</span></td><td>${phase?'<span class="phase-cell" data-symbol="'+esc(x.symbol)+'" data-k="'+k+'">'+esc(phase)+'</span>':'—'}</td><td>${x[`fft_next_turn_${k}`]===null?'—':esc(x[`fft_turn_type_${k}`])+' '+x[`fft_next_turn_${k}`].toFixed(1)+'d'}</td>`};
 document.getElementById('rows').innerHTML=shown.map(x=>{const ready=x.price!==null;const cls=!ready?'loading':x.change_pct>0?'up':x.change_pct<0?'down':'flat';return `<tr><td class="symbol">${esc(x.symbol)}</td><td class="company">${esc(x.company)}</td><td class="price ${cls}">${ready?'$'+x.price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}):'Loading…'}</td><td class="change ${cls}">${ready?(x.change_pct>=0?'+':'')+x.change_pct.toFixed(2)+'%':'—'}</td>${zig(x)}<td>${metric(x.first_10m_pct,8)}</td><td>${metric(x.next_50m_pct,8)}</td><td>${metric(x.after_30m_pct,5)}</td><td>${avg(x.avg_d5,x.n_d5)}</td><td>${osc(x.osc_d5,x.orank_d5,x.on_d5,'d5',x.symbol)}</td><td>${avg(x.avg_d10,x.n_d10)}</td><td>${osc(x.osc_d10,x.orank_d10,x.on_d10,'d10',x.symbol)}</td><td>${avg(x.avg_w2,x.n_w2)}</td><td>${osc(x.osc_w2,x.orank_w2,x.on_w2,'w2',x.symbol)}</td><td>${avg(x.avg_m1,x.n_m1)}</td><td>${osc(x.osc_m1,x.orank_m1,x.on_m1,'m1',x.symbol)}</td>${fft(x,1)}${fft(x,2)}${fft(x,3)}<td>${esc(x.fft_active_since||'—')}</td><td>${x.fft_cycles===null?'—':x.fft_cycles.toFixed(1)}</td><td class="regime-${esc(x.fft_regime||'')}">${esc(x.fft_regime||'—')}</td><td class="index">${esc(x.indexes)}</td></tr>`}).join('');}
async function jsonResponse(r){const type=r.headers.get('content-type')||'';if(!r.ok||!type.includes('application/json'))throw new Error(`השרת החזיר ${r.status}; נסה שוב בעוד רגע`);return r.json()}
async function load(){try{const d=await jsonResponse(await fetch(API_BASE+'/api/stocks',{cache:'no-store'}));stocks=d.stocks.map(x=>({...x,talk_score:talkCache[x.symbol]?.score??talkScores[x.symbol]??null}));document.getElementById('status').textContent=d.status;render()}catch(e){document.getElementById('status').textContent='שגיאת חיבור לשרת: '+e.message}}
async function pollFullScan(){const button=document.getElementById('refresh');try{const d=await jsonResponse(await fetch(API_BASE+'/api/status',{cache:'no-store'}));document.getElementById('status').textContent=d.status;button.disabled=d.running;button.textContent=d.running?`מחשב… ${d.completed}/${d.total}`:'חשב את כל המניות';if(d.running){setTimeout(pollFullScan,2000)}else{await load()}}catch(e){button.disabled=false;button.textContent='חשב את כל המניות';document.getElementById('status').textContent='שגיאת חיבור לשרת: '+e.message}}
document.getElementById('search').addEventListener('input',render);document.getElementById('refresh').addEventListener('click',async()=>{const button=document.getElementById('refresh');button.disabled=true;try{await jsonResponse(await fetch(API_BASE+'/api/refresh',{method:'POST'}));pollFullScan()}catch(e){button.disabled=false;document.getElementById('status').textContent='לא ניתן להתחיל סריקה: '+e.message}});
document.getElementById('scanTalk').addEventListener('click',async()=>{const button=document.getElementById('scanTalk');button.disabled=true;try{await jsonResponse(await fetch(API_BASE+'/api/talk-scan',{method:'POST'}));pollTalkScan()}catch(e){button.disabled=false}});
['zigzagFilter','periodFilter','qualityFilter','phaseFilter','turnFilter','cyclesFilter','regimeFilter'].forEach(id=>document.getElementById(id).addEventListener('input',render));
document.getElementById('clearFilters').addEventListener('click',()=>{['zigzagFilter','periodFilter','qualityFilter','phaseFilter','turnFilter','cyclesFilter','regimeFilter'].forEach(id=>document.getElementById(id).value='');render()});
document.querySelector('thead th:nth-child(4)').insertAdjacentHTML('afterend','<th class="sortable" data-sort="talk_score">Talk of the Day</th>');
document.querySelectorAll('th.sortable').forEach(th=>th.addEventListener('click',()=>{const key=th.dataset.sort;if(sortKey===key){sortDirection=sortDirection==='desc'?'asc':'desc'}else{sortKey=key;sortDirection='desc'}document.querySelectorAll('th.sortable').forEach(x=>{x.classList.toggle('active',x===th);x.textContent=x.textContent.replace(/\s[▲▼]$/,'')});th.textContent+=sortDirection==='desc'?' ▼':' ▲';render()}));
function talkCellContent(symbol){if(talkLoading.has(symbol))return '<button class="talk-check" disabled>בודק…</button>';const item=talkCache[symbol];if(item)return item.status==='ok'?esc(item.score):'<button class="talk-check">נסה שוב</button>';if(talkScores[symbol]!==undefined)return esc(talkScores[symbol]);return '<button class="talk-check">בדוק</button>'}
function updateTalkCell(symbol){document.querySelectorAll(`.talk-cell[data-symbol="${symbol}"]`).forEach(cell=>cell.innerHTML=talkCellContent(symbol))}
function decorateTalkRows(){document.querySelectorAll('#rows tr').forEach(row=>{if(row.querySelector('.talk-cell'))return;const symbol=row.cells[0]?.textContent.trim();if(!symbol)return;const cell=document.createElement('td');cell.className='talk-cell';cell.dataset.symbol=symbol;cell.innerHTML=talkCellContent(symbol);row.cells[3].after(cell)})}
async function requestTalk(symbol){if(talkLoading.has(symbol))return;const cached=talkCache[symbol];if(cached?.status==='ok')return;talkLoading.add(symbol);updateTalkCell(symbol);try{const data=await jsonResponse(await fetch(API_BASE+'/api/talk?symbol='+encodeURIComponent(symbol),{cache:'no-store'}));talkCache[symbol]=data;const stock=stocks.find(x=>x.symbol===symbol);if(stock)stock.talk_score=data.score;if(sortKey==='talk_score')render()}catch(e){talkCache[symbol]={score:null,articles:[],status:'error'}}finally{talkLoading.delete(symbol);updateTalkCell(symbol)}}
async function pollTalkScan(){try{const data=await jsonResponse(await fetch(API_BASE+'/api/talk-scan',{cache:'no-store'})),status=document.getElementById('talkStatus'),button=document.getElementById('scanTalk');talkScores={...talkScores,...(data.scores||{})};Object.entries(data.scores||{}).forEach(([symbol,score])=>{const stock=stocks.find(x=>x.symbol===symbol);if(stock)stock.talk_score=score;updateTalkCell(symbol)});status.style.display='block';status.textContent=data.running?`Talk scan: ${data.completed}/${data.total} · errors ${data.errors}`:`Talk scan completed: ${data.completed}/${data.total} · errors ${data.errors}`;button.disabled=data.running;button.textContent=data.running?`Scanning ${data.completed}/${data.total}`:'Scan all Talk';if(data.running)setTimeout(pollTalkScan,1500);else if(sortKey==='talk_score')render()}catch(e){document.getElementById('scanTalk').disabled=false}}
new MutationObserver(decorateTalkRows).observe(document.getElementById('rows'),{childList:true});
document.getElementById('rows').addEventListener('click',e=>{const button=e.target.closest('.talk-check');if(!button)return;const cell=button.closest('.talk-cell');if(cell)requestTalk(cell.dataset.symbol)});
const tip=document.getElementById('chartTip');
const headerHelp={
'Symbol':'סימול המסחר של המניה.','Company':'שם החברה.','Price':'המחיר האחרון שהתקבל מ-Yahoo Finance.','Daily':'השינוי מול סגירת יום המסחר הקודם.',
'Talk of the Day':'מספר אתרי החדשות הייחודיים שפרסמו ב-24 השעות האחרונות כתבה המזכירה את החברה או את הסימול. מספר כתבות מאותו אתר נספר פעם אחת.',
'7% Score':'דירוג מחזורי ZigZag: מספר המחזורים כפול גודל התנודה, עקביות ועדכניות.','7% Cycles':'מספר זוגות שהושלמו ובהם גם עלייה של 7% לפחות וגם ירידה של 7% לפחות במחירי סגירה.',
'Avg Up':'ממוצע הרגליים העולות המאושרות בין שפל לשיא.','Avg Down':'ממוצע גודל הרגליים היורדות המאושרות בין שיא לשפל.','Cycle Days':'מספר ימי המסחר הממוצע משפל לשפל או משיא לשיא.','Last Pivot':'נקודת השיא או השפל האחרונה שאושרה רק לאחר תנועה של 7% בכיוון ההפוך.','Since Pivot':'שינוי מחיר הסגירה מאז ה-pivot המאושר האחרון.','Possible Entry':'watch possible trough מציין ירידה פעילה וריבאונד של 1%–7% מהשפל הזמני. השפל עדיין לא מאושר ולכן זה אות מחקרי בסיכון גבוה, לא המלצת מסחר.',
'First 10m':'השינוי מפתיחת 09:30 עד סוף 09:39. מעל ±8% מודגש.','Next 50m':'השינוי מ-09:40 עד סוף 10:29. מעל ±8% מודגש.','From 10:00':'השינוי מ-10:00 עד הנר האחרון. עלייה מעל 5% מודגשת.',
'Avg 5d':'ממוצע ניקוד השעה הראשונה בחלון קלנדרי של 5 ימים.','Avg 10d':'ממוצע ניקוד השעה הראשונה ב-10 ימים.','Avg 2w':'ממוצע ניקוד השעה הראשונה בשבועיים.','Avg month':'ממוצע ניקוד השעה הראשונה בחודש.',
'Osc 5d':'דירוג אוסילציה ל-5 ימים, מתוקן לפי מספר האירועים.','Osc 10d':'דירוג אוסילציה ל-10 ימים.','Osc 2w':'דירוג אוסילציה לשבועיים.','Osc month':'דירוג אוסילציה לחודש.',
'Periods K1':'המחזור שנבחר במודל בעל תדר מרכזי אחד.','Periods K2':'שני המחזורים שנבחרו במודל בעל שני תדרים.','Periods K3':'שלושת המחזורים שנבחרו במודל בעל שלושה תדרים.',
'Freq Q1':'איכות מודל של תדר אחד: התאמה לעבר, בדיקה מחוץ למדגם, יציבות ועוצמה ספקטרלית.','Freq Q2':'איכות מודל של שני תדרים לפי אותם מבחנים. עברו על הערך כדי לראות את התדרים שנבחרו.','Freq Q3':'איכות מודל של שלושה תדרים לפי אותם מבחנים. תדרים הרמוניים מסומנים בחלונית.',
'Phase 1':'מצב הגל במודל K1: עולה, סמוך לשיא, יורד או סמוך לשפל.','Phase 2':'מצב הסכום של שני הגלים במודל K2.','Phase 3':'מצב הסכום של שלושת הגלים במודל K3.',
'Next Turn 1':'מספר ימי המסחר עד נקודת המפנה הבאה שחוזה מודל K1.','Next Turn 2':'מספר ימי המסחר עד נקודת המפנה הבאה שחוזה מודל K2.','Next Turn 3':'מספר ימי המסחר עד נקודת המפנה הבאה שחוזה מודל K3.','Active since':'התאריך שממנו המשטר הנוכחי נשמר ללא שבירה מאושרת.',
'Active cycles':'מספר המחזורים המשוער שעבר מאז Active Since; זה אינו מספר מחזורים שנצפו בפועל.','Regime':'active מעל 0.60, weakening בין 0.35 ל-0.60, broken מתחת 0.35.','Index':'המדד שבו המניה נכללת.'};
function showHelp(text,e){tip.className='';tip.innerHTML=`<div style="font-size:13px;line-height:1.55;direction:rtl">${esc(text)}</div>`;tip.style.display='block';moveTip(e)}
function showFrequency(target,e){const stock=stocks.find(x=>x.symbol===target.dataset.symbol),k=target.dataset.k,candidates=stock?.[`fft_candidates_${k}`];if(!candidates?.length)return;const rows=candidates.map(c=>`<div class="freq-row ${c.selected?'selected':''}"><span>${c.selected?'✓ selected':'candidate'}</span><span class="freq-period">${c.period.toFixed(2)}d${c.harmonic_of?' · harmonic of '+c.harmonic_of.toFixed(2)+'d':''}</span><span>${(c.power_share*100).toFixed(1)}%</span><span>${c.z_score.toFixed(2)}σ</span></div>`).join('');const std=stock[`fft_period_std_${k}`];tip.className='freq-tip';tip.innerHTML=`<div class="tip-title">${esc(stock.symbol)} · K${k} model</div><div style="font-size:11px;color:#91a1b3;margin-bottom:7px;direction:ltr">Weighted period std: ${std?.toFixed(2)??'—'}d</div><div class="freq-grid"><div class="freq-row header"><span>status</span><span>period</span><span>power</span><span>noise</span></div>${rows}</div>`;tip.style.display='block';moveTip(e)}
function showFourierChart(target,e){const stock=stocks.find(x=>x.symbol===target.dataset.symbol),k=target.dataset.k,chart=stock?.[`fft_chart_${k}`];if(!chart?.actual?.length||!chart?.fitted?.length)return;const w=494,h=210,p=14,all=[...chart.actual,...chart.fitted],lo=Math.min(...all.map(d=>d.value)),hi=Math.max(...all.map(d=>d.value)),span=hi-lo||1,maxX=Math.max(...all.map(d=>d.x)),xScale=(w-2*p)/Math.max(maxX,1),xy=d=>({...d,x:p+d.x*xScale,y:p+(hi-d.value)*(h-2*p)/span}),actual=chart.actual.map(xy),fitted=chart.fitted.map(xy),past=fitted.filter(d=>!d.forecast),future=[past[past.length-1],...fitted.filter(d=>d.forecast)].filter(Boolean),line=a=>a.map(d=>`${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(' '),todayX=p+chart.today_x*xScale,turn=fitted[fitted.length-1],periods=(stock[`fft_candidates_${k}`]||[]).filter(c=>c.selected).slice(0,Number(k)),colors=['#4da3ff','#b37cff','#ffd43b'],svgH=h+periods.length*27+12,markers=periods.map((_,i)=>`<marker id="periodArrow${i}" markerWidth="7" markerHeight="7" refX="3.5" refY="3.5" orient="auto-start-reverse"><path d="M7,0 L0,3.5 L7,7" fill="none" stroke="${colors[i]}" stroke-width="1.5"/></marker>`).join(''),rulers=periods.map((item,i)=>{const length=Math.min(w-70,item.period*xScale),x1=(w-length)/2,x2=x1+length,y=h+22+i*27,label=`${item.period.toFixed(1)} trading days${item.harmonic_of?' · H of '+item.harmonic_of.toFixed(1)+'d':''}`;return `<text x="${w/2}" y="${y-7}" fill="${colors[i]}" font-size="10.5" text-anchor="middle">${esc(label)}</text><line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}" stroke="${colors[i]}" stroke-width="1.8" marker-start="url(#periodArrow${i})" marker-end="url(#periodArrow${i})"/>`}).join('');tip.className='fourier-tip';tip.innerHTML=`<div class="tip-title">${esc(stock.symbol)} · K${k} inverse Fourier · ${esc(chart.turn_type)} in ${stock[`fft_next_turn_${k}`].toFixed(1)}d</div><svg width="100%" height="${svgH}" viewBox="0 0 494 ${svgH}"><defs>${markers}</defs><line x1="${todayX}" y1="8" x2="${todayX}" y2="202" stroke="#91a1b3" stroke-width="1" stroke-dasharray="3 4"/><polyline points="${line(actual)}" fill="none" stroke="#35d07f" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/><polyline points="${line(past)}" fill="none" stroke="#ff9f43" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/><polyline points="${line(future)}" fill="none" stroke="#ff9f43" stroke-width="3" stroke-dasharray="6 5" stroke-linejoin="round" stroke-linecap="round"/><circle cx="${turn.x}" cy="${turn.y}" r="6" fill="#ff9f43" stroke="#ffe0bd" stroke-width="2"/>${rulers}</svg><div style="display:flex;gap:14px;flex-wrap:wrap;font-size:11px;direction:ltr"><span style="color:#35d07f">● detrended closes</span><span style="color:#ff9f43">● K${k} reconstruction</span><span style="color:#ffb873">- - forecast to ${esc(chart.turn_type)}</span></div>`;tip.style.display='block';moveTip(e)}
function showTalk(target,e){const symbol=target.dataset.symbol,item=talkCache[symbol];if(!item||item.status!=='ok')return;const formatTime=value=>new Date(value).toLocaleString('he-IL',{timeZone:'Asia/Jerusalem',day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});const articles=item.articles.slice(0,8).map(a=>`<div class="talk-item"><span class="talk-source">${esc(a.source)}</span><span class="talk-title">${esc(a.title)}</span><span class="talk-time">${esc(formatTime(a.published))}</span></div>`).join('');tip.className='talk-tip';tip.innerHTML=`<div class="tip-title">${esc(symbol)} · ${item.score} unique sites / last 24h</div><div class="talk-list"><div class="talk-item header"><span>site</span><span>headline</span><span>Israel time</span></div>${articles||'<div class="talk-item"><span></span><span>No matching coverage found.</span><span></span></div>'}</div>`;tip.style.display='block';moveTip(e)}
function showPivotChart(target,e){const stock=stocks.find(x=>x.symbol===target.dataset.symbol),points=stock?.zz_chart||[];if(!points.length)return;const w=494,h=190,p=14,values=points.map(x=>x.value),lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;const xy=points.map((d,i)=>({x:p+i*(w-2*p)/Math.max(1,points.length-1),y:p+(hi-d.value)*(h-2*p)/span,...d}));const line=xy.map(d=>`${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(' '),pivot=xy.find(d=>d.pivot),confirmation=xy.find(d=>d.confirmation);const marks=(pivot?`<circle cx="${pivot.x}" cy="${pivot.y}" r="6" fill="#ff9f43" stroke="#ffe0bd" stroke-width="2"/>`:'')+(confirmation?`<circle cx="${confirmation.x}" cy="${confirmation.y}" r="6" fill="#ffd43b" stroke="#fff2a8" stroke-width="2"/>`:'');tip.className='pivot-tip';tip.innerHTML=`<div class="tip-title">${esc(stock.symbol)} · last ${esc(stock.zz_last_pivot||'pivot')}</div><svg width="100%" height="190" viewBox="0 0 494 190"><polyline points="${line}" fill="none" stroke="#35d07f" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>${marks}</svg><div style="display:flex;gap:14px;flex-wrap:wrap;font-size:11px;direction:ltr"><span style="color:#ff9f43">● extreme: ${esc(stock.zz_last_pivot_date||'—')}</span><span style="color:#ffd43b">● confirmed: ${esc(stock.zz_confirmation_date||'—')}</span></div>`;tip.style.display='block';moveTip(e)}
function showChart(target,e){const stock=stocks.find(x=>x.symbol===target.dataset.symbol),points=stock?.charts?.[target.dataset.range]||[];if(!points.length)return;tip.className='';
 const w=334,h=150,p=10,values=points.map(x=>x.value),lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;
 const xy=points.map((d,i)=>({x:p+i*(w-2*p)/Math.max(1,points.length-1),y:p+(hi-d.value)*(h-2*p)/span,...d}));
 const colors={detrended:'#ff5263',reversal:'#ff9f43',continuation:'#ffd43b'},strokes={detrended:'#ffd0d4',reversal:'#ffe0bd',continuation:'#fff2a8'};
 const line=xy.map(d=>`${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(' '),dots=xy.filter(d=>d.event).map(d=>`<circle cx="${d.x}" cy="${d.y}" r="4.7" fill="${colors[d.event]}" stroke="${strokes[d.event]}" stroke-width="1.5"/>`).join('');
 tip.innerHTML=`<div class="tip-title">${esc(stock.symbol)} · ${esc(target.dataset.range.toUpperCase())}</div><svg width="334" height="150" viewBox="0 0 334 150"><polyline points="${line}" fill="none" stroke="#35d07f" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>${dots}</svg><div style="display:flex;gap:12px;font-size:11px;color:#b9c7d5;direction:ltr"><span style="color:#ff5263">● detrended</span><span style="color:#ff9f43">● reversal</span><span style="color:#ffd43b">● continuation</span></div>`;tip.style.display='block';moveTip(e)}
function moveTip(e){const gap=16,w=tip.offsetWidth||360,h=tip.offsetHeight||190;let left=e.clientX+gap,top=e.clientY+gap;if(left+w>window.innerWidth-8)left=e.clientX-w-gap;if(top+h>window.innerHeight-8)top=e.clientY-h-gap;tip.style.left=Math.max(8,left)+'px';tip.style.top=Math.max(8,top)+'px'}
document.getElementById('rows').addEventListener('mouseover',e=>{const osc=e.target.closest('.osc-cell'),freq=e.target.closest('.freq-cell'),phase=e.target.closest('.phase-cell'),pivot=e.target.closest('.pivot-cell'),talk=e.target.closest('.talk-cell');if(osc)showChart(osc,e);else if(freq)showFrequency(freq,e);else if(phase)showFourierChart(phase,e);else if(pivot)showPivotChart(pivot,e);else if(talk)showTalk(talk,e)});document.getElementById('rows').addEventListener('mousemove',e=>{if(tip.style.display==='block')moveTip(e)});document.getElementById('rows').addEventListener('mouseout',e=>{if(e.target.closest('.osc-cell,.freq-cell,.phase-cell,.pivot-cell,.talk-cell'))tip.style.display='none'});
document.querySelector('thead').addEventListener('mouseover',e=>{const th=e.target.closest('th');if(!th)return;const key=th.textContent.replace(/\s[▲▼]$/,'').trim(),help=headerHelp[key];if(help)showHelp(help,e)});document.querySelector('thead').addEventListener('mousemove',e=>{if(tip.style.display==='block')moveTip(e)});document.querySelector('thead').addEventListener('mouseout',()=>tip.style.display='none');
load();
</script></body></html>"""


class TalkState:
    """Lazy, rate-limited cache of distinct news publishers per stock."""
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names
        self.lock = threading.Lock()
        self.gate = threading.Semaphore(2)
        self.scanning = False
        self.completed = 0
        self.total = 0
        self.errors = 0
        self.scan_running = False
        self.scan_done = 0
        self.scan_total = 0
        try:
            self.cache = json.loads(TALK_CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self.cache = {}

    def get(self, symbol: str) -> dict[str, object]:
        now = time.time()
        with self.lock:
            cached = self.cache.get(symbol)
            if cached and now - float(cached.get("fetched_at", 0)) < 1800:
                return cached["result"]
        try:
            with self.gate:
                result = _download_talk_score(symbol, self.names.get(symbol, symbol))
            result["status"] = "ok"
        except Exception as exc:
            result = {"score": None, "articles": [], "hours": 24,
                      "status": "error", "error": type(exc).__name__}
        with self.lock:
            self.cache[symbol] = {"fetched_at": now, "result": result}
            try:
                TALK_CACHE_FILE.write_text(
                    json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass
        return result

    def start_scan(self, symbols: list[str]) -> None:
        with self.lock:
            if self.scanning:
                return
            self.scanning, self.completed, self.total, self.errors = True, 0, len(symbols), 0
        threading.Thread(target=self._scan_all, args=(list(symbols),), daemon=True).start()

    def _scan_all(self, symbols: list[str]) -> None:
        def fetch(symbol: str) -> dict[str, object]:
            return self.get(symbol)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(fetch, symbol): symbol for symbol in symbols}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        failed = result.get("status") != "ok"
                    except Exception:
                        failed = True
                    with self.lock:
                        self.completed += 1
                        self.errors += int(failed)
        finally:
            with self.lock:
                self.scanning = False

    def status(self) -> dict[str, object]:
        with self.lock:
            now = time.time()
            scores = {symbol: item["result"].get("score") for symbol, item in self.cache.items()
                      if now - float(item.get("fetched_at", 0)) < 1800
                      and item.get("result", {}).get("status") == "ok"}
            return {"running": self.scanning, "completed": self.completed,
                    "total": self.total, "errors": self.errors, "scores": scores}

class StockState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.symbols, self.memberships, self.names = load_universe_details()
        self.talk = TalkState(self.names)
        self.quotes: dict[str, Quote] = {}
        self.analytics: dict[str, Analytics] = {}
        self.loading = False
        self.updated = ""
        self.phase = "idle"

    def refresh(self) -> None:
        with self.lock:
            if self.loading:
                return
            self.loading = True
            self.phase = "quotes"
            # A new manual scan replaces the previous snapshot. Dropping the
            # old one first prevents two full result sets living in RAM.
            self.quotes = {}
            self.analytics = {}
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            def update(batch: dict[str, Quote]) -> None:
                with self.lock:
                    self.quotes.update(batch)
            result = _download_quotes(self.symbols, update)
            with self.lock:
                self.quotes = result
                self.updated = datetime.now().strftime("%H:%M:%S")
                self.phase = "analytics"
            missing_analytics = [
                symbol for symbol in self.symbols
                if symbol not in self.analytics or self.analytics[symbol].fourier is None
            ]
            if missing_analytics:
                def update_analytics(batch: dict[str, Analytics]) -> None:
                    with self.lock:
                        self.analytics.update(batch)
                analytics = _download_analytics(missing_analytics, update_analytics)
                with self.lock:
                    self.analytics.update(analytics)
        finally:
            with self.lock:
                self.loading = False
                self.phase = "done"

    def status_payload(self) -> dict[str, object]:
        with self.lock:
            if self.phase == "idle":
                status = "מוכן לסריקה — לחץ על ‘חשב את כל המניות’"
                completed = 0
            elif self.phase == "quotes":
                completed = len(self.quotes)
                status = f"מוריד מחירים… {completed} מתוך {len(self.symbols)}"
            elif self.phase == "analytics":
                completed = len(self.analytics)
                status = f"מחשב סטטיסטיקה היסטורית… {completed} מתוך {len(self.symbols)}"
            else:
                completed = len(self.analytics)
                status = f"הסריקה הסתיימה: {self.updated} | {len(self.quotes)} מניות"
            return {"running": self.loading, "phase": self.phase, "completed": completed,
                    "total": len(self.symbols), "status": status}

    def payload(self) -> dict[str, object]:
        with self.lock:
            count = len(self.quotes)
            if self.phase == "idle":
                status = "מוכן לסריקה — לחץ על ‘חשב את כל המניות’"
            elif self.loading:
                if count < len(self.symbols):
                    status = f"מוריד מחירים… {count} מתוך {len(self.symbols)}"
                else:
                    status = f"מחשב סטטיסטיקה היסטורית… {len(self.analytics)} מתוך {len(self.symbols)}"
            else:
                status = f"הסריקה הסתיימה: {self.updated} | {count} מניות"
            rows = []
            for symbol in self.symbols:
                quote = self.quotes.get(symbol)
                analytics = self.analytics.get(symbol)
                row = {
                    "symbol": symbol, "company": self.names.get(symbol, symbol),
                    "indexes": self.memberships.get(symbol, ""),
                    "price": quote.price if quote else None,
                    "change_pct": quote.change_pct if quote else None,
                    "first_10m_pct": analytics.first_10m_pct if analytics else None,
                    "next_50m_pct": analytics.next_50m_pct if analytics else None,
                    "after_30m_pct": analytics.after_30m_pct if analytics else None,
                }
                for label in ("d5", "d10", "w2", "m1"):
                    value, n = analytics.averages.get(label, (None, 0)) if analytics else (None, 0)
                    osc_mean, osc_rank, osc_n = (
                        analytics.oscillations.get(label, (None, None, 0))
                        if analytics else (None, None, 0)
                    )
                    row[f"avg_{label}"] = value
                    row[f"n_{label}"] = n
                    row[f"osc_{label}"] = osc_mean
                    row[f"orank_{label}"] = osc_rank
                    row[f"on_{label}"] = osc_n
                row["charts"] = analytics.charts if analytics else {}
                zigzag = analytics.zigzag if analytics else None
                row.update({
                    "zz_score": zigzag.score if zigzag else None,
                    "zz_cycles": zigzag.cycles if zigzag else None,
                    "zz_avg_up": zigzag.average_up if zigzag else None,
                    "zz_avg_down": zigzag.average_down if zigzag else None,
                    "zz_avg_days": zigzag.average_days if zigzag else None,
                    "zz_last_pivot": zigzag.last_pivot if zigzag else None,
                    "zz_last_pivot_date": zigzag.last_pivot_date if zigzag else None,
                    "zz_confirmation_date": zigzag.confirmation_date if zigzag else None,
                    "zz_move": zigzag.move_since_pivot if zigzag else None,
                    "zz_entry": zigzag.possible_entry if zigzag else None,
                    "zz_chart": zigzag.chart if zigzag else [],
                })
                fourier = analytics.fourier if analytics else None
                models_by_k = ({model.frequencies_used: model for model in analytics.fourier_models}
                               if analytics else {})
                row.update({
                    "fft_period": fourier.period if fourier else None,
                    "fft_quality": fourier.quality if fourier else None,
                    "fft_phase": fourier.phase if fourier else None,
                    "fft_next_turn": fourier.next_turn if fourier else None,
                    "fft_active_since": fourier.active_since if fourier else None,
                    "fft_cycles": fourier.cycles_seen if fourier else None,
                    "fft_regime": fourier.regime if fourier else None,
                    "fft_period_std": fourier.period_std if fourier else None,
                    "fft_candidates": fourier.candidates if fourier else [],
                    "fft_k": fourier.frequencies_used if fourier else None,
                    "fft_turn_type": fourier.next_turn_type if fourier else None,
                })
                for k in (1, 2, 3):
                    model = models_by_k.get(k)
                    row[f"fft_periods_{k}"] = (
                        " + ".join(f"{item['period']:.1f}d" +
                            (f" (H of {item['harmonic_of']:.1f}d)" if item.get("harmonic_of") else "")
                            for item in model.candidates if item.get("selected")) if model else None
                    )
                    row[f"fft_quality_{k}"] = model.quality if model else None
                    row[f"fft_phase_{k}"] = model.phase if model else None
                    row[f"fft_next_turn_{k}"] = model.next_turn if model else None
                    row[f"fft_turn_type_{k}"] = model.next_turn_type if model else None
                    row[f"fft_candidates_{k}"] = model.candidates if model else []
                    row[f"fft_period_std_{k}"] = model.period_std if model else None
                    row[f"fft_chart_{k}"] = model.chart if model else None
                rows.append(row)
            return {"status": status, "stocks": rows}


def run_web_gui() -> None:
    state = StockState()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/stocks":
                body = json.dumps(state.payload(), ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            elif self.path == "/api/status":
                body = json.dumps(state.status_payload(), ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            elif self.path.startswith("/api/talk?"):
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                symbol = query.get("symbol", [""])[0].upper()
                if symbol not in state.symbols:
                    self.send_error(404); return
                body = json.dumps(state.talk.get(symbol), ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            elif self.path == "/api/talk-scan":
                body = json.dumps(state.talk.status(), ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            elif self.path == "/health":
                self._send(b'{"status":"ok"}', "application/json; charset=utf-8")
            else:
                self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            if self.path == "/api/refresh":
                state.refresh(); self._send(b"{}", "application/json")
            elif self.path == "/api/talk-scan":
                state.talk.start_scan(state.symbols)
                body = json.dumps(state.talk.status(), ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            return

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "0"))
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    if host == "127.0.0.1" and not os.environ.get("STOCKS_NO_BROWSER"):
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()
    print(f"Live stock table: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
