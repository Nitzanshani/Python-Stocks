"""Experimental rolling validation and historical-news study for 20 stocks.

This is deliberately separate from the production Freq Q implementation.  It
does not alter dashboard scores.  Each current K3 frequency is evaluated with
an adaptive walk-forward window, then linked to news published between the
previous trading session and the next session's open.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, time as clock_time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

from market_scanner import load_universe_details
from web_gui import _fit_frequency_model, _is_relevant_talk


SYMBOLS = [
    "AAOI", "COHR", "BA", "AJG", "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "MU",
    "TSLA", "META", "AMZN", "GOOGL", "NFLX", "PLTR", "QCOM", "ANET", "GE", "CCL",
]
OUTPUT_JSON = Path(__file__).with_name("hitting_wave_experiment.json")
OUTPUT_CSV = Path(__file__).with_name("hitting_wave_experiment.csv")


@dataclass
class RollingPoint:
    validation_start: str
    known_on: str
    q: float
    nrmse: float
    residual_sigma: float
    spectral_strength: float
    period_match: float | None
    good: bool


@dataclass
class WaveResult:
    symbol: str
    company: str
    frequency_number: int
    period: float
    train_days: int
    validation_days: int
    status: str
    estimated_start: str | None
    confirmed_on: str | None
    prior_session: str | None
    latest_q: float | None
    latest_nrmse: float | None
    latest_residual_sigma: float | None
    residual_trend: str
    observations: int
    news_sites: int | None
    news_articles: list[dict[str, str]]


def _adaptive_lengths(period: float) -> tuple[int, int]:
    train = max(14, math.ceil(3 * period))
    validation = max(4, math.ceil(0.20 * train))
    return train, validation


def _fixed_frequency_spectral_strength(values, period: float) -> tuple[float, float | None]:
    """Measure the target frequency directly instead of re-selecting today's peak."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    x = np.arange(len(values), dtype=float)
    residual = values - np.polyval(np.polyfit(x, values, 1), x)
    power = np.abs(np.fft.rfft(residual * np.hanning(len(residual)))) ** 2
    frequencies = np.fft.rfftfreq(len(residual), d=1.0)
    target = 1.0 / period
    index = int(np.argmin(np.abs(frequencies - target)))
    if index <= 0 or index >= len(power) - 1:
        return 0.0, None
    band = (frequencies >= 1 / min(63.0, max(4.0, len(values) / 2))) & (frequencies <= 1 / 4)
    band_power = power[band]
    if not len(band_power):
        return 0.0, None
    floor = float(np.median(band_power))
    spread = max(float(np.std(band_power)), 1e-12)
    is_peak = power[index] >= power[index - 1] and power[index] >= power[index + 1]
    matched_period = float(1 / frequencies[index]) if frequencies[index] else None
    if not is_peak or matched_period is None or abs(matched_period - period) / period > 0.20:
        return 0.0, matched_period
    return min(1.0, max(0.0, float((power[index] - floor) / spread) / 3)), matched_period


def _rolling_point(values, dates, period: float, end: int) -> RollingPoint | None:
    import numpy as np

    train_length, validation_length = _adaptive_lengths(period)
    split = end - validation_length
    start = split - train_length
    if start < 0:
        return None
    train = np.asarray(values[start:split], dtype=float)
    validation = np.asarray(values[split:end], dtype=float)
    train_x = np.arange(train_length, dtype=float)
    validation_x = np.arange(train_length, train_length + validation_length, dtype=float)
    frequency = 1.0 / period
    omega = 2 * np.pi * frequency
    design = np.column_stack((
        np.ones(train_length), train_x,
        np.sin(omega * train_x), np.cos(omega * train_x),
    ))
    coefficients = np.linalg.lstsq(design, train, rcond=None)[0]
    validation_design = np.column_stack((
        np.ones(validation_length), validation_x,
        np.sin(omega * validation_x), np.cos(omega * validation_x),
    ))
    residual = validation - validation_design @ coefficients
    trend = coefficients[0] + coefficients[1] * validation_x
    validation_detrended = validation - trend
    train_detrended = train - (coefficients[0] + coefficients[1] * train_x)
    scale = max(float(np.std(validation_detrended)), 0.25 * float(np.std(train_detrended)), 1e-6)
    rmse = math.sqrt(float(np.mean(residual ** 2)))
    nrmse = rmse / scale
    oos_q = 1 / (1 + nrmse)

    spectral, matched_period = _fixed_frequency_spectral_strength(train, period)
    # The rolling quality remains distinct from production Freq Q. Spectral
    # presence prevents a low-error straight trend from masquerading as a wave.
    combined_q = oos_q ** 0.80 * max(spectral, 1e-6) ** 0.20
    good = bool(matched_period is not None and spectral > 0 and combined_q >= 0.55)
    return RollingPoint(
        dates[split].date().isoformat(), dates[end - 1].date().isoformat(),
        round(combined_q, 4), round(nrmse, 4), round(float(np.std(residual)), 6),
        round(spectral, 4), round(matched_period, 2) if matched_period else None, good,
    )


def _find_current_regime(points: list[RollingPoint], period: float):
    if not points:
        return "insufficient", None, None
    confirmation = max(2, math.ceil(period / 4))
    good_indexes = [index for index, point in enumerate(points) if point.good]
    if not good_indexes:
        return "broken", None, None
    anchor = good_indexes[-1]
    if points[-1].good:
        status = "active"
    elif anchor >= len(points) - confirmation:
        status = "weakening"
    else:
        status = "broken"
    failures = 0
    start_index = anchor
    for index in range(anchor, -1, -1):
        if points[index].good:
            failures = 0
            start_index = index
        else:
            failures += 1
            if failures >= confirmation:
                start_index = index + confirmation
                break
    start_index = min(start_index, len(points) - 1)
    return status, points[start_index].validation_start, points[start_index].known_on


def _trend(points: list[RollingPoint]) -> str:
    recent = points[-10:]
    if len(recent) < 4:
        return "insufficient"
    x_mean = (len(recent) - 1) / 2
    y_mean = statistics.mean(point.nrmse for point in recent)
    denominator = sum((index - x_mean) ** 2 for index in range(len(recent)))
    slope = sum((index - x_mean) * (point.nrmse - y_mean)
                for index, point in enumerate(recent)) / max(denominator, 1e-9)
    if slope > 0.03:
        return "rising error"
    if slope < -0.03:
        return "falling error"
    return "stable"


def _previous_session(dates, start_date: str | None) -> str | None:
    if not start_date:
        return None
    target = datetime.fromisoformat(start_date).date()
    previous = [stamp.date() for stamp in dates if stamp.date() < target]
    return previous[-1].isoformat() if previous else None


def _news_for_start(symbol: str, company: str, prior_session: str | None,
                    start_date: str | None) -> tuple[int | None, list[dict[str, str]]]:
    if not prior_session or not start_date:
        return None, []
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    start = datetime.combine(datetime.fromisoformat(prior_session).date(), clock_time.min, eastern)
    end = datetime.combine(datetime.fromisoformat(start_date).date(), clock_time(9, 30), eastern)
    # before: is exclusive, so include the start session in the search and then
    # enforce the exact publication interval ourselves.
    before = datetime.fromisoformat(start_date).date() + timedelta(days=1)
    query = f'"{company}" stock after:{prior_session} before:{before.isoformat()}'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 wave-study/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            root = ElementTree.fromstring(response.read())
    except Exception:
        return None, []
    articles = []
    domains = set()
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source_node = item.find("source")
        source = ((source_node.text if source_node is not None else "") or "").strip()
        domain = urllib.parse.urlparse(
            source_node.attrib.get("url", "") if source_node is not None else ""
        ).netloc.lower().removeprefix("www.")
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(eastern)
        except (TypeError, ValueError):
            continue
        if not (start <= published <= end):
            continue
        if not _is_relevant_talk(title, source, domain, symbol, company):
            continue
        domains.add(domain or source.casefold())
        articles.append({
            "source": source or domain or "Unknown", "title": title, "url": link,
            "published": published.isoformat(),
        })
    articles.sort(key=lambda item: item["published"])
    return len(domains), articles[:10]


def run(include_news: bool = True) -> list[WaveResult]:
    import numpy as np
    import yfinance as yf

    _, _, names = load_universe_details()
    data = yf.download(
        SYMBOLS, period="2y", interval="1d", group_by="ticker", auto_adjust=True,
        actions=False, threads=True, progress=False, timeout=40, multi_level_index=True,
    )
    results: list[WaveResult] = []
    for symbol in SYMBOLS:
        try:
            close = data[symbol]["Close"].dropna().iloc[-504:]
        except (KeyError, TypeError):
            continue
        if len(close) < 80:
            continue
        values = np.log(close.to_numpy(dtype=float))
        current = _fit_frequency_model(values[-252:], exact_k=3)
        if not current:
            continue
        periods = [1 / float(frequency) for frequency in current["frequencies"]]
        for number, period in enumerate(periods, 1):
            train_days, validation_days = _adaptive_lengths(period)
            points = [point for end in range(train_days + validation_days, len(values) + 1)
                      if (point := _rolling_point(values, close.index, period, end))]
            status, estimated_start, confirmed_on = _find_current_regime(points, period)
            prior = _previous_session(close.index, estimated_start)
            sites, articles = (None, [])
            if include_news and estimated_start:
                sites, articles = _news_for_start(
                    symbol, names.get(symbol, symbol), prior, estimated_start
                )
                time.sleep(0.25)
            latest = points[-1] if points else None
            results.append(WaveResult(
                symbol, names.get(symbol, symbol), number, round(period, 2), train_days,
                validation_days, status, estimated_start, confirmed_on, prior,
                latest.q if latest else None, latest.nrmse if latest else None,
                latest.residual_sigma if latest else None, _trend(points), len(points),
                sites, articles,
            ))
    return results


def _write_results(results: list[WaveResult]) -> None:
    OUTPUT_JSON.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = [field for field in WaveResult.__dataclass_fields__ if field != "news_articles"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row.pop("news_articles")
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()
    results = run(include_news=not args.skip_news)
    _write_results(results)
    active = sum(result.status == "active" for result in results)
    news = sum(bool(result.news_articles) for result in results)
    print(f"waves={len(results)} active={active} waves_with_news={news}")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
