# S&P 500 + QQQ opening scanner

The script scans the union of the current S&P 500 and Nasdaq-100 (QQQ) constituents,
removes duplicates, downloads free one-minute bars from Yahoo Finance, and creates a CSV report.

## Definitions

- `first_10m`: change from the open of the 09:30 bar to the close of the 09:39 bar.
- `next_50m`: change from the open of the 09:40 bar to the close of the 10:29 bar.
- `after_30m`: change from the open of the 10:00 bar to the last available regular-session bar.
- Thresholds are strict: exactly 8.00% is not "more than 8%".
- All times are New York time. Early closes are handled by using the last available bar.

## Setup and use

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python market_scanner.py
```

This opens a live GUI table. Prices refresh every 60 seconds and are colored relative
to the previous trading day's close. Use the search field to filter by ticker or
company name.

To run the original opening-window CSV scanner instead:

```bash
python market_scanner.py --cli --date 2026-07-31
```

No brokerage account or API key is needed. Yahoo publishes US Nasdaq equity quotes as
real-time, but `yfinance` is an unofficial, best-effort interface: availability and
latency are not guaranteed, and large scans can occasionally be throttled. Bars are
updated at one-minute granularity. The complete report is written to
`scan_results.csv`; the terminal also prints only stocks whose `after_30m` gain is
greater than 5%.

For a controlled/custom universe, pass a text or CSV file whose first column contains
symbols:

```bash
python market_scanner.py --cli --date 2026-07-31 --symbols-file symbols.txt
```

Yahoo's one-minute history is intended for recent intraday dates, so use a date within
the last seven days for the most reliable result. Do not use a weekend/holiday date.
On the current trading day, run after 10:30 ET to obtain both complete opening windows.
