"""Create an interactive price/news chart for one selected stock."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, time as clock_time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from market_scanner import load_universe_details
from web_gui import _company_search_name, _is_relevant_talk, _reference_daily_frame

NEW_YORK = ZoneInfo("America/New_York")
CACHE_FILE = Path(__file__).with_name("historical_article_count_cache.json")


def _session_news_window(previous_session, session) -> tuple[datetime, datetime]:
    """Causal window: previous session open through this session's open."""
    start = datetime.combine(previous_session, clock_time(9, 30), NEW_YORK)
    end = datetime.combine(session, clock_time(9, 30), NEW_YORK)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _parse_historical_feed(xml: bytes, start: datetime, end: datetime,
                           symbol: str, company: str) -> list[dict[str, str]]:
    articles, seen = [], set()
    for item in ElementTree.fromstring(xml).findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source_node = item.find("source")
        source = (source_node.text or "Unknown").strip() if source_node is not None else "Unknown"
        source_url = source_node.get("url", "") if source_node is not None else ""
        title = re.sub(rf"\s+[-–—]\s*{re.escape(source)}\s*$", "", title,
                       flags=re.IGNORECASE).strip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        domain = urllib.parse.urlparse(source_url).netloc.lower().removeprefix("www.")
        if not start <= published < end:
            continue
        if not _is_relevant_talk(title, source, domain, symbol, company):
            continue
        key = re.sub(r"\W+", "", title.casefold()) or link
        if key in seen:
            continue
        seen.add(key)
        articles.append({"title": title, "source": source, "link": link,
                         "published": published.isoformat()})
    return sorted(articles, key=lambda article: article["published"])


def _load_cache() -> dict[str, object]:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict[str, object]) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _download_session_articles(symbol: str, company: str, sessions: list,
                               pause: float = 0.12) -> dict[str, list[dict[str, str]]]:
    cache, results = _load_cache(), {}
    company_term = _company_search_name(company)
    for index in range(1, len(sessions)):
        previous, session = sessions[index - 1], sessions[index]
        key = f"{symbol}:{session.isoformat()}"
        cached = cache.get(key)
        if isinstance(cached, dict) and isinstance(cached.get("articles"), list):
            results[session.isoformat()] = cached["articles"]
            continue
        start, end = _session_news_window(previous, session)
        query = (f'("{company_term}" OR ("{symbol}" (stock OR shares OR NASDAQ OR NYSE))) '
                 f'after:{start.date()} before:{(end + timedelta(days=1)).date()}')
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
            {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        request = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 stock-research-tool/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                articles = _parse_historical_feed(response.read(), start, end, symbol, company)
        except Exception as exc:
            print(f"  news {session}: {type(exc).__name__}; recorded as unavailable")
            articles = []
        results[session.isoformat()] = articles
        cache[key] = {"articles": articles, "fetched_at": datetime.now(timezone.utc).isoformat()}
        _save_cache(cache)
        time.sleep(pause)
    return results


def _download_prices(symbol: str, days: int):
    import pandas as pd
    import yfinance as yf

    start = (datetime.now(timezone.utc) - timedelta(days=max(days + 45, 150))).date().isoformat()
    daily = yf.download(symbol, start=start, interval="1d", auto_adjust=True,
                        actions=False, progress=False, timeout=30, multi_level_index=False)
    hourly = yf.download(symbol, start=start, interval="60m", auto_adjust=True,
                         actions=False, prepost=False, progress=False, timeout=30,
                         multi_level_index=False)
    close = _reference_daily_frame(daily, "close")
    first_hour = _reference_daily_frame(hourly, "first_hour")
    if close is None or first_hour is None:
        raise RuntimeError("Yahoo Finance did not return both daily and first-hour prices.")
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    first_hour.index = pd.to_datetime(first_hour.index).tz_localize(None).normalize()
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=days)
    joined = close.rename(columns={"Close": "close"}).join(
        first_hour.rename(columns={"Close": "first_hour"}), how="inner")
    return joined[joined.index >= cutoff].dropna()


def _build_report(symbol: str, company: str, rows: list[dict[str, object]], output: Path) -> None:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"{symbol} · {company} — price and pre-open news")
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="robots" content="noindex,nofollow"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--bg:#091521;--panel:#112438;--grid:#29435b;--text:#d7e4f3;--muted:#8fa7bd;--green:#32db8b;--orange:#ff9d3d;--blue:#55a6ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui;padding:28px}}.wrap{{max-width:1180px;margin:auto}}
h1{{font-size:25px;margin:0 0 5px}}p{{color:var(--muted);margin:0 0 22px}}.card{{background:var(--panel);border:1px solid #31516d;border-radius:18px;padding:18px;margin:14px 0}}
.label{{font-weight:700;margin-bottom:8px}}svg{{width:100%;height:285px;display:block;overflow:visible}}.tip{{position:fixed;display:none;max-width:440px;z-index:5;background:#07111c;border:1px solid #4b7192;border-radius:10px;padding:11px;pointer-events:none;box-shadow:0 10px 30px #0008}}
.tip b{{color:#fff}}.article{{margin-top:6px;color:#bcd0e2}}.source{{color:var(--orange)}}</style></head><body><div class="wrap"><h1>{title}</h1>
<p>Orange bars count unique relevant articles published after the previous session opened and before 09:30 ET on the displayed session.</p>
<div class="card"><div class="label">Daily adjusted close</div><svg id="close"></svg></div>
<div class="card"><div class="label">Price after first trading hour + articles known before open</div><svg id="hour"></svg></div></div><div class="tip" id="tip"></div><script>
const data={payload},NS='http://www.w3.org/2000/svg';function E(t,a={{}}){{let e=document.createElementNS(NS,t);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));return e}}
function chart(id,key,bars=false){{let s=document.getElementById(id),W=1100,H=270,m={{l:62,r:55,t:18,b:38}},iw=W-m.l-m.r,ih=H-m.t-m.b;s.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);let vs=data.map(d=>d[key]),lo=Math.min(...vs),hi=Math.max(...vs),pad=(hi-lo||1)*.09,y=v=>m.t+ih-(v-(lo-pad))/(hi-lo+2*pad)*ih,x=i=>m.l+(data.length===1?iw/2:i*iw/(data.length-1));
for(let j=0;j<5;j++){{let yy=m.t+j*ih/4;s.append(E('line',{{x1:m.l,x2:W-m.r,y1:yy,y2:yy,stroke:'#29435b'}}));let v=hi+pad-j*(hi-lo+2*pad)/4,t=E('text',{{x:4,y:yy+4,fill:'#8fa7bd','font-size':12}});t.textContent='$'+v.toFixed(2);s.append(t)}}
if(bars){{let max=Math.max(1,...data.map(d=>d.news_count));data.forEach((d,i)=>{{let h=d.news_count/max*ih*.72;s.append(E('rect',{{x:x(i)-Math.max(2,iw/data.length*.28),y:m.t+ih-h,width:Math.max(4,iw/data.length*.56),height:h,fill:'#ff9d3d',opacity:.45,rx:2}}))}})}}
let path=data.map((d,i)=>`${{i?'L':'M'}} ${{x(i)}} ${{y(d[key])}}`).join(' ');s.append(E('path',{{d:path,fill:'none',stroke:bars?'#55a6ff':'#32db8b','stroke-width':3,'stroke-linejoin':'round'}}));
data.forEach((d,i)=>{{if(i%10===0||i===data.length-1){{let t=E('text',{{x:x(i),y:H-12,fill:'#8fa7bd','font-size':11,'text-anchor':'middle'}});t.textContent=d.date.slice(5);s.append(t)}}let hit=E('circle',{{cx:x(i),cy:y(d[key]),r:9,fill:'transparent'}});hit.onmouseenter=e=>show(e,d);hit.onmousemove=move;hit.onmouseleave=hide;s.append(hit)}})}}
let tip=document.getElementById('tip');function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}function show(e,d){{let a=d.articles.slice(0,8).map(x=>`<div class="article"><span class="source">${{esc(x.source)}}</span> · ${{esc(x.title)}}<br><small>${{new Date(x.published).toLocaleString()}}</small></div>`).join('');tip.innerHTML=`<b>${{d.date}}</b><br>Close: $${{d.close.toFixed(2)}}<br>After first hour: $${{d.first_hour.toFixed(2)}}<br>Pre-open articles: <b>${{d.news_count}}</b>${{a}}`;tip.style.display='block';move(e)}}function move(e){{tip.style.left=Math.min(innerWidth-tip.offsetWidth-12,e.clientX+16)+'px';tip.style.top=Math.min(innerHeight-tip.offsetHeight-12,e.clientY+16)+'px'}}function hide(){{tip.style.display='none'}}chart('close','close');chart('hour','first_hour',true);
</script></body></html>'''
    output.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Ticker symbol, for example AAOI")
    parser.add_argument("--days", type=int, default=90, help="Calendar days to include")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    symbol = args.symbol.upper().replace(".", "-")
    _, _, names = load_universe_details()
    company = names.get(symbol, symbol)
    print(f"Downloading prices for {symbol} ({company})...")
    prices = _download_prices(symbol, args.days)
    sessions = [stamp.date() for stamp in prices.index]
    print(f"Checking historical news for {len(sessions)} sessions; cached sessions are reused...")
    news = _download_session_articles(symbol, company, sessions)
    rows = []
    for stamp, row in prices.iterrows():
        date = stamp.date().isoformat(); articles = news.get(date, [])
        rows.append({"date": date, "close": round(float(row["close"]), 4),
                     "first_hour": round(float(row["first_hour"]), 4),
                     "news_count": len(articles), "articles": articles})
    output = args.output or Path(f"{symbol}_news_first_hour_chart.html")
    _build_report(symbol, company, rows, output)
    print(f"Created {output.resolve()} ({len(rows)} sessions).")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
