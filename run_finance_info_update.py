# -*- coding: utf-8 -*-
"""
AI 台股研究中心｜財經資訊快取更新器

更新時間由 Windows Task Scheduler 控制：
- 08:30：與晨報一起更新
- 18:00：獨立更新一次

來源：
- Fugle 法說會備忘錄：最近 2 天
- 鉅亨 CNYES 財經新聞：最近 2 天

輸出：
output/research_reports/finance_info_latest.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any
from urllib.parse import quote_plus
import requests
import xml.etree.ElementTree as ET

import pandas as pd

from earnings_call_agent import EarningsCallAgent
from cnyes_news_crawler import CnyesNewsCrawler

TAIPEI = ZoneInfo("Asia/Taipei")
BASE = Path(__file__).resolve().parent
REPORT_DIR = BASE / "output" / "research_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def now_taipei() -> datetime:
    return datetime.now(TAIPEI)


def _parse_date(value: Any):
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(TAIPEI).tz_localize(None)
        return ts.date()
    except Exception:
        return None


def _filter_recent(items: list[dict[str, Any]], days: int, date_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    today = now_taipei().date()
    cutoff = today - timedelta(days=max(1, int(days)))
    out: list[dict[str, Any]] = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        dt = None
        for key in date_keys:
            dt = _parse_date(item.get(key))
            if dt is not None:
                break
        if dt is None or not (cutoff <= dt <= today):
            continue

        row = dict(item)
        row["_date"] = dt.isoformat()

        identity = (
            dt.isoformat(),
            str(row.get("symbol") or row.get("code") or row.get("stock_code") or ""),
            str(row.get("title") or row.get("headline") or row.get("source_url") or row.get("url") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        out.append(row)

    out.sort(key=lambda x: str(x.get("_date", "")), reverse=True)
    return out


def _extract_list(data: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _run_cnyes_refresh() -> dict[str, Any]:
    """
    優先執行專案既有 CNYES 夜間爬蟲。
    這裡不要求一定在 23:00 才能執行，08:30 / 18:00 也可以重新建立近兩日快取。
    """
    script = BASE / "run_cnyes_news_nightly.py"
    if not script.exists():
        return {"ok": False, "skipped": True, "reason": f"找不到 {script}"}

    log = BASE / "output" / "scheduler_logs" / f"finance_cnyes_{now_taipei().date().isoformat()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("a", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(
                [sys.executable, "-u", str(script)],
                cwd=str(BASE),
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=int(os.getenv("FINANCE_CNYES_TIMEOUT", "900")),
            )
        return {"ok": proc.returncode == 0, "returncode": int(proc.returncode), "log": str(log)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "log": str(log)}


def _google_news_fallback(days: int = 2) -> list[dict[str, Any]]:
    """CNYES 無法更新時的備援新聞來源；只作財經資訊頁可用性保底。"""
    query = quote_plus("台股 OR 台灣股市 OR AI 伺服器 OR 半導體 OR 電子業")
    url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    cutoff = now_taipei().date() - timedelta(days=max(1, int(days)))
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 TaiwanStockFinanceInfo/1.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        rows = []
        for item in root.findall("./channel/item")[:80]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            dt = _parse_date(pub)
            if not title or not link or dt is None or not (cutoff <= dt <= now_taipei().date()):
                continue
            rows.append({
                "title": title,
                "url": link,
                "published_at": dt.isoformat(),
                "source": "Google News RSS（CNYES備援）",
            })
        return rows
    except Exception:
        return []


def _load_cnyes(days: int = 2) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    crawler = CnyesNewsCrawler(
        BASE,
        max_articles=int(os.getenv("CNYES_NIGHTLY_MAX_ARTICLES", "600")),
        max_scrolls=int(os.getenv("CNYES_NEWS_MAX_SCROLLS", "18")),
    )

    try:
        payload = crawler.load_recent_cached(
            before_date=now_taipei().date().isoformat(),
            days=days,
            max_search_days=int(os.getenv("CNYES_RESEARCH_CACHE_SEARCH_DAYS", "7")),
        )
        articles = _extract_list(payload, ("articles", "items", "data"))
        articles = _filter_recent(
            articles,
            days,
            ("published_at", "published_time", "date", "created_at", "updated_at"),
        )
        if articles:
            return articles, errors
        errors.append("CNYES 最近 2 天快取沒有可用文章")
        return [], errors
    except Exception as exc:
        errors.append(f"CNYES cache 讀取失敗：{type(exc).__name__}: {exc}")
        return [], errors


def _load_earnings(days: int = 2, watchlist: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    try:
        agent = EarningsCallAgent(BASE, ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"))
        payload = agent.daily_run(
            days=days,
            limit=int(os.getenv("EARNINGS_MEMO_MAX_ARTICLES", "80")),
            force=_clean_bool(os.getenv("EARNINGS_FORCE_REFRESH", "false")),
            watchlist=watchlist or [],
        )
        items = _extract_list(payload, ("items", "earnings_calls", "events", "data"))
        items = _filter_recent(
            items,
            days,
            ("event_date", "eventDate", "date", "published_at", "published_time", "created_at"),
        )
        if items:
            return items, errors
        errors.append("Fugle 最近 2 天沒有可用法說會 memo")
        return [], errors
    except Exception as exc:
        errors.append(f"法說會資料更新失敗：{type(exc).__name__}: {exc}")
        return [], errors


def _clean_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def load_watchlist() -> list[str]:
    path = BASE / "user_watchlist.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return list(dict.fromkeys(str(x).strip().replace(".TW", "") for x in data if str(x).strip()))
    except Exception:
        pass
    return []


def update_finance_info(base_dir: str | Path = BASE, watchlist: list[str] | None = None, refresh_news: bool = True) -> dict[str, Any]:
    global BASE, REPORT_DIR
    BASE = Path(base_dir).resolve()
    REPORT_DIR = BASE / "output" / "research_reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    started = now_taipei()
    symbols = watchlist if watchlist is not None else load_watchlist()
    errors: list[str] = []

    news_refresh = _run_cnyes_refresh() if refresh_news else {"ok": False, "skipped": True, "reason": "refresh_news=False"}
    if not news_refresh.get("ok") and not news_refresh.get("skipped"):
        errors.append(str(news_refresh.get("error") or news_refresh.get("reason") or "CNYES refresh failed"))

    earnings, earn_errors = _load_earnings(2, symbols)
    news, news_errors = _load_cnyes(2)
    errors.extend(earn_errors)
    errors.extend(news_errors)
    if not news:
        fallback_news = _google_news_fallback(2)
        if fallback_news:
            news = fallback_news
            errors.append("CNYES 無可用最近 2 天文章，財經資訊改用 Google News RSS 備援。")

    payload = {
        "report_type": "finance_info",
        "report_date": started.date().isoformat(),
        "updated_at": started.isoformat(),
        "lookback_days": 2,
        "earnings": earnings,
        "news": news,
        "earnings_count": len(earnings),
        "news_count": len(news),
        "cnyes_refresh": news_refresh,
        "errors": errors,
        "watchlist_count": len(symbols),
    }

    latest = REPORT_DIR / "finance_info_latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    stamp = started.strftime("%Y%m%d_%H%M%S")
    snapshot = REPORT_DIR / f"finance_info_{stamp}.json"
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return {
        "ok": True,
        "updated_at": started.isoformat(),
        "latest": str(latest),
        "snapshot": str(snapshot),
        "earnings_count": len(earnings),
        "news_count": len(news),
        "errors": errors,
    }


def main() -> int:
    print("=" * 72)
    print("AI TW STOCK FINANCE INFO UPDATE")
    print(f"TIME={now_taipei().isoformat()}")
    try:
        result = update_finance_info(BASE, watchlist=load_watchlist(), refresh_news=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"ERROR={type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
