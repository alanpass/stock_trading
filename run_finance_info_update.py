# -*- coding: utf-8 -*-
"""
AI 台股研究中心｜財經資訊自動更新（新聞 + 法說會 → Agent 摘要／畫重點／分析）

排程（Windows Task Scheduler，每天）：
    08:10  11:00  13:30  16:00  18:00  23:00
另外 08:30 晨報、14:30 盤後分析會直接讀這裡產生的 finance_info_latest.json。

每次執行：
    1. 爬鉅亨新聞（前一日 00:00 ～ 現在，頭條 + 台股 + 國際股 + 外匯 + 期貨）
    2. 取得最近 5 天 Fugle 法說會備忘錄（已分析過的不重做）
    3. FinanceNewsAgent：每篇摘要成 1~3 點、標示關鍵詞、判斷利多利空與產業、算重要度、
       再產出「今日重點」總覽
    4. 原子式覆蓋 output/research_reports/finance_info_latest.json
       → 上一個時間點的資料直接被取代；若這次完全爬不到，保留上一版並標示 stale

輸出：output/research_reports/finance_info_latest.json
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from cnyes_news_crawler import CnyesNewsCrawler
from finance_agent import FinanceNewsAgent

TAIPEI = ZoneInfo("Asia/Taipei")
BASE = Path(__file__).resolve().parent
FINANCE_SCHEDULE = ["08:10", "11:00", "13:30", "16:00", "18:00", "23:00"]
NEWS_LOOKBACK_DAYS = 2
EARNINGS_LOOKBACK_DAYS = 5
NEWS_DATE_KEYS = ("published_ts", "published", "published_at", "published_time", "date", "created_at", "updated_at")
EARN_DATE_KEYS = ("event_date", "eventDate", "published_date", "modified_date", "date", "published_at", "published_time", "published_ts", "published", "created_at")


def now_taipei() -> datetime:
    return datetime.now(TAIPEI)


_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{now_taipei().strftime('%H:%M:%S')} +{time.time() - _T0:5.0f}s] {msg}", flush=True)


def _run_with_timeout(fn, timeout: int):
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"超過 {timeout} 秒仍未完成")
    if "error" in box:
        raise box["error"]
    return box.get("value")


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


def _filter_recent_calendar_days(items: list[dict[str, Any]], days: int, date_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """保留今天與前 N-1 個日曆日，讓新聞與法說會可各自設定範圍。"""
    today = now_taipei().date()
    cutoff = today - timedelta(days=max(0, int(days) - 1))
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
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
        title = str(item.get("title") or item.get("headline") or item.get("url") or item.get("source_url") or "")
        symbol = str(item.get("symbol") or item.get("code") or item.get("stock_code") or "")
        identity = (dt.isoformat(), symbol, title)
        if identity in seen:
            continue
        seen.add(identity)
        row = dict(item)
        row["_date"] = dt.isoformat()
        out.append(row)
    out.sort(key=lambda x: str(x.get("published_ts") or x.get("published") or x.get("_date") or ""), reverse=True)
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


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# ----------------------------------------------------------------------
# 新聞
# ----------------------------------------------------------------------
def _google_news_fallback(days: int = 2) -> list[dict[str, Any]]:
    """鉅亨抓不到時的備援；只當保底，來源會標註。"""
    query = quote_plus("台股 OR 台灣股市 OR AI 伺服器 OR 半導體 OR 電子業 OR 聯準會")
    url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    cutoff = now_taipei().date() - timedelta(days=max(0, int(days) - 1))
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 TaiwanStockFinanceInfo/2.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        rows = []
        for item in root.findall("./channel/item")[:100]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            dt = _parse_date(pub)
            if not title or not link or dt is None or not (cutoff <= dt <= now_taipei().date()):
                continue
            rows.append({
                "article_id": link, "title": title, "url": link, "published": dt.isoformat(), "published_ts": dt.isoformat(),
                "category": "Google News", "summary": (item.findtext("description") or "").strip(), "content": "",
                "source": "Google News RSS（鉅亨備援）", "list_category": "fallback",
            })
        return rows
    except Exception:
        return []


def _crawl_news(base: Path, errors: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """滾動爬取；失敗時退回讀取既有快取。"""
    crawler = CnyesNewsCrawler(
        base,
        max_articles=int(os.getenv("CNYES_NIGHTLY_MAX_ARTICLES", "600")),
        max_scrolls=int(os.getenv("CNYES_NEWS_MAX_SCROLLS", "18")),
    )
    crawl_info: dict[str, Any] = {"ok": False}
    log("新聞：開始爬取鉅亨（頭條／台股／國際股／外匯／期貨，前一日 00:00 至現在）")
    try:
        result = crawler.crawl_rolling(days=NEWS_LOOKBACK_DAYS)
        log(f"新聞：爬到 {result.get('article_count', 0)} 篇，各分類 {result.get('per_category_count')}")
        crawl_info = {
            "ok": result.get("article_count", 0) > 0,
            "article_count": result.get("article_count", 0),
            "per_category_count": result.get("per_category_count", {}),
            "source_method": result.get("source_method"),
            "reused_from_previous_run": result.get("reused_from_previous_run", 0),
            "window_start": result.get("window_start"),
            "window_end": result.get("window_end"),
        }
        errors.extend(str(x) for x in (result.get("errors") or [])[:10])
        articles = _extract_list(result, ("articles",))
    except Exception as exc:
        errors.append(f"CNYES 爬取失敗：{type(exc).__name__}: {exc}")
        crawl_info["error"] = f"{type(exc).__name__}: {exc}"
        articles = []

    if not articles:
        try:
            cached = crawler.load_latest() or crawler.load_recent_cached(
                before_date=(now_taipei().date() + timedelta(days=1)).isoformat(), days=NEWS_LOOKBACK_DAYS, max_search_days=7
            )
            articles = _extract_list(cached, ("articles",))
            if articles:
                errors.append("本次未取得新文章，改用既有 CNYES 快取。")
                crawl_info["used_cache"] = True
        except Exception as exc:
            errors.append(f"CNYES 快取讀取失敗：{type(exc).__name__}: {exc}")

    articles = _filter_recent_calendar_days(articles, NEWS_LOOKBACK_DAYS, NEWS_DATE_KEYS)
    min_news = int(os.getenv("FINANCE_MIN_NEWS", "8"))
    if len(articles) < min_news:
        extra = _filter_recent_calendar_days(_google_news_fallback(NEWS_LOOKBACK_DAYS), NEWS_LOOKBACK_DAYS, NEWS_DATE_KEYS)
        known = {str(a.get("title")) for a in articles}
        added = [x for x in extra if str(x.get("title")) not in known]
        if added:
            errors.append(f"鉅亨新聞只有 {len(articles)} 篇（少於 {min_news}），已補入 {len(added)} 篇 Google News 備援。")
            articles.extend(added)
            crawl_info["fallback_added"] = len(added)
    return articles, crawl_info


# ----------------------------------------------------------------------
# 法說會
# ----------------------------------------------------------------------
def _load_earnings(base: Path, previous: dict[str, Any], watchlist: list[str], errors: list[str]) -> list[dict[str, Any]]:
    prev_items = [x for x in (previous.get("earnings") or []) if isinstance(x, dict)]
    prev_items = _filter_recent_calendar_days(prev_items, EARNINGS_LOOKBACK_DAYS, EARN_DATE_KEYS)
    force = os.getenv("EARNINGS_FORCE_REFRESH", "false").strip().lower() in {"1", "true", "yes", "on"}
    skip = set() if force else {str(x.get("url") or x.get("source_url")) for x in prev_items if (x.get("url") or x.get("source_url"))}
    fresh: list[dict[str, Any]] = []
    if os.getenv("FINANCE_SKIP_EARNINGS", "").strip().lower() in {"1", "true", "yes", "on"}:
        log("法說會：略過（--no-earnings）")
        return prev_items
    per_run = int(os.getenv("EARNINGS_NEW_PER_RUN", "12"))      # 每次最多新分析幾場，多的留給下一個時段
    timeout = int(os.getenv("EARNINGS_TIMEOUT_SEC", "600"))
    try:
        from earnings_call_agent import EarningsCallAgent

        log(f"法說會：開始（已有 {len(prev_items)} 筆；本次最多新分析 {per_run} 場，逾時 {timeout} 秒）")

        def work():
            agent = EarningsCallAgent(base, ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"))
            return agent.daily_run(days=EARNINGS_LOOKBACK_DAYS, limit=per_run, force=force, watchlist=[], skip_urls=skip or None)

        payload = _run_with_timeout(work, timeout)
        fresh = _extract_list(payload, ("items", "earnings_calls", "events", "data"))
        if isinstance(payload, dict):
            discovered_count = payload.get("discovered_count", "未知")
            recent_count = payload.get("recent_count", "未知")
            selected_count = payload.get("selected_count", len(fresh))
            log(
                f"法說會：探索 {discovered_count} 個文章網址；最近 {EARNINGS_LOOKBACK_DAYS} 天符合 {recent_count} 個；"
                f"本次選取 {selected_count} 個、成功分析 {len(fresh)} 筆"
            )
            if not discovered_count or discovered_count == 0:
                debug = json.dumps(payload.get("discovery_debug") or {}, ensure_ascii=False)
                log(f"法說會：爬取診斷 {debug[:2400]}")
            for e in (payload.get("errors") or [])[:5]:
                if isinstance(e, dict):
                    message = e.get("warning") or e.get("error")
                    if message:
                        errors.append(f"法說會：{message}")
        log(f"法說會：新增 {len(fresh)} 筆")
    except TimeoutError as exc:
        errors.append(f"法說會逾時（{exc}），本次沿用既有資料；剩下的會在下一個時段繼續。")
        log(f"法說會：逾時，沿用既有 {len(prev_items)} 筆")
    except Exception as exc:
        errors.append(f"法說會資料更新失敗：{type(exc).__name__}: {exc}")
        log(f"法說會：失敗 {type(exc).__name__}: {exc}")
    return _filter_recent_calendar_days(fresh + prev_items, EARNINGS_LOOKBACK_DAYS, EARN_DATE_KEYS)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def load_watchlist(base: Path = BASE) -> list[str]:
    path = base / "user_watchlist.json"
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
    base = Path(base_dir).resolve()
    report_dir = base / "output" / "research_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    latest = report_dir / "finance_info_latest.json"
    previous = _read_json(latest)

    started = now_taipei()
    symbols = watchlist if watchlist is not None else load_watchlist(base)
    errors: list[str] = []

    def build_payload(news, news_digest, earnings, earnings_digest, crawl_info, stale, agent_stats):
        return {
            "report_type": "finance_info",
            "report_date": started.date().isoformat(),
            "updated_at": started.isoformat(timespec="seconds"),
            "finished_at": now_taipei().isoformat(timespec="seconds"),
            "lookback_days": NEWS_LOOKBACK_DAYS,
            "earnings_lookback_days": EARNINGS_LOOKBACK_DAYS,
            "schedule": FINANCE_SCHEDULE,
            "stale": stale,
            "news": news,
            "news_count": len(news),
            "news_digest": news_digest,
            "earnings": earnings,
            "earnings_count": len(earnings),
            "earnings_digest": earnings_digest,
            "crawl": crawl_info,
            "agent_stats": agent_stats,
            "errors": errors,
            "watchlist_count": len(symbols),
        }

    # 1) 新聞
    if refresh_news:
        articles, crawl_info = _crawl_news(base, errors)
    else:
        articles = _filter_recent_calendar_days([x for x in previous.get("news", []) if isinstance(x, dict)], NEWS_LOOKBACK_DAYS, NEWS_DATE_KEYS)
        crawl_info = {"ok": bool(articles), "skipped": True}
    log(f"新聞：進入 Agent 分析（{len(articles)} 篇）")

    # 2) Agent 分析新聞（完成後先寫入一次，儀表板馬上就能看到新聞；法說會稍後補上）
    agent = FinanceNewsAgent(base)
    stale = False
    if articles:
        news, news_digest = agent.analyze_news(articles, now=started)
    else:
        prev_news = [x for x in previous.get("news", []) if isinstance(x, dict)]
        if prev_news:
            news, news_digest, stale = prev_news, previous.get("news_digest", {}), True
            errors.append("本次完全沒有取得新聞，保留上一個時間點的資料（stale）。")
        else:
            news, news_digest = [], {}
    log(f"新聞：Agent 完成 {agent.stats}")
    prev_earn = _filter_recent_calendar_days([x for x in previous.get("earnings", []) if isinstance(x, dict)], EARNINGS_LOOKBACK_DAYS, EARN_DATE_KEYS)
    _atomic_write_json(latest, build_payload(news, news_digest, prev_earn, previous.get("earnings_digest", {}), crawl_info, stale, agent.stats))
    log("已先寫入新聞結果（儀表板現在就能看到）")

    # 3) 法說會
    earnings_raw = _load_earnings(base, previous, symbols, errors)
    earnings, earnings_digest = agent.analyze_earnings(earnings_raw, days=EARNINGS_LOOKBACK_DAYS)
    payload = build_payload(news, news_digest, earnings, earnings_digest, crawl_info, stale, agent.stats)
    _atomic_write_json(latest, payload)  # 直接取代上一個時間點的資料
    log(f"完成：新聞 {len(news)} 篇、法說會 {len(earnings)} 筆")

    # 只保留最近 30 次的執行摘要（不含全文），看得到每個時段到底抓了幾篇
    hist_path = report_dir / "finance_run_history.json"
    hist = _read_json(hist_path).get("runs", [])
    hist.append({
        "at": payload["updated_at"], "news": len(news), "earnings": len(earnings),
        "per_category": crawl_info.get("per_category_count", {}), "llm": agent.stats.get("llm_summarized", 0),
        "stale": stale, "errors": len(errors),
    })
    _atomic_write_json(hist_path, {"runs": hist[-30:]})

    return {
        "ok": True,
        "updated_at": payload["updated_at"],
        "latest": str(latest),
        "earnings_count": len(earnings),
        "news_count": len(news),
        "per_category_count": crawl_info.get("per_category_count", {}),
        "agent_stats": agent.stats,
        "stale": stale,
        "errors": errors,
    }


def main() -> int:
    print("=" * 72)
    print("AI TW STOCK FINANCE INFO UPDATE")
    print(f"TIME={now_taipei().isoformat()}")
    try:
        if "--no-llm" in sys.argv:
            os.environ["FINANCE_USE_LLM"] = "false"
        if "--no-earnings" in sys.argv:
            os.environ["FINANCE_SKIP_EARNINGS"] = "true"
        if "--quick" in sys.argv:  # 只爬新聞＋規則摘要，幾十秒內完成，用來確認流程
            os.environ["FINANCE_USE_LLM"] = "false"
            os.environ["FINANCE_SKIP_EARNINGS"] = "true"
        result = update_finance_info(BASE, watchlist=load_watchlist(BASE), refresh_news="--no-crawl" not in sys.argv)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"ERROR={type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
