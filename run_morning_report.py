# -*- coding: utf-8 -*-
"""Windows Task Scheduler 早報入口：讀夜間 CNYES 快取＋最近 5 天法說會，交給 Qwen3 產生早報。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from email_agent import EmailAgent
from morning_report_agent import MorningResearchAgent, build_morning_report, DEFAULT_WATCHLIST
from stock_api import clean_symbol

BASE = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE / "user_watchlist.json"
MORNING_HOUR = int(os.getenv("MORNING_REPORT_HOUR", "8"))
MORNING_MINUTE = int(os.getenv("MORNING_REPORT_MINUTE", "30"))


def load_watchlist() -> list[str]:
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                out = [clean_symbol(x) for x in data if clean_symbol(x)]
                if out:
                    return list(dict.fromkeys(out))
    except Exception:
        pass
    return DEFAULT_WATCHLIST.copy()


def main() -> int:
    print("=" * 72)
    print("AI TW STOCK MORNING REPORT")
    print("=" * 72)
    print(f"專案：{BASE}")
    print(f"Python：{os.sys.executable}")
    print(f"Watchlist：{', '.join(load_watchlist())}")
    print("新聞：只讀昨晚 23:00 建立的 CNYES 日檔，不重新爬網站")
    print("法說會：優先使用 Fugle memo cache；沒有 cache 才補抓")
    print("")

    try:
        report = build_morning_report(BASE, watchlist=load_watchlist())
    except Exception as exc:
        print(f"早報研究失敗：{exc}")
        return 2

    morning = report.get("morning_agent", {}) or {}
    print(f"report_date = {report.get('report_date')}")
    print(f"cnyes_cache_dates = {(report.get('cnyes_news') or {}).get('cache_dates', [])}")
    print(f"cnyes_articles = {(report.get('cnyes_news') or {}).get('article_count', 0)}")
    print(f"cnyes_agent_status = {(report.get('cnyes_research_digest') or {}).get('agent_status', '')}")
    print(f"cnyes_findings = {len((report.get('cnyes_research_digest') or {}).get('key_findings', []) or [])}")
    print(f"earnings_calls = {len(report.get('earnings_calls', []) or [])}")
    print(f"morning_agent_status = {morning.get('agent_status', '')}")
    print(f"news_summary = {len(morning.get('news_summary', []) or [])}")
    print(f"recommended_industries = {len(morning.get('recommended_industries', []) or [])}")
    print(f"recommended_stocks = {len(morning.get('recommended_stocks', []) or [])}")
    print(f"json_path = {report.get('json_path')}")

    try:
        email = EmailAgent(BASE)
        status = email.status()
        print(f"Email configured={status.get('configured')} auto_send={status.get('auto_send')} recipient={status.get('recipient')}")
        if email.config.auto_send:
            result = email.send_report(report)
        else:
            result = {"sent": False, "skipped": True, "reason": "EMAIL_AUTO_SEND=false", "status": status}
    except Exception as exc:
        result = {"sent": False, "skipped": False, "error": f"早報 Email 執行失敗：{exc}"}

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 3 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
