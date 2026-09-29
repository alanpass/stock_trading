# -*- coding: utf-8 -*-
"""Windows Task Scheduler entry point for the local AI Research Assistant."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from research_agent import ResearchAgent
from email_agent import EmailAgent
from stock_api import clean_symbol

BASE = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE / "user_watchlist.json"
DEFAULT_WATCHLIST = [
    "3481", "2327", "2492", "3037", "3044", "3533", "2303", "2330", "2344", "2408",
    "6515", "7769", "6488", "3374", "2377", "3450", "0050", "3006", "3661", "8299",
    "2308", "6274",
]

# Research is allowed only after this time.
# The Windows task should normally start at the same time.
AFTER_CLOSE_HOUR = int(os.getenv("AFTER_CLOSE_HOUR", "14"))
AFTER_CLOSE_MINUTE = int(os.getenv("AFTER_CLOSE_MINUTE", "30"))


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
    now = pd.Timestamp.now(tz="Asia/Taipei")
    cutoff = pd.Timestamp(
        year=now.year,
        month=now.month,
        day=now.day,
        hour=AFTER_CLOSE_HOUR,
        minute=AFTER_CLOSE_MINUTE,
        tz="Asia/Taipei",
    )

    print("=" * 72)
    print("AI TW STOCK AFTER-CLOSE RESEARCH")
    print("=" * 72)
    print(f"時間：{now}")
    print(f"專案：{BASE}")
    print(f"Python：{os.sys.executable}")
    print(f"Watchlist：{', '.join(load_watchlist())}")
    print("FUGLE_API_KEY：" + ("已設定" if os.getenv("FUGLE_API_KEY") else "由 stock_api.py / 環境提供的 Key 未確認"))
    print(f"OLLAMA_MODEL：{os.getenv('OLLAMA_MODEL', 'qwen3:8b')}")
    print(f"EARNINGS_MEMO_LOOKBACK_DAYS：{os.getenv('EARNINGS_MEMO_LOOKBACK_DAYS', '5')}")
    print(f"CNYES_RESEARCH_LOOKBACK_DAYS：{os.getenv('CNYES_RESEARCH_LOOKBACK_DAYS', '2')}")
    print("CNYES：盤後只讀夜間快取，不重新爬網站")
    print("盤後模式：快速模式，不重複做 CNYES 新聞摘要／Google News 搜尋／法說會全文分析")
    print("")

    if now < cutoff:
        print(f"尚未到 {AFTER_CLOSE_HOUR:02d}:{AFTER_CLOSE_MINUTE:02d}，略過研究。")
        return 0

    print("[1/2] Research Agent 啟動（FAST MODE）")

    # 盤後只做市場／財報／題材／法人等盤後整理；新聞深度摘要已在早報完成。
    os.environ["AFTER_CLOSE_FAST_MODE"] = "true"
    os.environ["AGENT_MAX_TOOL_ROUNDS"] = os.getenv("AFTER_CLOSE_AGENT_MAX_TOOL_ROUNDS", "6")
    os.environ["EARNINGS_USE_CACHE"] = "true"

    try:
        agent = ResearchAgent(
            BASE,
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        )
        report = agent.run_daily_research(
            symbols=load_watchlist(),
            sector_codes=["24", "26", "28"],
        )
    except Exception as exc:
        print(f"Research Agent 失敗：{exc}")
        return 2

    earnings_calls = [
        x for x in report.get("earnings_calls", [])
        if isinstance(x, dict) and not x.get("error")
    ]

    agent_notes = report.get("agent_research_notes", {}) or {}

    print("")
    print("Research Agent 完成")
    print(f"report_date = {report.get('report_date')}")
    print(f"earnings_calls = {len(earnings_calls)}")
    print(f"cnyes_cache_dates = {report.get('cnyes_news', {}).get('cache_dates', [])}")
    print(f"cnyes_news_articles = {report.get('cnyes_news', {}).get('article_count', 0)}")
    cnyes_digest = report.get("cnyes_research_digest", {}) or {}
    print(f"cnyes_agent_status = {cnyes_digest.get('agent_status', 'unknown')}")
    print(f"cnyes_agent_findings = {len(cnyes_digest.get('key_findings', []) or [])}")
    print(f"research_notes = {len(agent_notes.get('research_notes', []))}")
    print(f"agent_status = {agent_notes.get('agent_status', 'unknown')}")
    print(f"json_path = {report.get('json_path')}")
    print(f"markdown_path = {report.get('markdown_path')}")

    if agent_notes.get("agent_actions"):
        print(f"Agent tool actions = {len(agent_notes['agent_actions'])}")
        for action in agent_notes["agent_actions"]:
            print(f"  - {action.get('tool', '')}")

    print("")
    print("[2/2] Email Agent 啟動")

    try:
        email_agent = EmailAgent(BASE)
        status = email_agent.status()
        print(
            f"Email configured={status.get('configured')} "
            f"auto_send={status.get('auto_send')} "
            f"recipient={status.get('recipient')}"
        )

        if email_agent.config.auto_send:
            # 每執行一次就寄一次；不做同日去重。
            email_result = email_agent.send_report(report)
        else:
            email_result = {
                "sent": False,
                "skipped": True,
                "reason": "EMAIL_AUTO_SEND=false",
                "status": status,
            }
    except Exception as exc:
        email_result = {
            "sent": False,
            "skipped": False,
            "error": f"Email Agent 執行失敗：{exc}",
        }

    print("")
    print("Email result:")
    print(json.dumps(email_result, ensure_ascii=False, indent=2, default=str))

    # Email failure should be visible to Task Scheduler.
    # A skipped email due to auto_send=false is a deliberate configuration.
    if email_result.get("error"):
        return 3

    print("")
    print("=" * 72)
    print("AI AFTER-CLOSE RESEARCH COMPLETED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
