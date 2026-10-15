# -*- coding: utf-8 -*-
"""AI 研究共用設定。

所有需要 LLM 的研究（盤後研究、早報、新聞重點、法說會摘要、個股研究筆記）都由
Claude 排程任務依 .claude/skills/ 的流程完成。Python 程式只負責抓資料、計算、寄信與發布，
在 Claude 寫回結果之前，研究欄位一律標示為 pending_claude。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

AGENT_STATUS = "claude_agent"
PENDING_STATUS = "pending_claude"
AGENT_MODEL = "claude"
AGENT_LABEL = "Claude 研究 Agent"
TAIPEI = ZoneInfo("Asia/Taipei")


def is_agent_status(status: object) -> bool:
    """研究內容是否已由 AI Agent 完成（Claude；舊報告的 Qwen3 也算），而不是程式化 fallback。"""
    s = str(status or "")
    return s == AGENT_STATUS or s.startswith("qwen3")


def status_label(status: object) -> str:
    s = str(status or "")
    if s == AGENT_STATUS:
        return AGENT_LABEL
    if s == PENDING_STATUS:
        return "等待 Claude 研究"
    if s.startswith("qwen3"):
        return "Qwen3 Agent（舊報告）"
    return "保守 fallback"


def pending_agent_notes() -> dict[str, Any]:
    """盤後研究筆記的空白結構；Claude 完成後由 run_after_close_research.py --finalize 取代。"""
    return {
        "agent_status": PENDING_STATUS,
        "model": AGENT_MODEL,
        "generated_at": datetime.now(TAIPEI).isoformat(),
        "overview": "",
        "key_takeaways": [],
        "research_notes": [],
        "positive_signals": [],
        "negative_signals": [],
        "watch_items": [],
        "earnings_digest": "",
        "financial_focus": [],
        "model_alert": "",
        "agent_actions": [],
    }


def pending_news_digest() -> dict[str, Any]:
    return {
        "agent_status": PENDING_STATUS,
        "model": AGENT_MODEL,
        "overview": "",
        "key_findings": [],
        "market_drivers": [],
        "watch_topics": [],
    }
