# -*- coding: utf-8 -*-
"""AI Research Assistant Agent integration test.

This test deliberately refuses to use research_state.json as the research packet,
because that file is a model-health snapshot rather than the full daily evidence packet.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_research_assistant import ResearchOrchestratorAgent

BASE = Path(__file__).resolve().parent
REPORT_DIR = BASE / "output" / "research_reports"


def _latest_report() -> Path | None:
    reports = sorted(REPORT_DIR.glob("research_*.json"))
    return reports[-1] if reports else None


def main() -> int:
    path = _latest_report()
    if not path:
        state = REPORT_DIR / "research_state.json"
        if state.exists():
            print("找不到 research_YYYY-MM-DD.json。")
            print("目前只有 research_state.json；它是模型狀態快照，不足以進行完整自主研究。")
            print("請先執行完整 ResearchAgent 產生每日研究報告。")
        else:
            print("找不到 output/research_reports/research_YYYY-MM-DD.json")
        return 1

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"讀取研究報告失敗：{exc}")
        return 1

    print("=" * 72)
    print("AI RESEARCH ASSISTANT AGENT TEST")
    print("=" * 72)
    print(f"REPORT = {path}")
    print(f"DATE   = {report.get('report_date')}")
    print("")

    agent = ResearchOrchestratorAgent(
        BASE,
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen3:8b",
    )

    notes = agent.run(report)

    print("Agent status:", notes.get("agent_status"))
    print("Model:", notes.get("model"))
    print("Report date:", notes.get("report_date"))
    print("")

    print("=== OVERVIEW ===")
    print(notes.get("overview", ""))
    print("")

    print("=== KEY TAKEAWAYS ===")
    for i, item in enumerate(notes.get("key_takeaways", []), 1):
        print(f"{i}. {item}")
    print("")

    print("=== RESEARCH NOTES ===")
    for i, item in enumerate(notes.get("research_notes", []), 1):
        confidence = item.get("confidence", 0)
        print(
            f"[{i}] {item.get('importance')} | {item.get('type')} | "
            f"{item.get('symbol')} | {item.get('title')}"
        )
        print("    NOTE:", item.get("note"))
        print("    TAGS:", "、".join(item.get("tags", [])))
        print("    EVIDENCE:", "；".join(item.get("evidence", [])))
        print("    FOLLOW-UP:", "；".join(item.get("follow_up", [])))
        print("    CONFIDENCE:", confidence)
        print("")

    print("=== AGENT ACTIONS ===")
    actions = notes.get("agent_actions", [])
    for item in actions:
        print("-", item.get("tool"), "|", item.get("reason"))

    print("")
    print("Agent tool action count:", len(actions))
    print("Memory:", REPORT_DIR / "agent_memory.json")

    if notes.get("agent_status") not in {"qwen3_tool_agent", "qwen3"}:
        print("ERROR: Qwen3 Agent 沒有成功完成自主研究。")
        return 2

    if not actions:
        print("ERROR: Agent 沒有留下任何工具操作紀錄，這不符合自主研究測試要求。")
        return 3

    if not notes.get("research_notes"):
        print("ERROR: Agent 沒有產生研究筆記。")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
