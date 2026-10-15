# -*- coding: utf-8 -*-
"""以真正的 MCP stdio 連線測試 tw-stock-research server。

執行：python test_mcp_server.py [YYYY-MM-DD]   （日期需有 output/claude_jobs/afterclose_<date>/package.json）
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "get_quote", "get_price_history", "analyze_entry", "audit_entry_model", "get_institutional_flow", "get_holder_structure", "get_market_status",
    "research_market_overview", "research_company_financial", "research_company_profile", "research_earnings_events",
    "research_read_fugle_memo", "research_news_batch", "research_search_news", "research_search_live_news",
    "research_theme_evidence", "research_previous_notes", "pipeline_status", "get_job_brief", "get_stock_note",
    "validate_afterclose_notes", "validate_morning_notes", "validate_finance_digest", "validate_earnings_memo",
}


def _payload(result) -> object:
    if getattr(result, "structured_content", None):
        return result.structured_content
    text = "".join(getattr(c, "text", "") for c in result.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def main(report_date: str) -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server" / "tw_stock_server.py")],
                                   env={"PYTHONIOENCODING": "utf-8"}, cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            missing = EXPECTED - tools
            print(f"TOOLS {len(tools)}  missing={sorted(missing)}")
            assert not missing

            calls = [
                ("get_market_status", {}),
                ("get_quote", {"symbol": "2330"}),
                ("analyze_entry", {"symbol": "2330"}),
                ("research_market_overview", {"report_date": report_date}),
                ("research_search_news", {"query": "台積電 法說", "limit": 2, "report_date": report_date}),
                ("pipeline_status", {"report_date": report_date}),
                ("validate_afterclose_notes", {"report_date": report_date}),
            ]
            for name, args in calls:
                result = await session.call_tool(name, args)
                data = _payload(result)
                assert not result.is_error, f"{name} 失敗：{str(data)[:300]}"
                summary = json.dumps(data, ensure_ascii=False, default=str)[:160]
                print(f"OK {name}: {summary}")
    print("MCP_SERVER_OK")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "2026-10-14")))
