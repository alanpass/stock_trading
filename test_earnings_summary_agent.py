# -*- coding: utf-8 -*-
"""離線驗證：法說會 Agent（規則層，不需網路／Ollama）。執行：python test_earnings_summary_agent.py"""
import os, tempfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
os.environ["FINANCE_USE_LLM"] = "false"
from earnings_summary_agent import EarningsSummaryAgent

T = ZoneInfo("Asia/Taipei")
today = datetime.now(T).date()
items = [
    {"url": "u1", "symbol": "2330", "name": "台積電", "title": "台積電法說會", "published_date": today.isoformat(),
     "impact": "利多", "confidence": 80, "one_line_summary": "第三季營收年增 30%，毛利率 59%，AI 需求強勁。",
     "guidance": ["第四季營收預期季增 5%，全年資本支出上修至 420 億美元"], "key_risks": ["關稅與匯率波動"],
     "financial_highlights": ["營收 9,000 億元，EPS 15.2 元"], "in_watchlist": True},
    {"url": "u2", "symbol": "2603", "name": "長榮", "title": "長榮法說會", "published_date": (today - timedelta(days=4)).isoformat(),
     "impact": "利空", "one_line_summary": "運價下滑，獲利衰退，下修全年展望。", "key_risks": ["運價持續下滑"]},
    {"url": "u3", "symbol": "1101", "name": "台泥", "title": "台泥法說會", "published_date": (today - timedelta(days=1)).isoformat(),
     "memo_text": "水泥需求持平。" * 40},
]
with tempfile.TemporaryDirectory() as d:
    a = EarningsSummaryAgent(d)
    rows, dg = a.analyze_earnings(items, days=5)
    assert len(rows) == 3
    assert "最近 5 天共 3 場" in dg["headline"] and "2 天" not in dg["headline"], dg["headline"]
    top = rows[0]
    assert top["symbol"] == "2330" and top["sentiment"] == "利多" and top["ai_points"] and top["highlights"]
    assert any("%" in h or "營收" in h for h in top["highlights"]), top["highlights"]
    assert {r["symbol"]: r["sentiment"] for r in rows}["2603"] == "利空"   # 沿用法說 Agent 的 impact
    assert dg["利多"] == 1 and dg["利空"] == 1 and dg["key_points"] and dg["risks"]
    empty_rows, empty = a.analyze_earnings([], days=5)
    assert empty_rows == [] and "最近 5 天沒有可用" in empty["headline"]
    print(dg["headline"]); print(top["ai_points"]); print(top["highlights"]); print(dg["risks"])
print("OK")
