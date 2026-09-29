# -*- coding: utf-8 -*-
"""v6 Email 版面回歸測試：不顯示法說訊號/模型健康度，恢復題材與營收重點。"""
from __future__ import annotations

from email_agent import EmailAgent


def main() -> None:
    agent = EmailAgent(".")
    report = {
        "report_date": "2026-09-28",
        "research_scope": {"watchlist_symbols": ["3006", "2408", "8299"]},
        "market_status": "盤後測試",
        "research_conclusion": "測試研究完成",
        "business_profiles": {
            "3006": {"name": "晶豪科"},
            "2408": {"name": "南亞科"},
            "8299": {"name": "群聯"},
        },
        "market_movers": {"movers": [], "industry_summary": []},
        "earnings_calls": [],
        "bullish_themes": [
            {"theme": "CCL、高速材料與高階PCB", "strength": 55.4, "status": "成長題材"},
            {"theme": "AI／邊緣AI／Physical AI", "strength": 55.0, "status": "成長題材"},
            {"theme": "先進製程與AI ASIC", "strength": 46.4, "status": "初步題材"},
        ],
        "financial_snapshots": [
            {"symbol": "3006", "latest_month_revenue": 7904796.0, "revenue_yoy": 605.24, "eps": 27.58},
            {"symbol": "2408", "latest_month_revenue": 44690268.0, "revenue_yoy": 560.85, "eps": 23.38},
            {"symbol": "8299", "latest_month_revenue": 28275661.0, "revenue_yoy": 376.52, "eps": 0.27},
        ],
        "official_sources": {"cnyes_news_articles": 123},
        "agent_research_notes": {
            "agent_status": "qwen3_tool_agent",
            "overview": "電子零組件近期強勢需要用新聞進一步解釋。",
            "key_takeaways": ["測試"],
            "positive_signals": ["載板題材"],
            "negative_signals": [],
            "watch_items": [],
            "research_notes": [{
                "importance": "high", "type": "industry", "symbol": "3006",
                "title": "電子零組件產業異動", "note": "測試新聞因果鏈。",
                "evidence": ["鉅亨新聞"], "tags": ["載板", "待驗證"],
                "confidence": 65, "follow_up": ["再查官方資料"],
                "source_links": ["https://news.cnyes.com/news/cat/headline"],
            }],
            "cnyes_news_digest": ["鉅亨近兩日新聞顯示載板相關題材受到市場關注。"],
        },
    }
    html = agent.build_html(report)
    text = agent.build_text(report)
    assert "③ 法說會訊號" not in html
    assert "五、模型健康度" not in html
    assert "模型健康度：" not in html
    assert "④ 目前主要利多題材" in html
    assert "⑤ 營收成長重點" in html
    assert "晶豪科（3006）" in html
    assert "南亞科（2408）" in html
    assert "群聯（8299）" in html
    assert "Agent 研究筆記" in html
    assert "鉅亨近兩日新聞｜Agent 研究摘要" in html
    assert "法說會訊號" not in html
    assert "五、模型健康度" not in text
    assert "模型健康度：" not in text
    assert "④ 目前主要利多題材" in html and "④ 目前主要利多題材" in text
    assert "⑤ 營收成長重點" in html and "⑤ 營收成長重點" in text
    assert "晶豪科（3006）" in text
    print("EMAIL_LAYOUT_V6_TEST_OK")


if __name__ == "__main__":
    main()
