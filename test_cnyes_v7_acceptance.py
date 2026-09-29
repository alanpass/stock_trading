# -*- coding: utf-8 -*-
"""V7 最終驗收：新聞研究不會因主 Agent fallback 消失，Email 不含重複舊區塊。"""
from __future__ import annotations

from agent_research_assistant import _merge_cnyes_digest
from email_agent import EmailAgent


def main():
    digest = {
        "agent_status": "qwen3_cnyes_news_agent",
        "overview": "電子零組件新聞集中在高階載板與 AI 伺服器需求。",
        "key_findings": [{
            "title": "高階載板需求受到 AI 伺服器帶動",
            "summary": "近兩日新聞聚焦高階載板與 AI 伺服器需求。",
            "why_relevant": "可作為電子零組件產業異動的市場研究線索。",
            "impact": "利多",
            "related_symbols": ["6274"],
            "evidence": ["鉅亨：高階載板需求受到 AI 伺服器帶動"],
            "source_links": ["https://news.cnyes.com/news/id/123"],
            "confidence": 0.82,
        }],
    }
    result = _merge_cnyes_digest({
        "research_notes": [],
        "key_takeaways": [],
        "cnyes_news_digest": [],
    }, {"cnyes_research_digest": digest})
    assert result["research_notes"], result
    assert "新聞研究｜高階載板需求受到 AI 伺服器帶動" in result["research_notes"][0]["title"], result
    assert result["cnyes_news_digest"], result

    report = {
        "report_date": "2026-09-29",
        "research_scope": {"watchlist_symbols": ["6274"]},
        "agent_research_notes": result,
        "cnyes_news": {"cache_dates": ["2026-09-28", "2026-09-27"], "article_count": 161},
        "cnyes_research_digest": digest,
        "bullish_themes": [],
        "financial_snapshots": [],
        "earnings_calls": [],
        "market_movers": {"movers": []},
        "official_sources": {"cnyes_news_articles": 161},
    }
    html = EmailAgent(".").build_html(report)
    assert "高階載板需求受到 AI 伺服器帶動" in html
    assert "Agent 研究筆記" in html
    assert "③ 法說會訊號" not in html
    assert "五、模型健康度" not in html
    print("CNYES_V7_ACCEPTANCE_OK")


if __name__ == "__main__":
    main()
