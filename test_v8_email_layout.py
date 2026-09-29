# -*- coding: utf-8 -*-
"""v8 Email 版面驗收，不寄信、不需要 API。"""
from pathlib import Path
from email_agent import EmailAgent

BASE = Path(__file__).resolve().parent

def make_report():
    return {
        "report_type": "after_close",
        "report_date": "2026-09-29",
        "research_scope": {"watchlist_symbols": ["3044", "3006", "2408"]},
        "business_profiles": {
            "3044": {"name": "健鼎", "business_group": "高階PCB", "primary_chain": "PCB"},
            "3006": {"name": "晶豪科", "business_group": "記憶體IC", "primary_chain": "記憶體"},
            "2408": {"name": "南亞科", "business_group": "DRAM記憶體", "primary_chain": "記憶體"},
        },
        "market_movers": {
            "movers": [
                {"symbol": "3044", "name": "健鼎", "industry_name": "電子零組件業", "today_change_percent": 9.5, "ret_5d": 12.3},
                {"symbol": "3006", "name": "晶豪科", "industry_name": "半導體業", "today_change_percent": -5.2, "ret_5d": -1.1},
            ],
            "industry_summary": [
                {"industry_code": "28", "industry_name": "電子零組件業", "candidate_count": 5, "today_avg_change": 4.2, "ret_5d_avg": 6.1, "rising_count": 4, "falling_count": 1}
            ],
        },
        "bullish_themes": [
            {"theme": "CCL、高速材料與高階PCB", "strength": 54.9, "status": "成長題材"},
            {"theme": "AI／邊緣AI／Physical AI", "strength": 49.0, "status": "初步題材"},
        ],
        "financial_snapshots": [
            {"symbol": "3006", "latest_month_revenue": 7904796.0, "revenue_yoy": 605.24, "eps": 27.58},
            {"symbol": "2408", "latest_month_revenue": 44690268.0, "revenue_yoy": 560.85, "eps": 23.38},
        ],
        "earnings_calls": [],
        "agent_research_notes": {"overview": "市場出現明顯產業分化", "key_takeaways": ["高階PCB相關產業值得持續研究"], "positive_signals": ["健鼎今日+9.50%"], "negative_signals": ["晶豪科今日-5.20%"], "watch_items": ["持續觀察產業需求"], "research_notes": []},
        "official_sources": {"twse_announcements_rows": 1, "twse_revenue_rows": 2, "tpex_announcements_rows": 0, "tpex_revenue_rows": 0, "twse_income_rows": 2, "tpex_income_rows": 0, "cnyes_news_articles": 140},
    }


def make_morning():
    r = make_report()
    r["report_type"] = "morning"
    r["symbols"] = ["3044", "3006", "2408"]
    r["cnyes_news"] = {"cache_dates": ["2026-09-29", "2026-09-28"], "article_count": 140}
    r["morning_agent"] = {
        "agent_status": "qwen3_morning_agent",
        "overview": "近兩日 AI 與電子供應鏈新聞集中，早報聚焦高階 PCB 與記憶體需求。",
        "news_summary": [{"title": "AI 伺服器供應鏈動向", "summary": "近兩日新聞集中討論 AI 伺服器供應鏈需求。", "impact": "利多", "industries": ["高階PCB", "記憶體"], "evidence": ["鉅亨新聞 2026-09-29"], "source_links": [], "confidence": 0.8}],
        "recommended_industries": [{"industry": "高階PCB", "why": "近期新聞與產業鏈資料均有關聯。", "related_symbols": ["3044"]}],
        "recommended_stocks": [{"symbol": "3044", "name": "健鼎", "industry": "高階PCB", "reason": "公司業務屬高階PCB，與近期研究題材直接相關。", "evidence": ["公司業務資料"], "source_links": []}],
        "watch_items": [],
    }
    r["earnings_calls"] = []
    return r


a = EmailAgent(BASE)
after = a.build_html(make_report())
morning = a.build_morning_html(make_morning())
assert "③ 法說會訊號" not in after
assert "五、模型健康度" not in after
assert "原因候選" not in after
assert "產業／業務" in after
assert "健鼎（3044）" in morning
assert "2. 最近 2 天財經報導摘要" in morning
assert "3. Agent 看好的產業類別與股票" in morning
assert "4. Agent 觀察股票" not in morning
assert "3. Agent 觀察的產業類別" not in morning
print("V8_EMAIL_LAYOUT_OK")
