# -*- coding: utf-8 -*-
"""V7 CNYES pipeline unit test：快取 -> 新聞 Agent -> 主 Agent merge -> Email 顯示。"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def fake_ollama_module():
    class Client:
        def __init__(self, host=None):
            self.host = host

        def chat(self, **kwargs):
            text = json.dumps({
                "overview": "電子零組件近期受到載板與AI伺服器相關新聞關注。",
                "key_findings": [{
                    "title": "載板需求受到 AI 伺服器拉動",
                    "summary": "近兩日鉅亨新聞集中提到高階載板與AI伺服器需求。",
                    "why_relevant": "可作為電子零組件產業異動的新聞研究線索。",
                    "impact": "利多",
                    "related_symbols": ["6274"],
                    "evidence": ["鉅亨新聞：載板需求受到 AI 伺服器拉動"],
                    "source_links": ["https://news.cnyes.com/news/id/1"],
                    "confidence": 0.82,
                }],
                "market_drivers": ["AI伺服器需求與高階載板題材"],
                "watch_topics": ["ABF/BT載板後續需求"]
            }, ensure_ascii=False)
            return SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=[]))

    return SimpleNamespace(Client=Client)


def main():
    sys.modules["ollama"] = fake_ollama_module()
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        out = base / "output" / "research_reports"
        out.mkdir(parents=True)
        for d in ("2026-09-27", "2026-09-28"):
            payload = {
                "source_url": "https://news.cnyes.com/news/cat/headline",
                "source_method": "api",
                "crawl_date": d,
                "article_count": 1,
                "articles": [{
                    "article_id": d.replace("-", ""),
                    "title": "載板需求受到 AI 伺服器拉動",
                    "url": "https://news.cnyes.com/news/id/1",
                    "category": "台股",
                    "published": d + " 22:00:00",
                    "published_ts": d + "T22:00:00+08:00",
                    "summary": "近兩日鉅亨新聞集中提到高階載板與AI伺服器需求。",
                    "content": "高階載板與AI伺服器需求受到市場關注。",
                    "tags": ["載板", "AI"],
                    "stock_refs": [],
                    "crawled_at": d + "T23:00:00+08:00",
                    "source": "鉅亨網",
                    "source_type": "CNYES headline",
                }],
                "errors": []
            }
            (out / f"cnyes_news_{d}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        from cnyes_news_crawler import CnyesNewsCrawler
        from cnyes_news_agent import CnyesNewsDigestAgent
        crawler = CnyesNewsCrawler(base)
        cache = crawler.load_recent_cached(before_date="2026-09-29", days=2)
        assert cache["cache_dates"] == ["2026-09-28", "2026-09-27"], cache
        assert cache["article_count"] == 1, cache

        research_payload = {
            "report_date": "2026-09-29",
            "symbols": ["6274"],
            "market_movers": {"movers": [{"symbol": "6274", "name": "台燿", "today_change_percent": 5.0}]},
            "cnyes_news": cache,
        }
        digest = CnyesNewsDigestAgent(base).run(research_payload, limit=40)
        assert digest["agent_status"] == "qwen3_cnyes_news_agent", digest
        assert digest["key_findings"], digest
        assert digest["cache_dates"] == ["2026-09-28", "2026-09-27"], digest

    print("CNYES_V7_CACHE_AGENT_PIPELINE_OK")


if __name__ == "__main__":
    main()
