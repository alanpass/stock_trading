# -*- coding: utf-8 -*-
"""CNYES API/HTML parser + Agent tool smoke test（不需連外網/Ollama）。"""
from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from cnyes_news_crawler import CnyesNewsCrawler
from agent_research_assistant import ResearchOrchestratorAgent


def main() -> None:
    with TemporaryDirectory() as td:
        crawler = CnyesNewsCrawler(td)

        sample_payload = {
            "items": {
                "total": 2,
                "per_page": 30,
                "current_page": 1,
                "last_page": 1,
                "data": [
                    {
                        "newsId": 123456,
                        "title": "電子零組件族群上漲 載板需求成市場焦點",
                        "summary": "市場關注 PCB、IC 載板與 ABF 需求。",
                        "content": "近期市場聚焦 AI 伺服器需求與高階載板，電子零組件相關個股同步受到關注。",
                        "keyword": "PCB,ABF,載板",
                        "publishAt": 1782266520,
                        "categoryName": "台股",
                        "stock": ["3037", "3044"],
                    }
                ],
            }
        }
        rows, meta = crawler._api_data(sample_payload)
        assert len(rows) == 1
        assert meta["total"] == 2
        article = crawler._api_item_to_article(rows[0])
        assert article is not None
        assert article.article_id == "123456"
        assert "載板" in article.content
        assert article.category == "台股"
        assert article.stock_refs == ["3037", "3044"]

        html = """<!doctype html><html><head>
        <meta property="og:title" content="電子零組件族群上漲 載板需求成市場焦點">
        <meta property="article:published_time" content="2026-09-28T10:02:00+08:00">
        <meta property="article:section" content="台股">
        <meta property="og:description" content="市場關注 PCB、IC 載板與 ABF 需求。">
        </head><body><article>近期市場聚焦 AI 伺服器需求與高階載板。</article></body></html>"""
        html_article = crawler._parse_article_html(
            html, "https://news.cnyes.com/news/id/654321", ""
        )
        assert html_article is not None
        assert html_article.article_id == "654321"
        assert html_article.category == "台股"
        assert html_article.published.startswith("2026-09-28")

        agent = ResearchOrchestratorAgent(td)
        payload = {
            "cnyes_news": {
                "source_url": "https://news.cnyes.com/news/cat/headline",
                "crawl_date": "2026-09-28",
                "days": 2,
                "article_count": 1,
                "category_counts": {"台股": 1},
                "window_start": "2026-09-27T00:00:00+08:00",
                "window_end": "2026-09-28T16:00:00+08:00",
                "articles": [{
                    "article_id": article.article_id,
                    "title": article.title,
                    "url": article.url,
                    "category": article.category,
                    "published": article.published,
                    "published_ts": article.published_ts,
                    "summary": article.summary,
                    "content": article.content,
                    "tags": ["PCB", "ABF", "載板"],
                    "source": "鉅亨網",
                    "source_type": "CNYES headline",
                }],
                "errors": [],
            }
        }
        overview = agent._tool("get_cnyes_news_overview", {}, payload)
        assert overview["article_count"] == 1
        result = agent._tool(
            "search_cnyes_news",
            {"query": "電子零組件 上漲 原因 載板 ABF CCL", "limit": 12},
            payload,
        )
        assert result["match_count"] >= 1
        batch = agent._tool("get_cnyes_news_batch", {"offset": 0, "limit": 50}, payload)
        assert batch["total"] == 1
        assert "content" not in batch["articles"][0]
        print("CNYES_CRAWLER_TOOL_TEST_OK")
        print(json.dumps({"overview": overview, "search": result, "batch": batch}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
