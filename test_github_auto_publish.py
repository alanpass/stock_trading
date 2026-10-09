# -*- coding: utf-8 -*-
"""不觸發 GitHub push 的發布功能 smoke test。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from publish_research_data import build_public_payload


def main() -> int:
    sample = {
        "report_type": "finance_info",
        "report_date": "2026-10-08",
        "updated_at": "2026-10-08T18:00:00+08:00",
        "finished_at": "2026-10-08T18:05:00+08:00",
        "lookback_days": 2,
        "schedule": ["08:10", "11:00", "13:30", "16:00", "18:00", "23:00"],
        "stale": False,
        "news": [{
            "article_id": "123",
            "title": "測試新聞",
            "url": "https://example.com/news/123",
            "content": "這一大段正文不應該被公開發布。" * 100,
            "summary": "測試摘要",
            "ai_points": ["AI 測試重點"],
            "sentiment": "利多",
            "sectors": ["半導體"],
            "stocks": ["2330"],
            "highlights": ["AI"],
            "why": "測試",
            "importance": 8.2,
        }],
        "earnings": [{
            "title": "測試法說會",
            "symbol": "2330",
            "name": "測試公司",
            "event_date": "2026-10-08",
            "content": "法說會正文不應該被公開發布。" * 100,
            "ai_points": ["法說重點"],
            "sentiment": "中性",
            "source_url": "https://example.com/earnings/2330",
        }],
        "news_digest": {"headline": "測試總覽", "stats": {"total": 1}},
        "earnings_digest": {"headline": "測試法說總覽"},
        "crawl": {"ok": True},
        "agent_stats": {"llm_summarized": 1},
        "errors": [],
    }
    public = build_public_payload(sample)
    text = json.dumps(public, ensure_ascii=False)
    assert "正文不應該被公開發布" not in text
    assert "法說會正文不應該被公開發布" not in text
    assert public["news_count"] == 1
    assert public["earnings_count"] == 1
    assert public["news"][0]["ai_points"] == ["AI 測試重點"]
    print("OK: public finance payload only contains approved summary fields.")
    print(f"Serialized size: {len(text) / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
