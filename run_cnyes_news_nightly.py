# -*- coding: utf-8 -*-
"""手動／相容入口：爬取前一日 00:00 至現在的鉅亨新聞（滾動取代舊快取）。排程請改用 run_finance_info_update.py。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from cnyes_news_crawler import CnyesNewsCrawler

BASE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="夜間 CNYES 當日新聞爬蟲")
    parser.add_argument("--force", dest="force", action="store_true", default=True, help="重新建立當日日檔（夜間排程預設開啟）")
    parser.add_argument("--no-force", dest="force", action="store_false", help="需要時才沿用已有日檔")
    parser.add_argument("--max-articles", type=int, default=int(os.getenv("CNYES_NIGHTLY_MAX_ARTICLES", "600")))
    args = parser.parse_args()

    crawler = CnyesNewsCrawler(
        BASE,
        max_articles=args.max_articles,
        max_scrolls=int(os.getenv("CNYES_NEWS_MAX_SCROLLS", "18")),
    )
    result = crawler.crawl_rolling(days=2, max_articles=args.max_articles)
    cache_path = result.get("cache_path") or (
        BASE / "output" / "research_reports" / f"cnyes_news_{result.get('crawl_date', '')}.json"
    )
    print("=" * 72)
    print("CNYES NIGHTLY NEWS CRAWLER")
    print("=" * 72)
    print("研究日期 =", result.get("crawl_date"))
    print("方法 =", result.get("source_method"))
    print("文章數 =", result.get("article_count"))
    print("檔案 =", cache_path)
    print("檔案存在 =", Path(cache_path).exists())
    print("時間窗 =", result.get("window_start"), "~", result.get("window_end"))
    if result.get("errors"):
        print("Errors:")
        for x in result["errors"]:
            print("-", x)
    return 0 if result.get("article_count", 0) or not result.get("errors") else 2


if __name__ == "__main__":
    raise SystemExit(main())
