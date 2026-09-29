# -*- coding: utf-8 -*-
"""每天 23:00 執行：只負責爬取『當天』鉅亨頭條新聞並建立日檔快取。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from cnyes_news_crawler import CnyesNewsCrawler

BASE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="夜間 CNYES 當日新聞爬蟲")
    parser.add_argument("--force", action="store_true", help="忽略當日日檔並重新爬取")
    parser.add_argument("--max-articles", type=int, default=int(os.getenv("CNYES_NIGHTLY_MAX_ARTICLES", "600")))
    args = parser.parse_args()

    crawler = CnyesNewsCrawler(
        BASE,
        max_articles=args.max_articles,
        max_scrolls=int(os.getenv("CNYES_NEWS_MAX_SCROLLS", "18")),
    )
    result = crawler.crawl(days=1, force=args.force, max_articles=args.max_articles)
    print("=" * 72)
    print("CNYES NIGHTLY NEWS CRAWLER")
    print("=" * 72)
    print("研究日期 =", result.get("crawl_date"))
    print("方法 =", result.get("source_method"))
    print("文章數 =", result.get("article_count"))
    print("檔案 =", result.get("cache_path"))
    print("時間窗 =", result.get("window_start"), "~", result.get("window_end"))
    if result.get("errors"):
        print("Errors:")
        for x in result["errors"]:
            print("-", x)
    return 0 if result.get("article_count", 0) or not result.get("errors") else 2


if __name__ == "__main__":
    raise SystemExit(main())
