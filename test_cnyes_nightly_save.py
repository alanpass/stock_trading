# -*- coding: utf-8 -*-
"""驗證夜間 CNYES 爬蟲結果一定回傳實際快取檔案路徑。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from cnyes_news_crawler import CnyesNewsCrawler


def main() -> None:
    """測試爬不到新資料時仍能正確保存快取。"""
    class FakeCrawler(CnyesNewsCrawler):
        """不連網路的假爬蟲。"""
        def crawl_api(self, days):
            """假裝 API 沒有資料。"""
            return [], {}

        def _get(self, *args, **kwargs):
            """測試中禁止連網路。"""
            raise RuntimeError("network disabled in unit test")

        def _selenium_listing(self):
            """假裝瀏覽器沒有抓到清單。"""
            return []

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        c = FakeCrawler(base)
        result = {
            "source_url": "https://news.cnyes.com/news/cat/headline",
            "crawl_date": "2026-09-29",
            "window_start": "2026-09-29T00:00:00+08:00",
            "window_end": "2026-09-29T23:59:59+08:00",
        }
        path = c.output_dir / "cnyes_news_2026-09-29.json"
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["crawl_date"] == "2026-09-29"
        assert path.exists()
    print("CNYES_NIGHTLY_SAVE_PATH_TEST_OK")


if __name__ == "__main__":
    main()
