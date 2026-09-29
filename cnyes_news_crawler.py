# -*- coding: utf-8 -*-
"""
鉅亨網頭條新聞爬蟲（近 N 個日曆日）
============================================================

來源入口：
    https://news.cnyes.com/news/cat/headline

實作：
1. 優先呼叫鉅亨頭條頁面使用的公開新聞列表 API，帶 startAt/endAt 時間區間，
   因此可以可靠抓取「今天＋昨天」而不依賴無限滾動頁面的當前載入狀態。
2. API 回傳的新聞項目本身包含標題、正文、摘要、關鍵字、分類與 publishAt，
   直接保存為 ResearchAgent 的近期新聞資料庫。
3. API 失敗時回退到 HTML 頭條頁面；必要時再用 Selenium 取得動態載入的文章連結。
4. 每篇文章保存完整正文（去除 HTML 標籤）及來源 URL，給 Qwen3 Agent 後續摘要、查證與因果分析。
5. 不繞過登入、付費牆或其他存取控制；請遵守來源網站服務條款與合理請求頻率。
"""
from __future__ import annotations

import html as html_lib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://news.cnyes.com/news/cat/headline"
API_URL = "https://api.cnyes.com/media/api/v1/newslist/category/headline"
ARTICLE_PATTERN = re.compile(r"^https?://news\.cnyes\.com/news/id/\d+/?$")
TAIPEI = ZoneInfo("Asia/Taipei")
DEFAULT_DAYS = 2
DEFAULT_MAX_ARTICLES = 600
DEFAULT_API_PAGE_SIZE = 30
DEFAULT_MAX_API_PAGES = 40
DEFAULT_MAX_SCROLLS = 18
DEFAULT_DETAIL_WORKERS = 3
DEFAULT_TIMEOUT = 25
DEFAULT_SLEEP = 0.35
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TaiwanStockResearchAgent/2.0"


@dataclass
class CnyesArticle:
    article_id: str
    title: str
    url: str
    category: str
    published: str
    published_ts: str
    summary: str
    content: str
    tags: list[str]
    stock_refs: list[str]
    crawled_at: str
    source: str = "鉅亨網"
    source_type: str = "CNYES headline"


class CnyesNewsCrawler:
    def __init__(
        self,
        base_dir: str | Path = ".",
        timeout: int = DEFAULT_TIMEOUT,
        sleep_seconds: float = DEFAULT_SLEEP,
        max_articles: int = DEFAULT_MAX_ARTICLES,
        max_scrolls: int = DEFAULT_MAX_SCROLLS,
    ):
        self.base = Path(base_dir).resolve()
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.max_articles = max(30, int(max_articles))
        self.max_scrolls = max(3, int(max_scrolls))
        self.output_dir = self.base / "output" / "research_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": BASE_URL,
        })

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_text(value: Any, limit: int = 10000) -> str:
        text = html_lib.unescape(str(value or ""))
        text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit]

    @staticmethod
    def _article_id(url: str) -> str:
        m = re.search(r"/news/id/(\d+)", str(url))
        return m.group(1) if m else ""

    @staticmethod
    def _to_url(href: str) -> str:
        return urljoin(BASE_URL, str(href or "").strip()).split("#", 1)[0]

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            try:
                x = float(value)
                # CNYES API publishAt 是 Unix timestamp；容忍毫秒 timestamp。
                if x > 10_000_000_000:
                    x /= 1000.0
                return datetime.fromtimestamp(x, tz=TAIPEI)
            except Exception:
                return None

        text = str(value or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TAIPEI)
            return dt.astimezone(TAIPEI)
        except Exception:
            pass

        patterns = [
            (r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?", True),
            (r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", False),
            (r"(\d{4})(\d{2})(\d{2})[ T](\d{2})(\d{2})(\d{2})?", True),
        ]
        for pattern, has_time in patterns:
            m = re.search(pattern, text)
            if not m:
                continue
            vals = [int(x) if x is not None else 0 for x in m.groups()]
            try:
                if has_time:
                    y, mo, d, hh, mm = vals[:5]
                    ss = vals[5] if len(vals) > 5 else 0
                    return datetime(y, mo, d, hh, mm, ss, tzinfo=TAIPEI)
                y, mo, d = vals[:3]
                return datetime(y, mo, d, tzinfo=TAIPEI)
            except Exception:
                continue
        return None

    @staticmethod
    def _date_window(days: int) -> tuple[datetime, datetime]:
        now = datetime.now(TAIPEI)
        days = max(1, int(days))
        start = datetime(now.year, now.month, now.day, tzinfo=TAIPEI) - timedelta(days=days - 1)
        end = now
        return start, end

    def _within_window(self, value: Any, days: int) -> bool:
        dt = self._parse_datetime(value)
        if not dt:
            return False
        start, end = self._date_window(days)
        return start <= dt <= end + timedelta(minutes=2)

    def _get(self, url: str, params: dict[str, Any] | None = None, accept_json: bool = False) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                if accept_json:
                    r.encoding = r.apparent_encoding or r.encoding
                return r
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(0.9)
        raise RuntimeError(f"GET 失敗 {url}: {last_exc}")

    # ------------------------------------------------------------------
    # CNYES API：頁面實際使用的新聞列表介面
    # ------------------------------------------------------------------
    def _api_page(self, start: datetime, end: datetime, page: int, limit: int) -> dict[str, Any]:
        params = {
            "page": page,
            "limit": min(30, max(10, int(limit))),
            "isCategoryHeadline": 1,
            "startAt": int(start.timestamp()),
            "endAt": int(end.timestamp()),
        }
        response = self._get(API_URL, params=params, accept_json=True)
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"鉅亨 API JSON 解析失敗：{exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("鉅亨 API 回傳格式不是 JSON object。")
        return payload

    @staticmethod
    def _api_data(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        items = payload.get("items")
        if isinstance(items, dict):
            data = items.get("data")
            meta = items
        else:
            data = payload.get("data")
            meta = payload
        if not isinstance(data, list):
            data = []
        rows = [x for x in data if isinstance(x, dict)]
        return rows, meta if isinstance(meta, dict) else {}

    def _api_item_to_article(self, row: dict[str, Any]) -> CnyesArticle | None:
        article_id = str(row.get("newsId") or row.get("id") or "").strip()
        if not article_id:
            return None
        url = f"https://news.cnyes.com/news/id/{article_id}"
        title = self._clean_text(row.get("title", ""), 300)
        summary = self._clean_text(row.get("summary", ""), 1600)
        content = self._clean_text(row.get("content", ""), 7000)
        if not content:
            content = summary
        published_dt = self._parse_datetime(row.get("publishAt") or row.get("publishedAt") or row.get("published"))
        if not title:
            return None

        keywords = row.get("keyword") or row.get("keywords") or []
        if isinstance(keywords, str):
            tags = [x.strip() for x in re.split(r"[,，|;；]", keywords) if x.strip()]
        elif isinstance(keywords, list):
            tags = [str(x).strip() for x in keywords if str(x).strip()]
        else:
            tags = []
        tags = list(dict.fromkeys(tags))[:30]

        stock_refs = row.get("stock") or []
        if isinstance(stock_refs, str):
            stock_refs = [stock_refs]
        elif not isinstance(stock_refs, list):
            stock_refs = []
        stock_refs = [str(x).strip() for x in stock_refs if str(x).strip()][:30]

        return CnyesArticle(
            article_id=article_id,
            title=title,
            url=url,
            category=self._clean_text(row.get("categoryName") or row.get("category") or "頭條", 100) or "頭條",
            published=published_dt.strftime("%Y-%m-%d %H:%M:%S") if published_dt else self._clean_text(row.get("published", ""), 80),
            published_ts=published_dt.isoformat() if published_dt else "",
            summary=summary,
            content=content,
            tags=tags,
            stock_refs=stock_refs,
            crawled_at=datetime.now(TAIPEI).isoformat(),
        )

    def crawl_api(self, days: int) -> tuple[list[CnyesArticle], dict[str, Any]]:
        start, end = self._date_window(days)
        articles: list[CnyesArticle] = []
        errors: list[str] = []
        seen: set[str] = set()
        total = None
        last_page = None
        page = 1
        page_size = DEFAULT_API_PAGE_SIZE

        for _ in range(DEFAULT_MAX_API_PAGES):
            payload = self._api_page(start, end, page, page_size)
            rows, meta = self._api_data(payload)
            if total is None:
                total = meta.get("total")
            last_page = meta.get("last_page")
            if not rows:
                break

            oldest_dt: datetime | None = None
            for row in rows:
                article = self._api_item_to_article(row)
                if not article or article.article_id in seen:
                    continue
                seen.add(article.article_id)
                dt = self._parse_datetime(article.published_ts or article.published)
                if dt:
                    oldest_dt = dt if oldest_dt is None or dt < oldest_dt else oldest_dt
                if self._within_window(article.published_ts or article.published, days):
                    articles.append(article)

            # API 多半按照時間新→舊排序；一旦整頁最舊時間已早於視窗即可停止。
            if oldest_dt and oldest_dt < start:
                break
            if last_page is not None:
                try:
                    if page >= int(last_page):
                        break
                except Exception:
                    pass
            if len(articles) >= self.max_articles:
                break
            page += 1
            time.sleep(self.sleep_seconds)

        articles = sorted(
            articles,
            key=lambda x: x.published_ts or x.published or "",
            reverse=True,
        )[: self.max_articles]
        return articles, {
            "api_url": API_URL,
            "api_total": total,
            "api_last_page": last_page,
            "api_pages_fetched": page,
            "errors": errors,
        }

    def load_recent_cached(self, before_date: str | None = None, days: int = 2, max_search_days: int = 7) -> dict[str, Any]:
        """只讀取既有日檔快取，不進行任何網路請求。

        盤後研究應使用前一晚完成的日檔，因此預設從 report_date 的前一天開始，
        往前尋找最近的兩份可用日檔；遇到週末／缺檔時最多往前搜尋 max_search_days。
        """
        ref_dt = datetime.now(TAIPEI)
        if before_date:
            try:
                ref_dt = datetime.strptime(str(before_date)[:10], "%Y-%m-%d").replace(tzinfo=TAIPEI)
            except Exception:
                pass
        target = ref_dt.date()
        wanted = []
        for offset in range(1, max(1, int(max_search_days)) + 1):
            d = target - timedelta(days=offset)
            path = self.output_dir / f"cnyes_news_{d.isoformat()}.json"
            if not path.exists():
                continue
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("articles"), list):
                continue
            wanted.append((d.isoformat(), path, obj))
            if len(wanted) >= max(1, int(days)):
                break

        articles = []
        seen = set()
        errors = []
        for cache_date, path, obj in wanted:
            for article in obj.get("articles", []) or []:
                if not isinstance(article, dict):
                    continue
                # 優先以文章 URL 去重，避免不同日檔因邊界時間或重抓造成同一篇新聞重複。
                key = str(article.get("url") or article.get("article_id") or "").strip()
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                articles.append(article)
            for err in obj.get("errors", []) or []:
                errors.append(f"{cache_date}: {err}")

        articles.sort(key=lambda x: str(x.get("published_ts") or x.get("published") or ""), reverse=True)
        cache_dates = [x[0] for x in wanted]
        return {
            "source_url": BASE_URL,
            "source_method": "nightly_cache",
            "crawl_date": cache_dates[0] if cache_dates else "",
            "days": len(cache_dates),
            "cache_dates": cache_dates,
            "cache_files": [str(x[1]) for x in wanted],
            "window_start": cache_dates[-1] + " 00:00:00" if cache_dates else "",
            "window_end": cache_dates[0] + " 23:59:59" if cache_dates else "",
            "article_count": len(articles),
            "articles": articles,
            "errors": errors,
            "loaded_from_cache": True,
        }

    # ------------------------------------------------------------------
    # HTML/Selenium fallback
    # ------------------------------------------------------------------
    def _extract_links(self, html: str) -> list[dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            url = self._to_url(a.get("href"))
            if not ARTICLE_PATTERN.match(url) or url in seen:
                continue
            seen.add(url)
            rows.append({
                "url": url,
                "title_hint": self._clean_text(a.get_text(" ", strip=True), 300),
            })
        return rows

    def _selenium_listing(self) -> list[dict[str, str]]:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except Exception as exc:
            return [{"error": f"Selenium unavailable: {exc}"}]
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"--user-agent={USER_AGENT}")
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(45)
            driver.get(BASE_URL)
            time.sleep(1.2)
            links: dict[str, dict[str, str]] = {}
            previous_height = 0
            stable_rounds = 0
            for _ in range(self.max_scrolls):
                page_html = driver.page_source
                for row in self._extract_links(page_html):
                    links[row["url"]] = row
                current_height = driver.execute_script("return document.body.scrollHeight")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(max(self.sleep_seconds, 0.55))
                if current_height == previous_height:
                    stable_rounds += 1
                else:
                    stable_rounds = 0
                previous_height = current_height
                if len(links) >= self.max_articles or stable_rounds >= 3:
                    break
            return list(links.values())[: self.max_articles]
        except Exception as exc:
            return [{"error": f"Selenium listing failed: {exc}"}]
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def _parse_article_html(self, html: str, url: str, title_hint: str = "") -> CnyesArticle | None:
        soup = BeautifulSoup(html, "html.parser")
        title = ""
        for attrs in [
            {"property": "og:title"},
            {"name": "twitter:title"},
            {"name": "title"},
        ]:
            node = soup.find("meta", attrs=attrs)
            if node and node.get("content"):
                title = self._clean_text(node.get("content"), 300)
                break
        if not title:
            h1 = soup.find("h1")
            title = self._clean_text(h1.get_text(" ", strip=True), 300) if h1 else title_hint

        def meta(*pairs: tuple[str, str]) -> str:
            for attr, value in pairs:
                node = soup.find("meta", attrs={attr: value})
                if node and node.get("content"):
                    return str(node.get("content"))
            return ""

        published_raw = meta(
            ("property", "article:published_time"),
            ("name", "datePublished"),
            ("itemprop", "datePublished"),
        )
        category = meta(("property", "article:section"), ("name", "section"), ("name", "category"))
        description = meta(("property", "og:description"), ("name", "description"))
        tags_text = meta(("name", "keywords"), ("property", "article:tag"))

        published_dt = self._parse_datetime(published_raw)
        if not published_dt:
            time_node = soup.find("time", datetime=True)
            if time_node:
                published_dt = self._parse_datetime(time_node.get("datetime"))

        body_candidates = []
        for sel in [
            "article",
            "main article",
            "[class*='article-content']",
            "[class*='articleContent']",
            "[class*='post-content']",
            "[class*='content-body']",
            "main",
        ]:
            for node in soup.select(sel):
                text = self._clean_text(node.get_text(" ", strip=True), 12000)
                if len(text) >= 180:
                    body_candidates.append(text)
        content = max(body_candidates, key=len) if body_candidates else self._clean_text(description, 3500)
        if title and content.startswith(title):
            content = content[len(title):].strip()
        article_id = self._article_id(url)
        if not article_id or not title:
            return None
        tags = [x.strip() for x in re.split(r"[,，|;；]", tags_text) if x.strip()][:30]
        return CnyesArticle(
            article_id=article_id,
            title=title,
            url=url,
            category=self._clean_text(category, 100) or "頭條",
            published=published_dt.strftime("%Y-%m-%d %H:%M:%S") if published_dt else self._clean_text(published_raw, 80),
            published_ts=published_dt.isoformat() if published_dt else "",
            summary=self._clean_text(description, 1600),
            content=self._clean_text(content, 7000),
            tags=tags,
            stock_refs=[],
            crawled_at=datetime.now(TAIPEI).isoformat(),
        )

    def _fetch_detail(self, row: dict[str, str]) -> CnyesArticle | None:
        try:
            response = self._get(row.get("url", ""))
            return self._parse_article_html(response.text, row.get("url", ""), row.get("title_hint", ""))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # main crawl + cache
    # ------------------------------------------------------------------
    def crawl(self, days: int = DEFAULT_DAYS, force: bool = False, max_articles: int | None = None) -> dict[str, Any]:
        days = max(1, int(days))
        if max_articles is not None:
            self.max_articles = max(30, int(max_articles))
        today = datetime.now(TAIPEI).strftime("%Y-%m-%d")
        cache_path = self.output_dir / f"cnyes_news_{today}.json"
        if cache_path.exists() and not force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("articles") is not None:
                    return cached
            except Exception:
                pass

        errors: list[str] = []
        articles: list[CnyesArticle] = []
        source_method = "api"
        api_meta: dict[str, Any] = {}

        # 1) API：優先，能直接帶日期範圍並拿到正文。
        try:
            articles, api_meta = self.crawl_api(days)
            if not articles:
                errors.append("鉅亨 API 在指定日期範圍沒有回傳可用新聞，改用 HTML fallback。")
        except Exception as exc:
            errors.append(f"鉅亨 API 爬蟲失敗：{exc}")
            articles = []

        # 2) HTML fallback：API 暫時異常時保留原專案可用性。
        listing_link_count = 0
        if not articles:
            source_method = "html"
            listing: list[dict[str, str]] = []
            try:
                response = self._get(BASE_URL)
                listing = self._extract_links(response.text)
            except Exception as exc:
                errors.append(f"鉅亨頭條 HTML 讀取失敗：{exc}")

            if len(listing) < min(30, self.max_articles):
                selenium_rows = self._selenium_listing()
                if selenium_rows and not selenium_rows[0].get("error"):
                    listing.extend(selenium_rows)
                elif selenium_rows and selenium_rows[0].get("error"):
                    errors.append(str(selenium_rows[0].get("error")))

            dedup_listing = {row.get("url"): row for row in listing if ARTICLE_PATTERN.match(row.get("url", ""))}
            listing = list(dedup_listing.values())
            listing_link_count = len(listing)
            with ThreadPoolExecutor(max_workers=DEFAULT_DETAIL_WORKERS) as pool:
                futures = [pool.submit(self._fetch_detail, row) for row in listing[: self.max_articles]]
                for future in as_completed(futures):
                    article = future.result()
                    if article and self._within_window(article.published_ts or article.published, days):
                        articles.append(article)
                    if len(articles) >= self.max_articles:
                        break

        dedup: dict[str, CnyesArticle] = {}
        for article in articles:
            dedup[article.article_id] = article
        articles = sorted(dedup.values(), key=lambda x: x.published_ts or x.published or "", reverse=True)[: self.max_articles]

        categories: dict[str, int] = {}
        for x in articles:
            categories[x.category] = categories.get(x.category, 0) + 1

        start, end = self._date_window(days)
        result = {
            "source_url": BASE_URL,
            "api_url": API_URL,
            "crawl_date": today,
            "crawl_started_at": datetime.now(TAIPEI).isoformat(),
            "days": days,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "source_method": source_method,
            "article_count": len(articles),
            "listing_link_count": listing_link_count,
            "category_counts": dict(sorted(categories.items(), key=lambda x: (-x[1], x[0]))),
            "api_meta": api_meta,
            "articles": [asdict(x) for x in articles],
            "errors": errors,
        }
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CNYES 新聞爬蟲")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-articles", type=int, default=DEFAULT_MAX_ARTICLES)
    args = parser.parse_args()

    crawler = CnyesNewsCrawler(Path(__file__).resolve().parent, max_articles=args.max_articles)
    result = crawler.crawl(days=args.days, force=args.force)
    print("=" * 72)
    print("CNYES NEWS CRAWLER")
    print("=" * 72)
    print(f"SOURCE       = {result.get('source_url')}")
    print(f"METHOD       = {result.get('source_method')}")
    print(f"WINDOW       = {result.get('window_start')} ~ {result.get('window_end')}")
    print(f"ARTICLES     = {result.get('article_count')}")
    print(f"CATEGORIES   = {result.get('category_counts')}")
    if result.get("errors"):
        print("ERRORS:")
        for err in result["errors"]:
            print("-", err)
    for item in result.get("articles", [])[:10]:
        print(f"- {item.get('published')} | {item.get('category')} | {item.get('title')}")
        print(f"  {item.get('url')}")
