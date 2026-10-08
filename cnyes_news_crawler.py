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
import os
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
CATEGORY_API_URL = "https://api.cnyes.com/media/api/v1/newslist/category/{category}"
# 大、小新聞都要：頭條 + 台股 + 國際股 + 外匯 + 期貨（可用環境變數 CNYES_CATEGORIES 覆寫；不存在的分類會被略過）
DEFAULT_CATEGORIES = ("headline", "tw_stock", "wd_stock", "forex", "future")
KEEP_DAILY_FILES = 10
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
    content_length: int = 0
    summary_length: int = 0
    detail_enriched: bool = False
    detail_enriched_at: str = ""
    list_category: str = "headline"


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
    def _api_page(self, start: datetime, end: datetime, page: int, limit: int, category: str = "headline") -> dict[str, Any]:
        params = {
            "page": page,
            "limit": min(30, max(10, int(limit))),
            "startAt": int(start.timestamp()),
            "endAt": int(end.timestamp()),
        }
        if category == "headline":
            params["isCategoryHeadline"] = 1
        url = API_URL if category == "headline" else CATEGORY_API_URL.format(category=category)
        response = self._get(url, params=params, accept_json=True)
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
            content_length=len(content),
            summary_length=len(summary),
        )

    def crawl_api(self, days: int, category: str = "headline") -> tuple[list[CnyesArticle], dict[str, Any]]:
        start, end = self._date_window(days)
        articles: list[CnyesArticle] = []
        errors: list[str] = []
        seen: set[str] = set()
        total = None
        last_page = None
        page = 1
        page_size = DEFAULT_API_PAGE_SIZE

        for _ in range(DEFAULT_MAX_API_PAGES):
            payload = self._api_page(start, end, page, page_size, category)
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
                article.list_category = category
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
            "api_url": API_URL if category == "headline" else CATEGORY_API_URL.format(category=category),
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
            content_length=len(self._clean_text(content, 7000)),
            summary_length=len(self._clean_text(description, 1600)),
        )

    def _fetch_detail(self, row: dict[str, str]) -> CnyesArticle | None:
        try:
            response = self._get(row.get("url", ""))
            return self._parse_article_html(response.text, row.get("url", ""), row.get("title_hint", ""))
        except Exception:
            return None

    @staticmethod
    def _content_quality(article: dict[str, Any]) -> int:
        """估算新聞正文品質；API 只有摘要時，分數通常很低。"""
        content = str(article.get("content") or "").strip()
        summary = str(article.get("summary") or "").strip()
        return max(len(content), len(summary))

    def enrich_articles(
        self,
        articles: list[dict[str, Any]],
        max_articles: int | None = None,
        workers: int = DEFAULT_DETAIL_WORKERS,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """用文章頁補齊 API 缺少的正文。

        只對正文過短／只有標題摘要的新聞補抓，避免每次更新都重抓全部新聞。
        失敗時保留原資料，不讓單篇新聞影響整批資料。
        """
        rows = [dict(x) for x in articles if isinstance(x, dict)]
        limit = max_articles if max_articles is not None else len(rows)
        candidates = [
            x for x in rows
            if str(x.get("url") or "").strip()
            and self._content_quality(x) < 320
        ][:max(0, int(limit))]
        errors: list[str] = []
        if not candidates:
            return rows, errors

        def fetch_one(row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
            try:
                detail = self._fetch_detail({
                    "url": str(row.get("url") or ""),
                    "title_hint": str(row.get("title") or ""),
                })
                if detail and self._content_quality(asdict(detail)) >= self._content_quality(row):
                    merged = dict(row)
                    for key in ("title", "category", "published", "published_ts", "summary", "content", "tags", "stock_refs", "crawled_at", "source", "source_type"):
                        value = getattr(detail, key, None)
                        if value not in (None, "", [], {}):
                            merged[key] = value
                    merged["content_length"] = len(str(merged.get("content") or ""))
                    merged["summary_length"] = len(str(merged.get("summary") or ""))
                    merged["detail_enriched"] = True
                    merged["detail_enriched_at"] = datetime.now(TAIPEI).isoformat()
                    return merged, None
                return dict(row), None
            except Exception as exc:
                return dict(row), f"{row.get('url')}: {type(exc).__name__}: {exc}"

        index = {str(row.get("url") or row.get("article_id") or ""): i for i, row in enumerate(rows)}
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(fetch_one, row) for row in candidates]
            for future in as_completed(futures):
                merged, error = future.result()
                key = str(merged.get("url") or merged.get("article_id") or "")
                if key in index:
                    rows[index[key]] = merged
                if error:
                    errors.append(error)

        return rows, errors


    # ------------------------------------------------------------------
    # 滾動式爬取：每次都抓「前一日 00:00 ～ 現在」，爬完直接取代上一個時間點的資料
    # ------------------------------------------------------------------
    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _load_prior_articles(self) -> dict[str, dict[str, Any]]:
        """讀上一次的滾動快取，已補齊正文的文章不必重抓內文頁。"""
        path = self.output_dir / "cnyes_news_latest.json"
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            return {
                str(a.get("article_id") or a.get("url")): a
                for a in obj.get("articles", []) if isinstance(a, dict)
            }
        except Exception:
            return {}

    def _html_fallback_articles(self, days: int, errors: list[str]) -> list[CnyesArticle]:
        listing: list[dict[str, str]] = []
        try:
            response = self._get(BASE_URL)
            listing = self._extract_links(response.text)
        except Exception as exc:
            errors.append(f"鉅亨頭條 HTML 讀取失敗：{exc}")
        if len(listing) < min(30, self.max_articles):
            rows = self._selenium_listing()
            if rows and not rows[0].get("error"):
                listing.extend(rows)
            elif rows and rows[0].get("error"):
                errors.append(str(rows[0].get("error")))
        dedup = {row.get("url"): row for row in listing if ARTICLE_PATTERN.match(row.get("url", ""))}
        out: list[CnyesArticle] = []
        with ThreadPoolExecutor(max_workers=DEFAULT_DETAIL_WORKERS) as pool:
            futures = [pool.submit(self._fetch_detail, row) for row in list(dedup.values())[: self.max_articles]]
            for future in as_completed(futures):
                article = future.result()
                if article and self._within_window(article.published_ts or article.published, days):
                    out.append(article)
        return out

    def crawl_rolling(
        self,
        days: int = 2,
        categories: tuple[str, ...] | list[str] | None = None,
        max_articles: int | None = None,
    ) -> dict[str, Any]:
        """爬取最近 days 個日曆日（預設：前一日 00:00 至現在）的新聞並取代舊快取。

        輸出：
        - cnyes_news_latest.json：本次完整結果（財經資訊頁、晨報、盤後都讀這份）
        - cnyes_news_YYYY-MM-DD.json：依發佈日切成日檔（相容舊程式），同日檔直接被取代
        """
        days = max(1, int(days))
        if max_articles is not None:
            self.max_articles = max(30, int(max_articles))
        if categories is None:
            env = os.getenv("CNYES_CATEGORIES", "").strip()
            categories = [x.strip() for x in env.split(",") if x.strip()] or list(DEFAULT_CATEGORIES)

        errors: list[str] = []
        per_category: dict[str, int] = {}
        merged: dict[str, CnyesArticle] = {}
        api_meta: dict[str, Any] = {}
        for cat in categories:
            try:
                rows, meta = self.crawl_api(days, category=cat)
            except Exception as exc:
                errors.append(f"[{cat}] API 失敗：{exc}")
                per_category[cat] = 0
                continue
            per_category[cat] = len(rows)
            api_meta[cat] = {k: meta.get(k) for k in ("api_total", "api_last_page", "api_pages_fetched")}
            for a in rows:
                if a.article_id not in merged:
                    merged[a.article_id] = a
            time.sleep(self.sleep_seconds)

        source_method = "api"
        if not merged:
            source_method = "html"
            errors.append("鉅亨 API 在指定日期範圍沒有回傳新聞，改用 HTML fallback。")
            for a in self._html_fallback_articles(days, errors):
                merged[a.article_id] = a

        articles = sorted(merged.values(), key=lambda x: x.published_ts or x.published or "", reverse=True)[: self.max_articles]
        rows = [asdict(x) for x in articles]

        # 沿用上一次已補齊的正文，只對新文章補抓內文頁
        prior = self._load_prior_articles()
        reused = 0
        for row in rows:
            old = prior.get(str(row.get("article_id") or row.get("url")))
            if old and self._content_quality(old) > self._content_quality(row):
                for key in ("content", "summary", "tags", "stock_refs", "detail_enriched", "detail_enriched_at", "content_length", "summary_length"):
                    if old.get(key) not in (None, "", [], {}):
                        row[key] = old[key]
                reused += 1
        enrich_limit = int(os.getenv("CNYES_DETAIL_ENRICH_MAX", "150"))
        detail_errors: list[str] = []
        if enrich_limit > 0 and rows:
            rows, detail_errors = self.enrich_articles(rows, max_articles=enrich_limit)

        now = datetime.now(TAIPEI)
        start, end = self._date_window(days)
        by_date: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            dt = self._parse_datetime(row.get("published_ts") or row.get("published"))
            by_date.setdefault(dt.date().isoformat() if dt else now.date().isoformat(), []).append(row)

        categories_count: dict[str, int] = {}
        for row in rows:
            categories_count[row.get("category") or "未分類"] = categories_count.get(row.get("category") or "未分類", 0) + 1

        result = {
            "source_url": BASE_URL,
            "source_method": source_method,
            "crawl_date": now.date().isoformat(),
            "crawl_started_at": now.isoformat(),
            "days": days,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "categories": list(categories),
            "per_category_count": per_category,
            "api_meta": api_meta,
            "article_count": len(rows),
            "category_counts": dict(sorted(categories_count.items(), key=lambda x: (-x[1], x[0]))),
            "reused_from_previous_run": reused,
            "articles": rows,
            "errors": errors + detail_errors,
            "content_quality": {
                "articles_with_content": sum(1 for x in rows if len(str(x.get("content") or "")) >= 320),
                "articles_with_summary": sum(1 for x in rows if str(x.get("summary") or "").strip()),
                "detail_enriched_count": sum(1 for x in rows if x.get("detail_enriched")),
            },
        }
        latest = self.output_dir / "cnyes_news_latest.json"
        self._atomic_write(latest, json.dumps(result, ensure_ascii=False, indent=2))
        # 依發佈日寫日檔（取代同日舊檔），相容 load_recent_cached 與舊程式
        for d, items in by_date.items():
            daily = dict(result)
            daily.update({"crawl_date": d, "article_count": len(items), "articles": items})
            self._atomic_write(self.output_dir / f"cnyes_news_{d}.json", json.dumps(daily, ensure_ascii=False, indent=2))
        # 清掉太舊的日檔，避免資料夾無限增長
        try:
            files = sorted(self.output_dir.glob("cnyes_news_20??-??-??.json"))
            for old_file in files[:-KEEP_DAILY_FILES]:
                old_file.unlink(missing_ok=True)
        except Exception:
            pass
        result["cache_path"] = str(latest)
        result["cache_exists"] = latest.exists()
        return result

    def load_latest(self) -> dict[str, Any]:
        path = self.output_dir / "cnyes_news_latest.json"
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

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
                if isinstance(cached, dict) and isinstance(cached.get("articles"), list):
                    # 舊版快取可能只有 title/URL/很短摘要；新版本啟動時先升級正文，不直接沿用低品質 cache。
                    cached_articles = [x for x in cached.get("articles", []) if isinstance(x, dict)]
                    weak_count = sum(1 for x in cached_articles if self._content_quality(x) < 320)
                    if weak_count == 0:
                        return cached

                    upgraded, enrich_errors = self.enrich_articles(
                        cached_articles,
                        max_articles=int(os.getenv("CNYES_DETAIL_ENRICH_MAX", "120")),
                    )
                    cached["articles"] = upgraded
                    cached["article_count"] = len(upgraded)
                    cached["content_quality"] = {
                        "articles_with_content": sum(1 for x in upgraded if len(str(x.get("content") or "")) >= 320),
                        "articles_with_summary": sum(1 for x in upgraded if bool(str(x.get("summary") or "").strip())),
                        "detail_enriched_count": sum(1 for x in upgraded if bool(x.get("detail_enriched"))),
                        "upgraded_existing_cache": True,
                    }
                    if enrich_errors:
                        cached.setdefault("errors", [])
                        cached["errors"].extend(enrich_errors)
                    cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
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

        # API 常見只有 title/summary；夜間完整快取再補一批正文，讓早報與 18:00 財經資訊有可閱讀內容。
        detail_errors: list[str] = []
        enrich_limit = int(os.getenv("CNYES_DETAIL_ENRICH_MAX", "120"))
        if enrich_limit > 0 and articles:
            raw_rows = [asdict(x) for x in articles]
            raw_rows, detail_errors = self.enrich_articles(raw_rows, max_articles=enrich_limit)
            articles = [CnyesArticle(**{k: row.get(k) for k in CnyesArticle.__dataclass_fields__.keys()}) for row in raw_rows]

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
            "errors": errors + detail_errors,
            "content_quality": {
                "articles_with_content": sum(1 for x in articles if len(str(x.content or "")) >= 320),
                "articles_with_summary": sum(1 for x in articles if bool(str(x.summary or "").strip())),
                "detail_enriched_count": sum(1 for x in articles if bool(getattr(x, "detail_enriched", False))),
            },
        }
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        # 明確把實際快取檔案路徑寫回結果，讓夜間排程與人工測試可以直接驗證。
        result["cache_path"] = str(cache_path)
        result["cache_exists"] = cache_path.exists()
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
