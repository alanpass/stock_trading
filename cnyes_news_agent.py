# -*- coding: utf-8 -*-
"""鉅亨網（cnyes）頭條新聞 Agent — 爬取／快取／本機 Qwen3 8B 摘要

用法設計成兩個獨立階段，對應你描述的排程：

1. **23:00 排程（只爬取，不摘要）**：
       python cnyes_news_agent.py --refresh
   呼叫 `CnyesNewsAgent.refresh_cache()`：掃 https://news.cnyes.com/news/cat/headline
   等分類頁，抓到的文章併入本機快取 data/news/cnyes_seen.json（依文章 id 去重），
   同時清掉太舊的紀錄。這一步不呼叫 Ollama，跑起來快、也不怕 Ollama 還沒啟動。

2. **早報時（只讀快取＋摘要）**：
       python cnyes_news_agent.py --digest --days 2
   呼叫 `CnyesNewsAgent.build_digest(days=2)`：只讀快取裡最近 2 天的文章（不重新爬網頁），
   交給本機 Qwen3 8B 合併、篩選成「財經報導摘要」。

`collect()` = `refresh_cache()` + `build_digest()`，保留給想一次做完（例如手動測試、
或還沒排 23:00 排程之前先跑跑看）的情境用。

已知限制（本機沙盒沒有對外網路，以下是依實際觀察到的鉅亨網頁面結構撰寫，
尚未實際連線跑過，接進排程前務必先跑 `python test_cnyes_news_agent.py --dry-run` 確認）：

- 列表頁（/news/cat/headline）看起來是新聞聚合頁，一次 GET 大概只能看到目前最新的一批文章
  （可能是最近幾小時到大半天），不保證單次就能看到完整 48 小時。這裡改用更穩定的策略：
  只從列表頁掃出 `/news/id/<id>` 這種永久連結當「候選清單」，交給每天固定跑 --refresh
  的排程反覆累積快取，涵蓋度會隨每天執行逐步補齊到 2 天、而不是靠單次爬取。
- 標題／發布時間／分類／關鍵字一律讀文章詳細頁的 meta 標籤（og:title、
  article:published_time、meta name=category/keywords），這些比列表頁版面穩定，
  但如果鉅亨網改版拿掉這些 meta 標籤，這裡就會抓不到，請用 --dry-run 檢查。
- 如果 --refresh 抓到 0 篇候選，很可能是被 User-Agent／頻率限制擋下、或分類頁版面已改版；
  可以先調高 REQUEST_DELAY，或比照本專案 Fugle 備忘錄的三層讀取（requests → curl_cffi
  Chrome 偽裝 → Selenium headless）在 `_get()` 裡加第二層，這裡先只用 requests 保持簡單。
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 requests：pip install requests") from exc

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 beautifulsoup4：pip install beautifulsoup4") from exc


TAIPEI_TZ = timezone(timedelta(hours=8))
LISTING_URL_TMPL = "https://news.cnyes.com/news/cat/{slug}"
ARTICLE_URL_TMPL = "https://news.cnyes.com/news/id/{aid}"
ARTICLE_ID_RE = re.compile(r"/news/id/(\d+)")
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}
VALID_IMPACT = ("偏利多", "偏利空", "中性", "待觀察")
VALID_RELEVANCE = ("台股", "總經", "產業", "其他")


@dataclass
class CnyesArticle:
    id: str
    url: str
    title: str = ""
    published_at: str | None = None  # ISO 字串，已轉台北時間
    category: str = ""
    keywords: list[str] = field(default_factory=list)
    summary_points: list[str] = field(default_factory=list)  # 鉅亨網自家摘要或退回的本文前段
    lead_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CnyesNewsAgent:
    """鉅亨網頭條新聞：refresh_cache() 爬取存檔、build_digest() 讀快取＋本機 Qwen3 8B 摘要重點。"""

    def __init__(
        self,
        base_dir: str | Path = ".",
        categories: list[str] | None = None,
        max_new_per_run: int = 40,
        request_delay: float = 0.6,
        ollama_model: str = "qwen3:8b",
        ollama_host: str = "http://127.0.0.1:11434",
        timeout: int = 15,
        cache_keep_days: int = 4,
    ):
        self.base = Path(base_dir).resolve()
        self.categories = categories or ["headline"]
        self.max_new_per_run = max_new_per_run
        self.request_delay = request_delay
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host.rstrip("/")
        self.timeout = timeout
        self.cache_keep_days = max(cache_keep_days, 2)
        self.cache_path = self.base / "data" / "news" / "cnyes_seen.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    # ------------------------------- 底層抓取 -------------------------------
    def _get(self, url: str) -> str | None:
        """單一 GET，失敗回傳 None。若之後發現被擋（整批 403 或空內容），可仿照本專案
        Fugle 備忘錄的三層讀取，在這裡加 curl_cffi／Selenium 備援。"""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200 or not resp.text:
                return None
            resp.encoding = resp.encoding or "utf-8"
            return resp.text
        except requests.RequestException:
            return None

    def _candidate_ids(self) -> list[str]:
        """掃過設定的分類列表頁，取得候選文章 id（新到舊、依出現順序）。只依賴
        `/news/id/<id>` 這種穩定永久連結，不依賴列表頁版面（較容易改版）。"""
        seen: dict[str, None] = {}
        for slug in self.categories:
            text = self._get(LISTING_URL_TMPL.format(slug=slug))
            if text:
                for m in ARTICLE_ID_RE.finditer(text):
                    seen.setdefault(m.group(1), None)
            time.sleep(self.request_delay)
        return list(seen.keys())

    @staticmethod
    def _meta(soup: BeautifulSoup, *, prop: str | None = None, name: str | None = None) -> str | None:
        tag = soup.find("meta", attrs={"property": prop} if prop else {"name": name})
        if tag and tag.get("content"):
            return html.unescape(tag["content"]).strip()
        return None

    def _parse_article(self, article_id: str) -> CnyesArticle | None:
        """讀單篇文章詳細頁。刻意只依賴 meta 標籤（og:title / article:published_time /
        category / keywords）取得核心欄位，這些比正文版面穩定很多；本文前幾段只當
        補充（抓不到摘要時的退路），解析失敗不會讓整批爬取中斷。"""
        url = ARTICLE_URL_TMPL.format(aid=article_id)
        text = self._get(url)
        if not text:
            return None
        try:
            soup = BeautifulSoup(text, "html.parser")

            published_at = None
            published_raw = self._meta(soup, prop="article:published_time")
            if published_raw:
                try:
                    dt_utc = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                    published_at = dt_utc.astimezone(TAIPEI_TZ).isoformat()
                except ValueError:
                    published_at = None

            title = self._meta(soup, prop="og:title") or (soup.title.get_text(strip=True) if soup.title else "")
            title = re.sub(r"\s*[|｜]\s*鉅亨網\s*$", "", title).strip()

            category = self._meta(soup, name="category") or ""
            keywords = [k.strip() for k in (self._meta(soup, name="keywords") or "").split(",") if k.strip()]

            # 鉅亨網自家「AI新聞摘要」常整段塞進 og:description（\n 分行的條列句），
            # 比在正文找特定 class 穩定很多；抓不到就退回本文前幾段。
            desc = self._meta(soup, prop="og:description") or self._meta(soup, name="description") or ""
            summary_points = [re.sub(r"^\d+[.、]\s*", "", ln).strip() for ln in desc.split("\n") if ln.strip()]

            lead_text = ""
            try:
                paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                paras = [p for p in paras if len(p) >= 15 and not p.startswith("鉅亨網")]
                lead_text = " ".join(paras[:3])[:400]
            except Exception:
                pass
            if not summary_points and lead_text:
                summary_points = [lead_text[:150]]

            return CnyesArticle(
                id=article_id, url=url, title=title, published_at=published_at,
                category=category, keywords=keywords[:8], summary_points=summary_points[:6],
                lead_text=lead_text,
            )
        except Exception:
            return None

    # -------------------------------- 快取 ----------------------------------
    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("articles"), dict):
                    return data
            except Exception:
                pass
        return {"articles": {}}

    def _save_cache(self, articles: dict[str, Any]) -> None:
        payload = {"updated_at": datetime.now(TAIPEI_TZ).isoformat(), "articles": articles}
        try:
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _prune(self, articles: dict[str, Any], now: datetime) -> dict[str, Any]:
        keep_after = now - timedelta(days=self.cache_keep_days)
        kept = {}
        for aid, rec in articles.items():
            try:
                pub = datetime.fromisoformat(rec["published_at"])
            except Exception:
                continue
            if pub >= keep_after:
                kept[aid] = rec
        return kept

    # -------------------------------- 階段一 ---------------------------------
    def refresh_cache(self) -> dict[str, Any]:
        """23:00 排程呼叫這個：只爬取＋寫快取，不呼叫 Ollama。回傳這次執行的簡單統計，
        方便排程記 log。"""
        now = datetime.now(TAIPEI_TZ)
        cache = self._load_cache()
        articles: dict[str, Any] = cache.get("articles", {})

        ids = self._candidate_ids()
        new_count = 0
        for aid in ids:
            if aid in articles:
                continue
            art = self._parse_article(aid)
            time.sleep(self.request_delay)
            if art and art.published_at:
                articles[aid] = art.to_dict()
                new_count += 1
            if new_count >= self.max_new_per_run:
                break

        articles = self._prune(articles, now)
        self._save_cache(articles)
        return {
            "as_of": now.isoformat(),
            "candidates_seen": len(ids),
            "new_articles": new_count,
            "cache_size": len(articles),
        }

    # -------------------------------- 階段二 ---------------------------------
    def _articles_in_window(self, days: int) -> list[CnyesArticle]:
        now = datetime.now(TAIPEI_TZ)
        since = now - timedelta(days=days)
        cache = self._load_cache()
        rows = []
        for rec in cache.get("articles", {}).values():
            try:
                pub = datetime.fromisoformat(rec["published_at"])
            except Exception:
                continue
            if pub >= since:
                rows.append(rec)
        rows.sort(key=lambda r: r["published_at"], reverse=True)
        return [CnyesArticle(**r) for r in rows]

    @staticmethod
    def _safe_json(raw: str) -> dict[str, Any] | None:
        text = (raw or "").strip()
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

    def _build_prompt(self, articles: list[CnyesArticle], days: int) -> str:
        lines = []
        for i, a in enumerate(articles):
            facts = "；".join(a.summary_points) if a.summary_points else (a.lead_text or "(無摘要)")
            lines.append(
                f"[{i}] {a.title}\n時間：{a.published_at}｜分類：{a.category}｜關鍵字：{'、'.join(a.keywords)}\n重點：{facts}"
            )
        joined = "\n\n".join(lines)
        return f"""你是台股盤後研究團隊的新聞編輯。以下是鉅亨網最近 {days} 天的頭條新聞清單（依編號 [0]、[1]...），
每則已附上鉅亨網原始標題、時間、分類、關鍵字與重點摘要。

請完成：
1. 找出對台股投資人真正重要的新聞（總經／Fed／利率／匯率／關稅／地緣政治／AI供應鏈／半導體／記憶體等），
   同一事件被多篇報導時合併成一則，用你自己的話寫，不要照抄標題。
2. 不重要或純娛樂/單一則八卦可以略過；輸出 6~12 則即可，不必每篇都收錄。
3. 每則標註 impact（{"|".join(VALID_IMPACT)}，站在台股角度）與 relevance（{"|".join(VALID_RELEVANCE)}）。
4. 最後給一句 overall_take：這 {days} 天新聞對台股的整體意義。
5. 只能使用清單中出現的事實，不要編造清單以外的數字或事件。

只輸出下面這個 JSON 結構，不要有任何額外文字、不要用 ```：
{{"items": [{{"headline": "string", "summary": "string（1-2句）", "impact": "{VALID_IMPACT[0]}",
"relevance": "{VALID_RELEVANCE[0]}", "keywords": ["string"], "source_indices": [0]}}],
"overall_take": "string"}}

新聞清單：
{joined}
"""

    def _run_qwen_digest(self, articles: list[CnyesArticle], days: int) -> dict[str, Any]:
        prompt = self._build_prompt(articles, days)
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model, "prompt": prompt, "stream": False,
                    "format": "json", "options": {"temperature": 0.2},
                },
                timeout=max(self.timeout, 90),
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        except Exception as exc:
            return {"available": False, "model": self.ollama_model, "error": f"呼叫本機 Ollama 失敗：{exc}",
                    "items": [], "overall_take": ""}

        parsed = self._safe_json(raw)
        if not isinstance(parsed, dict):
            return {"available": False, "model": self.ollama_model, "validation": "failed",
                    "error": "Qwen 回傳內容不是合法 JSON，已捨棄。", "items": [], "overall_take": ""}

        items = []
        for it in parsed.get("items", []) or []:
            if not isinstance(it, dict) or not it.get("headline"):
                continue
            idxs = [i for i in (it.get("source_indices") or []) if isinstance(i, int) and 0 <= i < len(articles)]
            sources = [{"title": articles[i].title, "url": articles[i].url} for i in idxs] or (
                [{"title": articles[0].title, "url": articles[0].url}] if articles else []
            )
            items.append({
                "headline": str(it.get("headline"))[:80],
                "summary": str(it.get("summary", ""))[:300],
                "impact": it.get("impact") if it.get("impact") in VALID_IMPACT else "待觀察",
                "relevance": it.get("relevance") if it.get("relevance") in VALID_RELEVANCE else "其他",
                "keywords": [str(k) for k in (it.get("keywords") or [])][:6],
                "sources": sources,
            })
        return {
            "available": True, "model": self.ollama_model, "items": items[:14],
            "overall_take": str(parsed.get("overall_take", ""))[:200],
        }

    def build_digest(self, days: int = 2) -> dict[str, Any]:
        """早報呼叫這個：只讀快取＋跑 Qwen 摘要，不重新爬網頁（爬取交給 refresh_cache()）。"""
        now = datetime.now(TAIPEI_TZ)
        articles = self._articles_in_window(days)
        source_url = LISTING_URL_TMPL.format(slug=self.categories[0])
        if not articles:
            return {
                "as_of": now.isoformat(), "source_url": source_url, "window_days": days,
                "since": (now - timedelta(days=days)).isoformat(), "total_in_window": 0,
                "articles": [],
                "digest": {"available": False, "model": self.ollama_model,
                           "error": "快取裡沒有近期文章，請確認 23:00 的 --refresh 排程有正常執行。",
                           "items": [], "overall_take": ""},
            }
        digest = self._run_qwen_digest(articles, days)
        return {
            "as_of": now.isoformat(), "source_url": source_url, "window_days": days,
            "since": (now - timedelta(days=days)).isoformat(), "total_in_window": len(articles),
            "articles": [a.to_dict() for a in articles],
            "digest": digest,
        }

    # -------------------------------- 便利入口 -------------------------------
    def collect(self, days: int = 2) -> dict[str, Any]:
        """= refresh_cache() + build_digest(days)。給手動測試或還沒排程前先跑跑看用；
        正式排程請分開呼叫 refresh_cache()（23:00）與 build_digest()（早報時）。"""
        self.refresh_cache()
        return self.build_digest(days=days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="鉅亨網頭條新聞 Agent")
    parser.add_argument("--refresh", action="store_true", help="只爬取＋寫快取（23:00 排程用）")
    parser.add_argument("--digest", action="store_true", help="只讀快取＋跑 Qwen 摘要（早報用）")
    parser.add_argument("--days", type=int, default=2, help="--digest 用：摘要涵蓋最近幾天")
    parser.add_argument("--base-dir", default=".", help="專案根目錄（快取存在 <base-dir>/data/news/）")
    args = parser.parse_args()

    agent = CnyesNewsAgent(base_dir=args.base_dir)
    if args.refresh:
        print(json.dumps(agent.refresh_cache(), ensure_ascii=False, indent=2))
    elif args.digest:
        result = agent.build_digest(days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        result = agent.collect(days=args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
