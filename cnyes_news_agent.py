# -*- coding: utf-8 -*-
"""
鉅亨新聞研究 Agent
============================================================
用途：
1. 讀取「前一晚 23:00 已完成」的鉅亨網每日新聞快取。
2. 在盤後研究時間由本機 Ollama / Qwen3 8B 做新聞摘要與市場因果研究。
3. 只閱讀已快取新聞，不在盤後重新爬網站，因此不拖慢盤後研究。
4. 產出結構化新聞研究結果，供 ResearchAgent 與 EmailAgent 使用。

注意：
- 夜間爬蟲負責「資料蒐集」。
- 本 Agent 負責「新聞研究」。
- 新聞是市場線索；重大事項仍需回到 TWSE / TPEx / MOPS / 公司原始資料查證。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
TAIPEI = ZoneInfo("Asia/Taipei")


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        x = float(value)
        return x if x == x and abs(x) != float("inf") else default
    except Exception:
        return default


def _clean_text(value: Any, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


class CnyesNewsDigestAgent:
    """使用本機 Qwen3 對已快取 CNYES 新聞進行盤後研究。"""

    def __init__(self, base_dir: str | Path = ".", ollama_host: str | None = None, ollama_model: str | None = None):
        self.base = Path(base_dir).resolve()
        self.ollama_host = (ollama_host or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.ollama_model = ollama_model or DEFAULT_OLLAMA_MODEL

    @staticmethod
    def _article_blob(article: dict[str, Any]) -> str:
        return " ".join([
            str(article.get("title", "")),
            str(article.get("summary", "")),
            str(article.get("content", ""))[:1800],
            str(article.get("category", "")),
            " ".join(map(str, article.get("tags", []) or [])),
        ]).lower()

    def _select_articles(self, articles: list[dict[str, Any]], payload: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        movers = (payload.get("market_movers", {}) or {}).get("movers", []) or []
        focus_symbols = [str(x).strip().upper() for x in payload.get("symbols", []) or []]
        focus_keywords = [
            "AI", "AI伺服器", "伺服器", "GPU", "ASIC", "HPC", "資料中心",
            "半導體", "記憶體", "HBM", "DRAM", "NAND", "晶圓", "先進製程",
            "封裝", "Chiplet", "PCB", "載板", "ABF", "BT", "CCL", "高速材料",
            "電子零組件", "光電", "OLED", "面板", "散熱", "電源", "BBU", "UPS",
            "營收", "獲利", "接單", "訂單", "漲價", "降價", "關稅", "原油", "油價",
            "Fed", "利率", "出口", "台股",
        ]
        for x in focus_symbols:
            focus_keywords.append(x)
        mover_names = []
        for m in movers:
            if isinstance(m, dict):
                focus_keywords.append(str(m.get("symbol", "")).strip())
                mover_names.append(str(m.get("name", "")).strip())
        focus_keywords.extend([x for x in mover_names if x])

        scored: list[tuple[float, dict[str, Any]]] = []
        for article in articles:
            if not isinstance(article, dict) or not article.get("title"):
                continue
            title = str(article.get("title", "" )).lower()
            blob = self._article_blob(article)
            score = 0.0
            for kw in focus_keywords:
                kw = kw.strip().lower()
                if len(kw) < 2:
                    continue
                if kw in title:
                    score += 5.0
                elif kw in blob:
                    score += 2.0
            # 市場方向詞讓盤後市場解釋更容易被選入。
            for kw in ("大漲", "大跌", "漲停", "跌停", "領漲", "領跌", "爆量", "新高", "新低", "升息", "降息"):
                if kw in title:
                    score += 2.0
            # 讓較新的新聞稍微優先，但不以日期取代內容相關度。
            published = str(article.get("published_ts", "") or article.get("published", ""))
            score += min(1.5, 0.05 * len(published)) if published else 0
            scored.append((score, article))

        scored.sort(key=lambda x: (x[0], str(x[1].get("published_ts", ""))), reverse=True)
        chosen = []
        seen = set()
        for _, article in scored:
            aid = str(article.get("article_id", ""))
            if aid and aid in seen:
                continue
            if aid:
                seen.add(aid)
            chosen.append(article)
            if len(chosen) >= limit:
                break
        # 若相關度候選太少，仍保留最新新聞的一部分，讓 Agent 可以看到市場背景。
        if len(chosen) < min(40, len(articles)):
            for article in sorted(articles, key=lambda x: str(x.get("published_ts", "")), reverse=True):
                aid = str(article.get("article_id", ""))
                if aid and aid in seen:
                    continue
                chosen.append(article)
                if len(chosen) >= min(limit, max(40, len(chosen))):
                    break
        return chosen[:limit]

    def _build_context(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for a in articles:
            rows.append({
                "article_id": a.get("article_id", ""),
                "title": _clean_text(a.get("title", ""), 300),
                "published": a.get("published", ""),
                "category": a.get("category", ""),
                "summary": _clean_text(a.get("summary", "") or a.get("content", ""), 700),
                "content_excerpt": _clean_text(a.get("content", ""), 700),
                "tags": (a.get("tags") or [])[:12],
                "url": a.get("url", ""),
            })
        return rows

    def _fallback(self, selected: list[dict[str, Any]], cache_meta: dict[str, Any], error: str = "") -> dict[str, Any]:
        findings = []
        for a in selected[:8]:
            findings.append({
                "title": _clean_text(a.get("title", "新聞事件"), 100),
                "summary": _clean_text(a.get("summary", "") or a.get("content", ""), 500),
                "why_relevant": "鉅亨近兩日新聞中出現，列為盤後研究候選事件；尚未由 Agent 完成跨來源歸因。",
                "impact": "待查證",
                "related_symbols": [],
                "evidence": [f"鉅亨新聞：{_clean_text(a.get('title', ''), 200)}"],
                "source_links": [str(a.get("url", ""))] if a.get("url") else [],
                "confidence": 0.20,
            })
        return {
            "agent_status": "cnyes_digest_fallback",
            "model": self.ollama_model,
            "generated_at": datetime.now(TAIPEI).isoformat(),
            "cache_dates": cache_meta.get("cache_dates", []),
            "input_article_count": cache_meta.get("article_count", 0),
            "selected_article_count": len(selected),
            "overview": "鉅亨近兩日新聞已完成離線快取，但本次 Qwen3 新聞摘要未完成；以下為研究候選，未視為已驗證結論。",
            "key_findings": findings,
            "market_drivers": [],
            "watch_topics": [],
            "error": error,
        }

    def run(self, payload: dict[str, Any], limit: int = 100) -> dict[str, Any]:
        corpus = payload.get("cnyes_news", {}) or {}
        articles = [x for x in (corpus.get("articles", []) or []) if isinstance(x, dict)]
        cache_dates = corpus.get("cache_dates", []) or ([corpus.get("crawl_date")] if corpus.get("crawl_date") else [])
        cache_meta = {
            "cache_dates": cache_dates,
            "article_count": len(articles),
        }
        if not articles:
            return {
                "agent_status": "no_cnyes_cache",
                "model": self.ollama_model,
                "generated_at": datetime.now(TAIPEI).isoformat(),
                "cache_dates": cache_dates,
                "input_article_count": 0,
                "selected_article_count": 0,
                "overview": "目前沒有可供盤後 Agent 閱讀的鉅亨新聞夜間快取。",
                "key_findings": [],
                "market_drivers": [],
                "watch_topics": [],
                "error": "沒有 CNYES 日檔快取",
            }

        selected = self._select_articles(articles, payload, limit=max(40, int(limit)))
        context = self._build_context(selected)

        # 本機 Qwen3；不要把完整數百篇全文塞進 context。
        try:
            import ollama
        except Exception as exc:
            return self._fallback(selected, cache_meta, f"ollama 套件不可用：{exc}")

        watchlist = payload.get("symbols", []) or []
        movers = (payload.get("market_movers", {}) or {}).get("movers", [])[:30]
        prompt = f"""
你是「鉅亨新聞盤後研究 Agent」。

今天盤後研究日期：{payload.get('report_date', '')}
研究股票範圍：{watchlist}
今日市場異動候選：{movers}

以下是『前一晚／前兩晚已經爬好並保存』的鉅亨網頭條新聞研究資料。你現在的任務不是逐篇摘要，而是把兩日新聞轉成盤後研究：
1. 找出今天市場、產業或個股最值得注意的事件。
2. 找出可能解釋今日產業漲跌的新聞脈絡。
3. 特別檢查電子零組件、PCB、IC載板、ABF/BT、CCL、高速材料、半導體、記憶體、AI伺服器、光電等產業是否存在共同驅動因素。
4. 找出互相呼應或互相矛盾的新聞。
5. 不要重複法說會／財報模組已整理的完整內容；只補充新聞層面的新發現。
6. 不確定時必須寫「待查證」，不得把單一新聞敘事當成事實。
7. 每個重點盡量回答：「發生什麼、可能為什麼、影響誰、證據是哪篇新聞」。
8. 只允許引用下面提供的新聞；不得引用你自己的記憶或杜撰資料。

請只輸出 JSON：
{{
  "overview": "1~2句盤後新聞總結",
  "key_findings": [
    {{
      "title": "新聞研究標題",
      "summary": "發生什麼",
      "why_relevant": "為什麼與今天市場或產業有關",
      "impact": "利多|利空|混合|待查證",
      "related_symbols": ["股票代號，可空"],
      "evidence": ["具體新聞標題／日期／證據"],
      "source_links": ["只能填輸入資料中的 URL"],
      "confidence": 0.0
    }}
  ],
  "market_drivers": ["從新聞歸納出的市場／產業驅動因素；每項都要有新聞證據"],
  "watch_topics": ["值得盤後持續追蹤的新聞題材"]
}}

新聞資料：
{json.dumps(context, ensure_ascii=False, default=str)}
""".strip()

        try:
            client = ollama.Client(host=self.ollama_host)
            try:
                resp = client.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": "你是完全本機的鉅亨新聞研究 Agent，只能使用使用者提供的新聞資料。"},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.05},
                    format="json",
                    think=True,
                )
            except TypeError:
                resp = client.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": "你是完全本機的鉅亨新聞研究 Agent，只能使用使用者提供的新聞資料。"},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.05},
                    format="json",
                )
            content = getattr(getattr(resp, "message", None), "content", "") or ""
            raw = json.loads(content)
            if not isinstance(raw, dict):
                raise ValueError("Qwen 新聞摘要不是 JSON object")

            valid_urls = {str(a.get("url", "")).strip() for a in selected if a.get("url")}
            key_findings = []
            for item in raw.get("key_findings", []) or []:
                if not isinstance(item, dict):
                    continue
                links = [u for u in (item.get("source_links") or []) if str(u).strip() in valid_urls]
                if not links:
                    title_text = str(item.get("title", "")).strip()
                    best = next((a.get("url") for a in selected if title_text and title_text in str(a.get("title", ""))), "")
                    if best:
                        links = [best]
                key_findings.append({
                    "title": _clean_text(item.get("title", "新聞研究"), 120),
                    "summary": _clean_text(item.get("summary", ""), 800),
                    "why_relevant": _clean_text(item.get("why_relevant", ""), 800),
                    "impact": _clean_text(item.get("impact", "待查證"), 30),
                    "related_symbols": [str(x).strip() for x in (item.get("related_symbols") or []) if str(x).strip()][:15],
                    "evidence": [_clean_text(x, 500) for x in (item.get("evidence") or []) if str(x).strip()][:8],
                    "source_links": links[:5],
                    "confidence": max(0.0, min(1.0, _safe_float(item.get("confidence"), 0.0) or 0.0)),
                })

            result = {
                "agent_status": "qwen3_cnyes_news_agent",
                "model": self.ollama_model,
                "generated_at": datetime.now(TAIPEI).isoformat(),
                "cache_dates": cache_dates,
                "input_article_count": len(articles),
                "selected_article_count": len(selected),
                "overview": _clean_text(raw.get("overview", ""), 1600),
                "key_findings": key_findings[:10],
                "market_drivers": [_clean_text(x, 600) for x in (raw.get("market_drivers") or []) if str(x).strip()][:10],
                "watch_topics": [_clean_text(x, 500) for x in (raw.get("watch_topics") or []) if str(x).strip()][:10],
            }
            if not result["key_findings"] and not result["market_drivers"]:
                raise ValueError("Qwen 新聞摘要沒有產生有效研究內容")
            return result
        except Exception as exc:
            return self._fallback(selected, cache_meta, str(exc))
