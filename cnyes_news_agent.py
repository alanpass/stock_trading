# -*- coding: utf-8 -*-
"""
鉅亨新聞研究 Agent
============================================================
用途：
1. 讀取「前一晚 23:00 已完成」的鉅亨網每日新聞快取。
2. 依研究主題挑出值得研究的新聞候選（規則式），新聞摘要與市場因果研究由 Claude 完成。
3. 只閱讀已快取新聞，不在盤後重新爬網站，因此不拖慢盤後研究。
4. 產出結構化新聞研究結果，供 ResearchAgent 與 EmailAgent 使用。

注意：
- 夜間爬蟲負責「資料蒐集」。
- 本 Agent 負責「新聞研究」。
- 新聞是市場線索；重大事項仍需回到 TWSE / TPEx / MOPS / 公司原始資料查證。
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


from ai_backend import AGENT_MODEL, PENDING_STATUS

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
    """從已快取的 CNYES 新聞挑出研究候選，交給 Claude 研究。"""

    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir).resolve()

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

    def _fallback(self, selected: list[dict[str, Any]], cache_meta: dict[str, Any], error: str = "") -> dict[str, Any]:
        findings = []
        for a in selected[:8]:
            findings.append({
                "title": _clean_text(a.get("title", "新聞事件"), 100),
                "summary": (_clean_text(a.get("content", "") or a.get("summary", ""), 780)
                            or f"{_clean_text(a.get('title', '新聞事件'), 160)}：已進入最近兩日新聞研究範圍。"),
                "why_relevant": "鉅亨近兩日新聞中出現，已取得文章內容；若尚未完成跨來源歸因，標記為待查證。",
                "impact": "待查證",
                "related_symbols": [],
                "evidence": [f"鉅亨新聞：{_clean_text(a.get('title', ''), 200)}"],
                "source_links": [str(a.get("url", ""))] if a.get("url") else [],
                "confidence": 0.20,
            })
        return {
            "agent_status": "cnyes_digest_fallback",
            "model": AGENT_MODEL,
            "generated_at": datetime.now(TAIPEI).isoformat(),
            "cache_dates": cache_meta.get("cache_dates", []),
            "input_article_count": cache_meta.get("article_count", 0),
            "selected_article_count": len(selected),
            "overview": "鉅亨近兩日新聞已完成離線快取，尚待 Claude 新聞研究；以下為研究候選，未視為已驗證結論。",
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
                "model": AGENT_MODEL,
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
        # 新聞研究由 Claude 排程任務完成；這裡先提供規則挑選的研究候選讓研究包可用。
        return {**self._fallback(selected, cache_meta, "等待 Claude 新聞研究"), "agent_status": PENDING_STATUS}
