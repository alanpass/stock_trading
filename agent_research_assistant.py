# -*- coding: utf-8 -*-
"""
AI 台股盤後研究助理 Agent
============================================================

核心概念：
    一個 AI 研究助理幫使用者讀完一堆資料後，
    自己決定哪些事情值得記錄、哪些需要標記、哪些需要進一步查證。

ResearchAgent 負責資料工程與證據蒐集；本模組提供研究工具與研究記憶：
Claude 盤後研究（.claude/skills/after-close-research）透過 claude_tools.py 呼叫這些工具，
自主決定要看哪些資料，再寫出研究筆記；--finalize 時用 _normalize 驗證並存入研究記憶。

可用工具：
    - 市場概況
    - 公司財務
    - 法說會事件
    - Fugle 法說會詳細備忘錄
    - 本次研究包新聞
    - 即時 Google News RSS 補充搜尋
    - 題材證據
    - 模型健康度
    - 公司研究
    - 上一期 Agent 研究筆記

注意：
    - Agent 不取代資料來源，也不直接修改 production model。
    - Agent 不保存 chain-of-thought，只保存最終研究結果與工具使用摘要。
    - Agent 不輸出買賣建議，也不做投資標的排名。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import requests

from ai_backend import AGENT_STATUS


MAX_NEWS_RESULTS = 10
MEMORY_FILE_NAME = "agent_memory.json"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, ""))
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}



def _tool_schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


TOOLS = [
    _tool_schema(
        "get_market_overview",
        "讀取本次研究包的市場強弱、主要漲跌候選與產業表現。當你需要判斷今日異常市場訊號時使用。",
        {"focus": {"type": "string", "description": "例如：漲跌異常、產業強弱、今日市場概況"}},
        ["focus"],
    ),
    _tool_schema(
        "get_company_financial",
        "取得指定股票本次研究包中的營收、獲利、EPS 等財務資料。",
        {"symbol": {"type": "string"}},
        ["symbol"],
    ),
    _tool_schema(
        "get_earnings_events",
        "取得最近法說會事件與已有的 Agent 摘要。適合先找值得深讀的法說事件。",
        {"symbol": {"type": "string", "description": "可留空表示全部近期法說會"}},
        ["symbol"],
    ),
    _tool_schema(
        "read_fugle_memo",
        "重新開啟 Fugle 法說會備忘錄詳細文章並取得正文。只有需要確認法說內容時使用。",
        {"url": {"type": "string"}},
        ["url"],
    ),
    _tool_schema(
        "search_news_evidence",
        "在本次研究包的新聞與查證候選中，依關鍵字尋找相關證據。",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _tool_schema(
        "search_live_news",
        "使用 Google News RSS 做額外即時新聞搜尋。當研究包沒有足夠資訊、遇到不知道的原因、產業或個股異常漲跌原因不明、財報/營收缺資料時，應主動使用。新聞只能作為研究線索，需與官方資料、財務或多來源證據交叉比對。",
        {
            "query": {"type": "string", "description": "具體搜尋問題，例如：電子零組件 上漲 原因 載板 ABF CCL；3006 最新月營收 EPS"},
            "days": {"type": "integer", "description": "近幾天新聞，預設 3 天，最多 30 天"},
        },
        ["query"],
    ),
    _tool_schema(
        "get_cnyes_news_overview",
        "讀取已由爬蟲從鉅亨網頭條頁面抓下來的近兩日全部新聞索引、分類數量與代表性標題。先用此工具了解兩日新聞全貌，再決定需要深入查哪個市場/產業。",
        {},
    ),
    _tool_schema(
        "get_cnyes_research_digest",
        "取得盤後已由 CNYES News Agent 使用夜間快取新聞整理出的研究摘要。優先讀取此摘要，再針對重要事件用 search_cnyes_news 深讀原文。",
        {},
    ),
    _tool_schema(
        "search_cnyes_news",
        "在已爬下來的鉅亨網近兩日新聞全文/摘要中搜尋研究問題。遇到今日產業漲跌原因不明、熱門題材、公司異動或市場事件時，優先用此工具找近期新聞脈絡，再與 TWSE/TPEx/MOPS/Fugle 財務資料交叉比對。",
        {
            "query": {"type": "string", "description": "例如：電子零組件 上漲 原因 載板 ABF CCL；半導體 記憶體 HBM；3006 晶豪科 營收"},
            "limit": {"type": "integer", "description": "最多回傳幾篇，預設 12"},
        },
        ["query"],
    ),
    _tool_schema(
        "get_cnyes_news_batch",
        "按批次閱讀鉅亨近兩日新聞的標題、分類、發布時間與短摘要。用來建立兩日新聞全貌；需要深入原因時再用 search_cnyes_news 查全文。offset 從 0 開始。",
        {
            "offset": {"type": "integer"},
            "limit": {"type": "integer", "description": "每批 20～60 篇，建議 50"},
        },
        ["offset"],
    ),
    _tool_schema(
        "get_theme_evidence",
        "取得指定題材的支持證據、接單/出貨、營收支撐與矛盾/查證訊號。",
        {"theme": {"type": "string"}},
        ["theme"],
    ),
    _tool_schema(
        "get_model_health",
        "取得本次研究包的模型健康度、方向準確率、MAE/RMSE、模型年齡與重訓旗標。",
        {},
    ),
    _tool_schema(
        "get_company_research",
        "取得指定公司在本次研究包的業務定位、產業鏈、新聞證據與研究關鍵字。",
        {"symbol": {"type": "string"}},
        ["symbol"],
    ),
    _tool_schema(
        "get_previous_notes",
        "讀取上一期 AI 研究助理保存的研究筆記。這些筆記只能作為變化比較線索，不可當成當日事實，必要時需用本次工具重新驗證。",
        {},
    ),
]


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _clean_symbol(v: Any) -> str:
    return str(v or "").strip().upper().replace(".TW", "")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()]


CNYES_QUERY_EXPANSIONS = {
    "電子零組件": ["電子零組件", "PCB", "IC載板", "載板", "ABF", "BT", "CCL", "高速材料", "MLCC", "連接器"],
    "載板": ["載板", "IC載板", "ABF", "BT", "substrate"],
    "PCB": ["PCB", "印刷電路板", "載板", "ABF", "CCL", "高速材料"],
    "半導體": ["半導體", "晶圓", "先進製程", "封裝", "HBM", "記憶體", "ASIC", "GPU"],
    "AI": ["AI", "人工智慧", "AI伺服器", "伺服器", "GPU", "ASIC", "HPC", "資料中心", "Physical AI"],
    "記憶體": ["記憶體", "HBM", "DRAM", "NAND", "美光", "SK海力士"],
    "光電": ["光電", "面板", "OLED", "顯示器", "Micro LED"],
    "電源": ["電源", "電源供應器", "PSU", "BBU", "UPS", "資料中心"],
}

def _cnyes_query_terms(query: str) -> list[str]:
    raw_terms = [x.strip().lower() for x in re.findall(r"[0-9A-Za-z一-龥]+", str(query or "")) if len(x.strip()) >= 2]
    expanded = []
    for term in raw_terms:
        expanded.append(term)
        for key, values in CNYES_QUERY_EXPANSIONS.items():
            if term == key.lower() or term in [v.lower() for v in values]:
                expanded.extend(v.lower() for v in values)
    # 維持順序去重
    return list(dict.fromkeys(expanded))


def _normalize(raw: dict[str, Any], tool_history: list[dict[str, Any]], model: str) -> dict[str, Any]:
    notes = []
    for item in raw.get("research_notes") or []:
        if not isinstance(item, dict):
            continue
        importance = str(item.get("importance", "medium")).lower()
        if importance not in {"high", "medium", "low"}:
            importance = "medium"
        raw_conf = _safe_float(item.get("confidence"), 0.0) or 0.0
        normalized_conf = (raw_conf / 100.0) if raw_conf > 1 else raw_conf
        evidence = _as_list(item.get("evidence"))[:8]
        if raw_conf <= 0 and evidence:
            normalized_conf = 0.35
        notes.append({
            "importance": importance,
            "type": str(item.get("type", "observation")).strip()[:60],
            "symbol": _clean_symbol(item.get("symbol")),
            "title": str(item.get("title", "研究發現")).strip()[:120],
            "note": str(item.get("note", "")).strip()[:1200],
            "evidence": evidence,
            "tags": _as_list(item.get("tags"))[:10],
            "confidence": max(0.0, min(1.0, normalized_conf)),
            "follow_up": _as_list(item.get("follow_up"))[:8],
            "source_links": [str(x).strip() for x in (item.get("source_links") or []) if str(x).strip()][:8],
        })

    notes = [x for x in notes if x["note"]]
    return {
        "agent_status": AGENT_STATUS,
        "model": model,
        "generated_at": datetime.now().isoformat(),
        "overview": str(raw.get("overview", "")).strip()[:1600],
        "key_takeaways": _as_list(raw.get("key_takeaways"))[:10],
        "research_notes": notes[:30],
        "positive_signals": _as_list(raw.get("positive_signals"))[:12],
        "negative_signals": _as_list(raw.get("negative_signals"))[:12],
        "watch_items": _as_list(raw.get("watch_items"))[:12],
        "earnings_digest": str(raw.get("earnings_digest", "")).strip()[:2200],
        "financial_focus": _as_list(raw.get("financial_focus"))[:12],
        "cnyes_news_digest": _as_list(raw.get("cnyes_news_digest"))[:15],
        "model_alert": str(raw.get("model_alert", "")).strip()[:1200],
        "agent_actions": [
            {
                "tool": x.get("tool", ""),
                "reason": x.get("reason", ""),
                "result": x.get("result", "")[:500],
            }
            for x in tool_history[-30:]
        ],
    }


def _merge_cnyes_digest(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """把獨立 CNYES News Agent 的研究結果確實併入主 Agent，避免新聞有爬到卻在 Email 看不到。"""
    digest = payload.get("cnyes_research_digest", {}) or {}
    findings = digest.get("key_findings", []) or []
    if not isinstance(findings, list) or not findings:
        return result

    existing_titles = {str(x.get("title", "")).strip() for x in result.get("research_notes", []) if isinstance(x, dict)}
    notes = list(result.get("research_notes", []) or [])
    cnyes_text = list(result.get("cnyes_news_digest", []) or [])

    for item in findings:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "鉅亨新聞研究")).strip() or "鉅亨新聞研究"
        if title in existing_titles:
            continue
        symbols = [str(x).strip() for x in (item.get("related_symbols") or []) if str(x).strip()]
        evidence = [str(x).strip() for x in (item.get("evidence") or []) if str(x).strip()]
        links = [str(x).strip() for x in (item.get("source_links") or []) if str(x).strip()]
        summary = str(item.get("summary", "")).strip()
        why = str(item.get("why_relevant", "")).strip()
        note = summary + (f"；研究意義：{why}" if why else "")
        if not note:
            continue
        notes.append({
            "importance": "high" if float(item.get("confidence") or 0) >= 0.75 else "medium",
            "type": "industry" if symbols else "event",
            "symbol": _clean_symbol(symbols[0]) if symbols else "",
            "title": f"新聞研究｜{title}"[:120],
            "note": note[:1200],
            "evidence": evidence[:8] or [f"鉅亨新聞：{title}"],
            "tags": ["鉅亨新聞", str(item.get("impact", "待查證"))],
            "confidence": max(0.0, min(1.0, float(item.get("confidence") or 0))),
            "follow_up": ["與公司公告、財報、法人或其他產業資料交叉驗證"],
            "source_links": links[:8],
        })
        existing_titles.add(title)
        cnyes_text.append(note[:500])

    result["research_notes"] = notes[:30]
    result["cnyes_news_digest"] = cnyes_text[:15]
    result["cnyes_research_digest"] = digest
    if digest.get("overview"):
        result["key_takeaways"] = list(result.get("key_takeaways", []) or [])[:8]
        result["key_takeaways"].append(f"鉅亨近兩日新聞：{str(digest.get('overview'))[:500]}")
    return result


def _memory_path(base: Path) -> Path:
    p = base / "output" / "research_reports"
    p.mkdir(parents=True, exist_ok=True)
    return p / MEMORY_FILE_NAME


class ResearchToolbox:
    """盤後研究工具（市場、財務、法說會、新聞、題材、公司研究）與跨日研究記憶。"""

    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir).resolve()
        self._earnings_agent = None

    def _load_memory(self) -> dict[str, Any]:
        path = _memory_path(self.base)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_memory(self, notes: dict[str, Any]) -> None:
        path = _memory_path(self.base)
        existing = self._load_memory()
        history = existing.get("history", []) if isinstance(existing, dict) else []
        if not isinstance(history, list):
            history = []

        snapshot = {
            "saved_at": datetime.now().isoformat(),
            "report_date": notes.get("report_date", ""),
            "overview": notes.get("overview", ""),
            "key_takeaways": notes.get("key_takeaways", []),
            "research_notes": notes.get("research_notes", []),
            "watch_items": notes.get("watch_items", []),
            "positive_signals": notes.get("positive_signals", []),
            "negative_signals": notes.get("negative_signals", []),
            "earnings_digest": notes.get("earnings_digest", ""),
            "financial_focus": notes.get("financial_focus", []),
            "cnyes_news_digest": notes.get("cnyes_news_digest", []),
            "model_alert": notes.get("model_alert", ""),
            "agent_actions": notes.get("agent_actions", []),
        }

        history.append(snapshot)
        history = history[-30:]

        path.write_text(
            json.dumps(
                {
                    "latest": snapshot,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def _get_earnings_agent(self):
        if self._earnings_agent is None:
            from earnings_call_agent import EarningsCallAgent
            self._earnings_agent = EarningsCallAgent(self.base)
        return self._earnings_agent

    def _tool(self, name: str, args: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        mm = payload.get("market_movers", {}) or {}

        if name == "get_market_overview":
            movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]
            industries = [x for x in mm.get("industry_summary", []) if isinstance(x, dict)]
            movers = sorted(movers, key=lambda x: abs(_safe_float(x.get("today_change_percent"), 0) or 0), reverse=True)[:20]
            industries = sorted(industries, key=lambda x: abs(_safe_float(x.get("today_avg_change"), 0) or 0), reverse=True)[:15]
            return {"focus": args.get("focus", ""), "movers": movers, "industry_summary": industries}

        if name == "get_company_financial":
            symbol = _clean_symbol(args.get("symbol"))
            rows = [x for x in payload.get("financial_snapshots", []) if isinstance(x, dict) and _clean_symbol(x.get("symbol")) == symbol]
            profiles = payload.get("business_profiles", {}) or {}
            profile = profiles.get(symbol, {}) if isinstance(profiles, dict) else {}
            return {
                "symbol": symbol,
                "name": profile.get("name", ""),
                "industry_name": profile.get("industry_name", ""),
                "business_group": profile.get("business_group", ""),
                "rows": rows[:10],
                "available": bool(rows),
            }

        if name == "get_earnings_events":
            symbol = _clean_symbol(args.get("symbol"))
            rows = [x for x in payload.get("earnings_calls", []) if isinstance(x, dict)]
            if symbol:
                rows = [x for x in rows if _clean_symbol(x.get("symbol")) == symbol]
            return {
                "events": [
                    {k: x.get(k) for k in (
                        "symbol", "name", "event_date", "title", "impact", "confidence",
                        "one_line_summary", "summary", "financial_highlights", "operating_highlights",
                        "guidance", "positive_factors", "negative_factors", "key_risks",
                        "qa_highlights", "memo_opened", "detail_read_verified", "source_url",
                    )}
                    for x in rows[:20]
                ]
            }

        if name == "read_fugle_memo":
            url = str(args.get("url", "")).strip()
            if not url.startswith("https://blog.fugle.tw/post/earnings-call-"):
                return {"ok": False, "error": "只允許讀取 Fugle earnings-call memo URL。"}
            agent = self._get_earnings_agent()
            article = agent.crawler.read_memo_tool(url)
            return {
                "ok": True,
                "symbol": article.get("symbol"),
                "title": article.get("title"),
                "published_date": article.get("published_date"),
                "detail_read_verified": article.get("detail_read_verified"),
                "sections": article.get("sections", {}),
                "memo_text": str(article.get("memo_text", ""))[:28000],
                "source_url": article.get("url", url),
            }

        if name == "search_news_evidence":
            query = str(args.get("query", "")).strip().lower()
            terms = [x for x in re.findall(r"[0-9A-Za-z一-龥]+", query) if len(x) >= 2]
            rows = []
            for x in payload.get("news", [])[:400]:
                if not isinstance(x, dict):
                    continue
                text = f"{x.get('title', '')} {x.get('summary', '')} {x.get('source', '')}".lower()
                hit = sum(1 for t in terms if t in text)
                if hit:
                    rows.append((hit, x))
            rows.sort(key=lambda z: (z[0], z[1].get("published", "")), reverse=True)
            return {"query": query, "matches": [x for _, x in rows[:15]]}

        if name == "search_live_news":
            if _env_bool("AFTER_CLOSE_FAST_MODE", False):
                return {"query": str(args.get("query", "")), "matches": [], "skipped": True, "reason": "快速盤後模式只使用夜間 CNYES 快取，不再連線即時新聞。"}
            query = str(args.get("query", "")).strip()
            if not query:
                return {"query": "", "matches": []}
            try:
                try:
                    days = int(args.get("days", 3) or 3)
                except Exception:
                    days = 3
                days = max(1, min(30, days))
                if not re.search(r"\bwhen:\d+d\b", query, flags=re.I):
                    query = f"{query} when:{days}d"

                r = requests.get(
                    GOOGLE_NEWS_RSS,
                    params={"q": query, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"},
                    headers={"User-Agent": "Mozilla/5.0 TaiwanStockResearchAgent/1.0"},
                    timeout=20,
                )
                r.raise_for_status()
                root = ET.fromstring(r.text)
                matches = []
                for item in root.findall(".//item")[:MAX_NEWS_RESULTS]:
                    title = item.findtext("title", default="")
                    link = item.findtext("link", default="")
                    pub = item.findtext("pubDate", default="")
                    source = item.findtext("source", default="")
                    desc = item.findtext("description", default="")
                    matches.append({
                        "title": title,
                        "link": link,
                        "published": pub,
                        "source": source,
                        "summary": re.sub(r"<[^>]+>", " ", desc).strip()[:1200],
                    })
                return {"query": query, "matches": matches}
            except Exception as exc:
                return {"query": query, "matches": [], "error": str(exc)}

        if name == "get_cnyes_news_overview":
            corpus = payload.get("cnyes_news", {}) or {}
            articles = [x for x in corpus.get("articles", []) if isinstance(x, dict)]
            by_cat: dict[str, list[dict[str, Any]]] = {}
            for x in articles:
                cat = str(x.get("category", "頭條") or "頭條")
                by_cat.setdefault(cat, []).append(x)
            category_preview = []
            for cat, rows in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                category_preview.append({
                    "category": cat,
                    "count": len(rows),
                    "titles": [{"published": r.get("published", ""), "title": r.get("title", ""), "url": r.get("url", "")} for r in rows[:8]],
                })
            return {
                "source_url": corpus.get("source_url", "https://news.cnyes.com/news/cat/headline"),
                "crawl_date": corpus.get("crawl_date", ""),
                "days": corpus.get("days", 2),
                "window_start": corpus.get("window_start", ""),
                "window_end": corpus.get("window_end", ""),
                "article_count": len(articles),
                "listing_link_count": corpus.get("listing_link_count", 0),
                "category_counts": corpus.get("category_counts", {}),
                "category_preview": category_preview[:20],
                "errors": corpus.get("errors", []),
            }

        if name == "get_cnyes_research_digest":
            return payload.get("cnyes_research_digest", {}) or {}

        if name == "search_cnyes_news":
            corpus = payload.get("cnyes_news", {}) or {}
            articles = [x for x in corpus.get("articles", []) if isinstance(x, dict)]
            query = str(args.get("query", "")).strip()
            terms = _cnyes_query_terms(query)
            try:
                limit = max(3, min(30, int(args.get("limit", 12) or 12)))
            except Exception:
                limit = 12
            ranked = []
            for x in articles:
                blob = " ".join([str(x.get("title", "")), str(x.get("summary", "")), str(x.get("content", "")), str(x.get("category", "")), " ".join(map(str, x.get("tags", []) or []))]).lower()
                exact_hits = sum(1 for term in terms if term and term in blob)
                title_hits = sum(1 for term in terms if term and term in str(x.get("title", "")).lower())
                score = exact_hits + title_hits * 2
                if score > 0:
                    ranked.append((score, x))
            ranked.sort(key=lambda z: (z[0], z[1].get("published_ts", "")), reverse=True)
            return {
                "query": query,
                "expanded_terms": terms,
                "match_count": len(ranked),
                "matches": [
                    {k: x.get(k) for k in ("article_id", "title", "url", "category", "published", "summary", "content", "tags", "source", "source_type")}
                    for _, x in ranked[:limit]
                ],
            }

        if name == "get_cnyes_news_batch":
            corpus = payload.get("cnyes_news", {}) or {}
            articles = [x for x in corpus.get("articles", []) if isinstance(x, dict)]
            try:
                offset = max(0, int(args.get("offset", 0) or 0))
            except Exception:
                offset = 0
            try:
                limit = max(20, min(60, int(args.get("limit", 50) or 50)))
            except Exception:
                limit = 50
            batch = articles[offset:offset + limit]
            # 廣泛閱讀時只給短摘要，避免一次回傳兩日數百篇全文；
            # Agent 若發現研究價值，再用 search_cnyes_news 取得完整正文。
            compact_articles = []
            for x in batch:
                compact_articles.append({
                    "article_id": x.get("article_id", ""),
                    "title": x.get("title", ""),
                    "url": x.get("url", ""),
                    "category": x.get("category", ""),
                    "published": x.get("published", ""),
                    "summary": str(x.get("summary", "") or x.get("content", ""))[:220],
                    "tags": (x.get("tags") or [])[:8],
                })
            return {
                "offset": offset,
                "limit": limit,
                "total": len(articles),
                "has_more": offset + len(batch) < len(articles),
                "articles": compact_articles,
            }

        if name == "get_theme_evidence":
            theme = str(args.get("theme", "")).strip().lower()
            rows = []
            for x in payload.get("bullish_themes", [])[:40]:
                if not isinstance(x, dict):
                    continue
                text = json.dumps(x, ensure_ascii=False).lower()
                if theme and theme in text:
                    rows.append(x)
            return {"theme": theme, "matches": rows[:10]}

        if name == "get_model_health":
            return {
                "model_health": payload.get("model_health", []),
                "retrain_flags": payload.get("retrain_flags", []),
                "research_conclusion": payload.get("research_conclusion", ""),
            }

        if name == "get_company_research":
            symbol = _clean_symbol(args.get("symbol"))
            rows = [x for x in payload.get("company_research", []) if isinstance(x, dict) and _clean_symbol(x.get("symbol")) == symbol]
            if not rows:
                profiles = payload.get("business_profiles", {}) or {}
                profile = profiles.get(symbol, {})
                rows = [{"symbol": symbol, **profile}] if profile else []
            return {"symbol": symbol, "matches": rows[:4]}

        if name == "get_previous_notes":
            memory = self._load_memory()
            latest = memory.get("latest", memory) if isinstance(memory, dict) else {}
            return {
                "saved_at": latest.get("saved_at", ""),
                "report_date": latest.get("report_date", ""),
                "overview": latest.get("overview", ""),
                "research_notes": latest.get("research_notes", [])[:20],
                "watch_items": latest.get("watch_items", [])[:10],
                "earnings_digest": latest.get("earnings_digest", ""),
                "financial_focus": latest.get("financial_focus", []),
                "cnyes_news_digest": latest.get("cnyes_news_digest", []),
                "model_alert": latest.get("model_alert", ""),
                "agent_actions": latest.get("agent_actions", [])[-15:],
            }

        return {"ok": False, "error": f"未知工具：{name}"}
