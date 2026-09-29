# -*- coding: utf-8 -*-
"""
AI 台股盤後研究助理 Agent
============================================================

核心概念：
    一個 AI 研究助理幫使用者讀完一堆資料後，
    自己決定哪些事情值得記錄、哪些需要標記、哪些需要進一步查證。

ResearchAgent 負責資料工程與證據蒐集；本模組負責 Agent Orchestration：
Qwen3 8B 透過 Tool Calling 自主決定要看哪些資料，再產生可保存的研究筆記。

Agent 可使用：
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
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import numpy as np
import requests


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_MAX_TOOL_ROUNDS = 14
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


def _extract_json(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _assistant_message(resp: Any) -> dict[str, Any]:
    msg = getattr(resp, "message", None)
    out: dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
    calls: list[dict[str, Any]] = []
    for call in getattr(msg, "tool_calls", None) or []:
        try:
            args = call.function.arguments
            if not isinstance(args, dict):
                args = json.loads(str(args))
            calls.append({"function": {"name": str(call.function.name), "arguments": args}})
        except Exception:
            continue
    if calls:
        out["tool_calls"] = calls
    return out


def _call_name(call: Any) -> str:
    try:
        return str(call.function.name)
    except Exception:
        return ""


def _call_args(call: Any) -> dict[str, Any]:
    try:
        args = call.function.arguments
        if isinstance(args, dict):
            return args
        value = json.loads(str(args))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


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
        "agent_status": "qwen3",
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


class ResearchOrchestratorAgent:
    """Qwen3 自主研究／工具選擇／研究筆記 Agent。"""

    def __init__(self, base_dir: str | Path = ".", ollama_host: str | None = None, ollama_model: str | None = None):
        self.base = Path(base_dir).resolve()
        self.ollama_host = (ollama_host or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.ollama_model = ollama_model or DEFAULT_OLLAMA_MODEL
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
            self._earnings_agent = EarningsCallAgent(
                self.base,
                ollama_host=self.ollama_host,
                ollama_model=self.ollama_model,
            )
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
            # 廣泛閱讀時只給短摘要，避免將兩日數百篇全文一次塞進 Qwen context；
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

    def _fallback(self, payload: dict[str, Any], error: str) -> dict[str, Any]:
        mm = payload.get("market_movers", {}) or {}
        movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]
        movers.sort(key=lambda x: abs(_safe_float(x.get("today_change_percent"), 0) or 0), reverse=True)
        calls = [x for x in payload.get("earnings_calls", []) if isinstance(x, dict) and not x.get("error")]
        notes = []
        for x in movers[:6]:
            notes.append({
                "importance": "medium",
                "type": "market",
                "symbol": _clean_symbol(x.get("symbol")),
                "title": "市場波動候選",
                "note": f"今日變動 {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%；可能原因：{str(x.get('reason', '待查證'))[:250]}",
                "evidence": [f"今日變動 {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%"],
                "tags": ["市場波動", "待查證"],
                "confidence": 35.0,
                "follow_up": ["需要公司公告或多來源新聞確認原因"],
            })
        # 讓即使主研究 Agent 進入 fallback，夜間新聞 Agent 的研究也不會消失。
        cnyes_research = payload.get("cnyes_research_digest", {}) or {}
        for item in cnyes_research.get("key_findings", []) or []:
            if not isinstance(item, dict) or not item.get("summary"):
                continue
            notes.append({
                "importance": "medium",
                "type": "industry",
                "symbol": _clean_symbol((item.get("related_symbols") or [""])[0]) if item.get("related_symbols") else "",
                "title": f"新聞研究｜{str(item.get('title','鉅亨新聞')).strip()[:90]}",
                "note": (str(item.get("summary", "")) + (f"；研究意義：{item.get('why_relevant')}" if item.get("why_relevant") else ""))[:1200],
                "evidence": [str(x) for x in (item.get("evidence") or [])][:8] or ["鉅亨近兩日新聞快取"],
                "tags": ["鉅亨新聞", str(item.get("impact", "待查證"))],
                "confidence": max(0.0, min(1.0, float(item.get("confidence") or 0))),
                "follow_up": ["與公司公告、財報、法人或其他產業資料交叉驗證"],
                "source_links": [str(x) for x in (item.get("source_links") or []) if str(x).strip()][:8],
            })

        cnyes_digest = []
        cnyes_research = payload.get("cnyes_research_digest", {}) or {}
        if isinstance(cnyes_research, dict):
            for item in cnyes_research.get("key_findings", []) or []:
                if isinstance(item, dict) and item.get("summary"):
                    text = str(item.get("summary"))
                    if item.get("why_relevant"):
                        text += f"；{item.get('why_relevant')}"
                    cnyes_digest.append(text[:700])
        if not cnyes_digest:
            cnyes = payload.get("cnyes_news", {}) or {}
            for item in (cnyes.get("articles", []) or [])[:8]:
                if isinstance(item, dict) and item.get("title"):
                    cnyes_digest.append(f"{item.get('title')}（鉅亨網，{item.get('published', '')}）")

        return {
            "agent_status": "fallback",
            "model": self.ollama_model,
            "generated_at": datetime.now().isoformat(),
            "overview": "Qwen3 研究助理本次無法完成自主工具分析；以下僅為保守資料摘要。",
            "key_takeaways": [f"近期市場波動候選 {len(movers)} 筆", f"近期法說會可用資料 {len(calls)} 筆"],
            "research_notes": notes,
            "positive_signals": [],
            "negative_signals": [],
            "watch_items": [f"Agent 錯誤：{error}"],
            "earnings_digest": "；".join(f"{x.get('symbol')} {x.get('one_line_summary') or x.get('summary','')}" for x in calls[:5]),
            "financial_focus": [],
            "cnyes_news_digest": cnyes_digest[:8],
            "model_alert": "",
            "agent_actions": [],
            "error": error,
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import ollama
        except Exception as exc:
            return self._fallback(payload, f"ollama 套件不可用：{exc}")

        client = ollama.Client(host=self.ollama_host)
        previous_memory = self._load_memory()
        evidence_index = {
            "研究日期": payload.get("report_date", ""),
            "研究範圍": payload.get("research_scope", {}),
            "研究包已存在的資料": {
                "market_movers": len((payload.get("market_movers", {}) or {}).get("movers", []) or []),
                "financial_snapshots": len(payload.get("financial_snapshots", []) or []),
                "earnings_calls": len(payload.get("earnings_calls", []) or []),
                "news": len(payload.get("news", []) or []),
                "cnyes_news": int((payload.get("cnyes_news", {}) or {}).get("article_count", 0) or 0),
                "bullish_themes": len(payload.get("bullish_themes", []) or []),
                "company_research": len(payload.get("company_research", []) or []),
                "model_health": len(payload.get("model_health", []) or []),
            },
            "可用工具": [x["function"]["name"] for x in TOOLS],
            "上一期筆記存在": bool(previous_memory),
        }

        system_prompt = """
你是「AI 台股盤後研究助理」，不是報表排版器，也不是只會重述資料的摘要器。

你的工作方式是：先讀研究包 → 發現問題／異常 → 主動選擇工具補證據 → 必要時上網查近期新聞 → 交叉比對 → 最後整理成研究筆記。
研究的目標不是把所有資料列出來，而是回答：「今天市場到底發生了什麼？為什麼？哪些只是題材、哪些有基本面支撐、哪些仍然不知道？」

【最重要的研究規則】
1. 你必須先使用 Tool Calling 取得具體證據，再輸出最後 JSON。至少 1 次；視研究問題可使用多個不同工具。
2. 你不是被固定工具流程綁住，但有「條件式必查」：
   - 如果看到某個產業／個股明顯上漲或下跌，而原因不清楚、只有「可能」或「待查證」，就要主動 search_news_evidence 或 search_live_news。
   - 對產業異動，搜尋時要從「股價異動 → 新聞原因 → 供應鏈／產業題材 → 財務或公司資料」一路追下去。例如電子零組件族群上漲，不要停在「電子零組件上漲」，要檢查 PCB、IC 載板、ABF/BT、CCL、高速材料、AI 伺服器等可能的產業驅動因素，哪一個有近期新聞支持，再與公司／產業資料交叉比對。
   - 如果財報／營收資料缺失、過時或不足以解釋市場變化，就主動 get_company_financial；若工具仍沒有資料，再 search_live_news 查近期營收、EPS、需求、訂單等公開資訊，並明確標記來源與查證程度。
   - 如果題材很強但原因不清楚，主動 get_theme_evidence；若研究包證據不足，再 search_live_news。
   - 如果法說會看起來會影響股價，就先 get_earnings_events，再視需要 read_fugle_memo 讀詳細正文。
3. 上網查詢不是最後才使用的裝飾功能。只要你真的不知道、現有資料不足、或需要解釋近期市場異動，就應該上網查。沒有網路證據時，不准假裝知道。
   - 本研究包中的鉅亨新聞是「前一晚 23:00 已完成的日檔快取」，盤後研究時間不應再次爬網站。若 cnyes_news > 0，先使用 get_cnyes_research_digest 讀取已完成的新聞 Agent 摘要，再依需要用 get_cnyes_news_overview / search_cnyes_news 深讀原文。
   - 對「為什麼某產業今天上漲／下跌」這類問題，優先從鉅亨近兩日新聞尋找市場原因。例如電子零組件族群上漲，要查是否由載板、ABF/BT、PCB、CCL、高速材料、AI 伺服器等新聞驅動，而不是停在產業漲幅描述。
   - 你不是要把所有新聞逐篇重複到 Email；你要從大量新聞中找出有關聯的事件、共同敘事、矛盾資訊與真正值得研究的因果鏈。
4. 新聞是線索，不是自動等於事實。優先尋找多來源、官方公告、公司說法、財務數據或產業公開資訊交叉驗證；若只有單一媒體報導，confidence 要降低，並在 follow_up 標示查證事項。
5. 主動尋找「因果鏈」與「矛盾訊號」，例如：
   - 產業上漲，但真正帶動的是載板／CCL／散熱／電源中的哪一段？
   - 股價上漲，但營收沒有跟上，是否只是題材？
   - 營收大增，但市場股價沒有反應，是否已反映或存在其他風險？
   - 法說偏正向，但新聞或產業供給訊息是否相反？
6. 不要只寫「市場漲跌參半」「族群受到關注」等空泛句。每個重要筆記盡量回答「發生什麼、為什麼、證據是什麼、還缺什麼」。
7. 自己決定研究筆記數量，不要硬湊 3 個、5 個或固定數量。
8. 每個筆記自行決定 importance、type、tags。type 可使用 market|financial|earnings|theme|risk|model|contradiction|industry|event|data_quality|other。
9. 每個重要筆記都應有 evidence；如果有新聞來源，盡量在 source_links 放入實際網址；如果沒有足夠證據，就降低 confidence 並標記待查證。
10. confidence 使用 0～100 的百分比數字：多來源＋直接數據可高；單一新聞或間接推論要低；沒有可用證據才可以是 0。不要把 0 當成預設值。
11. 可以參考上一期 Agent 筆記，但上一期不是今日事實；今日結論必須由今日工具重新驗證。
12. 不得捏造數字、新聞、公司關係、管理層說法或因果關係。無法證明就說「尚未確認」。
13. 不輸出 chain-of-thought、隱藏推理或逐步思考，只輸出最終研究結果。
14. 不提供買進、賣出、持有、投資建議，也不做股票排名。
15. 若同一股票代號對應到矛盾公司名稱，標記「資料一致性問題／需要查證」，不能自行猜測。

【研究輸出】
最後只輸出 JSON：
{
  "overview": "一句話總結今天市場最重要的研究發現",
  "key_takeaways": ["Agent 自主挑出的重要發現；可自行決定數量"],
  "research_notes": [
    {
      "importance": "high|medium|low",
      "type": "market|financial|earnings|theme|risk|model|contradiction|industry|event|data_quality|other",
      "symbol": "可空白",
      "title": "研究筆記標題",
      "note": "簡潔說明發生什麼＋為何發生＋目前證據強度；不要空泛描述",
      "evidence": ["具體資料、新聞標題／來源、官方資料或交叉驗證結果"],
      "source_links": ["若有新聞／公開資料網址就填入"],
      "tags": ["Agent 自己決定的記號"],
      "confidence": 0,
      "follow_up": ["後續要觀察或查證的事項"]
    }
  ],
  "positive_signals": ["真正值得記錄的正向訊號；可空"],
  "negative_signals": ["真正值得記錄的負向訊號；可空"],
  "watch_items": ["真正值得持續追蹤或需要查證的事項；可空"],
  "earnings_digest": "不要重複列出每一篇法說摘要，只保留跨法說的共同影響或新發現；如果 Email 已有最近5天法說會區塊，可寫空字串",
  "financial_focus": ["只放本次額外查到、且值得補充的財報／營收研究；不要重複 Email 結構化表格已有的內容"],
  "cnyes_news_digest": ["從鉅亨近兩日新聞中整理出的市場／產業研究重點；不要逐篇複述，應整理成事件、原因、影響與查證狀態"],
  "model_alert": "只有真的存在值得提醒的模型異常才填；否則空字串"
}
        """.strip()
        if _env_bool("AFTER_CLOSE_FAST_MODE", False):
            system_prompt += "\n\n【快速盤後模式】本次研究以速度優先；禁止使用 search_live_news。鉅亨新聞只讀本機夜間快取與早報已完成的新聞摘要。仍需使用工具查證，但不得重新爬網站或重複進行新聞深度摘要。"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "先開始自主研究。這是研究包索引，不是答案。"
                    "你必須先使用工具取得具體證據，再整理最終研究筆記。"
                    "若研究包有 CNYES 近兩日新聞，第一個工具呼叫請使用 get_cnyes_news_overview，然後再自主選擇最需要的新聞搜尋／批次與其他工具。"
                    "不要把所有工具機械式呼叫一次。\n\n"
                    + json.dumps(evidence_index, ensure_ascii=False, default=str)
                ),
            },
        ]
        tool_history: list[dict[str, Any]] = []
        had_tool_call = False
        no_tool_rounds = 0
        try:
            max_tool_rounds = max(2, min(DEFAULT_MAX_TOOL_ROUNDS, int(os.getenv("AGENT_MAX_TOOL_ROUNDS", str(DEFAULT_MAX_TOOL_ROUNDS)))))
        except Exception:
            max_tool_rounds = DEFAULT_MAX_TOOL_ROUNDS

        for round_no in range(max_tool_rounds):
            try:
                resp = client.chat(
                    model=self.ollama_model,
                    messages=messages,
                    tools=TOOLS,
                    think=True,
                    options={"temperature": 0.1},
                )
            except TypeError:
                resp = client.chat(
                    model=self.ollama_model,
                    messages=messages,
                    tools=TOOLS,
                    options={"temperature": 0.1},
                )

            calls = getattr(getattr(resp, "message", None), "tool_calls", None) or []

            if not calls:
                no_tool_rounds += 1
                raw = _extract_json(getattr(getattr(resp, "message", None), "content", "") or "")

                if raw is not None and had_tool_call:
                    cnyes_count = int(evidence_index["研究包已存在的資料"].get("cnyes_news", 0) or 0)
                    cnyes_actions = {"get_cnyes_research_digest", "search_cnyes_news", "get_cnyes_news_batch"}
                    if cnyes_count > 0 and not any(x.get("tool") in cnyes_actions for x in tool_history):
                        messages.append({
                            "role": "assistant",
                            "content": getattr(getattr(resp, "message", None), "content", "") or "",
                        })
                        messages.append({
                            "role": "user",
                            "content": (
                                "研究包已有鉅亨近兩日新聞，但你尚未實際搜尋／閱讀新聞內容。"
                                "先使用 get_cnyes_news_batch 廣泛閱讀兩日新聞，再針對最需要解釋的市場／產業異動使用 search_cnyes_news 深讀，最後才整理 JSON。"
                            ),
                        })
                        continue
                    result = _normalize(raw, tool_history, self.ollama_model)
                    result["report_date"] = payload.get("report_date", "")
                    result["agent_status"] = "qwen3_tool_agent"
                    result = _merge_cnyes_digest(result, payload)
                    self._save_memory(result)
                    return result

                # Qwen3 在第一輪可能直接回答而沒有產生 tool call。
                # 不接受這種答案；先給一次明確的工具使用要求。
                messages.append({
                    "role": "assistant",
                    "content": getattr(getattr(resp, "message", None), "content", "") or "",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "你目前尚未使用任何研究工具。請不要直接下結論。"
                        "現在請根據研究目標自行選擇最需要的工具，至少實際查一項市場、財務、法說會、新聞、題材或模型證據。"
                        "取得工具結果後再繼續研究。"
                    ),
                })
                if no_tool_rounds >= 2 and not had_tool_call:
                    # 最後一道保險：只在模型連續不使用工具時，建立一個最小的市場/法說探查，
                    # 探查內容本身仍不決定最後研究重點。
                    if (evidence_index["研究包已存在的資料"].get("cnyes_news", 0) or 0) > 0:
                        starter_name = "get_cnyes_research_digest"
                        starter_args = {}
                    elif (evidence_index["研究包已存在的資料"].get("earnings_calls", 0) or 0) > 0:
                        starter_name = "get_earnings_events"
                        starter_args = {"symbol": ""}
                    else:
                        starter_name = "get_market_overview"
                        starter_args = {"focus": "找出需要深入研究的異常訊號"}
                    starter_result = self._tool(starter_name, starter_args, payload)
                    compact = json.dumps(starter_result, ensure_ascii=False, default=str)
                    tool_history.append({
                        "tool": starter_name,
                        "reason": "bootstrap：模型連續兩輪未產生工具呼叫，先提供最低限度偵察資料；最終重點仍由 Agent 決定。",
                        "result": compact,
                    })
                    messages.append({"role": "tool", "tool_name": starter_name, "content": compact[:52000]})
                    had_tool_call = True
                continue

            # Ollama 官方 tool-calling 流程要求先把 assistant message（含 tool_calls）
            # 放回 messages，再逐一加入 tool result。
            messages.append(resp.message)

            for call in calls:
                name = _call_name(call)
                args = _call_args(call)
                result = self._tool(name, args, payload)
                compact = json.dumps(result, ensure_ascii=False, default=str)
                tool_history.append({
                    "tool": name,
                    "reason": f"Agent 自主選擇工具；arguments={json.dumps(args, ensure_ascii=False)}",
                    "result": compact,
                })
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": compact[:52000],
                })
                had_tool_call = True

        # 工具輪數用完後，要求 Agent 只輸出最後 JSON。
        try:
            resp = client.chat(
                model=self.ollama_model,
                messages=messages + [{"role": "user", "content": "停止查資料，現在輸出最後 JSON 研究結果。不要輸出思考過程。只輸出最終研究筆記。"}],
                tools=TOOLS,
                options={"temperature": 0.0},
                format="json",
                think=True,
            )
            raw = _extract_json(getattr(getattr(resp, "message", None), "content", "") or "")
            if raw is not None:
                result = _normalize(raw, tool_history, self.ollama_model)
                result["report_date"] = payload.get("report_date", "")
                result["agent_status"] = "qwen3_tool_agent" if tool_history else "qwen3"
                result = _merge_cnyes_digest(result, payload)
                self._save_memory(result)
                return result
        except Exception as exc:
            return self._fallback(payload, f"Agent 最終 JSON 失敗：{exc}")

        return self._fallback(payload, "Agent 未產生有效 JSON 結果")
