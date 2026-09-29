# -*- coding: utf-8 -*-
"""AI 台股早報研究 Agent

每日早報只做三件事：
1. 最近 5 天法說會
2. 最近 2 天鉅亨財經報導摘要
3. Agent 根據新聞、法說會、產業與公司資料整理「值得關注的產業／股票」與研究原因

設計重點：
- CNYES 新聞由前一晚 23:00 夜間爬蟲建立日檔；早報只讀本機快取。
- 法說會優先使用前一日盤後已建立的 Fugle memo cache；若沒有可用 cache 才補抓。
- 產業／股票研究由本機 Qwen3 8B 完成，不寫死固定產業排名。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from cnyes_news_crawler import CnyesNewsCrawler
from cnyes_news_agent import CnyesNewsDigestAgent
from earnings_call_agent import EarningsCallAgent
from business_master import business_map_for_symbols, build_business_research_context

TAIPEI = ZoneInfo("Asia/Taipei")
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_WATCHLIST = [
    "3481", "2327", "2492", "3037", "3044", "3533", "2303", "2330", "2344", "2408",
    "6515", "7769", "6488", "3374", "2377", "3450", "0050", "3006", "3661", "8299",
    "2308", "6274",
]


def _clean_symbol(v: Any) -> str:
    s = str(v or "").strip().upper()
    if s.endswith(".TW"):
        s = s[:-3]
    return s


def _clean_text(v: Any, limit: int = 2000) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:limit]


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x
    except Exception:
        return default


class MorningResearchAgent:
    def __init__(self, base_dir: str | Path = ".", ollama_host: str | None = None, ollama_model: str | None = None):
        self.base = Path(base_dir).resolve()
        self.output_dir = self.base / "output" / "research_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_host = (ollama_host or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).rstrip("/")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    def _load_latest_research(self) -> dict[str, Any]:
        files = sorted(self.output_dir.glob("research_*.json"))
        for p in reversed(files):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return {}

    def _load_recent_earnings_cache(self, days: int = 5, watchlist: list[str] | None = None) -> dict[str, Any]:
        files = sorted(self.output_dir.glob("fugle_earnings_memo_*.json"), reverse=True)
        cutoff = datetime.now(TAIPEI).date() - timedelta(days=max(1, int(days)))
        chosen: Path | None = None
        for p in files:
            m = re.search(r"fugle_earnings_memo_(20\d{2}-\d{2}-\d{2})\.json$", p.name)
            if not m:
                continue
            try:
                d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            except Exception:
                continue
            if d >= cutoff:
                chosen = p
                break
        if chosen:
            try:
                data = json.loads(chosen.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data["cache_file"] = str(chosen)
                    return data
            except Exception:
                pass
        # 沒有現成 cache：補抓一次，但 EarningsCallAgent 本身使用 DB 避免重複分析。
        try:
            agent = EarningsCallAgent(self.base, ollama_model=self.ollama_model)
            out = agent.daily_run(
                days=int(os.getenv("EARNINGS_MEMO_LOOKBACK_DAYS", "5")),
                limit=int(os.getenv("EARNINGS_MEMO_MAX_ARTICLES", "80")),
                force=_clean_text(os.getenv("EARNINGS_FORCE_REFRESH", "false")).lower() in {"1", "true", "yes", "on"},
                watchlist=watchlist or [],
            )
            return out if isinstance(out, dict) else {}
        except Exception as exc:
            return {"items": [], "errors": [{"error": f"早報法說會 cache 補抓失敗：{exc}"}], "daily_digest": {}}

    def _load_cnyes_cache(self, report_date: str) -> dict[str, Any]:
        crawler = CnyesNewsCrawler(
            self.base,
            max_articles=int(os.getenv("CNYES_NIGHTLY_MAX_ARTICLES", "600")),
            max_scrolls=int(os.getenv("CNYES_NEWS_MAX_SCROLLS", "18")),
        )
        try:
            return crawler.load_recent_cached(
                before_date=report_date,
                days=int(os.getenv("CNYES_MORNING_NEWS_DAYS", "2")),
                max_search_days=int(os.getenv("CNYES_RESEARCH_CACHE_SEARCH_DAYS", "7")),
            )
        except Exception as exc:
            return {
                "source_url": "https://news.cnyes.com/news/cat/headline",
                "source_method": "nightly_cache",
                "cache_dates": [],
                "cache_files": [],
                "article_count": 0,
                "articles": [],
                "errors": [str(exc)],
                "loaded_from_cache": True,
            }

    def _build_payload(self, watchlist: list[str]) -> dict[str, Any]:
        now = datetime.now(TAIPEI)
        report_date = now.strftime("%Y-%m-%d")
        research = self._load_latest_research()
        earnings = self._load_recent_earnings_cache(5, watchlist)
        cnyes = self._load_cnyes_cache(report_date)

        try:
            profiles_raw = business_map_for_symbols(watchlist, path=self.base / "data" / "business_master.csv", force=False)
            profiles = {k: build_business_research_context(v) for k, v in profiles_raw.items()}
        except Exception:
            profiles = {}

        # 早報新聞先由專門 CNYES Agent 做一次「兩日新聞研究」，主 Agent 再整合。
        digest_agent = CnyesNewsDigestAgent(self.base, self.ollama_host, self.ollama_model)
        digest_payload = {
            "report_date": report_date,
            "symbols": watchlist,
            "cnyes_news": cnyes,
            "market_movers": research.get("market_movers", {}),
        }
        cnyes_digest = digest_agent.run(
            digest_payload,
            limit=int(os.getenv("CNYES_MORNING_AGENT_INPUT_ARTICLES", "100")),
        )

        return {
            "report_date": report_date,
            "generated_at": now.isoformat(),
            "report_type": "morning",
            "symbols": watchlist,
            "business_profiles": profiles,
            "earnings_calls": [x for x in (earnings.get("items", []) or []) if isinstance(x, dict) and not x.get("error")],
            "earnings_memo_report": earnings,
            "cnyes_news": cnyes,
            "cnyes_research_digest": cnyes_digest,
            "previous_research": {
                "bullish_themes": research.get("bullish_themes", [])[:12],
                "financial_snapshots": research.get("financial_snapshots", [])[:20],
                "market_movers": research.get("market_movers", {}),
            },
        }

    def _fallback(self, payload: dict[str, Any], error: str) -> dict[str, Any]:
        profiles = payload.get("business_profiles", {}) or {}
        themes = payload.get("previous_research", {}).get("bullish_themes", []) or []
        stocks = []
        for sym, p in profiles.items():
            if not isinstance(p, dict):
                continue
            stocks.append({
                "symbol": sym,
                "name": p.get("name", sym),
                "business": p.get("business_group") or p.get("primary_chain") or p.get("industry_name") or "",
                "reason": "公司業務與近期產業研究可作為早報觀察標的，仍需自行確認最新資訊。",
                "evidence": [],
            })
            if len(stocks) >= 8:
                break
        return {
            "agent_status": "morning_fallback",
            "model": self.ollama_model,
            "generated_at": datetime.now(TAIPEI).isoformat(),
            "overview": "早報 Agent 本次未完成完整自主整理；已保留最近兩日新聞與法說會資料。",
            "news_summary": [x.get("summary", "") for x in (payload.get("cnyes_research_digest", {}).get("key_findings", []) or []) if isinstance(x, dict) and x.get("summary")][:8],
            "recommended_industries": [{
                "industry": x.get("theme", ""),
                "why": "前一日研究題材，早報中列為待觀察。",
                "evidence": [],
            } for x in themes[:5] if isinstance(x, dict)],
            "recommended_stocks": stocks,
            "watch_items": [error],
            "source_links": [],
            "error": error,
        }

    def run(self, watchlist: list[str], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload if isinstance(payload, dict) else self._build_payload(watchlist)
        digest = payload.get("cnyes_research_digest", {}) or {}
        earnings = payload.get("earnings_calls", []) or []
        profiles = payload.get("business_profiles", {}) or {}
        prev = payload.get("previous_research", {}) or {}
        theme_hint = prev.get("bullish_themes", []) or []
        financial_hint = prev.get("financial_snapshots", []) or []

        try:
            import ollama
        except Exception as exc:
            return self._fallback(payload, f"ollama 套件不可用：{exc}")

        earnings_compact = []
        for x in earnings[:20]:
            earnings_compact.append({
                "date": x.get("event_date"),
                "symbol": x.get("symbol"),
                "name": x.get("name"),
                "impact": x.get("impact"),
                "confidence": x.get("confidence"),
                "summary": x.get("one_line_summary") or x.get("summary"),
                "financial": (x.get("financial_highlights") or [])[:4],
                "outlook": (x.get("guidance") or [])[:4],
                "risks": (x.get("negative_factors") or [])[:4],
            })

        profile_compact = {
            str(sym): {
                "name": p.get("name", ""),
                "industry_name": p.get("industry_name", ""),
                "business_group": p.get("business_group", ""),
                "primary_chain": p.get("primary_chain", ""),
                "research_keywords": (p.get("research_keywords") or [])[:8],
            }
            for sym, p in profiles.items() if isinstance(p, dict)
        }

        prompt = f"""
你是「AI 台股早報研究 Agent」。

早報日期：{payload.get('report_date')}
研究自選股：{watchlist}

早報只有三個任務：
1. 最近 5 天法說會：法說會完整內容會由 Email 的獨立區塊呈現；你只需抓跨公司共同趨勢或值得特別注意的因素，不要逐篇重複。
2. 最近 2 天財經報導摘要：必須根據已經由 23:00 夜間爬蟲保存的鉅亨新聞與 CNYES News Agent 摘要，整理真正重要的市場／產業事件。
3. Agent 觀察的產業與股票：自行判斷近期值得關注的產業類別（例如 PCB、CCL、MLCC、IC 代工、封測、記憶體、AI 伺服器等），並從自選股中挑出與這些產業有直接業務關聯的股票，說明「它做什麼＋為什麼值得早報關注」。不要使用買進／賣出指令，也不要寫成投資保證。

重要規則：
- 優先使用輸入資料；不能杜撰新聞、法說內容、公司業務。
- 新聞是市場研究線索，若只是單一新聞敘事，請標示為「待查證」。
- 公司產業請使用提供的 business_profiles，不要自行猜測公司業務。
- 「推薦股票」其實是「研究觀察股票」，要給產業／業務與理由，不要寫目標價。
- 不要把法說會完整摘要重複進 news_summary。
- news_summary 應該像財經早報，而不是新聞標題清單：要回答「發生了什麼、影響哪個產業、為什麼值得注意」。

請只輸出 JSON：
{{
  "overview": "早報一句話總結",
  "news_summary": [
    {{
      "title": "財經事件",
      "summary": "事件摘要",
      "impact": "利多|利空|混合|待查證",
      "industries": ["產業"],
      "evidence": ["鉅亨新聞標題／日期"],
      "source_links": ["輸入資料中的 URL"],
      "confidence": 0.0
    }}
  ],
  "recommended_industries": [
    {{
      "industry": "產業類別",
      "why": "為什麼近期值得關注",
      "evidence": ["新聞／法說／公司資料"],
      "related_symbols": ["股票代號"]
    }}
  ],
  "recommended_stocks": [
    {{
      "symbol": "股票代號",
      "name": "公司名稱",
      "industry": "產業／業務類別",
      "reason": "這家公司做什麼，以及為什麼被本次早報選為研究觀察標的",
      "evidence": ["具體資料"],
      "source_links": ["輸入資料中的 URL"]
    }}
  ],
  "watch_items": ["需要後續追蹤的事件"]
}}

【CNYES 近兩日新聞 Agent 摘要】
{json.dumps(digest, ensure_ascii=False, default=str)}

【最近 5 天法說會】
{json.dumps(earnings_compact, ensure_ascii=False, default=str)}

【自選股公司與產業資料】
{json.dumps(profile_compact, ensure_ascii=False, default=str)}

【前一日研究可用線索】
題材：{json.dumps(theme_hint, ensure_ascii=False, default=str)}
營收：{json.dumps(financial_hint, ensure_ascii=False, default=str)}
""".strip()

        try:
            client = ollama.Client(host=self.ollama_host)
            try:
                resp = client.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": "你是本機 AI 台股早報研究 Agent，只能使用輸入的資料。"},
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
                        {"role": "system", "content": "你是本機 AI 台股早報研究 Agent，只能使用輸入的資料。"},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.05},
                    format="json",
                )
            raw_text = getattr(getattr(resp, "message", None), "content", "") or ""
            raw = json.loads(raw_text)
            if not isinstance(raw, dict):
                raise ValueError("早報 Agent 輸出不是 JSON object")

            valid_urls = {str(a.get("url", "")).strip() for a in (payload.get("cnyes_news", {}).get("articles", []) or []) if isinstance(a, dict) and a.get("url")}
            news_summary = []
            for x in raw.get("news_summary", []) or []:
                if not isinstance(x, dict):
                    continue
                links = [u for u in (x.get("source_links") or []) if str(u).strip() in valid_urls]
                news_summary.append({
                    "title": _clean_text(x.get("title", "財經事件"), 120),
                    "summary": _clean_text(x.get("summary", ""), 900),
                    "impact": _clean_text(x.get("impact", "待查證"), 20),
                    "industries": [_clean_text(v, 80) for v in (x.get("industries") or []) if str(v).strip()][:8],
                    "evidence": [_clean_text(v, 500) for v in (x.get("evidence") or []) if str(v).strip()][:6],
                    "source_links": links[:4],
                    "confidence": max(0.0, min(1.0, _safe_float(x.get("confidence"), 0.0) or 0.0)),
                })

            recommended_industries = []
            for x in raw.get("recommended_industries", []) or []:
                if not isinstance(x, dict):
                    continue
                syms = [_clean_symbol(v) for v in (x.get("related_symbols") or []) if _clean_symbol(v) in watchlist]
                recommended_industries.append({
                    "industry": _clean_text(x.get("industry", ""), 100),
                    "why": _clean_text(x.get("why", ""), 800),
                    "evidence": [_clean_text(v, 500) for v in (x.get("evidence") or []) if str(v).strip()][:6],
                    "related_symbols": syms[:15],
                })

            recommended_stocks = []
            profile_lookup = {str(k).strip().upper(): v for k, v in profiles.items()}
            for x in raw.get("recommended_stocks", []) or []:
                if not isinstance(x, dict):
                    continue
                sym = _clean_symbol(x.get("symbol"))
                if not sym or sym not in watchlist:
                    continue
                p = profile_lookup.get(sym, {}) or {}
                name = _clean_text(x.get("name") or p.get("name") or sym, 80)
                industry = _clean_text(x.get("industry") or p.get("business_group") or p.get("primary_chain") or p.get("industry_name") or "", 120)
                recommended_stocks.append({
                    "symbol": sym,
                    "name": name,
                    "industry": industry,
                    "reason": _clean_text(x.get("reason", ""), 900),
                    "evidence": [_clean_text(v, 500) for v in (x.get("evidence") or []) if str(v).strip()][:6],
                    "source_links": [str(v).strip() for v in (x.get("source_links") or []) if str(v).strip() in valid_urls][:4],
                })

            result = {
                "agent_status": "qwen3_morning_agent",
                "model": self.ollama_model,
                "generated_at": datetime.now(TAIPEI).isoformat(),
                "overview": _clean_text(raw.get("overview", ""), 1200),
                "news_summary": news_summary[:10],
                "recommended_industries": recommended_industries[:10],
                "recommended_stocks": recommended_stocks[:12],
                "watch_items": [_clean_text(x, 500) for x in (raw.get("watch_items") or []) if str(x).strip()][:10],
                "cnyes_cache_dates": payload.get("cnyes_news", {}).get("cache_dates", []),
                "cnyes_article_count": int(payload.get("cnyes_news", {}).get("article_count", 0) or 0),
                "earnings_count": len(earnings),
            }
            if not result["news_summary"] and not result["recommended_industries"] and not result["recommended_stocks"]:
                raise ValueError("早報 Agent 沒有產生有效內容")
            return result
        except Exception as exc:
            return self._fallback(payload, str(exc))


def build_morning_report(base_dir: str | Path = ".", watchlist: list[str] | None = None) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    symbols = [_clean_symbol(x) for x in (watchlist or DEFAULT_WATCHLIST) if _clean_symbol(x)]
    agent = MorningResearchAgent(base)
    payload = agent._build_payload(symbols)
    morning = agent.run(symbols, payload=payload)
    payload["morning_agent"] = morning
    payload["agent_research_notes"] = {
        "agent_status": morning.get("agent_status", ""),
        "overview": morning.get("overview", ""),
        "cnyes_news_digest": [x.get("summary", "") for x in morning.get("news_summary", []) if isinstance(x, dict)],
        "research_notes": [],
        "positive_signals": [],
        "negative_signals": [],
        "watch_items": morning.get("watch_items", []),
    }
    out_path = agent.output_dir / f"morning_{payload['report_date']}.json"
    payload["json_path"] = str(out_path)
    md_path = agent.output_dir / f"morning_{payload['report_date']}.md"
    payload["markdown_path"] = str(md_path)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# AI 台股早報｜{payload['report_date']}",
        "",
        f"## 早報 Agent 總結\n{morning.get('overview', '')}",
        "",
        "## 1. 最近 5 天法說會",
    ]
    for x in payload.get("earnings_calls", [])[:20]:
        lines.append(f"- {x.get('event_date')} {x.get('name', x.get('symbol'))}({x.get('symbol')})｜{x.get('impact')}｜{x.get('one_line_summary') or x.get('summary', '')}")
    if not payload.get("earnings_calls"):
        lines.append("- 最近 5 天沒有可用的法說會快取資料。")
    lines += ["", "## 2. 最近 2 天財經報導摘要"]
    for x in morning.get("news_summary", [])[:10]:
        if isinstance(x, dict):
            lines.append(f"- **{x.get('title','')}**｜{x.get('impact','待查證')}｜{x.get('summary','')}")
    lines += ["", "## 3. Agent 觀察產業"]
    for x in morning.get("recommended_industries", [])[:10]:
        lines.append(f"- **{x.get('industry','')}**｜{x.get('why','')}")
    lines += ["", "## 4. Agent 觀察股票"]
    for x in morning.get("recommended_stocks", [])[:12]:
        lines.append(f"- **{x.get('name','')}（{x.get('symbol','')}）**｜{x.get('industry','')}｜{x.get('reason','')}")
    lines += ["", "## CNYES 快取", f"- 日期：{payload.get('cnyes_news', {}).get('cache_dates', [])}", f"- 篇數：{payload.get('cnyes_news', {}).get('article_count', 0)}"]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload
