# -*- coding: utf-8 -*-
"""盤後「今日漲跌原因」研究 Agent。

設計原則：
1. 先用可量化的市場資料描述「今天發生了什麼」：個股、TAIEX、同細分業務群、法人、成交量。
2. 再依公司主要業務與產業鏈，建立針對性新聞/事件查詢。
3. 逐則標示證據層級，不把單一新聞直接視為因果。
4. 對國際事件、主管/董事談話、財報、接單、供應鏈等內容做交叉驗證。
5. Qwen3 只做證據統整，不可以自行捏造「原因」或修改量化模型。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
from datetime import datetime

import numpy as np
import pandas as pd

from stock_api import (
    clean_symbol, FugleClient, fetch_institution_daily, fetch_taiex_intraday,
)
from business_master import profile_for_symbol, build_business_research_context, business_map_for_symbols

TAIPEI_TZ = "Asia/Taipei"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


class PostMarketAnalyzer:
    def __init__(self, base_dir: str | Path = ".", ollama_model: str = "qwen3:8b", ollama_host: str = "http://127.0.0.1:11434"):
        self.base = Path(base_dir).resolve()
        self.data_dir = self.base / "data"
        self.output_dir = self.base / "output"
        self.out_dir = self.output_dir / "post_market"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.data_dir / "post_market_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.client = FugleClient()
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 TaiwanStockPostMarketAgent/1.0",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        })

    @staticmethod
    def _norm_text(v: Any) -> str:
        s = str(v or "").lower()
        return re.sub(r"[^0-9a-zA-Z一-龥]+", "", s)

    @staticmethod
    def _safe_float(v, default=np.nan):
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def _quote(self, symbol: str) -> dict[str, Any]:
        try:
            return self.client.quote(clean_symbol(symbol)) or {}
        except Exception:
            return {}

    def _ticker(self, symbol: str) -> dict[str, Any]:
        try:
            return self.client.ticker(clean_symbol(symbol)) or {}
        except Exception:
            return {}

    def _history(self, symbol: str, days: int = 40) -> pd.DataFrame:
        end = pd.Timestamp.now(tz=TAIPEI_TZ).tz_localize(None).normalize().date()
        start = (pd.Timestamp(end) - pd.Timedelta(days=days)).date()
        try:
            return self.client.historical_candles(clean_symbol(symbol), start, end, "D", adjusted=False)
        except Exception:
            return pd.DataFrame()

    def _news(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        try:
            from urllib.parse import quote_plus
            import xml.etree.ElementTree as ET
            url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            r = self.session.get(url, timeout=12)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            out = []
            for item in root.findall("./channel/item")[:limit]:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                published = item.findtext("pubDate", "")
                summary = re.sub(r"<[^>]+>", " ", item.findtext("description", "") or "").strip()
                source_el = item.find("source")
                source = source_el.text if source_el is not None else "Google News"
                out.append({"title": title, "link": link, "published": published, "summary": summary, "source": source, "query": query})
            return out
        except Exception:
            return []

    def _business_profile(self, symbol: str) -> dict[str, Any]:
        try:
            path = self.data_dir / "business_master.csv"
            p = profile_for_symbol(symbol, path, force=False)
            return build_business_research_context(p)
        except Exception:
            return {"symbol": symbol, "name": symbol, "industry_name": "", "business_group": "", "primary_chain": "", "research_keywords": []}

    def _official_mentions(self, symbol: str, name: str = "") -> list[dict[str, Any]]:
        """從現有官方快取中抓出該公司可能相關的 TWSE/TPEx 重大訊息。"""
        rows: list[dict[str, Any]] = []
        for fn, market in [("twse_announcements.json", "TWSE"), ("tpex_announcements.json", "TPEx")]:
            p = self.data_dir / "research" / fn
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                df = pd.DataFrame(data if isinstance(data, list) else [])
            except Exception:
                continue
            if df.empty:
                continue
            text = df.astype(str).agg(" ".join, axis=1)
            mask = text.str.contains(self._norm_text(symbol), regex=False, na=False)
            if name:
                mask = mask | text.str.contains(self._norm_text(name), regex=False, na=False)
            for r in df.loc[mask].tail(20).to_dict("records"):
                r["market_source"] = market
                rows.append(r)
        return rows[-30:]

    def _price_context(self, symbol: str, industry_symbols: Iterable[str]) -> dict[str, Any]:
        q = self._quote(symbol) or {}
        last_price = self._safe_float(q.get("lastPrice", q.get("closePrice")))
        change = self._safe_float(q.get("change"))
        change_pct = self._safe_float(q.get("changePercent"))
        hist = self._history(symbol, 40)
        hist["date"] = pd.to_datetime(hist.get("date"), errors="coerce") if not hist.empty else pd.Series(dtype="datetime64[ns]")
        hist = hist.dropna(subset=["date"]) if not hist.empty else hist
        today = pd.Timestamp.now(tz=TAIPEI_TZ).date()
        qdate = pd.to_datetime(q.get("date"), errors="coerce")
        # Quote 在盤後/非交易日可能仍是前一交易日；若今天日K已存在，優先用今天最後收盤。
        if (pd.isna(qdate) or qdate.date() != today) and not hist.empty:
            latest_row = hist.iloc[-1]
            if pd.Timestamp(latest_row.get("date")).date() == today:
                last_price = self._safe_float(latest_row.get("close"), last_price)
                if len(hist) >= 2:
                    prev_hist = self._safe_float(hist.iloc[-2].get("close"))
                    if np.isfinite(prev_hist) and prev_hist != 0 and np.isfinite(last_price):
                        change = last_price - prev_hist
                        change_pct = change / prev_hist * 100
        prev_close = self._safe_float(q.get("previousClose"))
        if not np.isfinite(prev_close) and not hist.empty:
            prev = hist.iloc[-2] if len(hist) >= 2 else None
            prev_close = self._safe_float(prev.get("close")) if prev is not None else np.nan
        today_close = last_price
        if not np.isfinite(today_close) and not hist.empty:
            today_close = self._safe_float(hist.iloc[-1].get("close"))
        today_return = (today_close / prev_close - 1) if np.isfinite(today_close) and np.isfinite(prev_close) and prev_close != 0 else np.nan

        peer_rows = []
        for peer in list(dict.fromkeys(clean_symbol(s) for s in industry_symbols if clean_symbol(s)))[:12]:
            try:
                pq = self._quote(peer) or {}
                pcp = self._safe_float(pq.get("changePercent"))
                peer_rows.append({"symbol": peer, "change_percent": pcp})
            except Exception:
                continue
        pcts = [r["change_percent"] for r in peer_rows if np.isfinite(r["change_percent"])]
        peer_median = float(np.median(pcts)) if pcts else np.nan

        taiex = fetch_taiex_intraday()
        taiex_latest = taiex.iloc[-1].to_dict() if isinstance(taiex, pd.DataFrame) and not taiex.empty else {}
        taiex_pct = self._safe_float(taiex_latest.get("change_pct"))

        return {
            "quote": q,
            "current_price": today_close,
            "previous_close": prev_close,
            "today_change": change,
            "today_change_percent": change_pct,
            "today_return": today_return,
            "peer_median_change_percent": peer_median,
            "relative_to_taiex_pct": (change_pct - taiex_pct) if np.isfinite(change_pct) and np.isfinite(taiex_pct) else np.nan,
            "taiex": taiex_latest,
            "peer_returns": peer_rows,
        }

    def _institution_context(self, symbol: str, market: str | None) -> dict[str, Any]:
        try:
            inst = fetch_institution_daily(symbol, market=market or "TSE")
            if inst is None or inst.empty:
                return {"available": False, "rows": []}
            rows = inst.tail(10).copy()
            for c in ["foreign_net", "trust_net", "dealer_net", "total_net"]:
                if c in rows.columns:
                    rows[c] = pd.to_numeric(rows[c], errors="coerce")
            totals = {c: float(rows[c].sum(skipna=True)) for c in ["foreign_net", "trust_net", "dealer_net", "total_net"] if c in rows.columns}
            latest = rows.iloc[-1].to_dict()
            return {"available": True, "latest": latest, "totals_10d": totals, "rows": rows.tail(10).to_dict("records")}
        except Exception as exc:
            return {"available": False, "error": str(exc), "rows": []}

    def _query_plan(self, symbol: str, profile: dict[str, Any]) -> list[str]:
        name = profile.get("name") or symbol
        group = profile.get("business_group") or ""
        chain = profile.get("primary_chain") or ""
        kws = list(profile.get("research_keywords") or [])[:6]
        core = " ".join(kws)
        return [
            f'"{symbol}" "{name}" {group} 財報 營收 毛利率 EPS',
            f'"{symbol}" "{name}" {chain} 董事長 總經理 法說會 展望 訂單 接單',
            f'"{symbol}" {core} 競爭者 客戶 供應鏈 產能 報價',
            f'"{name}" 官方公告 重大訊息 董事長 總經理 法說會',
            f'"{name}" 國際情勢 中美 關稅 伊朗 戰爭 油價 央行 利率 AI',
        ]

    def _collect_targeted_news(self, symbol: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
        all_items = []
        seen = set()
        for query in self._query_plan(symbol, profile):
            for item in self._news(query, 7):
                key = self._norm_text(item.get("title"))
                if not key or key in seen:
                    continue
                seen.add(key)
                all_items.append(item)
        return all_items[:60]

    def _verify_events(self, symbol: str, profile: dict[str, Any], news: list[dict[str, Any]], official: list[dict[str, Any]], fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
        official_text = self._norm_text(" ".join(" ".join(str(v) for v in r.values()) for r in official))
        business_text = self._norm_text(" ".join([profile.get("business_group", ""), profile.get("primary_chain", ""), " ".join(profile.get("research_keywords", []))]))
        title_norms = [self._norm_text(x.get("title")) for x in news]
        results = []
        for item in news:
            title = item.get("title", "")
            text = self._norm_text(f"{title} {item.get('summary','')}")
            business_hits = sum(1 for k in profile.get("research_keywords", []) if self._norm_text(k) and self._norm_text(k) in text)
            entity = self._norm_text(symbol) in text or self._norm_text(profile.get("name", "")) in text
            official_match = bool(text[:24] and text[:24] in official_text)
            corroboration = 0
            base = self._norm_text(title)
            chunks = {base[i:i+6] for i in range(0, max(0, len(base)-5), 3)}
            for other in title_norms:
                if not other or other == base:
                    continue
                och = {other[i:i+6] for i in range(0, max(0, len(other)-5), 3)}
                if chunks and len(chunks & och) / max(1, len(chunks | och)) >= 0.18:
                    corroboration += 1
            positive_words = ["成長","創高","接單","擴產","需求","獲利","上調","量產","導入","降本","AI"]
            negative_words = ["下滑","衰退","砍單","停工","裁員","下修","制裁","關稅","戰爭","成本上升","庫存"]
            pos = sum(w in f"{title} {item.get('summary','')}" for w in positive_words)
            neg = sum(w in f"{title} {item.get('summary','')}" for w in negative_words)
            sentiment = float(np.clip((pos - neg) / max(1, pos + neg), -1, 1))
            relevance = min(1.0, (0.45 if entity else 0.0) + min(0.35, 0.05 * business_hits) + (0.15 if business_text and any(k in text for k in business_text.split()[:8]) else 0))
            if official_match and corroboration >= 1 and relevance >= 0.45:
                verification = "已找到官方/原始資料支持＋多來源佐證"
            elif official_match and relevance >= 0.45:
                verification = "有官方/原始資料匹配，但尚缺第二來源"
            elif corroboration >= 2 and relevance >= 0.45:
                verification = "多來源一致，但尚未完成原始文件核驗"
            elif relevance < 0.20:
                verification = "低業務相關度，不宜直接歸因"
            else:
                verification = "待查證：需核對公司公告/主管機關/原始報告"
            results.append({**item, "sentiment_score": sentiment, "business_relevance": relevance, "official_match": official_match, "corroboration": corroboration, "verification": verification})
        return results

    def _fundamental_snapshot(self, symbol: str) -> dict[str, Any]:
        try:
            # 優先使用研究 Agent 已快取的官方資料，避免重複大量請求。
            from research_agent import ResearchAgent
            r = ResearchAgent(self.base)
            rows = r.company_financial_snapshot([symbol])
            return rows[0] if rows else {}
        except Exception:
            return {}

    def _reason_candidates(self, symbol: str, move: dict[str, Any], inst: dict[str, Any], verified: list[dict[str, Any]], fundamentals: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        # 市場共振：個股是否明顯跑輸/跑贏大盤與同業
        rel = self._safe_float(move.get("relative_to_taiex_pct"))
        peer = self._safe_float(move.get("peer_median_change_percent"))
        stock = self._safe_float(move.get("today_change_percent"))
        if np.isfinite(stock) and np.isfinite(peer):
            spread = stock - peer
            candidates.append({"factor": "同細分業務群相對表現", "score": float(min(1.0, abs(spread)/5.0)), "direction": "利空偏壓" if spread < 0 else "利多偏壓", "evidence": f"個股 {stock:+.2f}%；同業中位數 {peer:+.2f}%"})
        if np.isfinite(rel):
            candidates.append({"factor": "大盤相對強弱", "score": float(min(1.0, abs(rel)/5.0)), "direction": "利空偏壓" if rel < 0 else "利多偏壓", "evidence": f"相對 TAIEX 差異 {rel:+.2f} 個百分點"})
        if inst.get("available"):
            totals = inst.get("totals_10d", {})
            foreign = self._safe_float(totals.get("foreign_net"))
            total = self._safe_float(totals.get("total_net"))
            if np.isfinite(foreign):
                candidates.append({"factor": "外資 10 日累計交易", "score": float(min(1.0, abs(foreign)/5_000_000)), "direction": "利空偏壓" if foreign < 0 else "利多偏壓", "evidence": f"外資 10 日累計 {foreign:,.0f} 股"})
            if np.isfinite(total):
                candidates.append({"factor": "三大法人 10 日累計交易", "score": float(min(1.0, abs(total)/5_000_000)), "direction": "利空偏壓" if total < 0 else "利多偏壓", "evidence": f"三大法人 10 日累計 {total:,.0f} 股"})
        for x in verified:
            if x.get("verification", "").startswith("低業務"):
                continue
            s = self._safe_float(x.get("sentiment_score"), 0.0)
            relv = self._safe_float(x.get("business_relevance"), 0.0)
            corr = self._safe_float(x.get("corroboration"), 0.0)
            score = min(1.0, abs(s) * (0.5 + 0.5 * relv) * (1 + min(2, corr)*0.15))
            if score < 0.08:
                continue
            candidates.append({"factor": x.get("title"), "score": score, "direction": "利空偏壓" if s < 0 else "利多偏壓" if s > 0 else "中性", "evidence": x.get("verification", "待查證"), "source": x.get("source"), "link": x.get("link")})
        rev_yoy = self._safe_float(fundamentals.get("revenue_yoy"))
        eps = self._safe_float(fundamentals.get("eps"))
        if np.isfinite(rev_yoy):
            candidates.append({"factor": "最新營收年增率", "score": float(min(1.0, abs(rev_yoy)/30.0)), "direction": "利空偏壓" if rev_yoy < 0 else "利多偏壓", "evidence": f"最新可得 YoY {rev_yoy:+.2f}%"})
        if np.isfinite(eps):
            candidates.append({"factor": "最新可得 EPS", "score": 0.15, "direction": "中性", "evidence": f"EPS {eps:.2f}"})
        candidates.sort(key=lambda x: x.get("score",0), reverse=True)
        return candidates[:12]

    def ollama_explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import ollama
            client = ollama.Client(host=self.ollama_host)
            prompt = (
                "你是台股盤後事件歸因研究 Agent。以下是程式先完成的可驗證資料。\n"
                "你的任務：用繁體中文解釋『今天這檔股票為何上漲/下跌』，但不得把相關性當因果。\n"
                "請依序回答：1. 已確認的事件/數據；2. 最可能的 3 個驅動因素；3. 哪些消息尚未被原始資料證實；4. 法人交易是否與價格方向一致；5. 財報/營收是否支持消息；6. 國際/總經因素如何可能傳導到這家公司；7. 供應鏈或同業是否共振；8. 哪些內容可能是市場誇大/反應過度；9. 最終用『高/中/低』信心標示每個原因。\n"
                "不可自行新增未提供的事實，不可把新聞標題直接當成真相；若缺乏證據，明確寫『待查證』。\n\n" + json.dumps(payload, ensure_ascii=False, default=str)[:50000]
            )
            resp = client.chat(model=self.ollama_model, messages=[{"role":"system","content":"完全本機 Research Agent；只使用提供證據；不可修改模型。"},{"role":"user","content":prompt}], options={"temperature":0.1})
            content = getattr(resp.message, "content", "") or ""
            return {"available": True, "model": self.ollama_model, "content": content}
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def analyze(self, symbol: str, market: str | None = None, industry_symbols: Iterable[str] = ()) -> dict[str, Any]:
        symbol = clean_symbol(symbol)
        if not symbol:
            raise ValueError("沒有選取股票")
        info = self._ticker(symbol) or {}
        profile = self._business_profile(symbol)
        peer_symbols = [clean_symbol(s) for s in industry_symbols if clean_symbol(s) and clean_symbol(s) != symbol]
        try:
            profiles = business_map_for_symbols(peer_symbols + [symbol], path=self.data_dir / "business_master.csv", force=False)
            selected_p = profiles.get(symbol, {})
            selected_group = str(selected_p.get("business_group") or profile.get("business_group") or "")
            selected_ind = str(selected_p.get("industry") or profile.get("industry") or "00")
            same_group = [s for s in peer_symbols if str(profiles.get(s, {}).get("business_group") or "") == selected_group and selected_group]
            peer_symbols = same_group or [s for s in peer_symbols if str(profiles.get(s, {}).get("industry") or "00") == selected_ind]
        except Exception:
            peer_symbols = peer_symbols[:10]
        move = self._price_context(symbol, peer_symbols)
        inst = self._institution_context(symbol, market or info.get("market"))
        official = self._official_mentions(symbol, profile.get("name", ""))
        fundamentals = self._fundamental_snapshot(symbol)
        news = self._collect_targeted_news(symbol, profile)
        verified = self._verify_events(symbol, profile, news, official, fundamentals)
        reasons = self._reason_candidates(symbol, move, inst, verified, fundamentals, profile)
        payload = {
            "as_of": pd.Timestamp.now(tz=TAIPEI_TZ).isoformat(),
            "symbol": symbol,
            "name": profile.get("name") or info.get("name") or symbol,
            "market": info.get("market") or market,
            "industry": profile.get("industry_name"),
            "business_group": profile.get("business_group"),
            "primary_chain": profile.get("primary_chain"),
            "price_context": move,
            "institution_context": inst,
            "official_mentions": official[-20:],
            "fundamentals": fundamentals,
            "verified_news": verified[:40],
            "reason_candidates": reasons,
            "research_questions": self._query_plan(symbol, profile),
        }
        payload["ollama_analysis"] = self.ollama_explain(payload)
        out_path = self.out_dir / f"{symbol}_latest.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return payload

    def load(self, symbol: str) -> dict[str, Any] | None:
        p = self.out_dir / f"{clean_symbol(symbol)}_latest.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
