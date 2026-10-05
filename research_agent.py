# -*- coding: utf-8 -*-
"""Local Model Research Agent for the Taiwan-stock dashboard.

完全不使用 OpenAI API。

定位：盤後研究與模型治理，不直接修改 production model，不直接下單。
主要工作：
1. 檢查各產業 Random Forest / 44 Features 模型健康度、OOF 指標與資料新鮮度。
2. 檢查近期特徵漂移與模型集中度。
3. 讀取 TWSE / TPEx 官方重大訊息與月營收資料。
4. 讀取 Google News RSS，做新聞搜尋與跨來源比對。
5. 整理總體經濟 / 國際情勢 / 產業新聞證據。
6. 使用本機 Ollama / Qwen3 做「證據統整與待查證判斷」，不是直接相信新聞標題。
7. 抓取 MOPS 法人說明會資料，判斷偏利多／偏利空，並用法說後股價反應交叉驗證。
8. 分析近期市場漲幅／跌幅個股，依產業聚合，並用新聞與公司主要業務產生利多／利空原因候選。
9. 產生「是否建議重新訓練、哪些特徵值得研究」報告，不自動改模型。
"""
from __future__ import annotations

import json
import os
import math
import re
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import joblib
import numpy as np
import pandas as pd
import requests

from ai_engine import FEATURES, _normalise_metrics
from stock_api import INDUSTRIES, INDUSTRY_OVERRIDES, clean_symbol, FugleClient
from business_master import business_map_for_symbols, build_business_research_context
from theme_agent import discover_bullish_themes
from earnings_call_agent import EarningsCallAgent
from agent_research_assistant import ResearchOrchestratorAgent
from market_mover_analysis import MarketMoverAnalyzer
from market_calendar import market_status_text, is_twse_trading_day
from cnyes_news_crawler import CnyesNewsCrawler
# 相容不同版本的 CNYES Agent：
# 舊版類別名稱為 CnyesNewsDigestAgent；目前專案可能使用 CnyesNewsAgent。
try:
    from cnyes_news_agent import CnyesNewsDigestAgent  # type: ignore
except ImportError:
    try:
        from cnyes_news_agent import CnyesNewsAgent as CnyesNewsDigestAgent  # type: ignore
    except ImportError as exc:
        CnyesNewsDigestAgent = None  # type: ignore
        _CNYES_AGENT_IMPORT_ERROR = exc
    else:
        _CNYES_AGENT_IMPORT_ERROR = None
else:
    _CNYES_AGENT_IMPORT_ERROR = None

TAIPEI_TZ = "Asia/Taipei"
TWSE_OPENAPI = "https://openapi.twse.com.tw/v1"
TPEX_OPENAPI = "https://www.tpex.org.tw/openapi/v1"
TWSE_SUPPLY_CHAIN = f"{TWSE_OPENAPI}/opendata/t187ap46_L_13"
TPEX_SUPPLY_CHAIN = f"{TPEX_OPENAPI}/t187ap46_O_13"
TWSE_INCOME = f"{TWSE_OPENAPI}/opendata/t187ap06_L_ci"
TPEX_INCOME = f"{TPEX_OPENAPI}/mopsfin_t187ap06_O_ci"
TWSE_BALANCE = f"{TWSE_OPENAPI}/opendata/t187ap07_L_ci"
TPEX_BALANCE = f"{TPEX_OPENAPI}/mopsfin_t187ap07_O_ci"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, ""))
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


POSITIVE_WORDS = [
    "成長", "增加", "上升", "創高", "接單", "擴產", "需求強", "回溫", "獲利", "毛利率",
    "漲價", "改善", "復甦", "強勁", "創新高", "AI需求", "訂單", "突破", "正向", "上調",
]
NEGATIVE_WORDS = [
    "下滑", "衰退", "減少", "虧損", "停工", "裁員", "下修", "砍單", "需求弱", "跌價", "降價",
    "庫存", "風險", "制裁", "關稅", "戰爭", "原料漲價", "成本上升", "放緩", "取消", "負面",
]
SECTOR_QUERY_HINTS = {
    "01": "水泥 水泥建材",
    "02": "食品 食品工業",
    "03": "塑膠 塑化",
    "04": "紡織 紡織纖維",
    "05": "電機機械 自動化",
    "06": "電器電纜 電線電纜",
    "08": "玻璃 陶瓷",
    "09": "造紙 紙業",
    "10": "鋼鐵 鋼價 鋼材",
    "11": "橡膠 輪胎",
    "12": "汽車 零件",
    "14": "營建 建材 房市",
    "15": "航運 貨櫃 運價",
    "16": "觀光 餐旅 旅遊",
    "17": "金融 銀行 保險",
    "19": "綜合產業",
    "20": "其他產業 台股",
    "21": "化學 化工",
    "22": "生技 醫療 藥品",
    "23": "油電燃氣 天然氣 電價",
    "24": "半導體 晶圓 AI GPU 記憶體",
    "25": "電腦及週邊 伺服器 筆電",
    "26": "光電 面板 顯示器",
    "27": "通信網路 網通 通訊",
    "28": "電子零組件 PCB MLCC 連接器",
    "29": "電子通路 IC通路",
    "30": "資訊服務 軟體 雲端",
    "31": "其他電子 電子",
    "32": "文化創意 遊戲",
    "33": "農業科技 農業科技",
    "35": "綠能環保 綠能 太陽能 儲能",
    "36": "數位雲端 雲端 AI 軟體",
    "37": "運動休閒 運動 體育",
    "38": "居家生活 居家 零售",
}
MACRO_QUERY_HINTS = [
    ("台灣通膨", "台灣 CPI PPI 通膨 DGBAS"),
    ("央行利率", "台灣央行 利率 貨幣政策"),
    ("出口貿易", "台灣 出口 進口 貿易 財政部"),
    ("美元利率", "Fed 利率 美債 美元"),
    ("國際油價", "原油 油價 OPEC"),
    ("地緣政治", "中東 地緣政治 制裁 關稅"),
    ("中國景氣", "中國 經濟 PMI 房地產"),
    ("AI產業", "AI 資料中心 GPU 半導體 server"),
]

# 本專案目前將盤後研究焦點限縮在自選股中的三個核心產業，避免泛化新聞淹沒真正有用的證據。
FOCUS_INDUSTRIES = {"24": "半導體業", "26": "光電業", "28": "電子零組件業"}
FOCUS_WATCHLIST_FALLBACK = {
    "3481": "26", "2327": "28", "2492": "28", "3037": "28", "3044": "28",
    "3533": "28", "2303": "24", "2330": "24", "2344": "24", "2408": "24",
    "6515": "24", "7769": "24", "6488": "24", "3374": "24", "3450": "24",
    "3006": "24", "3661": "24", "8299": "24", "2377": "25", "2308": "28",
    "6274": "28", "0050": "00",
}
FOCUS_ORDER_TERMS = [
    "接單", "訂單", "在手訂單", "backlog", "量產", "放量", "產能", "稼動率", "擴產",
    "資本支出", "CAPEX", "design win", "design-win", "導入", "認證", "供貨", "出貨",
]
FOCUS_THEMES = {
    "CCL/高速材料": ["CCL 銅箔基板 高速材料 PCB AI 伺服器", "玻纖布 銅箔基板 高頻高速 PCB", "聯茂 台光電 台燿 南亞 CCL"],
    "電源供應器": ["電源供應器 AI 伺服器 PSU 800V BBU", "AI 伺服器 電源供應 台達 光寶 康舒 全漢 群電", "OCP rack power PSU server power"],
    "半導體先進製程": ["2奈米 先進製程 AI GPU CoWoS 先進封裝 台積電", "HBM 記憶體 DDR5 NAND AI server", "半導體設備 光罩 化學品 封裝 測試"],
    "光電/顯示": ["面板 光電 Mini LED Micro LED AR VR", "顯示器 面板 驅動IC 光學元件", "光電 車用顯示"],
}
INDUSTRY_CHAIN_SOURCES = {
    "24": "https://ic.tpex.org.tw/introduce.php?ic=D000",
    "28": "https://ic.tpex.org.tw/introduce.php?ic=L000",
    "25": "https://ic.tpex.org.tw/introduce.php?ic=F000",
    "26": "https://ic.tpex.org.tw/introduce.php",
}


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: str
    summary: str
    query: str
    sentiment: str = "中性/待判定"
    sentiment_score: float = 0.0
    corroboration: int = 0
    official_match: bool = False
    verification: str = "待查證"
    business_relevance: float = 0.0
    business_match: str = "一般產業相關"
    entity_match: bool = False


@dataclass
class SectorResearch:
    code: str
    name: str
    news_count: int
    weighted_sentiment: float
    confidence: float
    dominant_risk: str
    dominant_opportunity: str
    evidence: list[dict[str, Any]]


class ResearchAgent:
    def __init__(
        self,
        base_dir: str | Path = ".",
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
    ):
        self.base = Path(base_dir).resolve()
        self.data_dir = self.base / "data"
        self.models_dir = self.base / "models"
        self.output_dir = self.base / "output"
        self.research_dir = self.output_dir / "research_reports"
        self.cache_dir = self.data_dir / "research"
        for p in [self.research_dir, self.cache_dir]:
            p.mkdir(parents=True, exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.fugle_client = FugleClient()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 TaiwanStockResearchAgent/1.0",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        })

    # ---------------------------- utils ----------------------------
    @staticmethod
    def _safe_float(v, default=np.nan):
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    @staticmethod
    def _norm_text(s: Any) -> str:
        text = str(s or "").lower()
        text = re.sub(r"[^0-9a-zA-Z一-龥]+", "", text)
        return text

    @staticmethod
    def _word_score(title: str, summary: str) -> float:
        text = f"{title} {summary}"
        pos = sum(text.count(w) for w in POSITIVE_WORDS)
        neg = sum(text.count(w) for w in NEGATIVE_WORDS)
        if pos == neg == 0:
            return 0.0
        return float(np.clip((pos - neg) / max(1, pos + neg), -1, 1))

    def _get_json(self, url: str, timeout: int = 20) -> Any:
        r = self.session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _cache_json(self, path: Path, payload: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")

    def _load_json(self, path: Path, default=None):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    # ---------------------------- official market / company data ----------------------------
    def fetch_twse_announcements(self) -> pd.DataFrame:
        path = self.cache_dir / "twse_announcements.json"
        try:
            data = self._get_json(f"{TWSE_OPENAPI}/opendata/t187ap04_L")
            self._cache_json(path, data)
        except Exception:
            data = self._load_json(path, [])
        return pd.DataFrame(data if isinstance(data, list) else [])

    def fetch_twse_revenue(self) -> pd.DataFrame:
        path = self.cache_dir / "twse_monthly_revenue.json"
        try:
            data = self._get_json(f"{TWSE_OPENAPI}/opendata/t187ap05_L")
            self._cache_json(path, data)
        except Exception:
            data = self._load_json(path, [])
        return pd.DataFrame(data if isinstance(data, list) else [])

    def fetch_tpex_announcements(self) -> pd.DataFrame:
        path = self.cache_dir / "tpex_announcements.json"
        try:
            data = self._get_json(f"{TPEX_OPENAPI}/mopsfin_t187ap04_O")
            self._cache_json(path, data)
        except Exception:
            data = self._load_json(path, [])
        return pd.DataFrame(data if isinstance(data, list) else [])

    def fetch_tpex_revenue(self) -> pd.DataFrame:
        path = self.cache_dir / "tpex_monthly_revenue.json"
        try:
            data = self._get_json(f"{TPEX_OPENAPI}/mopsfin_t187ap05_O")
            self._cache_json(path, data)
        except Exception:
            data = self._load_json(path, [])
        return pd.DataFrame(data if isinstance(data, list) else [])

    def fetch_financial_statement(self, market: str, statement: str = "income") -> pd.DataFrame:
        key = f"{market.lower()}_{statement}_statement.json"
        path = self.cache_dir / key
        if market == "TWSE":
            url = TWSE_INCOME if statement == "income" else TWSE_BALANCE
        else:
            url = TPEX_INCOME if statement == "income" else TPEX_BALANCE
        try:
            data = self._get_json(url)
            self._cache_json(path, data)
        except Exception:
            data = self._load_json(path, [])
        return pd.DataFrame(data if isinstance(data, list) else [])

    def _load_focus_industry_map(self) -> dict[str, str]:
        mapping = dict(FOCUS_WATCHLIST_FALLBACK)
        path = self.data_dir / "industry_master.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, dtype={"symbol": str, "industry": str})
                for row in df.to_dict("records"):
                    code = clean_symbol(row.get("symbol", ""))
                    ind = str(row.get("industry", "00")).zfill(2)
                    if code:
                        mapping[code] = INDUSTRY_OVERRIDES.get(code, ind)
            except Exception:
                pass
        mapping.update({k: INDUSTRY_OVERRIDES.get(k, v) for k, v in FOCUS_WATCHLIST_FALLBACK.items()})
        return mapping

    def focus_watchlist_symbols(self, symbols: Iterable[str]) -> list[str]:
        mapping = self._load_focus_industry_map()
        out = []
        for s in symbols:
            code = clean_symbol(s)
            if code and mapping.get(code) in FOCUS_INDUSTRIES:
                out.append(code)
        return list(dict.fromkeys(out))

    @staticmethod
    def _pick_symbol_row(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        symbol = clean_symbol(symbol)
        # 優先使用代號欄位，找不到時才掃描字串。
        for col in df.columns:
            c = str(col).strip().lower()
            if c in {"公司代號", "證券代號", "code", "symbol", "公司代碼"}:
                m = df[col].astype(str).map(clean_symbol) == symbol
                if m.any():
                    return df.loc[m].copy()
        text = df.astype(str).agg(" ".join, axis=1)
        return df.loc[text.str.contains(rf"(?<!\d){re.escape(symbol)}(?!\d)", regex=True, na=False)].copy()

    @staticmethod
    def _numeric_field(row: dict[str, Any], keywords: list[str]) -> float:
        for key, value in row.items():
            k = str(key).replace(" ", "")
            if any(word in k for word in keywords):
                val = ResearchAgent._safe_float(str(value).replace(",", ""))
                if np.isfinite(val):
                    return float(val)
        return np.nan

    def company_financial_snapshot(self, symbols: Iterable[str]) -> list[dict[str, Any]]:
        twse_inc = self.fetch_financial_statement("TWSE", "income")
        tpex_inc = self.fetch_financial_statement("TPEx", "income")
        twse_rev = self.fetch_twse_revenue()
        tpex_rev = self.fetch_tpex_revenue()
        rows = []
        for symbol in self.focus_watchlist_symbols(symbols):
            market_parts = []
            for market, df in [("TWSE", twse_inc), ("TPEx", tpex_inc)]:
                hit = self._pick_symbol_row(df, symbol)
                if not hit.empty:
                    market_parts.extend(hit.to_dict("records")[-1:])
                    break
            inc = market_parts[0] if market_parts else {}
            rev_parts = []
            for df in [twse_rev, tpex_rev]:
                hit = self._pick_symbol_row(df, symbol)
                if not hit.empty:
                    rev_parts.extend(hit.to_dict("records")[-3:])
            rev = rev_parts[-1] if rev_parts else {}
            revenue = self._numeric_field(inc, ["營業收入", "營業收入淨額", "OperatingRevenue"])
            gross_profit = self._numeric_field(inc, ["營業毛利", "營業毛利（損）", "GrossProfit"])
            op_profit = self._numeric_field(inc, ["營業利益", "營業利益（損失）", "OperatingIncome"])
            net_income = self._numeric_field(inc, ["本期淨利", "本期淨利（淨損）", "NetIncome"])
            eps = self._numeric_field(inc, ["基本每股盈餘", "基本每股盈餘（元）", "EPS"])
            latest_month_revenue = self._numeric_field(rev, ["當月營收", "營業收入", "Revenue"])
            yoy = self._numeric_field(rev, ["去年同月增減", "年增率", "YoY"])
            rows.append({
                "symbol": symbol,
                "industry_code": self._load_focus_industry_map().get(symbol, "00"),
                "market": "TWSE" if not tpex_inc.empty and self._pick_symbol_row(twse_inc, symbol).shape[0] else "TPEx",
                "statement_revenue": revenue, "gross_profit": gross_profit, "operating_profit": op_profit,
                "net_income": net_income, "eps": eps, "latest_month_revenue": latest_month_revenue,
                "revenue_yoy": yoy,
                "raw_financial": inc, "raw_revenue": rev,
            })
        return rows

    def focus_peer_panel(self, symbols: Iterable[str], revenue_frames: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> list[dict[str, Any]]:
        mapping = self._load_focus_industry_map()
        focus_symbols = self.focus_watchlist_symbols(symbols)
        twse_rev, tpex_rev = revenue_frames if revenue_frames is not None else (self.fetch_twse_revenue(), self.fetch_tpex_revenue())
        # 先以自選股作為核心競爭者，再補同產業最新營收較高的外部同業。
        result = []
        for code, industry in FOCUS_INDUSTRIES.items():
            watch = [s for s in focus_symbols if mapping.get(s) == code]
            rows = []
            for s in watch:
                rows.append({"symbol": s, "source": "watchlist", "industry_code": code})
            candidates = []
            for df in (twse_rev, tpex_rev):
                if df is None or df.empty:
                    continue
                for row in df.to_dict("records"):
                    sym = ""
                    for k in row:
                        if str(k).strip() in {"公司代號", "證券代號", "symbol", "code"}:
                            sym = clean_symbol(row[k]); break
                    if not sym or mapping.get(sym) != code:
                        continue
                    rev = self._numeric_field(row, ["當月營收", "營業收入", "Revenue"])
                    candidates.append((sym, rev))
            seen = {r["symbol"] for r in rows}
            for sym, rev in sorted(candidates, key=lambda z: (np.isfinite(z[1]), z[1]), reverse=True)[:8]:
                if sym not in seen:
                    rows.append({"symbol": sym, "source": "same_industry_peer", "industry_code": code, "latest_month_revenue": rev})
                    seen.add(sym)
            result.append({"industry_code": code, "industry_name": industry, "watchlist_symbols": watch, "peer_rows": rows})
        return result

    def focus_theme_research(self, symbols: Iterable[str]) -> list[dict[str, Any]]:
        focus_symbols = self.focus_watchlist_symbols(symbols)
        if not focus_symbols:
            return []
        mapping = self._load_focus_industry_map()
        theme_rows = []
        for theme, queries in FOCUS_THEMES.items():
            hits = []
            for q in queries:
                hits.extend(self.google_news(q, max_items=5))
            # 去重 + 情緒 + 訂單關鍵詞。
            seen = set(); clean = []
            for item in hits:
                key = self._norm_text(item.title)
                if not key or key in seen:
                    continue
                seen.add(key)
                item.sentiment_score = self._word_score(item.title, item.summary)
                text = (item.title + " " + item.summary).lower()
                item.verification = "待查證"
                clean.append(item)
            self._verify_news(clean)
            order_hits = [x for x in clean if any(term.lower() in (x.title + " " + x.summary).lower() for term in FOCUS_ORDER_TERMS)]
            theme_rows.append({
                "theme": theme,
                "related_watchlist": [s for s in focus_symbols if mapping.get(s) in FOCUS_INDUSTRIES],
                "news_count": len(clean),
                "order_evidence_count": len(order_hits),
                "evidence": [asdict(x) for x in clean[:12]],
                "official_chain_source": (
                    "https://ic.tpex.org.tw/introduce.php?ic=D000" if theme == "半導體先進製程" else
                    "https://ic.tpex.org.tw/introduce.php?ic=L000" if theme == "CCL/高速材料" else
                    "https://ic.tpex.org.tw/introduce.php?ic=F000" if theme == "電源供應器" else
                    "https://ic.tpex.org.tw/introduce.php"
                ),
            })
        return theme_rows

    def fetch_supply_chain(self) -> dict[str, pd.DataFrame]:
        """取得公開揭露的供應鏈管理資料；它不是完整供應商/客戶網路，而是公司自揭露的供應鏈管理資訊。"""
        out = {}
        specs = [("TWSE", TWSE_SUPPLY_CHAIN), ("TPEx", TPEX_SUPPLY_CHAIN)]
        for name, url in specs:
            path = self.cache_dir / f"{name.lower()}_supply_chain.json"
            try:
                data = self._get_json(url)
                self._cache_json(path, data)
            except Exception:
                data = self._load_json(path, [])
            out[name] = pd.DataFrame(data if isinstance(data, list) else [])
        return out

    # ---------------------------- news ----------------------------
    def google_news(self, query: str, max_items: int = 8) -> list[NewsItem]:
        url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception:
            return []
        out: list[NewsItem] = []
        for item in root.findall("./channel/item")[:max_items]:
            title = item.findtext("title", default="")
            link = item.findtext("link", default="")
            pub = item.findtext("pubDate", default="")
            desc = item.findtext("description", default="")
            source_el = item.find("source")
            source = source_el.text if source_el is not None else "Google News"
            # description 常帶 HTML，這裡只做輕度清理
            desc = re.sub(r"<[^>]+>", " ", desc)
            out.append(NewsItem(title=title, link=link, source=source or "Google News", published=pub, summary=desc.strip(), query=query))
        return out

    def _business_profiles_for_symbols(self, symbols: Iterable[str]) -> dict[str, dict[str, Any]]:
        codes = [clean_symbol(s) for s in symbols if clean_symbol(s)]
        try:
            profiles = business_map_for_symbols(codes, path=self.data_dir / "business_master.csv", force=False)
            return {s: build_business_research_context(p) for s, p in profiles.items()}
        except Exception:
            return {}

    def _company_research_queries(self, symbol: str, profile: dict[str, Any]) -> list[str]:
        name = str(profile.get("name") or symbol).strip()
        group = str(profile.get("business_group") or "").strip()
        chain = str(profile.get("primary_chain") or "").strip()
        keys = list(profile.get("research_keywords") or [])
        queries = []
        core = " ".join(keys[:5])
        queries.append(f'"{symbol}" {name} {group} 財報 營收 毛利率 EPS')
        queries.append(f'"{symbol}" {name} {chain} 訂單 接單 出貨 產能 客戶')
        if core:
            queries.append(f'"{symbol}" {name} {core} 需求 價格 稼動率')
        return queries[:3]

    def _business_relevance(self, item: NewsItem, profiles: dict[str, dict[str, Any]]) -> tuple[float, str, bool]:
        text = self._norm_text(f"{item.title} {item.summary}")
        best = 0.0
        best_label = "一般產業相關"
        entity_match = False
        for symbol, p in profiles.items():
            name = self._norm_text(p.get("name", ""))
            group = self._norm_text(p.get("business_group", ""))
            chain = self._norm_text(p.get("primary_chain", ""))
            keys = [self._norm_text(k) for k in p.get("research_keywords", []) if len(self._norm_text(k)) >= 2]
            score = 0.0
            if symbol and symbol in text:
                score += 0.60; entity_match = True
            if name and name in text:
                score += 0.30; entity_match = True
            group_tokens = [t for t in re.split(r"[／/、,，\s]+", group) if len(t) >= 2]
            chain_tokens = [t for t in re.split(r"[ >｜/]+", chain) if len(t) >= 2]
            if group_tokens and any(tok in text for tok in group_tokens):
                score += 0.18
            if chain_tokens and any(tok in text for tok in chain_tokens):
                score += 0.16
            key_hits = sum(1 for k in keys if k and k in text)
            score += min(0.30, 0.06 * key_hits)
            if score > best:
                best = min(1.0, score)
                if score >= 0.75:
                    best_label = f"高度相關：{symbol}／{p.get('business_group','')}"
                elif score >= 0.45:
                    best_label = f"業務鏈相關：{symbol}／{p.get('primary_chain','')}"
                else:
                    best_label = f"產業背景相關：{symbol}"
        return best, best_label, entity_match

    def collect_news(self, symbols: Iterable[str] = (), sectors: Iterable[str] = (), max_per_query: int = 6) -> list[NewsItem]:
        profiles = self._business_profiles_for_symbols(symbols)
        queries: list[str] = []
        focused = list(profiles.keys())
        # 1) 每家公司：用「公司名稱 + 主要業務 + 產業鏈 + 驅動因子」建立精準查詢。
        for s in focused:
            queries.extend(self._company_research_queries(s, profiles[s]))
        # 2) 同一細分業務群組：補抓產業需求、價格、接單、競爭者資訊。
        groups = {}
        for s, p in profiles.items():
            key = (p.get("industry"), p.get("business_group"))
            groups.setdefault(key, p)
        for (_, group), p in list(groups.items())[:20]:
            chain = str(p.get("primary_chain") or "").strip()
            keys = " ".join((p.get("research_keywords") or [])[:5])
            if group:
                queries.append(f'"{group}" {chain} 台灣 競爭者 接單 報價 需求 {keys}')
        # 3) 只補三大核心產業與明確主題。
        for code in sectors:
            code = str(code).zfill(2)
            if code not in FOCUS_INDUSTRIES:
                continue
            hint = SECTOR_QUERY_HINTS.get(code, INDUSTRIES.get(code, "台股"))
            queries.append(f"{hint} 台灣 AI 供應鏈 訂單 產能")
        # 4) 有針對性的總體因素，不做泛台股新聞灌水。
        queries.extend([q for name, q in MACRO_QUERY_HINTS if name in {"美元利率", "出口貿易", "國際油價", "地緣政治", "AI產業"}])
        seen = set(); items: list[NewsItem] = []
        for q in list(dict.fromkeys(queries)):
            for item in self.google_news(q, max_items=max_per_query):
                key = self._norm_text(item.title)
                if not key or key in seen:
                    continue
                seen.add(key)
                item.sentiment_score = self._word_score(item.title, item.summary)
                rel, label, entity = self._business_relevance(item, profiles)
                item.business_relevance = rel
                item.business_match = label
                item.entity_match = entity
                # 高相關資料保留；完全無法連到任何研究標的的泛新聞只作宏觀補充。
                if focused and rel < 0.10 and any(s in q for s in focused):
                    continue
                items.append(item)
        self._verify_news(items)
        return items

    def _verify_news(self, items: list[NewsItem]):
        if not items:
            return
        norm_titles = [self._norm_text(x.title) for x in items]
        twse = self.fetch_twse_announcements()
        tpex = self.fetch_tpex_announcements()
        official_texts = []
        for df in [twse, tpex]:
            if df is not None and not df.empty:
                for row in df.to_dict("records"):
                    official_texts.append(" ".join(str(v) for v in row.values()))
        official_norm = [self._norm_text(t) for t in official_texts]
        strong_sources = {"Taiwan Stock Exchange", "Taipei Exchange", "TWSE", "TPEx"}
        for idx, item in enumerate(items):
            base = norm_titles[idx]
            # 透過 token-like chunk overlap 粗估跨來源重複；避免完全依賴同字串。
            chunks = {base[i:i+6] for i in range(0, max(0, len(base)-5), 3)}
            corroboration = 0
            for j, other in enumerate(norm_titles):
                if j == idx or not other:
                    continue
                other_chunks = {other[k:k+6] for k in range(0, max(0, len(other)-5), 3)}
                overlap = len(chunks & other_chunks) / max(1, len(chunks | other_chunks))
                if overlap >= 0.18:
                    corroboration += 1
            official_match = any(base and base[:18] in o for o in official_norm if o)
            item.corroboration = corroboration
            item.official_match = official_match
            if item.sentiment_score >= 0.20:
                item.sentiment = "利多"
            elif item.sentiment_score <= -0.20:
                item.sentiment = "利空"
            else:
                item.sentiment = "中性"
            source_score = 0.75 if item.source in strong_sources or "證交所" in item.source or "櫃買" in item.source else 0.5
            evidence_score = source_score + min(0.2, 0.07 * corroboration) + (0.25 if official_match else 0.0) + min(0.15, 0.15 * item.business_relevance)
            if official_match and corroboration >= 1 and item.business_relevance >= 0.35:
                item.verification = "高可信：官方/原始資料＋跨來源＋與公司主要業務高度相關"
            elif official_match and item.business_relevance >= 0.35:
                item.verification = "較可信：有官方資料且與公司主要業務直接相關"
            elif corroboration >= 2 and item.business_relevance >= 0.35:
                item.verification = "中度可信：多來源且業務高度相關，仍需核對原始文件"
            elif item.business_relevance < 0.15:
                item.verification = "低相關：僅屬泛產業消息，不宜直接推論該公司"
            elif evidence_score < 0.55:
                item.verification = "待查證：來源單一或缺乏原始資料"
            else:
                item.verification = "待查證：建議查看公司/主管機關原始公告"

    # ---------------------------- model health ----------------------------
    def list_model_bundles(self) -> list[Path]:
        return sorted(self.models_dir.glob("ai_sector_*.joblib"))

    def _load_bundle(self, path: Path):
        try:
            return joblib.load(path)
        except Exception as exc:
            return {"_load_error": str(exc)}

    def feature_drift(self, bundle: dict[str, Any], current_df: pd.DataFrame) -> dict[str, Any]:
        profile = bundle.get("feature_profile") or {}
        if not profile or current_df is None or current_df.empty:
            return {"available": False, "score": np.nan, "top_features": []}
        scores = []
        for feat in FEATURES:
            if feat not in current_df.columns:
                continue
            cur = pd.to_numeric(current_df[feat], errors="coerce").dropna()
            if cur.empty:
                continue
            p = profile.get(feat) or {}
            mu = self._safe_float(p.get("mean"))
            sd = self._safe_float(p.get("std"), 1.0)
            if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 1e-9:
                continue
            cm = float(cur.mean())
            z = abs(cm - mu) / sd
            scores.append((feat, float(min(1.0, z / 3.0)), z))
        scores.sort(key=lambda x: x[1], reverse=True)
        return {
            "available": bool(scores),
            "score": float(np.mean([x[1] for x in scores])) if scores else np.nan,
            "top_features": [{"feature": f, "score": s, "z": z} for f, s, z in scores[:8]],
        }

    def model_health(self) -> list[dict[str, Any]]:
        records = []
        now = pd.Timestamp.now(tz=TAIPEI_TZ)
        for path in self.list_model_bundles():
            bundle = self._load_bundle(path)
            sector_code = path.stem.replace("ai_sector_", "").zfill(2)
            metrics = _normalise_metrics(bundle.get("result", bundle.get("result_meta", {}))) if isinstance(bundle, dict) else {}
            trained_at = pd.to_datetime(bundle.get("trained_at"), errors="coerce") if isinstance(bundle, dict) else pd.NaT
            age_days = (now.tz_localize(None) - trained_at.tz_localize(None)).days if pd.notna(trained_at) else np.nan
            imp = []
            for key in ["reg", "clf"]:
                model = bundle.get(key) if isinstance(bundle, dict) else None
                if model is not None and hasattr(model, "feature_importances_"):
                    arr = np.asarray(model.feature_importances_, dtype=float)
                    if len(arr) == len(FEATURES):
                        order = np.argsort(arr)[::-1][:5]
                        imp.extend([{"model": key, "feature": FEATURES[i], "importance": float(arr[i])} for i in order])
            records.append({
                "sector_code": sector_code,
                "sector_name": INDUSTRIES.get(sector_code, "未知產業"),
                "model_file": str(path),
                "training_samples": int(bundle.get("training_samples", 0)) if isinstance(bundle, dict) else 0,
                "training_stocks": int(bundle.get("training_stock_count", 0)) if isinstance(bundle, dict) else 0,
                "training_date": str(bundle.get("training_data_date", "")) if isinstance(bundle, dict) else "",
                "trained_at": str(bundle.get("trained_at", "")) if isinstance(bundle, dict) else "",
                "age_days": age_days,
                "metrics": metrics,
                "top_features": imp,
                "load_error": bundle.get("_load_error") if isinstance(bundle, dict) else None,
            })
        return records

    def _retrains(self, records: list[dict[str, Any]], previous: dict[str, Any]) -> list[dict[str, Any]]:
        prev_models = previous.get("models", {}) if isinstance(previous, dict) else {}
        flags = []
        for rec in records:
            code = rec["sector_code"]
            m = rec["metrics"]
            reasons = []
            dir_acc = self._safe_float(m.get("ClassificationDirectionAccuracy"))
            if np.isfinite(dir_acc) and dir_acc < 0.55:
                reasons.append("分類方向準確率低於55%")
            age = rec.get("age_days")
            if np.isfinite(age) and age > 7:
                reasons.append("模型超過7天未重新訓練")
            prev = prev_models.get(code, {}) if isinstance(prev_models, dict) else {}
            prev_mae = self._safe_float(prev.get("MAE"))
            cur_mae = self._safe_float(m.get("MAE"))
            if np.isfinite(prev_mae) and np.isfinite(cur_mae) and cur_mae > prev_mae * 1.15:
                reasons.append("OOF MAE 較前次研究基準惡化超過15%")
            top = rec.get("top_features", [])
            if top:
                max_imp = max(float(x.get("importance", 0)) for x in top)
                if max_imp > 0.25:
                    reasons.append("特徵重要性集中於少數特徵，建議檢查過度依賴")
            flags.append({"sector_code": code, "sector_name": rec["sector_name"], "retrain": bool(reasons), "reasons": reasons})
        return flags

    # ---------------------------- fundamentals / market evidence ----------------------------
    def _select_symbol_revenue(self, symbol: str, twse_df: pd.DataFrame, tpex_df: pd.DataFrame) -> dict[str, Any]:
        symbol = clean_symbol(symbol)
        frames = []
        for df, market in [(twse_df, "TWSE"), (tpex_df, "TPEx")]:
            if df is None or df.empty:
                continue
            tmp = df.copy()
            text = tmp.astype(str).agg(" ".join, axis=1)
            mask = text.str.contains(re.escape(symbol), na=False)
            hit = tmp.loc[mask].tail(3)
            if not hit.empty:
                frames.append((market, hit))
        out = {"symbol": symbol, "found": False, "rows": []}
        for market, df in frames:
            out["found"] = True
            for row in df.to_dict("records"):
                out["rows"].append({"market": market, **row})
        return out

    def company_research_summary(self, symbols: Iterable[str], items: list[NewsItem]) -> list[dict[str, Any]]:
        profiles = self._business_profiles_for_symbols(symbols)
        rows = []
        for symbol, p in profiles.items():
            related = [x for x in items if x.entity_match or p.get("business_group", "") in (x.business_match or "")]
            # 再用代號/名稱/業務關鍵字做第二層嚴格篩選。
            strict = []
            text_terms = [str(symbol), self._norm_text(p.get("name", ""))] + [self._norm_text(k) for k in (p.get("research_keywords") or [])]
            for x in items:
                text = self._norm_text(f"{x.title} {x.summary}")
                hits = sum(1 for t in text_terms if t and t in text)
                if hits >= 2 or x.entity_match:
                    strict.append(x)
            related = strict[:12] if strict else related[:8]
            positive = sorted([asdict(x) for x in related if x.sentiment_score >= 0.20], key=lambda r: (r.get("business_relevance",0), r.get("sentiment_score",0)), reverse=True)[:5]
            negative = sorted([asdict(x) for x in related if x.sentiment_score <= -0.20], key=lambda r: (r.get("business_relevance",0), r.get("sentiment_score",0)))[:5]
            watch_terms = list(p.get("research_keywords") or [])[:12]
            rows.append({
                "symbol": symbol,
                "name": p.get("name"),
                "industry_name": p.get("industry_name"),
                "business_group": p.get("business_group"),
                "primary_chain": p.get("primary_chain"),
                "research_keywords": watch_terms,
                "query_plan": self._company_research_queries(symbol, p),
                "related_news_count": len(related),
                "positive_evidence": positive,
                "negative_evidence": negative,
                "source_url": p.get("source_url"),
            })
        return rows

    def sector_news_summary(self, items: list[NewsItem]) -> list[SectorResearch]:
        results = []
        for code, name in INDUSTRIES.items():
            if code == "00":
                continue
            keywords = [k for k in SECTOR_QUERY_HINTS.get(code, name).replace("/", " ").split() if len(k) >= 2]
            hits = [x for x in items if any(k.lower() in (x.title + x.summary).lower() for k in keywords)]
            if not hits:
                continue
            scores = np.array([h.sentiment_score for h in hits], dtype=float)
            weights = np.array([1.0 + 0.25 * h.corroboration + (0.5 if h.official_match else 0) for h in hits])
            weighted = float(np.average(scores, weights=weights)) if weights.sum() else 0.0
            confidence = float(np.clip(0.35 + 0.1 * np.mean([min(3, h.corroboration) for h in hits]) + 0.2 * np.mean([h.official_match for h in hits]), 0, 1))
            risks = sorted((h for h in hits if h.sentiment_score < 0), key=lambda h: h.sentiment_score)
            ops = sorted((h for h in hits if h.sentiment_score > 0), key=lambda h: h.sentiment_score, reverse=True)
            results.append(SectorResearch(
                code=code, name=name, news_count=len(hits), weighted_sentiment=weighted,
                confidence=confidence,
                dominant_risk=risks[0].title if risks else "目前未發現明顯利空主題",
                dominant_opportunity=ops[0].title if ops else "目前未發現明顯利多主題",
                evidence=[asdict(x) for x in hits[:8]],
            ))
        return results

    # ---------------------------- earnings call fallback ----------------------------
    def _announcement_earnings_fallback(self, df: pd.DataFrame, requested_symbols: Iterable[str], market: str, existing_keys: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
        """從 TWSE/TPEx 官方重大訊息補抓法說會公告。

        MOPS 法說查詢偶爾會因入口/防爬機制回傳空頁；官方重大訊息仍會留下「法人說明會/業績說明會」公告，
        因此用精確的 (代號, 法說日期) 去重，而不是用「同公司就跳過」，避免漏掉同一公司新的法說。
        """
        if df is None or df.empty:
            return []
        wanted = {clean_symbol(x) for x in requested_symbols if clean_symbol(x)}
        mapping = self._load_focus_industry_map()
        existing_keys = existing_keys or set()
        rows = []
        code_cols = [c for c in df.columns if str(c).strip() in {"公司代號", "證券代號", "SecuritiesCompanyCode", "代號", "公司代碼"}]
        name_cols = [c for c in df.columns if str(c).strip() in {"公司名稱", "CompanyName", "名稱"}]

        def parse_any_date(v: Any) -> str | None:
            text = str(v or "").strip()
            patterns = [
                r"(?<!\d)(\d{3})[./-](\d{1,2})[./-](\d{1,2})(?!\d)",
                r"(?<!\d)(\d{2,3})/(\d{1,2})/(\d{1,2})(?!\d)",
                r"(?<!\d)(\d{7,8})(?!\d)",
                r"(?<!\d)(20\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)",
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if not m:
                    continue
                vals = list(map(int, m.groups()))
                try:
                    y, mo, d = vals
                    if y < 1911:
                        y += 1911
                    return pd.Timestamp(year=y, month=mo, day=d).strftime("%Y-%m-%d")
                except Exception:
                    continue
            return None

        report_today = pd.Timestamp.now(tz=TAIPEI_TZ).date()
        for _, row in df.iterrows():
            raw = " ".join(str(v) for v in row.to_dict().values() if pd.notna(v))
            if not re.search(r"法人說明會|法說會|業績說明會|業績發表會", raw):
                continue
            sym = ""
            if code_cols:
                for c in code_cols:
                    v = clean_symbol(row.get(c, ""))
                    if v:
                        sym = v; break
            if not sym:
                for w in wanted:
                    if re.search(rf"(?<!\d){re.escape(w)}(?!\d)", raw):
                        sym = w; break
            if not sym or (wanted and sym not in wanted):
                continue
            name = ""
            for c in name_cols:
                if str(row.get(c, "")).strip():
                    name = str(row.get(c)).strip(); break
            if not name:
                name = sym
            text_norm = re.sub(r"\s+", " ", raw).strip()
            event_date = None
            for key in ["召開日期", "法說會日期", "說明會日期", "會議日期", "發表日期"]:
                hit_cols = [c for c in row.index if key in str(c)]
                if hit_cols:
                    event_date = parse_any_date(row.get(hit_cols[0]))
                    if event_date: break
            dates = []
            for m in re.finditer(r"(20\d{2}|\d{3})[./-]\d{1,2}[./-]\d{1,2}|\d{7,8}", text_norm):
                d = parse_any_date(m.group(0))
                if d: dates.append(d)
            # 優先找未來日期；其次找最近的歷史日期。不要只把「公告日期」當法說日期，除非找不到其他日期。
            if not event_date:
                future = sorted(set(d for d in dates if d >= report_today))
                event_date = future[0] if future else (sorted(set(dates))[-1] if dates else None)
            if not event_date:
                continue
            key = (sym, event_date)
            if key in existing_keys:
                continue
            # 使用與研究新聞一致的文字規則，明確標示官方公告證據。
            pos_hits = [w for w in POSITIVE_WORDS if w in text_norm]
            neg_hits = [w for w in NEGATIVE_WORDS if w in text_norm]
            ps, ns = len(pos_hits), len(neg_hits)
            if ps > ns and ps:
                label = "偏利多"
            elif ns > ps and ns:
                label = "偏利空"
            else:
                label = "中性／待確認"
            rows.append({
                "symbol": sym,
                "name": name,
                "industry_code": mapping.get(sym, "00"),
                "industry_name": INDUSTRIES.get(mapping.get(sym, "00"), "未知產業"),
                "event_date": event_date,
                "event_time": "",
                "venue": "",
                "summary": text_norm[:1600],
                "impact": label,
                "impact_score": round((ps - ns) / max(1, ps + ns), 4),
                "positive_terms": pos_hits[:6],
                "negative_terms": neg_hits[:6],
                "evidence_level": "官方重大訊息補漏",
                "source_url": "https://mops.twse.com.tw/mops/web/t100sb07_1",
                "source_type": "MOPS t100sb07_1 / 重大訊息補漏",
                "related_links": [],
                "post_1d_return": None,
                "post_5d_return": None,
                "price_reaction_label": "尚無法計算",
                "market": market,
            })
            existing_keys.add(key)
        return rows

    # ---------------------------- Ollama synthesis ----------------------------
    def _deterministic_llm_fallback(self, payload: dict[str, Any]) -> str:
        """當 Qwen 輸出混入研究包外年份/資料時，產生只使用本次 payload 的摘要。"""
        lines = ["### 本次盤後研究摘要", ""]
        movers = (payload.get("market_movers", {}) or {}).get("movers", [])
        if movers:
            rises = [m for m in movers if m.get("direction") == "上漲"][:3]
            falls = [m for m in movers if m.get("direction") == "下跌"][:3]
            if rises:
                lines.append("- 上漲樣本：" + "、".join(f"{m.get('name', m.get('symbol'))}({m.get('symbol')}) {float(m.get('today_change_percent',0)):+.2f}%" for m in rises))
            if falls:
                lines.append("- 下跌樣本：" + "、".join(f"{m.get('name', m.get('symbol'))}({m.get('symbol')}) {float(m.get('today_change_percent',0)):+.2f}%" for m in falls))
        ind = (payload.get("market_movers", {}) or {}).get("industry_summary", [])
        if ind:
            strongest = sorted(ind, key=lambda x: float(x.get("today_avg_change", 0) or 0), reverse=True)[:2]
            weakest = sorted(ind, key=lambda x: float(x.get("today_avg_change", 0) or 0))[:2]
            lines.append("- 產業強弱：強勢 " + "、".join(f"{x.get('industry_code')} {x.get('industry_name')} {float(x.get('today_avg_change',0)):+.2f}%" for x in strongest))
            lines.append("- 產業弱勢：" + "、".join(f"{x.get('industry_code')} {x.get('industry_name')} {float(x.get('today_avg_change',0)):+.2f}%" for x in weakest))
        calls = [x for x in payload.get("earnings_calls", []) if isinstance(x, dict) and x.get("event_date")][:5]
        if calls:
            lines.append("- 最近法說會：" + "；".join(f"{x.get('event_date')} {x.get('name',x.get('symbol'))}（{x.get('impact','中性／待確認')}）" for x in calls))
            for x in calls[:5]:
                if x.get("memo_opened"):
                    lines.append(f"  - {x.get('symbol')} 備忘錄摘要：{str(x.get('summary',''))[:700]}")
                    lines.append(f"  - Fugle來源：{x.get('source_url')}")
        else:
            lines.append("- 最近法說會：本次研究包沒有取得可驗證的法說事件。")
        themes = payload.get("bullish_themes", [])[:3]
        if themes:
            lines.append("- 利多題材：" + "；".join(f"{x.get('theme')}（{x.get('status')}）" for x in themes))
        fin = payload.get("financial_snapshots", [])
        positive_fin = []
        for x in fin:
            try:
                yoy = float(x.get("revenue_yoy"))
                if np.isfinite(yoy) and yoy > 0:
                    positive_fin.append((yoy, x))
            except Exception:
                pass
        if positive_fin:
            positive_fin.sort(reverse=True, key=lambda z: z[0])
            lines.append("- 營收成長樣本：" + "、".join(f"{x.get('symbol')} YoY {y:+.1f}%" for y,x in positive_fin[:5]))
        lines.append("- 判讀原則：法說文字中的利多／利空只表示營運訊號；新聞若缺官方或多來源交叉驗證，維持待查證。")
        return "\n".join(lines)

    def ollama_synthesis(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import ollama
        except Exception as exc:
            return {"available": False, "error": f"未安裝 ollama：{exc}"}
        client = ollama.Client(host=self.ollama_host)
        report_date = str(payload.get("report_date", ""))
        packet = {
            "研究日期": report_date,
            "市場狀態": payload.get("market_status", ""),
            "近期漲跌產業與個股": (payload.get("market_movers", {}) or {}).get("movers", [])[:20],
            "產業彙整": (payload.get("market_movers", {}) or {}).get("industry_summary", [])[:15],
            "法說會": [
                {
                    **{k: x.get(k) for k in ("symbol","name","event_date","published_date","title","summary","impact","impact_score","positive_terms","negative_terms","evidence_level","source_url","topic_url","memo_opened","memo_sections","post_1d_return","post_5d_return","price_reaction_label")},
                    "memo_text": x.get("memo_text", "")[:12000],
                }
                for x in payload.get("earnings_calls", [])
                if isinstance(x, dict) and x.get("event_date")
            ][:30],
            "利多題材": payload.get("bullish_themes", [])[:10],
            "財報營收": payload.get("financial_snapshots", [])[:25],
            "模型健康": payload.get("model_health", [])[:10],
            "公司研究": payload.get("company_research", [])[:20],
        }
        prompt = (
            "你是台灣股市盤後研究 Agent。只能使用下方『本次研究包』，禁止引用你記憶中的舊報告。"
            "如果資料不足，寫『資料不足／待查證』，不要自行補數字。"
            f"本次研究日期為 {report_date}。輸出不得出現研究包內沒有的年份、公司、數字或事件。"
            "請用繁體中文，以手機容易閱讀的格式輸出。第一段固定是『今日五個重點』，後面依序：產業強弱、法說會、利多題材、財報營收、待查證風險。"
            "不要提供買賣建議或預測式結論。法說會利多／利空只代表公開營運文字的訊號分類。法說會摘要必須直接引用/改寫本次研究包中的 Fugle memo_text；memo_opened=false 的資料只能寫成『官方日程，沒有備忘錄摘要』，不得把網址或網站導覽當成摘要。"
            "全文不超過 1600 字。\n\n"
            + json.dumps(packet, ensure_ascii=False, default=str)[:30000]
        )
        try:
            resp = client.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "完全本機研究 Agent；只能根據本次資料，不得引用記憶中的舊內容。"},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0},
            )
            content = getattr(resp.message, "content", "") or ""
            # 只允許「本次研究包本身出現過」的年份。
            # 法說備忘錄可能合法提到 2027/2028 展望，因此不能再只允許 report_date 的年份；
            # 只有 Qwen 自行冒出研究包外年份（例如舊報告 2023）才觸發 fallback。
            packet_years = set(re.findall(r"20\d{2}", json.dumps(packet, ensure_ascii=False, default=str)))
            packet_years.add(report_date[:4])
            bad_years = sorted(set(y for y in re.findall(r"20\d{2}", content) if y not in packet_years))
            if bad_years:
                fallback = self._deterministic_llm_fallback(payload)
                return {
                    "available": True, "model": self.ollama_model, "content": fallback,
                    "validation": "fallback",
                    "error": f"Qwen 原始輸出含研究包外年份 {bad_years}，已改用本次資料的安全摘要。",
                }
            return {"available": True, "model": self.ollama_model, "content": content.strip(), "validation": "passed"}
        except Exception as exc:
            fallback = self._deterministic_llm_fallback(payload)
            return {"available": True, "model": self.ollama_model, "content": fallback, "validation": "fallback", "error": str(exc)}

    # ---------------------------- final report ----------------------------
    def run_daily_research(self, symbols: Iterable[str] = (), sector_codes: Iterable[str] | None = None) -> dict[str, Any]:
        # 研究範圍：自選股優先，聚焦半導體/光電/電子零組件；只在需要時補同產業少量競品。
        requested_symbols = [clean_symbol(s) for s in symbols if clean_symbol(s)]
        symbols = self.focus_watchlist_symbols(requested_symbols)
        if sector_codes is None:
            sector_codes = list(FOCUS_INDUSTRIES.keys())
        sector_codes = [str(c).zfill(2) for c in sector_codes if str(c).zfill(2) in FOCUS_INDUSTRIES]
        now_taipei = pd.Timestamp.now(tz=TAIPEI_TZ)
        report_date = now_taipei.strftime("%Y-%m-%d")
        fast_mode = _env_bool("AFTER_CLOSE_FAST_MODE", False)

        twse_rev = self.fetch_twse_revenue()
        tpex_rev = self.fetch_tpex_revenue()
        twse_ann = self.fetch_twse_announcements()
        tpex_ann = self.fetch_tpex_announcements()
        supply_chain = self.fetch_supply_chain()
        twse_income = self.fetch_financial_statement("TWSE", "income")
        tpex_income = self.fetch_financial_statement("TPEx", "income")
        # 15:00 盤後模式不再等待 Google News 多輪搜尋；新聞研究改用前一晚 CNYES 日檔。
        news = [] if fast_mode else self.collect_news(symbols=symbols, sectors=sector_codes, max_per_query=6)

        # ============================================================
        # 鉅亨網新聞：盤後研究「只讀夜間快取」，不在 15:00 重新爬網站。
        # 前一晚 23:00 的排程負責建立每日 cnyes_news_YYYY-MM-DD.json。
        # 盤後讀取最近兩份可用日檔，再交給新聞 Agent 摘要。
        # ============================================================
        crawler = CnyesNewsCrawler(
            self.base,
            max_articles=int(os.getenv("CNYES_NIGHTLY_MAX_ARTICLES", "600")),
            max_scrolls=int(os.getenv("CNYES_NEWS_MAX_SCROLLS", "18")),
        )
        try:
            cnyes_news = crawler.load_recent_cached(
                before_date=report_date,
                days=int(os.getenv("CNYES_RESEARCH_LOOKBACK_DAYS", "2")),
                max_search_days=int(os.getenv("CNYES_RESEARCH_CACHE_SEARCH_DAYS", "7")),
            )
            if not cnyes_news.get("articles"):
                cnyes_news = {
                    **cnyes_news,
                    "source_method": "nightly_cache",
                    "errors": list(cnyes_news.get("errors", [])) + ["沒有找到可用的夜間 CNYES 日檔快取；盤後研究不會重新爬網站。"],
                }
        except Exception as exc:
            cnyes_news = {
                "source_url": "https://news.cnyes.com/news/cat/headline",
                "source_method": "nightly_cache",
                "crawl_date": "",
                "days": 0,
                "cache_dates": [],
                "cache_files": [],
                "article_count": 0,
                "articles": [],
                "errors": [f"讀取 CNYES 夜間快取失敗：{exc}"],
                "loaded_from_cache": True,
            }

        model_records = [r for r in self.model_health() if str(r.get("sector_code", "")).zfill(2) in sector_codes]
        previous = self._load_json(self.research_dir / "research_state.json", {}) or {}
        retrain_flags = self._retrains(model_records, previous)
        sector_news = [x for x in self.sector_news_summary(news) if str(x.code).zfill(2) in sector_codes]

        financials = self.company_financial_snapshot(symbols)
        peer_panel = self.focus_peer_panel(symbols, (twse_rev, tpex_rev))
        business_profiles_all = business_map_for_symbols(requested_symbols, client=self.fugle_client, path=self.data_dir / "business_master.csv", force=False)
        theme_research = [] if fast_mode else self.focus_theme_research(symbols)
        order_evidence = [asdict(x) for x in news if any(term.lower() in (x.title + " " + x.summary).lower() for term in FOCUS_ORDER_TERMS)]
        if fast_mode:
            for item in cnyes_news.get("articles", []) or []:
                if not isinstance(item, dict):
                    continue
                text_blob = f"{item.get('title','')} {item.get('summary','')} {item.get('content','')}"
                if any(term.lower() in text_blob.lower() for term in FOCUS_ORDER_TERMS):
                    order_evidence.append({
                        "title": item.get("title", ""), "link": item.get("url", ""), "source": "鉅亨網",
                        "published": item.get("published", ""), "summary": item.get("summary", ""),
                        "verification": "市場新聞線索，待官方交叉驗證",
                    })
        fundamentals = {s: self._select_symbol_revenue(s, twse_rev, tpex_rev) for s in symbols}

        # 法說會已移到早報。盤後 FAST MODE 不再讀取／分析法說會，
        # 避免盤後重複內容、重複呼叫 Agent，也避免 Email 再次顯示法說。
        # 法說會已移到早報；盤後報告永遠不建立 earnings_calls。
        # 即使非 FAST MODE 也不要把舊 cache 帶進盤後 Email。
        earnings_calls = []
        earnings_memo_report = {}
        if not fast_mode and str(os.getenv("RESEARCH_INCLUDE_EARNINGS", "false")).lower() in {"1", "true", "yes", "on"}:
            try:
                cache_files = sorted(self.research_dir.glob("fugle_earnings_memo_*.json"), reverse=True)
                loaded_cache = None
                for cache_file in cache_files:
                    try:
                        obj = json.loads(cache_file.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                        loaded_cache = obj
                        obj["cache_file"] = str(cache_file)
                        break

                use_cache_only = _env_bool("EARNINGS_USE_CACHE", False)
                if use_cache_only and loaded_cache:
                    earnings_memo_report = loaded_cache
                else:
                    ea = EarningsCallAgent(self.base, ollama_model=getattr(self, "ollama_model", DEFAULT_OLLAMA_MODEL))
                    earnings_memo_report = ea.daily_run(
                        days=int(os.getenv("EARNINGS_MEMO_LOOKBACK_DAYS", "5")),
                        limit=int(os.getenv("EARNINGS_MEMO_MAX_ARTICLES", "80")),
                        force=_env_bool("EARNINGS_FORCE_REFRESH", False),
                        watchlist=requested_symbols,
                    )
                earnings_calls = earnings_memo_report.get("items", [])
            except Exception as exc:
                earnings_memo_report = {"items": [], "errors": [{"error": f"Fugle 法說會 Agent 失敗：{exc}"}], "daily_digest": {}}
                earnings_calls = []

        # 市場近期漲跌：全市場當日漲跌前段候選 + 自選股補充，再以 5/20 交易日驗證。
        try:
            mover_analyzer = MarketMoverAnalyzer(self.base)
            market_movers = mover_analyzer.analyze(watchlist_symbols=requested_symbols, max_candidates=30)
        except Exception as exc:
            market_movers = {"movers": [], "industry_summary": [], "errors": [str(exc)]}
        business_profiles = business_map_for_symbols(symbols, client=self.fugle_client, path=self.data_dir / "business_master.csv", force=False)
        business_contexts = {k: build_business_research_context(v) for k, v in business_profiles.items()}

        # CNYES 新聞也納入題材偵測；與既有 Google News 資料並列，而非取代官方資料。
        cnyes_theme_rows = []
        for item in cnyes_news.get("articles", []) or []:
            if not isinstance(item, dict):
                continue
            text = f"{item.get('title', '')} {item.get('summary', '')} {item.get('content', '')}"
            cnyes_theme_rows.append({
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "source": "鉅亨網",
                "published": item.get("published", ""),
                "summary": item.get("summary", "") or item.get("content", "")[:800],
                "query": "CNYES headline",
                "sentiment": "中性/待判定",
                "sentiment_score": self._word_score(item.get("title", ""), text),
                "corroboration": 0,
                "official_match": False,
                "verification": "市場新聞線索，待官方交叉驗證",
                "business_relevance": 0.0,
                "business_match": "待 Agent 判讀",
                "entity_match": False,
            })

        bullish_themes = discover_bullish_themes(
            business_profiles=business_contexts,
            news_rows=[asdict(x) for x in news] + cnyes_theme_rows,
            financial_snapshots=financials,
            order_evidence=order_evidence,
            source_urls=INDUSTRY_CHAIN_SOURCES,
        )
        # 第二層 fallback：FOCUS_THEMES 是較寬的主題查詢，第一層若因新聞標題
        # 沒有直接出現主題詞而空白，就把這些已做過查證標記的新聞再交給 Theme Agent。
        if not bullish_themes and theme_research:
            fallback_theme_news = []
            for tr in theme_research:
                for item in tr.get("evidence", []) or []:
                    item = dict(item)
                    item["query"] = tr.get("theme", "")
                    item["title"] = item.get("title", "")
                    item["summary"] = item.get("summary", "") or item.get("title", "")
                    fallback_theme_news.append(item)
            bullish_themes = discover_bullish_themes(
                business_profiles=business_contexts,
                news_rows=[asdict(x) for x in news] + cnyes_theme_rows + fallback_theme_news,
                financial_snapshots=financials,
                order_evidence=order_evidence,
                source_urls=INDUSTRY_CHAIN_SOURCES,
            )
        market_status = market_status_text(now_taipei, self.base)
        # 盤後選取標的事件歸因：由 Dashboard 每次選股時保存最後選取標的，
        # 讓 Windows 盤後任務也能針對「你最後看的那檔股票」做今日漲跌原因研究。
        post_market_selected = None
        try:
            if fast_mode:
                raise RuntimeError("快速盤後模式略過個股即時事件歸因；避免再次啟動新聞搜尋。")
            sel_path = self.data_dir / "last_selected_symbol.json"
            last_selected = None
            if sel_path.exists():
                last_selected = (json.loads(sel_path.read_text(encoding="utf-8")) or {}).get("symbol")
            if last_selected and clean_symbol(last_selected) in requested_symbols:
                from post_market_analysis import PostMarketAnalyzer
                analyzer = PostMarketAnalyzer(self.base, ollama_model=self.ollama_model if hasattr(self, "ollama_model") else DEFAULT_OLLAMA_MODEL)
                # 使用同一份自選股作為 peer set；實際新聞查詢會依公司業務做精準化。
                post_market_selected = analyzer.analyze(clean_symbol(last_selected), industry_symbols=requested_symbols)
        except Exception as exc:
            post_market_selected = {"symbol": clean_symbol(last_selected) if 'last_selected' in locals() and last_selected else "", "error": str(exc)}

        payload = {
            "report_date": report_date,
            "report_generated_at": now_taipei.isoformat(),
            "market_status": market_status,
            "research_scope": {
                "focus_industries": FOCUS_INDUSTRIES,
                "watchlist_symbols": requested_symbols,
                "focused_symbols": symbols,
                "excluded_symbols": [s for s in requested_symbols if s not in symbols],
                "scope_policy": "自選股優先；核心研究限定半導體/光電/電子零組件；競品僅補同產業少量資料。",
            },
            "symbols": symbols,
            "business_profiles": business_profiles,
            "company_research": self.company_research_summary(symbols, news),
            "model_health": model_records,
            "retrain_flags": retrain_flags,
            "sector_news": [asdict(x) for x in sector_news],
            "news": [asdict(x) for x in news[:100]],
            "cnyes_news": cnyes_news,
            "order_evidence": order_evidence[:60],
            "fundamentals": fundamentals,
            "financial_snapshots": financials,
            "earnings_calls": earnings_calls[:160],
            "earnings_memo_report": earnings_memo_report,
            "earnings_call_primary_source": "Fugle 法說會備忘錄：https://blog.fugle.tw/topic/earnings-call-memo；每篇摘要都必須由 Ollama Agent 呼叫 read_fugle_memo 開啟詳細文章後產生。",
            "earnings_call_count": int(sum(1 for x in earnings_calls if isinstance(x, dict) and x.get("memo_opened"))),
            "earnings_memo_daily_digest": earnings_memo_report.get("daily_digest", {}) if isinstance(earnings_memo_report, dict) else {},
            "market_movers": market_movers,
            "peer_panel": peer_panel,
            "theme_research": theme_research,
            "bullish_themes": bullish_themes,
            "post_market_selected": post_market_selected,
            "official_sources": {
                "twse_announcements_rows": int(len(twse_ann)),
                "twse_revenue_rows": int(len(twse_rev)),
                "tpex_announcements_rows": int(len(tpex_ann)),
                "tpex_revenue_rows": int(len(tpex_rev)),
                "twse_income_rows": int(len(twse_income)),
                "tpex_income_rows": int(len(tpex_income)),
                "cnyes_news_articles": int(cnyes_news.get("article_count", 0) or 0),
                "twse_supply_chain_rows": int(len(supply_chain.get("TWSE", pd.DataFrame()))),
                "tpex_supply_chain_rows": int(len(supply_chain.get("TPEx", pd.DataFrame()))),
            },
            "official_industry_chain_sources": INDUSTRY_CHAIN_SOURCES,
            "supply_chain_scope": "公開揭露的供應鏈管理資訊＋TPEx 產業價值鏈公開頁面；不是完整客戶/供應商關係網路。",
            "cnyes_news_source": "https://news.cnyes.com/news/cat/headline",
            "cnyes_news_scope": "前一晚夜間爬蟲建立每日新聞快取；盤後讀取最近兩份可用日檔，再由 CNYES News Agent 摘要與研究。盤後不再重新爬網站。",
            "cnyes_news_cache_dates": cnyes_news.get("cache_dates", []),
            "cnyes_news_cache_files": cnyes_news.get("cache_files", []),
            "cnyes_news_loaded_from_cache": True,
        }
        should_retrain = [x for x in retrain_flags if x["retrain"]]
        payload_research_conclusion = (
            "建議安排重新訓練/檢查" if should_retrain else "目前沒有足夠證據要求立即重新訓練；維持監控"
        )
        payload["research_conclusion"] = payload_research_conclusion

        # ============================================================
        # CNYES News Research Agent / 晨報快取
        # 快速盤後模式直接讀取早報已產生的 digest，不再次呼叫 Qwen3 新聞摘要。
        # ============================================================
        cnyes_research_digest = {}
        try:
            morning_files = sorted(self.research_dir.glob(f"morning_{report_date}.json"), reverse=True)
            morning_loaded = None
            if morning_files:
                try:
                    morning_loaded = json.loads(morning_files[0].read_text(encoding="utf-8"))
                except Exception:
                    morning_loaded = None
            if fast_mode and isinstance(morning_loaded, dict) and isinstance(morning_loaded.get("cnyes_research_digest"), dict):
                cnyes_research_digest = {**morning_loaded.get("cnyes_research_digest", {}), "reused_from_morning_report": True}
            else:
                if CnyesNewsDigestAgent is None:
                    raise ImportError(f"無法匯入 CNYES Agent：{_CNYES_AGENT_IMPORT_ERROR}")

                # 相容不同版本的建構子：
                # 1) (base_dir, ollama_host=..., ollama_model=...)
                # 2) (base_dir, host, model)
                # 3) (base_dir)
                try:
                    cnyes_digest_agent = CnyesNewsDigestAgent(
                        self.base,
                        ollama_host=self.ollama_host,
                        ollama_model=self.ollama_model,
                    )
                except TypeError:
                    try:
                        cnyes_digest_agent = CnyesNewsDigestAgent(
                            self.base, self.ollama_host, self.ollama_model
                        )
                    except TypeError:
                        cnyes_digest_agent = CnyesNewsDigestAgent(self.base)

                payload_for_cnyes = {
                    "report_date": report_date,
                    "symbols": requested_symbols,
                    "cnyes_news": cnyes_news,
                    "market_movers": market_movers,
                }

                # 相容不同版本的執行方法名稱。
                cnyes_runner = getattr(cnyes_digest_agent, "run", None)
                if not callable(cnyes_runner):
                    for method_name in ("analyze", "summarize", "digest"):
                        candidate = getattr(cnyes_digest_agent, method_name, None)
                        if callable(candidate):
                            cnyes_runner = candidate
                            break
                if not callable(cnyes_runner):
                    raise AttributeError(
                        "CNYES Agent 沒有 run/analyze/summarize/digest 方法"
                    )

                try:
                    cnyes_research_digest = cnyes_runner(
                        payload_for_cnyes,
                        limit=int(os.getenv("CNYES_RESEARCH_AGENT_INPUT_ARTICLES", "100")),
                    )
                except TypeError:
                    cnyes_research_digest = cnyes_runner(payload_for_cnyes)

                if not isinstance(cnyes_research_digest, dict):
                    cnyes_research_digest = {
                        "agent_status": "cnyes_digest_text",
                        "model": self.ollama_model,
                        "overview": str(cnyes_research_digest or ""),
                        "key_findings": [],
                        "market_drivers": [],
                        "watch_topics": [],
                    }
        except Exception as exc:
            cnyes_research_digest = {
                "agent_status": "cnyes_digest_error",
                "model": self.ollama_model,
                "overview": "鉅亨新聞快取已取得，但新聞研究 Agent 執行失敗。",
                "key_findings": [],
                "market_drivers": [],
                "watch_topics": [],
                "error": str(exc),
            }
        
        # ============================================================
        # AI Research Assistant Agent
        #
        # 這一層不再由 EmailAgent 固定挑選「前 3 / 前 5」資訊。
        # Qwen3 會自己選工具、深入查證、建立研究筆記與標籤。
        # ============================================================
        try:
            research_assistant = ResearchOrchestratorAgent(
                self.base,
                ollama_host=self.ollama_host,
                ollama_model=self.ollama_model,
            )
            agent_notes = research_assistant.run(payload)
        except Exception as exc:
            agent_notes = {
                "agent_status": "fallback",
                "model": self.ollama_model,
                "error": f"Research Assistant Agent 失敗：{exc}",
                "overview": "本次 AI 研究助理未完成自主分析。",
                "key_takeaways": [],
                "research_notes": [],
                "positive_signals": [],
                "negative_signals": [],
                "watch_items": [str(exc)],
                "earnings_digest": "",
                "financial_focus": [],
                "model_alert": "",
                "agent_actions": [],
            }

        payload["agent_research_notes"] = agent_notes

        # 舊版欄位保留，避免 Dashboard / 舊程式讀不到 ollama_analysis。
        payload["ollama_analysis"] = {
            "available": agent_notes.get("agent_status") == "qwen3",
            "model": self.ollama_model,
            "content": agent_notes.get("overview", ""),
            "validation": "agent_orchestrator",
            "agent_status": agent_notes.get("agent_status", "fallback"),
        }

        payload["recommend_retrain"] = bool(should_retrain)
        payload["research_conclusion"] = payload_research_conclusion

        # Save snapshot for next-day drift comparison.
        state_models = {}
        for rec in model_records:
            state_models[rec["sector_code"]] = rec["metrics"]
        self._cache_json(self.research_dir / "research_state.json", {
            "saved_at": pd.Timestamp.now(tz=TAIPEI_TZ).isoformat(),
            "models": state_models,
        })

        json_path = self.research_dir / f"research_{report_date}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path = self.research_dir / f"research_{report_date}.md"
        md_path.write_text(self.to_markdown(payload), encoding="utf-8")
        payload["json_path"] = str(json_path)
        payload["markdown_path"] = str(md_path)
        return payload

    def to_markdown(self, p: dict[str, Any]) -> str:
        lines = [
            f"# 台股模型研究 Agent 報告｜{p.get('report_date', '')}",
            "",
            f"## 結論\n**{p.get('research_conclusion', '')}**",
            "",
            "## 模型健康度",
        ]
        for rec in p.get("model_health", []):
            m = rec.get("metrics", {})
            lines.append(
                f"- {rec['sector_code']} {rec['sector_name']}：樣本 {rec['training_samples']:,}、股票 {rec['training_stocks']}、"
                f"MAE {m.get('MAE', np.nan):.6f}、RMSE {m.get('RMSE', np.nan):.6f}、"
                f"方向準確率 {m.get('ClassificationDirectionAccuracy', np.nan) if np.isfinite(self._safe_float(m.get('ClassificationDirectionAccuracy'))) else float('nan'):.3f}、"
                f"模型年齡 {rec.get('age_days', np.nan)} 天"
            )
        lines += ["", "## 是否建議重新訓練"]
        for flag in p.get("retrain_flags", []):
            if flag["retrain"]:
                lines.append(f"- **{flag['sector_code']} {flag['sector_name']}：建議檢查** → {'；'.join(flag['reasons'])}")
        if not any(x["retrain"] for x in p.get("retrain_flags", [])):
            lines.append("- 目前沒有產業符合立即重訓條件。")
        lines += ["", "## 研究範圍"]
        scope = p.get("research_scope", {}) or {}
        lines.append(f"- 聚焦產業：{scope.get('focus_industries', {})}")
        lines.append(f"- 自選股研究標的：{', '.join(scope.get('focused_symbols', [])) or '目前沒有符合三大核心產業的自選股'}")
        lines.append(f"- 排除標的：{', '.join(scope.get('excluded_symbols', [])) or '無'}")
        lines += ["", "## 公司主要業務／產業鏈細分"]
        for sym, bp in (p.get("business_profiles") or {}).items():
            lines.append(f"- {sym}：{bp.get('industry_name')}｜{bp.get('business_group')}｜{bp.get('primary_chain', '')}")
        lines += ["", "## 財報 / 營收"]
        for row in p.get("financial_snapshots", [])[:40]:
            lines.append(f"- {row.get('symbol')}：月營收 {row.get('latest_month_revenue')}、營收YoY {row.get('revenue_yoy')}、營業利益 {row.get('operating_profit')}、淨利 {row.get('net_income')}、EPS {row.get('eps')}")
        lines += ["", "## 同業 / 競爭面"]
        for g in p.get("peer_panel", []):
            peers = [r.get('symbol') for r in g.get('peer_rows', [])]
            lines.append(f"- {g.get('industry_code')} {g.get('industry_name')}：自選股 {g.get('watchlist_symbols', [])}｜比較池 {peers[:12]}")
        lines += ["", "## 接單 / 出貨證據"]
        for n in p.get("order_evidence", [])[:20]:
            lines.append(f"- {n.get('title')}｜{n.get('verification')}｜{n.get('link')}")
        lines += ["", "## 法人說明會／法說會"]
        if not any(isinstance(e, dict) and e.get("event_date") for e in p.get("earnings_calls", [])):
            lines.append("- 本次沒有取得可驗證的法說會資料；請檢查 MOPS t100sb07_1 / t05st01 連線與 cache。")
        for e in p.get("earnings_calls", [])[:40]:
            if e.get("error"):
                lines.append(f"- {e.get('error')}")
                continue
            r1 = e.get("post_1d_return")
            r5 = e.get("post_5d_return")
            lines.append(
                f"- {e.get('event_date')} {e.get('symbol')}｜{e.get('impact')} {float(e.get('impact_score',0)):+.2f}｜證據 {e.get('evidence_level')}｜法說後1日 {'' if r1 is None else f'{float(r1)*100:+.2f}%'}｜法說後5日 {'' if r5 is None else f'{float(r5)*100:+.2f}%'}"
            )
            if e.get("memo_opened"):
                lines.append(f"  - Ollama Agent 一句話摘要：{e.get('one_line_summary') or e.get('summary','')}")
                for field_label, field_key in [("財務重點","financial_highlights"),("營運重點","operating_highlights"),("展望/指引","guidance"),("利多因素","positive_factors"),("利空/風險","negative_factors"),("Q&A重點","qa_highlights")]:
                    vals=e.get(field_key) or []
                    if vals:
                        lines.append("  - " + field_label + "：" + "；".join(str(v) for v in vals[:6]))
            else:
                lines.append("  - 摘要：（未取得Fugle法說會備忘錄正文，僅保留官方日程）")
            lines.append(f"  - 來源：{e.get('source_url')}" if e.get("memo_opened") else f"  - 官方日程來源：{e.get('source_url')}" )
        lines += ["", "## 近期上漲／下跌個股與產業原因"]
        mm = p.get("market_movers", {}) or {}
        for s in mm.get("industry_summary", [])[:30]:
            lines.append(
                f"- {s.get('industry_code')} {s.get('industry_name')}｜今日樣本平均 {float(s.get('today_avg_change',0)):+.2f}%｜5日 {float(s.get('ret_5d_avg',0)):+.2f}%｜20日 {float(s.get('ret_20d_avg',0)):+.2f}%｜上漲 {s.get('rising_count')}／下跌 {s.get('falling_count')}"
            )
            if s.get('top_risers'): lines.append("  - 漲幅樣本：" + "、".join(s.get('top_risers', [])[:3]))
            if s.get('top_fallers'): lines.append("  - 跌幅樣本：" + "、".join(s.get('top_fallers', [])[:3]))
        lines += ["", "## 利多題材雷達"]
        for t in p.get("bullish_themes", [])[:12]:
            lines.append(
                f"- {t.get('theme')}｜強度 {float(t.get('strength',0)):.1f}/100｜{t.get('status')}｜涉及 {t.get('affected_count',0)} 家｜營收支撐 {t.get('revenue_support_count',0)} 家｜接單證據 {t.get('order_evidence_count',0)} 筆"
            )
            if t.get("bullish_drivers"):
                lines.append("  - " + "；".join(t.get("bullish_drivers", [])[:4]))
        lines += ["", "## 產業消息",]
        for s in p.get("sector_news", []):
            lines.append(
                f"### {s['code']} {s['name']}\n"
                f"新聞數 {s['news_count']}｜加權消息情緒 {s['weighted_sentiment']:+.2f}｜可信度 {s['confidence']:.0%}\n"
                f"- 利多：{s['dominant_opportunity']}\n"
                f"- 風險：{s['dominant_risk']}"
            )
        lines += ["", "## 新聞查證樣本"]
        for n in p.get("news", [])[:30]:
            lines.append(
                f"- {n['title']}｜來源：{n['source']}｜跨來源 {n['corroboration']}｜"
                f"官方匹配：{'是' if n['official_match'] else '否'}｜{n['verification']}\n  {n['link']}"
            )
        lines += ["", "## AI 研究助理 Agent 筆記"]
        agent = p.get("agent_research_notes", {}) or {}
        if agent.get("overview"):
            lines.append(f"### Agent 總結\n{agent.get('overview')}")
        if agent.get("key_takeaways"):
            lines.append("### Agent 自主挑選的重點")
            lines.extend(f"- {x}" for x in agent.get("key_takeaways", []))
        if agent.get("research_notes"):
            lines.append("### Agent 研究筆記 / 記號")
            for n in agent.get("research_notes", []):
                tags = "、".join(n.get("tags", []))
                evidence = "；".join(n.get("evidence", []))
                follow = "；".join(n.get("follow_up", []))
                lines.append(f"- [{n.get('importance','medium')}] {n.get('title','')} | {n.get('symbol','')} | {n.get('note','')}")
                if tags: lines.append(f"  - Tags：{tags}")
                if evidence: lines.append(f"  - Evidence：{evidence}")
                if follow: lines.append(f"  - Follow-up：{follow}")
        if agent.get("positive_signals"):
            lines.append("### 正向訊號")
            lines.extend(f"- {x}" for x in agent.get("positive_signals", []))
        if agent.get("negative_signals"):
            lines.append("### 負向訊號")
            lines.extend(f"- {x}" for x in agent.get("negative_signals", []))
        if agent.get("watch_items"):
            lines.append("### 待追蹤 / 待查證")
            lines.extend(f"- {x}" for x in agent.get("watch_items", []))
        if agent.get("earnings_digest"):
            lines.append(f"### 法說會 Agent 摘要\n{agent.get('earnings_digest')}")
        if agent.get("financial_focus"):
            lines.append("### 財務重點")
            lines.extend(f"- {x}" for x in agent.get("financial_focus", []))
        if agent.get("model_alert"):
            lines.append(f"### 模型提醒\n{agent.get('model_alert')}")
        lines.append(f"### Agent 狀態\n{agent.get('agent_status','unknown')}｜工具動作 {len(agent.get('agent_actions', []))} 次")
        if agent.get("agent_actions"):
            lines.append("### Agent 工具操作紀錄")
            for action in agent.get("agent_actions", [])[-30:]:
                lines.append(f"- {action.get('tool','')}｜{action.get('reason','')}")
        lines += ["", "## 本機 Qwen3 相容摘要", p.get("ollama_analysis", {}).get("content", "未產生本機 LLM 摘要。")]
        lines += ["", "> 本報告是研究與模型治理工具，不保證新聞真偽或投資結果；涉及重大消息時應人工查看原始公告。"]
        return "\n".join(lines)


def run_after_close(base_dir: str | Path = ".", symbols: Iterable[str] = ()) -> dict[str, Any] | None:
    """盤後研究入口：每天 15:00 後皆可執行；休市日也可研究最近交易日資料。"""
    now = pd.Timestamp.now(tz=TAIPEI_TZ)
    if now.hour < 15:
        return None
    agent = ResearchAgent(base_dir)
    return agent.run_daily_research(symbols=symbols)
