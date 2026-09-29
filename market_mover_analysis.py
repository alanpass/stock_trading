# -*- coding: utf-8 -*-
"""近期台股漲跌產業與個股分析。

設計：
1. 用 Fugle 全市場 snapshot 建立當日漲跌候選池。
2. 對候選股再抓 5 / 20 交易日歷史，驗證近期趨勢。
3. 依產業代碼彙整上漲 / 下跌股票。
4. 對前段漲跌個股抓少量新聞，提供利多 / 利空原因的原始證據。
5. Qwen 可再對這些證據做摘要，但規則式結果本身也可獨立運作。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET

from stock_api import FugleClient, clean_symbol, INDUSTRIES, INDUSTRY_OVERRIDES
from business_master import business_map_for_symbols, build_business_research_context

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
POSITIVE_WORDS = ["漲價", "報價上調", "需求增加", "訂單", "接單", "擴產", "AI需求", "出貨增加", "營收成長", "獲利成長", "供給吃緊", "漲價循環", "庫存回補"]
NEGATIVE_WORDS = ["跌價", "降價", "需求下滑", "砍單", "庫存過高", "庫存調整", "營收下滑", "獲利衰退", "成本上升", "供給過剩", "需求疲弱", "關稅", "制裁"]


class MarketMoverAnalyzer:
    def __init__(self, base_dir: str | Path = ".", client: FugleClient | None = None):
        self.base = Path(base_dir).resolve()
        self.cache_dir = self.base / "data" / "research" / "movers"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or FugleClient()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 TaiwanStockResearchAgent/1.0", "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8"})

    @staticmethod
    def _float(value):
        try:
            s = str(value).replace(",", "").replace("%", "").strip()
            if s in {"", "-", "--", "None", "nan"}:
                return np.nan
            return float(s)
        except Exception:
            return np.nan

    def _industry_map_from_local(self) -> dict[str, str]:
        mapping = dict(INDUSTRY_OVERRIDES)
        path = self.base / "data" / "industry_master.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, dtype={"symbol": str, "industry": str})
                for row in df.to_dict("records"):
                    code = clean_symbol(row.get("symbol", ""))
                    ind = str(row.get("industry", "00")).zfill(2)
                    if code and ind != "nan":
                        mapping[code] = INDUSTRY_OVERRIDES.get(code, ind)
            except Exception:
                pass
        return mapping

    def _load_cached_snapshot(self) -> tuple[pd.DataFrame, str]:
        path = self.cache_dir / "market_snapshot_latest.json"
        if not path.exists():
            return pd.DataFrame(), ""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            asof = str(payload.get("asof", "")) if isinstance(payload, dict) else ""
            return pd.DataFrame(rows), asof
        except Exception:
            return pd.DataFrame(), ""

    def _save_snapshot(self, df: pd.DataFrame, asof: str) -> None:
        try:
            rows = df.to_dict("records")
            payload = {"asof": asof, "rows": rows, "source": "Fugle snapshot_quotes"}
            (self.cache_dir / "market_snapshot_latest.json").write_text(
                json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
            )
            stamp = str(asof).replace(":", "-").replace("+", "_")[:19].replace(" ", "_")
            (self.cache_dir / f"market_snapshot_{stamp}.json").write_text(
                json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    def _enrich_industries(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        local_map = self._industry_map_from_local()
        df = df.copy()
        if "industry" not in df.columns:
            df["industry"] = "00"
        df["industry"] = df["industry"].fillna("00").astype(str).map(
            lambda x: str(x).zfill(2) if x.strip() not in {"", "nan", "none"} else "00"
        )
        df["industry"] = [
            INDUSTRY_OVERRIDES.get(str(sym), local_map.get(str(sym), ind))
            for sym, ind in zip(df["symbol"].astype(str), df["industry"].tolist())
        ]
        df["industry"] = df["industry"].map(lambda x: str(x).zfill(2))
        df["industry_name"] = df["industry"].map(lambda x: INDUSTRIES.get(str(x).zfill(2), "其他"))
        return df

    def _snapshot_universe(self) -> pd.DataFrame:
        rows = []
        for market in ["TSE", "OTC"]:
            try:
                snap = self.client.snapshot_quotes(market)
                for r in snap:
                    code = clean_symbol(r.get("symbol", ""))
                    if not code:
                        continue
                    rows.append({
                        "symbol": code,
                        "name": r.get("name", code),
                        "last_price": self._float(r.get("lastPrice", r.get("price"))),
                        "change": self._float(r.get("change")),
                        "change_percent": self._float(r.get("changePercent")),
                        "volume": self._float(r.get("volume")),
                        "market": market,
                    })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        source = "Fugle即時snapshot"
        asof = pd.Timestamp.now(tz="Asia/Taipei").isoformat()

        if not df.empty:
            df = df.drop_duplicates("symbol", keep="first")
            if df["change_percent"].isna().all() and "last_price" in df and "change" in df:
                prev = df["last_price"] - df["change"]
                df["change_percent"] = np.where(prev.abs() > 1e-9, df["change"] / prev * 100, np.nan)
            # snapshot 通常沒有 industry，因此另外抓 ticker；失敗時回退本地產業主檔。
            try:
                ticker_rows = []
                for market in ["TSE", "OTC"]:
                    ticker_rows.extend(self.client.tickers(market))
                tdf = pd.DataFrame(ticker_rows)
                if not tdf.empty and "symbol" in tdf.columns:
                    keep = [c for c in ["symbol", "name", "industry"] if c in tdf.columns]
                    tdf = tdf[keep].copy().drop_duplicates("symbol")
                    df = df.merge(tdf, on="symbol", how="left", suffixes=("", "_ticker"))
                    if "name_ticker" in df.columns:
                        df["name"] = df["name"].replace("", np.nan).fillna(df["name_ticker"])
            except Exception:
                pass
            df = self._enrich_industries(df)
            if np.isfinite(df["change_percent"]).any():
                self._save_snapshot(df, asof)
                df["data_source"] = source
                df["data_asof"] = asof
                return df

        # 休市／盤後若 Fugle 沒有可用 snapshot，改用最近一次成功交易日快取。
        cached, cached_asof = self._load_cached_snapshot()
        if not cached.empty:
            cached = self._enrich_industries(cached)
            cached["data_source"] = "最近一次成功的Fugle市場snapshot"
            cached["data_asof"] = cached_asof
            return cached
        return pd.DataFrame()

    def _history(self, symbol: str) -> pd.DataFrame:
        path = self.cache_dir / f"{clean_symbol(symbol)}.csv"
        try:
            # 每日只抓一次，避免排程與前端重複拉資料。
            if path.exists() and time.time() - path.stat().st_mtime < 20 * 3600:
                return pd.read_csv(path, parse_dates=["date"])
        except Exception:
            pass
        try:
            end = pd.Timestamp.today().normalize().date()
            start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=60)).date()
            df = self.client.historical_candles(clean_symbol(symbol), start, end, "D", adjusted=False)
            if not df.empty:
                df.to_csv(path, index=False)
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _returns(df: pd.DataFrame) -> tuple[float | None, float | None]:
        if df is None or df.empty or "close" not in df.columns:
            return None, None
        d = df.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
        if len(d) < 2:
            return None, None
        last = float(d.iloc[-1]["close"])
        r5 = last / float(d.iloc[max(0, len(d)-6)]["close"]) - 1 if len(d) >= 6 else None
        r20 = last / float(d.iloc[max(0, len(d)-21)]["close"]) - 1 if len(d) >= 21 else None
        return r5, r20

    def _news(self, symbol: str, name: str, business: str = "") -> list[dict[str, Any]]:
        query = f'"{name}" {symbol} {business} 股票 漲跌 原因'
        try:
            r = self.session.get(GOOGLE_NEWS_RSS, params={"q": query, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}, timeout=15)
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception:
            return []
        out = []
        for item in root.findall(".//item")[:5]:
            title = item.findtext("title", default="")
            link = item.findtext("link", default="")
            published = item.findtext("pubDate", default="")
            desc = item.findtext("description", default="")
            score = self._word_score(title + " " + desc)
            out.append({"title": title, "link": link, "published": published, "summary": desc, "score": score})
        return out

    @staticmethod
    def _word_score(text: str) -> float:
        p = sum(text.count(x) for x in POSITIVE_WORDS)
        n = sum(text.count(x) for x in NEGATIVE_WORDS)
        if p == n == 0:
            return 0.0
        return float(np.clip((p-n)/max(1, p+n), -1, 1))

    def _reason(self, news: list[dict[str, Any]], direction: str) -> tuple[str, str, float]:
        if not news:
            return "缺少足夠公司/產業新聞，無法直接歸因", "無原始新聞證據", 0.0
        pos = [x for x in news if x["score"] >= 0.20]
        neg = [x for x in news if x["score"] <= -0.20]
        if direction == "上漲":
            pool = sorted(pos, key=lambda x: x["score"], reverse=True)
            reason = pool[0]["title"] if pool else news[0]["title"]
            kind = "利多證據" if pool else "待查證"
            conf = min(1.0, 0.35 + 0.15 * len(pos))
        else:
            pool = sorted(neg, key=lambda x: x["score"])
            reason = pool[0]["title"] if pool else news[0]["title"]
            kind = "利空證據" if pool else "待查證"
            conf = min(1.0, 0.35 + 0.15 * len(neg))
        return reason, kind, conf

    def analyze(self, watchlist_symbols: Iterable[str] = (), max_candidates: int = 30) -> dict[str, Any]:
        universe = self._snapshot_universe()
        if universe.empty:
            return {"generated_at": pd.Timestamp.now().isoformat(), "movers": [], "industry_summary": [], "errors": ["目前沒有可用市場 snapshot；請先於交易日成功取得一次全市場行情快照。"]}
        universe = universe[np.isfinite(universe["change_percent"])].copy()
        if universe.empty:
            return {"generated_at": pd.Timestamp.now().isoformat(), "movers": [], "industry_summary": [], "errors": ["目前 snapshot 沒有有效漲跌幅；可能是休市日。"]}
        up = universe.sort_values("change_percent", ascending=False).head(max_candidates // 2)
        down = universe.sort_values("change_percent", ascending=True).head(max_candidates // 2)
        wl = set(clean_symbol(s) for s in watchlist_symbols)
        extra = universe[universe["symbol"].isin(wl)]
        candidates = pd.concat([up, down, extra], ignore_index=True).drop_duplicates("symbol")
        business_profiles = business_map_for_symbols(candidates["symbol"].tolist(), path=self.base / "data" / "business_master.csv", force=False)
        records = []
        for _, r in candidates.iterrows():
            code = clean_symbol(r["symbol"])
            hist = self._history(code)
            r5, r20 = self._returns(hist)
            direction = "上漲" if r["change_percent"] > 0 else "下跌" if r["change_percent"] < 0 else "平盤"
            p = business_profiles.get(code, {})
            # 只對跨市場排名前段的公司抓新聞，控制 request 數量。
            news = []
            if abs(float(r["change_percent"])) >= float(universe["change_percent"].abs().quantile(0.88)) or code in wl:
                news = self._news(code, str(r.get("name", code)), str(p.get("business_group", "")))
            reason, evidence, confidence = self._reason(news, direction)
            records.append({
                "symbol": code,
                "name": str(r.get("name", code)),
                "industry_code": str(r.get("industry", "00")).zfill(2),
                "industry_name": r.get("industry_name", INDUSTRIES.get(str(r.get("industry", "00")).zfill(2), "其他")),
                "business_group": p.get("business_group", ""),
                "primary_chain": p.get("primary_chain", ""),
                "today_change_percent": round(float(r["change_percent"]), 3),
                "ret_5d": None if r5 is None else round(r5 * 100, 3),
                "ret_20d": None if r20 is None else round(r20 * 100, 3),
                "direction": direction,
                "reason": reason,
                "reason_type": evidence,
                "reason_confidence": round(confidence, 3),
                "news": news[:5],
                "selection_basis": "全市場當日漲跌前段候選＋自選股補充",
                "market_data_source": str(r.get("data_source", "Fugle snapshot")),
                "market_data_asof": str(r.get("data_asof", "")),
            })
        mdf = pd.DataFrame(records)
        if not mdf.empty:
            # 每個期間重新排序，保留最近最明顯的個股樣本。
            mdf["strength_5d"] = pd.to_numeric(mdf["ret_5d"], errors="coerce").fillna(0)
            mdf["strength_20d"] = pd.to_numeric(mdf["ret_20d"], errors="coerce").fillna(0)
        ind_rows = []
        for (code, name), g in mdf.groupby(["industry_code", "industry_name"], dropna=False):
            up_count = int((g["today_change_percent"] > 0).sum())
            down_count = int((g["today_change_percent"] < 0).sum())
            ind_rows.append({
                "industry_code": code, "industry_name": name, "candidate_count": int(len(g)),
                "today_avg_change": round(float(g["today_change_percent"].mean()), 3),
                "ret_5d_avg": round(float(pd.to_numeric(g["ret_5d"], errors="coerce").mean()), 3),
                "ret_20d_avg": round(float(pd.to_numeric(g["ret_20d"], errors="coerce").mean()), 3),
                "rising_count": up_count, "falling_count": down_count,
                "breadth_confidence": "較高" if len(g) >= 5 else "中" if len(g) >= 3 else "低：樣本少",
                "top_risers": [f"{x['name']}({x['symbol']}) {x['today_change_percent']:+.2f}%" for _, x in g.sort_values("today_change_percent", ascending=False).head(3).iterrows()],
                "top_fallers": [f"{x['name']}({x['symbol']}) {x['today_change_percent']:+.2f}%" for _, x in g.sort_values("today_change_percent").head(3).iterrows()],
            })
        idf = pd.DataFrame(ind_rows)
        if not idf.empty:
            idf = idf.sort_values("today_avg_change", ascending=False)
        # JSON 可序列化。
        return {
            "generated_at": pd.Timestamp.now().isoformat(),
            "universe_count": int(len(universe)),
            "candidate_count": int(len(mdf)),
            "movers": mdf.drop(columns=[c for c in ["strength_5d", "strength_20d"] if c in mdf.columns]).to_dict("records"),
            "industry_summary": idf.to_dict("records") if not idf.empty else [],
            "method": "全市場當日漲跌前段候選＋自選股補充；再以5/20交易日收盤報酬驗證；休市日若無即時snapshot則使用最近成功交易日snapshot；新聞只作原因候選，需原始公告交叉驗證。",
            "data_source": str(mdf["market_data_source"].iloc[0]) if not mdf.empty and "market_data_source" in mdf.columns else "",
            "data_asof": str(mdf["market_data_asof"].iloc[0]) if not mdf.empty and "market_data_asof" in mdf.columns else "",
        }
