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
import os
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
TWSE_DAILY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_DAILY_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
POSITIVE_WORDS = ["漲價", "報價上調", "需求增加", "訂單", "接單", "擴產", "AI需求", "出貨增加", "營收成長", "獲利成長", "供給吃緊", "漲價循環", "庫存回補"]
NEGATIVE_WORDS = ["跌價", "降價", "需求下滑", "砍單", "庫存過高", "庫存調整", "營收下滑", "獲利衰退", "成本上升", "供給過剩", "需求疲弱", "關稅", "制裁"]


def parse_tw_date(value: Any) -> str:
    """將 TWSE/TPEx 的西元或民國日期轉成 YYYY-MM-DD。"""
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    text = text.replace("/", "-").replace(".", "-")
    m = __import__("re").match(r"^(\d{3})-(\d{1,2})-(\d{1,2})$", text)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{y + 1911:04d}-{mo:02d}-{d:02d}"
    m = __import__("re").match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else ""


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

    @staticmethod
    def _roc_to_iso(value: Any) -> str:
        """將交易所民國年月日字串轉成 YYYY-MM-DD。"""
        text = str(value or "").strip().replace("/", "").replace("-", "")
        if not text.isdigit():
            return ""
        if len(text) == 7:
            try:
                return f"{int(text[:3]) + 1911:04d}-{int(text[3:5]):02d}-{int(text[5:7]):02d}"
            except Exception:
                return ""
        return ""

    def _official_daily_snapshot(self, required_date: str | None = None) -> tuple[pd.DataFrame, str, list[str]]:
        """讀取 TWSE / TPEx 官方日行情，並嚴格以實際資料日期作為 data_asof。

        required_date 若指定，會記錄是否取得該日期；不會把其他日期冒充成 required_date。
        """
        today = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
        errors: list[str] = []
        frames: list[pd.DataFrame] = []

        endpoints = [
            ("TWSE", TWSE_DAILY_ALL_URL),
            ("TPEx", TPEX_DAILY_URL),
        ]

        for label, url in endpoints:
            data = None
            last_error = ""
            for attempt in range(1, 4):
                try:
                    r = self.session.get(url, timeout=25)
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, list) and data:
                        break
                    last_error = f"{label}: API returned no rows"
                except Exception as exc:
                    last_error = f"{label}: attempt {attempt} failed: {type(exc).__name__}: {exc}"
                    if attempt < 3:
                        time.sleep(1.0 * attempt)
            if not isinstance(data, list) or not data:
                if last_error:
                    errors.append(last_error)
                continue

            try:
                df = pd.DataFrame(data)
                if df.empty:
                    continue

                if label == "TWSE":
                    date_col = next((c for c in ["Date", "date"] if c in df.columns), None)
                    code_col = next((c for c in ["Code", "股票代號", "證券代號"] if c in df.columns), None)
                    name_col = next((c for c in ["Name", "股票名稱", "證券名稱"] if c in df.columns), None)
                    close_col = next((c for c in ["ClosingPrice", "收盤價", "Close"] if c in df.columns), None)
                    vol_col = next((c for c in ["TradeVolume", "成交股數", "Volume"] if c in df.columns), None)
                    change_col = next((c for c in ["Change", "漲跌價差", "PriceChange"] if c in df.columns), None)
                    high_col = next((c for c in ["HighestPrice", "最高價", "High"] if c in df.columns), None)
                    low_col = next((c for c in ["LowestPrice", "最低價", "Low"] if c in df.columns), None)
                    open_col = next((c for c in ["OpeningPrice", "開盤價", "Open"] if c in df.columns), None)
                else:
                    date_col = next((c for c in ["Date", "date"] if c in df.columns), None)
                    code_col = next((c for c in ["SecuritiesCompanyCode", "Code", "證券代號"] if c in df.columns), None)
                    name_col = next((c for c in ["CompanyName", "Name", "證券名稱"] if c in df.columns), None)
                    close_col = next((c for c in ["Close", "ClosingPrice", "收盤價"] if c in df.columns), None)
                    vol_col = next((c for c in ["TradingShares", "TradeVolume", "成交股數", "Volume"] if c in df.columns), None)
                    change_col = next((c for c in ["Change", "漲跌", "漲跌價差", "PriceChange"] if c in df.columns), None)
                    high_col = next((c for c in ["High", "HighestPrice", "最高價"] if c in df.columns), None)
                    low_col = next((c for c in ["Low", "LowestPrice", "最低價"] if c in df.columns), None)
                    open_col = next((c for c in ["Open", "OpeningPrice", "開盤價"] if c in df.columns), None)

                if not date_col or not code_col or not close_col:
                    errors.append(f"{label}: 缺少日期／代號／收盤欄位")
                    continue

                out = pd.DataFrame({
                    "source_date": df[date_col].map(parse_tw_date),
                    "symbol": df[code_col].map(clean_symbol),
                    "name": df[name_col] if name_col else df[code_col].map(clean_symbol),
                    "close": df[close_col].map(self._float),
                    "change_value": df[change_col].map(self._float) if change_col else np.nan,
                    "volume": df[vol_col].map(self._float) if vol_col else np.nan,
                    "high": df[high_col].map(self._float) if high_col else np.nan,
                    "low": df[low_col].map(self._float) if low_col else np.nan,
                    "open": df[open_col].map(self._float) if open_col else np.nan,
                })
                out = out[out["symbol"].astype(str).str.len().between(4, 6)]
                out = out[np.isfinite(out["close"])].copy()
                out["name"] = out["name"].astype(str).str.strip()
                out = out[out["source_date"].notna()].copy()
                out["source"] = label
                frames.append(out)
            except Exception as exc:
                errors.append(f"{label}: parse failed: {type(exc).__name__}: {exc}")

        if not frames:
            return pd.DataFrame(), "", errors

        combined = pd.concat(frames, ignore_index=True)
        valid_dates = pd.to_datetime(combined["source_date"], errors="coerce").dropna()
        if valid_dates.empty:
            return pd.DataFrame(), "", errors

        # 官方來源若存在不同日期，嚴格只取最新實際交易日。
        trade_date = valid_dates.max().date().isoformat()
        out = combined[combined["source_date"].astype(str).str[:10] == trade_date].copy()

        if out.empty:
            return pd.DataFrame(), "", errors

        # 同一股票可能同時存在 TWSE / TPEx 來源；保留第一筆有效名稱與價格。
        out = out.sort_values(["symbol", "source"]).drop_duplicates("symbol", keep="first")
        out["date"] = pd.to_datetime(out["source_date"], errors="coerce")
        out["previous_close"] = out["close"] - out["change_value"]
        out.loc[~np.isfinite(out["previous_close"]), "previous_close"] = np.nan
        out["change"] = out["change_value"]
        out["change_percent"] = np.where(
            np.isfinite(out["change"]) & np.isfinite(out["previous_close"]) & (out["previous_close"] != 0),
            out["change"] / out["previous_close"] * 100.0,
            0.0,
        )
        out["data_asof"] = trade_date
        out["is_trading_day"] = trade_date == today
        out["data_date_verified"] = True

        if required_date and trade_date != required_date:
            errors.append(f"official latest date {trade_date} != required date {required_date}")

        out = self._enrich_industries(out)
        return out, trade_date, errors

    @staticmethod
    def _cache_date(asof: str) -> str:
        text = str(asof or "")
        m = __import__("re").search(r"(20\d{2}-\d{2}-\d{2})", text)
        return m.group(1) if m else ""

    @staticmethod
    def _normalize_snapshot_response(response: Any, market: str, fallback_date: str) -> pd.DataFrame:
        """將 Fugle snapshot/quotes v1.0 回應統一成 DataFrame。

        Fugle 回應格式為 {date,time,market,data:[...]}；股票資料位於 data。
        """
        if isinstance(response, dict):
            rows = response.get("data", [])
            if not isinstance(rows, list):
                return pd.DataFrame()
            top_date = str(response.get("date") or fallback_date).strip()
            top_time = str(response.get("time") or "").strip()
        elif isinstance(response, list):
            rows = [x for x in response if isinstance(x, dict)]
            top_date = fallback_date
            top_time = ""
        else:
            return pd.DataFrame()

        rows = [x for x in rows if isinstance(x, dict)]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).copy()
        df["snapshot_date"] = top_date
        df["snapshot_time"] = top_time
        df["snapshot_market"] = market

        alias_map = {
            "tradePrice": "close", "lastPrice": "close", "closePrice": "close",
            "changePercent": "change_percent", "openPrice": "open",
            "highPrice": "high", "lowPrice": "low", "tradeVolume": "volume",
            "tradeValue": "trade_value", "lastUpdated": "last_updated",
        }
        for old, new in alias_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

        if "symbol" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()

        df["symbol"] = df["symbol"].map(clean_symbol)
        df["close"] = df["close"].map(MarketMoverAnalyzer._float)
        if "change" in df.columns:
            df["change"] = df["change"].map(MarketMoverAnalyzer._float)
        else:
            df["change"] = np.nan
        if "change_percent" in df.columns:
            df["change_percent"] = df["change_percent"].map(MarketMoverAnalyzer._float)
        else:
            df["change_percent"] = np.nan

        previous = None
        if "previousClose" in df.columns:
            previous = df["previousClose"].map(MarketMoverAnalyzer._float)
        elif "previous_close" in df.columns:
            previous = df["previous_close"].map(MarketMoverAnalyzer._float)

        if previous is not None:
            missing_pct = ~np.isfinite(df["change_percent"])
            calc_pct = np.where(
                np.isfinite(df["change"]) & np.isfinite(previous) & (previous != 0),
                df["change"] / previous * 100.0,
                np.nan,
            )
            df.loc[missing_pct, "change_percent"] = calc_pct[missing_pct]

            missing_change = ~np.isfinite(df["change"])
            calc_change = np.where(
                missing_change & np.isfinite(df["change_percent"]) & np.isfinite(previous),
                previous * df["change_percent"] / 100.0,
                np.nan,
            )
            df.loc[missing_change, "change"] = calc_change[missing_change]

        df = df[df["symbol"].astype(str).str.len().between(4, 6) & np.isfinite(df["close"])].copy()
        if df.empty:
            return pd.DataFrame()

        parsed_dates = pd.to_datetime(df["snapshot_date"], errors="coerce")
        df["date"] = parsed_dates
        df["data_date_verified"] = parsed_dates.notna()
        df["data_asof"] = df["snapshot_date"].astype(str).str[:10]
        df.loc[df["data_asof"].isin(["", "nan", "None"]), "data_asof"] = fallback_date
        return df

    def _snapshot_universe(self) -> pd.DataFrame:
        """取得市場全集；盤後優先使用 Fugle snapshot/quotes。

        重要修正：Fugle v1.0 snapshot/quotes 回傳 dict，股票資料在 data。
        舊版直接 pd.DataFrame(response) 會導致整個 snapshot 被丟棄。
        """
        now = pd.Timestamp.now(tz="Asia/Taipei")
        today = now.strftime("%Y-%m-%d")
        after_close = os.getenv("AFTER_CLOSE_FAST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
        after_close = after_close or (now.hour > 14 or (now.hour == 14 and now.minute >= 30))

        frames: list[pd.DataFrame] = []
        fugle_errors: list[str] = []

        # 1) Fugle snapshot/quotes：一次拿上市、上櫃全市場。
        for market in ("TSE", "OTC"):
            try:
                response = self.client.snapshot_quotes(market=market)
                df = self._normalize_snapshot_response(response, market, today)
                if df.empty:
                    fugle_errors.append(f"Fugle snapshot/quotes/{market}: 無有效資料")
                    continue

                # API 文件本身提供頂層 date。若沒有 date，盤後因為交易日已由
                # 上層 market calendar 確認，使用本次執行日作為時間錨點。
                if after_close:
                    df["data_asof"] = today
                    df["data_date_verified"] = True
                    df["is_trading_day"] = True
                else:
                    df["is_trading_day"] = df["data_asof"].astype(str).str[:10].eq(today)

                df["data_source"] = f"Fugle snapshot_quotes/{market}"
                df = self._enrich_industries(df)
                frames.append(df)
            except Exception as exc:
                fugle_errors.append(f"Fugle snapshot/quotes/{market}: {type(exc).__name__}: {exc}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined[np.isfinite(combined["close"])].copy()
            if not combined.empty:
                # 先取今天的資料；若盤後沒有日期欄位，前面已用今天做錨定。
                if after_close:
                    combined = combined[combined["data_asof"].astype(str).str[:10] == today].copy()
                else:
                    combined = combined[combined["data_asof"].astype(str).str[:10] == today].copy()

                if not combined.empty:
                    combined["data_asof"] = today
                    combined["is_trading_day"] = True
                    combined["data_date_verified"] = True

                    # 重新確保 changePercent 可計算。
                    if "previousClose" in combined.columns:
                        prev = combined["previousClose"].map(self._float)
                    else:
                        prev = combined["close"] - combined["change"]
                    missing_pct = ~np.isfinite(combined["change_percent"])
                    calc_pct = np.where(
                        np.isfinite(combined["change"]) & np.isfinite(prev) & (prev != 0),
                        combined["change"] / prev * 100.0,
                        np.nan,
                    )
                    combined.loc[missing_pct, "change_percent"] = calc_pct[missing_pct]
                    self._save_snapshot(combined, today)
                    return combined

        # 2) 官方 TWSE / TPEx 日收盤。
        official, official_date, official_errors = self._official_daily_snapshot(required_date=today)
        if not official.empty:
            if official_date == today:
                official["data_source"] = "TWSE/TPEx官方日收盤"
                official["data_asof"] = today
                official["is_trading_day"] = True
                official["data_date_verified"] = True
                self._save_snapshot(official, today)
                return official
            official["data_source"] = "TWSE/TPEx官方最近交易日"
            official["data_asof"] = official_date
            official["is_trading_day"] = False
            official["data_date_verified"] = True
            return official

        # 3) 今日已驗證 cache。
        cached, cached_asof = self._load_cached_snapshot()
        cache_date = self._cache_date(cached_asof)
        if not cached.empty and cache_date == today:
            cached = self._enrich_industries(cached)
            cached["data_source"] = "當日已驗證市場快取"
            cached["data_asof"] = today
            cached["is_trading_day"] = True
            cached["data_date_verified"] = True
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
            end = pd.Timestamp.now(tz="Asia/Taipei").date()
            start = end - pd.Timedelta(days=60)
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
        if not universe.empty and "is_trading_day" in universe.columns:
            flags = pd.to_numeric(universe["is_trading_day"], errors="coerce").dropna()
            if not flags.empty and bool((flags == 0).all()):
                asof = str(universe.get("data_asof", pd.Series(dtype=object)).dropna().iloc[0]) if "data_asof" in universe.columns and not universe["data_asof"].dropna().empty else ""
                return {
                    "data_asof": asof,
                    "is_trading_day": False,
                    "data_source": str(universe.get("data_source", pd.Series(["official"])).iloc[0]) if "data_source" in universe.columns else "official",
                    "movers": [],
                    "industry_summary": [],
                    "errors": [],
                    "note": "今天沒有台股交易，未將最近交易日資料冒充今日盤後行情。",
                }
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
        data_asof = str(mdf["market_data_asof"].iloc[0]) if not mdf.empty and "market_data_asof" in mdf.columns else ""
        data_source = str(mdf["market_data_source"].iloc[0]) if not mdf.empty and "market_data_source" in mdf.columns else ""
        verified = bool(universe["data_date_verified"].dropna().astype(bool).all()) if "data_date_verified" in universe.columns and not universe["data_date_verified"].dropna().empty else False
        trading_flag = bool(universe["is_trading_day"].dropna().astype(bool).all()) if "is_trading_day" in universe.columns and not universe["is_trading_day"].dropna().empty else None
        return {
            "generated_at": pd.Timestamp.now().isoformat(),
            "universe_count": int(len(universe)),
            "candidate_count": int(len(mdf)),
            "movers": mdf.drop(columns=[c for c in ["strength_5d", "strength_20d"] if c in mdf.columns]).to_dict("records"),
            "industry_summary": idf.to_dict("records") if not idf.empty else [],
            "method": "全市場官方日收盤／可驗證市場資料＋自選股補充；再以5/20交易日收盤報酬驗證；新聞只作原因候選，需原始公告交叉驗證。",
            "data_source": data_source,
            "data_asof": data_asof,
            "is_trading_day": trading_flag,
            "data_date_verified": verified,
        }
