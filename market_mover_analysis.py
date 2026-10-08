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

    def _http_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET JSON；遇到企業網路／憑證攔截時，再嘗試 verify=False。"""
        last_exc = None
        for verify in (True, False):
            try:
                r = self.session.get(url, params=params, timeout=30, verify=verify)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                last_exc = exc
        raise last_exc if last_exc else RuntimeError(f"無法取得 JSON：{url}")

    def _official_daily_snapshot(self, required_date: str | None = None) -> tuple[pd.DataFrame, str, list[str]]:
        """取得 TWSE/TPEx 最新日收盤，並只用來源實際日期驗證。

        盤後優先使用 TWSE RWD / TPEx 最新日行情；TWSE OpenAPI STOCK_DAY_ALL
        僅作最後的最新快照來源，若日期不是 required_date，不會冒充成當日。
        """
        today = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
        errors: list[str] = []
        frames: list[pd.DataFrame] = []
        request_date = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y%m%d")

        def build_frame(label: str, rows: list, fields: list[str] | None = None) -> pd.DataFrame:
            if not rows:
                return pd.DataFrame()
            if fields and all(isinstance(x, (list, tuple)) for x in rows):
                df = pd.DataFrame(rows, columns=fields[:len(rows[0])])
            else:
                df = pd.DataFrame(rows)
            if df.empty:
                return pd.DataFrame()

            if label.startswith("TWSE"):
                date_col = next((c for c in ["Date", "日期", "date"] if c in df.columns), None)
                code_col = next((c for c in ["Code", "證券代號", "股票代號"] if c in df.columns), None)
                name_col = next((c for c in ["Name", "證券名稱", "股票名稱"] if c in df.columns), None)
                close_col = next((c for c in ["ClosingPrice", "收盤價", "Close"] if c in df.columns), None)
                vol_col = next((c for c in ["TradeVolume", "成交股數", "Volume"] if c in df.columns), None)
                change_col = next((c for c in ["Change", "漲跌價差", "PriceChange"] if c in df.columns), None)
                high_col = next((c for c in ["HighestPrice", "最高價", "High"] if c in df.columns), None)
                low_col = next((c for c in ["LowestPrice", "最低價", "Low"] if c in df.columns), None)
                open_col = next((c for c in ["OpeningPrice", "開盤價", "Open"] if c in df.columns), None)
            else:
                date_col = next((c for c in ["Date", "日期", "date"] if c in df.columns), None)
                code_col = next((c for c in ["SecuritiesCompanyCode", "Code", "證券代號"] if c in df.columns), None)
                name_col = next((c for c in ["CompanyName", "Name", "證券名稱"] if c in df.columns), None)
                close_col = next((c for c in ["Close", "ClosingPrice", "收盤價"] if c in df.columns), None)
                vol_col = next((c for c in ["TradingShares", "TradeVolume", "成交股數", "Volume"] if c in df.columns), None)
                change_col = next((c for c in ["Change", "漲跌", "漲跌價差", "PriceChange"] if c in df.columns), None)
                high_col = next((c for c in ["High", "HighestPrice", "最高價"] if c in df.columns), None)
                low_col = next((c for c in ["Low", "LowestPrice", "最低價"] if c in df.columns), None)
                open_col = next((c for c in ["Open", "OpeningPrice", "開盤價"] if c in df.columns), None)

            if not date_col or not code_col or not close_col:
                return pd.DataFrame()
            out = pd.DataFrame({
                "source_date": df[date_col].map(parse_tw_date),
                "symbol": df[code_col].map(clean_symbol),
                "name": df[name_col].astype(str).str.strip() if name_col else df[code_col].map(clean_symbol),
                "close": df[close_col].map(self._float),
                "change_value": df[change_col].map(self._float) if change_col else np.nan,
                "volume": df[vol_col].map(self._float) if vol_col else np.nan,
                "high": df[high_col].map(self._float) if high_col else np.nan,
                "low": df[low_col].map(self._float) if low_col else np.nan,
                "open": df[open_col].map(self._float) if open_col else np.nan,
            })
            out = out[out["symbol"].astype(str).str.len().between(4, 6)]
            out = out[np.isfinite(out["close"]) & out["source_date"].astype(bool)].copy()
            out["source"] = label
            return out

        # 1) TWSE RWD：同日盤後資料較適合 14:30 之後的報告。
        try:
            payload = self._http_json(
                "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL",
                params={"date": request_date, "response": "json"},
            )
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            fields = payload.get("fields", []) if isinstance(payload, dict) else []
            out = build_frame("TWSE-RWD", rows, fields if isinstance(fields, list) else None)
            if not out.empty:
                frames.append(out)
            else:
                errors.append("TWSE RWD STOCK_DAY_ALL：無法解析或 data 為空")
        except Exception as exc:
            errors.append(f"TWSE RWD：{type(exc).__name__}: {exc}")

        # 2) TPEx 最新日收盤。
        try:
            data = self._http_json(TPEX_DAILY_URL)
            out = build_frame("TPEx", data if isinstance(data, list) else [])
            if not out.empty:
                frames.append(out)
            else:
                errors.append("TPEx OpenAPI：無法解析或回應為空")
        except Exception as exc:
            errors.append(f"TPEx OpenAPI：{type(exc).__name__}: {exc}")

        # 3) TWSE OpenAPI 作為第三層 fallback。此端點可能有資料延遲，
        # 所以只有真的等於 required_date 才接受成今日盤後資料。
        if not any(x.get("source")=="TWSE-RWD" for x in frames):
            try:
                data = self._http_json(TWSE_DAILY_ALL_URL)
                out = build_frame("TWSE-OpenAPI", data if isinstance(data, list) else [])
                if not out.empty:
                    latest = str(out["source_date"].max())
                    if required_date and latest != required_date:
                        errors.append(f"TWSE OpenAPI：最新日期 {latest} != 要求日期 {required_date}，不採用")
                    else:
                        frames.append(out)
                else:
                    errors.append("TWSE OpenAPI：無法解析或回應為空")
            except Exception as exc:
                errors.append(f"TWSE OpenAPI：{type(exc).__name__}: {exc}")

        if not frames:
            return pd.DataFrame(), "", errors

        combined = pd.concat(frames, ignore_index=True)
        combined["source_date"] = combined["source_date"].astype(str).str[:10]
        valid_dates = pd.to_datetime(combined["source_date"], errors="coerce").dropna()
        if valid_dates.empty:
            return pd.DataFrame(), "", errors + ["官方行情：日期全部無法解析"]
        trade_date = valid_dates.max().date().isoformat()
        out = combined[combined["source_date"] == trade_date].copy()
        if out.empty:
            return pd.DataFrame(), "", errors + ["官方行情：最新日期沒有有效資料"]

        out = out.sort_values(["symbol", "source"]).drop_duplicates("symbol", keep="first")
        out["date"] = pd.to_datetime(out["source_date"], errors="coerce")
        out["previous_close"] = out["close"] - out["change_value"]
        out.loc[~np.isfinite(out["previous_close"]), "previous_close"] = np.nan
        out["change"] = out["change_value"]
        out["change_percent"] = np.where(
            np.isfinite(out["change"]) & np.isfinite(out["previous_close"]) & (out["previous_close"] != 0),
            out["change"] / out["previous_close"] * 100.0,
            np.nan,
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
    def _normalize_snapshot_response(response: Any, market: str, fallback_date: str = "") -> pd.DataFrame:
        """將 Fugle snapshot/quotes 回應轉成 DataFrame，並盡可能從來源本身取得日期。

        Fugle 文件的回應為 {date,time,market,data:[...]}；data 列另含 type、symbol、
        closePrice、changePercent、lastUpdated 等欄位。若 wrapper 把頂層 date 拿掉，
        再從列內 date 或 lastUpdated 還原；絕不單靠程式當天日期冒充來源日期。
        """
        if isinstance(response, dict):
            rows = response.get("data", [])
            if not isinstance(rows, list):
                return pd.DataFrame()
            top_date = str(response.get("date") or "").strip()
            top_time = str(response.get("time") or "").strip()
        elif isinstance(response, list):
            rows = [x for x in response if isinstance(x, dict)]
            top_date = ""
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
        df["change"] = df["change"].map(MarketMoverAnalyzer._float) if "change" in df.columns else np.nan
        df["change_percent"] = df["change_percent"].map(MarketMoverAnalyzer._float) if "change_percent" in df.columns else np.nan

        previous = None
        if "previousClose" in df.columns:
            previous = df["previousClose"].map(MarketMoverAnalyzer._float)
        elif "previous_close" in df.columns:
            previous = df["previous_close"].map(MarketMoverAnalyzer._float)
        # 漲跌幅一律優先由「現價 + 漲跌額 / 昨收」重新計算，
        # 不直接相信 wrapper 轉換後的 changePercent。
        # 這可避免 API 欄位是比例值(0.0988)卻被當成百分比 0.10%，
        # 也可避免某些錯誤欄位造成 100%~1000% 的假漲跌。
        if previous is None:
            previous = pd.Series(np.nan, index=df.index, dtype=float)
        derived_previous = df["close"] - df["change"]
        previous = previous.where(np.isfinite(previous), derived_previous)
        calc_pct = np.where(
            np.isfinite(df["change"]) & np.isfinite(previous) & (previous != 0),
            df["change"] / previous * 100.0,
            np.nan,
        )
        df["change_percent"] = np.where(
            np.isfinite(calc_pct),
            calc_pct,
            np.where(
                np.isfinite(df["change_percent"]),
                np.where(np.abs(df["change_percent"]) <= 1, df["change_percent"] * 100.0, df["change_percent"]),
                np.nan,
            ),
        )
        df["previous_close"] = previous

        # 還原實際日期。
        date_series = pd.Series([top_date] * len(df), index=df.index, dtype="object")
        for c in ("date", "tradeDate", "snapshotDate", "dataDate"):
            if c in df.columns:
                vals = df[c].astype(str).str.strip()
                mask = date_series.astype(str).str.strip().isin(["", "nan", "None", "NaT"])
                date_series.loc[mask] = vals.loc[mask]

        parsed = pd.to_datetime(date_series, errors="coerce")
        # 若沒有日期，再由 lastUpdated（Fugle 為 epoch microseconds）推導。
        if "last_updated" in df.columns:
            try:
                epoch = pd.to_numeric(df["last_updated"], errors="coerce")
                # 常見為 microseconds；也容忍 milliseconds / seconds。
                unit = "us"
                finite = epoch.dropna()
                if not finite.empty and float(finite.median()) < 10_000_000_000:
                    unit = "s"
                elif not finite.empty and float(finite.median()) < 10_000_000_000_000:
                    unit = "ms"
                ep = pd.to_datetime(epoch, unit=unit, errors="coerce", utc=True).dt.tz_convert("Asia/Taipei")
                parsed = parsed.fillna(ep)
            except Exception:
                pass
        df["date"] = parsed
        df["data_date_verified"] = df["date"].notna()
        df["data_asof"] = df["date"].dt.strftime("%Y-%m-%d").fillna("")

        df = df[df["symbol"].astype(str).str.len().between(4, 6) & np.isfinite(df["close"])].copy()
        return df

    @staticmethod
    def _filter_tradeable_equity_rows(df: pd.DataFrame) -> pd.DataFrame:
        """只保留現貨股票／ETF，排除權證、指數、零股等非主體市場標的。

        Fugle snapshot/quotes 的 `type` 欄位會區分 EQUITY / WARRANT / INDEX /
        ODDLOT；`EQUITY` 同時涵蓋一般股票與 ETF，因此適合盤後「股票市場
        漲跌重點」使用。若某些 client wrapper 未保留 `type`，再以名稱做
        保守的權證排除，避免把權證 +800%、+1500% 等倍率誤當成股票。
        """
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()

        if "type" in out.columns:
            t = out["type"].astype(str).str.upper().str.strip()
            # 明確只留 Fugle 的現貨 EQUITY。
            known = t[t.notna() & ~t.isin({"", "NAN", "NONE"})]
            if not known.empty:
                out = out[t.eq("EQUITY")].copy()

        if out.empty:
            return out

        # 某些 wrapper 可能把 type 丟掉；以權證常見名稱特徵做第二層保護。
        if "name" in out.columns:
            name = out["name"].astype(str).str.strip()
            warrant_terms = r"認購|認售|購[0-9A-Za-z]*|售[0-9A-Za-z]*|牛證|熊證|權證"
            out = out[~name.str.contains(warrant_terms, regex=True, na=False)].copy()

        # 明確排除常見指數／非個股代碼，以及臺股市場常見 6 碼權證。
        # 一般上市櫃股票與 ETF 的代號通常為 4 碼；這裡只讓 4 碼現貨進入
        # 「今日市場漲跌重點」，避免再次把權證倍率列入股票排名。
        sym = out["symbol"].astype(str).str.upper().str.strip()
        out = out[~sym.str.startswith(("IX", "TX"), na=False)].copy()
        out = out[sym.str.len().eq(4)].copy()

        return out

    def _resolve_fugle_api_key(self) -> str:
        """從環境、FugleClient 或 stock_api 模組找 Key；不在 log 印出 Key。"""
        candidates = [os.getenv("FUGLE_API_KEY", "").strip()]
        for attr in ("api_key", "_api_key", "key", "_key"):
            try:
                value = str(getattr(self.client, attr, "") or "").strip()
                if value:
                    candidates.append(value)
            except Exception:
                pass
        try:
            if hasattr(self.client, "session"):
                value = str(getattr(self.client.session, "headers", {}).get("X-API-KEY", "") or "").strip()
                if value:
                    candidates.append(value)
        except Exception:
            pass
        try:
            import stock_api as _stock_api
            for name in ("FUGLE_API_KEY", "API_KEY", "fugle_api_key"):
                value = str(getattr(_stock_api, name, "") or "").strip()
                if value:
                    candidates.append(value)
        except Exception:
            pass
        for value in candidates:
            if value and value.lower() not in {"none", "null", "your_api_key"}:
                return value
        return ""

    def _direct_fugle_snapshot(self, market: str) -> Any:
        """直接呼叫 Fugle v1.0 snapshot API，作為 FugleClient wrapper 的 fallback。"""
        key = self._resolve_fugle_api_key()
        if not key:
            raise RuntimeError("找不到 FUGLE_API_KEY")
        url = f"https://api.fugle.tw/marketdata/v1.0/stock/snapshot/quotes/{market}"
        r = self.session.get(
            url,
            params={"type": "ALLBUT0999"},
            headers={"X-API-KEY": key},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def _snapshot_universe(self) -> pd.DataFrame:
        """取得全市場股票／ETF行情，盤後只接受來源可驗證的今日資料。"""
        now = pd.Timestamp.now(tz="Asia/Taipei")
        today = now.strftime("%Y-%m-%d")
        after_close = os.getenv("AFTER_CLOSE_FAST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
        after_close = after_close or (now.hour > 14 or (now.hour == 14 and now.minute >= 30))

        # 1) Fugle snapshot：盤後即時／收盤快照。優先使用現有 wrapper，再用
        # 直接 REST API fallback；Fugle 文件明確提供 date/time/data 及 type。
        frames: list[pd.DataFrame] = []
        fugle_errors: list[str] = []
        for market in ("TSE", "OTC"):
            responses = []
            try:
                try:
                    responses.append(self.client.snapshot_quotes(market=market, type="ALLBUT0999"))
                except TypeError:
                    responses.append(self.client.snapshot_quotes(market=market))
            except Exception as exc:
                fugle_errors.append(f"FugleClient {market}: {type(exc).__name__}: {exc}")
            try:
                responses.append(self._direct_fugle_snapshot(market))
            except Exception as exc:
                fugle_errors.append(f"Fugle REST {market}: {type(exc).__name__}: {exc}")

            accepted = False
            seen = set()
            for response in responses:
                key = repr(response)[:200]
                if key in seen:
                    continue
                seen.add(key)
                try:
                    df = self._normalize_snapshot_response(response, market)
                    if df.empty:
                        continue
                    df = self._filter_tradeable_equity_rows(df)
                    if df.empty:
                        continue
                    actual_dates = pd.to_datetime(df["date"], errors="coerce").dropna() if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
                    actual_date = actual_dates.max().date().isoformat() if not actual_dates.empty else ""
                    if not actual_date:
                        # 最後更新時間無法解析時，不允許盤後把今天當資料日期。
                        fugle_errors.append(f"Fugle {market}: 有行情但無法驗證資料日期")
                        continue
                    if after_close and actual_date != today:
                        fugle_errors.append(f"Fugle {market}: 實際日期 {actual_date} != {today}")
                        continue
                    df["data_asof"] = actual_date
                    df["is_trading_day"] = actual_date == today
                    df["data_date_verified"] = True
                    df["data_source"] = f"Fugle snapshot_quotes/{market}"
                    frames.append(self._enrich_industries(df))
                    accepted = True
                    break
                except Exception as exc:
                    fugle_errors.append(f"Fugle {market} parse: {type(exc).__name__}: {exc}")
            if not accepted:
                continue

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = self._filter_tradeable_equity_rows(combined)
            combined = combined[np.isfinite(combined["close"])].copy()
            if not combined.empty:
                dates = combined["data_asof"].astype(str).str[:10]
                if after_close:
                    combined = combined[dates == today].copy()
                if not combined.empty:
                    combined["data_asof"] = today if after_close else combined["data_asof"].astype(str).str[:10]
                    combined["is_trading_day"] = True if after_close else combined["data_asof"].eq(today)
                    combined["data_date_verified"] = True
                    self._save_snapshot(combined, today)
                    return combined

        # 2) 官方同日盤後行情：RWD / TPEx。
        official, official_date, official_errors = self._official_daily_snapshot(required_date=today)
        if not official.empty:
            if official_date == today:
                official["data_source"] = "TWSE/TPEx官方日收盤"
                official["data_asof"] = today
                official["is_trading_day"] = True
                official["data_date_verified"] = True
                self._save_snapshot(official, today)
                return official
            # 官方只拿到舊日期：可以用來判斷休市，但不會冒充今日。
            official["data_source"] = "TWSE/TPEx官方最近交易日"
            official["data_asof"] = official_date
            official["is_trading_day"] = False
            official["data_date_verified"] = True
            return official

        # 3) 今天已驗證快取。
        cached, cached_asof = self._load_cached_snapshot()
        cache_date = self._cache_date(cached_asof)
        if not cached.empty and cache_date == today:
            cached = self._filter_tradeable_equity_rows(self._enrich_industries(cached))
            if not cached.empty:
                cached["data_source"] = "當日已驗證市場快取"
                cached["data_asof"] = today
                cached["is_trading_day"] = True
                cached["data_date_verified"] = True
                return cached

        empty = pd.DataFrame()
        empty.attrs["official_errors"] = official_errors
        empty.attrs["fugle_errors"] = fugle_errors
        empty.attrs["today"] = today
        return empty

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
            details = []
            details.extend(list(universe.attrs.get("official_errors", [])))
            details.extend(list(universe.attrs.get("fugle_errors", [])))
            if not details:
                details.append("目前沒有可用市場行情。")
            return {
                "generated_at": pd.Timestamp.now().isoformat(),
                "movers": [],
                "industry_summary": [],
                "errors": details,
                "data_asof": "",
                "is_trading_day": None,
                "data_date_verified": False,
            }
        universe = self._filter_tradeable_equity_rows(universe)
        # 第二道防線：只允許四碼現貨代號進入市場漲跌排名。
        universe = universe[universe["symbol"].astype(str).str.fullmatch(r"\d{4}", na=False)].copy()
        # 漲跌幅若極端異常，多半是權證／比例誤讀；直接排除，避免污染 Email。
        universe = universe[np.isfinite(universe["change_percent"]) & (universe["change_percent"].abs() <= 30)].copy()
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
