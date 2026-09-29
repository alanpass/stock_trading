# -*- coding: utf-8 -*-
"""Fugle / TWSE / TPEx 資料存取層。

本模組負責：
1. Fugle 即時報價、Ticker、分 K、歷史日 K。
2. TWSE T86 上市三大法人。
3. TPEx 上櫃三大法人（OpenAPI + 歷史日報下載 fallback）。
"""
from __future__ import annotations

import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

try:
    from industry_master import INDUSTRY_OVERRIDES, normalize_industry
except Exception:
    INDUSTRY_OVERRIDES = {}
    def normalize_industry(value):
        return str(value or "00").zfill(2)

BASE = "https://api.fugle.tw/marketdata/v1.0/stock"
TWSE_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_3INSTI = "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading"
# TPEx 舊版三大法人日報下載端點，能指定民國日期，可補足 OpenAPI 只提供最新日的限制。
TPEX_3INSTI_DOWNLOAD = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_download.php"

FUGLE_API_KEY = "MjQwNTMxZTgtYTIwOS00Y2VkLTg5ZTktZjVlMDgxNjMzMWUxIGE0ZjA1NDIxLWZmZjAtNDZhOC1hN2RmLTZjNmVjYzdiZjA1OQ=="

INDUSTRIES = {
    "01":"水泥工業","02":"食品工業","03":"塑膠工業","04":"紡織纖維","05":"電機機械",
    "06":"電器電纜","08":"玻璃陶瓷","09":"造紙工業","10":"鋼鐵工業","11":"橡膠工業",
    "12":"汽車工業","14":"建材營造","15":"航運業","16":"觀光餐旅","17":"金融保險",
    "19":"綜合","20":"其他","21":"化學工業","22":"生技醫療業","23":"油電燃氣業",
    "24":"半導體業","25":"電腦及週邊設備業","26":"光電業","27":"通信網路業","28":"電子零組件業",
    "29":"電子通路業","30":"資訊服務業","31":"其他電子業","32":"文化創意業","33":"農業科技業",
    "35":"綠能環保","36":"數位雲端","37":"運動休閒","38":"居家生活","80":"管理股票","00":"ETF/無產業分類"
}


class APIError(RuntimeError):
    pass


class FugleClient:
    def __init__(self, api_key: str | None = None, timeout: int = 20):
        self.api_key = (api_key or FUGLE_API_KEY).strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 stock-dashboard/3.0"})
        if self.api_key:
            self.session.headers.update({"X-API-KEY": self.api_key})

    def _get(self, path: str, params: dict[str, Any] | None = None, retry: int = 3) -> dict:
        if not self.api_key:
            raise APIError("FUGLE_API_KEY 尚未設定。請在目前 PowerShell 視窗設定 $env:FUGLE_API_KEY。")
        url = f"{BASE}/{path.lstrip('/')}"
        last = None
        for i in range(retry):
            try:
                r = self.session.get(url, params=params or {}, timeout=self.timeout)
                if r.status_code == 429:
                    last = f"429 Rate Limit: {r.text[:300]}"
                    time.sleep(2 ** i)
                    continue
                if r.status_code in (401, 403):
                    raise APIError(f"Fugle 認證失敗 HTTP {r.status_code}：{r.text[:500]}")
                if not r.ok:
                    raise APIError(f"Fugle HTTP {r.status_code}：{r.text[:800]}")
                data = r.json()
                if not isinstance(data, dict):
                    raise APIError("Fugle 回傳不是 JSON object")
                return data
            except requests.RequestException as e:
                last = str(e)
                time.sleep(1.5 * (i + 1))
        raise APIError(last or "Fugle API 請求失敗")

    def quote(self, symbol: str) -> dict:
        return self._get(f"intraday/quote/{clean_symbol(symbol)}")

    def ticker(self, symbol: str) -> dict:
        return self._get(f"intraday/ticker/{clean_symbol(symbol)}")

    def tickers(self, exchange: str) -> list[dict]:
        p = {"type": "EQUITY", "exchange": exchange, "isNormal": "true"}
        data = self._get("intraday/tickers", p)
        rows = data.get("data", []) or []
        for row in rows:
            code = clean_symbol(row.get("symbol", ""))
            row["symbol"] = code
            row["industry"] = INDUSTRY_OVERRIDES.get(code, normalize_industry(row.get("industry", "00")))
        return rows

    def snapshot_quotes(self, market: str) -> list[dict]:
        data = self._get(f"snapshot/quotes/{market}", {"type": "ALLBUT0999"})
        return data.get("data", []) or []

    def intraday_candles(self, symbol: str, timeframe: str = "1") -> pd.DataFrame:
        """取得當日盤中 K 線。

        timeframe：1 / 3 / 5 / 10 / 15 / 30 / 60。
        """
        symbol = clean_symbol(symbol)
        timeframe = str(timeframe)
        if timeframe not in {"1", "3", "5", "10", "15", "30", "60"}:
            raise ValueError("intraday timeframe 必須是 1/3/5/10/15/30/60")
        data = self._get(f"intraday/candles/{symbol}", {"timeframe": timeframe, "sort": "asc"})
        rows = data.get("data", []) or []
        return self._candles_df(rows, symbol)

    def historical_candles(self, symbol: str, start: date, end: date, timeframe: str = "D", adjusted: bool = False) -> pd.DataFrame:
        params = {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "timeframe": timeframe,
            "fields": "open,high,low,close,volume,average,change",
            "sort": "asc",
        }
        if timeframe in {"D", "W", "M"}:
            params["adjusted"] = "true" if adjusted else "false"
        data = self._get(f"historical/candles/{clean_symbol(symbol)}", params)
        return self._candles_df(data.get("data", []) or [], clean_symbol(symbol))

    def history_3y(self, symbol: str, cache_dir: Path, years: int = 3) -> pd.DataFrame:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{clean_symbol(symbol)}.csv"
        end = pd.Timestamp.today().normalize().date()
        start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=365 * years + 10)).date()
        parts = []
        cur = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        while cur <= end_ts:
            chunk_end = min(cur + pd.Timedelta(days=329), end_ts)
            part = self.historical_candles(clean_symbol(symbol), cur.date(), chunk_end.date(), "D", adjusted=False)
            if not part.empty:
                parts.append(part)
            cur = chunk_end + pd.Timedelta(days=1)
        if not parts:
            raise APIError(f"{symbol} 沒有歷史日 K 資料")
        out = pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return out

    @staticmethod
    def _candles_df(rows: list[dict], symbol: str) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "average", "change", "stock_code"])
        for c in ["open", "high", "low", "close", "volume", "average", "change"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date"] = pd.NaT
        df["stock_code"] = clean_symbol(symbol)
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df


def clean_symbol(s: str) -> str:
    s = str(s).strip().upper()
    if "." in s:
        s = s.split(".")[0]
    return s


def _num(s: Any) -> float:
    try:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return np.nan
        text = str(s).strip().replace(",", "").replace(" ", "").replace("%", "")
        if text in {"", "-", "--", "—"}:
            return np.nan
        return float(text)
    except Exception:
        return np.nan


def _extract_json_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    return []


def _find_numeric_by_exact_or_suffix(record: dict, names: list[str]) -> float:
    """依官方欄位完整名稱優先，找不到時再用模糊字串 fallback。"""
    for name in names:
        if name in record:
            return _num(record[name])
    lowered = {str(k).strip().lower(): v for k, v in record.items()}
    for name in names:
        needle = name.strip().lower()
        for key, value in lowered.items():
            if needle == key or needle in key:
                val = _num(value)
                if np.isfinite(val):
                    return val
    return np.nan


def _tpex_record_from_json(row: dict) -> dict | None:
    code = clean_symbol(row.get("SecuritiesCompanyCode", ""))
    if not code:
        return None
    raw_date = str(row.get("Date", "")).strip()
    if raw_date.isdigit() and len(raw_date) == 7:
        ad_date = f"{int(raw_date[:3]) + 1911}{raw_date[3:]}"
        ad_date = pd.to_datetime(ad_date, format="%Y%m%d", errors="coerce")
    else:
        ad_date = pd.to_datetime(raw_date, errors="coerce")
    foreign = _find_numeric_by_exact_or_suffix(row, [
        "Foreign Investors include Mainland Area Investors (Foreign Dealers excluded)-Difference",
        "Foreign Investors include Mainland Area Investors (Foreign Dealers excluded) - Difference",
    ])
    trust = _find_numeric_by_exact_or_suffix(row, ["SecuritiesInvestmentTrustCompanies-Difference"])
    dealer = _find_numeric_by_exact_or_suffix(row, ["Dealers-Difference"])
    total = _find_numeric_by_exact_or_suffix(row, ["TotalDifference"])
    if not np.isfinite(foreign) or not np.isfinite(trust) or not np.isfinite(dealer):
        return None
    return {
        "date": ad_date,
        "symbol": code,
        "foreign_net": foreign,
        "trust_net": trust,
        "dealer_net": dealer,
        "total_net": total if np.isfinite(total) else foreign + trust + dealer,
        "market": "上櫃",
        "source": "TPEx-OpenAPI",
    }


def _tpex_legacy_download(day: date, symbol: str) -> dict | None:
    """TPEx 歷史三大法人日報。

    日期格式採民國年，例如 2026-09-09 -> 115/09/09。
    回傳為文字 CSV/HTML 時，透過 pandas 嘗試解析表格。
    """
    roc = f"{day.year - 1911:03d}/{day.month:02d}/{day.day:02d}"
    params = {"l": "zh-tw", "t": "D", "d": roc, "s": "0,asc,0"}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge.php",
    }
    try:
        r = requests.get(TPEX_3INSTI_DOWNLOAD, params=params, headers=headers, timeout=20)
        if not r.ok or not r.text:
            return None
        text = r.text
        # 若伺服器返回 CSV，先直接處理；若返回 HTML，從 table 讀取。
        tables = []
        if "<table" in text.lower():
            try:
                tables = pd.read_html(text)
            except Exception:
                tables = []
        else:
            try:
                import io
                for enc in ["utf-8-sig", "big5", "utf-8"]:
                    try:
                        tables = [pd.read_csv(io.StringIO(text), encoding=enc)]
                        break
                    except Exception:
                        continue
            except Exception:
                tables = []

        if not tables:
            return None

        # 選擇包含「證券代號」或「代號」的表格。
        table = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("代號" in c or "SecuritiesCompanyCode" in c for c in cols):
                table = t.copy()
                break
        if table is None:
            # 常見情況：第一張表就是主資料。
            table = tables[-1].copy()

        table.columns = [str(c).strip() for c in table.columns]
        code_col = next((c for c in table.columns if "代號" in c or "SecuritiesCompanyCode" in c), None)
        if code_col is None:
            return None
        table[code_col] = table[code_col].astype(str).str.extract(r"(\w+)")[0].str.strip()
        row = table[table[code_col].map(clean_symbol) == clean_symbol(symbol)]
        if row.empty:
            return None
        row = row.iloc[0].to_dict()

        def pick_contains(*parts: str) -> float:
            for k, v in row.items():
                kk = str(k).replace(" ", "")
                if all(p.replace(" ", "") in kk for p in parts):
                    val = _num(v)
                    if np.isfinite(val):
                        return val
            return np.nan

        foreign = pick_contains("外資及陸資", "淨買股數")
        if not np.isfinite(foreign):
            foreign = pick_contains("外資", "淨買股數")
        trust = pick_contains("投信", "淨買股數")
        dealer = pick_contains("自營商", "淨買股數")
        total = pick_contains("三大法人", "買賣超股數")
        if not np.isfinite(foreign) or not np.isfinite(trust) or not np.isfinite(dealer):
            return None
        return {
            "date": pd.Timestamp(day),
            "symbol": clean_symbol(symbol),
            "foreign_net": foreign,
            "trust_net": trust,
            "dealer_net": dealer,
            "total_net": total if np.isfinite(total) else foreign + trust + dealer,
            "market": "上櫃",
            "source": "TPEx-歷史日報",
        }
    except Exception:
        return None


def fetch_institution_daily(symbol: str, d: date | None = None, market: str | None = None) -> pd.DataFrame:
    """取得指定股票最近 10 個交易日三大法人買賣超。

    上市：TWSE T86，可查歷史日。
    上櫃：TPEx OpenAPI 提供最新日；歷史 10 日優先使用 TPEx 日報下載端點，失敗才退回 OpenAPI 最新日。
    """
    symbol = clean_symbol(symbol)
    d = d or pd.Timestamp.today().date()
    market_u = str(market or "").upper()
    frames: list[dict] = []

    # -------------------- 上市 TWSE --------------------
    if market_u in {"TSE", "TWSE", "上市"} or not market_u:
        for offset in range(0, 22):
            day = d - timedelta(days=offset)
            try:
                url = TWSE_T86
                params = {"date": day.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"}
                r = requests.get(url, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                if not r.ok:
                    continue
                js = r.json()
                data = js.get("data", []) if isinstance(js, dict) else []
                fields = js.get("fields", []) if isinstance(js, dict) else []
                if not data or not fields:
                    continue
                try:
                    idx = {name: fields.index(name) for name in fields}
                except Exception:
                    continue
                for row in data:
                    if not row or clean_symbol(row[idx.get("證券代號", 0)]) != symbol:
                        continue
                    def fld(name: str) -> float:
                        pos = idx.get(name)
                        return _num(row[pos]) if pos is not None and pos < len(row) else np.nan
                    # 官方 T86 把外資及陸資（不含外資自營商）與外資自營商分欄。
                    # 為避免低估「外資」，這裡合併兩欄；另外保留投信、自營商及三大法人合計。
                    foreign_core = fld("外陸資買賣超股數(不含外資自營商)")
                    foreign_dealer = fld("外資自營商買賣超股數")
                    foreign = foreign_core + foreign_dealer if np.isfinite(foreign_core) and np.isfinite(foreign_dealer) else foreign_core
                    rec = {
                        "date": pd.Timestamp(day),
                        "symbol": symbol,
                        "foreign_net": foreign,
                        "trust_net": fld("投信買賣超股數"),
                        "dealer_net": fld("自營商買賣超股數"),
                        "total_net": fld("三大法人買賣超股數"),
                        "market": "上市",
                        "source": "TWSE-T86",
                    }
                    if np.isfinite(rec["foreign_net"]) and np.isfinite(rec["trust_net"]) and np.isfinite(rec["dealer_net"]):
                        frames.append(rec)
                    break
            except Exception:
                continue
            if len(frames) >= 10:
                break

    # -------------------- 上櫃 TPEx --------------------
    if market_u in {"OTC", "TPEX", "TPEx", "上櫃"} or (not market_u and len(frames) < 10):
        # 先用可查歷史日期的日報端點，最多找 10 個交易日。
        for offset in range(0, 22):
            day = d - timedelta(days=offset)
            rec = _tpex_legacy_download(day, symbol)
            if rec is not None:
                frames.append(rec)
            if len(frames) >= 10:
                break

        # 若歷史日報被站方擋下，至少保留 OpenAPI 最新日，不捏造歷史。
        if not frames:
            try:
                r = requests.get(TPEX_3INSTI, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                if r.ok:
                    for raw in _extract_json_rows(r.json()):
                        rec = _tpex_record_from_json(raw)
                        if rec and rec["symbol"] == symbol:
                            frames.append(rec)
                            break
            except Exception:
                pass

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "foreign_net", "trust_net", "dealer_net", "total_net", "market", "source"])

    out = pd.DataFrame(frames)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    return out.tail(10).reset_index(drop=True)


TDCC_URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
TDCC_CACHE_DIR = Path(__file__).resolve().parent / "data" / "tdcc"
TDCC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TDCC_CACHE_CSV = TDCC_CACHE_DIR / "TDCC_OD_1-5.csv"
TDCC_META = TDCC_CACHE_DIR / "tdcc_meta.txt"
TDCC_CACHE_MAX_AGE_HOURS = 24


def _download_tdcc_csv() -> pd.DataFrame:
    """從 TDCC 官方 Open Data 下載股權分散表（1-5），並保存本地快取。

    官方目前提供的開放資料 URL 為 opendata.tdcc.com.tw/getOD.ashx?id=1-5；
    該資料是每週最後一個營業日收盤後的股權分散資料。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/octet-stream,text/plain,*/*",
        "Referer": "https://www.tdcc.com.tw/portal/zh/smWeb/qryStock",
    }
    last_error = None
    for timeout in (15, 30):
        try:
            r = requests.get(TDCC_URL, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            content = r.content
            if not content:
                raise RuntimeError("TDCC 回傳空內容")
            text = content.decode("utf-8-sig", errors="ignore")
            if "資料日期" not in text[:500]:
                text = content.decode("cp950", errors="ignore")
            from io import StringIO
            df = pd.read_csv(StringIO(text), dtype=str)
            if df.empty:
                raise RuntimeError("TDCC CSV 沒有資料")
            TDCC_CACHE_CSV.write_bytes(content)
            TDCC_META.write_text(pd.Timestamp.now(tz="Asia/Taipei").isoformat(), encoding="utf-8")
            return df
        except Exception as e:
            last_error = e
    raise APIError(f"TDCC Open Data 下載失敗：{last_error}")


def _read_tdcc_cache() -> pd.DataFrame:
    if not TDCC_CACHE_CSV.exists():
        return pd.DataFrame()
    try:
        from io import StringIO
        content = TDCC_CACHE_CSV.read_bytes()
        text = content.decode("utf-8-sig", errors="ignore")
        if "資料日期" not in text[:500]:
            text = content.decode("cp950", errors="ignore")
        return pd.read_csv(StringIO(text), dtype=str)
    except Exception:
        return pd.DataFrame()


def fetch_tdcc_big_retail(symbol: str) -> pd.DataFrame:
    """取得 TDCC 最新一週股權分散資料；優先使用本地快取，避免頁面每 2 秒重抓 TDCC。"""
    symbol = clean_symbol(symbol)
    now = pd.Timestamp.now(tz="Asia/Taipei")
    df = _read_tdcc_cache()

    stale = True
    if not df.empty and TDCC_META.exists():
        try:
            ts = pd.Timestamp(TDCC_META.read_text(encoding="utf-8").strip())
            if ts.tzinfo is None:
                ts = ts.tz_localize("Asia/Taipei")
            stale = (now - ts) >= pd.Timedelta(hours=TDCC_CACHE_MAX_AGE_HOURS)
        except Exception:
            stale = True

    if df.empty or stale:
        try:
            df = _download_tdcc_csv()
        except Exception:
            # TDCC 暫時無法連線時，沿用舊快取；絕不因週資料來源短暫 timeout 讓整個 Dashboard 失敗。
            df = _read_tdcc_cache()
            if df.empty:
                raise

    aliases = {
        "Date":"資料日期",
        "Securities Code":"證券代號",
        "Securities Holding Range":"持股分級",
        "Number of Holders":"人數",
        "Number of Shares/Units":"股數",
        "Percentage of Centrally Deposited Securities":"占集保庫存數比例%",
    }
    df = df.rename(columns=aliases)
    needed = ["資料日期","證券代號","持股分級","人數","股數","占集保庫存數比例%"]
    if not all(c in df.columns for c in needed):
        raise APIError("TDCC 欄位格式無法辨識")

    df["證券代號"] = df["證券代號"].astype(str).str.strip().str.replace(".0", "", regex=False)
    df = df[df["證券代號"].map(clean_symbol) == symbol].copy()
    if df.empty:
        return pd.DataFrame()

    df["level"] = pd.to_numeric(df["持股分級"], errors="coerce")
    df["shares"] = df["股數"].map(_num)
    df["holders"] = df["人數"].map(_num)
    df["proportion"] = df["占集保庫存數比例%"].map(_num)
    df["date"] = pd.to_datetime(df["資料日期"].astype(str).str.replace("/", "", regex=False), errors="coerce")
    latest = df["date"].max()
    df = df[df["date"] == latest]

    # TDCC 1-5 的 17 個持股級距：1~8 為 50 張以下，12~15 為 400 張以上；16、17 為統計彙總級距。
    retail = df[df.level.between(1, 8)]["shares"].sum(min_count=1)
    big = df[df.level.between(12, 15)]["shares"].sum(min_count=1)
    total = df[df.level == 17]["shares"].sum(min_count=1)
    if not np.isfinite(total) or total <= 0:
        total = df[df.level < 16]["shares"].sum(min_count=1)
    if not all(np.isfinite(v) for v in [retail, big, total]) or total <= 0:
        return pd.DataFrame()

    big_pct = big / total * 100
    retail_pct = retail / total * 100
    return pd.DataFrame([{
        "date": latest,
        "symbol": symbol,
        "big_shares": big,
        "retail_shares": retail,
        "total_shares": total,
        "big_pct": big_pct,
        "retail_pct": retail_pct,
        "middle_pct": max(0, 100 - big_pct - retail_pct),
        "definition": "大戶>=400張；散戶<=50張",
        "source": "TDCC Open Data",
    }])


def fetch_taiex_intraday() -> pd.DataFrame:
    """取得 TWSE 加權指數；盤中取最近 5 秒值，非交易時段回退最近交易日。

    成功資料會快取到 data/taiex_last.csv，避免週末／盤後因 TWSE API 暫時
    不可用而讓右上 TAIEX 空白。
    """
    cache_path = Path(__file__).resolve().parent / "data" / "taiex_last.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://www.twse.com.tw/exchangeReport/MI_5MINS_INDEX"
    tz = "Asia/Taipei"
    now = pd.Timestamp.now(tz=tz)
    dates = [now.date()] + [(now - pd.Timedelta(days=i)).date() for i in range(1, 11)]
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.twse.com.tw/",
    }

    for day in dates:
        try:
            r = requests.get(url, params={"response": "json", "date": day.strftime("%Y%m%d")},
                             headers=headers, timeout=10)
            if not r.ok:
                continue
            js = r.json()
            rows = js.get("data", []) if isinstance(js, dict) else []
            vals = []
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                t = str(row[0]).strip()
                v = _num(row[1])
                if np.isfinite(v):
                    vals.append((t, v))
            if not vals:
                continue
            df = pd.DataFrame(vals, columns=["time", "index"])
            ref = float(df.iloc[0]["index"])
            if len(df) > 1:
                intraday = df[df["time"].astype(str) != "09:00:00"].copy()
                if not intraday.empty:
                    df = intraday
            df["change"] = df["index"] - ref
            df["change_pct"] = df["change"] / ref * 100 if ref else np.nan
            df["datetime"] = pd.to_datetime(day.strftime("%Y-%m-%d") + " " + df["time"].astype(str), errors="coerce")
            df["datetime"] = df["datetime"].dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
            df = df.dropna(subset=["datetime", "index"]).sort_values("datetime").reset_index(drop=True)
            if not df.empty:
                df.to_csv(cache_path, index=False, encoding="utf-8-sig")
                return df
        except Exception:
            continue

    # 若 TWSE 當下無法連線，直接回傳最後一次成功資料。
    try:
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime", "index"]).sort_values("datetime").reset_index(drop=True)
            if not df.empty:
                return df
    except Exception:
        pass
    return pd.DataFrame(columns=["time", "index", "change", "change_pct", "datetime"])

