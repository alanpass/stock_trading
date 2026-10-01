# -*- coding: utf-8 -*-
"""AI 台股即時互動式分析系統 v57

重點：
1. 自選股上漲整列紅字、下跌整列綠字。
2. 右上固定即時卡：選取股票 + 即時價格 + TAIEX；盤中 2 秒更新。
3. 盤中即時資料以 Streamlit browser-side rerun 自動刷新，預設每 5 秒更新；盤後嚴格驗證今日行情日期。
4. 技術 K 線：日K / 當日 5 / 30 / 60 分K。
4. 當日逢低進場預測；主要使用 5 分 K 學習 K 線特徵，輔助量能、VWAP、日內位置與前一交易日資料。
5. 大戶 VS 散戶持股比例。
6. 交易明細：台灣時間；買進價紅字、賣出價綠字。
7. 三大法人 10 交易日買賣超：正紅、負綠。
8. 法人說明會：抓取 MOPS 法說資料、判斷偏利多/偏利空，並比較法說後 1/5 交易日股價反應。
9. 近期產業漲跌：全市場漲跌候選、5/20日報酬、產業聚合與利多/利空原因。
10. AI：Regression + Classification 整合成單一最終方向與可信度，不要求使用者自行選擇。
11. 所有即時/選取股票資料盤中預設每 5 秒更新；AI 模型本身不隨即時刷新重訓。
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh


def _load_cloud_secret_env() -> None:
    """將 Streamlit Community Cloud Secrets 同步成環境變數。

    本機仍可使用原本的 stock_api.py；部署到 Cloud 時，
    可在 App Settings -> Secrets 設定 FUGLE_API_KEY。
    """
    try:
        key = st.secrets.get("FUGLE_API_KEY")
        if key:
            os.environ["FUGLE_API_KEY"] = str(key)
    except Exception:
        pass

_load_cloud_secret_env()

from market_calendar import is_twse_trading_day, market_status_text
from stock_api import (
    FugleClient, clean_symbol, INDUSTRIES,
    fetch_institution_daily, fetch_tdcc_big_retail, fetch_taiex_intraday,
)
from industry_master import build_master, apply_overrides, INDUSTRY_OVERRIDES
from business_master import business_map_for_symbols, profile_for_symbol
from ai_engine import (add_features, train, save_model_bundle, load_model_bundle,
                         bundle_is_usable, predict_from_bundle)
from entry_model import (load_training_intraday, fit_model, predict_entry,
                         load_saved_model, save_saved_model)
from market_agent import AgentContext, MarketSupervisorAgent
from local_ollama_agent import LocalOllamaAgent, ollama_available
from research_agent import ResearchAgent
from market_intelligence import render_market_intelligence
from theme_agent import theme_sankey_frame
from post_market_analysis import PostMarketAnalyzer
from email_agent import EmailAgent

BASE = Path(__file__).resolve().parent
DATA, MODELS, OUTPUT = BASE / "data", BASE / "models", BASE / "output"
DAILY_DATA = DATA / "daily"
for d in (DATA, MODELS, OUTPUT, DAILY_DATA):
    d.mkdir(exist_ok=True)

WATCHLIST_DEFAULT = [
    "3481", "2327", "2492", "3037", "3044", "3533", "2303", "2330", "2344", "2408",
    "6515", "7769", "6488", "3374", "2377", "3450", "0050", "3006"
]
# 啟動時若尚未建立全市場產業主檔，仍可先正確顯示預設自選股的分類；
# 完整上市櫃產業主檔只在搜尋／AI 等需要時再載入。
WATCHLIST_INDUSTRY_FALLBACK = {
    "3481": "26", "2327": "28", "2492": "28", "3037": "28", "3044": "28",
    "3533": "28", "2303": "24", "2330": "24", "2344": "24", "2408": "24",
    "6515": "24", "7769": "24", "6488": "24", "3374": "24", "2377": "25",
    "3450": "24", "0050": "00", "3006": "24",
}
WATCHLIST_FILE = BASE / "user_watchlist.json"

def load_watchlist():
    """讀取使用者上次加入的自選股；若檔案不存在則使用預設清單。"""
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                cleaned = [clean_symbol(x) for x in data if clean_symbol(x)]
                if cleaned:
                    return list(dict.fromkeys(cleaned))
    except Exception:
        pass
    return WATCHLIST_DEFAULT.copy()

def save_watchlist(items):
    """將自選股保存到本機，下一次開啟 Dashboard 仍會保留。"""
    try:
        WATCHLIST_FILE.write_text(json.dumps(list(dict.fromkeys(items)), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass



st.set_page_config(page_title="AI 台股即時互動式分析系統", layout="wide", initial_sidebar_state="expanded")

# 手動立即更新：只清除即時資料相關快取，不影響慢資料與模型快取。
# 這對 Cloud 使用者特別有用：若瀏覽器剛喚醒 App，可以直接取得最新行情。

def _manual_live_refresh() -> None:
    try:
        watch_rows.clear()
        get_quote.clear()
        get_intraday.clear()
        get_trades.clear()
        get_taiex.clear()
    except Exception:
        # 這些函式尚未宣告時不應阻止 App 啟動；按鈕僅在它們完成定義後才會正常工作。
        pass
    st.session_state["manual_refresh_at"] = taiwan_now().strftime("%Y-%m-%d %H:%M:%S") if "TW_TZ" in globals() else ""

@st.cache_resource(show_spinner=False)
def get_client():
    return FugleClient()

client = get_client()
TW_TZ = ZoneInfo("Asia/Taipei")
AGENT_LOG_DIR = OUTPUT / "agent_logs"
AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
AGENT_SUPERVISOR = MarketSupervisorAgent(AGENT_LOG_DIR)

# 台股集中市場正常盤中：09:00～13:30。
# 只有在盤中才啟用每 2 秒自動刷新；盤前、盤後與週末不自動輪詢。
def is_taiwan_cash_session_open() -> bool:
    now = datetime.now(TW_TZ)
    if not is_twse_trading_day(now, BASE):
        return False
    t = now.time()
    return dt_time(9, 0) <= t <= dt_time(13, 30)

# 即時行情刷新：預設每 5 秒一次。
# 2 秒雖然更即時，但對 Streamlit Cloud + Fugle API 會造成較高的重跑與請求壓力，
# 因此改為 5 秒，並可用 LIVE_REFRESH_SECONDS 自訂。
try:
    LIVE_REFRESH_SECONDS = max(2, int(os.getenv("LIVE_REFRESH_SECONDS", "5")))
except Exception:
    LIVE_REFRESH_SECONDS = 5

REFRESH_INTERVAL = LIVE_REFRESH_SECONDS if is_taiwan_cash_session_open() else None
MARKET_OPEN = dt_time(9, 0)
MARKET_CLOSE = dt_time(13, 30)

def taiwan_now():
    return pd.Timestamp.now(tz=TW_TZ)

def is_market_open_now():
    now = taiwan_now()
    return is_twse_trading_day(now, BASE) and MARKET_OPEN <= now.time() <= MARKET_CLOSE

def is_before_open_or_stale(data_date=None):
    """今日尚未開盤，或 API 回傳日期不是今天時，使用最近一個有效交易日。"""
    now = taiwan_now()
    today = now.date()
    if is_twse_trading_day(now, BASE) and now.time() < MARKET_OPEN:
        return True
    if data_date is not None:
        try:
            d = pd.Timestamp(data_date).date()
            if d != today and now.time() < MARKET_OPEN:
                return True
        except Exception:
            pass
    return False

def latest_trading_date(code):
    """取得今天以前最近一個有效交易日，供開盤前顯示。"""
    now = taiwan_now().tz_localize(None).normalize()
    end = (now - pd.Timedelta(days=1)).date()
    start = (now - pd.Timedelta(days=10)).date()
    try:
        h = client.historical_candles(clean_symbol(code), start, end, "D")
        if not h.empty:
            h["date"] = pd.to_datetime(h["date"], errors="coerce")
            h = h.dropna(subset=["date"]).sort_values("date")
            if not h.empty:
                return h.iloc[-1]["date"].date()
    except Exception:
        pass
    return None

def previous_day_daily_quote(code):
    """開盤前以最近交易日的日K作為目前價格顯示資料。"""
    code = clean_symbol(code)
    try:
        end = (taiwan_now().tz_localize(None).normalize() - pd.Timedelta(days=1)).date()
        start = (taiwan_now().tz_localize(None).normalize() - pd.Timedelta(days=10)).date()
        h = client.historical_candles(code, start, end, "D")
        if h.empty:
            return {}
        h = h.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        h = h.dropna(subset=["date"]).sort_values("date")
        if h.empty:
            return {}
        last = h.iloc[-1]
        prev = h.iloc[-2] if len(h) >= 2 else None
        close = float(last["close"])
        prev_close = float(prev["close"]) if prev is not None else np.nan
        change = float(last["change"]) if pd.notna(last.get("change")) else (close-prev_close if pd.notna(prev_close) else np.nan)
        cp = change / prev_close * 100 if pd.notna(change) and pd.notna(prev_close) and prev_close != 0 else np.nan
        name = code
        try:
            tk = client.ticker(code)
            name = tk.get("name") or name
        except Exception:
            pass
        return {
            "date": last["date"].date().isoformat(),
            "symbol": code, "name": name,
            "closePrice": close, "tradePrice": close, "lastPrice": close,
            "previousClose": prev_close, "referencePrice": prev_close,
            "change": change, "changePercent": cp,
            "openPrice": last.get("open", np.nan), "highPrice": last.get("high", np.nan), "lowPrice": last.get("low", np.nan),
            "limitUpPrice": np.nan, "limitDownPrice": np.nan,
            "is_previous_trading_day": True,
            "previous_trading_date": last["date"].date().isoformat(),
        }
    except Exception:
        return {}

if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if "selected" not in st.session_state:
    st.session_state.selected = st.session_state.watchlist[0]


def persist_selected_symbol():
    try:
        (DATA / "last_selected_symbol.json").write_text(
            json.dumps({"symbol": clean_symbol(st.session_state.selected), "saved_at": taiwan_now().isoformat()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

persist_selected_symbol()
if "ai_busy" not in st.session_state:
    st.session_state.ai_busy = False

if "search_result" not in st.session_state:
    st.session_state.search_result = None
if "ai_selected" not in st.session_state:
    st.session_state.ai_selected = None
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "ollama_agent_result" not in st.session_state:
    st.session_state.ollama_agent_result = None
if "ai_reused" not in st.session_state:
    st.session_state.ai_reused = False
if "ai_trained_at" not in st.session_state:
    st.session_state.ai_trained_at = None
if "ai_training_data_date" not in st.session_state:
    st.session_state.ai_training_data_date = None
if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "research_busy" not in st.session_state:
    st.session_state.research_busy = False

# ============================================================
# Data layer - 即時資料 2 秒；慢資料分開快取
# ============================================================
@st.cache_data(ttl=2, show_spinner=False)
def watch_rows(watchlist_key):
    rows, errors, maps = [], [], {}
    market_open = is_market_open_now()
    now = taiwan_now()
    today = now.date()
    trading_day = is_twse_trading_day(now, BASE)
    after_close = trading_day and now.time() > MARKET_CLOSE

    # 盤中：用全市場 snapshot，一次取得自選股最新行情，避免每 5 秒逐檔打 API。
    # 重要：若 snapshot 明確帶有日期，日期不是今天就丟掉，避免把舊快照當成即時資料。
    if market_open:
        for market in ["TSE", "OTC"]:
            try:
                for x in client.snapshot_quotes(market):
                    code = clean_symbol(x.get("symbol", ""))
                    if not code:
                        continue
                    raw_date = x.get("date") or x.get("dataDate") or x.get("asOfDate")
                    if raw_date:
                        d = pd.to_datetime(raw_date, errors="coerce")
                        if pd.notna(d) and d.date() != today:
                            continue
                    maps[code] = x
            except Exception as e:
                errors.append(f"snapshot/{market}: {e}")

    watchlist = list(watchlist_key)
    for code in watchlist:
        # 盤後是本次問題的關鍵：不要使用可能停留在前一交易日的 snapshot。
        # 直接走「今日 Quote -> 今日分K -> 今日日K」的嚴格日期驗證流程。
        if after_close:
            x = get_quote(code) or {}
        elif not trading_day or (trading_day and now.time() < MARKET_OPEN):
            # 盤前／週末／休市日只顯示最近完成交易日，不拿舊 snapshot 冒充即時行情。
            x = previous_day_daily_quote(code)
        else:
            x = maps.get(code, {})
            if not x:
                try:
                    x = client.quote(code)
                except Exception as e:
                    errors.append(f"{code}: {e}")

        price_candidates = [x.get("lastPrice"), x.get("closePrice"), x.get("tradePrice"), x.get("avgPrice")]
        p = np.nan
        for candidate in price_candidates:
            value = pd.to_numeric(candidate, errors="coerce")
            if pd.notna(value):
                p = value
                break

        prev_candidates = [x.get("referencePrice"), x.get("previousClose")]
        prev = np.nan
        for candidate in prev_candidates:
            value = pd.to_numeric(candidate, errors="coerce")
            if pd.notna(value):
                prev = value
                break

        ch = pd.to_numeric(x.get("change", np.nan), errors="coerce")
        cp = pd.to_numeric(x.get("changePercent", np.nan), errors="coerce")
        if pd.isna(ch) and pd.notna(p) and pd.notna(prev):
            ch = p - prev
        if pd.isna(cp) and pd.notna(ch) and pd.notna(prev) and float(prev) != 0:
            cp = ch / prev * 100

        name = x.get("name") or x.get("stockName") or x.get("securityName") or code
        limit_up = pd.to_numeric(x.get("limitUpPrice", x.get("limitUp", np.nan)), errors="coerce")
        limit_down = pd.to_numeric(x.get("limitDownPrice", x.get("limitDown", np.nan)), errors="coerce")
        row_date = x.get("date") or x.get("dataDate") or x.get("asOfDate")
        try:
            row_date = pd.to_datetime(row_date, errors="coerce").date().isoformat() if pd.notna(pd.to_datetime(row_date, errors="coerce")) else ""
        except Exception:
            row_date = ""
        rows.append({
            "code": code, "name": name, "price": p, "change": ch, "change_pct": cp,
            "limit_up": limit_up, "limit_down": limit_down, "market": x.get("market", ""),
            "data_date": row_date,
            "data_source": "Fugle Quote／今日收盤" if after_close else ("Fugle Snapshot／盤中" if market_open else "Fugle 歷史日K／最近交易日"),
            "industry": str(x.get("industry", WATCHLIST_INDUSTRY_FALLBACK.get(code, "00"))).zfill(2),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        result["industry"] = pd.Series(dtype=str)
        return result, errors

    # 只讀本機產業主檔；若不存在，不在首頁啟動時阻塞建立全市場主檔。
    industry_map = {}
    cache_path = DATA / "industry_master.csv"
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, dtype={"symbol": str, "industry": str})
            cached["symbol"] = cached["symbol"].astype(str).str.strip().map(clean_symbol)
            cached["industry"] = cached["industry"].map(lambda v: str(v).zfill(2))
            industry_map = cached.set_index("symbol")["industry"].to_dict()
        except Exception:
            industry_map = {}

    result["industry"] = result.apply(
        lambda r: INDUSTRY_OVERRIDES.get(
            clean_symbol(r["code"]),
            industry_map.get(clean_symbol(r["code"]), WATCHLIST_INDUSTRY_FALLBACK.get(clean_symbol(r["code"]), str(r.get("industry", "00")).zfill(2)))
        ), axis=1
    )

    # 第二層：公司主要業務／產業鏈子分類。
    profiles = get_business_profiles_cached(tuple(result["code"].tolist()))
    result["business_group"] = result["code"].map(
        lambda c: str((profiles.get(clean_symbol(c)) or {}).get("business_group", "主要業務待建立"))
    )
    result["primary_chain"] = result["code"].map(
        lambda c: str((profiles.get(clean_symbol(c)) or {}).get("primary_chain", ""))
    )
    return result, errors

@st.cache_data(ttl=21600, show_spinner=False)
def get_all_tickers() -> pd.DataFrame:
    # 6 小時更新一次全市場產業主檔；盤中 2 秒刷新不會反覆查詢全市場。
    cache_path = DATA / "industry_master.csv"
    try:
        df = build_master(client, cache_path, force=False)
        if not df.empty:
            return apply_overrides(df)
    except Exception:
        pass

    items = []
    for ex in ["TWSE", "TPEx"]:
        try:
            items.extend(client.tickers(ex))
        except Exception:
            pass
    rows = []
    seen = set()
    for x in items:
        code = clean_symbol(x.get("symbol", ""))
        if not code or code in seen:
            continue
        seen.add(code)
        rows.append({
            "symbol": code,
            "name": x.get("name", ""),
            "industry": INDUSTRY_OVERRIDES.get(code, str(x.get("industry", "00")).zfill(2)),
            "exchange": x.get("exchange", ""),
        })
    return apply_overrides(pd.DataFrame(rows))



@st.cache_data(ttl=21600, show_spinner=False)
def get_business_profiles_cached(symbols: tuple[str, ...]) -> dict:
    """讀取／補齊公司主要業務與產業鏈子分類；只對 watchlist/目前研究標的做需要的查詢。"""
    path = DATA / "business_master.csv"
    try:
        return business_map_for_symbols(symbols, client=client, path=path, force=False)
    except Exception:
        return {}


def search_stock(keyword: str) -> pd.DataFrame:
    q = str(keyword or "").strip().upper()
    if not q:
        return pd.DataFrame()
    df = get_all_tickers()
    if df.empty:
        return df
    code_s = df["symbol"].astype(str).str.upper()
    name_s = df["name"].astype(str).str.upper()
    exact = df[(code_s == q) | (name_s == q)]
    if not exact.empty:
        return exact.head(10)
    matched = df[code_s.str.contains(q, regex=False, na=False) | name_s.str.contains(q, regex=False, na=False)]
    return matched.head(20)


@st.cache_data(ttl=2, show_spinner=False)
def get_quote(code: str) -> dict:
    """取得最新且日期可驗證的行情。

    規則：
    - 盤中：使用 Fugle 即時 Quote。
    - 交易日盤後：只接受「今天」資料；依序使用 Quote、今日分K、今日日K。
    - 盤前／非交易日：使用最近一個已完成交易日。
    - 盤後若今天資料尚未取得，寧可回傳空值，也不把昨天價格冒充成今天收盤。
    """
    code = clean_symbol(code)
    now = taiwan_now()
    today = now.date()
    trading_day = is_twse_trading_day(now, BASE)
    after_close = trading_day and now.time() > MARKET_CLOSE

    # ------------------------------------------------------------
    # 1. Fugle 即時 Quote：date 必須能確認。
    # Fugle Quote 的 closePrice 是最後成交價，date 是資料日期。
    # ------------------------------------------------------------
    try:
        q = client.quote(code)
        qdate = pd.to_datetime(q.get("date"), errors="coerce") if q.get("date") else pd.NaT
        price_candidates = [q.get("lastPrice"), q.get("closePrice"), q.get("tradePrice")]
        has_price = any(pd.notna(pd.to_numeric(v, errors="coerce")) for v in price_candidates)

        if has_price and pd.notna(qdate):
            qday = qdate.date()
            if trading_day and qday == today:
                return q
            if not trading_day and qday <= today:
                return q

        # 盤中若 API 沒提供 date，仍可讓即時畫面使用 Quote；
        # 但盤後絕對不能接受日期不明的資料。
        if has_price and is_market_open_now() and pd.isna(qdate):
            return q
    except Exception:
        q = {}

    # ------------------------------------------------------------
    # 2. 交易日盤後：只接受今天的 5/1 分鐘K最後一根。
    # ------------------------------------------------------------
    if after_close:
        for tf in ("5", "1"):
            try:
                k = normalize_intraday(client.intraday_candles(code, tf))
                if k.empty or "date" not in k.columns:
                    continue
                k["date"] = pd.to_datetime(k["date"], errors="coerce")
                k = k.dropna(subset=["date"]).sort_values("date")
                if k.empty or k.iloc[-1]["date"].date() != today:
                    continue

                last = k.iloc[-1]
                prev = k.iloc[-2] if len(k) >= 2 else None
                close = float(last["close"])
                prev_close = float(prev["close"]) if prev is not None else np.nan
                change = close - prev_close if np.isfinite(prev_close) else np.nan
                cp = change / prev_close * 100 if np.isfinite(change) and prev_close != 0 else np.nan
                return {
                    "date": str(last["date"].date()),
                    "symbol": code,
                    "name": q.get("name", code) if isinstance(q, dict) else code,
                    "closePrice": close, "tradePrice": close, "lastPrice": close,
                    "previousClose": prev_close, "referencePrice": prev_close,
                    "change": change, "changePercent": cp,
                    "openPrice": float(last["open"]), "highPrice": float(last["high"]), "lowPrice": float(last["low"]),
                }
            except Exception:
                continue

        # 歷史日K同步通常比即時 Quote 慢，所以最後再明確查今天日K。
        try:
            k = client.historical_candles(code, today, today, "D")
            k = k.copy() if isinstance(k, pd.DataFrame) else pd.DataFrame(k)
            if not k.empty and "date" in k.columns:
                k["date"] = pd.to_datetime(k["date"], errors="coerce")
                k = k.dropna(subset=["date"]).sort_values("date")
                if not k.empty and k.iloc[-1]["date"].date() == today:
                    last = k.iloc[-1]
                    close = float(last["close"])
                    prev_close = np.nan
                    try:
                        kh = client.historical_candles(code, today - pd.Timedelta(days=7), today - pd.Timedelta(days=1), "D")
                        kh = kh.copy() if isinstance(kh, pd.DataFrame) else pd.DataFrame(kh)
                        if not kh.empty and "close" in kh.columns:
                            prev_close = float(kh.sort_values("date").iloc[-1]["close"])
                    except Exception:
                        pass
                    change = close - prev_close if np.isfinite(prev_close) else np.nan
                    cp = change / prev_close * 100 if np.isfinite(change) and prev_close != 0 else np.nan
                    return {
                        "date": str(last["date"].date()), "symbol": code,
                        "name": q.get("name", code) if isinstance(q, dict) else code,
                        "closePrice": close, "tradePrice": close, "lastPrice": close,
                        "previousClose": prev_close, "referencePrice": prev_close,
                        "change": change, "changePercent": cp,
                        "openPrice": last.get("open", np.nan),
                        "highPrice": last.get("high", np.nan),
                        "lowPrice": last.get("low", np.nan),
                    }
        except Exception:
            pass

        # 今天資料尚未取得時，禁止回退到昨天；否則首頁又會出現「看起來像今天、實際是昨天」的價格。
        return {}

    # ------------------------------------------------------------
    # 3. 盤前／非交易日：只使用最近完成交易日。
    # ------------------------------------------------------------
    fallback = previous_day_daily_quote(code)
    return fallback if fallback else {}

@st.cache_data(ttl=2, show_spinner=False)
def get_intraday(code: str, timeframe: str) -> pd.DataFrame:
    """取得盤中或最近一個有效交易日的分 K。

    盤中：今日 intraday。
    盤後：優先今日 intraday，再回退今日 historical 5m。
    盤前/週末：最近完成交易日 historical K。
    """
    code = clean_symbol(code)
    now = taiwan_now()

    if is_market_open_now():
        try:
            return normalize_intraday(client.intraday_candles(code, timeframe))
        except Exception:
            return pd.DataFrame()

    # 盤後仍然需要顯示今天完整 K 線，而不是昨天。
    if is_twse_trading_day(now, BASE) and now.time() > MARKET_CLOSE:
        try:
            k = normalize_intraday(client.intraday_candles(code, timeframe))
            if not k.empty and "date" in k.columns:
                k["date"] = pd.to_datetime(k["date"], errors="coerce")
                if pd.notna(k["date"].max()) and k["date"].max().date() == now.date():
                    return k
        except Exception:
            pass
        try:
            today = now.date()
            k = client.historical_candles(code, today, today, str(timeframe))
            k = normalize_intraday(k)
            if not k.empty and "date" in k.columns:
                k["date"] = pd.to_datetime(k["date"], errors="coerce")
                if pd.notna(k["date"].max()) and k["date"].max().date() == today:
                    return k
        except Exception:
            pass

    prev = latest_trading_date(code)
    if prev:
        try:
            k = client.historical_candles(code, prev, prev, str(timeframe))
            k = normalize_intraday(k) if str(timeframe) not in {"D", "W", "M"} else k
            if not k.empty:
                return k
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=2, show_spinner=False)
def get_trades(code: str) -> pd.DataFrame:
    payload = client._get(f"intraday/trades/{clean_symbol(code)}")
    return pd.DataFrame(payload.get("data", []) or [])

@st.cache_data(ttl=5, show_spinner=False)
def get_taiex() -> pd.DataFrame:
    return fetch_taiex_intraday()

@st.cache_data(ttl=60, show_spinner=False)
def get_ticker(code: str) -> dict:
    return client.ticker(code)

@st.cache_data(ttl=60, show_spinner=False)
def get_institutions(code: str, market_name: str) -> pd.DataFrame:
    return fetch_institution_daily(code, market=market_name)

@st.cache_data(ttl=3600, show_spinner=False)
def get_tdcc(code: str) -> pd.DataFrame:
    return fetch_tdcc_big_retail(code)

@st.cache_data(ttl=60, show_spinner=False)
def get_history(code: str, days: int, adjusted: bool = False) -> pd.DataFrame:
    """持續保存日 K 到 data/daily，不再每次重頭下載。"""
    code = clean_symbol(code)
    path = DAILY_DATA / f"{code}.csv"
    end = taiwan_now().tz_localize(None).normalize()
    start_date = end - pd.Timedelta(days=days)
    cached = pd.DataFrame()
    if path.exists():
        try:
            cached = pd.read_csv(path, parse_dates=["date"])
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        except Exception:
            cached = pd.DataFrame()

    parts = []
    if not cached.empty:
        # 保留完整歷史資料；僅從最後快取日前幾天重新抓取，避免漏掉最新更新。
        parts.append(cached)
        latest_cached = cached["date"].max().normalize()
        fetch_start = max(start_date, latest_cached - pd.Timedelta(days=3))
    else:
        fetch_start = start_date

    cur = fetch_start
    while cur <= end:
        ce = min(cur + pd.Timedelta(days=329), end)
        try:
            part = client.historical_candles(code, cur.date(), ce.date(), "D", adjusted=adjusted)
            if not part.empty:
                parts.append(part)
        except Exception:
            # 有舊 cache 時保留舊資料；完全沒有 cache 才在最後報錯。
            pass
        cur = ce + pd.Timedelta(days=1)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out[(out["date"] >= start_date) & (out["date"] <= end + pd.Timedelta(days=1))].copy()


def taipei_time(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    try:
        text = str(value).strip()
        if isinstance(value, (int, np.integer, float, np.floating)) or text.isdigit():
            n = float(value)
            unit = "us" if n > 1e13 else "ms" if n > 1e11 else "s"
            return pd.to_datetime(n, unit=unit, utc=True, errors="coerce").tz_convert("Asia/Taipei").tz_localize(None)
        t = pd.Timestamp(value)
        if t.tzinfo is None:
            return t.tz_localize("Asia/Taipei").tz_localize(None)
        return t.tz_convert("Asia/Taipei").tz_localize(None)
    except Exception:
        return pd.NaT


def normalize_intraday(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["date"].map(taipei_time)
    d = d.dropna(subset=["date"])
    d = d[(d["date"].dt.time >= dt_time(9, 0)) & (d["date"].dt.time <= dt_time(13, 30))]
    return d.sort_values("date").reset_index(drop=True)


def color_for_value(v):
    try:
        x = float(v)
        if x > 0: return "color:#ef4444;font-weight:800"
        if x < 0: return "color:#10b981;font-weight:800"
    except Exception:
        pass
    return "color:#64748b"


st.markdown("""
<style>
.live-card{position:fixed;right:16px;top:14px;width:340px;min-height:96px;padding:9px 10px;background:rgba(15,23,42,.97);border:1px solid rgba(148,163,184,.45);border-radius:12px;box-shadow:0 6px 22px rgba(0,0,0,.30);z-index:999999;font-family:Arial,"Microsoft JhengHei",sans-serif;}
.live-grid{display:grid;grid-template-columns:1fr 1fr;gap:0}.live-cell{padding:0 10px;min-width:0}.live-cell + .live-cell{border-left:1px solid rgba(148,163,184,.25)}
.live-title{font-size:10px;color:#94a3b8;margin-bottom:2px}.live-main{font-size:14px;font-weight:800;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.live-price{font-size:22px;line-height:1.03;font-weight:900;color:#f8fafc;margin-top:2px}.live-change{font-size:12px;font-weight:800;margin-top:3px}.live-time{font-size:9px;color:#64748b;margin-top:3px}
.top-header-safe-space{height:92px;width:100%;}
.title-search-row{position:relative;z-index:2;}

/* ============================================================
   Mobile responsive layout
   Streamlit Cloud / iPhone / Android
   ============================================================ */
@media (max-width: 768px){
  /* 頁面左右留白縮小，避免手機出現水平捲軸。 */
  .main .block-container{
    padding-left:12px !important;
    padding-right:12px !important;
    padding-top:8px !important;
    padding-bottom:24px !important;
    max-width:100% !important;
  }

  /* 右上即時卡改成手機寬度，避免固定 340px 撐破畫面。 */
  .live-card{
    position:relative !important;
    left:auto !important;
    right:auto !important;
    top:auto !important;
    width:100% !important;
    max-width:100% !important;
    min-height:0 !important;
    margin:0 0 10px 0 !important;
    box-sizing:border-box !important;
    z-index:20 !important;
  }

  .top-header-safe-space{height:8px !important;}

  /* 一般頁面區塊在手機改成單欄，避免標題／分析卡片被擠壓。 */
  [data-testid="stHorizontalBlock"]{
    flex-wrap:wrap !important;
    row-gap:10px !important;
    column-gap:0 !important;
    box-sizing:border-box !important;
    min-width:0 !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="column"]{
    flex:1 1 100% !important;
    width:100% !important;
    min-width:0 !important;
    max-width:100% !important;
    box-sizing:border-box !important;
  }

  /*
     ============================================================
     自選股手機版 UX：一筆股票固定「左 → 右」四欄
     股票名稱 | 當前市價 | +/-價格 | +/-百分比

     舊版問題：
     1. 全域 st.columns 被改成 100%，watchlist 也被一起壓成單欄。
     2. watchlist 再用 width:auto + flex，遇到 Streamlit column padding
        後容易發生名稱／按鈕／價格互相覆蓋。
     3. 查看／刪除按鈕 56px 太大，在 390px 左右的手機會擠壓文字。

     新版做法：
     - watchlist row 改成明確的 43/19/19/19 比例。
     - 所有欄位 box-sizing:border-box + min-width:0。
     - 名稱欄內的查看／刪除固定 38px，避免按鈕撐破欄位。
     - 不讓整列換行，因此不會再出現「價格掉到下一行」的情況。
     ============================================================
  */
  .st-key-watchlist_scroll_container{
    width:100% !important;
    max-width:100% !important;
    overflow-x:hidden !important;
    box-sizing:border-box !important;
  }

  /* 表頭保持四欄。 */
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(.watch-head){
    flex-wrap:nowrap !important;
    width:100% !important;
    gap:0 !important;
  }
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(.watch-head) > [data-testid="column"]:nth-child(1){flex:0 0 43% !important;width:43% !important;}
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(.watch-head) > [data-testid="column"]:nth-child(2),
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(.watch-head) > [data-testid="column"]:nth-child(3),
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(.watch-head) > [data-testid="column"]:nth-child(4){
    flex:0 0 19% !important;width:19% !important;
  }

  /* 股票資料列：只鎖定「含有查看按鈕」且有四個直接欄位的 row。 */
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]){
    flex-wrap:nowrap !important;
    width:100% !important;
    max-width:100% !important;
    gap:0 !important;
    row-gap:0 !important;
    box-sizing:border-box !important;
    align-items:center !important;
  }
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]{
    min-width:0 !important;
    max-width:none !important;
    box-sizing:border-box !important;
  }
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:nth-child(1){
    flex:0 0 43% !important;width:43% !important;
  }
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:nth-child(2),
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:nth-child(3),
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:nth-child(4){
    flex:0 0 19% !important;width:19% !important;
  }

  /* 名稱 + 查看 + 刪除仍保持同一行。 */
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:first-child [data-testid="stHorizontalBlock"]{
    flex-wrap:nowrap !important;
    width:100% !important;
    gap:2px !important;
    margin:0 !important;
  }
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:first-child [data-testid="stHorizontalBlock"] > [data-testid="column"]{
    min-width:0 !important;
    box-sizing:border-box !important;
  }
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:first-child [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1){flex:1 1 auto !important;width:auto !important;}
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:first-child [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2),
  .st-key-watchlist_scroll_container [data-testid="stHorizontalBlock"]:has(> [data-testid="column"] button[title="查看此股票"]) > [data-testid="column"]:first-child [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3){
    flex:0 0 38px !important;width:38px !important;
  }

  .st-key-watchlist_scroll_container .watch-name-display{
    width:100% !important;
    max-width:100% !important;
    min-width:0 !important;
    padding:8px 5px !important;
    font-size:14px !important;
    line-height:1.25 !important;
    white-space:nowrap !important;
    overflow:hidden !important;
    text-overflow:ellipsis !important;
    box-sizing:border-box !important;
  }
  .st-key-watchlist_scroll_container .watch-value{
    min-width:0 !important;
    width:100% !important;
    padding:9px 3px !important;
    font-size:15px !important;
    line-height:1.15 !important;
    justify-content:center !important;
    text-align:center !important;
    white-space:nowrap !important;
    overflow:hidden !important;
    text-overflow:ellipsis !important;
    box-sizing:border-box !important;
  }
  .st-key-watchlist_scroll_container .watch-price-cell{
    border-left:1px solid rgba(148,163,184,.18) !important;
    border-top:0 !important;
    padding-left:3px !important;
  }
  .st-key-watchlist_scroll_container .watch-head{
    min-width:0 !important;
    padding:6px 3px !important;
    font-size:11px !important;
    line-height:1.2 !important;
    white-space:nowrap !important;
    overflow:hidden !important;
    text-overflow:ellipsis !important;
    box-sizing:border-box !important;
    text-align:center !important;
  }
  .st-key-watchlist_scroll_container button[title="查看此股票"],
  .st-key-watchlist_scroll_container button[title="查看並切換至此股票"],
  .st-key-watchlist_scroll_container button[title^="從自選股移除"]{
    width:38px !important;
    min-width:38px !important;
    max-width:38px !important;
    height:38px !important;
    min-height:38px !important;
    max-height:38px !important;
    padding:0 !important;
    margin:0 !important;
    border-radius:8px !important;
    box-sizing:border-box !important;
  }
  .st-key-watchlist_scroll_container button[title="查看此股票"] span[data-testid="stIconMaterial"],
  .st-key-watchlist_scroll_container button[title="查看並切換至此股票"] span[data-testid="stIconMaterial"],
  .st-key-watchlist_scroll_container button[title^="從自選股移除"] span[data-testid="stIconMaterial"]{
    font-size:22px !important;width:22px !important;height:22px !important;
  }

  /* 手機上的產業／業務標籤不要搶走過多垂直空間。 */
  .watch-industry-header{font-size:15px !important;padding:8px 9px !important;margin:7px 0 5px !important;}
  .watch-business-header{font-size:13px !important;padding:5px 8px !important;margin:4px 0 !important;}

  /* 標題與搜尋列在手機改為上下排列。 */
  .title-search-row{display:block !important;}
  [data-testid="stTextInput"] input{
    height:44px !important;
    font-size:16px !important;
  }
  button[title='搜尋股票代號或名稱']{
    width:100% !important;
    min-width:100% !important;
    height:44px !important;
    margin-top:2px !important;
  }

  /* 自選股區：手機不使用桌面固定欄寬，改成卡片式資訊。 */
  .watch-head{
    font-size:14px !important;
    min-height:36px !important;
    padding:5px 8px !important;
  }
  .watch-name-display{
    font-size:16px !important;
    padding:10px 8px !important;
  }
  .watch-value{
    justify-content:flex-start !important;
    text-align:left !important;
    font-size:19px !important;
    min-height:42px !important;
    padding:7px 8px !important;
  }
  .watch-price-cell{
    border-left:0 !important;
    border-top:1px solid rgba(148,163,184,.18) !important;
    padding-left:8px !important;
  }
  .watch-name-cell{padding-right:0 !important;}

  .watch-industry-header{
    font-size:16px !important;
    padding:8px 10px !important;
    margin-top:8px !important;
  }
  .watch-business-header{
    font-size:14px !important;
    padding:6px 9px !important;
  }

  /* 搜尋結果在手機上改成一張一張的卡片。 */
  .search-panel{
    padding:11px 12px !important;
    margin:8px 0 !important;
  }
  .search-code{font-size:16px !important;padding:7px 2px !important;}
  .search-name{font-size:15px !important;padding:5px 2px !important;}

  /* 圖表在手機上必須吃滿可用寬度，不使用窄欄位。 */
  .stPlotlyChart,
  [data-testid="stPlotlyChart"]{
    width:100% !important;
    max-width:100% !important;
    min-width:0 !important;
    overflow:visible !important;
  }
  .stPlotlyChart > div,
  [data-testid="stPlotlyChart"] > div,
  .stPlotlyChart iframe{
    width:100% !important;
    max-width:100% !important;
    min-width:0 !important;
  }
  /* 表格保留水平滑動，避免欄位被壓到看不清楚。 */
  [data-testid="stDataFrame"],
  [data-testid="stTable"]{
    max-width:100% !important;
    overflow-x:auto !important;
  }

  /* Tabs 在手機上允許橫向滑動，不把內容擠成極窄欄位。 */
  [data-baseweb="tab-list"]{
    overflow-x:auto !important;
    flex-wrap:nowrap !important;
    scrollbar-width:none !important;
  }
  [data-baseweb="tab-list"]::-webkit-scrollbar{display:none;}
  [data-baseweb="tab"]{
    flex:0 0 auto !important;
    white-space:nowrap !important;
    padding-left:12px !important;
    padding-right:12px !important;
  }

  /* Expander、資訊卡與按鈕不要超出手機寬度。 */
  [data-testid="stExpander"],
  [data-testid="stAlert"],
  [data-testid="stMetric"],
  .element-container{
    max-width:100% !important;
    box-sizing:border-box !important;
  }
  .stButton > button{
    width:100% !important;
    min-height:42px !important;
  }

  /* 手機上的主標題縮小，避免第一屏被標題吃掉。 */
  h1{font-size:1.55rem !important;line-height:1.25 !important;}
  h2{font-size:1.28rem !important;}
  h3{font-size:1.08rem !important;}
  p, li{font-size:0.93rem;}
}

@media (max-width: 430px){
  .main .block-container{
    padding-left:9px !important;
    padding-right:9px !important;
  }
  .live-grid{grid-template-columns:1fr 1fr !important;}
  .live-cell{padding:0 7px !important;}
  .live-price{font-size:19px !important;}
  .live-main{font-size:12px !important;}
  .live-change{font-size:11px !important;}
  .watch-industry-header{font-size:15px !important;}
  .watch-business-header{font-size:13px !important;}
}
</style>
""", unsafe_allow_html=True)

# 開盤前資料模式：使用最近一個交易日資料，避免頁面顯示今天尚未存在的假行情。
_now_tw = taiwan_now()
if _now_tw.time() < MARKET_OPEN:
    st.info(f"📌 今日尚未開盤（台灣時間 {_now_tw.strftime('%H:%M:%S')}），目前價格、趨勢與 K 線以最近一個有效交易日資料顯示；開盤後自動切換回即時行情。")
elif not is_twse_trading_day(_now_tw, BASE):
    st.info(f"📌 {market_status_text(_now_tw, BASE)}")

# 頁面頂端保留空白區，避免右上即時資訊卡遮住搜尋欄。
st.markdown('<div class="top-header-safe-space"></div>', unsafe_allow_html=True)

title_col, search_col = st.columns([6.6, 3.4], gap="medium")
with title_col:
    st.title("AI 台股即時互動式分析系統")
with search_col:
    s1, s2 = st.columns([12, 1], gap=None, vertical_alignment="center")
    with s1:
        search_text = st.text_input("搜尋股票", placeholder="輸入股票代號或名稱，例如 2330", label_visibility="collapsed", key="stock_search")
    with s2:
        do_search = st.button(
            "",
            icon=":material/search:",
            use_container_width=True,
            key="do_stock_search",
            help="搜尋股票代號或名稱",
        )

if do_search:
    found = search_stock(search_text)
    if found.empty:
        st.warning("找不到符合的股票代號或名稱。")
        st.session_state.search_result = None
    elif len(found) == 1:
        code = found.iloc[0]["symbol"]
        st.session_state.selected = code
        st.session_state.search_result = found.iloc[0].to_dict()
        st.session_state.ai_selected = None
        st.session_state.ai_result = None
        st.rerun()
    else:
        st.session_state.search_result = None
        st.session_state.search_candidates = found.to_dict("records")

if st.session_state.get("search_candidates"):
    st.markdown("""
    <div class="search-panel">
      <div class="search-panel-title">🔎 搜尋結果</div>
      <div class="search-panel-subtitle">點選「查看」在<strong>同一分頁</strong>切換所有分析視窗；點選「＋ 加入自選股」會保存到下次開啟。</div>
    </div>
    """, unsafe_allow_html=True)
    candidates = st.session_state.search_candidates
    for item in candidates[:10]:
        c1, c2, c3 = st.columns([1.4, 4.6, 2.2], gap="small")
        with c1:
            st.markdown(f"<div class='search-code'>{item['symbol']}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='search-name'><b>{item['name']}</b><span>｜產業 {item.get('industry','00')}</span></div>", unsafe_allow_html=True)
        with c3:
            b1,b2=st.columns(2, gap="small")
            with b1:
                if st.button("", icon=":material/visibility:", key=f"search_select_{item['symbol']}", use_container_width=True, help="查看並切換至此股票"):
                    st.session_state.selected = item["symbol"]
                    st.session_state.search_result = item
                    st.session_state.search_candidates = None
                    st.session_state.ai_selected = None
                    st.session_state.ai_result = None
                    st.rerun()
            with b2:
                if item["symbol"] in st.session_state.watchlist:
                    st.markdown("<div class='added-badge'>✓ 已加入</div>", unsafe_allow_html=True)
                else:
                    if st.button("＋ 加入", key=f"search_add_{item['symbol']}", use_container_width=True):
                        st.session_state.watchlist.append(item["symbol"])
                        save_watchlist(st.session_state.watchlist)
                        st.session_state.selected = item["symbol"]
                        st.session_state.search_candidates = None
                        st.session_state.search_result = item
                        st.session_state.ai_selected = None
                        st.session_state.ai_result = None
                        st.rerun()

# 搜尋後的目前標的：其它分析視窗立即改為此股票；若來自搜尋且尚未加入自選股，顯示加入按鈕。
# 不在每次首頁啟動時呼叫全市場 tickers，避免非交易日初次載入卡住。
_search_code = st.session_state.selected
_search_row = st.session_state.get("search_result")
if isinstance(_search_row, dict) and _search_row.get("symbol") == _search_code and _search_code not in st.session_state.watchlist:
    add_c1, add_c2, add_c3 = st.columns([3, 5, 1])
    with add_c1:
        st.markdown(f"**搜尋股票：{_search_row.get('symbol')}｜{_search_row.get('name', _search_code)}**")
    with add_c2:
        st.caption("目前已切換所有分析視窗顯示此股票")
    with add_c3:
        if st.button("＋", help="加入視窗一自選股", key=f"add_watch_{_search_code}"):
            st.session_state.watchlist.append(_search_code)
            save_watchlist(st.session_state.watchlist)
            st.rerun()

st.caption("即時選取股票相關行情、盤中 K 線、成交明細於 09:00～13:30 盤中自動更新；盤後首頁只接受可驗證的今日收盤資料，不把前一交易日快照冒充成今天。")
st.caption("🔄 台股盤中 09:00～13:30：每 5 秒自動更新；盤前／非交易日顯示最近完成交易日；盤後顯示今天資料，若今日資料暫時無法驗證則不顯示舊價。")

# ============================================================
# Watchlist fragment
# ============================================================
st.markdown("""
<style>
.watch-head{font-size:20px;font-weight:950;letter-spacing:.4px;min-height:44px;display:flex;align-items:center;padding:4px 10px 8px 8px;color:#f8fafc;}
.search-wrap-spacer{height:0;}
[data-testid="stTextInput"] input{height:46px!important;min-height:46px!important;box-sizing:border-box!important;padding-top:0!important;padding-bottom:0!important;}
[data-testid="stTextInput"]{margin:0!important;display:flex!important;align-items:center!important;width:100%!important;}
[data-testid="stTextInput"]>div{width:100%!important;}
[data-testid="stTextInput"]>div>div{height:46px!important;min-height:46px!important;margin:0!important;width:100%!important;}
[data-testid="stTextInput"] input{height:46px!important;min-height:46px!important;line-height:46px!important;}
/* 搜尋：使用 Streamlit 原生 Material Symbols，避免 emoji / SVG 背景在不同版本縮放不一致。 */
button[title='搜尋股票代號或名稱']{
  font-size:0!important; line-height:1!important;
  width:46px!important; min-width:46px!important; height:46px!important;
  margin:0!important; padding:0!important;
  border-radius:10px!important;
  display:flex!important; align-items:center!important; justify-content:center!important;
  color:#fff!important; background-color:#0f172a!important;
  border:1px solid rgba(255,255,255,.28)!important;
  box-sizing:border-box!important;
  transform:none!important;
  vertical-align:middle!important;
  align-self:center!important;
  position:relative!important;
  top:0!important;
}

button[title='搜尋股票代號或名稱'] span[data-testid='stIconMaterial'],
button[title='搜尋股票代號或名稱'] .material-symbols-rounded{
  font-size:34px!important; line-height:1!important; color:#fff!important;
  width:34px!important; height:34px!important; font-variation-settings:'FILL' 0,'wght' 500,'GRAD' 0,'opsz' 32;
}
button[title='搜尋股票代號或名稱']:hover{background-color:#1e293b!important;border-color:rgba(255,255,255,.42)!important;}
[data-testid="column"]:has(button[title='搜尋股票代號或名稱']){min-width:46px!important;}
.watch-remove-btn button{color:#ef4444!important;font-weight:900!important;}
div[data-testid="stButton"] > button[kind="secondary"]{border:0;background:transparent;padding:8px 8px;text-align:left;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
div[data-testid="stButton"] > button[kind="secondary"]{color:#ffffff !important;}
div[data-testid="stButton"] > button[kind="secondary"]:hover{background:rgba(148,163,184,.12);border:0;}
.watch-name{display:block;padding:8px 10px;border-radius:6px;font-weight:900;line-height:1.25;}
.watch-name.limit{box-shadow:inset 0 0 0 1px rgba(255,255,255,.10);}
.watch-value{padding:15px 10px;font-size:21px;font-weight:900;text-align:right;min-height:58px;display:flex;align-items:center;justify-content:flex-end;}
.watch-price-cell{border-left:2px solid rgba(148,163,184,.24);padding-left:16px;}
.watch-name-cell{padding-right:12px;}
.watch-row-divider{height:1px;background:rgba(148,163,184,.13);margin:1px 0;}
.search-panel{padding:14px 18px;margin:10px 0 12px;border-radius:14px;background:linear-gradient(135deg,#eff6ff,#f8fafc);border:1px solid #bfdbfe;box-shadow:0 6px 20px rgba(15,23,42,.08);}
.search-panel-title{font-size:18px;font-weight:900;color:#0f172a}.search-panel-subtitle{margin-top:4px;color:#475569;font-size:13px}
.search-code{font-size:18px;font-weight:900;padding:10px 4px;color:#0f172a}.search-name{padding:10px 4px;font-size:16px;color:#0f172a}.search-name span{font-size:12px;color:#64748b;margin-left:8px}.added-badge{padding:9px 6px;text-align:center;border-radius:8px;background:#dcfce7;color:#166534;font-weight:800}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.watch-select-row{display:flex;align-items:center;gap:8px;width:100%;}
.watch-name-display{flex:1;min-width:0;padding:9px 10px;border-radius:7px;font-weight:900;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border:1px solid transparent;}
.watch-name-display.up{color:#ef4444;}
.watch-name-display.down{color:#10b981;}
.watch-name-display.flat{color:#64748b;}
.watch-name-display.limit-up{background:#991b1b;color:#fff;border-color:#7f1d1d;}
.watch-name-display.limit-down{background:#14532d;color:#fff;border-color:#064e3b;}
.watch-pick-placeholder{font-size:0;height:1px;}
/* 查看／刪除：使用 Streamlit 原生 Material Symbols，直接放大圖示本身。 */
button[title='查看此股票'], button[title='查看並切換至此股票'], button[title^='從自選股移除']{
  font-size:0!important; line-height:1!important; color:#fff!important;
  width:56px!important; min-width:56px!important; height:56px!important;
  padding:0!important; margin:0!important; border-radius:10px!important;
  display:flex!important; align-items:center!important; justify-content:center!important;
  box-sizing:border-box!important;
}
button[title='查看此股票'] span[data-testid='stIconMaterial'],
button[title='查看並切換至此股票'] span[data-testid='stIconMaterial'],
button[title^='從自選股移除'] span[data-testid='stIconMaterial'],
button[title='查看此股票'] .material-symbols-rounded,
button[title='查看並切換至此股票'] .material-symbols-rounded,
button[title^='從自選股移除'] .material-symbols-rounded{
  font-size:36px!important; line-height:1!important; color:#fff!important;
  width:36px!important; height:36px!important; font-variation-settings:'FILL' 0,'wght' 500,'GRAD' 0,'opsz' 36;
}
button[title='查看此股票']:hover, button[title='查看並切換至此股票']:hover{background-color:rgba(148,163,184,.14)!important;}
button[title^='從自選股移除']:hover{background-color:rgba(239,68,68,.16)!important;}
.watch-industry-header{margin:10px 0 7px;padding:9px 14px;border:1px solid rgba(96,165,250,.28);background:linear-gradient(90deg,rgba(59,130,246,.12),rgba(59,130,246,.03));border-radius:10px;font-size:19px;font-weight:800;letter-spacing:.3px;color:#e5e7eb;}
.watch-industry-dot{display:inline-block;width:9px;height:9px;border-radius:999px;background:#60a5fa;margin-right:9px;vertical-align:2px;}
.watch-industry-code{font-size:14px;color:#94a3b8;font-weight:700;margin-left:4px;}
.watch-industry-separator{height:2px;margin:14px 0 8px;background:linear-gradient(90deg,rgba(96,165,250,.65),rgba(148,163,184,.18),transparent);}
.watch-business-header{margin:6px 0 5px;padding:6px 12px;border-left:4px solid rgba(96,165,250,.5);background:rgba(15,23,42,.35);border-radius:7px;color:#cbd5e1;font-size:16px;font-weight:800;}

</style>
""", unsafe_allow_html=True)

def render_watchlist():
    rows, errors = watch_rows(tuple(st.session_state.watchlist))
    st.subheader("自選股即時行情")
    st.caption("上漲：股票中文名、當前市價、漲跌與漲跌幅均為紅色；下跌則均為綠色；持平為灰色。")
    if not rows.empty and "data_date" in rows.columns:
        valid_dates = sorted({str(v) for v in rows["data_date"].tolist() if str(v).strip()})
        if valid_dates:
            st.caption(f"資料日期：{', '.join(valid_dates)}｜盤後行情僅接受可驗證的當日資料")
        else:
            st.caption("資料日期：目前尚未取得可驗證的行情資料")
    with st.container(height=365, border=True, key="watchlist_scroll_container"):
        heads = st.columns([2.55, 1.85, 1.85, 1.85], gap="small")
        head_labels = ["股票中文名", "當前市價", "+/−價格", "+/−價格%"]
        for i, (c, t) in enumerate(zip(heads, head_labels)):
            with c:
                sep = " border-left:2px solid rgba(148,163,184,.28); padding-left:18px;" if i else ""
                st.markdown(
                    f"<div class='watch-head' style='{sep}'>{t}</div>",
                    unsafe_allow_html=True,
                )
        # 使用「樣式名稱 + 選取按鈕」組合，避免 Streamlit button 的預設 CSS 覆蓋股票名稱顏色。
        # 名稱欄位會真正呈現：上漲紅字／下跌綠字；漲停深紅底白字；跌停深綠底白字。
        # 第一層依官方產業分類；第二層依公司主要業務／產業鏈子分類。
        industry_groups = {}
        for original_idx, record in enumerate(rows.to_dict("records")):
            industry_code = str(record.get("industry", "00")).zfill(2)
            business_group = str(record.get("business_group", "主要業務待建立")) or "主要業務待建立"
            industry_groups.setdefault(industry_code, {})
            industry_groups[industry_code].setdefault(business_group, []).append((original_idx, record))

        first_group = True
        display_index = 0
        for industry_code, business_groups in industry_groups.items():
            if not first_group:
                st.markdown("<div class='watch-industry-separator'></div>", unsafe_allow_html=True)
            first_group = False
            industry_name = INDUSTRIES.get(industry_code, "其他/未分類")
            st.markdown(
                f"<div class='watch-industry-header'><span class='watch-industry-dot'></span>{industry_name} <span class='watch-industry-code'>({industry_code})</span></div>",
                unsafe_allow_html=True,
            )

            for business_group, group_records in business_groups.items():
                st.markdown(
                    f"<div class='watch-business-header'>▸ {business_group}</div>",
                    unsafe_allow_html=True,
                )
                for original_idx, r in group_records:
                    idx = display_index
                    display_index += 1
                    ch = pd.to_numeric(r["change"], errors="coerce")
                    cp = pd.to_numeric(r["change_pct"], errors="coerce")
                    price = pd.to_numeric(r["price"], errors="coerce")
                    lup = pd.to_numeric(r.get("limit_up"), errors="coerce")
                    ldn = pd.to_numeric(r.get("limit_down"), errors="coerce")

                    limit_up = pd.notna(price) and pd.notna(lup) and abs(float(price)-float(lup)) < 1e-9
                    limit_down = pd.notna(price) and pd.notna(ldn) and abs(float(price)-float(ldn)) < 1e-9
                    if not limit_up and not limit_down:
                        limit_up = pd.notna(cp) and cp >= 9.95
                        limit_down = pd.notna(cp) and cp <= -9.95

                    if limit_up:
                        name_class = "limit-up"
                    elif limit_down:
                        name_class = "limit-down"
                    elif pd.notna(ch) and ch > 0:
                        name_class = "up"
                    elif pd.notna(ch) and ch < 0:
                        name_class = "down"
                    else:
                        name_class = "flat"

                    vcolor = "#ef4444" if pd.notna(ch) and ch > 0 else "#10b981" if pd.notna(ch) and ch < 0 else "#64748b"
                    price_txt = "--" if pd.isna(price) else f"{price:,.2f}"
                    ch_txt = "--" if pd.isna(ch) else f"{ch:+,.2f}"
                    cp_txt = "--" if pd.isna(cp) else f"{cp:+.2f}%"

                    b1,b2,b3,b4 = st.columns([2.4,1.8,1.8,1.8], gap="small")
                    with b1:
                        n1, n2, n3 = st.columns([4.1, 1.1, 1.0], gap="small")
                        with n1:
                            st.markdown(
                                f"<div class='watch-name-display {name_class}'>{r['name']}（{r['code']}）</div>",
                                unsafe_allow_html=True,
                            )
                        with n2:
                            st.markdown("<div class='watch-action'>", unsafe_allow_html=True)
                            if st.button("", icon=":material/visibility:", key=f"watch_select_{r['code']}_{idx}", use_container_width=True, type="secondary", help="查看此股票"):
                                st.session_state.selected = r['code']
                                st.session_state.ai_selected = None
                                st.session_state.ai_result = None
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                        with n3:
                            st.markdown("<div class='watch-action watch-remove-action'>", unsafe_allow_html=True)
                            if st.button("", icon=":material/delete:", key=f"watch_remove_{r['code']}_{idx}", use_container_width=True, help=f"從自選股移除 {r['code']}"):
                                new_watchlist = [x for x in st.session_state.watchlist if x != r['code']]
                                st.session_state.watchlist = new_watchlist
                                save_watchlist(new_watchlist)
                                if st.session_state.get('selected') == r['code']:
                                    if new_watchlist:
                                        st.session_state.selected = new_watchlist[0]
                                    else:
                                        st.session_state.selected = WATCHLIST_DEFAULT[0]
                                st.session_state.ai_selected = None
                                st.session_state.ai_result = None
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                    with b2:
                        st.markdown(f"<div class='watch-value watch-price-cell' style='color:{vcolor}'>{price_txt}</div>", unsafe_allow_html=True)
                    with b3:
                        st.markdown(f"<div class='watch-value watch-price-cell' style='color:{vcolor}'>{ch_txt}</div>", unsafe_allow_html=True)
                    with b4:
                        st.markdown(f"<div class='watch-value watch-price-cell' style='color:{vcolor}'>{cp_txt}</div>", unsafe_allow_html=True)
                    # 只有同一產業內的股票之間使用細分隔線；產業之間使用更醒目的分組線。
                    if original_idx != group_records[-1][0]:
                        st.markdown("<div class='watch-row-divider'></div>", unsafe_allow_html=True)
    if errors:
        with st.expander("資料連線診斷"):
            st.code("\n".join(errors[:30]))


# ============================================================
# Selected header + fixed live card
# ============================================================
def render_selected_and_live():
    selected = st.session_state.selected
    rows, _ = watch_rows(tuple(st.session_state.watchlist))
    sr = rows[rows.code == selected]
    sr = sr.iloc[0] if not sr.empty else None
    info = {}
    if is_market_open_now():
        try: info = get_ticker(selected)
        except Exception: info = {}
    name = info.get("name") or (sr["name"] if sr is not None else selected)
    industry_code = INDUSTRY_OVERRIDES.get(selected, str((sr["industry"] if sr is not None else info.get("industry")) or WATCHLIST_INDUSTRY_FALLBACK.get(selected, "00")).zfill(2))
    industry_name = INDUSTRIES.get(industry_code, "未知產業")
    market = str(info.get("market") or (sr["market"] if sr is not None else "TSE"))
    st.markdown(f"## 分析標的：{name}（{selected}）")
    st.caption(f"市場：{market}｜產業：{industry_code} {industry_name}")
    try:
        bp = get_business_profiles_cached((selected,)).get(selected, {})
        bg = bp.get("business_group") or "主要業務待建立"
        chain = bp.get("primary_chain") or ""
        st.markdown(f"**主要業務：** {bg}" + (f"　｜　**產業鏈：** {chain}" if chain else ""))
    except Exception:
        pass

    live = {}
    if is_market_open_now():
        try: live = get_quote(selected)
        except Exception: live = {}
    elif sr is not None:
        live = {
            "tradePrice": sr.get("price", np.nan),
            "closePrice": sr.get("price", np.nan),
            "change": sr.get("change", np.nan),
            "changePercent": sr.get("change_pct", np.nan),
            "referencePrice": np.nan,
            "previousClose": np.nan,
        }
    lp = pd.to_numeric(live.get("tradePrice", live.get("closePrice", np.nan)), errors="coerce")
    lch = pd.to_numeric(live.get("change", np.nan), errors="coerce")
    lcp = pd.to_numeric(live.get("changePercent", np.nan), errors="coerce")
    ref = pd.to_numeric(live.get("referencePrice", live.get("previousClose", np.nan)), errors="coerce")
    if pd.isna(lch) and pd.notna(lp) and pd.notna(ref): lch = lp - ref
    if pd.isna(lcp) and pd.notna(lch) and pd.notna(ref) and ref != 0: lcp = lch / ref * 100
    color = "#ef4444" if pd.notna(lch) and lch > 0 else "#10b981" if pd.notna(lch) and lch < 0 else "#e5e7eb"

    try:
        taiex = get_taiex()
        idx = taiex.iloc[-1] if not taiex.empty else None
    except Exception:
        idx = None
    if idx is not None:
        ix = float(idx["index"]); ixch = float(idx["change"]); ixcp = float(idx["change_pct"])
        ixcolor = "#ef4444" if ixch > 0 else "#10b981" if ixch < 0 else "#e5e7eb"
        ixline, ixsub = f"{ix:,.2f}", f"{ixch:+,.2f}  {ixcp:+.2f}%"
        ixtime = f"最新指數資料 {idx['time']}（台灣時間）"
    else:
        ixline, ixsub, ixcolor, ixtime = "--", "--", "#e5e7eb", "目前無法取得 TWSE 指數資料"

    st.markdown(f"""
    <div class="live-card"><div class="live-grid">
      <div class="live-cell"><div class="live-title">目前選取股票｜盤中每 2 秒刷新</div>
        <div class="live-main">{name}（{selected}）</div>
        <div class="live-price">{'--' if pd.isna(lp) else f'{lp:,.2f}'}</div>
        <div class="live-change" style="color:{color};">{'--' if pd.isna(lch) else f'{lch:+,.2f}'}　{'--' if pd.isna(lcp) else f'{lcp:+.2f}%'} </div>
      </div>
      <div class="live-cell"><div class="live-title">台灣加權指數</div>
        <div class="live-main">TAIEX</div><div class="live-price">{ixline}</div>
        <div class="live-change" style="color:{ixcolor};">{ixsub}</div>
        <div class="live-time">{ixtime}</div>
      </div>
    </div></div>""", unsafe_allow_html=True)


# ============================================================
# Trend
# ============================================================
def render_trend():
    selected = st.session_state.selected
    st.subheader("股價趨勢")
    period = st.radio("期間", ["日內", "五日", "近月", "三月"], horizontal=True, key="period")
    try:
        if period == "日內":
            if is_market_open_now():
                d = normalize_intraday(get_intraday(selected, "1"))
            else:
                prev = latest_trading_date(selected)
                d = pd.DataFrame() if prev is None else normalize_intraday(client.historical_candles(selected, prev, prev, "1"))
            if d.empty: raise RuntimeError("目前沒有可用的盤中 1 分鐘資料")
            day_label = pd.Timestamp(d.date.iloc[-1]).strftime("%Y-%m-%d")
            title, x, y = f"{day_label} 1 分鐘價格走勢", d.date, d.close
        else:
            days = {"五日": 15, "近月": 45, "三月": 120}[period]
            d = get_history(selected, days)
            if d.empty: raise RuntimeError("沒有足夠日 K 資料")
            title, x, y = f"{period}收盤價", d.date, d.close
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name="價格"))
        fig.update_layout(height=400, margin=dict(l=20,r=20,t=45,b=20), title=title, xaxis_title="時間", yaxis_title="價格")
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False}, key=f"trend_{selected}_{period}")
    except Exception as e:
        st.error(f"股價趨勢資料取得失敗：{e}")

# ============================================================
# Technical K
# ============================================================
def render_kline():
    selected = st.session_state.selected
    st.subheader("技術 K 線")
    k_period = st.radio("K線週期", ["日K", "當日 5 分K", "當日 30 分K", "當日 60 分K"], horizontal=True, key="technical_k_period")
    try:
        if k_period == "日K":
            d = get_history(selected, 180); title = "日 K 線"
        else:
            tf = {"當日 5 分K":"5", "當日 30 分K":"30", "當日 60 分K":"60"}[k_period]
            if is_market_open_now():
                d = normalize_intraday(get_intraday(selected, tf))
            else:
                prev = latest_trading_date(selected)
                d = pd.DataFrame() if prev is None else normalize_intraday(client.historical_candles(selected, prev, prev, tf))
            day_label = pd.Timestamp(d.date.iloc[-1]).strftime("%Y-%m-%d") if not d.empty else ""
            title = f"{day_label} {tf} 分鐘 K 線"
        if d.empty: raise RuntimeError("目前無有效 K 線資料")
        fig = go.Figure(go.Candlestick(x=d.date, open=d.open, high=d.high, low=d.low, close=d.close,
                                        name="K線", increasing_line_color="#ef4444", decreasing_line_color="#10b981"))
        if len(d) >= 5: fig.add_trace(go.Scatter(x=d.date, y=d.close.rolling(5).mean(), name="MA5", mode="lines"))
        if len(d) >= 20: fig.add_trace(go.Scatter(x=d.date, y=d.close.rolling(20).mean(), name="MA20", mode="lines"))
        if len(d) >= 60: fig.add_trace(go.Scatter(x=d.date, y=d.close.rolling(60).mean(), name="MA60", mode="lines"))
        fig.update_layout(
            height=580,
            autosize=True,
            title=title,
            xaxis_rangeslider_visible=False,
            margin=dict(l=45, r=10, t=68, b=28),
            yaxis_title="價格",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False}, key=f"kline_{selected}_{k_period}")
        vf = go.Figure(go.Bar(x=d.date, y=d.volume, name="成交量")); vf.update_layout(height=220, title="成交量", margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(vf, use_container_width=True, config={"responsive": True, "displaylogo": False}, key=f"vol_{selected}_{k_period}")
    except Exception as e:
        st.error(f"{k_period} 資料取得失敗：{e}")

# ============================================================
# 進場建議分析系統
# ============================================================
@st.cache_resource(show_spinner=False)
def get_entry_model(selected: str, training_end_key: str):
    """優先載入模型；若舊模型損壞則自動重訓。"""
    try:
        saved = load_saved_model(selected, training_end_key)
        if saved is not None and hasattr(saved, "reg") and hasattr(saved, "clf"):
            return saved
    except Exception:
        pass
    hist = load_training_intraday(client, selected, days=140)
    if hist is None or hist.empty:
        raise RuntimeError("無法取得最近交易日 5 分鐘歷史 K 線")
    model = fit_model(hist, horizon_bars=6)
    save_saved_model(selected, training_end_key, model)
    return model


def render_entry_prediction():
    selected = st.session_state.selected
    st.subheader("進場建議分析系統")
    st.caption(
        "以 5 分鐘 K 線為主：將 K 線型態、短線趨勢、價格結構、VWAP、量能、日內位置與前一交易日背景統合，"
        "只在上漲趨勢中尋找回檔進場機會；若高檔急拉則優先給出『不追高』建議。"
    )

    try:
        train_end = latest_trading_date(selected)
        train_end_key = str(train_end or "latest")
        model = get_entry_model(selected, train_end_key)
    except Exception as e:
        st.warning(f"進場建議模型建立失敗：{e}")
        return

    try:
        k5 = normalize_intraday(get_intraday(selected, "5"))
        if k5.empty:
            raise RuntimeError("目前沒有可用的最近交易日 5 分鐘 K 線")
        q = get_quote(selected)
        live_price = pd.to_numeric(q.get("lastPrice", q.get("closePrice", np.nan)), errors="coerce")
        # 先判斷 Quote 是否為今天；若盤後 Quote 尚未同步，優先用今天最後一根 5 分 K。
        today = taiwan_now().date()
        qdate = pd.to_datetime(q.get("date"), errors="coerce") if q else pd.NaT
        bar_date = pd.to_datetime(k5["date"].iloc[-1], errors="coerce") if "date" in k5.columns and not k5.empty else pd.NaT
        if (pd.isna(qdate) or qdate.date() != today) and pd.notna(bar_date) and bar_date.date() == today:
            candidate = pd.to_numeric(k5["close"].iloc[-1], errors="coerce")
            if pd.notna(candidate):
                live_price = candidate
        if pd.isna(live_price):
            live_price = float(k5.close.iloc[-1])
        result = predict_entry(model, k5, float(live_price))
        result["reference_price"] = float(live_price)
    except Exception as e:
        st.error(f"進場建議分析失敗：{e}")
        return

    action = result["action"]
    if action == "適合回檔進場":
        action_color = "#dc2626"
        action_bg = "#fef2f2"
    elif "不建議" in action:
        action_color = "#047857"
        action_bg = "#ecfdf5"
    elif "不追高" in action:
        action_color = "#b45309"
        action_bg = "#fffbeb"
    else:
        action_color = "#475569"
        action_bg = "#f8fafc"

    confidence = float(result["confidence"])
    entry_score = float(result["entry_score"])
    trend_score = float(result["trend_score"])

    st.markdown(
        f"""
        <div style="background:{action_bg};border:2px solid {action_color};border-radius:14px;padding:14px 18px;margin:4px 0 16px 0;">
            <div style="font-size:16px;color:#64748b;font-weight:700;">目前單一進場決策</div>
            <div style="font-size:30px;color:{action_color};font-weight:900;line-height:1.2;margin-top:3px;">{action}</div>
            <div style="font-size:15px;color:#475569;margin-top:6px;">{result['decision_reason']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前價格", f"{result['reference_price']:.2f}")
    c2.metric("最佳進場價格", f"{result['best_entry_price']:.2f}")
    c3.metric("進場評分", f"{entry_score:.1f} / 100")
    c4.metric("進場可信度", f"{confidence*100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("趨勢強度", f"{trend_score:.1f} / 100")
    c6.metric("K線型態", result["pattern_name"])
    c7.metric("未來30分進場機率", f"{result['opportunity_probability']*100:.1f}%")
    c8.metric("VWAP", f"{result['vwap']:.2f}")

    st.markdown("#### K 線型態與趨勢分析")
    a1, a2, a3 = st.columns(3)
    a1.markdown(f"**目前趨勢**  \n### {result['trend_state']}")
    a2.markdown(f"**目前 K 線**  \n### {result['pattern_name']}")
    a3.markdown(f"**型態偏多分數**  \n### {result['pattern_score']:+.2f}")

    st.progress(float(np.clip(trend_score / 100, 0, 1)), text=f"趨勢強度 {trend_score:.1f} / 100")
    st.progress(float(np.clip(entry_score / 100, 0, 1)), text=f"進場評分 {entry_score:.1f} / 100")

    gap_pct = (result["reference_price"] / result["best_entry_price"] - 1) * 100
    overext_text = "目前價格有明顯延伸，系統禁止追高。" if result["overextended"] else "目前未偵測到明顯過熱延伸。"
    st.info(
        f"**建議進場區間：{result['entry_lower']:.2f} ～ {result['entry_upper']:.2f}**　｜ "
        f"預測未來約 **{model.horizon_bars*5} 分鐘最低價：{result['predicted_future_low']:.2f}**　｜ "
        f"相對最佳進場價溢價：**{gap_pct:.2f}%**\n\n{overext_text}"
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("近期支撐", "--" if not np.isfinite(result['support_price']) else f"{result['support_price']:.2f}")
    d2.metric("預期反彈價格", f"{result['predicted_rebound_price']:.2f}")
    d3.metric("ATR(12)", f"{result['atr_price']:.2f}")
    d4.metric("6K量能比", f"{result['volume_ratio_6']:.2f}x")

    # 這一區把決策理由直接翻成使用者可讀的分析。
    st.markdown("#### 進場判斷理由")
    reasons = result["decision_reason"].split("；") if result["decision_reason"] else ["等待價格結構進一步確認"]
    # 固定成單一 Markdown 元件，避免盤中 2 秒更新時因理由數量變動造成 React DOM
    # insertBefore / removeChild reconciliation 問題。
    reason_text = "\n".join(f"- {str(reason).strip()}" for reason in reasons if str(reason).strip())
    st.markdown(reason_text or "- 等待價格結構進一步確認")

    st.markdown(
        f"模型：**Random Forest + 5 分鐘 K 線型態/趨勢進場模型 v2**｜"
        f"訓練樣本：**{model.train_samples:,}**｜交易日：**{model.train_days}**｜"
        f"OOF 最低價 MAE：**{model.oof_mae*100:.3f}%**｜"
        f"OOF 回檔方向準確率：**{model.oof_direction_accuracy*100:.1f}%**｜"
        f"OOF 進場機會準確率：**{model.oof_opportunity_accuracy*100:.1f}%**"
    )
    feature_time = pd.Timestamp(result["feature_time"])
    st.caption(
        f"最新 5 分 K：{feature_time.strftime('%Y-%m-%d %H:%M:%S')}（台灣時間）｜"
        "K線型態參照 OANDA 14 種 K 線概念；實際決策由本專案歷史台股資料訓練結果與當下市場結構共同決定。"
    )
    st.caption("提醒：這是機率式進場決策參考，不保證觸及最低價，也不保證成交；實際下單仍應搭配停損與部位管理。")

    # ---------------- AI Agent 協作分析 ----------------
    try:
        daily_for_agent = get_history(selected, 120)
        try:
            q_agent = get_quote(selected) or {}
        except Exception:
            q_agent = {}
        inst_agent = pd.DataFrame()
        tdcc_agent = pd.DataFrame()
        try:
            sr = watch_rows(tuple(st.session_state.watchlist))[0]
            market_agent = str(sr.loc[sr.code == selected, "market"].iloc[0]) if (not sr.empty and (sr.code == selected).any()) else "TSE"
            inst_agent = get_institutions(selected, market_agent)
        except Exception:
            pass
        try:
            tdcc_agent = get_tdcc(selected)
        except Exception:
            pass
        agent_ctx = AgentContext(symbol=selected, as_of=str(taiwan_now()), market_open=is_market_open_now(),
                                 quote=q_agent, intraday=k5, daily=daily_for_agent, entry_result=result,
                                 ai_result=st.session_state.get("ai_result"), institutions=inst_agent, tdcc=tdcc_agent)
        agent_result = AGENT_SUPERVISOR.run(agent_ctx)
        st.markdown("#### AI Agent 進場決策協作")
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Agent 綜合分數", f"{agent_result['score']:.1f} / 100")
        ac2.metric("Agent 信心", f"{agent_result['confidence']*100:.1f}%")
        ac3.metric("資料層", agent_result['findings']['data_quality']['status'])
        ac4.metric("風險層", agent_result['findings']['risk_guard']['status'])
        st.caption("流程：資料品質 Agent → 技術 Agent → 籌碼 Agent → Risk Guard → Decision Agent。數值結果由量化模型決定；本機 Qwen3 只作工具編排與證據說明，不可覆寫數值結果。")
        ollama_model = st.session_state.get("ollama_model", "qwen3:8b")
        agent_col1, agent_col2 = st.columns([1, 3])
        with agent_col1:
            run_local_agent = st.button("啟動本機 Agent", type="secondary", use_container_width=True, key=f"run_ollama_agent_{selected}")
        with agent_col2:
            ok_local, local_msg = ollama_available(model=ollama_model)
            if ok_local:
                st.success(f"Ollama / {ollama_model} 已就緒｜完全本機，不使用 OpenAI API")
            else:
                st.warning(local_msg)
        if run_local_agent:
            context = {
                "symbol": selected, "market_open": is_market_open_now(), "quote": q_agent,
                "intraday": k5, "daily": daily_for_agent, "entry_result": result,
                "ai_result": st.session_state.get("ai_result"), "institutions": inst_agent,
                "tdcc": tdcc_agent, "output_dir": str(OUTPUT),
            }
            try:
                local_agent = LocalOllamaAgent(context, AGENT_LOG_DIR, model=ollama_model)
                prompt = (
                    "請分析目前這檔股票的進場狀態。必須先使用工具查證行情、技術面與既有模型結果；"
                    "再說明是否符合上漲趨勢回檔進場條件、目前最大的風險，以及下一步應觀察什麼。"
                    "不得改寫既有模型的價格、分數、方向與可信度。"
                )
                with st.spinner(f"本機 {ollama_model} Agent 正在使用工具分析…"):
                    agent_out = local_agent.run(prompt)
                st.session_state["ollama_agent_result"] = agent_out
                st.rerun()
            except Exception as exc:
                st.error(f"本機 Agent 執行失敗：{exc}")
        local_out = st.session_state.get("ollama_agent_result")
        if isinstance(local_out, dict) and local_out.get("symbol") == selected:
            st.markdown("**Ollama Agent 分析**")
            st.info(local_out.get("content") or "Agent 沒有產生文字摘要。")
            trace = local_out.get("trace") or []
            if trace:
                with st.expander(f"工具使用紀錄（{len(trace)} 次）", expanded=False):
                    for i, item in enumerate(trace, 1):
                        st.write(f"{i}. {item.get('tool')}")
                        st.json(item.get("output"), expanded=False)
    except Exception as e:
        st.caption(f"AI Agent 協作層暫不可用，不影響原本進場模型：{e}")


# ============================================================
# Big vs retail
# ============================================================
def render_tdcc():
    selected = st.session_state.selected
    st.subheader("大戶 VS 散戶持有股比率")
    st.caption("TDCC 股權分散是週資料；頁面 2 秒更新只會重新顯示最新已取得資料，不會虛構新一期籌碼。")
    try:
        tdcc = get_tdcc(selected)
        if tdcc.empty:
            st.info("目前找不到此股票的 TDCC 股權分散資料。")
            return
        row = tdcc.iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("大戶持股比例", f"{row.big_pct:.2f}%")
        c2.metric("散戶持股比例", f"{row.retail_pct:.2f}%")
        c3.metric("中間持股比例", f"{row.middle_pct:.2f}%")
        pie = go.Figure(go.Pie(labels=["大戶（≥400張）","散戶（≤50張）","中間持股"], values=[row.big_pct,row.retail_pct,row.middle_pct], hole=.45))
        pie.update_layout(height=380, title=f"{selected}｜最新一期籌碼結構（{pd.Timestamp(row.date).date()}）", margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(pie, use_container_width=True, config={"responsive": True, "displaylogo": False}, key=f"tdcc_{selected}")
    except Exception as e:
        st.warning(f"大戶/散戶資料取得失敗：{e}")

# ============================================================
# Trades - Taiwan time + colored bid/ask
# ============================================================
def render_trades():
    selected = st.session_state.selected
    st.subheader("交易明細（台灣時間）")
    try:
        td = get_trades(selected)
        if td.empty:
            st.info("目前沒有成交明細")
            return
        td["時間"] = td.get("time", pd.Series(dtype=object)).map(taipei_time).dt.strftime("%H:%M:%S")
        td = td.rename(columns={"bid":"買進價格", "ask":"賣出價格", "price":"成交價", "size":"單量"})
        wanted = ["時間","買進價格","賣出價格","成交價","單量"]
        show = td[[c for c in wanted if c in td.columns]].head(300).copy()
        for c in ["買進價格","賣出價格","成交價","單量"]:
            if c in show.columns: show[c] = pd.to_numeric(show[c], errors="coerce")
        styler = show.style
        if "買進價格" in show.columns:
            styler = styler.map(lambda _: "color:#ef4444;font-weight:800", subset=["買進價格"])
        if "賣出價格" in show.columns:
            styler = styler.map(lambda _: "color:#10b981;font-weight:800", subset=["賣出價格"])
        st.dataframe(styler, use_container_width=True, height=360, hide_index=True)
    except Exception as e:
        st.error(f"交易明細取得失敗：{e}")

# ============================================================
# Institutions
# ============================================================
def render_institutions():
    selected = st.session_state.selected
    rows, _ = watch_rows(tuple(st.session_state.watchlist)); sr=rows[rows.code==selected]
    try: info=get_ticker(selected)
    except Exception: info={}
    market=str(info.get("market") or (sr.iloc[0]["market"] if not sr.empty else "TSE"))
    st.subheader("三大法人 10 交易日買賣超")
    try:
        inst = get_institutions(selected, market)
        if inst.empty:
            st.warning("目前抓不到此股票的三大法人資料")
            return
        labels={"foreign_net":"外資買賣超","trust_net":"投信買賣超","dealer_net":"自營商買賣超","total_net":"三大法人買賣超"}
        show=st.multiselect("選擇顯示法人",list(labels),default=["foreign_net","trust_net","dealer_net"],format_func=lambda x:labels[x],key=f"inst_show_{selected}")
        if show:
            fig=go.Figure()
            for c in show:
                vals=pd.to_numeric(inst[c],errors="coerce")
                fig.add_trace(go.Scatter(x=inst.date,y=vals,mode="lines+markers",name=labels[c]))
            fig.add_hline(y=0,line_dash="dash",line_width=1)
            fig.update_layout(height=450,title=f"{selected}｜最近 10 個交易日買賣超",yaxis_title="買賣超（股）",xaxis_title="交易日",margin=dict(l=20,r=20,t=55,b=20))
            st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False}, key=f"inst_chart_{selected}")
        disp=inst.copy(); disp["date"]=disp.date.dt.strftime("%Y-%m-%d")
        disp=disp.rename(columns={"date":"日期","symbol":"股票代號","foreign_net":"外資買賣超","trust_net":"投信買賣超","dealer_net":"自營商買賣超","total_net":"三大法人買賣超","market":"市場","source":"資料來源"})
        for c in ["外資買賣超","投信買賣超","自營商買賣超","三大法人買賣超"]:
            if c in disp.columns: disp[c]=pd.to_numeric(disp[c],errors="coerce")
        sty=disp.style
        for c in ["外資買賣超","投信買賣超","自營商買賣超","三大法人買賣超"]:
            if c in disp.columns: sty=sty.map(color_for_value,subset=[c])
        st.dataframe(sty,use_container_width=True,hide_index=True)
    except Exception as e:
        st.error(f"法人資料取得失敗：{e}")

# ============================================================
# AI - manual training only; result is a single integrated decision
# ============================================================
def get_ai_adjusted_history(code: str, days: int = 1100) -> pd.DataFrame:
    """取得 AI 訓練用的還原股價日 K。

    除權息會造成名目股價跳空，但不等同於持有人經濟損失。
    AI 訓練使用 adjusted=True，避免模型把除權息造成的機械性價格跳跌當成一般賣壓。
    快取獨立保存，避免與畫面一般日 K 混用。
    """
    code = clean_symbol(code)
    path = DAILY_DATA / f"{code}_adjusted.csv"
    end = taiwan_now().tz_localize(None).normalize()
    start_date = end - pd.Timedelta(days=days)
    cached = pd.DataFrame()
    if path.exists():
        try:
            cached = pd.read_csv(path, parse_dates=["date"])
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        except Exception:
            cached = pd.DataFrame()
    parts=[]
    if not cached.empty:
        parts.append(cached)
        latest = cached["date"].max().normalize()
        fetch_start=max(start_date, latest-pd.Timedelta(days=3))
    else:
        fetch_start=start_date
    cur=fetch_start
    while cur <= end:
        ce=min(cur+pd.Timedelta(days=329), end)
        try:
            part=client.historical_candles(code, cur.date(), ce.date(), "D", adjusted=True)
            if not part.empty:
                parts.append(part)
        except Exception:
            pass
        cur=ce+pd.Timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    out=pd.concat(parts, ignore_index=True)
    out["date"]=pd.to_datetime(out["date"], errors="coerce")
    out=out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    out.to_csv(path,index=False,encoding="utf-8-sig")
    return out[(out["date"]>=start_date)&(out["date"]<=end+pd.Timedelta(days=1))].copy()


def history_with_latest_market_bar(code: str, history_days: int = 1100, adjusted: bool = False) -> pd.DataFrame:
    """
    取得 AI 使用的日 K。

    Fugle Historical Candles 的日 K 盤後才更新，因此 13:30 收盤後、
    但歷史日 K 尚未同步時，必須用 Intraday Quote 的今日收盤/最後成交
    補上一根「今日 K」。這讓隔日模型使用真正的今日收盤價，而不是昨天。
    """
    code = clean_symbol(code)
    hist = get_ai_adjusted_history(code, history_days).copy() if adjusted else get_history(code, history_days).copy()
    if hist.empty:
        return hist

    now = taiwan_now()
    q = {}
    try:
        q = get_quote(code) or {}
    except Exception:
        q = {}

    qdate = pd.to_datetime(q.get("date"), errors="coerce")
    close = pd.to_numeric(q.get("closePrice", q.get("lastPrice", np.nan)), errors="coerce")
    if pd.isna(qdate) or pd.isna(close) or qdate.date() != now.date():
        return hist

    # 非交易日不要把舊報價硬塞成今日 K。
    if not is_twse_trading_day(now, BASE):
        return hist

    today = qdate.normalize()
    open_p = pd.to_numeric(q.get("openPrice", np.nan), errors="coerce")
    high_p = pd.to_numeric(q.get("highPrice", np.nan), errors="coerce")
    low_p = pd.to_numeric(q.get("lowPrice", np.nan), errors="coerce")
    avg_p = pd.to_numeric(q.get("avgPrice", np.nan), errors="coerce")
    total = q.get("total") or {}
    volume = pd.to_numeric(total.get("tradeVolume", np.nan), errors="coerce")
    prev_close = pd.to_numeric(q.get("previousClose", np.nan), errors="coerce")

    # 某些報價情況可能缺少 OHLC，必要時以可用欄位退化。
    if pd.isna(open_p):
        open_p = close
    if pd.isna(high_p):
        high_p = close
    if pd.isna(low_p):
        low_p = close
    if pd.isna(volume):
        old = hist.loc[hist["date"] < today].tail(1)
        volume = float(old["volume"].iloc[0]) if not old.empty and "volume" in old.columns else 0.0

    today_row = pd.DataFrame([{
        "date": today,
        "open": float(open_p),
        "high": float(high_p),
        "low": float(low_p),
        "close": float(close),
        "volume": float(volume),
        "average": float(avg_p) if pd.notna(avg_p) else np.nan,
        "change": float(close - prev_close) if pd.notna(prev_close) else np.nan,
        "stock_code": code,
    }])

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.tz_localize(None)
    hist = hist.dropna(subset=["date"])
    hist = hist[hist["date"] != today]
    out = pd.concat([hist, today_row], ignore_index=True)
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def train_selected_ai(selected, industry_code, industry_name):
    selected = clean_symbol(selected)
    industry_code = str(industry_code).zfill(2)

    # 優先使用完整的本機產業主檔，再用 Fugle 最新 tickers 補充/校正。
    # 避免舊版快取只剩目標股，導致「共同訓練股票 = 1 檔」。
    sector_map = {}
    master_path = DATA / "industry_master.csv"
    if master_path.exists():
        try:
            master = pd.read_csv(master_path, dtype={"symbol": str, "industry": str})
            if not master.empty:
                master["symbol"] = master["symbol"].astype(str).map(clean_symbol)
                master["industry"] = master["industry"].map(lambda v: str(v).zfill(2))
                for _, row in master.iterrows():
                    code = clean_symbol(row.get("symbol", ""))
                    if code:
                        sector_map[code] = INDUSTRY_OVERRIDES.get(code, str(row.get("industry", "00")).zfill(2))
        except Exception:
            pass

    # 線上資料可以補上主檔沒有的最新股票；若線上 industry=00，仍優先採用主檔/override。
    ticks = []
    for ex in ["TWSE", "TPEx"]:
        try:
            ticks += client.tickers(ex)
        except Exception:
            pass
    for item in ticks:
        code = clean_symbol(item.get("symbol", ""))
        if not code:
            continue
        live_ind = str(item.get("industry", "00")).zfill(2)
        sector_map[code] = INDUSTRY_OVERRIDES.get(code, sector_map.get(code, live_ind))

    sector = sorted([code for code, ind in sector_map.items() if ind == industry_code])
    if industry_code == "00":
        sector = [selected]
    if selected not in sector:
        sector.append(selected)
    sector = sorted(set(sector))

    frames=[]; failed=[]; latest_dates=[]
    benchmark=history_with_latest_market_bar("0050",1200)
    for code in sector:
        try:
            # 訓練集只使用已完成、已有隔日標籤的歷史日 K；
            # 今日 K 只拿來建立「明日預測」的最新特徵，避免資料洩漏。
            hist=get_ai_adjusted_history(code,1100)
            if len(hist)>=80:
                latest_dates.append(pd.to_datetime(hist["date"],errors="coerce").max().date())
                frames.append(hist.assign(stock_code=code))
            else:
                failed.append(code)
        except Exception:
            failed.append(code)
    if not frames: raise RuntimeError("沒有可用的產業訓練資料")
    training_data_date = str(min(latest_dates)) if latest_dates else None

    # 若同產業、同一批股票、同一資料日已有模型，直接載入，不重訓。
    cached_bundle = load_model_bundle(MODELS, industry_code)
    if bundle_is_usable(cached_bundle, sector, training_data_date):
        selected_hist = history_with_latest_market_bar(selected,1100, adjusted=True).assign(stock_code=selected)
        selected_feat = add_features(selected_hist, benchmark)
        result = predict_from_bundle(cached_bundle, selected_feat, selected)
        trained_at = cached_bundle.get("trained_at")
        reused = True
    else:
        feat=add_features(pd.concat(frames,ignore_index=True),benchmark)
        # 共同訓練模型使用已完成且可形成隔日標籤的歷史資料。
        # 預測時另外補入「今日」最新行情，避免把昨天收盤當成今日基準價。
        out=train(feat,selected)
        selected_hist = history_with_latest_market_bar(selected,1100, adjusted=True).assign(stock_code=selected)
        selected_feat = add_features(selected_hist, benchmark)
        # 重新以今天最新特徵做一次只預測、不重訓的結果產生。
        result = predict_from_bundle({
            "reg": out["reg"],
            "clf": out["clf"],
            "result": out["result"],
            "residuals": out.get("residuals", []),
            "regression_weight": out.get("regression_weight", 0.5),
            "classification_weight": out.get("classification_weight", 0.5),
            "sector_codes": sector,
            "training_samples": out.get("training_samples", 0),
            "training_stock_count": out.get("training_stock_count", len(sector)),
        }, selected_feat, selected)
        save_model_bundle(MODELS, industry_code, out, sector, training_data_date)
        trained_at = pd.Timestamp.now(tz="Asia/Taipei").isoformat()
        reused = False

    # 每次完成/載入一次隔日預測，都把該股票結果留下，方便追蹤歷史模型輸出。
    pred_dir = OUTPUT / "ai_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{clean_symbol(selected)}.json"
    try:
        payload = dict(result)
        payload.update({"stock_code": clean_symbol(selected), "industry_code": industry_code, "industry_name": industry_name,
                        "training_data_date": training_data_date, "model_file": f"models/ai_sector_{industry_code}.joblib",
                        "saved_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat()})
        pred_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return result, failed, reused, trained_at, training_data_date


def render_ai():
    selected=st.session_state.selected
    st.subheader("AI 隔日預測（優先任務）")
    # 首頁載入時不先呼叫 Fugle ticker；先讀本機產業主檔／預設分類。
    industry_code = INDUSTRY_OVERRIDES.get(selected, WATCHLIST_INDUSTRY_FALLBACK.get(selected, "00"))
    cache_path = DATA / "industry_master.csv"
    if cache_path.exists():
        try:
            cm = pd.read_csv(cache_path, dtype={"symbol": str, "industry": str})
            cm["symbol"] = cm["symbol"].astype(str).map(clean_symbol)
            hit = cm[cm["symbol"] == clean_symbol(selected)]
            if not hit.empty:
                industry_code = INDUSTRY_OVERRIDES.get(selected, str(hit.iloc[0].get("industry", industry_code)).zfill(2))
        except Exception:
            pass
    industry_name=INDUSTRIES.get(industry_code,"未知產業")
    a,b=st.columns([1,3])
    with a:
        run=st.button("開始訓練本產業模型",type="primary",use_container_width=True,key=f"run_ai_{selected}")
    with b:
        st.info(f"{selected} → {industry_name}（{industry_code}）→ 使用該產業全部股票共同訓練，再預測 {selected}")
    if run:
        st.session_state.ai_busy = True
        try:
            with st.spinner("正在執行明日股價預測；其他視窗暫停更新，請等待預測任務完成…"):
                result,failed,reused,trained_at,training_data_date=train_selected_ai(selected,industry_code,industry_name)
            st.session_state.ai_selected=selected; st.session_state.ai_result=result; st.session_state.ai_failed=failed
            st.session_state.ai_reused=reused; st.session_state.ai_trained_at=trained_at; st.session_state.ai_training_data_date=training_data_date
            st.session_state.ai_busy = False
            st.rerun()
        except Exception as e:
            st.session_state.ai_busy = False
            st.error(f"AI 模型建立失敗：{e}")
            return
    if st.session_state.ai_selected==selected and st.session_state.ai_result is not None:
        r=st.session_state.ai_result
        direction=r.get("direction","持平")
        dcolor="#ef4444" if direction=="上漲" else "#10b981" if direction=="下跌" else "#64748b"
        c1,c2,c3,c4=st.columns(4)
        c1.metric("模型基準價格",f"{r['current_price']:.2f}")
        c2.metric("預測隔日價格",f"{r['predicted_price']:.2f}")
        c3.metric("預測報酬率",f"{r['predicted_return']*100:+.2f}%")
        c4.markdown(f"**最終預測方向**<div style='font-size:28px;font-weight:900;color:{dcolor}'>{direction}</div>",unsafe_allow_html=True)
        st.markdown(f"**統合預測可信度：{r['confidence']*100:.2f}%**　｜上漲機率：{r['probability_up']*100:.2f}%　｜下跌機率：{r['probability_down']*100:.2f}%")
        st.progress(float(np.clip(r['confidence'],0,1)))
        st.write(f"模型基準日：**{r['prediction_base_date']}**　｜預測日：**{r['prediction_date']}**　｜僅使用最後有效交易日 K 資料，不使用電腦目前時間作為交易時間。")
        st.write(f"價格區間：**{r['lower_price']:.2f} ~ {r['upper_price']:.2f}**　｜訓練樣本：{r['train_samples']:,}　｜共同訓練股票：{r['train_stocks']} 檔")
        cache_status = "沿用已保存模型（不重新訓練）" if st.session_state.get("ai_reused") else "本次完成訓練並已保存模型"
        st.caption(f"模型狀態：{cache_status}｜模型資料基準日：{st.session_state.get('ai_training_data_date', r.get('prediction_base_date', '-'))}｜模型檔：models/ai_sector_{industry_code}.joblib")
        st.markdown(f"Regression 決策權重：**{r['regression_weight']*100:.1f}%**　｜Classification 決策權重：**{r['classification_weight']*100:.1f}%**")
        m = r.get("metrics") or {}
        metric_keys = [
            "MAE", "RMSE", "Accuracy", "Precision", "Recall", "F1",
            "RegressionDirectionAccuracy", "ClassificationDirectionAccuracy"
        ]
        metric_row = {k: m.get(k, np.nan) for k in metric_keys}
        # 相容舊模型：舊版模型可能沒有完整的 OOF metrics。
        # 不讓單一缺失欄位造成整個 Dashboard KeyError。
        st.markdown("**5-Fold Time Series OOF 驗證**")
        st.dataframe(
            pd.DataFrame([metric_row]),
            use_container_width=True,
            hide_index=True,
        )
        if st.session_state.get("ai_failed"):
            st.caption(f"資料不足/失敗而未納入訓練的股票：{len(st.session_state.ai_failed)} 檔")
    else:
        st.info("按下「開始訓練本產業模型」後才會建立模型。模型建立後，結果會持續顯示，但不會每 2 秒重新訓練。")


def render_model_research_agent():
    """盤後模型研究 Agent：研究模型健康、特徵漂移、產業新聞與官方證據，不自動改 production model。"""
    st.markdown("## 模型研究 Agent")
    st.caption(
        "盤後研究模式目前聚焦自選股中的：半導體業（24）、光電業（26）、電子零組件業（28）；"
        "優先做財報、營收、接單/訂單、法人、產業競爭與公開供應鏈關係分析，再用新聞與總經資料補強。"
        "本機 Qwen3 僅負責證據統整與待查證判斷，不會自動修改 production model。"
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    now = taiwan_now()
    # 研究 Agent 與交易日無關：週末、國定休市日也可以手動執行。
    # 盤後自動排程依 setup_research_task.ps1 設定時間執行；Dashboard 不再因為休市日而鎖住研究按鈕。
    allowed = True
    with c1:
        run_research = st.button(
            "執行盤後研究",
            type="primary",
            use_container_width=True,
            key="run_model_research",
            disabled=not allowed or st.session_state.get("research_busy", False),
        )
    with c2:
        load_latest = st.button("載入最新研究報告", use_container_width=True, key="load_latest_research")
    with c3:
        if not is_twse_trading_day(now, BASE):
            st.info(f"{market_status_text(now, BASE)}")
        elif now.time() < dt_time(15, 0):
            st.info("目前尚未到一般盤後時間；研究 Agent 仍可執行，行情相關內容以最近可驗證資料為準。")
        else:
            st.success("目前可執行盤後研究。")

    # --------------------------------------------------------
    # AI 盤後財報 Email Agent
    # --------------------------------------------------------
    email_agent = EmailAgent(BASE)
    email_status = email_agent.status()
    email_send_latest = False
    email_force = False
    with st.expander("AI 盤後財報 Email Agent", expanded=False):
        if email_status.get("configured"):
            st.success(f"Email 已設定：{email_status.get('sender')} → {email_status.get('recipient')}｜SMTP {email_status.get('smtp_host')}:{email_status.get('smtp_port')}")
            st.caption("Windows 盤後自動研究完成後，若 EMAIL_AUTO_SEND=true，會自動寄送當日盤後財報；每次執行都會寄送，方便測試。")
        else:
            missing = "、".join(email_status.get("missing", [])) or "未知設定"
            st.warning(f"Email Agent 尚未完成設定：{missing}")
            st.caption("請在專案根目錄建立 .env，設定 Gmail 帳號與 App Password。一般 Gmail 登入密碼不要直接放入程式。")
        st.code("""EMAIL_ENABLED=true
EMAIL_AUTO_SEND=true
EMAIL_SENDER=你的Gmail帳號@gmail.com
EMAIL_APP_PASSWORD=你的16位AppPassword
EMAIL_RECIPIENT=10501500978aaa@gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USE_SSL=false
EMAIL_ATTACH_REPORT=true""", language="dotenv")
        b1, b2 = st.columns(2)
        with b1:
            email_send_latest = st.button("寄送最新盤後財報", use_container_width=True, key="send_latest_email")
        with b2:
            email_force = st.checkbox("忽略今日已寄送標記", value=False, key="force_email_send")

    if run_research:
        st.session_state.research_busy = True
        try:
            symbols = list(dict.fromkeys(st.session_state.watchlist))
            with st.spinner("本機模型研究 Agent 正在分析：財報、營收、法說會、接單、近期產業漲跌、新聞與模型健康…"):
                agent = ResearchAgent(BASE, ollama_model=st.session_state.get("ollama_model", "qwen3:8b"))
                report = agent.run_daily_research(symbols=symbols, sector_codes=["24", "26", "28"])
            st.session_state.research_result = report
            # Dashboard 手動執行也可沿用與 Windows 排程相同的 Email Agent。
            if email_status.get("configured") and email_status.get("auto_send"):
                mail_result = email_agent.send_report(report)
                st.session_state.email_send_result = mail_result
        except Exception as exc:
            st.error(f"模型研究 Agent 執行失敗：{exc}")
        finally:
            st.session_state.research_busy = False

    if load_latest or st.session_state.research_result is None:
        latest = sorted((OUTPUT / "research_reports").glob("research_*.json")) if (OUTPUT / "research_reports").exists() else []
        if latest:
            try:
                st.session_state.research_result = json.loads(latest[-1].read_text(encoding="utf-8"))
            except Exception:
                pass

    report = st.session_state.get("research_result")
    if email_send_latest:
        if not isinstance(report, dict):
            report = email_agent.load_latest_report()
            if isinstance(report, dict):
                st.session_state.research_result = report
        if isinstance(report, dict):
            mail_result = email_agent.send_report(report, force=email_force)
            st.session_state.email_send_result = mail_result
            if mail_result.get("sent"):
                st.success(f"盤後財報已寄送至 {email_status.get('recipient')}")
            elif mail_result.get("skipped"):
                st.info(str(mail_result.get("reason", "Email Agent 略過寄送")))
            else:
                st.error(str(mail_result.get("error", "Email Agent 寄送失敗")))
        else:
            st.warning("目前沒有可寄送的盤後研究報告。")

    last_email_result = st.session_state.get("email_send_result")
    if isinstance(last_email_result, dict) and last_email_result.get("sent"):
        st.caption(f"最近 Email Agent 寄送成功：{last_email_result.get('recipient', email_status.get('recipient'))}｜{last_email_result.get('report_date','')}")
    elif isinstance(last_email_result, dict) and last_email_result.get("error"):
        st.warning(f"最近一次 Email Agent：{last_email_result.get('error')}")

    if not isinstance(report, dict):
        st.info("尚未有盤後研究報告；平日 15:00 後按「執行盤後研究」。")
        return

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("重新訓練建議", "是" if report.get("recommend_retrain") else "暫不需要")
    rc2.metric("產業模型數", len(report.get("model_health", [])))
    rc3.metric("新聞樣本", len(report.get("news", [])))
    st.info(report.get("research_conclusion", ""))

    # ========================================================
    # AI 研究助理 Agent 筆記
    # ========================================================
    agent_notes = report.get("agent_research_notes", {}) or {}
    with st.expander("🤖 AI 研究助理 Agent｜自主研究筆記與記號", expanded=True):
        if agent_notes.get("agent_status") == "qwen3":
            st.success(f"Qwen3 8B Agent 已完成自主研究｜工具動作 {len(agent_notes.get('agent_actions', []))} 次")
        else:
            st.warning("本次未完成 Qwen3 自主研究，以下可能為保守 fallback。")

        overview = str(agent_notes.get("overview", "")).strip()
        if overview:
            st.markdown("### 今日研究助理總結")
            st.info(overview)

        takeaways = agent_notes.get("key_takeaways", []) or []
        if takeaways:
            st.markdown("### Agent 自主挑選的重點")
            for item in takeaways:
                st.markdown(f"- {item}")

        notes = agent_notes.get("research_notes", []) or []
        if notes:
            st.markdown("### Agent 研究筆記 / 記號")
            note_rows = []
            for n in notes:
                note_rows.append({
                    "重要性": str(n.get("importance", "medium")).upper(),
                    "類型": n.get("type", ""),
                    "股票": n.get("symbol", ""),
                    "標題": n.get("title", ""),
                    "筆記": n.get("note", ""),
                    "信心": n.get("confidence", ""),
                    "Tags": "、".join(n.get("tags", []) or []),
                })
            st.dataframe(pd.DataFrame(note_rows), use_container_width=True, hide_index=True)

            for idx, n in enumerate(notes):
                title = f"{n.get('symbol','') or '市場'}｜{n.get('title','研究筆記')}｜{n.get('importance','medium')}"
                with st.expander(title, expanded=False):
                    st.write(n.get("note", ""))
                    evidence = n.get("evidence", []) or []
                    if evidence:
                        st.markdown("**Evidence**")
                        for e in evidence:
                            st.write("• " + str(e))
                    tags = n.get("tags", []) or []
                    if tags:
                        st.markdown("**Tags：** " + "、".join(map(str, tags)))
                    follow = n.get("follow_up", []) or []
                    if follow:
                        st.markdown("**Follow-up**")
                        for f in follow:
                            st.write("• " + str(f))
        else:
            st.info("本次研究沒有產生 Agent 研究筆記。")

        if agent_notes.get("earnings_digest"):
            st.markdown("### 法說會 Agent 筆記")
            st.write(agent_notes.get("earnings_digest"))

        if agent_notes.get("watch_items"):
            st.markdown("### 待追蹤 / 待查證")
            for item in agent_notes.get("watch_items", [])[:12]:
                st.write("• " + str(item))

        if agent_notes.get("agent_actions"):
            with st.expander("Agent 實際工具使用紀錄", expanded=False):
                action_rows = []
                for a in agent_notes.get("agent_actions", []):
                    action_rows.append({
                        "工具": a.get("tool", ""),
                        "原因": a.get("reason", ""),
                        "結果摘要": a.get("result", ""),
                    })
                st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

    with st.expander("模型健康 / OOF / 重新訓練檢查", expanded=True):
        rows = []
        flag_map = {x.get("sector_code"): x for x in report.get("retrain_flags", [])}
        for rec in report.get("model_health", []):
            m = rec.get("metrics", {}) or {}
            flag = flag_map.get(rec.get("sector_code"), {})
            rows.append({
                "產業": f"{rec.get('sector_code')} {rec.get('sector_name')}",
                "訓練股票": rec.get("training_stocks"),
                "樣本": rec.get("training_samples"),
                "MAE": m.get("MAE"),
                "RMSE": m.get("RMSE"),
                "方向準確率": m.get("ClassificationDirectionAccuracy"),
                "模型年齡(天)": rec.get("age_days"),
                "重訓研究": "建議" if flag.get("retrain") else "監控",
                "原因": "；".join(flag.get("reasons", [])),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("公司主要業務／產業鏈定位與研究策略", expanded=True):
        company_rows = report.get("company_research", []) or []
        if not company_rows:
            st.info("本次報告尚未產生公司層級研究策略。")
        else:
            view = []
            for row in company_rows:
                view.append({
                    "股票": row.get("symbol"),
                    "公司": row.get("name"),
                    "產業": row.get("industry_name"),
                    "主要業務": row.get("business_group"),
                    "產業鏈位置": row.get("primary_chain"),
                    "相關消息": row.get("related_news_count", 0),
                })
            st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)
            selected = str(st.session_state.get("selected") or "")
            current = next((r for r in company_rows if str(r.get("symbol")) == selected), None)
            if current:
                st.markdown(f"**目前選取：{selected}｜{current.get('business_group','主要業務待建立')}**")
                if current.get("research_keywords"):
                    st.caption("Agent 研究關鍵字：" + "、".join(current.get("research_keywords", [])[:12]))
                if current.get("query_plan"):
                    st.markdown("**本次實際研究問題**")
                    for q in current.get("query_plan", [])[:5]:
                        st.write("• " + str(q))

    # ========================================================
    # 法說會 + 近期漲跌產業分析
    # ========================================================
    with st.expander("法人說明會：法說內容利多／利空 + 法說後股價反應", expanded=True):
        calls = report.get("earnings_calls", []) or []
        calls = [x for x in calls if isinstance(x, dict)]
        if not calls:
            st.info("目前研究包沒有取得可開啟的富果法說會備忘錄；請重新執行盤後研究。系統不會把 MOPS 導覽文字冒充法說摘要。")
        else:
            err_rows = [x for x in calls if x.get("error")]
            data = [x for x in calls if not x.get("error")]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("法說事件", len(data))
            c2.metric("偏利多", sum(str(x.get("impact")) == "偏利多" for x in data))
            c3.metric("偏利空", sum(str(x.get("impact")) == "偏利空" for x in data))
            c4.metric("中性／待確認", sum(str(x.get("impact")) == "中性／待確認" for x in data))
            if err_rows:
                for e in err_rows[:3]:
                    st.warning(str(e.get("error")))
            if data:
                rows = []
                for x in data[:80]:
                    r1 = x.get("post_1d_return")
                    r5 = x.get("post_5d_return")
                    rows.append({
                        "日期": x.get("event_date"),
                        "股票": f"{x.get('name','') or x.get('symbol','')}（{x.get('symbol','')}）",
                        "產業": x.get("industry_name"),
                        "判斷": x.get("impact"),
                        "影響分數": float(x.get("impact_score", 0) or 0),
                        "法說後1日": None if r1 is None else f"{float(r1)*100:+.2f}%",
                        "法說後5日": None if r5 is None else f"{float(r5)*100:+.2f}%",
                        "市場反應": x.get("price_reaction_label"),
                        "證據": x.get("evidence_level"),
                        "摘要": str(x.get("summary", ""))[:180],
                    })
                cdf = pd.DataFrame(rows)
                st.dataframe(cdf, use_container_width=True, hide_index=True)
                st.caption("主要來源固定為 Fugle 法說會備忘錄；Ollama Agent 必須先開啟詳細文章再分析。原文網址僅作來源查閱。")
                for x in data[:12]:
                    label = x.get("impact", "中性／待確認")
                    title = f"{x.get('event_date')}｜{x.get('symbol')}｜{label}"
                    with st.expander(title, expanded=False):
                        if x.get("memo_opened") and x.get("agent_tool_used") and (x.get("one_line_summary") or x.get("summary")):
                            st.markdown("**🤖 Ollama AI Agent 已開啟富果『法說會備忘錄』詳細文章並完成摘要**")
                            st.markdown(f"**一句話摘要：** {x.get('one_line_summary') or x.get('summary','')}")
                            st.markdown(f"**判斷：** `{x.get('impact','混合')}`｜**AI 信心度：** {float(x.get('confidence',0) or 0):.0f}%")
                            for label,key in [("財務重點","financial_highlights"),("營運重點","operating_highlights"),("展望／指引","guidance"),("利多因素","positive_factors"),("利空／風險因素","negative_factors"),("Q&A 重點","qa_highlights")]:
                                vals=x.get(key) or []
                                if vals:
                                    st.markdown(f"**{label}**")
                                    for v in vals[:8]: st.write("• "+str(v))
                            if x.get("memo_sections"):
                                with st.expander("查看法說會重點原文分段", expanded=False):
                                    for sec, body in x.get("memo_sections", {}).items():
                                        st.markdown(f"**{sec}**")
                                        st.write(body)
                            st.markdown(f"**富果備忘錄原文（僅供來源查閱）：** [開啟文章]({x.get('source_url','')})")
                        else:
                            st.warning("目前只有官方法說日程，尚未取得富果法說會備忘錄正文；這筆資料不會冒充摘要。")
                            if x.get("source_url"):
                                st.markdown(f"**官方日程來源：** [開啟]({x.get('source_url','')})")
                        if x.get("positive_terms"):
                            st.write("**利多證據關鍵字：** " + "、".join(x.get("positive_terms", [])[:8]))
                        if x.get("negative_terms"):
                            st.write("**利空證據關鍵字：** " + "、".join(x.get("negative_terms", [])[:8]))
                        st.write(f"**證據程度：** {x.get('evidence_level','')}")
                        st.write(f"**法說後市場反應：** {x.get('price_reaction_label','')}")

    with st.expander("近期上漲／下跌產業個股：1日、5日、20日 + 利多利空原因", expanded=True):
        mm = report.get("market_movers", {}) or {}
        movers = pd.DataFrame(mm.get("movers", []) or [])
        industry_summary = pd.DataFrame(mm.get("industry_summary", []) or [])
        if mm.get("errors"):
            for err in mm.get("errors", [])[:3]:
                st.warning(str(err))
        if movers.empty:
            st.info("目前沒有近期漲跌樣本。請確認盤後 Fugle API 可取得 snapshot 與歷史日K。")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("全市場 snapshot 股票", int(mm.get("universe_count", 0) or 0))
            c2.metric("漲跌候選", int(mm.get("candidate_count", len(movers)) or len(movers)))
            c3.metric("產業樣本數", len(industry_summary))

            period = st.radio("排行依據", ["今日", "5日", "20日"], horizontal=True, key="mover_period")
            period_col = {"今日": "today_change_percent", "5日": "ret_5d", "20日": "ret_20d"}[period]
            view = movers.copy()
            if "today_change_percent" in view:
                view["today_change_percent"] = pd.to_numeric(view["today_change_percent"], errors="coerce")
            if "ret_5d" in view:
                view["ret_5d"] = pd.to_numeric(view["ret_5d"], errors="coerce")
            if "ret_20d" in view:
                view["ret_20d"] = pd.to_numeric(view["ret_20d"], errors="coerce")
            gainers = view.sort_values(period_col, ascending=False).head(12).copy()
            losers = view.sort_values(period_col, ascending=True).head(12).copy()
            st.markdown(f"**近期漲幅較明顯的個股樣本（{period}）**")
            gcols = [c for c in ["symbol", "name", "industry_name", "business_group", "today_change_percent", "ret_5d", "ret_20d", "reason_type", "reason_confidence", "reason"] if c in gainers.columns]
            st.dataframe(gainers[gcols].rename(columns={
                "symbol":"代號", "name":"公司", "industry_name":"產業", "business_group":"主要業務",
                "today_change_percent":"今日漲跌%", "ret_5d":"5日%", "ret_20d":"20日%",
                "reason_type":"原因類型", "reason_confidence":"原因信心", "reason":"可能原因"
            }), use_container_width=True, hide_index=True)
            st.markdown(f"**近期跌幅較明顯的個股樣本（{period}）**")
            lcols = [c for c in ["symbol", "name", "industry_name", "business_group", "today_change_percent", "ret_5d", "ret_20d", "reason_type", "reason_confidence", "reason"] if c in losers.columns]
            st.dataframe(losers[lcols].rename(columns={
                "symbol":"代號", "name":"公司", "industry_name":"產業", "business_group":"主要業務",
                "today_change_percent":"今日漲跌%", "ret_5d":"5日%", "ret_20d":"20日%",
                "reason_type":"原因類型", "reason_confidence":"原因信心", "reason":"可能原因"
            }), use_container_width=True, hide_index=True)

            if not industry_summary.empty:
                ishow = industry_summary.copy()
                for c in ["today_avg_change", "ret_5d_avg", "ret_20d_avg"]:
                    if c in ishow.columns:
                        ishow[c] = pd.to_numeric(ishow[c], errors="coerce")
                ishow = ishow.head(20)
                st.markdown("**產業強弱彙整**")
                st.dataframe(ishow.rename(columns={
                    "industry_code":"代碼", "industry_name":"產業", "candidate_count":"樣本股數",
                    "today_avg_change":"今日平均%", "ret_5d_avg":"5日平均%", "ret_20d_avg":"20日平均%",
                    "rising_count":"上漲數", "falling_count":"下跌數", "top_risers":"上漲樣本", "top_fallers":"下跌樣本"
                }), use_container_width=True, hide_index=True)

            st.caption(mm.get("method", ""))
            st.caption("近期漲跌的『可能原因』先由公司主要業務 + 新聞關鍵字建立候選原因；沒有官方公告、法說或財報交叉驗證時，請視為待查證而非因果結論。")

    with st.expander("產業利多 / 利空與新聞查證", expanded=False):
        for s in report.get("sector_news", []):
            st.markdown(f"**{s['code']} {s['name']}**｜消息 {s['news_count']}｜情緒 {s['weighted_sentiment']:+.2f}｜證據信心 {s['confidence']:.0%}")
            st.write(f"利多：{s['dominant_opportunity']}")
            st.write(f"風險：{s['dominant_risk']}")
        news_df = pd.DataFrame(report.get("news", []))
        if not news_df.empty:
            cols = [c for c in ["title", "source", "published", "sentiment_score", "corroboration", "official_match", "verification", "link"] if c in news_df.columns]
            st.dataframe(news_df[cols].head(50), use_container_width=True, hide_index=True)

    with st.expander("利多題材雷達：哪些題材正在帶動產業發展", expanded=True):
        themes = report.get("bullish_themes", []) or []
        if not themes:
            st.info("目前沒有足夠資料辨識明確的利多題材。")
        else:
            rows = [{
                "題材": t.get("theme"),
                "題材強度": float(t.get("strength", 0)),
                "狀態": t.get("status"),
                "涉及公司": "、".join(
                    f"{c.get('name','')}（{c.get('symbol','')}）"
                    for c in (t.get("affected_companies") or [])
                ) or "無",
                "涉及自選股": int(t.get("affected_count", 0) or 0),
                "正向消息": int(t.get("positive_news_count", 0) or 0),
                "官方佐證": int(t.get("official_evidence_count", 0) or 0),
                "接單/出貨證據": int(t.get("order_evidence_count", 0) or 0),
                "營收支撐家數": int(t.get("revenue_support_count", 0) or 0),
            } for t in themes[:12]]
            tdf = pd.DataFrame(rows)
            fig = px.bar(tdf.sort_values("題材強度"), x="題材強度", y="題材", orientation="h",
                         hover_data=["涉及公司", "狀態", "涉及自選股", "正向消息", "官方佐證", "接單/出貨證據", "營收支撐家數"],
                         title="利多題材強度（研究訊號，不是投資評級）")
            fig.update_layout(height=max(360, 44 * len(tdf)), margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False})
            st.dataframe(tdf, use_container_width=True, hide_index=True)
            st.caption("介紹利多題材時同步列出受影響的自選股公司、主要業務與產業鏈位置，方便從題材一路追到實際受惠企業。")
            for t in themes[:8]:
                title = f"{t.get('theme')}｜{float(t.get('strength',0)):.1f}/100｜{t.get('status')}"
                with st.expander(title, expanded=False):
                    companies = t.get("affected_companies") or []
                    if companies:
                        st.markdown("**相關公司：**")
                        company_rows = [{
                            "公司": f"{c.get('name','')}（{c.get('symbol','')}）",
                            "主要業務": c.get("business_group", ""),
                            "產業鏈位置": c.get("primary_chain", ""),
                        } for c in companies]
                        st.dataframe(pd.DataFrame(company_rows), use_container_width=True, hide_index=True)
                    else:
                        syms = "、".join(str(x) for x in (t.get("affected_symbols") or [])[:15]) or "無"
                        st.markdown(f"**相關自選股：** {syms}")
                    drivers = t.get("bullish_drivers") or []
                    if drivers:
                        st.markdown("**支持題材的證據：**")
                        for d in drivers[:6]:
                            st.write("• " + str(d))
                    contradictions = t.get("contradiction_flags") or []
                    if contradictions:
                        st.markdown("**需要反證／查證：**")
                        for c in contradictions[:6]:
                            st.write("• " + str(c))
                    evidence = t.get("supporting_evidence") or []
                    if evidence:
                        st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)
                    queries = t.get("research_queries") or []
                    if queries:
                        st.caption("Agent 後續研究問題：" + "；".join(queries[:5]))

    llm = report.get("ollama_analysis", {}) or {}
    with st.expander("Qwen3 本機研究摘要", expanded=True):
        if llm.get("available"):
            st.markdown(llm.get("content") or "未產生摘要。")
        else:
            st.warning(llm.get("error", "Qwen3 尚不可用；仍可查看規則式研究結果。"))

    st.caption(
        "新聞『疑似假利多/假利空』不做絕對斷言；系統優先使用官方公告、法說會、營收與法人資料交叉驗證，"
        "若消息與基本面或官方資料不一致，標示為『待查證／疑似誇大／市場反應與基本面不一致』。"
    )
    st.caption(
        "報告檔案：output/research_reports/research_YYYY-MM-DD.json / .md；Email Agent 會依 .env 設定自動寄送盤後財報；每次執行都會寄送，data/email/sent_YYYY-MM-DD.json 僅作寄送紀錄；研究 Agent 只提出重訓與調參建議，不自動修改 production model。"
    )



st.caption("資料時間統一依台灣時間處理；盤中現貨資料只顯示 09:00:00～13:30:00。")

# ============================================================
# 一般市場與交易資訊區
# ============================================================
# 依使用者操作流程，先呈現看盤、K線、進場、籌碼與交易資訊。
# AI 研究相關的三個重量級區塊統一移到頁面最後三項。
# 使用者可手動強制清除即時資料快取。
_refresh_col, _status_col = st.columns([1, 5])
with _refresh_col:
    if st.button("🔄 立即更新行情", use_container_width=True):
        _manual_live_refresh()
        st.rerun()
with _status_col:
    if is_market_open_now():
        st.caption(
            f"即時模式：每 {REFRESH_INTERVAL} 秒刷新｜"
            "報價、成交、盤中 K 線、加權指數會優先更新；法人/公司基本資料依各自 TTL 更新。"
        )
    else:
        st.caption("目前非交易時段；收盤資料不會每秒變動，仍可使用「立即更新行情」重新取得最新可得資料。")

render_watchlist()
render_selected_and_live()
render_trend()
render_kline()
render_entry_prediction()
render_tdcc()
render_trades()
render_institutions()

# ============================================================
# 最後三項：AI 研究區
# ============================================================
# AI 隔日預測仍是「優先任務」：按下後先執行高運算任務，完成後 st.rerun。
# 但版面位置依需求固定在 Dashboard 最後三項。
render_ai()

# 市場情報與利多 / 利空分析
_selected_symbol = clean_symbol(st.session_state.get("selected", ""))
_selected_industry_code = str(INDUSTRY_OVERRIDES.get(_selected_symbol, WATCHLIST_INDUSTRY_FALLBACK.get(_selected_symbol, "00"))).zfill(2)
render_market_intelligence(
    BASE, OUTPUT,
    selected_symbol=_selected_symbol,
    selected_industry=_selected_industry_code,
    blocked=st.session_state.get("ai_busy", False),
)

# 模型研究 Agent
render_model_research_agent()

# ============================================================
# Stable full-page refresh
# ============================================================
# Streamlit Cloud 不會因為畫面停著就自動重新執行整份 Python；
# 這裡使用 st_autorefresh 在交易時段建立固定週期的 browser-side rerun。
# 即時資料函式本身再用短 TTL（2~5 秒），避免同一次 rerun 讀到舊快取。
if REFRESH_INTERVAL and not st.session_state.get("ai_busy", False):
    st_autorefresh(
        interval=int(REFRESH_INTERVAL * 1000),
        key="market_live_refresh",
    )

# 顯示目前資料刷新狀態，避免使用者不知道頁面到底有沒有在更新。
if is_market_open_now():
    st.caption(
        f"🟢 即時行情更新中｜每 {REFRESH_INTERVAL} 秒自動刷新｜"
        f"最後頁面刷新：{taiwan_now().strftime('%H:%M:%S')}"
    )
else:
    st.caption(
        f"⚪ 目前非台股交易時段｜即時行情不進行高頻輪詢｜"
        f"頁面檢查時間：{taiwan_now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
