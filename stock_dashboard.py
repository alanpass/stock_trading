# -*- coding: utf-8 -*-
"""AI 台股即時互動式分析系統 v10

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
import re
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh


def _load_cloud_secret_env() -> None:
    """將 Streamlit Community Cloud Secrets 同步成環境變數。

    本機仍可使用原本的 stock_api.py；部署到 Cloud 時，
    可在 App Settings -> Secrets 設定 FUGLE_API_KEY。
    """
    try:
        secret_keys = [
            "FUGLE_API_KEY",
            "EMAIL_ENABLED",
            "EMAIL_AUTO_SEND",
            "EMAIL_SENDER",
            "EMAIL_APP_PASSWORD",
            "EMAIL_RECIPIENT",
            "EMAIL_SMTP_HOST",
            "EMAIL_SMTP_PORT",
            "EMAIL_USE_SSL",
            "EMAIL_ATTACH_REPORT",
            "SUBSCRIBER_SYNC_URL",
            "SUBSCRIBER_SYNC_TOKEN",
        ]
        for name in secret_keys:
            try:
                value = st.secrets.get(name)
            except Exception:
                value = None
            if value not in (None, ""):
                os.environ[name] = str(value)
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
from subscriber_service import register_subscriber, subscription_status
from industry_rotation import ROTATION_THEMES, ROTATION_DEFAULT_THEMES, ROTATION_DISPLAY_NAMES, NAME_FALLBACKS, load_published_rotation

BASE = Path(__file__).resolve().parent
DATA, MODELS, OUTPUT = BASE / "data", BASE / "models", BASE / "output"
DAILY_DATA = DATA / "daily"
SITE_IMAGE_DIR = BASE / "assets" / "site_images"
for d in (DATA, MODELS, OUTPUT, DAILY_DATA, SITE_IMAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

WATCHLIST_DEFAULT = [
    "3481", "2327", "2492", "3037", "3044", "3533", "2303", "2330", "2344", "2408",
    "6515", "7769", "6488", "3374", "2377", "3450", "0050", "3006", "3661", "8299", "2308", "6274"
]
# 啟動時若尚未建立全市場產業主檔，仍可先正確顯示預設自選股的分類；
# 完整上市櫃產業主檔只在搜尋／AI 等需要時再載入。
WATCHLIST_INDUSTRY_FALLBACK = {
    "3481": "26", "2327": "28", "2492": "28", "3037": "28", "3044": "28",
    "3533": "28", "2303": "24", "2330": "24", "2344": "24", "2408": "24",
    "6515": "24", "7769": "24", "6488": "24", "3374": "24", "2377": "25",
    "3450": "26", "0050": "00", "3006": "24", "3661": "24", "8299": "24", "2308": "28", "6274": "28",
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



def _site_favicon():
    """瀏覽器分頁圖示：使用網站 logo（assets/logo/favicon.png）；找不到檔案時退回 emoji。"""
    try:
        from PIL import Image
        icon = BASE / "assets" / "logo" / "favicon.png"
        if icon.exists():
            return Image.open(icon)
    except Exception:
        pass
    return "📈"


st.set_page_config(page_title="台股即時互動式分析系統", page_icon=_site_favicon(), layout="wide", initial_sidebar_state="expanded")


# ---- 所有 Plotly 圖表統一使用與網頁背景相配的淺色系 ----
_plotly_chart_orig = st.plotly_chart
def _light_plotly_chart(fig, *args, **kwargs):
    try:
        fig.update_layout(
            template="plotly_white",
            title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.55)",
            font=dict(color="#34404a"),
            title_font=dict(color="#8e2b2f"),
            colorway=["#6c8ba3", "#8e2b2f", "#9cb0c0", "#c98f8f", "#58758e", "#b7c4cf"],
            xaxis=dict(gridcolor="#dfe5ea", linecolor="#c6d0d8", zerolinecolor="#c6d0d8"),
            yaxis=dict(gridcolor="#dfe5ea", linecolor="#c6d0d8", zerolinecolor="#c6d0d8"),
        )
    except Exception:
        pass
    kwargs.setdefault("theme", None)
    return _plotly_chart_orig(fig, *args, **kwargs)
st.plotly_chart = _light_plotly_chart

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

        # 只要同時有現價與昨日參考價，就以價格重新計算漲跌額與漲跌幅。
        # 不直接沿用某些 API payload 的 changePercent，避免單位不同造成
        # +112%、-123% 等不可能的顯示。
        api_change = pd.to_numeric(x.get("change", np.nan), errors="coerce")
        api_change_pct = pd.to_numeric(x.get("changePercent", np.nan), errors="coerce")
        if pd.notna(p) and pd.notna(prev) and float(prev) != 0:
            ch = p - prev
            cp = ch / prev * 100
        else:
            ch = api_change
            cp = api_change_pct
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
.live-card{position:relative;width:100%;min-height:96px;padding:9px 10px;background:rgba(15,23,42,.97);border:1px solid rgba(148,163,184,.45);border-radius:12px;box-shadow:0 6px 22px rgba(0,0,0,.30);z-index:999999;font-family:Arial,"Microsoft JhengHei",sans-serif;}
.live-grid{display:grid;grid-template-columns:1fr 1fr;gap:0}.live-cell{padding:0 10px;min-width:0}.live-cell + .live-cell{border-left:1px solid rgba(148,163,184,.25)}
.live-title{font-size:10px;color:#94a3b8;margin-bottom:2px}.live-main{font-size:14px;font-weight:800;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.live-price{font-size:22px;line-height:1.03;font-weight:900;color:#f8fafc;margin-top:2px}.live-change{font-size:12px;font-weight:800;margin-top:3px}.live-time{font-size:9px;color:#64748b;margin-top:3px}
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
    padding-top:0 !important;
    padding-bottom:0 !important;
  }


  /* 標題與搜尋列在手機改為上下排列。 */
  .title-search-row{display:block !important;}
  [data-testid="stTextInput"] input{
    height:44px !important;
    font-size:16px !important;
  }
  button[title='搜尋股票代號或名稱']{
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

  /* 圖表、表格只保留必要的橫向捲動，不限制欄位本身寬度。 */
  .stPlotlyChart,
  [data-testid="stDataFrame"],
  [data-testid="stTable"]{
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
    box-sizing:border-box !important;
  }
  .stButton > button{
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
    padding-top:0 !important;
    padding-bottom:0 !important;
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


# ============================================================
# 新版研究中心視覺：參照入口型股票網站的資訊架構
# ============================================================
st.markdown(r"""
<style>
:root{--brand:#0f172a;--brand2:#1e293b;--accent:#2563eb;--muted:#64748b;--line:#e2e8f0;--up:#ef4444;--down:#10b981;}
body{background:#f5f7fb;}
.main .block-container{padding-top:0!important;}
.dashboard-hero{background:linear-gradient(135deg,#0f172a 0%,#172554 56%,#1d4ed8 100%);color:#fff;border-radius:22px;padding:28px 30px 24px;margin:0 0 16px;box-shadow:0 14px 34px rgba(15,23,42,.16);position:relative;overflow:hidden;}
.dashboard-hero:after{content:"";position:absolute;right:-85px;top:-95px;width:290px;height:290px;border-radius:50%;background:rgba(255,255,255,.07);box-shadow:-110px 115px 0 rgba(255,255,255,.035);}
.dashboard-hero-kicker{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#bfdbfe;font-weight:800;position:relative;z-index:2;}
.dashboard-hero-title{font-size:34px;font-weight:950;line-height:1.12;margin:7px 0;position:relative;z-index:2;}
.dashboard-hero-sub{font-size:14px;color:#dbeafe;max-width:900px;line-height:1.65;position:relative;z-index:2;}
.dashboard-hero-pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;position:relative;z-index:2;}
.dashboard-pill{background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.16);padding:7px 11px;border-radius:999px;font-size:12px;color:#eff6ff;}
.search-wrap{background:#fff;border:1px solid #dbe2ea;border-radius:15px;padding:10px 12px 8px;margin:10px 0 14px;box-shadow:0 5px 18px rgba(15,23,42,.05);}
.search-help{font-size:11px;color:#64748b;margin:4px 2px 0;}
.info-strip{background:#f8fafc;border:1px solid #e2e8f0;border-radius:13px;padding:10px 13px;margin:8px 0 12px;color:#475569;font-size:13px;}
.section-nav-label{font-size:11px;font-weight:900;color:#94a3b8;letter-spacing:1.4px;margin:10px 2px 5px;text-transform:uppercase;}
[data-baseweb="tab-list"]{gap:8px!important;background:transparent!important;border-bottom:1px solid #dbe2ea!important;padding-bottom:3px!important;}
[data-baseweb="tab"]{border-radius:11px 11px 0 0!important;font-weight:850!important;padding:11px 16px!important;color:#475569!important;}
[data-baseweb="tab"][aria-selected="true"]{color:#1d4ed8!important;background:#eff6ff!important;}
.section-shell{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px 18px 9px;margin:10px 0 14px;box-shadow:0 6px 20px rgba(15,23,42,.05);}
.section-title{font-size:23px;font-weight:950;color:#0f172a;margin-bottom:4px;}
.section-sub{font-size:13px;color:#64748b;line-height:1.6;}
.overview-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0;}
.overview-card{background:linear-gradient(180deg,#fff,#f8fafc);border:1px solid #e2e8f0;border-radius:14px;padding:14px 15px;min-height:90px;}
.overview-label{font-size:11px;color:#64748b;font-weight:800;letter-spacing:.4px;}
.overview-value{font-size:21px;font-weight:950;color:#0f172a;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.overview-note{font-size:11px;color:#94a3b8;margin-top:4px;}
.feature-card-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0;}
.feature-card{background:#fff;border:1px solid #e2e8f0;border-radius:15px;padding:16px;box-shadow:0 5px 16px rgba(15,23,42,.04);}
.feature-card h3{margin:0 0 7px;color:#0f172a;font-size:17px;}
.feature-card p{margin:0;color:#64748b;font-size:13px;line-height:1.6;}
.future-banner{background:linear-gradient(135deg,#eff6ff,#eef2ff);border:1px solid #bfdbfe;border-radius:17px;padding:17px 18px;margin:4px 0 14px;}
.future-banner-title{font-size:22px;font-weight:950;color:#1e3a8a;}
.future-banner-sub{font-size:13px;color:#475569;margin-top:3px;line-height:1.55;}
@media(max-width:900px){.overview-grid,.feature-card-grid{grid-template-columns:1fr 1fr}.dashboard-hero-title{font-size:28px}.main .block-container{padding-left:14px!important;padding-right:14px!important;}}
@media(max-width:600px){.overview-grid,.feature-card-grid{grid-template-columns:1fr}.dashboard-hero{padding:20px 18px;border-radius:18px}.dashboard-hero-title{font-size:24px}.main .block-container{padding-left:10px!important;padding-right:10px!important}.search-wrap{padding:9px}.section-shell{padding:15px 14px 8px}.section-title{font-size:20px}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Site design overrides - 依 PDF 設計稿的視覺系統
# ============================================================
st.markdown(r'''<style>
:root{--site-red:#8e2b2f;--site-red-dark:#6f2024;--site-ink:#252525;--site-muted:#787878;--site-line:#d8cfcf;--site-blush:#fbf1f1;--site-footer:#242424;}
body,.stApp{background:#fff!important;color:var(--site-ink)!important;}
.main .block-container{padding-top:0!important;padding-bottom:0!important;}
header[data-testid="stHeader"]{background:rgba(255,255,255,.96)!important;min-height:0!important;}
#MainMenu{visibility:hidden!important;}
footer:not(.site-footer){display:none!important;}

.brand-block{text-align:center;padding:0 0 0}.brand-mark{width:54px;height:54px;margin:0 auto 8px;position:relative}.brand-mark:before{content:"";position:absolute;inset:8px;border:4px solid var(--site-red);border-radius:50% 50% 46% 54%;transform:rotate(-22deg)}.brand-mark:after{content:"";position:absolute;width:23px;height:15px;border-left:4px solid var(--site-red);border-bottom:4px solid var(--site-red);left:17px;top:21px;transform:rotate(32deg);border-radius:0 0 0 8px}.brand-mark span{position:absolute;width:12px;height:12px;background:#fff;left:15px;top:16px;border-radius:50%;z-index:2}.brand-name{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:20px;letter-spacing:1.5px;color:#555}.site-status-line{text-align:center;font-size:10px;color:#989898;margin:4px 0 10px;letter-spacing:.4px}.site-nav{display:flex;justify-content:center;align-items:center;gap:28px;flex-wrap:wrap;padding:8px 4px 12px}.site-nav-link{font-family:"Noto Sans TC","Microsoft JhengHei",sans-serif;color:#777!important;text-decoration:none!important;font-size:11px;letter-spacing:.6px;padding:3px 0;border-bottom:1px solid transparent;transition:.18s ease}.site-nav-link:hover{color:var(--site-red)!important}.site-nav-link.active{color:var(--site-red)!important;border-bottom-color:var(--site-red);font-weight:800}.site-rule{border-top:1px dashed #a9a1a1;margin:1px 0 20px}
.page-image-banner{position:relative;width:100%;height:180px;border-radius:10px;overflow:hidden;margin:0 0 24px;box-shadow:0 8px 24px rgba(0,0,0,.08)}.page-image-banner>img{display:block;width:100%;height:100%;object-fit:cover}.page-image-shade{position:absolute;inset:0;background:rgba(15,15,20,.42)}.page-image-title{position:absolute;left:34px;bottom:24px;color:#fff;font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:28px;font-weight:500;letter-spacing:1px}.contact-page-banner{margin-bottom:20px}.hero-visual{width:100%;height:350px;border-radius:10px;overflow:hidden;margin:0 0 0;box-shadow:0 8px 25px rgba(0,0,0,.08)}
.home-band{padding:48px 42px;margin:0}.home-band.blush{background:var(--site-blush)}.home-band.white-band{background:#fff}.home-two-col{width:100%;margin:0;display:grid;grid-template-columns:1.1fr .9fr;gap:60px;align-items:center}.home-two-col.reverse-on-mobile{grid-template-columns:.92fr 1.08fr}.home-copy{max-width:540px}.eyebrow{color:var(--site-red);font-size:10px;letter-spacing:2px;font-weight:800;margin-bottom:7px}.home-copy h2,.content-heading h1{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;color:var(--site-red);font-size:32px;line-height:1.2;margin:0 0 14px;font-weight:500}.home-copy p,.content-heading p{font-size:13px;line-height:1.9;color:#5f5a5a;margin:0 0 10px}.home-link,.contact-link{display:inline-block;margin-top:8px;color:var(--site-red)!important;text-decoration:none!important;font-size:12px;font-weight:800;letter-spacing:.5px;border-bottom:1px solid rgba(142,43,47,.45);padding-bottom:3px}.home-image-stack{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:end}.small-visual{height:170px;border-radius:8px;overflow:hidden;box-shadow:0 5px 14px rgba(0,0,0,.12)}.home-image-grid{display:grid;gap:10px}.home-image-grid.three-grid{grid-template-columns:repeat(3,1fr)}.home-image-grid.six-grid{grid-template-columns:repeat(3,1fr)}.home-image-grid>div,.focus-mini{min-width:0;border-radius:8px;overflow:hidden;box-shadow:0 5px 12px rgba(0,0,0,.09)}.home-image-grid.three-grid>div{height:142px}.home-image-grid.six-grid>div{height:118px}.home-accordion-hints{margin:14px 0 8px;border-top:1px solid #ded4d4}.home-accordion-hints>div{display:flex;justify-content:space-between;gap:16px;padding:12px 2px;border-bottom:1px solid #ded4d4;font-size:12px}.home-accordion-hints span{color:#8b8282}
.contact-band{position:relative;min-height:280px;overflow:hidden;margin:0}.contact-overlay{position:absolute;inset:0}.contact-overlay svg,.contact-overlay img{width:100%;height:100%;display:block;object-fit:cover;}.contact-shade{position:absolute;inset:0;background:rgba(20,20,20,.76)}.contact-content{position:relative;z-index:2;max-width:780px;margin:0 auto;padding:44px 42px;color:#fff}.contact-content .eyebrow{color:#cf9494}.contact-content h2{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:30px;font-weight:500;margin:0 0 14px;color:#fff}.contact-content p{font-size:13px;line-height:1.9;color:#ddd;margin:0;max-width:660px}.contact-link{color:#e8b2b2!important;border-color:rgba(232,178,178,.4)}
.content-heading{padding:8px 0 22px;max-width:760px}.content-heading h1{font-size:34px;margin-bottom:8px}.feature-section-title{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:22px;color:var(--site-red);font-weight:500;border-bottom:1px solid var(--site-line);padding:8px 0 10px;margin:36px 0 18px}.selected-strip{display:flex;gap:7px;align-items:center;background:var(--site-blush);border-left:3px solid var(--site-red);padding:10px 13px;font-size:12px;color:#766f6f;margin:12px 0 20px}.selected-strip strong{color:var(--site-red);font-size:15px}.data-panel-heading,.finance-card-title,.form-title{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;color:var(--site-red);font-size:20px;margin:20px 0 8px}.search-code-site{color:var(--site-red);font-weight:900;font-size:16px}.search-name-site{font-size:13px;color:#3e3a3a;padding:9px 0}.search-name-site span{color:#8d8585;margin-left:4px}.finance-card-sub{color:#8c8585;font-size:11px;margin-bottom:12px}.finance-refresh-time{font-size:11px;color:#7f6d6d;margin:0 0 12px;text-align:right}
[data-testid="stDataFrame"],[data-testid="stTable"],[data-testid="stExpander"]{border-color:#ddd4d4!important}[data-testid="stExpander"] summary{font-size:13px!important;color:#5b5555!important}.stButton>button{border:1px solid #c9bbbb!important;background:#fff!important;color:#6d5b5b!important;border-radius:3px!important;box-shadow:none!important}.stButton>button:hover{border-color:var(--site-red)!important;color:var(--site-red)!important}[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{border-radius:2px!important;border:0!important;border-bottom:1px solid #cfc7c7!important;background:#fff!important;box-shadow:none!important}[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-bottom-color:var(--site-red)!important;box-shadow:none!important}
.contact-info-card{background:#fff;color:#4a4545;padding:6px 0;min-height:360px}.contact-info-card h2{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:28px;color:var(--site-red);font-weight:500;margin:0 0 10px}.contact-info-card p{color:#777;font-size:12px;line-height:1.8;max-width:470px}.contact-item{display:flex;gap:12px;align-items:flex-start;padding:12px 0;border-top:1px solid #e2dada}.contact-item span{color:var(--site-red);font-size:18px;line-height:1}.contact-item b{display:block;font-size:12px;margin-bottom:3px;color:#4a4545}.contact-item small{display:block;color:#8c8585;font-size:11px;line-height:1.6}.form-title{margin-top:0;font-size:21px}.subscribe-note{margin:8px 0 12px;color:var(--site-red);font-size:12px;line-height:1.6}.stFormSubmitButton>button{background:#252525!important;color:#fff!important;border-color:#252525!important;border-radius:16px!important;padding:6px 22px!important}.site-footer{background:var(--site-footer);color:#eee;margin:34px 0 0;padding:0 0 0;width:100%}.footer-inner{width:100%;margin:0;display:grid;grid-template-columns:1fr 1fr;gap:50px;padding:38px 42px 30px}.footer-contact,.footer-form-note{padding:4px 0}.footer-title{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;color:#d68f8f;font-size:16px;margin-bottom:8px}.footer-contact p,.footer-form-note p{color:#bdbdbd;font-size:11px;line-height:1.8;margin:0 0 10px}.footer-small{color:#8f8f8f;font-size:10px;line-height:1.8}.footer-form-note a{color:#d7a2a2!important;text-decoration:none!important;font-size:11px}.footer-bottom{border-top:1px solid rgba(255,255,255,.08);text-align:center;padding:14px 12px 18px;color:#777;font-size:9px;letter-spacing:.5px}
@media(max-width:900px){.main .block-container{padding-left:16px!important;padding-right:16px!important}.site-nav{gap:18px}.home-band,.site-footer,.contact-band{margin-left:0;margin-right:0}.home-band{padding:42px 22px}.home-two-col,.home-two-col.reverse-on-mobile{grid-template-columns:1fr;gap:28px}.home-copy{max-width:none}.footer-inner{padding:30px 22px;grid-template-columns:1fr;gap:24px}}
@media(max-width:600px){.main .block-container{padding-left:10px!important;padding-right:10px!important}.brand-name{font-size:17px}.site-nav{gap:14px 18px;padding-bottom:10px}.site-nav-link{font-size:11px}.hero-visual{height:220px;margin-bottom:0}.page-image-banner{height:150px;margin-bottom:18px}.page-image-title{left:18px;bottom:18px;font-size:24px}.home-band,.site-footer,.contact-band{margin-left:0;margin-right:0}.home-band{padding:28px 16px}.home-copy h2,.content-heading h1{font-size:26px}.home-image-grid.three-grid{grid-template-columns:1fr 1fr 1fr}.home-image-grid.six-grid{grid-template-columns:1fr 1fr}.home-image-grid.three-grid>div{height:108px}.home-image-grid.six-grid>div{height:96px}.small-visual{height:128px}.feature-section-title{font-size:18px}.footer-inner{padding:26px 18px}}
</style>''', unsafe_allow_html=True)


# 網站圖片固定槽位：
# 0~6：首頁專用；7：操作策略頁；8：未來分析頁；11：訂閱／聯絡頁。
# 財經資訊頁依設計要求不放裝飾圖。
SITE_IMAGE_FILES = [
    "01_generated_hero.jpg",          # 0 首頁：主視覺
    "02_generated_laptop.jpg",        # 1 首頁：關於台股
    "03_generated_news.jpg",          # 2 首頁：關於台股
    "04_generated_strategy.jpg",      # 3 首頁：每日財經資訊
    "05_generated_ai.jpg",            # 4 首頁：每日財經資訊
    "06_generated_coins.jpg",         # 5 首頁：每日財經資訊
    "07_original_market_chart.jpg",   # 6 首頁：專注台股左半部大圖
    "08_original_data_analysis.jpg",  # 7 操作策略頁 Banner
    "09_original_kline_detail.jpg",   # 8 未來分析頁 Banner
    "10_original_candle_focus.jpg",   # 9 備用
    "11_original_market_numbers.jpg", # 10 備用
    "12_original_alt_chart.jpg",      # 11 訂閱／聯絡頁 Banner
]

def _find_site_image(slot: int) -> Path | None:
    """依固定槽位讀取網站圖片；Cloud 與本機使用相同順序。"""
    try:
        if not SITE_IMAGE_DIR.exists():
            return None
        name = SITE_IMAGE_FILES[int(slot) % len(SITE_IMAGE_FILES)]
        path = SITE_IMAGE_DIR / name
        return path if path.exists() else None
    except Exception:
        return None

# 內嵌圖片備援：即使只把 stock_dashboard.py 單檔部署到 Streamlit Cloud，首頁仍會保留使用者提供的六張圖片。
_EMBEDDED_SITE_IMAGES = {
    0: "data:image/jpeg;base64,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",
    1: "data:image/jpeg;base64,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",
    2: "data:image/jpeg;base64,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",
    3: "data:image/jpeg;base64,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",
    4: "data:image/jpeg;base64,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",
    5: "data:image/jpeg;base64,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",
}

def _site_image_html(slot: int, wide: bool = False, dark: bool = False) -> str:
    """優先讀取 assets/site_images 的 10 個固定素材；若不存在則回退到內嵌舊素材。"""
    import base64
    try:
        img = _find_site_image(slot)
        if img is not None:
            mime = {".png":"image/png", ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".webp":"image/webp", ".gif":"image/gif"}[img.suffix.lower()]
            data = base64.b64encode(img.read_bytes()).decode("ascii")
            return f'<img src="data:{mime};base64,{data}" alt="台股市場視覺素材" style="display:block;width:100%;height:100%;object-fit:cover;" />'
        data_uri = _EMBEDDED_SITE_IMAGES.get(int(slot) % len(_EMBEDDED_SITE_IMAGES))
        if data_uri:
            return f'<img src="{data_uri}" alt="台股市場視覺素材" style="display:block;width:100%;height:100%;object-fit:cover;" />'
    except Exception:
        pass
    return _stock_svg(slot, wide=wide, dark=dark)


def _stock_svg(variant: int = 0, wide: bool = False, dark: bool = False) -> str:
    """建立不依賴外部圖片的台股視覺素材，避免 Cloud 外部圖片失效。"""
    bg = "#161616" if dark else ("#111820" if variant % 2 == 0 else "#1a2029")
    grid = "#334155" if dark else "#2c3746"
    candle_up = "#c94f4f"
    candle_down = "#5fb89d"
    line = "#d7c0c0" if dark else "#9ca3af"
    w, h = (1060, 420) if wide else (520, 330)
    points = [
        (42, 250, 72, 214, 27), (108, 228, 148, 245, 25), (172, 242, 212, 182, 38),
        (236, 202, 276, 218, 27), (300, 235, 340, 145, 55), (364, 166, 404, 188, 31),
        (428, 192, 468, 128, 48), (492, 136, 532, 160, 30), (556, 169, 596, 108, 48),
        (620, 118, 660, 138, 33), (684, 145, 724, 94, 42), (748, 104, 788, 124, 30),
    ]
    if not wide:
        points = [(x * 0.78, y * 0.86, x2 * 0.78, y2 * 0.86, bh * 0.86) for x, y, x2, y2, bh in points]
    candles = []
    for i, (x, y, x2, y2, body_h) in enumerate(points):
        color = candle_up if i % 3 else candle_down
        wick_top = min(y, y2) - 20
        wick_bottom = max(y, y2) + body_h * 0.35
        body_y = min(y2, y)
        body_top = max(18, min(body_y, 270))
        candles.append(
            f'<line x1="{x+10}" y1="{wick_top}" x2="{x+10}" y2="{wick_bottom}" stroke="{color}" stroke-width="3"/>'
            f'<rect x="{x}" y="{body_top}" width="20" height="{max(18, body_h*0.55):.1f}" rx="2" fill="{color}"/>'
        )
    vols = []
    for i, x in enumerate(range(46, 820, 48)):
        vh = 32 + (i * 17 % 76)
        color = candle_up if i % 3 else candle_down
        vols.append(f'<rect x="{x}" y="354" width="22" height="{vh}" rx="2" fill="{color}" opacity="0.75"/>')
    return f'''
    <svg viewBox="0 0 {w} {h}" width="100%" height="100%" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="台股 K 線與成交量視覺">
      <rect width="{w}" height="{h}" rx="18" fill="{bg}"/>
      <g opacity="0.45" stroke="{grid}" stroke-width="1">
        <line x1="0" y1="72" x2="{w}" y2="72"/><line x1="0" y1="142" x2="{w}" y2="142"/>
        <line x1="0" y1="212" x2="{w}" y2="212"/><line x1="0" y1="282" x2="{w}" y2="282"/>
        <line x1="100" y1="0" x2="100" y2="{h}"/><line x1="220" y1="0" x2="220" y2="{h}"/>
        <line x1="340" y1="0" x2="340" y2="{h}"/><line x1="460" y1="0" x2="460" y2="{h}"/>
        <line x1="580" y1="0" x2="580" y2="{h}"/><line x1="700" y1="0" x2="700" y2="{h}"/>
        <line x1="820" y1="0" x2="820" y2="{h}"/>
      </g>
      <path d="M 14 290 C 120 260, 165 274, 240 242 S 365 250, 450 194 S 585 214, 675 158 S 792 174, 904 126 S 1010 124, 1040 90" fill="none" stroke="{line}" stroke-width="3" opacity="0.9"/>
      <g>{''.join(candles)}</g>
      <g>{''.join(vols)}</g>
      <text x="26" y="36" font-family="Arial, sans-serif" font-size="15" fill="#f8fafc" opacity="0.86">TAIWAN STOCK · MARKET VIEW</text>
      <text x="26" y="56" font-family="Arial, sans-serif" font-size="11" fill="#cbd5e1" opacity="0.72">PRICE / VOLUME / MOMENTUM</text>
    </svg>'''


def _site_header_svg() -> str:
    return '''
    <div class="brand-block" aria-label="台股即時互動式分析系統">
      <div class="brand-mark"><span></span></div>
      <div class="brand-name">台股即時互動式分析系統</div>
    </div>
    '''


def render_header_and_search() -> None:
    """依設計文件統一呈現所有頁面的頁首品牌區。"""
    now = taiwan_now()
    st.markdown(_site_header_svg(), unsafe_allow_html=True)

def render_strategy_search() -> None:
    """操作策略頁的股票查詢；選取後同步整個操作策略與未來分析。"""
    st.markdown('<div class="content-heading"><div class="eyebrow">STOCK OPERATION</div><h1>操作策略</h1><p>從自選股即時行情到 K 線、量能、籌碼與法人資料，集中處理單一分析標的。</p></div>', unsafe_allow_html=True)
    left, right = st.columns([8.2, 1.35], gap="small", vertical_alignment="bottom")
    with left:
        search_text = st.text_input(
            "股票查詢",
            placeholder="輸入股票代號或公司名稱，例如 2330、台積電",
            label_visibility="collapsed",
            key="stock_search",
        )
    with right:
        do_search = st.button("\u200b", use_container_width=True, key="do_stock_search_site", help="搜尋")

    if do_search:
        found = search_stock(search_text)
        if found.empty:
            st.warning("找不到符合的股票代號或名稱。")
            st.session_state.search_result = None
            st.session_state.search_candidates = None
        elif len(found) == 1:
            row = found.iloc[0].to_dict()
            code = clean_symbol(row.get("symbol"))
            st.session_state.selected = code
            st.session_state.search_result = row
            st.session_state.search_candidates = None
            st.session_state.ai_selected = None
            st.session_state.ai_result = None
            st.rerun()
        else:
            st.session_state.search_result = None
            st.session_state.search_candidates = found.to_dict("records")

    candidates = st.session_state.get("search_candidates") or []
    if candidates:
        st.markdown('<div class="data-panel-heading">搜尋結果</div>', unsafe_allow_html=True)
        for item in candidates[:12]:
            c1, c2, c3 = st.columns([1.1, 4.7, 1.7], gap="small", vertical_alignment="center")
            with c1:
                st.markdown(f'<div class="search-code-site">{item.get("symbol", "")}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="search-name-site"><b>{item.get("name", "")}</b><span>｜產業 {item.get("industry", "00")}</span></div>', unsafe_allow_html=True)
            with c3:
                if st.button("查看", key=f"search_select_site_{item.get('symbol')}", use_container_width=True):
                    st.session_state.selected = clean_symbol(item.get("symbol"))
                    st.session_state.search_result = item
                    st.session_state.search_candidates = None
                    st.rerun()

    code = clean_symbol(st.session_state.get("selected", ""))
    result = st.session_state.get("search_result") or {}
    if result and code not in st.session_state.watchlist:
        st.info(f'目前分析標的：{result.get("name", code)}（{code}）尚未加入自選股。')
        if st.button("加入自選股", key=f"add_watch_site_{code}"):
            st.session_state.watchlist.append(code)
            st.session_state.watchlist = list(dict.fromkeys(st.session_state.watchlist))
            save_watchlist(st.session_state.watchlist)
            st.rerun()

    st.markdown(f'<div class="selected-strip"><span>目前分析標的</span><strong>{code}</strong><span>｜自選股 {len(st.session_state.watchlist)} 檔</span></div>', unsafe_allow_html=True)


def _collect_json_candidates(patterns: list[str]) -> list[Path]:
    roots = [OUTPUT, OUTPUT / "research_reports", BASE / "data"]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.extend(root.rglob(pattern))
    unique = []
    seen = set()
    for p in sorted(found, key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
        if str(p) not in seen:
            unique.append(p)
            seen.add(str(p))
    return unique


def _read_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_date(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _flatten_records(obj):
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "articles", "items", "events", "earnings_calls", "memos", "records"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [obj]
    return []


def _finance_cache_path() -> Path:
    """本機完整財經資料的備援路徑。"""
    return OUTPUT / "research_reports" / "finance_info_latest.json"


@st.cache_data(ttl=180, show_spinner=False)
def _load_finance_info_cache() -> dict:
    """
    優先讀取 GitHub finance-data 分支的公開 JSON。
    若 GitHub 無法連線或回傳格式異常，才回退本機完整 JSON。
    快取 180 秒，避免每次 Streamlit rerun 都重新下載。
    """
    import urllib.request
    import time

    public_url = (
        "https://raw.githubusercontent.com/"
        "alanpass/stock_trading/finance-data/"
        "output/research_reports/finance_info_public.json"
    )
    request_url = f"{public_url}?v={int(time.time())}"

    # 第一優先：GitHub 公開財經資料
    try:
        request = urllib.request.Request(
            request_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        if (
            isinstance(data, dict)
            and data.get("report_type") == "finance_info_public"
            and data.get("updated_at")
            and isinstance(data.get("news"), list)
            and isinstance(data.get("earnings"), list)
        ):
            data["_data_source"] = (
                "GitHub finance-data / finance_info_public.json"
            )
            return data

    except Exception:
        # GitHub 讀取失敗時，繼續使用本機備援。
        pass

    # 第二優先：本機完整財經資料
    path = _finance_cache_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                data["_data_source"] = (
                    "本機備援 / finance_info_latest.json"
                )
                return data
    except Exception:
        pass

    return {"_data_source": "資料讀取失敗"}


def _recent_earnings_items(days: int = 2):
    """讀取 08:30 / 18:00 財經資訊快取，並以實際事件日期過濾。"""
    today = taiwan_now().date()
    cutoff = today - pd.Timedelta(days=max(0, int(days) - 1))
    cache = _load_finance_info_cache()
    items = cache.get("earnings", []) if isinstance(cache, dict) else []

    def parse_item(item):
        if not isinstance(item, dict):
            return None
        dt = _as_date(
            item.get("event_date")
            or item.get("published_date")
            or item.get("date")
            or item.get("published_at")
            or item.get("published_time")
            or item.get("published_ts")
            or item.get("published")
            or item.get("created_at")
            or item.get("modified_date")
        )
        if dt is None or not (cutoff <= dt <= today):
            return None
        row = dict(item)
        row["_date"] = dt
        return row

    records = [x for x in (parse_item(i) for i in items) if x]
    if not records:
        # 相容舊版 Fugle memo 日檔。
        for path in _collect_json_candidates(["fugle_earnings_memo_*.json", "*earnings*memo*.json"]):
            obj = _read_json_file(path)
            for item in _flatten_records(obj):
                row = parse_item(item)
                if row:
                    row["_source_file"] = path.name
                    records.append(row)

    unique = []
    seen = set()
    for row in sorted(records, key=lambda x: x["_date"], reverse=True):
        key = (
            str(row.get("event_date") or row.get("date") or row["_date"]),
            clean_symbol(row.get("symbol") or row.get("code") or row.get("stock_code") or ""),
            str(row.get("source_url") or row.get("url") or row.get("title") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _recent_news_items(days: int = 2):
    """讀取 08:30 / 18:00 財經資訊快取，並以實際發佈日期過濾。"""
    today = taiwan_now().date()
    cutoff = today - pd.Timedelta(days=max(0, int(days) - 1))
    cache = _load_finance_info_cache()
    items = cache.get("news", []) if isinstance(cache, dict) else []

    def parse_item(item):
        if not isinstance(item, dict):
            return None
        dt = _as_date(
            item.get("published_at")
            or item.get("published_time")
            or item.get("published_ts")
            or item.get("published")
            or item.get("date")
            or item.get("created_at")
            or item.get("updated_at")
        )
        if dt is None or not (cutoff <= dt <= today):
            return None
        row = dict(item)
        row["_date"] = dt
        return row

    records = [x for x in (parse_item(i) for i in items) if x]
    if not records:
        for path in _collect_json_candidates(["cnyes_news_*.json", "*cnyes*news*.json", "news*.json"]):
            obj = _read_json_file(path)
            for item in _flatten_records(obj):
                row = parse_item(item)
                if row:
                    row["_source_file"] = path.name
                    records.append(row)

    unique = []
    seen = set()
    for row in sorted(records, key=lambda x: x["_date"], reverse=True):
        key = (
            str(row.get("title") or row.get("headline") or row.get("name") or ""),
            str(row["_date"]),
            str(row.get("url") or row.get("source_url") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _record_company(item: dict) -> str:
    code = clean_symbol(item.get("symbol") or item.get("code") or item.get("stock_code") or "")
    name = str(item.get("name") or item.get("company") or item.get("company_name") or code).strip()
    return f"{name}（{code}）" if code and code not in name else name


def _record_summary(item: dict) -> str:
    """財經資訊頁永遠優先顯示實際正文／Agent 摘要，避免只剩「快取尚未更新」。"""
    for key in (
        "one_line_summary", "summary", "brief_summary", "description", "abstract",
        "content", "memo_text", "text", "ollama_summary", "agent_summary"
    ):
        value = item.get(key)
        if isinstance(value, list):
            value = "；".join(str(x) for x in value if str(x).strip())
        if value:
            text = re.sub(r"\s+", " ", str(value)).strip()
            if text:
                return text[:1800]
    title = str(item.get("title") or item.get("headline") or item.get("company") or item.get("name") or "財經事件").strip()
    return f"{title}：已列入最近兩日財經資訊更新範圍；目前只有標題資料，請查看原文確認。"


_FIN_SCHEDULE_TEXT = "08:10、11:00、13:30、16:00、18:00、23:00"


def _fin_hl(text, terms) -> str:
    """HTML 跳脫後，把關鍵詞用螢光紅字標出（單次比對，避免巢狀標籤）。"""
    from html import escape as _esc
    safe = _esc(str(text or ""))
    words = sorted({str(t).strip() for t in (terms or []) if str(t).strip()}, key=len, reverse=True)
    words = [_esc(w) for w in words if _esc(w) in safe]
    if not words:
        return safe
    pattern = "|".join(re.escape(w) for w in words)
    return re.sub(pattern, lambda m: f'<mark class="fin-hl">{m.group(0)}</mark>', safe)


def _fin_badge(sentiment: str) -> str:
    s = str(sentiment or "中性")
    cls = {"利多": "up", "利空": "down", "混合": "mix"}.get(s, "flat")
    return f'<span class="fin-badge {cls}">{s}</span>'


def _fin_chips(items, cls: str = "") -> str:
    from html import escape as _esc
    return "".join(f'<span class="fin-chip {cls}">{_esc(str(x))}</span>' for x in items if str(x).strip())


def _inject_finance_css() -> None:
    """樣式已移到全站樣式區塊（檔案最底部 V16），這裡不再於頁面內插入 <style>。"""
    return None


def _render_news_digest(digest: dict, total_news: int) -> None:
    if not digest:
        return
    stats = digest.get("stats", {}) or {}
    headline = digest.get("agent_headline") or digest.get("headline") or ""
    by = "Qwen3 Agent" if digest.get("generated_by") == "qwen3" else "規則摘要"
    parts = [f'<div class="fin-digest"><h3>📌 AI 今日重點 <small style="font-weight:400;font-size:12px">（{by}整理）</small></h3>']
    if headline:
        parts.append(f'<div class="fin-headline">{_fin_hl(headline, [])}</div>')
    parts.append(
        '<div class="fin-stats">'
        f'<div class="fin-stat">新聞<b>{stats.get("total", total_news)}</b></div>'
        f'<div class="fin-stat">利多<b style="color:#b3121a">{stats.get("利多", 0)}</b></div>'
        f'<div class="fin-stat">利空<b style="color:#14532d">{stats.get("利空", 0)}</b></div>'
        f'<div class="fin-stat">中性／混合<b>{stats.get("中性", 0) + stats.get("混合", 0)}</b></div></div>'
    )
    points = digest.get("agent_key_points") or []
    if points:
        parts.append('<ul class="fin-list">' + "".join(f"<li>{_fin_hl(p, [])}</li>" for p in points) + "</ul>")
    else:
        rows = digest.get("key_points", []) or []
        parts.append(
            '<ul class="fin-list">'
            + "".join(f'<li>{_fin_badge(x.get("sentiment"))}{_fin_hl(x.get("point") or x.get("title"), x.get("highlights"))}</li>' for x in rows[:8])
            + "</ul>"
        )
    bull, bear = digest.get("bullish", []) or [], digest.get("bearish", []) or []
    if bull or bear:
        def col(title, rows, cls):
            body = "".join(f'<li>{_fin_hl(x.get("theme", ""), [])}</li>' for x in rows[:4]) or "<li>—</li>"
            return f'<div><div class="fin-sub">{_fin_badge(cls)}{title}</div><ul class="fin-list">{body}</ul></div>'
        parts.append('<div class="fin-two">' + col("利多主題", bull, "利多") + col("利空主題", bear, "利空") + "</div>")
    heat = digest.get("sector_heat", []) or []
    if heat:
        chips = "".join(
            f'<span class="fin-chip {"hot" if i < 3 else ""}">{x["sector"]} {x["count"]}'
            f'{"　▲" if x.get("tone") == "偏多" else "　▼" if x.get("tone") == "偏空" else ""}</span>'
            for i, x in enumerate(heat[:10])
        )
        parts.append(f'<div class="fin-sub" style="margin-top:12px">產業熱度（則數）</div>{chips}')
    watch, risks = digest.get("watch_items", []) or [], digest.get("risks", []) or []
    if watch or risks:
        def col2(title, rows):
            body = "".join(f"<li>{_fin_hl(x, [])}</li>" for x in rows) or "<li>—</li>"
            return f'<div><div class="fin-sub">{title}</div><ul class="fin-list">{body}</ul></div>'
        parts.append('<div class="fin-two">' + col2("👀 待追蹤", watch) + col2("⚠️ 風險", risks) + "</div>")
    parts.append("</div>")
    st.markdown('<div translate="no" class="notranslate">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def _safe_url(url) -> str:
    from html import escape as _esc
    u = str(url or "").strip()
    return _esc(u, quote=True) if u.startswith(("http://", "https://")) else ""


def _news_item_html(item: dict, is_open: bool = False) -> str:
    from html import escape as _esc
    title = str(item.get("title") or item.get("headline") or "未命名新聞").strip()
    hl = item.get("highlights") or []
    badge = _fin_badge(item.get("sentiment")) if item.get("sentiment") else ""
    imp = item.get("importance")
    stars = ""
    if isinstance(imp, (int, float)):
        n = max(1, min(5, int(round(float(imp) / 2))))
        stars = f'<span class="fin-meta" style="margin-left:8px">{"★" * n}{"☆" * (5 - n)}</span>'
    body = []
    points = item.get("ai_points") or []
    if points:
        body.append('<ul class="fin-list">' + "".join(f"<li>{_fin_hl(p, hl)}</li>" for p in points) + "</ul>")
    else:
        body.append(f'<div>{_esc(str(_record_summary(item)))}</div>')
    if item.get("why"):
        body.append(f'<div class="fin-why">💡 {_fin_hl(item["why"], hl)}</div>')
    meta = []
    when = str(item.get("published") or item.get("published_at") or "")[:16].replace("T", " ")
    if when:
        meta.append(_esc(when))
    if item.get("category"):
        meta.append(_esc(str(item["category"])))
    if meta:
        body.append(f'<div class="fin-meta">{"　｜　".join(meta)}</div>')
    chips = _fin_chips(item.get("sectors") or []) + _fin_chips(item.get("stocks") or [], "stock")
    if chips:
        body.append(f"<div>{chips}</div>")
    url = _safe_url(item.get("url") or item.get("source_url"))
    if url:
        body.append(f'<div style="margin-top:6px"><a href="{url}" target="_blank" rel="noopener noreferrer">查看原文 ↗</a></div>')
    return (
        f'<details class="fin-item"{" open" if is_open else ""}>'
        f'<summary>{badge}{_esc(title)}{stars}</summary><div class="fin-body">{"".join(body)}</div></details>'
    )


def _earning_item_html(item: dict, label: str, is_open: bool = False) -> str:
    from html import escape as _esc
    hl = item.get("highlights") or []
    pts = item.get("ai_points") or []
    body = []
    if pts:
        body.append('<ul class="fin-list">' + "".join(f"<li>{_fin_hl(p, hl)}</li>" for p in pts) + "</ul>")
    else:
        body.append(f'<div>{_esc(str(_record_summary(item)))}</div>')
    signal = item.get("impact") or item.get("signal") or item.get("judgement") or item.get("sentiment")
    if signal:
        body.append(f'<div class="fin-meta"><b>判斷：</b>{_esc(str(signal))}</div>')
    url = _safe_url(item.get("source_url") or item.get("url"))
    if url:
        body.append(f'<div><a href="{url}" target="_blank" rel="noopener noreferrer">查看來源 ↗</a></div>')
    badge = _fin_badge(item.get("sentiment")) if item.get("sentiment") else ""
    return (
        f'<details class="fin-item"{" open" if is_open else ""}>'
        f'<summary>{badge}{_esc(label)}</summary><div class="fin-body">{"".join(body)}</div></details>'
    )


def _render_news_item(item: dict, expanded: bool = False) -> None:
    """相容舊呼叫：單篇新聞。"""
    st.markdown(f'<div translate="no" class="notranslate">{_news_item_html(item, expanded)}</div>', unsafe_allow_html=True)


def render_finance_workspace() -> None:
    # 財經頁在瀏覽器開啟期間每分鐘自動 rerun；180 秒資料快取到期後會重新向 GitHub 取最新 JSON。
    st_autorefresh(interval=60_000, key="finance_data_auto_refresh")
    st.markdown(
        '<div class="content-heading"><div class="eyebrow">DAILY FINANCE</div><h1>財經資訊</h1></div>',
        unsafe_allow_html=True,
    )
    cache = _load_finance_info_cache()
    earnings = _recent_earnings_items(days=2)
    news = _recent_news_items(days=2)

    if cache.get("updated_at"):
        updated = str(cache["updated_at"]).replace("T", " ")[:19]
        st.markdown(f"<div class='finance-refresh-time'>最後更新：{updated}（台灣時間）</div>", unsafe_allow_html=True)

    # 顯示實際讀取來源及發布時間，方便驗證上架網站是否同步
    data_source = cache.get("_data_source", "未知資料來源")
    published_at = cache.get("published_at", "無發布時間")
    st.caption(
        f"資料來源：{data_source}｜GitHub 發布時間：{published_at}"
    )
    if cache.get("stale"):
        st.markdown("<div class='fin-stale'>⚠️ 最近一次更新沒有取得新資料，目前顯示的是上一個時間點的內容。</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='info-strip'>法說會：{len(earnings)} 筆｜財經新聞：{len(news)} 筆｜"
        f"自動更新：每天 {_FIN_SCHEDULE_TEXT}（涵蓋前一日 00:00 至現在）。晨報 08:30、盤後分析 14:30。</div>",
        unsafe_allow_html=True,
    )

    _render_news_digest(cache.get("news_digest") or {}, len(news))

    left, right = st.columns([1, 1.45], gap="large")

    # ---------------- 法說會 ----------------
    with left:
        st.markdown('<div class="finance-card-title">法說會摘要</div>', unsafe_allow_html=True)
        edig = cache.get("earnings_digest") or {}
        if edig.get("headline"):
            st.markdown(f'<div class="fin-meta">{_fin_hl(edig["headline"], [])}</div>', unsafe_allow_html=True)
        if not earnings:
            st.info("最近 2 天目前沒有發現新的可用 Fugle 法說會摘要；這不代表財經資訊快取未更新。")
        else:
            html_items = "".join(
                _earning_item_html(item, f"{item['_date'].strftime('%Y-%m-%d')}｜{_record_company(item)}", i == 0)
                for i, item in enumerate(earnings[:20])
            )
            st.markdown(f'<div translate="no" class="notranslate">{html_items}</div>', unsafe_allow_html=True)

    # ---------------- 新聞 ----------------
    with right:
        st.markdown('<div class="finance-card-title">財經新聞摘要</div>', unsafe_allow_html=True)
        if not news:
            st.info("最近 2 天目前沒有可用的財經新聞；請確認 08:10 等排程是否有執行（output/scheduler_logs）。")
            return

        sectors_all = sorted({s for n in news for s in (n.get("sectors") or [])})
        with st.container(key="fin_filters"):
            c1, c2, c3 = st.columns([1.3, 1, 1])
            senti = c1.radio("情緒", ["全部", "利多", "利空", "中性／混合"], horizontal=True, key="fin_senti")
            order = c2.selectbox("排序", ["重要度", "最新時間"], key="fin_order")
            sector = c3.selectbox("產業", ["全部產業"] + sectors_all, key="fin_sector")

        rows = list(news)
        if senti == "利多":
            rows = [x for x in rows if x.get("sentiment") == "利多"]
        elif senti == "利空":
            rows = [x for x in rows if x.get("sentiment") == "利空"]
        elif senti == "中性／混合":
            rows = [x for x in rows if x.get("sentiment") in (None, "中性", "混合")]
        if sector != "全部產業":
            rows = [x for x in rows if sector in (x.get("sectors") or [])]
        if order == "重要度":
            rows.sort(key=lambda x: float(x.get("importance") or 0), reverse=True)
        else:
            rows.sort(key=lambda x: str(x.get("published_ts") or x.get("published") or ""), reverse=True)

        shown = int(st.session_state.get("fin_news_n", 15))
        st.caption(f"符合條件 {len(rows)} 則，顯示前 {min(shown, len(rows))} 則")
        html_items = "".join(_news_item_html(item, i == 0) for i, item in enumerate(rows[:shown]))
        # translate="no"：避免瀏覽器自動翻譯改動文字節點，造成 React removeChild 錯誤
        st.markdown(f'<div translate="no" class="notranslate">{html_items}</div>', unsafe_allow_html=True)
        if len(rows) > shown:
            if st.button(f"顯示更多（再 15 則，尚有 {len(rows) - shown} 則）", key="fin_more"):
                st.session_state["fin_news_n"] = shown + 15
                st.rerun()


HERO_DRIVE_FOLDER_ID = "1oSa6fv8F7YrVbUmX5UXVs8v0r4sNO9Fy"
HERO_LOCAL_DIR = BASE / "assets" / "hero_images"
HERO_INTERVAL_SEC = 5


@st.cache_data(show_spinner=False)
def _hero_local_image_uris() -> list:
    """讀取 assets/hero_images 內的圖片（依檔名排序），轉成 data URI。"""
    import base64
    out = []
    try:
        if HERO_LOCAL_DIR.exists():
            mimes = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
            for f in sorted(HERO_LOCAL_DIR.iterdir()):
                if f.suffix.lower() in mimes:
                    out.append(f"data:{mimes[f.suffix.lower()]};base64," + base64.b64encode(f.read_bytes()).decode("ascii"))
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _drive_folder_image_urls(folder_id: str) -> list:
    """備援：讀取公開 Google Drive 資料夾內的圖片；失敗時回傳空清單。"""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"https://drive.google.com/embeddedfolderview?id={folder_id}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        html = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", "ignore")
        ids = []
        for m in re.finditer(r'id="entry-([A-Za-z0-9_-]{15,})"(.*?)(?=id="entry-|$)', html, re.S):
            title = re.search(r'flip-entry-title">([^<]+)<', m.group(2))
            name = (title.group(1) if title else "").lower()
            if name and not name.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                continue
            ids.append(m.group(1))
        return [f"https://drive.google.com/thumbnail?id={i}&sz=w1920" for i in dict.fromkeys(ids)]
    except Exception:
        return []


def _render_hero_carousel() -> None:
    """首頁第一張圖：每 5 秒交叉淡入換圖，緩慢推近 + 底部進度條 + 圓點指示，循環播放（純 CSS）。"""
    urls = _hero_local_image_uris() or _drive_folder_image_urls(HERO_DRIVE_FOLDER_ID)
    if len(urls) < 2:
        st.markdown(f'<div class="hero-visual">{_site_image_html(0, wide=True)}</div>', unsafe_allow_html=True)
        return
    n = len(urls)
    total = n * HERO_INTERVAL_SEC
    slides, dots = [], []
    for i, u in enumerate(urls):
        slides.append(f'<div class="hero-slide" style="background-image:url(\'{u}\');animation-delay:{i * HERO_INTERVAL_SEC}s"></div>')
        dots.append(f'<span style="animation-delay:{i * HERO_INTERVAL_SEC}s"></span>')
    a = 100 / n                              # 每張圖佔的比例
    trans = 0.9                              # 滑動耗時（秒）
    e = trans / total * 100
    for i in range(n):                       # 第 1 張一載入就在定位，其餘依序從右側滑入
        slides[i] = slides[i].replace(f"animation-delay:{i * HERO_INTERVAL_SEC}s", f"animation-delay:{i * HERO_INTERVAL_SEC - trans:.2f}s")
    css = f"""
    .hero-carousel{{position:relative;width:100%;height:350px;border-radius:10px;overflow:hidden;box-shadow:0 8px 25px rgba(0,0,0,.08);background:#e9edf1;}}
    .hero-slide{{position:absolute;inset:0;background-size:cover;background-position:center;transform:translateX(100%);
      will-change:transform;animation:heroSlide {total}s infinite;animation-fill-mode:backwards;}}
    @keyframes heroSlide{{
      0%{{transform:translateX(100%);animation-timing-function:cubic-bezier(.65,0,.25,1)}}
      {e:.2f}%{{transform:translateX(0)}}
      {a:.2f}%{{transform:translateX(0);animation-timing-function:cubic-bezier(.65,0,.25,1)}}
      {a + e:.2f}%{{transform:translateX(-100%)}}
      100%{{transform:translateX(-100%)}}
    }}
    .hero-carousel::after{{content:"";position:absolute;left:0;bottom:0;height:3px;width:100%;
      background:linear-gradient(90deg,#8e2b2f,#58758e);transform-origin:left;animation:heroBar {HERO_INTERVAL_SEC}s linear infinite;}}
    @keyframes heroBar{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}
    .hero-dots{{position:absolute;right:16px;bottom:14px;display:flex;gap:7px;z-index:3;}}
    .hero-dots span{{width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.55);border:1px solid rgba(0,0,0,.25);
      animation:heroDot {total}s infinite;}}
    @keyframes heroDot{{0%{{background:#8e2b2f;transform:scale(1.35)}}{a:.2f}%{{background:#8e2b2f;transform:scale(1.35)}}{a + .01:.2f}%{{background:rgba(255,255,255,.55);transform:scale(1)}}100%{{background:rgba(255,255,255,.55);transform:scale(1)}}}}
    @media(max-width:600px){{.hero-carousel{{height:220px}}}}
    """
    # 樣式與圖片分開送出：含 <style> 的區塊會被「收掉頂端空白」的規則隱藏
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-carousel">{"".join(slides)}<div class="hero-dots">{"".join(dots)}</div></div>', unsafe_allow_html=True)


def render_home_workspace() -> None:
    """首頁：依 PDF 的節點節奏，以實際提供的六張台股圖片做視覺主體。"""
    _render_hero_carousel()

    st.markdown('''
    <section translate="no" class="home-band blush"><div class="home-two-col">
      <div class="home-image-stack">
        <div class="small-visual">''' + _site_image_html(1, wide=False) + '''</div>
        <div class="small-visual">''' + _site_image_html(2, wide=False) + '''</div>
      </div>
      <div class="home-copy"><div class="eyebrow">ABOUT TAIWAN STOCKS</div><h2>關於台股</h2>
      <p>把即時行情、股價趨勢、技術 K 線、成交量、籌碼與法人資料集中到同一套操作流程，使用者可以先從自選股與分析標的開始，再逐層檢查市場訊號。</p>
      <p>從「看行情」到「做判斷」都維持同一個分析標的，減少頁面切換造成的資訊斷裂。</p>
      <a class="home-link" href="?section=strategy" target="_self">操作策略 →</a></div>
    </div></section>''', unsafe_allow_html=True)

    st.markdown('''
    <section translate="no" class="home-band white-band"><div class="home-two-col reverse-on-mobile">
      <div class="home-image-grid three-grid">
        <div>''' + _site_image_html(3, wide=False) + '''</div>
        <div>''' + _site_image_html(4, wide=False) + '''</div>
        <div>''' + _site_image_html(5, wide=False) + '''</div>
      </div>
      <div class="home-copy"><div class="eyebrow">DAILY FINANCE</div><h2>每日財經資訊</h2>
      <p>集中查看最近 2 天的法說會摘要與財經新聞摘要，先理解市場事件，再回到操作策略與未來分析。</p>
      <div class="home-accordion-hints"><div><strong>01．財經新聞摘要</strong><span>最新財經事件重點</span></div><div><strong>02．法說會摘要</strong><span>企業展望與營運訊息</span></div></div>
      <a class="home-link" href="?section=finance" target="_self">財經資訊 →</a></div>
    </div></section>''', unsafe_allow_html=True)

    # 專注台股左側只保留一張主視覺，佔整個節點約一半寬。
    six = _site_image_html(6, wide=True)
    st.markdown(f'''
    <section translate="no" class="home-band blush"><div class="home-two-col focus-taiwan-layout"><div class="focus-main-visual">{six}</div>
      <div class="home-copy"><div class="eyebrow">FOCUS ON TAIWAN STOCKS</div><h2>專注台股</h2>
      <p>未來分析把 AI 隔日預測、市場情報與利多／利空分析、模型研究 Agent 串在一起，讓使用者從今日資料延伸到明日的可能方向。</p>
      <a class="home-link" href="?section=future" target="_self">未來分析 →</a></div>
    </div></section>''' , unsafe_allow_html=True)

    st.markdown(f'''
    <section translate="no" class="contact-band"><div class="contact-overlay">{_site_image_html(5, wide=True, dark=True)}<div class="contact-shade"></div></div>
      <div class="contact-content"><div class="eyebrow">CONTACT</div><h2>聯絡資訊</h2>
      <p>對股票資料、操作策略、財經資訊或 AI 研究功能有問題，可從訂閱系統頁面的聯絡表單留下訊息。</p>
      <a class="contact-link" href="?section=subscribe" target="_self">訂閱系統／聯絡我們 →</a></div>
    </section>''' , unsafe_allow_html=True)

def render_subscription_workspace() -> None:
    st.markdown(
        f'<div class="page-image-banner contact-page-banner">{_site_image_html(11, wide=True)}'
        '<div class="page-image-shade"></div><div class="page-image-title">聯絡資訊</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="content-heading"><div class="eyebrow">CONTACT & SUBSCRIPTION</div><h1>聯絡資訊</h1></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="contact-info-card">
              <h2>聯絡資訊</h2>
              <p>歡迎針對網站功能、股票資料、AI 分析或每日財經資訊提出問題。</p>
              <div class="contact-item"><span>◎</span><div><b>台灣股市研究</b><small>AI 台股即時互動式分析系統</small></div></div>
              <div class="contact-item"><span>⌕</span><div><b>研究服務</b><small>股票操作策略／未來分析／財經資訊</small></div></div>
              <div class="contact-item"><span>✉</span><div><b>電子郵件</b><small>a1113359@mail.nuk.edu.tw</small></div></div>
              <div class="contact-item"><span>◎</span><div><b>聯絡電話</b><small>+886-907-611-728</small></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="form-title">訂閱／聯絡表單</div>', unsafe_allow_html=True)
        with st.form("site_contact_form", clear_on_submit=False):
            name = st.text_input("姓名 *", key="contact_name")
            email = st.text_input("電子郵件 *", key="contact_email")
            message = st.text_area("訊息（可選）", height=150, key="contact_message")
            subscribe = st.checkbox("訂閱每日晨報與盤後分析（可選）", value=True, key="contact_subscribe")
            st.markdown(
                "<div class='subscribe-note'>勾選後會加入每日晨報（08:30）與盤後分析（14:30）收件名單，"
                "並自動寄一封「訂閱成功」信到你的信箱；不勾選則只送出訊息，不會加入名單。</div>",
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("送出")

        if submitted:
            name_v = name.strip()
            email_v = email.strip()
            message_v = message.strip()

            if not name_v or not email_v:
                st.warning("請填寫姓名與電子郵件。")
            elif not subscribe and not message_v:
                st.warning("沒有勾選訂閱時，請至少留下訊息。")
            else:
                try:
                    result = register_subscriber(
                        BASE,
                        name=name_v,
                        email=email_v,
                        message=message_v,
                        subscribed=bool(subscribe),
                    )
                    if not result.get("saved"):
                        st.error(result.get("error", "資料保存失敗。"))
                    else:
                        if result.get("requested_subscribe"):
                            if result.get("already_subscribed"):
                                st.info("這個信箱已經在訂閱名單內，不會重複寄送確認信。")
                            elif result.get("welcome_sent"):
                                st.success(f"訂閱完成！訂閱成功信已寄到 {result.get('email')}，沒看到的話請檢查垃圾郵件匣。")
                            else:
                                st.success("訂閱完成；之後的晨報與盤後分析會寄到此電子郵件。")
                                st.warning(f"訂閱成功信沒有寄出：{result.get('welcome_error') or '原因不明'}")
                        else:
                            st.success("訊息已送出，謝謝你！（未加入訂閱）")
                        if result.get("owner_notify_error") and not result.get("owner_notified"):
                            st.caption("（管理者通知暫時失敗，資料已保存。）")
                        if result.get("persistence_warning"):
                            st.caption(result["persistence_warning"])
                except Exception as exc:
                    st.error(f"訂閱系統錯誤：{type(exc).__name__}: {exc}")
    try:
        sub_status = subscription_status(BASE)
        st.caption(f"目前有效訂閱者：{int(sub_status.get('active_count', 0) or 0)} 人")
    except Exception:
        pass

def _get_nav_section() -> str:
    default = str(st.session_state.get("nav_section", "home"))
    try:
        value = st.query_params.get("section", default)
        if isinstance(value, list):
            value = value[0] if value else default
        value = str(value or default)
    except Exception:
        value = default
    allowed = {"home", "strategy", "future", "finance", "rotation", "subscribe"}
    return value if value in allowed else "home"


def _set_nav_section(section: str) -> None:
    st.session_state["nav_section"] = section
    try:
        st.query_params["section"] = section
    except Exception:
        pass
    st.rerun()


def render_site_navigation() -> str:
    current = _get_nav_section()
    nav = [("home", "首頁"), ("strategy", "操作策略"), ("future", "未來分析"), ("finance", "財經資訊"), ("rotation", "產業分析"), ("subscribe", "訂閱系統")]
    links = []
    for key, label in nav:
        active = " active" if current == key else ""
        links.append(f'<a class="site-nav-link{active}" href="?section={key}" target="_self">{label}</a>')
    st.markdown('<nav translate="no" class="site-nav">' + ''.join(links) + '</nav>', unsafe_allow_html=True)
    st.markdown('<div class="site-rule"></div>', unsafe_allow_html=True)
    return current


def _render_footer() -> None:
    """每一頁固定放頁尾，並顯示資料來源、新聞著作權與投資風險聲明。"""
    st.markdown('''
    <footer translate="no" class="site-footer"><div class="footer-inner"><div class="footer-contact">
      <div class="footer-title">台股即時互動式分析系統</div><p>股票即時行情、操作策略、未來分析與每日財經資訊。</p>
      <div class="footer-small">電子郵件：a1113359@mail.nuk.edu.tw</div><div class="footer-small">聯絡電話：+886-907-611-728</div>
    </div><div class="footer-form-note"><div class="footer-title">訂閱每日財經資訊</div>
      <p>輸入電子郵件後，可由訂閱系統留下需求。</p><a href="?section=subscribe" target="_self">前往訂閱系統 →</a>
    </div></div>
    <div class="footer-disclaimer" style="border-top:1px solid rgba(255,255,255,.08);padding:18px clamp(18px,4vw,42px) 16px;color:#a9a9a9;font-size:11px;line-height:1.9;overflow-wrap:anywhere;">
      <div style="color:#d68f8f;font-size:12px;font-weight:700;letter-spacing:.3px;margin-bottom:5px;">資料來源與使用聲明</div>
      <p style="margin:0 0 4px;">資料來源：Fugle、鉅亨網等公開資訊來源。</p>
      <p style="margin:0 0 4px;">新聞以標題聯播方式呈現，著作權屬原媒體與原作者所有。本網站不重製、不儲存新聞內文、不轉載全文，亦不主張任何權利。</p>
      <p style="margin:0;">本站資料僅供參考，不構成投資建議。</p>
    </div>
    <div class="footer-bottom">© 2026 台股即時互動式分析系統</div></footer>''', unsafe_allow_html=True)

def render_strategy_workspace() -> None:
    """操作策略頁：維持文件的功能順序；各功能節點使用淺色交錯背景，形成同一套網站視覺。"""
    st.markdown(f'<div class="page-image-banner">{_site_image_html(7, wide=True)}</div>', unsafe_allow_html=True)
    render_strategy_search()

    # 自選股固定在左側欄（Streamlit 側邊欄本身固定、可獨立捲動），不會蓋到右側的分析圖表。
    with st.sidebar:
        st.markdown('<div class="feature-section-title side-watch-title">自選股即時行情</div>', unsafe_allow_html=True)
        render_watchlist(compact=True)

    with st.container(key="strategy-02-selected"):
        st.markdown('<div class="feature-section-title">01｜分析標的</div>', unsafe_allow_html=True)
        render_selected_and_live(show_heading=False)

    with st.container(key="strategy-03-trend"):
        st.markdown('<div class="feature-section-title">02｜股價趨勢</div>', unsafe_allow_html=True)
        render_trend()

    with st.container(key="strategy-04-kline"):
        st.markdown('<div class="feature-section-title">03｜技術 K 線</div>', unsafe_allow_html=True)
        render_kline()

    with st.container(key="strategy-05-volume"):
        st.markdown('<div class="feature-section-title">04｜成交量</div>', unsafe_allow_html=True)
        render_volume_snapshot()

    with st.container(key="strategy-06-entry"):
        st.markdown('<div class="feature-section-title">05｜進場建議分析系統</div>', unsafe_allow_html=True)
        render_entry_prediction()

    with st.container(key="strategy-07-tdcc"):
        st.markdown('<div class="feature-section-title">06｜大戶 VS 散戶持有股比率</div>', unsafe_allow_html=True)
        render_tdcc()

    with st.container(key="strategy-08-trades"):
        st.markdown('<div class="feature-section-title">07｜交易明細（台灣時間）</div>', unsafe_allow_html=True)
        render_trades()

    with st.container(key="strategy-09-institutions"):
        st.markdown('<div class="feature-section-title">08｜三大法人 10 交易日買賣超</div>', unsafe_allow_html=True)
        render_institutions()

def render_future_workspace() -> None:
    st.markdown(f'<div class="page-image-banner">{_site_image_html(8, wide=True)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-heading"><div class="eyebrow">FUTURE ANALYSIS</div><h1>未來分析</h1><p>把預測、市場情報與研究 Agent 集中在同一頁，讓操作策略之後的判斷有明確出口。</p></div>', unsafe_allow_html=True)

    with st.container(key="future-01-ai"):
        st.markdown('<div class="feature-section-title">01｜AI 隔日預測</div>', unsafe_allow_html=True)
        render_ai()

    with st.container(key="future-02-intelligence"):
        st.markdown('<div class="feature-section-title">02｜市場情報與利多／利空分析</div>', unsafe_allow_html=True)
        selected = clean_symbol(st.session_state.get("selected", ""))
        industry_code = str(INDUSTRY_OVERRIDES.get(selected, WATCHLIST_INDUSTRY_FALLBACK.get(selected, "00"))).zfill(2)
        render_market_intelligence(BASE, OUTPUT, selected_symbol=selected, selected_industry=industry_code, blocked=st.session_state.get("ai_busy", False))

    with st.container(key="future-03-agent"):
        st.markdown('<div class="feature-section-title">03｜模型研究 Agent</div>', unsafe_allow_html=True)
        render_model_research_agent()


# ============================================================
# 產業輪動：供應鏈族群相對強弱與輪動象限
# ============================================================
# 產業主題清單集中在 industry_rotation.py，供頁面與自動排程共用。
def _rotation_period_return(series: pd.Series, bars: int) -> float:
    """以有效交易 K 棒計算報酬率；資料不足時回傳 NaN。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= bars:
        return np.nan
    base = float(s.iloc[-bars - 1])
    latest = float(s.iloc[-1])
    return (latest / base - 1.0) * 100.0 if base else np.nan


def _rotation_quadrant(rs: float, momentum: float) -> str:
    """RRG 四象限：RS 與相對動能均以 100 為中線。"""
    if not np.isfinite(rs) or not np.isfinite(momentum):
        return "資料不足"
    if rs >= 100 and momentum >= 100:
        return "領先"
    if rs >= 100 and momentum < 100:
        return "轉弱"
    if rs < 100 and momentum < 100:
        return "落後"
    return "改善"


def _rotation_make_index(history_map: dict[str, pd.DataFrame], symbols: list[str]) -> tuple[pd.Series, dict]:
    """以代表股每日報酬等權平均建構族群指數，並回傳個股最近一期報酬。"""
    return_frames = []
    latest_returns = {}
    valid_symbols = []
    for code in symbols:
        h = history_map.get(code)
        if h is None or h.empty or "close" not in h.columns:
            continue
        d = h.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d = d.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
        if len(d) < 22 or d["close"].iloc[-1] <= 0:
            continue
        returns = d.set_index("date")["close"].pct_change()
        returns.name = code
        return_frames.append(returns)
        latest_returns[code] = float(returns.dropna().iloc[-1] * 100.0) if not returns.dropna().empty else np.nan
        valid_symbols.append(code)

    if not return_frames:
        return pd.Series(dtype=float), {"valid_symbols": [], "latest_returns": {}}

    # 不補造休市日或缺少的 K 棒；各交易日只平均當日真正有有效報酬的代表股。
    frame = pd.concat(return_frames, axis=1).sort_index()
    daily_returns = frame.mean(axis=1, skipna=True).dropna()
    index_series = (1.0 + daily_returns).cumprod() * 100.0
    index_series.name = "族群等權指數"
    return index_series, {"valid_symbols": valid_symbols, "latest_returns": latest_returns}


def _rotation_build_frame(group_series: dict[str, pd.Series], history_map: dict[str, pd.DataFrame],
                          period_bars: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """產生族群績效表，以及相對 0050 基準的 RRG 歷史座標。"""
    benchmark, _ = _rotation_make_index(history_map, ["0050"])
    # 0050 是可取得的廣泛市場代理；清楚揭露它不是完整加權指數。
    if benchmark.empty:
        raise RuntimeError("無法取得 0050 基準日 K，暫時無法計算相對強弱。")

    # 族群報酬表以樣本股等權平均計算；同一股票可出現在不同主題。
    rows = []
    rrg_map: dict[str, pd.DataFrame] = {}
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min
    for name, group_index in group_series.items():
        aligned = pd.concat([group_index.rename("group"), benchmark.rename("benchmark")], axis=1).dropna()
        if len(aligned) < 70:
            continue
        aligned["relative"] = aligned["group"] / aligned["benchmark"] * 100.0
        aligned["rs"] = aligned["relative"] / aligned["relative"].rolling(50, min_periods=50).mean() * 100.0
        aligned["momentum"] = aligned["rs"] / aligned["rs"].shift(10) * 100.0
        aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(subset=["rs", "momentum"])
        if aligned.empty:
            continue

        view = aligned.tail(period_bars).copy()
        rrg_map[name] = view
        last = aligned.iloc[-1]
        metrics = {
            "族群": name,
            "樣本數": int(len(group_index.attrs.get("valid_symbols", []))) if hasattr(group_index, "attrs") else 0,
            "近 1 日": _rotation_period_return(group_index, 1),
            "近 5 日": _rotation_period_return(group_index, 5),
            "近 20 日": _rotation_period_return(group_index, 20),
            "近 60 日": _rotation_period_return(group_index, 60),
            "今年以來": np.nan,
            "相對強弱": float(last["rs"]),
            "相對動能": float(last["momentum"]),
            "輪動象限": _rotation_quadrant(float(last["rs"]), float(last["momentum"])),
            "_group_index": group_index,
        }
        dates = pd.to_datetime(group_index.index, errors="coerce")
        if not dates.empty:
            this_year = pd.Timestamp.now(tz="Asia/Taipei").year
            ytd_start = pd.Timestamp(year=this_year, month=1, day=1)
            ytd_hist = group_index[dates >= ytd_start]
            prior_hist = group_index[dates < ytd_start]
            if not ytd_hist.empty and not prior_hist.empty:
                metrics["今年以來"] = (float(ytd_hist.iloc[-1]) / float(prior_hist.iloc[-1]) - 1.0) * 100.0
        rows.append(metrics)
        min_date = min(min_date, aligned.index.min())
        max_date = max(max_date, aligned.index.max())

    if not rows:
        raise RuntimeError("有效歷史樣本不足，至少需要約 70 個交易日資料。")
    result = pd.DataFrame(rows).sort_values("近 20 日", ascending=False, na_position="last").reset_index(drop=True)
    return result, rrg_map


def _render_published_rotation(payload: dict) -> None:
    """顯示排程產生的最新產業快照；不在網站端重複下載大量歷史資料。"""
    import html as _html

    st.markdown(
        '<div class="content-heading"><div class="eyebrow">SECTOR INTELLIGENCE</div>'
        '<h1>產業分析</h1><p>整合供應鏈輪動、相對強弱、短中期報酬、成分股廣度與技術動能。</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <style>
    .rotation-note{background:#fbf1f1;border:1px solid #eadada;border-left:4px solid #8e2b2f;border-radius:10px;padding:13px 16px;margin:0 0 18px;color:#594b4b;font-size:12px;line-height:1.85}
    .rotation-quadrants{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 14px}
    .rotation-quadrants div{border-radius:8px;padding:10px 12px;font-size:12px;line-height:1.6}
    .rotation-leading{background:#fce8e8;color:#8e2b2f}.rotation-weakening{background:#fff1df;color:#9a5b10}
    .rotation-lagging{background:#edf1f5;color:#455468}.rotation-improving{background:#e5f4ed;color:#17694b}
    </style>
    """, unsafe_allow_html=True)

    groups = [
        g for g in payload.get("groups", [])
        if isinstance(g, dict) and str(g.get("name", "")) in ROTATION_THEMES
    ]
    by_name = {str(g.get("name", "")): g for g in groups}

    def rotation_label(name: str) -> str:
        return ROTATION_DISPLAY_NAMES.get(str(name), str(name))

    def sign_style(value: object) -> str:
        """台股習慣：上漲／流入紅色、下跌／流出綠色。"""
        import re
        match = re.search(r"[-+]?\\d+(?:\\.\\d+)?", str(value).replace(",", ""))
        if not match:
            return ""
        try:
            number = float(match.group(0))
        except (TypeError, ValueError):
            return ""
        if number > 0:
            return "color: #c62828; font-weight: 650;"
        if number < 0:
            return "color: #16803c; font-weight: 650;"
        return "color: #64748b;"

    def styled_frame(frame: pd.DataFrame, columns: list[str]):
        valid = [col for col in columns if col in frame.columns]
        if valid:
            return frame.style.applymap(sign_style, subset=valid)
        return frame
    asof = str(payload.get("data_asof") or "未知")
    updated_at = str(payload.get("generated_at") or "未知")
    benchmark = str(payload.get("benchmark") or "加權指數")
    st.markdown(
        f'<div class="rotation-note"><strong>資料日期：{_html.escape(asof)}</strong>｜最後更新：{_html.escape(updated_at)}'
        f'<br>基準：{_html.escape(benchmark)}｜上漲主題 {payload.get("up_group_count", 0)} 群／下跌主題 {payload.get("down_group_count", 0)} 群'
        '<br>資料僅供研究參考，不構成投資建議。</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"追蹤 {payload.get('tracked_unique_stocks', 0)} 檔不重複代表股；最新交易日有資料 {payload.get('fresh_stock_count', 0)} 檔。"
        "本頁資料於台股交易日 15:10 起由本機排程更新，再發布到 GitHub。"
    )

    a, b, c = st.columns([1, 3, 1.5])
    with a:
        period = st.radio("輪動軌跡", ["1 個月", "3 個月", "半年", "1 年"], index=1, key="rotation_published_period")
    display_to_internal = {rotation_label(name): name for name in by_name}
    with b:
        defaults = [rotation_label(n) for n in ROTATION_DEFAULT_THEMES if n in by_name]
        selected_labels = st.multiselect(
            "觀察主題", list(display_to_internal.keys()), default=defaults,
            key="rotation_published_groups",
        )
        selected = [display_to_internal[label] for label in selected_labels if label in display_to_internal]
    with c:
        sort_by = st.selectbox(
            "排行依據",
            ["近 20 日", "近 5 日", "今日", "相對強弱", "相對動能", "資金流向"],
            key="rotation_published_sort",
        )

    quadrants = {q: sum(1 for g in groups if g.get("quadrant") == q) for q in ["領先", "改善", "轉弱", "落後"]}
    kpis = st.columns(5)
    kpis[0].metric("追蹤主題", f"{len(groups)} 群")
    kpis[1].metric("領先", f"{quadrants['領先']} 群")
    kpis[2].metric("改善", f"{quadrants['改善']} 群")
    kpis[3].metric("轉弱", f"{quadrants['轉弱']} 群")
    kpis[4].metric("落後", f"{quadrants['落後']} 群")

    # 資金流向圖：以代表股成交值在本頁樣本中的占比變化（百分點）作為資金聚焦代理。
    st.markdown('<div class="feature-section-title">01｜資金流向輪動圖</div>', unsafe_allow_html=True)
    has_yesterday_flow = any(
        g.get("yesterday_sample_turnover_share_delta_pp") is not None for g in groups
    )
    flow_options = ["今日", "昨日"] if has_yesterday_flow else ["今日"]
    flow_day = st.radio(
        "資金流向日期",
        flow_options,
        horizontal=True,
        key="rotation_published_flow_day",
    )
    if flow_day == "昨日":
        flow_delta_key = "yesterday_sample_turnover_share_delta_pp"
        flow_share_key = "yesterday_sample_turnover_share_pct"
        flow_date_label = payload.get("previous_data_asof") or "前一交易日"
        flow_compare_label = payload.get("previous_previous_data_asof") or "再前一交易日"
    else:
        flow_delta_key = "sample_turnover_share_delta_pp"
        flow_share_key = "sample_turnover_share_pct"
        flow_date_label = asof
        flow_compare_label = payload.get("previous_data_asof") or "前一交易日"

    flow_rows = []
    for group in groups:
        raw_delta = group.get(flow_delta_key)
        if raw_delta is None:
            continue
        try:
            delta_value = float(raw_delta)
        except (TypeError, ValueError):
            continue
        share_value = group.get(flow_share_key)
        flow_rows.append({
            "name": rotation_label(group.get("name", "")),
            "delta": delta_value,
            "share": float(share_value) if share_value is not None else float("nan"),
        })
    flow_rows.sort(key=lambda row: row["delta"], reverse=True)
    if not flow_rows:
        st.info("這份產業快照尚無資金流向資料。請執行一次最新版 run_industry_rotation_update.py --force 產生資料。")
    else:
        flow_colors = [
            "#c62828" if row["delta"] > 0 else "#16803c" if row["delta"] < 0 else "#94a3b8"
            for row in flow_rows
        ]
        flow_chart = go.Figure(go.Bar(
            x=[row["delta"] for row in flow_rows],
            y=[row["name"] for row in flow_rows],
            orientation="h",
            marker=dict(color=flow_colors),
            text=[f"{row['delta']:+.2f} pp" for row in flow_rows],
            textposition="outside",
            cliponaxis=False,
            customdata=[[row["share"]] for row in flow_rows],
            hovertemplate=(
                "%{y}<br>成交值占比變化：%{x:+.2f} pp"
                "<br>成交值占比：%{customdata[0]:.2f}%<extra></extra>"
            ),
        ))
        flow_chart.add_vline(x=0, line_color="#8793a1", line_width=1)
        flow_chart.update_layout(
            height=max(500, len(flow_rows) * 23),
            margin=dict(l=165, r=78, t=12, b=48),
            showlegend=False,
            xaxis_title="樣本成交值占比變化（百分點 pp）",
            yaxis_title=None,
            hovermode="closest",
            bargap=0.24,
        )
        flow_chart.update_xaxes(zeroline=False, gridcolor="rgba(130,145,160,0.18)")
        flow_chart.update_yaxes(autorange="reversed", automargin=True)
        st.plotly_chart(
            flow_chart,
            use_container_width=True,
            key=f"rotation_published_cashflow_{flow_day}_{len(flow_rows)}",
            config={"displaylogo": False, "responsive": True},
        )
        st.caption(
            f"資料日期：{flow_date_label}；比較基準：{flow_compare_label}。"
            "紅色代表本頁樣本成交值占比增加，綠色代表占比下降。"
            "這是代表股樣本中的成交值占比變化，不等同全市場真實資金流入／流出或法人買賣超。"
        )

    st.markdown('<div class="feature-section-title">02｜產業輪動圖</div>', unsafe_allow_html=True)
    st.caption("每條線代表主題相對加權指數的軌跡，每 5 個交易日取一點。RS 高於 100 代表相對強弱高於自身 50 日均值；動能高於 100 代表較 10 個交易日前增強。")
    n_points = {"1 個月": 6, "3 個月": 15, "半年": 28, "1 年": 53}[period]
    fig = go.Figure()
    colors = {"領先": "#c65b62", "轉弱": "#d9a441", "落後": "#64748b", "改善": "#319879"}
    all_x, all_y = [], []
    for name in selected:
        g = by_name.get(name, {})
        display_label = rotation_label(name)
        pts = [x for x in g.get("rrg", []) if isinstance(x, dict)][-n_points:]
        if not pts:
            continue
        xs = [float(x["rs"]) for x in pts]
        ys = [float(x["momentum"]) for x in pts]
        labels = [str(x.get("date", ""))[5:] for x in pts]
        color = colors.get(str(g.get("quadrant")), "#64748b")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=display_label,
            line=dict(width=2, color=color),
            marker=dict(size=[5] * max(0, len(pts) - 1) + [10], color=color),
            text=labels,
            hovertemplate=display_label + "<br>日期 %{text}<br>相對強弱 %{x:.1f}<br>相對動能 %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]], mode="text", text=[display_label],
            textposition="top center", textfont=dict(size=10, color=color),
            showlegend=False, hoverinfo="skip",
        ))
        all_x.extend(xs)
        all_y.extend(ys)
    if all_x and all_y:
        xpad = max(2.5, (max(all_x) - min(all_x)) * .12)
        ypad = max(2.5, (max(all_y) - min(all_y)) * .12)
        xr = [min(min(all_x) - xpad, 97), max(max(all_x) + xpad, 103)]
        yr = [min(min(all_y) - ypad, 97), max(max(all_y) + ypad, 103)]
        fig.add_shape(type="line", x0=100, x1=100, y0=yr[0], y1=yr[1], line=dict(color="#9aa7b5", dash="dash"))
        fig.add_shape(type="line", x0=xr[0], x1=xr[1], y0=100, y1=100, line=dict(color="#9aa7b5", dash="dash"))
        fig.update_xaxes(range=xr, title_text="相對強弱", zeroline=False)
        fig.update_yaxes(range=yr, title_text="相對動能", zeroline=False)
    fig.update_layout(height=600, margin=dict(l=25, r=20, t=12, b=24), hovermode="closest", dragmode="pan",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0), showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key=f"rotation_published_rrg_{period}_{len(selected)}",
                    config={"scrollZoom": True, "doubleClick": "reset", "displaylogo": False, "responsive": True})

    st.markdown('<div class="feature-section-title">02｜產業報酬與輪動狀態</div>', unsafe_allow_html=True)
    fields = {"近 20 日": "return_20d", "近 5 日": "return_5d", "今日": "today_return",
              "相對強弱": "relative_strength", "相對動能": "relative_momentum",
              "資金流向": "sample_turnover_share_delta_pp"}
    field = fields[sort_by]
    table_groups = [g for g in groups if not selected or g.get("name") in selected]
    table_groups.sort(key=lambda g: g.get(field) if g.get(field) is not None else -999999, reverse=True)
    def pct(g, k):
        value = g.get(k)
        return f"{float(value):+.2f}%" if value is not None and np.isfinite(float(value)) else "—"
    table_rows = []
    for g in table_groups:
        share = g.get("sample_turnover_share_pct")
        delta = g.get("sample_turnover_share_delta_pp")
        rs, momentum = g.get("relative_strength"), g.get("relative_momentum")
        table_rows.append({
            "產業主題": rotation_label(g.get("name", "")),
            "有效／樣本": f"{g.get('members_fresh', 0)}/{g.get('members_total', 0)}",
            "資料覆蓋率": f"{g.get('data_coverage_pct', 0):.0f}%",
            "今日": pct(g, "today_return"), "昨日": pct(g, "yesterday_return"),
            "5 日": pct(g, "return_5d"), "20 日": pct(g, "return_20d"),
            "60 日": pct(g, "return_60d"), "今年以來": pct(g, "return_ytd"),
            "上漲／下跌": f"{g.get('rising', 0)}/{g.get('falling', 0)}",
            "52 週新高": g.get("year_high_count", 0),
            "樣本成交值占比": f"{float(share):.1f}%" if share is not None else "—",
            "資金流向": f"{float(delta):+.2f} pp" if delta is not None else "—",
            "RS": f"{float(rs):.1f}" if rs is not None else "—",
            "動能": f"{float(momentum):.1f}" if momentum is not None else "—",
            "象限": g.get("quadrant", "資料不足"),
        })
    metrics_df = pd.DataFrame(table_rows)
    st.dataframe(
        styled_frame(metrics_df, ["今日", "昨日", "5 日", "20 日", "60 日", "今年以來", "資金流向"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="feature-section-title">04｜本週象限變化</div>', unsafe_allow_html=True)
    changes = payload.get("quadrant_changes") or []
    if changes:
        changes_df = pd.DataFrame(changes).rename(columns={"name": "產業主題", "from": "前一象限", "to": "目前象限"})
        if "產業主題" in changes_df.columns:
            changes_df["產業主題"] = changes_df["產業主題"].map(rotation_label)
        st.dataframe(changes_df, use_container_width=True, hide_index=True)
    else:
        st.caption("目前沒有可用的象限轉換紀錄，或輪動方向尚未跨越象限。")

    component_names = [n for n in (selected or list(by_name)) if by_name.get(n, {}).get("components")]
    if component_names:
        st.markdown('<div class="feature-section-title">04｜成分股技術資訊</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            chosen_label = st.selectbox(
                "檢視主題成分股",
                [rotation_label(n) for n in component_names],
                key="rotation_published_component_group",
            )
            chosen = display_to_internal.get(chosen_label, chosen_label)
        with c2:
            component_sort = st.selectbox("排序", ["今日漲幅", "5 日報酬", "20 日報酬", "成交量"], key="rotation_published_component_sort")
        comps = [dict(x) for x in by_name[chosen].get("components", [])]
        skey = {"今日漲幅": "change_1d", "5 日報酬": "return_5d", "20 日報酬": "return_20d", "成交量": "volume"}[component_sort]
        comps.sort(key=lambda x: x.get(skey) if x.get(skey) is not None else -999999, reverse=True)
        components_df = pd.DataFrame([{
            "代號": x.get("symbol"), "名稱": x.get("name"), "資料日期": x.get("data_date"),
            "收盤價": x.get("close"), "今日": pct(x, "change_1d"), "5 日": pct(x, "return_5d"),
            "20 日": pct(x, "return_20d"), "60 日": pct(x, "return_60d"), "今年以來": pct(x, "return_ytd"),
            "成交量": x.get("volume"), "52 週新高": "是" if x.get("year_high") else "否",
        } for x in comps])
        st.dataframe(
            styled_frame(components_df, ["今日", "5 日", "20 日", "60 日", "今年以來"]),
            use_container_width=True,
            hide_index=True,
        )

        # 完整分類總覽列出每個主題對應的所有代表股，即使該股票當天沒有有效行情。
        with st.expander("查看全部 27 個產業主題與成分股分類", expanded=False):
            component_lookup = {}
            for group_data in groups:
                for item in group_data.get("components", []):
                    if item.get("symbol") is not None:
                        component_lookup[(str(group_data.get("name")), str(item.get("symbol")))] = item
            categorized_rows = []
            for group_name in by_name:
                for symbol in ROTATION_THEMES.get(group_name, []):
                    item = component_lookup.get((group_name, str(symbol)), {})
                    categorized_rows.append({
                        "產業主題": rotation_label(group_name),
                        "代號": str(symbol),
                        "名稱": item.get("name") or NAME_FALLBACKS.get(str(symbol), str(symbol)),
                        "資料日期": item.get("data_date") or "尚無行情",
                        "收盤價": item.get("close"),
                        "今日": pct(item, "change_1d"),
                        "5 日": pct(item, "return_5d"),
                        "20 日": pct(item, "return_20d"),
                        "成交量": item.get("volume"),
                    })
            category_df = pd.DataFrame(categorized_rows)
            st.dataframe(
                styled_frame(category_df, ["今日", "5 日", "20 日"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("分類依代表股主要產品與供應鏈用途整理；同一檔股票可能屬於多個主題。『尚無行情』表示尚未取得可用日 K，不代表該股不屬於此類。")

    st.markdown('<div class="feature-section-title">06｜成分股漲跌前十</div>', unsafe_allow_html=True)
    upcol, downcol = st.columns(2)
    for col, data_key, label in [(upcol, "gainers", "漲幅前十"), (downcol, "losers", "跌幅前十")]:
        with col:
            st.markdown(f"**{label}**")
            movers_df = pd.DataFrame([{
                "代號": x.get("symbol"), "名稱": x.get("name"), "今日": pct(x, "change_1d"),
                "5 日": pct(x, "return_5d"), "20 日": pct(x, "return_20d"),
                "52 週新高": "是" if x.get("year_high") else "否",
            } for x in payload.get(data_key, [])])
            st.dataframe(
                styled_frame(movers_df, ["今日", "5 日", "20 日"]),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown('<div class="feature-section-title">07｜輪動解讀與資料限制</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="rotation-quadrants">
      <div class="rotation-leading"><strong>領先｜右上</strong><br>相對大盤偏強，動能持續增強。</div>
      <div class="rotation-weakening"><strong>轉弱｜右下</strong><br>相對大盤仍強，但動能減弱。</div>
      <div class="rotation-lagging"><strong>落後｜左下</strong><br>相對大盤偏弱，動能仍弱。</div>
      <div class="rotation-improving"><strong>改善｜左上</strong><br>相對大盤偏弱，但動能回升。</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<div class="rotation-note"><strong>資料來源與方法</strong><br>' + _html.escape(str(payload.get("method") or "")) +
        '<br>' + _html.escape(str(payload.get("turnover_method") or "")) +
        '<br>「樣本成交值占比變化」不是全市場真實資金流向；成分股可重複出現在不同主題，請同時參考有效樣本數與資料覆蓋率。</div>',
        unsafe_allow_html=True,
    )
    if payload.get("errors"):
        with st.expander("更新診斷"):
            st.write(payload["errors"][:20])


def render_rotation_workspace() -> None:
    """產業分析頁：優先讀取每日 15:10 發布的快照，首次尚未有快照時才使用互動計算。"""
    published = load_published_rotation(BASE)
    if published.get("groups") and published.get("data_asof"):
        _render_published_rotation(published)
        return

    st.markdown(
        '<div class="content-heading"><div class="eyebrow">SECTOR ROTATION</div>'
        '<h1>產業分析</h1>'
        '<p>比較台股供應鏈主題的相對強弱、短中期報酬與輪動方向，協助辨識領先、轉弱、落後與改善族群。</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rotation-note"><strong>計算方式</strong>：使用各主題代表股的每日報酬等權平均建立族群指數，'
        '並以元大台灣 50（0050）作為市場基準代理。主題分類為研究用途整理、不是交易所官方產業指數；'
        '同一股票可能屬於多個主題，樣本結果不代表整個族群所有成分股。資料僅供參考，不構成投資建議。</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <style>
    .rotation-note{background:#fbf1f1;border:1px solid #eadada;border-left:4px solid #8e2b2f;
        border-radius:10px;padding:13px 16px;margin:0 0 18px;color:#594b4b;font-size:12px;line-height:1.85;}
    .rotation-note strong{color:#8e2b2f;}
    .rotation-quadrants{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 14px;}
    .rotation-quadrants div{border-radius:8px;padding:10px 12px;font-size:12px;line-height:1.6;}
    .rotation-leading{background:#fce8e8;color:#8e2b2f;}
    .rotation-weakening{background:#fff1df;color:#9a5b10;}
    .rotation-lagging{background:#edf1f5;color:#455468;}
    .rotation-improving{background:#e5f4ed;color:#17694b;}
    </style>
    """, unsafe_allow_html=True)

    col_period, col_groups, col_run = st.columns([1.2, 2.8, 1])
    with col_period:
        period_choice = st.radio(
            "輪動軌跡",
            ["1 個月", "3 個月", "半年", "1 年"],
            index=1,
            horizontal=False,
            key="rotation_period_choice",
        )
    with col_groups:
        selected_groups = st.multiselect(
            "分析主題（可自行增減）",
            list(ROTATION_THEMES.keys()),
            default=[x for x in ROTATION_DEFAULT_THEMES if x in ROTATION_THEMES],
            key="rotation_selected_themes",
            help="每個主題最多使用 5 檔代表股；可重複出現在不同主題。",
        )
    with col_run:
        st.markdown("<div style='height:29px'></div>", unsafe_allow_html=True)
        run_rotation = st.button("計算產業輪動", type="primary", use_container_width=True, key="run_rotation_analysis")

    period_bars = {"1 個月": 22, "3 個月": 66, "半年": 132, "1 年": 250}[period_choice]
    # 需要留出 50 日 RS 均線與 10 日動能的 warm-up 區間。
    fetch_days = {"1 個月": 170, "3 個月": 250, "半年": 370, "1 年": 520}[period_choice]
    signature = period_choice + "|" + "|".join(sorted(selected_groups))
    stored = st.session_state.get("rotation_result")
    stored_signature = st.session_state.get("rotation_signature")

    if run_rotation:
        if not selected_groups:
            st.warning("請至少選擇一個產業主題。")
        else:
            symbols = sorted(set(["0050"] + [
                code for group in selected_groups for code in ROTATION_THEMES.get(group, [])
            ]))
            history_map = {}
            progress = st.progress(0, text="準備下載代表股歷史日 K…")
            failures = []
            with st.spinner(f"正在整理 {len(symbols)} 檔股票的歷史行情，第一次執行可能需要一些時間…"):
                for idx, code in enumerate(symbols):
                    try:
                        hist = get_history(code, fetch_days)
                        if hist is None or hist.empty:
                            failures.append(code)
                        else:
                            history_map[code] = hist.copy()
                    except Exception:
                        failures.append(code)
                    progress.progress((idx + 1) / len(symbols), text=f"讀取 {code} 歷史日 K（{idx + 1}/{len(symbols)}）")
            progress.empty()

            group_series = {}
            diagnostics = {}
            for group in selected_groups:
                series, meta = _rotation_make_index(history_map, ROTATION_THEMES[group])
                if len(series) >= 70:
                    series.attrs["valid_symbols"] = meta["valid_symbols"]
                    group_series[group] = series
                    diagnostics[group] = meta

            try:
                metrics, rrg_map = _rotation_build_frame(group_series, history_map, period_bars)
                for _, metric in metrics.iterrows():
                    name = metric["族群"]
                    metric_symbols = diagnostics.get(name, {}).get("valid_symbols", [])
                    metric["樣本數"] = len(metric_symbols)
                # 在結果物件中把內部時間序列欄位取出，避免表格直接顯示 Series。
                visible_metrics = metrics.drop(columns=["_group_index"], errors="ignore").copy()
                visible_metrics["樣本數"] = visible_metrics["族群"].map(
                    lambda name: len(diagnostics.get(name, {}).get("valid_symbols", []))
                )
                st.session_state["rotation_result"] = {
                    "metrics": visible_metrics,
                    "rrg": rrg_map,
                    "diagnostics": diagnostics,
                    "asof": max((pd.to_datetime(h["date"], errors="coerce").max() for h in history_map.values() if h is not None and not h.empty), default=pd.NaT),
                    "failures": failures,
                }
                st.session_state["rotation_signature"] = signature
                stored = st.session_state["rotation_result"]
                stored_signature = signature
            except Exception as e:
                st.error(f"產業輪動計算失敗：{e}")
                st.caption("請檢查 Fugle 歷史 K 線是否可用，或先減少選取的主題數。")
                return

    if stored is None or stored_signature != signature:
        st.info("請選擇觀察期間與主題，再按「計算產業輪動」。結果會快取在目前頁面工作階段，切換主題後需重新計算。")
        return

    metrics = stored["metrics"].copy()
    asof = pd.to_datetime(stored.get("asof"), errors="coerce")
    if pd.notna(asof):
        st.caption(f"行情資料基準：{asof.strftime('%Y-%m-%d')}｜基準代理：0050｜主題數：{len(metrics)}")
    failed = stored.get("failures") or []
    if failed:
        st.caption(f"未取得有效日 K 的股票：{', '.join(failed[:15])}" + (f" 等 {len(failed)} 檔" if len(failed) > 15 else ""))

    counts = metrics["輪動象限"].value_counts()
    leaders = metrics.loc[metrics["輪動象限"] == "領先", "族群"].tolist()
    improvers = metrics.loc[metrics["輪動象限"] == "改善", "族群"].tolist()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("可分析主題", f"{len(metrics)}")
    m2.metric("領先象限", f"{int(counts.get('領先', 0))} 群")
    m3.metric("改善象限", f"{int(counts.get('改善', 0))} 群")
    m4.metric("落後／轉弱", f"{int(counts.get('落後', 0) + counts.get('轉弱', 0))} 群")

    st.markdown('<div class="feature-section-title">01｜產業輪動圖</div>', unsafe_allow_html=True)
    st.caption("每條線代表一個主題的相對強弱軌跡；尾端是最新位置。X、Y 軸的 100 為中線。滑鼠移到線上可查看日期與數值。")
    fig = go.Figure()
    quadrant_colors = {
        "領先": "#c65b62", "轉弱": "#d9a441", "落後": "#64748b", "改善": "#319879",
    }
    all_x, all_y = [], []
    for group, frame in stored["rrg"].items():
        if frame.empty:
            continue
        # 每五個交易日取一點，並確保最新點保留在軌跡末端。
        trail = frame.iloc[::5].copy()
        if trail.empty or trail.index[-1] != frame.index[-1]:
            trail = pd.concat([trail, frame.tail(1)])
        color = quadrant_colors.get(
            metrics.loc[metrics["族群"] == group, "輪動象限"].iloc[0], "#64748b"
        )
        fig.add_trace(go.Scatter(
            x=trail["rs"], y=trail["momentum"],
            mode="lines+markers",
            name=group,
            line=dict(width=2, color=color),
            marker=dict(size=[6] * max(0, len(trail) - 1) + [11], color=color),
            text=[pd.Timestamp(d).strftime("%m/%d") for d in trail.index],
            hovertemplate=group + "<br>日期 %{text}<br>相對強弱 %{x:.1f}<br>相對動能 %{y:.1f}<extra></extra>",
        ))
        last = frame.iloc[-1]
        all_x.extend(frame["rs"].tail(20).astype(float).tolist())
        all_y.extend(frame["momentum"].tail(20).astype(float).tolist())
        fig.add_trace(go.Scatter(
            x=[float(last["rs"])], y=[float(last["momentum"])],
            mode="text",
            text=[group],
            textposition="top center",
            textfont=dict(size=10, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        # 以 100 為中心提供適度邊界，避免軸距過窄。
        x_pad = max(2.5, (max_x - min_x) * 0.12)
        y_pad = max(2.5, (max_y - min_y) * 0.12)
        x_range = [min(min_x - x_pad, 97), max(max_x + x_pad, 103)]
        y_range = [min(min_y - y_pad, 97), max(max_y + y_pad, 103)]
        fig.add_shape(type="line", x0=100, x1=100, y0=y_range[0], y1=y_range[1],
                      line=dict(color="#9aa7b5", width=1, dash="dash"))
        fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=100, y1=100,
                      line=dict(color="#9aa7b5", width=1, dash="dash"))
        fig.update_xaxes(range=x_range, title_text="相對強弱（100＝近 50 日平均）", zeroline=False)
        fig.update_yaxes(range=y_range, title_text="相對動能（100＝10 日前水準）", zeroline=False)
    fig.update_layout(
        height=620, margin=dict(l=30, r=20, t=16, b=25),
        hovermode="closest", dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True,
    )
    st.plotly_chart(
        fig, use_container_width=True, key=f"rotation_rrg_{signature}",
        config={"scrollZoom": True, "doubleClick": "reset", "displaylogo": False, "responsive": True},
    )

    st.markdown('<div class="feature-section-title">02｜族群績效與輪動狀態</div>', unsafe_allow_html=True)
    sort_by = st.selectbox(
        "排序方式",
        ["近 20 日報酬", "近 5 日報酬", "今日／最近交易日", "相對強弱", "相對動能"],
        key="rotation_sort_by",
    )
    sort_col = {
        "近 20 日報酬": "近 20 日",
        "近 5 日報酬": "近 5 日",
        "今日／最近交易日": "近 1 日",
        "相對強弱": "相對強弱",
        "相對動能": "相對動能",
    }[sort_by]
    table = metrics.sort_values(sort_col, ascending=False, na_position="last").copy()
    for col in ["近 1 日", "近 5 日", "近 20 日", "近 60 日", "今年以來"]:
        table[col] = table[col].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
    for col in ["相對強弱", "相對動能"]:
        table[col] = table[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    st.dataframe(
        table[["族群", "樣本數", "近 1 日", "近 5 日", "近 20 日", "近 60 日", "今年以來", "相對強弱", "相對動能", "輪動象限"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="feature-section-title">03｜四象限判讀</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="rotation-quadrants">
      <div class="rotation-leading"><strong>領先｜右上</strong><br>相對大盤偏強，而且相對動能仍在增強。</div>
      <div class="rotation-weakening"><strong>轉弱｜右下</strong><br>相對大盤仍偏強，但動能正在減弱。</div>
      <div class="rotation-lagging"><strong>落後｜左下</strong><br>相對大盤偏弱，而且動能仍弱。</div>
      <div class="rotation-improving"><strong>改善｜左上</strong><br>目前相對偏弱，但動能正在改善。</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("RRG 是相對績效與動能的描述工具，不等同買賣訊號。不同主題樣本股數、產業重疊及個股流動性會影響結果；請搭配原始 K 線與基本面資料判讀。")


# ============================================================
# Watchlist fragment
# ============================================================
st.markdown("""
<style>
.watch-head{font-size:20px;font-weight:950;letter-spacing:.4px;min-height:44px;display:flex;align-items:center;padding:4px 10px 8px 8px;color:#f8fafc;}
.search-wrap-spacer{height:0;}
[data-testid="stTextInput"] input{height:46px!important;min-height:46px!important;box-sizing:border-box!important;padding-top:0!important;padding-bottom:0!important;}
[data-testid="stTextInput"]{margin:0!important;display:flex!important;align-items:center!important;}
[data-testid="stTextInput"]>div>div{height:46px!important;min-height:46px!important;margin:0!important;}
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

def _watch_select(code: str) -> None:
    st.session_state.selected = code
    st.session_state.ai_selected = None
    st.session_state.ai_result = None
    st.rerun()


def _watch_remove(code: str) -> None:
    new_watchlist = [x for x in st.session_state.watchlist if x != code]
    st.session_state.watchlist = new_watchlist
    save_watchlist(new_watchlist)
    if st.session_state.get('selected') == code:
        st.session_state.selected = new_watchlist[0] if new_watchlist else WATCHLIST_DEFAULT[0]
    st.session_state.ai_selected = None
    st.session_state.ai_result = None
    st.rerun()


def render_watchlist(compact: bool = False):
    rows, errors = watch_rows(tuple(st.session_state.watchlist))
    if not compact:
        st.subheader("自選股即時行情")
    if not rows.empty and "data_date" in rows.columns:
        valid_dates = sorted({str(v) for v in rows["data_date"].tolist() if str(v).strip()})
    # compact=True：放在左側欄，不需要固定高度（側邊欄自己會捲動）
    with (st.container() if compact else st.container(height=500, border=True)):
        if not compact:
            heads = st.columns([3.25, 0.82, 0.82, 1.7, 1.65, 1.75], gap="small")
            head_labels = ["股票中文名", "查看", "刪除", "當前市價", "+/−價格", "+/−價格%"]
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
                    value_class = ""
                    if name_class in ("limit-up", "limit-down"):
                        vcolor = "#ffffff"
                        value_class = " " + name_class
                    price_txt = "--" if pd.isna(price) else f"{price:,.2f}"
                    ch_txt = "--" if pd.isna(ch) else f"{ch:+,.2f}"
                    cp_txt = "--" if pd.isna(cp) else f"{cp:+.2f}%"

                    if compact:
                        # 一檔股票 = 一張完整卡片：名稱、操作按鈕、現價／漲跌／漲跌幅都在同一個框裡
                        with st.container(key=f"wc_{r['code']}_{idx}"):
                            b1, b2, b3 = st.columns([5.4, 1, 1], gap="small")
                            with b1:
                                st.markdown(
                                    f"<div class='wc-info {name_class}'>"
                                    f"<div class='wc-name'>{r['name']}<span>（{r['code']}）</span></div>"
                                    "<div class='wc-vals'>"
                                    f"<span class='wc-price' style='color:{vcolor}'>{price_txt}</span>"
                                    f"<span style='color:{vcolor}'>{ch_txt}</span>"
                                    f"<span style='color:{vcolor}'>{cp_txt}</span>"
                                    "</div></div>",
                                    unsafe_allow_html=True,
                                )
                            with b2:
                                if st.button("\u200b", key=f"watch_select_{r['code']}_{idx}", icon=":material/visibility:", help="選取此股票", use_container_width=True):
                                    _watch_select(r['code'])
                            with b3:
                                if st.button("\u200b", key=f"watch_remove_{r['code']}_{idx}", icon=":material/delete:", help=f"從自選股移除 {r['code']}", use_container_width=True):
                                    _watch_remove(r['code'])
                    else:
                        b1,b2,b3,b4,b5,b6 = st.columns([3.25, 0.82, 0.82, 1.7, 1.65, 1.75], gap="small")
                        with b1:
                            st.markdown(
                                f"<div class='watch-name-display {name_class}'>{r['name']}（{r['code']}）</div>",
                                unsafe_allow_html=True,
                            )
                        with b2:
                            if st.button("\u200b", key=f"watch_select_{r['code']}_{idx}", icon=":material/visibility:", use_container_width=True, type="secondary", help="查看此股票"):
                                _watch_select(r['code'])
                        with b3:
                            if st.button("\u200b", key=f"watch_remove_{r['code']}_{idx}", icon=":material/delete:", use_container_width=True, type="secondary", help=f"從自選股移除 {r['code']}"):
                                _watch_remove(r['code'])
                        with b4:
                            st.markdown(f"<div class='watch-value watch-price-cell{value_class}' style='color:{vcolor}'>{price_txt}</div>", unsafe_allow_html=True)
                        with b5:
                            st.markdown(f"<div class='watch-value watch-price-cell{value_class}' style='color:{vcolor}'>{ch_txt}</div>", unsafe_allow_html=True)
                        with b6:
                            st.markdown(f"<div class='watch-value watch-price-cell{value_class}' style='color:{vcolor}'>{cp_txt}</div>", unsafe_allow_html=True)
                    # 只有同一產業內的股票之間使用細分隔線；產業之間使用更醒目的分組線。
                    if original_idx != group_records[-1][0]:
                        st.markdown("<div class='watch-row-divider'></div>", unsafe_allow_html=True)
    if errors:
        with st.expander("資料連線診斷"):
            st.code("\n".join(errors[:30]))


# ============================================================
# Selected header + fixed live card
# ============================================================
def render_selected_and_live(show_heading: bool = True):
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
    if show_heading:
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

    # 重要：即時看板不再使用 fixed/absolute 或固定寬度 HTML 包裝。
    # 直接交給 Streamlit 原生 st.columns，在桌面並排、窄螢幕依原生規則折疊。
    c_stock, c_index = st.columns([2, 1], gap="large", vertical_alignment="top")
    with c_stock:
        st.markdown(f"""
        <div class="live-card"><div class="live-cell">
          <div class="live-title">目前選取股票｜盤中即時資料</div>
          <div class="live-main">{name}（{selected}）</div>
          <div class="live-price">{'--' if pd.isna(lp) else f'{lp:,.2f}'}</div>
          <div class="live-change" style="color:{color};">{'--' if pd.isna(lch) else f'{lch:+,.2f}'}　{'--' if pd.isna(lcp) else f'{lcp:+.2f}%'} </div>
        </div></div>""", unsafe_allow_html=True)
    with c_index:
        st.markdown(f"""
        <div class="live-card"><div class="live-cell">
          <div class="live-title">台灣加權指數</div>
          <div class="live-main">TAIEX</div><div class="live-price">{ixline}</div>
          <div class="live-change" style="color:{ixcolor};">{ixsub}</div>
          <div class="live-time">{ixtime}</div>
        </div></div>""", unsafe_allow_html=True)


# ============================================================
# Trend
# ============================================================
def _sample_axis_ticks(category_values, display_labels, max_ticks: int = 9):
    """減少 X 軸日期標籤；保留完整類別，避免非交易日留下空白或日期碰撞。"""
    values = list(category_values)
    labels = list(display_labels)
    n = min(len(values), len(labels))
    values, labels = values[:n], labels[:n]
    if n <= max_ticks:
        return values, labels
    step = max(1, int(np.ceil((n - 1) / max(1, max_ticks - 1))))
    indexes = list(range(0, n, step))
    if indexes[-1] != n - 1:
        indexes.append(n - 1)
    return [values[i] for i in indexes], [labels[i] for i in indexes]


def _chart_time_labels(d: pd.DataFrame, daily: bool = False):
    """回傳唯一的類別軸值與簡短顯示文字。"""
    stamps = pd.to_datetime(d["date"], errors="coerce")
    if daily:
        return stamps.dt.strftime("%Y-%m-%d").tolist(), stamps.dt.strftime("%m/%d").tolist()
    if stamps.dt.date.nunique() > 1:
        return stamps.dt.strftime("%Y-%m-%d %H:%M").tolist(), stamps.dt.strftime("%m/%d %H:%M").tolist()
    return stamps.dt.strftime("%H:%M").tolist(), stamps.dt.strftime("%H:%M").tolist()


def render_trend():
    selected = st.session_state.selected
    st.subheader("股價趨勢")
    period = st.radio("期間", ["日內", "五日", "近月", "三月", "一年"], horizontal=True, key="period")
    try:
        if period == "日內":
            if is_market_open_now():
                d = normalize_intraday(get_intraday(selected, "1"))
            else:
                prev = latest_trading_date(selected)
                d = pd.DataFrame() if prev is None else normalize_intraday(client.historical_candles(selected, prev, prev, "1"))
            if d.empty:
                raise RuntimeError("目前沒有可用的盤中 1 分鐘資料")
            chart_x, display_labels = _chart_time_labels(d, daily=False)
            y = d["close"]
        else:
            days = {"五日": 15, "近月": 45, "三月": 120, "一年": 420}[period]
            d = get_history(selected, days)
            if d.empty:
                raise RuntimeError("沒有足夠日 K 資料")
            chart_x, display_labels = _chart_time_labels(d, daily=True)
            y = d["close"]

        tickvals, ticktext = _sample_axis_ticks(chart_x, display_labels, max_ticks=9)
        fig = go.Figure(go.Scatter(x=chart_x, y=y, mode="lines", name="價格"))
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=16, b=24),
            showlegend=False,
            yaxis_title="價格",
        )
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=chart_x,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"trend_{selected}_{period}")
    except Exception as e:
        st.error(f"股價趨勢資料取得失敗：{e}")

# ============================================================
# Technical K
# ============================================================
def render_kline():
    """互動式技術 K 線：日期簡寫、交易日等距排列，價格與成交量共用時間軸。"""
    selected = st.session_state.selected
    st.subheader("技術 K 線")
    k_period = st.radio(
        "K 線週期",
        ["日K", "當日 5 分K", "當日 30 分K", "當日 60 分K"],
        horizontal=True,
        key="technical_k_period",
    )

    period_days = {"3 個月": 100, "6 個月": 210, "1 年": 420, "2 年": 800, "5 年": 1850}
    if k_period == "日K":
        control_col, reset_col = st.columns([5, 1])
        with control_col:
            history_period = st.radio(
                "顯示期間",
                list(period_days.keys()),
                index=1,
                horizontal=True,
                key=f"technical_history_period_{selected}",
            )
    else:
        history_period = "當日"
        control_col, reset_col = st.columns([5, 1])
        with control_col:
            st.caption("盤中 K 線顯示最近可用交易日；可縮放查看細節。")

    reset_state_key = f"technical_chart_reset_version_{selected}_{k_period}"
    with reset_col:
        st.markdown(
            """<style>
            div[class*="st-key-technical-reset-button"] [data-testid="stButton"] > button {
                background:#fbf1f1 !important;
                background-color:#fbf1f1 !important;
                color:#8e2b2f !important;
                border:1px solid #e6d3d3 !important;
                border-radius:8px !important;
                box-shadow:none !important;
                font-weight:700 !important;
                white-space:nowrap !important;
            }
            div[class*="st-key-technical-reset-button"] [data-testid="stButton"] > button:hover {
                background:#f5e4e4 !important;
                border-color:#d8b9b9 !important;
                color:#6f2024 !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        with st.container(key="technical-reset-button"):
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            if st.button(
                "回到最新",
                key=f"technical_reset_latest_{selected}_{k_period}",
                help="重設圖表範圍，回到此期間的最新資料",
            ):
                st.session_state[reset_state_key] = int(st.session_state.get(reset_state_key, 0)) + 1
    reset_version = int(st.session_state.get(reset_state_key, 0))

    try:
        if k_period == "日K":
            d = get_history(selected, period_days[history_period])
        else:
            tf = {"當日 5 分K": "5", "當日 30 分K": "30", "當日 60 分K": "60"}[k_period]
            if is_market_open_now():
                d = normalize_intraday(get_intraday(selected, tf))
            else:
                prev = latest_trading_date(selected)
                d = pd.DataFrame() if prev is None else normalize_intraday(
                    client.historical_candles(selected, prev, prev, tf)
                )

        if d.empty:
            raise RuntimeError("目前沒有可用的 K 線資料")

        d = d.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        d = (
            d.dropna(subset=["date", "open", "high", "low", "close"])
             .drop_duplicates(subset=["date"], keep="last")
             .sort_values("date")
             .reset_index(drop=True)
        )
        if d.empty:
            raise RuntimeError("K 線資料中的開、高、低、收欄位皆無有效數值")
        if "volume" not in d.columns:
            d["volume"] = 0
        d["volume"] = d["volume"].fillna(0).clip(lower=0)

        chart_x, display_labels = _chart_time_labels(d, daily=(k_period == "日K"))
        tickvals, ticktext = _sample_axis_ticks(chart_x, display_labels, max_ticks=9)

        ma_specs = [
            ("MA5", 5, True), ("MA10", 10, False), ("MA20", 20, True),
            ("MA60", 60, True), ("MA120", 120, False), ("MA240", 240, False),
        ]
        ma_cols = st.columns(6)
        enabled_ma = {}
        for col, (label, window, default_on) in zip(ma_cols, ma_specs):
            with col:
                enabled_ma[window] = st.checkbox(
                    label,
                    value=default_on,
                    key=f"technical_ma_{selected}_{k_period}_{history_period}_{label}",
                )

        # 沒有圖內主標題／子圖標題，讓 K 棒與 X 軸有更大的可用空間。
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                            row_heights=[0.76, 0.24])
        fig.add_trace(
            go.Candlestick(
                x=chart_x,
                open=d["open"],
                high=d["high"],
                low=d["low"],
                close=d["close"],
                name="K 線",
                increasing_line_color="#ef4444",
                decreasing_line_color="#10b981",
                increasing_fillcolor="#ef4444",
                decreasing_fillcolor="#10b981",
            ),
            row=1,
            col=1,
        )

        ma_colors = {5: "#2563eb", 10: "#f59e0b", 20: "#8b5cf6", 60: "#0891b2", 120: "#64748b", 240: "#a16207"}
        for label, window, _default_on in ma_specs:
            if enabled_ma.get(window):
                ma = d["close"].rolling(window=window, min_periods=window).mean()
                fig.add_trace(
                    go.Scatter(
                        x=chart_x,
                        y=ma,
                        mode="lines",
                        name=label,
                        line=dict(width=1.4, color=ma_colors[window]),
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )

        volume_colors = [
            "#ef4444" if close >= open_ else "#10b981"
            for open_, close in zip(d["open"], d["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=chart_x,
                y=d["volume"],
                name="成交量",
                marker_color=volume_colors,
                hovertemplate="%{x}<br>成交量 %{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=680,
            margin=dict(l=18, r=24, t=28, b=24),
            hovermode="x unified",
            dragmode="pan",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            uirevision=f"{selected}-{k_period}-{history_period}-{reset_version}",
            bargap=0.18,
        )
        for row_no in (1, 2):
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=chart_x,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=True,
                gridcolor="#e6eaee",
                rangeslider_visible=False,
                row=row_no,
                col=1,
            )
        fig.update_yaxes(showgrid=True, gridcolor="#e6eaee", row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#e6eaee", row=2, col=1)

        st.caption("操作：滾輪／雙指縮放、按住拖曳平移、點兩下重設；也可使用「回到最新」。非交易日不占用圖表空間。")
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"kline_{selected}_{k_period}_{history_period}_{reset_version}",
            config={"scrollZoom": True, "doubleClick": "reset", "displaylogo": False, "responsive": True},
        )
    except Exception as e:
        st.error(f"{k_period} 資料取得失敗：{e}")

# ============================================================
# 成交量概況
# ============================================================
def render_volume_snapshot():
    selected = st.session_state.selected
    st.subheader("成交量概況")
    try:
        if is_market_open_now():
            d = normalize_intraday(get_intraday(selected, "5"))
            label = "當日 5 分K"
            daily = False
        else:
            d = get_history(selected, 45)
            label = "最近 45 日"
            daily = True
        if d.empty or "volume" not in d.columns:
            st.info("目前沒有足夠的成交量資料。")
            return

        d = d.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
        # 僅留下實際存在的有效成交量紀錄，休市日／週末不補列、不產生空值。
        d = (
            d.dropna(subset=["date", "volume"])
             .drop_duplicates(subset=["date"], keep="last")
             .sort_values("date")
             .tail(60)
             .reset_index(drop=True)
        )
        if d.empty:
            st.info("目前沒有有效的成交量資料。")
            return

        current = float(d.iloc[-1]["volume"])
        avg5 = float(d["volume"].tail(5).mean()) if len(d) >= 5 else float(d["volume"].mean())
        ratio = current / avg5 if avg5 else np.nan
        a, b, c = st.columns(3)
        a.metric("最新成交量", f"{current:,.0f}")
        b.metric("5期平均", f"{avg5:,.0f}")
        c.metric("量／均量", "--" if pd.isna(ratio) else f"{ratio:.2f}x")

        chart_x, display_labels = _chart_time_labels(d, daily=daily)
        tickvals, ticktext = _sample_axis_ticks(chart_x, display_labels, max_ticks=8)
        colors = [
            "#ef4444" if float(close) >= float(open_) else "#10b981"
            for open_, close in zip(d.get("open", d["volume"]), d.get("close", d["volume"]))
        ]
        vf = go.Figure(go.Bar(
            x=chart_x,
            y=d["volume"],
            name="成交量",
            marker_color=colors,
            hovertemplate="%{x}<br>成交量 %{y:,.0f}<extra></extra>",
        ))
        vf.update_layout(height=300, margin=dict(l=20, r=20, t=14, b=24), showlegend=False)
        vf.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=chart_x,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        )
        vf.update_yaxes(title_text="量")
        st.plotly_chart(vf, use_container_width=True, key=f"volume_snapshot_{selected}_{label}")
    except Exception as e:
        st.warning(f"成交量資料取得失敗：{e}")

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
        st.plotly_chart(pie, use_container_width=True, key=f"tdcc_{selected}")
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
            st.plotly_chart(fig,use_container_width=True,key=f"inst_chart_{selected}")
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
    # Email Agent：保留「手動執行後依設定自動寄信」與「最近一次寄送結果」提示；
    # 原本的「AI 盤後財報 Email Agent」寄送面板已依需求移除，所以寄送按鈕固定為 False。
    try:
        email_agent = EmailAgent(BASE)
        email_status = email_agent.status()
    except Exception:
        email_agent, email_status = None, {}
    email_send_latest = False
    email_force = False
    st.markdown("## 模型研究 Agent")
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

    if run_research:
        st.session_state.research_busy = True
        try:
            symbols = list(dict.fromkeys(st.session_state.watchlist))
            with st.spinner("本機模型研究 Agent 正在分析：財報、營收、法說會、接單、近期產業漲跌、新聞與模型健康…"):
                agent = ResearchAgent(BASE, ollama_model=st.session_state.get("ollama_model", "qwen3:8b"))
                report = agent.run_daily_research(symbols=symbols, sector_codes=["24", "26", "28"])
            st.session_state.research_result = report
            # Dashboard 手動執行也可沿用與 Windows 排程相同的 Email Agent。
            if email_agent is not None and email_status.get("configured") and email_status.get("auto_send"):
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

    rc2, rc3 = st.columns(2)
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
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(tdf, use_container_width=True, hide_index=True)
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



# ============================================================
# V9 site visual system - only header/footer typography and section backgrounds
# ============================================================
st.markdown(r"""
<style>
/* 頁首：刪除狀態列，放大導覽字體，維持 PDF 的簡潔淺色風格 */
.brand-name{font-size:24px!important;letter-spacing:1.8px!important;}
.site-nav{gap:34px!important;padding:10px 4px 14px!important;}
.site-nav-link{font-size:15px!important;letter-spacing:.7px!important;padding:4px 0!important;}
.site-rule{margin:0 0 18px!important;}

/* 頁尾：字體明顯放大，但仍保留設計稿的深色收尾 */
.footer-title{font-size:20px!important;}
.footer-contact p,.footer-form-note p{font-size:13px!important;line-height:1.9!important;}
.footer-small{font-size:12px!important;line-height:1.9!important;}
.footer-form-note a{font-size:13px!important;}
.footer-bottom{font-size:11px!important;padding:16px 12px 20px!important;}

/* 每一個功能節點皆用淺色背景，避免只有首頁有完整視覺語言 */
.st-key-strategy-01-watchlist,.st-key-strategy-02-selected,.st-key-strategy-03-trend,
.st-key-strategy-04-kline,.st-key-strategy-05-volume,.st-key-strategy-06-entry,
.st-key-strategy-07-tdcc,.st-key-strategy-08-trades,.st-key-strategy-09-institutions,
.st-key-future-01-ai,.st-key-future-02-intelligence,.st-key-future-03-agent{
  border-radius:14px!important;
  padding:22px 24px 18px!important;
  margin:0 0 14px!important;
  border:1px solid rgba(142,43,47,.10)!important;
  box-shadow:none!important;
}
.st-key-strategy-01-watchlist{background:#fbf1f1!important;}
.st-key-strategy-02-selected{background:#fffaf7!important;}
.st-key-strategy-03-trend{background:#f8f4f2!important;}
.st-key-strategy-04-kline{background:#f9f5ef!important;}
.st-key-strategy-05-volume{background:#f6f5f1!important;}
.st-key-strategy-06-entry{background:#fbf3f3!important;}
.st-key-strategy-07-tdcc{background:#f4f7f6!important;}
.st-key-strategy-08-trades{background:#f8f6f3!important;}
.st-key-strategy-09-institutions{background:#f5f3f4!important;}
.st-key-future-01-ai{background:#f4f6fa!important;}
.st-key-future-02-intelligence{background:#fbf5f5!important;}
.st-key-future-03-agent{background:#f5f7f4!important;}

/* 功能標題與內容跟著節點一起呼吸，避免內容貼死邊緣 */
.feature-section-title{margin-top:0!important;}

/* Footer 後延伸同色底，避免網頁尾端再出現白色區塊 */
.site-footer{position:relative!important;margin-bottom:0!important;}
@media(max-width:900px){.site-nav{gap:22px!important;}.site-nav-link{font-size:14px!important;}.footer-title{font-size:19px!important;}}
@media(max-width:600px){.brand-name{font-size:21px!important;}.site-nav{gap:16px 20px!important;}.site-nav-link{font-size:13px!important;}.st-key-strategy-01-watchlist,.st-key-strategy-02-selected,.st-key-strategy-03-trend,
.st-key-strategy-04-kline,.st-key-strategy-05-volume,.st-key-strategy-06-entry,.st-key-strategy-07-tdcc,.st-key-strategy-08-trades,.st-key-strategy-09-institutions,
.st-key-future-01-ai,.st-key-future-02-intelligence,.st-key-future-03-agent{padding:16px 12px 12px!important;}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Final web layout polish - 只處理頁首／頁尾外框留白，不改內容版型
# ============================================================
st.markdown(r"""
<style>
/* ===== 只移除 Streamlit 頁首產生的空白 ===== */
html, body, #root { margin:0 !important; padding:0 !important; }
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"] {
  display:none !important;
  height:0 !important;
  min-height:0 !important;
}
[data-testid="stAppViewContainer"] {
  padding:0 !important;
  background:#fff !important;
}
section.main,
.main {
  padding:0 !important;
  margin:0 !important;
  background:transparent !important;
}
section.main > div,
.main > div,
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
  padding-top:0 !important;
  margin-top:0 !important;
}
.main .block-container {
  padding-top:0 !important;
  padding-bottom:0 !important;
  margin-top:0 !important;
  margin-bottom:0 !important;
}
[data-testid="stVerticalBlock"] { gap:.5rem !important; }

/* ===== 頁面背景：依 PDF 維持整體淺色 =====
   深色只用於頁尾，不把整個 Streamlit App 染成深色。 */
html, body, #root, .stApp {
  background:#fff !important;
  color:var(--site-ink) !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
section.main,
.main,
.main .block-container {
  background:transparent !important;
  min-height:0 !important;
}
.site-footer {
  margin-bottom:0 !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Final layout safety - based on PDF: light body + native Streamlit columns
# ============================================================
st.markdown(r"""
<style>
html,body,#root,.stApp{background:#fff!important;color:var(--site-ink)!important;}
[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],[data-testid="stMainBlockContainer"],section.main,.main,.main .block-container{background:transparent!important;}
.main .block-container{padding-top:0!important;padding-bottom:0!important;margin-top:0!important;margin-bottom:0!important;}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stAppDeployButton"]{display:none!important;height:0!important;min-height:0!important;}
section.main>div,.main>div{padding-top:0!important;margin-top:0!important;min-height:0!important;height:auto!important;}
[data-testid="stVerticalBlock"]{gap:.5rem!important;}
.st-key-strategy-01-watchlist{background:#fbf1f1!important;}
.st-key-strategy-02-selected{background:#fffaf7!important;}
.st-key-strategy-03-trend{background:#f8f4f2!important;}
.st-key-strategy-04-kline{background:#f9f5ef!important;}
.st-key-strategy-05-volume{background:#f6f5f1!important;}
.st-key-strategy-06-entry{background:#fbf3f3!important;}
.st-key-strategy-07-tdcc{background:#f4f7f6!important;}
.st-key-strategy-08-trades{background:#f8f6f3!important;}
.st-key-strategy-09-institutions{background:#f5f3f4!important;}
.st-key-future-01-ai{background:#f4f6fa!important;}
.st-key-future-02-intelligence{background:#fbf5f5!important;}
.st-key-future-03-agent{background:#f5f7f4!important;}
</style>
""", unsafe_allow_html=True)


st.markdown(r'''
<style>
/* ============================================================
   V11 visual correction
   - 不限制 Streamlit 原生 columns
   - 全站維持 PDF 的淺色系，頁尾維持原深色並做滿版
   - 頁首與內容保留呼吸距離
   ============================================================ */
html, body, #root, .stApp {
  background:#fff !important;
  color:var(--site-ink) !important;
}
.main .block-container {
  max-width:none !important;
  width:auto !important;
  padding-top:0 !important;
  padding-bottom:0 !important;
  margin-top:0 !important;
  margin-bottom:0 !important;
}

/* 頁首：不增加頂端大空白，只增加頁首與第一個內容區的距離 */
.brand-block { padding-top:8px !important; }
.site-rule { margin-top:4px !important; margin-bottom:30px !important; }
.site-nav-link { font-size:15px !important; }
.site-status-line { display:none !important; }

/* 所有頁面標題下方的介紹型小字移除 */
.content-heading p,
.finance-card-sub,
.section-sub {
  display:none !important;
}
.content-heading { padding-bottom:8px !important; }

/* 大標與下面的按鈕／表格保留間距 */
.feature-section-title {
  margin-top:0 !important;
  margin-bottom:26px !important;
  padding-top:10px !important;
  padding-bottom:12px !important;
}

/* 搜尋欄：藍灰、圓角、乾淨的視覺 */
[data-testid="stTextInput"] input {
  background:#eef2f5 !important;
  border:1px solid #a8b7c5 !important;
  border-radius:999px !important;
  color:#253746 !important;
  height:48px !important;
  min-height:48px !important;
  padding:0 18px !important;
  box-shadow:inset 0 1px 2px rgba(37,55,70,.05) !important;
}
[data-testid="stTextInput"] input:focus {
  border-color:#58758e !important;
  box-shadow:0 0 0 3px rgba(88,117,142,.12) !important;
}
.stButton > button {
  min-height:44px !important;
  border:1px solid #92a6b6 !important;
  background:#eef2f5 !important;
  color:#263746 !important;
  font-weight:800 !important;
  border-radius:10px !important;
  box-shadow:none !important;
}
.stButton > button:hover {
  border-color:#58758e !important;
  background:#e4ebf0 !important;
  color:#1f3343 !important;
}
.stButton > button[kind="primary"] {
  background:#58758e !important;
  border-color:#58758e !important;
  color:#fff !important;
}
.stButton > button[kind="primary"]:hover {
  background:#47657c !important;
  border-color:#47657c !important;
  color:#fff !important;
}

/* 自選股表格：原生 6 columns，避免巢狀 columns 造成名稱欄過窄 */
.watch-head {
  color:#334155 !important;
  background:rgba(255,255,255,.55) !important;
  font-size:16px !important;
  min-height:40px !important;
  padding:9px 8px !important;
  border-radius:8px !important;
}
.watch-name-display {
  background:rgba(255,255,255,.62) !important;
  min-height:40px !important;
  display:flex !important;
  align-items:center !important;
  padding:8px 10px !important;
  color:#334155 !important;
}
.watch-name-display.up { color:#b91c1c !important; }
.watch-name-display.down { color:#047857 !important; }
.watch-name-display.flat { color:#475569 !important; }
.watch-name-display.limit-up { background:#991b1b !important; color:#fff !important; }
.watch-name-display.limit-down { background:#166534 !important; color:#fff !important; }
.watch-value {
  min-height:40px !important;
  padding:8px !important;
  font-size:18px !important;
  background:rgba(255,255,255,.45) !important;
  border-radius:7px !important;
}
.watch-price-cell {
  border-left:1px solid rgba(100,116,139,.20) !important;
  padding-left:10px !important;
}
.watch-industry-header {
  color:#334155 !important;
  background:linear-gradient(90deg,#e8f1f7,#f7f9fb) !important;
  border-color:#c6d4df !important;
  font-size:18px !important;
}
.watch-industry-code { color:#64748b !important; }
.watch-industry-dot { background:#6c8ba3 !important; }
.watch-business-header {
  color:#475569 !important;
  background:#f5f7f9 !important;
  border-left-color:#9cb0c0 !important;
  font-size:14px !important;
}
button[title="查看此股票"],
button[title^="從自選股移除"] {
  width:100% !important;
  min-width:0 !important;
  height:40px !important;
  min-height:40px !important;
  padding:4px 6px !important;
  font-size:14px !important;
  line-height:1 !important;
}

/* 保留頁尾深色，不改色；讓深色直接橫跨整個視窗，消除左右淺色落差 */
.site-footer {
  background:var(--site-footer) !important;
  width:100vw !important;
  max-width:none !important;
  margin-left:calc(50% - 50vw) !important;
  margin-right:calc(50% - 50vw) !important;
  margin-bottom:0 !important;
  box-sizing:border-box !important;
}
.footer-inner { width:100% !important; box-sizing:border-box !important; }

/* 不覆寫 Streamlit column 的寬度；窄螢幕由 Streamlit 自己處理折疊 */
@media(max-width:768px){
  .site-rule { margin-bottom:22px !important; }
  .site-nav { gap:18px 24px !important; }
  .site-nav-link { font-size:14px !important; }
  .feature-section-title { margin-bottom:22px !important; }
  .watch-head { font-size:13px !important; }
  .watch-value { font-size:16px !important; }
  .watch-name-display { font-size:14px !important; }
}
</style>
''', unsafe_allow_html=True)


st.markdown(r'''<style>
/* ============================================================
   V12 final UI audit
   ============================================================ */
.focus-taiwan-layout {
  grid-template-columns:1fr 1fr !important;
  gap:42px !important;
}
.focus-main-visual {
  width:100%;
  min-width:0;
  min-height:300px;
  height:100%;
  border-radius:10px;
  overflow:hidden;
  box-shadow:0 8px 24px rgba(0,0,0,.08);
}
.focus-main-visual img {
  display:block;
  width:100%;
  height:100%;
  min-height:300px;
  object-fit:cover;
}

/* 搜尋：輸入框較長，按鈕緊鄰 */
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) {
  column-gap:8px !important;
}
.search-field-row {
  column-gap:8px !important;
}
.search-field-row [data-testid="stTextInput"],
.search-field-row [data-testid="stTextInput"] > div {
  width:100% !important;
}

/* 所有按鈕：淺色底＋深色字 */
.stButton > button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
[data-testid="stFormSubmitButton"] > button {
  background:#edf2f5 !important;
  color:#263746 !important;
  border:1px solid #9eb0bf !important;
  border-radius:10px !important;
  font-weight:800 !important;
  box-shadow:none !important;
  opacity:1 !important;
}
.stButton > button:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="baseButton-primary"]:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  background:#e1e9ef !important;
  color:#1f3343 !important;
  border-color:#58758e !important;
}

/* 查看／刪除 */
button[title="查看此股票"],
button[title^="從自選股移除"],
button[title="查看並切換至此股票"] {
  background:#edf2f5 !important;
  color:#263746 !important;
}
button[title="查看此股票"] span,
button[title^="從自選股移除"] span,
button[title="查看並切換至此股票"] span,
button[title="查看此股票"] .material-symbols-rounded,
button[title^="從自選股移除"] .material-symbols-rounded,
button[title="查看並切換至此股票"] .material-symbols-rounded {
  color:#263746 !important;
}

/* 自選股分類標題 */
.watch-industry-header,
.watch-industry-header * { color:#334155 !important; }
.watch-business-header,
.watch-business-header * { color:#475569 !important; }

/* 訂閱／聯絡表單 */
[data-testid="stForm"] label,
[data-testid="stForm"] [data-testid="stWidgetLabel"] {
  color:#34404a !important;
  opacity:1 !important;
}
[data-testid="stForm"] input,
[data-testid="stForm"] textarea {
  color:#263746 !important;
  background:#fff !important;
  border:1px solid #c7d0d8 !important;
}
[data-testid="stForm"] input::placeholder,
[data-testid="stForm"] textarea::placeholder { color:#8a949d !important; }

/* 大標與下面內容不要黏住 */
.feature-section-title { margin-bottom:24px !important; }

@media(max-width:700px){
  .focus-taiwan-layout { grid-template-columns:1fr !important; }
  .focus-main-visual, .focus-main-visual img { min-height:220px !important; }
}
</style>''', unsafe_allow_html=True)

st.markdown(r'''<style>
/* 最後一層按鈕對比檢查：壓過前面任何 secondary/primary 的舊 CSS */
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button {
  background:#edf2f5 !important;
  color:#263746 !important;
  border:1px solid #9eb0bf !important;
  font-weight:800 !important;
  opacity:1 !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  background:#e1e9ef !important;
  color:#1f3343 !important;
  border-color:#58758e !important;
}
</style>''', unsafe_allow_html=True)

st.markdown(r'''<style>
/* ============================================================
   V13 user tweaks
   ============================================================ */
:root{--title-red:#8e2b2f;--soft-blue:#dbeafe;--soft-blue-hover:#c7dcf8;}

/* 所有下拉選單（expander）：統一淺灰藍色，滑過時變成稍深的淺色，不再出現深色底 */
.stApp [data-testid="stExpander"],
.stApp [data-testid="stExpander"] details{
  background:#f4f6f9 !important;
  border:1px solid #dde3ea !important;
  border-radius:10px !important;
  overflow:hidden;
}
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] details > summary{
  background:#ebecef !important;
  transition:background .15s ease;
}
.stApp [data-testid="stExpander"] summary:hover,
.stApp [data-testid="stExpander"] details > summary:hover,
.stApp [data-testid="stExpander"] details[open] > summary:hover{
  background:#dde5ef !important;
}
.stApp [data-testid="stExpander"] summary svg{color:var(--title-red) !important;fill:var(--title-red) !important;}
.stApp [data-testid="stExpanderDetails"]{background:transparent !important;}

/* 圖表／表格外框改淺色 */
.stApp [data-testid="stPlotlyChart"],
.stApp [data-testid="stDataFrame"]{
  background:rgba(255,255,255,.55) !important;
  border-radius:10px;
}

/* 搜尋欄：輸入框撐滿到搜尋按鈕旁 */
[data-testid="stTextInput"],
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] > div > div{
  width:100% !important;
  max-width:none !important;
  flex:1 1 auto !important;
}
[data-testid="stTextInput"] input{width:100% !important;}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]){column-gap:6px !important;}

/* 查看／刪除按鈕：淺藍填滿 + 黑色框線 */
[class*="st-key-watch_select_"] button,
[class*="st-key-watch_remove_"] button,
.stApp button[title="查看此股票"],
.stApp button[title^="從自選股移除"]{
  background:var(--soft-blue) !important;
  border:1px solid #000 !important;
  color:#263746 !important;
}
[class*="st-key-watch_select_"] button:hover,
[class*="st-key-watch_remove_"] button:hover,
.stApp button[title="查看此股票"]:hover,
.stApp button[title^="從自選股移除"]:hover{
  background:var(--soft-blue-hover) !important;
  border-color:#000 !important;
}

/* 白字 → 與標題相同的紅色。只改容器本身，深色底區塊（頁尾、橫幅、即時報價卡、漲跌停標籤）自有的白字設定不受影響 */
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] *,
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] *,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary *,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stAlertContainer"] p{
  color:var(--title-red) !important;
}
</style>''', unsafe_allow_html=True)


st.markdown(r'''<style>
/* ============================================================
   V14 layout trim + form cleanup + tech polish
   ============================================================ */

/* 頁首上方留白：每個只放 <style> 的 st.markdown 都會佔一格 .5rem 的間距，
   累積起來造成頂端大片空白。把這些空容器收掉（樣式本身仍然生效）。 */
[data-testid="stElementContainer"]:has(style),
.element-container:has(style){
  display:none !important;
}
[data-testid="stMainBlockContainer"],
.main .block-container{padding-top:0 !important;}
.brand-block{padding-top:6px !important;}
.brand-mark{margin-bottom:4px !important;}
.site-nav{padding-top:6px !important;padding-bottom:10px !important;}
.site-rule{
  border-top:0 !important;
  height:1px !important;
  background:linear-gradient(90deg,rgba(142,43,47,0),rgba(142,43,47,.55) 50%,rgba(142,43,47,0)) !important;
  margin:0 0 22px !important;
}

/* ---------- 訂閱／聯絡表單 ---------- */
[data-testid="stForm"]{
  background:#fff !important;
  border:1px solid #dde3ea !important;
  border-radius:14px !important;
  padding:22px 24px 18px !important;
  box-shadow:0 6px 20px rgba(15,23,42,.05) !important;
}
[data-testid="stForm"] [data-testid="stVerticalBlock"]{gap:.9rem !important;}
/* 標籤改回在欄位上方（原本被全站的 flex 設定擠成同一列） */
[data-testid="stForm"] [data-testid="stTextInput"],
[data-testid="stForm"] [data-testid="stTextArea"]{
  display:block !important;
  width:100% !important;
}
[data-testid="stForm"] [data-testid="stWidgetLabel"]{
  margin-bottom:6px !important;
  min-height:0 !important;
}
[data-testid="stForm"] [data-testid="stWidgetLabel"] p{
  font-size:13px !important;
  font-weight:800 !important;
  letter-spacing:.6px !important;
}
/* 欄位外框（baseweb 包了一層深色底，造成圓角處露出深色） */
[data-testid="stForm"] [data-baseweb="input"],
[data-testid="stForm"] [data-baseweb="base-input"],
[data-testid="stForm"] [data-baseweb="textarea"]{
  background:#fff !important;
  border:1px solid #c7d0d8 !important;
  border-radius:10px !important;
  box-shadow:none !important;
  overflow:hidden;
  transition:border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stForm"] [data-baseweb="input"]:focus-within,
[data-testid="stForm"] [data-baseweb="textarea"]:focus-within{
  border-color:#58758e !important;
  box-shadow:0 0 0 3px rgba(88,117,142,.14) !important;
}
[data-testid="stForm"] [data-baseweb="input"] > div,
[data-testid="stForm"] [data-baseweb="base-input"]{background:transparent !important;}
[data-testid="stForm"] input,
[data-testid="stForm"] textarea{
  background:transparent !important;
  border:0 !important;
  border-radius:0 !important;
  box-shadow:none !important;
  height:44px !important;
  min-height:44px !important;
  padding:0 14px !important;
  color:#263746 !important;
}
[data-testid="stForm"] textarea{
  height:auto !important;
  min-height:150px !important;
  padding:12px 14px !important;
  line-height:1.6 !important;
}
/* 勾選框：淺色框，勾選後用標題紅 */
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > span:first-of-type{
  background:#fff !important;
  border:1.5px solid #58758e !important;
  border-radius:5px !important;
}
[data-testid="stCheckbox"] label:has(input:checked) > span:first-of-type{
  background:var(--title-red) !important;
  border-color:var(--title-red) !important;
}
[data-testid="stCheckbox"] label p{font-size:14px !important;}
/* 送出按鈕 */
[data-testid="stFormSubmitButton"] > button{
  min-width:132px !important;
  height:44px !important;
  border-radius:10px !important;
  letter-spacing:2px !important;
}

/* ---------- 整體俐落／科技感 ---------- */
:root{--tech-line:#dde3ea;--tech-radius:12px;}
html{scroll-behavior:smooth;}
/* 數字等寬，行情表格與指標對齊更整齊 */
[data-testid="stMetricValue"],[data-testid="stDataFrame"],.watch-value,.live-price{
  font-variant-numeric:tabular-nums;
}
/* 指標卡 */
[data-testid="stMetric"]{
  background:#fff;
  border:1px solid var(--tech-line);
  border-left:3px solid var(--title-red);
  border-radius:var(--tech-radius);
  padding:12px 16px;
  box-shadow:0 3px 12px rgba(15,23,42,.04);
}
/* 提示框、表格、圖表：統一細線框與圓角 */
[data-testid="stAlert"]{border-radius:var(--tech-radius) !important;border:1px solid rgba(88,117,142,.18) !important;}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{
  border:1px solid var(--tech-line);
  border-radius:var(--tech-radius);
  overflow:hidden;
}
/* 導覽列：滑過時底線由中間展開 */
.site-nav-link{position:relative;border-bottom:0 !important;}
.site-nav-link::after{
  content:"";position:absolute;left:50%;right:50%;bottom:-2px;height:2px;
  background:var(--title-red);transition:left .2s ease,right .2s ease;
}
.site-nav-link:hover::after,.site-nav-link.active::after{left:0;right:0;}
/* 按鈕：輕微浮起回饋 */
.stButton > button,[data-testid="stFormSubmitButton"] > button{
  transition:background .15s ease,border-color .15s ease,transform .15s ease,box-shadow .15s ease !important;
}
.stButton > button:hover,[data-testid="stFormSubmitButton"] > button:hover{
  transform:translateY(-1px);
  box-shadow:0 4px 12px rgba(37,55,70,.12) !important;
}
/* 細捲軸 */
*{scrollbar-width:thin;scrollbar-color:#b9c6d2 transparent;}
</style>''', unsafe_allow_html=True)


st.markdown(r'''<style>
/* ============================================================
   V15 搜尋欄新樣式 / 表單淺膚色 / 按鈕特效 / 漲跌停整列
   ============================================================ */

/* ---------- 搜尋欄：灰框膠囊 + 藍色放大鏡端蓋 ---------- */
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]){
  column-gap:0 !important;gap:0 !important;align-items:stretch !important;
}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-baseweb="input"],
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-baseweb="base-input"]{
  background:#fff !important;
  border:4px solid #b3b3b3 !important;
  border-right:0 !important;
  border-radius:999px 0 0 999px !important;
  height:52px !important;
  box-shadow:none !important;
}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) input{
  background:transparent !important;border:0 !important;border-radius:0 !important;
  height:44px !important;padding:0 20px !important;color:#263746 !important;
}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-testid="stButton"] > button,
.st-key-do_stock_search_site button{
  height:52px !important;min-height:52px !important;width:100% !important;
  background:#0072c6 !important;border:0 !important;
  border-radius:0 999px 999px 0 !important;
  font-size:0 !important;color:transparent !important;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'><circle cx='20' cy='20' r='11' fill='none' stroke='white' stroke-width='5'/><line x1='28.5' y1='28.5' x2='40' y2='40' stroke='white' stroke-width='6' stroke-linecap='round'/></svg>") !important;
  background-repeat:no-repeat !important;background-position:center !important;background-size:26px 26px !important;
  box-shadow:none !important;
}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-testid="stButton"] > button:hover,
.st-key-do_stock_search_site button:hover{background-color:#005fa8 !important;}
/* 按鈕內任何文字／段落一律不顯示，只留放大鏡圖示 */
.st-key-do_stock_search_site button *,
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-testid="stButton"] > button *{
  display:none !important;font-size:0 !important;color:transparent !important;
}
.st-key-do_stock_search_site button::before,
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-testid="stButton"] > button::before{display:none !important;}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-baseweb="input"]:focus-within{
  border-color:#0072c6 !important;
}

/* ---------- 訂閱／聯絡表單：淺膚色（與頁面 blush 同色），去掉深色底 ---------- */
[data-testid="stForm"]{background:#fff !important;}
[data-testid="stForm"] [data-testid="stTextInputRootElement"],
[data-testid="stForm"] [data-testid="stTextAreaRootElement"],
[data-testid="stForm"] [data-baseweb="input"],
[data-testid="stForm"] [data-baseweb="base-input"],
[data-testid="stForm"] [data-baseweb="textarea"],
[data-testid="stForm"] [data-baseweb="textarea"] > div{
  background:var(--site-blush) !important;
  background-color:var(--site-blush) !important;
}
[data-testid="stForm"] [data-testid="stTextInputRootElement"],
[data-testid="stForm"] [data-testid="stTextAreaRootElement"]{
  border:1px solid #e6d3d3 !important;border-radius:10px !important;overflow:hidden;
}
[data-testid="stForm"] [data-baseweb="input"],
[data-testid="stForm"] [data-baseweb="textarea"]{border:0 !important;}
[data-testid="stForm"] input,[data-testid="stForm"] textarea{
  background:transparent !important;background-color:transparent !important;
  color:#3a2f2f !important;-webkit-text-fill-color:#3a2f2f !important;caret-color:#8e2b2f;
}
[data-testid="stForm"] input::placeholder,[data-testid="stForm"] textarea::placeholder{color:#a89a9a !important;}
[data-testid="stForm"] [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stForm"] [data-testid="stTextAreaRootElement"]:focus-within{
  border-color:#8e2b2f !important;box-shadow:0 0 0 3px rgba(142,43,47,.12) !important;
}
[data-testid="stCheckbox"] label > span:first-child,
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > span:first-child,
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child{
  background:var(--site-blush) !important;background-color:var(--site-blush) !important;
  border:1.5px solid #8e2b2f !important;border-radius:5px !important;
}
[data-testid="stCheckbox"] label:has(input:checked) > span:first-child,
[data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input:checked) > span:first-child,
[data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input:checked) > div:first-child{
  background:#8e2b2f !important;background-color:#8e2b2f !important;
}

/* ---------- 所有按鈕特效：掃光 + 浮起 + 按下回彈 + 聚焦光環 ---------- */
.stButton > button,
[data-testid="stFormSubmitButton"] > button,
[data-testid="stDownloadButton"] > button,
a[data-testid^="stBaseLinkButton"]{
  position:relative !important;overflow:hidden !important;
  transition:transform .18s ease, box-shadow .18s ease, background-color .18s ease, border-color .18s ease !important;
}
.stButton > button::before,
[data-testid="stFormSubmitButton"] > button::before,
[data-testid="stDownloadButton"] > button::before{
  content:"";position:absolute;top:0;left:-70%;width:45%;height:100%;
  background:linear-gradient(105deg,rgba(255,255,255,0),rgba(255,255,255,.55),rgba(255,255,255,0));
  transform:skewX(-20deg);pointer-events:none;
}
.stButton > button:hover::before,
[data-testid="stFormSubmitButton"] > button:hover::before,
[data-testid="stDownloadButton"] > button:hover::before{
  left:130%;transition:left .6s ease;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover{
  transform:translateY(-2px) !important;
  box-shadow:0 8px 18px rgba(37,55,70,.18) !important;
}
.stButton > button:active,
[data-testid="stFormSubmitButton"] > button:active,
[data-testid="stDownloadButton"] > button:active{
  transform:translateY(0) scale(.96) !important;
  box-shadow:0 2px 6px rgba(37,55,70,.18) !important;
}
.stButton > button:focus-visible,
[data-testid="stFormSubmitButton"] > button:focus-visible{
  outline:0 !important;box-shadow:0 0 0 3px rgba(0,114,198,.35) !important;
}

/* ---------- 自選股：漲停深紅底白字／跌停深綠底白字（整列） ---------- */
.watch-name-display.limit-up,.watch-value.limit-up{
  background:#7f1d1d !important;color:#fff !important;border:1px solid #5f1414 !important;
}
.watch-name-display.limit-down,.watch-value.limit-down{
  background:#14532d !important;color:#fff !important;border:1px solid #0b3a1f !important;
}
.watch-value.limit-up *,.watch-value.limit-down *,
.watch-name-display.limit-up *,.watch-name-display.limit-down *{color:#fff !important;}

/* ---------- V16 財經資訊頁 ---------- */

        .fin-digest{background:#fff;border:1px solid #dde3ea;border-left:4px solid #8e2b2f;border-radius:12px;padding:18px 22px;margin:8px 0 18px;box-shadow:0 4px 16px rgba(15,23,42,.05);}
        .fin-digest h3{margin:0 0 6px;font-size:18px;letter-spacing:.5px;}
        .fin-digest .fin-headline{font-size:16px;font-weight:800;line-height:1.7;margin-bottom:10px;}
        .fin-stats{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 12px;}
        .fin-stat{background:#f4f6f9;border:1px solid #dde3ea;border-radius:10px;padding:6px 14px;font-size:13px;}
        .fin-stat b{font-size:18px;margin-left:4px;}
        .fin-list{margin:6px 0 0 0;padding-left:20px;line-height:1.85;font-size:14.5px;}
        .fin-two{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:12px;}
        @media(max-width:800px){.fin-two{grid-template-columns:1fr;}}
        .fin-sub{font-weight:800;font-size:14px;margin:4px 0;letter-spacing:.5px;}
        .fin-chip{display:inline-block;background:#ebecef;border:1px solid #dde3ea;border-radius:999px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12.5px;}
        .fin-chip.stock{background:#fbf1f1;border-color:#e6d3d3;}
        .fin-chip.hot{background:#fde8e8;border-color:#e6b9b9;font-weight:800;}
        .fin-badge{display:inline-block;border-radius:6px;padding:1px 8px;margin-right:6px;font-size:12px;font-weight:800;color:#fff;}
        .fin-badge.up{background:#b3121a;} .fin-badge.down{background:#14532d;} .fin-badge.mix{background:#8a6d1d;} .fin-badge.flat{background:#64748b;}
        mark.fin-hl{background:linear-gradient(transparent 55%,#ffd3d3 55%);color:#b3121a !important;font-weight:800;padding:0 1px;}
        .fin-meta{font-size:12.5px;opacity:.85;margin:6px 0;}
        .fin-why{font-size:13.5px;margin:6px 0;padding:6px 10px;background:#fbf1f1;border-radius:8px;}
        .fin-stale{background:#fff4e5;border:1px solid #f0c27a;border-radius:10px;padding:8px 14px;margin:8px 0;font-size:13.5px;}
        
        details.fin-item{background:#fff;border:1px solid #dde3ea;border-radius:10px;margin:8px 0;overflow:hidden;}
        details.fin-item > summary{cursor:pointer;list-style:none;padding:10px 14px;background:#ebecef;font-weight:700;line-height:1.6;transition:background .15s ease;}
        details.fin-item > summary::-webkit-details-marker{display:none;}
        details.fin-item > summary::before{content:"▸";display:inline-block;margin-right:8px;color:#8e2b2f;transition:transform .15s ease;}
        details.fin-item[open] > summary::before{transform:rotate(90deg);}
        details.fin-item > summary:hover{background:#dde5ef;}
        details.fin-item .fin-body{padding:8px 16px 12px;}
        details.fin-item a{color:#0072c6 !important;font-weight:700;}

/* ============================================================
   V17 操作策略：自選股固定在左側欄
   ============================================================ */
[data-testid="stSidebar"]{background:#fbf1f1 !important;border-right:1px solid #e6d3d3 !important;}
[data-testid="stSidebar"][aria-expanded="true"]{min-width:360px !important;max-width:360px !important;width:360px !important;}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{padding:10px 14px 0 !important;min-height:0 !important;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-bottom:24px;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:4px 14px 20px !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.35rem !important;}
[data-testid="stSidebar"] .side-watch-title{margin:2px 0 8px !important;font-size:18px !important;}
.watch-cv-legend{display:grid;grid-template-columns:1.25fr 1fr 1fr;gap:6px;padding:0 4px 2px;font-size:11.5px;letter-spacing:1px;opacity:.7;text-align:right;}
.watch-cv{display:grid;grid-template-columns:1.25fr 1fr 1fr;gap:6px;margin:-2px 0 4px;}
.watch-cv > div{background:#fff;border:1px solid #e6d3d3;border-radius:8px;padding:6px 8px;font-size:14px;font-weight:900;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}
.watch-cv > div.limit-up{background:#7f1d1d;border-color:#5f1414;color:#fff !important;}
.watch-cv > div.limit-down{background:#14532d;border-color:#0b3a1f;color:#fff !important;}
[data-testid="stSidebar"] .watch-name-display{font-size:14px;padding:8px 10px;}
[data-testid="stSidebar"] .watch-industry-header{margin:8px 0 5px;padding:7px 10px;font-size:15px !important;}
[data-testid="stSidebar"] .watch-business-header{font-size:13px !important;padding:4px 10px;}
/* 查看／刪除：側邊欄裡縮成小按鈕（保留淺藍底＋黑框） */
[data-testid="stSidebar"] [data-testid="stButton"] > button{
  height:36px !important;min-height:36px !important;width:100% !important;
  padding:0 4px !important;font-size:13px !important;line-height:1 !important;
  display:flex !important;align-items:center !important;justify-content:center !important;
  white-space:nowrap !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button p{font-size:13px !important;margin:0 !important;}

/* 不翻譯用的零高度元件：不要佔版面 */
[data-testid="stElementContainer"]:has(iframe[height="0"]),
.element-container:has(iframe[height="0"]){height:0 !important;min-height:0 !important;margin:0 !important;overflow:hidden !important;}

/* ---------- V18 自選股卡片：一檔一張整體卡片、圖示按鈕（資訊垂直置中） ---------- */
[class*="st-key-wc_"]{
  background:#fff;border:1px solid #e6d3d3;border-radius:12px;
  padding:10px 12px !important;margin:0 0 8px;gap:0 !important;
  box-shadow:0 2px 8px rgba(142,43,47,.06);
  transition:box-shadow .15s ease,border-color .15s ease;
}
[class*="st-key-wc_"]:hover{border-color:#c98f8f;box-shadow:0 4px 14px rgba(142,43,47,.14);}
[class*="st-key-wc_"]:has(.wc-info.limit-up){background:#7f1d1d;border-color:#5f1414;}
[class*="st-key-wc_"]:has(.wc-info.limit-down){background:#14532d;border-color:#0b3a1f;}
[class*="st-key-wc_"] [data-testid="stHorizontalBlock"]{align-items:center !important;gap:6px !important;margin:0 !important;}
[class*="st-key-wc_"] [data-testid="stColumn"]{display:flex;flex-direction:column;justify-content:center;}
[class*="st-key-wc_"] [data-testid="stElementContainer"],[class*="st-key-wc_"] .element-container{margin:0 !important;}
.wc-info{display:flex;flex-direction:column;justify-content:center;gap:3px;min-height:56px;}
.wc-name{font-size:15px;font-weight:900;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.wc-name span{font-weight:700;font-size:12.5px;opacity:.85;margin-left:2px;}
.wc-info.up .wc-name{color:#c62828;} .wc-info.down .wc-name{color:#0f8a5f;} .wc-info.flat .wc-name{color:#475569;}
.wc-info.limit-up .wc-name,.wc-info.limit-down .wc-name{color:#fff !important;}
.wc-vals{display:flex;align-items:baseline;justify-content:space-between;gap:10px;font-variant-numeric:tabular-nums;font-weight:800;font-size:14px;line-height:1.2;}
.wc-vals .wc-price{font-size:20px;font-weight:900;letter-spacing:.3px;}
[class*="st-key-wc_"]:has(.wc-info.limit-up) .wc-vals span,
[class*="st-key-wc_"]:has(.wc-info.limit-down) .wc-vals span{color:#fff !important;}
/* 選取／刪除：Material Symbols 圖示按鈕（淺藍底＋黑框，圓角正方形，圖示置中）
   注意：全站有 div[data-testid=stButton] > button[kind=secondary] 的 !important 規則，這裡的選擇器要更具體才蓋得過 */
.stApp [class*="st-key-wc_"] div[data-testid="stButton"]{display:flex !important;justify-content:center !important;align-items:center !important;}
.stApp [class*="st-key-wc_"] div[data-testid="stButton"] > button[kind]{
  width:38px !important;min-width:38px !important;max-width:38px !important;
  height:38px !important;min-height:38px !important;max-height:38px !important;
  padding:0 !important;margin:0 !important;border-radius:10px !important;
  background:#dbeafe !important;border:1.5px solid #000 !important;box-shadow:none !important;
  display:flex !important;align-items:center !important;justify-content:center !important;
  gap:0 !important;line-height:1 !important;
}
.stApp [class*="st-key-wc_"] div[data-testid="stButton"] > button[kind]:hover{background:#c7dcf8 !important;transform:translateY(-1px);}
.stApp [class*="st-key-wc_"] div[data-testid="stButton"] > button[kind] > div,
.stApp [class*="st-key-wc_"] div[data-testid="stButton"] > button[kind] [data-testid="stMarkdownContainer"]{
  display:flex !important;align-items:center !important;justify-content:center !important;
  width:100% !important;height:100% !important;margin:0 !important;padding:0 !important;
}
.stApp [class*="st-key-wc_"] div[data-testid="stButton"] > button[kind] p{
  margin:0 !important;padding:0 !important;line-height:1 !important;text-align:center !important;
  font-family:"Noto Serif TC","Source Han Serif TC","Songti TC","PMingLiU","MingLiU",Georgia,serif !important;
  font-size:19px !important;font-weight:900 !important;letter-spacing:0 !important;
  transform:translateY(-1px);   /* 補償明體字形重心偏下，讓字看起來在正中心 */
}
/* Material Symbols 圖示本身著色；按鈕不再顯示「選／刪」文字。 */
.stApp [class*="st-key-wc_"] div[data-testid="stButton"] > button [data-testid="stIconMaterial"],
.stApp [class*="st-key-wc_"] div[data-testid="stButton"] > button .material-symbols-rounded {
  display:inline-flex !important;align-items:center !important;justify-content:center !important;
  width:22px !important;height:22px !important;font-size:22px !important;line-height:1 !important;
}
.stApp [class*="st-key-watch_select_"] button p,
.stApp [class*="st-key-watch_select_"] button [data-testid="stIconMaterial"],
.stApp [class*="st-key-watch_select_"] button .material-symbols-rounded{color:#0b5cad !important;}
.stApp [class*="st-key-watch_remove_"] button p,
.stApp [class*="st-key-watch_remove_"] button [data-testid="stIconMaterial"],
.stApp [class*="st-key-watch_remove_"] button .material-symbols-rounded{color:#b3121a !important;}

/* ============================================================
   V19 淺色系：搜尋框、下拉選單、單選鈕、輸入框（不再出現深色底）
   ============================================================ */
/* 搜尋框：白底灰框膠囊，右接藍色放大鏡 */
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-testid="stTextInputRootElement"]{
  background:#fff !important;background-color:#fff !important;
  border:3px solid #c3ccd5 !important;border-right:0 !important;
  border-radius:999px 0 0 999px !important;height:52px !important;
  box-shadow:none !important;overflow:hidden;
}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-testid="stTextInputRootElement"]:focus-within{
  border-color:#0072c6 !important;box-shadow:0 0 0 3px rgba(0,114,198,.14) !important;
}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-baseweb="input"],
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) [data-baseweb="base-input"]{
  background:transparent !important;background-color:transparent !important;border:0 !important;border-radius:0 !important;height:100% !important;
}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) input{
  background:transparent !important;color:#263746 !important;-webkit-text-fill-color:#263746 !important;
  caret-color:#0072c6;padding:0 22px !important;font-size:16px !important;
}
div[data-testid="stHorizontalBlock"]:has(input[placeholder*="股票代號或公司名稱"]) input::placeholder{color:#8a97a3 !important;-webkit-text-fill-color:#8a97a3 !important;}

/* 其他文字輸入框：淺膚色 */
.stApp [data-testid="stTextInputRootElement"],
.stApp [data-testid="stNumberInputContainer"],
.stApp [data-testid="stTextAreaRootElement"]{
  background:#fbf1f1 !important;border:1px solid #e6d3d3 !important;border-radius:10px !important;
}
.stApp [data-testid="stTextInputRootElement"] [data-baseweb],
.stApp [data-testid="stNumberInputContainer"] [data-baseweb],
.stApp [data-testid="stTextAreaRootElement"] [data-baseweb]{background:transparent !important;border:0 !important;}
.stApp [data-testid="stTextInputRootElement"] input,
.stApp [data-testid="stNumberInputContainer"] input,
.stApp [data-testid="stTextAreaRootElement"] textarea{
  background:transparent !important;color:#3a2f2f !important;-webkit-text-fill-color:#3a2f2f !important;
}

/* 下拉選單（selectbox / multiselect）：淺色 */
.stApp [data-baseweb="select"] > div{
  background:#fbf1f1 !important;background-color:#fbf1f1 !important;
  border:1px solid #e6d3d3 !important;border-radius:10px !important;box-shadow:none !important;
}
.stApp [data-baseweb="select"] > div:hover,.stApp [data-baseweb="select"] > div:focus-within{border-color:#8e2b2f !important;}
.stApp [data-baseweb="select"] *{color:#3a2f2f !important;-webkit-text-fill-color:#3a2f2f !important;}
.stApp [data-baseweb="select"] svg{fill:#8e2b2f !important;color:#8e2b2f !important;}
.stApp [data-baseweb="tag"]{background:#e6d3d3 !important;}
/* 展開後的選項清單（浮在最上層，不在 .stApp 底下） */
[data-baseweb="popover"] > div,
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul{background:#fff !important;background-color:#fff !important;border-radius:10px !important;}
[data-baseweb="popover"] li,[data-baseweb="popover"] li *{color:#3a2f2f !important;background:transparent !important;}
[data-baseweb="popover"] li:hover,[data-baseweb="popover"] li[aria-selected="true"]{background:#fbf1f1 !important;}

/* 單選鈕：未選取＝白底灰框（原本是黑色實心） */
.stApp [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child{background:#fff !important;border:2px solid #9aa8b5 !important;}
.stApp [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child{background:#8e2b2f !important;border-color:#8e2b2f !important;}
.stApp [data-testid="stRadio"] label p{color:#3a2f2f !important;}

/* 財經新聞篩選列：淺膚色面板 */
[class*="st-key-fin_filters"]{
  background:#fbf1f1;border:1px solid #e6d3d3;border-radius:12px;padding:10px 14px 4px !important;margin:6px 0 10px;
}
</style>''', unsafe_allow_html=True)

# ============================================================
# App entry point
# ============================================================
# Streamlit 預設把 <html lang="en"> 送給瀏覽器，Chrome 會把中文頁面當成英文並跳出「翻譯」，
# 翻譯後會重排中文字（例如「操作策略 →」變成「操作→策略」）並造成 React removeChild 錯誤。
# 這裡改成 zh-TW 並加上 notranslate。
import streamlit.components.v1 as _components
_components.html(
    """<script>
    try {
      const d = window.parent.document;
      d.documentElement.setAttribute('lang', 'zh-TW');
      d.documentElement.setAttribute('translate', 'no');
      if (!d.querySelector('meta[name="google"]')) {
        const m = d.createElement('meta'); m.name = 'google'; m.content = 'notranslate'; d.head.appendChild(m);
      }
      d.body.classList.add('notranslate');
    } catch (e) {}
    </script>""",
    height=0,
)
render_header_and_search()
current_section = render_site_navigation()

if current_section == "home":
    render_home_workspace()
elif current_section == "strategy":
    render_strategy_workspace()
elif current_section == "future":
    render_future_workspace()
elif current_section == "finance":
    render_finance_workspace()
elif current_section == "rotation":
    render_rotation_workspace()
else:
    render_subscription_workspace()

if REFRESH_INTERVAL and not st.session_state.get("ai_busy", False):
    st_autorefresh(interval=int(REFRESH_INTERVAL * 1000), key="site_market_live_refresh")

# Footer 必須是整個網頁最後一個可見內容。
_render_footer()