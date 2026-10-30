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
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
import re
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import extra_streamlit_components as stx


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
            "AUTH_USERS_JSON",
            "AUTH_SESSION_SECRET",
            "ACCOUNT_DB_URL",
            "ACCOUNT_DB_SERVICE_ROLE_KEY",
            "ACCOUNT_PUBLIC_BASE_URL",
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


def _refresh_helper_modules() -> None:
    """Streamlit Cloud 推新版後常只重跑主程式、不重新 import 子模組，網站會拿舊模組配新主程式而出錯
    （2026-10-26 系統推薦顯示「沒有紀錄」、2026-10-29 AttributeError: symbol_flow_rows）。
    每個伺服器程序第一次執行時把常改的子模組各重新載入一次，之後檔案比載入時新才再載入。"""
    import importlib
    import sys as _sys
    names = ("market_calendar", "watchlist_store", "stock_features", "chip_signals", "intraday_signals",
             "industry_rotation", "value_chain", "section_ui", "site_logo", "future_picks_ui", "trend_pattern_ui", "stock_notes_ui")
    for name in names:
        mod = _sys.modules.get(name)
        if mod is None or not getattr(mod, "__file__", None):
            continue
        try:
            mtime = os.path.getmtime(mod.__file__)
            if getattr(mod, "_loaded_mtime", None) != mtime:
                mod = importlib.reload(mod)
                mod._loaded_mtime = mtime
        except Exception:
            pass


_refresh_helper_modules()

import watchlist_store
from market_calendar import is_twse_trading_day, market_status_text
from stock_api import (
    FugleClient, clean_symbol, INDUSTRIES,
    fetch_institution_daily, fetch_tdcc_big_retail, fetch_taiex_intraday,
)
from industry_master import build_master, apply_overrides, INDUSTRY_OVERRIDES
from business_master import business_map_for_symbols
# ai_engine／entry_model 會載入 scikit-learn（約 1～2 秒），只在用到模型的函式內才 import。
from market_agent import AgentContext, MarketSupervisorAgent
from market_intelligence import render_market_intelligence
from email_agent import EmailAgent
from subscriber_service import register_subscriber, subscription_status
from auth_service import (authenticate, request_account, create_session_token, verify_session_token,
                         SESSION_COOKIE_NAME, SESSION_TTL_SECONDS)
from industry_rotation import ROTATION_THEMES, ROTATION_DISPLAY_NAMES, NAME_FALLBACKS, load_published_rotation
from trend_pattern_ui import render_trend_patterns_workspace
from trend_pattern_engine import CATEGORY_ORDER as TREND_CATEGORY_ORDER
from stock_notes_ui import render_stock_notes_workspace, render_stock_note_for_symbol
from ai_backend import is_agent_status, status_label, AGENT_LABEL
from us_industry_indicators import US_INDUSTRY_INDICATORS, get_us_indicator_symbols, fetch_us_indicators
from celebrity_holdings import POLITICIANS, fetch_all_gurus
import chip_signals
import stock_features
import intraday_signals
import section_ui
import site_logo
from future_picks_ui import render_system_picks  # 模組改名過：雲端只改子模組時不會重新載入
from section_ui import section as _sec, section_box as _sec_box
from global_markets import GLOBAL_INDICATORS, fetch_indicator, vix_regime
from html import escape as _esc


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_us_industry_quotes() -> pd.DataFrame:
    """全部美股指標股＋SPY 一次查詢並快取 1 小時（台股時段美股休市，價格不會變）。
    產業頁各區塊共用同一份，不再因篩選條件不同而重複下載。"""
    return pd.DataFrame(fetch_us_indicators(get_us_indicator_symbols() + ["SPY"]))

BASE = Path(__file__).resolve().parent
DATA, MODELS, OUTPUT = BASE / "data", BASE / "models", BASE / "output"
DAILY_DATA = DATA / "daily"
SITE_IMAGE_DIR = BASE / "static" / "img"  # Streamlit 靜態檔，網址為 app/static/img/<檔名>
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
    """讀取自選股：先讀 Supabase（Cloud 重新部署會重置本機檔案），再退回 user_watchlist.json、預設清單。"""
    remote = watchlist_store.load()
    if remote:
        return remote
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
    """將自選股存進 Supabase（永久）與本機檔案；資料庫寫入失敗回傳 False。"""
    try:
        WATCHLIST_FILE.write_text(json.dumps(list(dict.fromkeys(items)), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return watchlist_store.save(items)



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


st.set_page_config(page_title="台股即時互動式分析系統", page_icon=_site_favicon(), layout="wide", initial_sidebar_state="collapsed")


# ---- 開站載入動畫：第一次開啟或重新整理時蓋住頁面，整頁跑完才淡出；站內換頁不播放 ----
# 一輪 3.6 秒：logo 圓環轉入 → 環內 K 線一根根長出（紅綠）、趨勢線往上 → K 線轉成 logo 色、外側斜線長出，停在完整 logo
# 頁面載入時主執行緒會被圖表塞住 1 秒以上，所以只動 transform／opacity 並讓每個圖形自成一層 div（GPU 合成，主執行緒忙也不停）；
# 不要改回 stroke-dashoffset、color 這類動畫，同一個動畫也不要混進 visibility（會整個退回主執行緒），也不要用 :has()（每次 DOM 變動都要重算樣式）。
_BOOT_BASE_CSS = """
.boot-loader{position:fixed;inset:0;z-index:1000000;display:flex;align-items:center;justify-content:center;
  background:radial-gradient(circle at 50% 42%,#fffdfc 0%,#fbf3f2 62%,#f4e6e5 100%);will-change:opacity;
  animation:boot-out .4s ease 90s forwards,boot-gone .01s linear 90.4s forwards}
.boot-loader.boot-done{pointer-events:none;animation:boot-out .65s cubic-bezier(.4,0,.2,1) .18s forwards,boot-gone .01s linear .83s forwards}
.boot-loader.boot-done .boot-inner{animation:boot-inner-out .4s cubic-bezier(.4,0,.2,1) forwards}
@keyframes boot-out{to{opacity:0}}
@keyframes boot-gone{to{visibility:hidden;pointer-events:none}}
@keyframes boot-inner-out{to{opacity:0;transform:scale(1.05)}}
.boot-inner{display:flex;flex-direction:column;align-items:center;width:min(340px,84vw);will-change:transform,opacity}
.boot-mark{position:relative;width:min(148px,42vw);aspect-ratio:1}
.boot-layer{position:absolute;inset:0;will-change:transform,opacity}
.boot-layer svg{display:block;width:100%;height:100%;overflow:visible}
.boot-ring{transform-origin:50% 50%;animation:boot-ring 3.6s cubic-bezier(.3,.7,.3,1) infinite both}
@keyframes boot-ring{0%{opacity:0;transform:rotate(-40deg) scale(.86)}16%{opacity:1;transform:none}93%{opacity:1;transform:none}100%{opacity:0;transform:none}}
.boot-sweep{transform-origin:50% 50%;animation:boot-sweep 1.5s linear infinite}
@keyframes boot-sweep{to{transform:rotate(360deg)}}
.boot-candle{opacity:0;animation:boot-candle 3.6s cubic-bezier(.2,.8,.2,1) infinite both}
@keyframes boot-candle{0%,14%{opacity:0;transform:scaleY(.08)}26%{opacity:1;transform:none}62%{opacity:1;transform:none}74%,100%{opacity:0;transform:none}}
.boot-candle-end{opacity:0;animation:boot-candle-end 3.6s ease-in-out infinite both}
@keyframes boot-candle-end{0%,62%{opacity:0}74%{opacity:1}93%{opacity:1}100%{opacity:0}}
.boot-trend{opacity:0;transform-origin:31.5% 66.7%;animation:boot-trend 3.6s cubic-bezier(.3,.7,.3,1) infinite both}
@keyframes boot-trend{0%,26%{opacity:0;transform:scale(.05)}30%{opacity:1}44%{opacity:1;transform:none}56%{opacity:1;transform:none}66%,100%{opacity:0;transform:none}}
.boot-tail{opacity:0;transform-origin:73% 73%;animation:boot-tail 3.6s cubic-bezier(.2,.8,.2,1) infinite both}
@keyframes boot-tail{0%,64%{opacity:0;transform:scale(.05)}68%{opacity:1}78%{opacity:1;transform:none}93%{opacity:1;transform:none}100%{opacity:0;transform:none}}
.boot-title{margin-top:24px;font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:21px;letter-spacing:.14em;color:#8e2b2f;text-align:center;line-height:1.5}
.boot-track{position:relative;width:148px;height:2px;margin-top:16px;border-radius:2px;background:rgba(142,43,47,.14);overflow:hidden}
.boot-track::after{content:"";position:absolute;top:0;bottom:0;left:0;width:38%;border-radius:2px;background:#8e2b2f;will-change:transform;animation:boot-slide 1.4s ease-in-out infinite}
@keyframes boot-slide{0%{transform:translateX(-110%)}100%{transform:translateX(300%)}}
.boot-note{margin-top:12px;font-size:12.5px;letter-spacing:.08em;color:#8a7676}
@media (prefers-reduced-motion:reduce){
  .boot-ring,.boot-candle-end,.boot-tail{animation:none;opacity:1;transform:none}
  .boot-candle,.boot-trend,.boot-sweep{animation:none;opacity:0}
  .boot-track::after{animation:none;width:100%}
}
"""
# 播放時：不顯示任何按鈕（#sb-toggle、回到頂端是 JS 加到 body 的，層級比動畫高）。放在單獨的 CSS，跑完就拿掉；
# 90 秒和 .boot-loader 的保險同步，頁面跑太久時動畫自動消失，按鈕也一起回來。
# 不要用 content-visibility 藏住底下的頁面：實測反而更慢（跑完 33 秒對 21 秒）。
_BOOT_PLAYING_CSS = """
body > *:not(#root),.stApp button,.stApp a,.stApp [role="button"],.stApp [data-testid="stSidebar"],.stApp [data-testid="stHeader"],
.stApp [data-testid="stExpandSidebarButton"],.stApp [data-testid="stSidebarCollapsedControl"]{animation:boot-controls 90s step-end}
@keyframes boot-controls{0%,100%{visibility:hidden}}
"""
# 跑完時：頁面內容在覆蓋層淡出的同時往上浮入，不是直接露出來（下一輪重跑這段 CSS 就消失）
_BOOT_DONE_CSS = """
.stApp [class*="st-key-page_"]{animation:boot-page-in .8s cubic-bezier(.2,.7,.2,1) .12s both}
.stApp .site-nav,.stApp .brand-block{animation:boot-page-in .7s cubic-bezier(.2,.7,.2,1) .08s both}
@keyframes boot-page-in{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
/* 收合、回到頂端等層級比動畫高，要等動畫快退完才淡入；只寫 from、不留終態，隱藏中的回到頂端才不會被強制顯示 */
body > *:not(#root){animation:boot-fade-in .4s ease .62s backwards}
@keyframes boot-fade-in{from{opacity:0}}
@media (prefers-reduced-motion:reduce){.stApp [class*="st-key-page_"],.stApp .site-nav,.stApp .brand-block,body > *:not(#root){animation:none}}
"""
BOOT_MIN_SECONDS = 1.2   # 頁面很快跑完時也至少播這麼久，否則動畫閃一下就消失


def _boot_css(extra: str) -> str:
    return "<style>" + _BOOT_BASE_CSS + extra + "</style>"


def _boot_loader_html(done: bool = False) -> str:
    """動畫用 site_logo 的真實圖形，最後一格就是網站 logo。整段不能有換行，否則 markdown 會把 HTML 拆開。"""
    def layer(cls: str, inner: str, style: str = "") -> str:
        # 動畫一定要掛在外層 div：掛在 <svg> 上 Chrome 不會交給 GPU（trace 的 compositeFailed=1024），主執行緒一忙就停
        return f'<div class="boot-layer {cls}" style="{style}"><svg viewBox="0 0 54 54" aria-hidden="true">{inner}</svg></div>'

    colors = ("#c0392f", "#1f8a58", "#c0392f")   # 台股慣例：紅漲綠跌；最後淡入 logo 色的同一根 K 線取代變色動畫
    candles = ""
    for i, color in enumerate(colors):
        x, _top, _bottom, _wick_top, wick_bottom = site_logo.CANDLES[i]
        timing = f"animation-delay:{i * 0.1:.1f}s;"
        origin = f"transform-origin:{x / 54 * 100:.1f}% {wick_bottom / 54 * 100:.1f}%;"
        candles += layer("boot-candle", site_logo.candle_svg(i, f'fill="{color}" stroke="{color}"'), timing + origin)
        candles += layer("boot-candle-end", site_logo.candle_svg(i), timing)
    sweep = ('<circle cx="27" cy="27" r="17" pathLength="100" fill="none" stroke="#fff" stroke-opacity=".5" '
             'stroke-width="4.4" stroke-dasharray="14 86" stroke-linecap="round"/>')
    trend = ('<path d="M17 36 L24 30 L30 25 L37 17" fill="none" stroke="#8e2b2f" stroke-width="1.1" '
             'stroke-linecap="round" stroke-linejoin="round"/>')
    return (f'<div class="boot-loader{" boot-done" if done else ""}" role="status" aria-label="網站載入中"><div class="boot-inner">'
            f'<div class="boot-mark">{layer("boot-ring", site_logo.ring_svg())}{layer("boot-sweep", sweep)}{candles}'
            f'{layer("boot-trend", trend)}{layer("boot-tail", site_logo.tail_svg())}</div>'
            '<div class="boot-title">台股即時互動式分析系統</div><div class="boot-track"></div>'
            '<div class="boot-note">正在準備盤面資料</div></div></div>')


# 兩個位置每一輪都要留著（之後留空）：元素數量一變，後面的元件全部位移重建，
# 攔截換頁連結的 iframe 也會被換掉，站內換頁就變成整頁重新載入。
_boot_css_slot, _boot_slot = st.empty(), st.empty()
# 帳號審核連結會中途 st.stop()，走不到淡出那一行，不播放
_boot_active = not st.session_state.get("_site_booted") and "account_review" not in st.query_params
_boot_started = time.time()
if _boot_active:
    _boot_css_slot.markdown(_boot_css(_BOOT_PLAYING_CSS), unsafe_allow_html=True)   # CSS 要單獨一個 markdown，含 <style> 的區塊會被整個隱藏
    _boot_slot.markdown(_boot_loader_html(), unsafe_allow_html=True)


def _finish_boot_loader() -> None:
    if _boot_active:
        time.sleep(max(0.0, BOOT_MIN_SECONDS - (time.time() - _boot_started)))
        _boot_css_slot.markdown(_boot_css(_BOOT_DONE_CSS), unsafe_allow_html=True)
        _boot_slot.markdown(_boot_loader_html(done=True), unsafe_allow_html=True)
        st.session_state["_site_booted"] = True


# ---- 所有 Plotly 圖表統一使用與網頁背景相配的淺色系 ----
# Streamlit 每次重跑都會重新執行本檔，但 st 模組在伺服器程序中只有一份：
# 一定要取 Streamlit 內部的原始函式（st._main），不能取 st.plotly_chart，否則每次重跑多包一層，
# 久了就 maximum recursion depth exceeded（舊程序中已疊好的包裝也會因此被丟掉）。
_plotly_chart_orig = getattr(getattr(st, "_main", None), "plotly_chart", None) or st.plotly_chart
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

@st.cache_resource(show_spinner=False)
def get_client():
    # 網站用：Fugle 每分鐘額度用完時立即放棄該次請求（顯示上一份快取），不讓整頁等待重試。
    return FugleClient(wait_on_limit=False)

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

@st.cache_data(ttl=21600, show_spinner=False)
def _local_stock_names() -> dict:
    """本機產業主檔的股票名稱；有名稱就不必再為每檔股票多打一次 Fugle ticker API。"""
    try:
        df = pd.read_csv(DATA / "industry_master.csv", dtype=str, usecols=["symbol", "name"])
        return {clean_symbol(k): v for k, v in zip(df["symbol"], df["name"]) if isinstance(v, str) and v.strip()}
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False, max_entries=512)
def _ticker_name(code: str) -> str:
    """Fugle ticker 的股票名稱（一天快取）；查不到就丟例外，不把失敗結果快取起來。"""
    name = (client.ticker(code) or {}).get("name")
    if not name:
        raise ValueError(f"{code} 沒有名稱")
    return name


def stock_name(code: str) -> str:
    """股票名稱：本機產業主檔 → Fugle ticker；都查不到才回傳代號。
    雲端網站沒有產業主檔，行情 API 撞到額度改用日 K 時也不帶名稱（2026-10-27 上銀只顯示 2049）。"""
    code = clean_symbol(code)
    name = _local_stock_names().get(code)
    if name:
        return name
    try:
        return _ticker_name(code)
    except Exception:
        return code


@st.cache_data(ttl=1800, show_spinner=False)
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
        name = stock_name(code)
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
if "ai_reused" not in st.session_state:
    st.session_state.ai_reused = False
if "ai_trained_at" not in st.session_state:
    st.session_state.ai_trained_at = None
if "ai_training_data_date" not in st.session_state:
    st.session_state.ai_training_data_date = None
if "research_result" not in st.session_state:
    st.session_state.research_result = None

# ============================================================
# Data layer - 即時資料 2 秒；慢資料分開快取
# ============================================================
def _quote_bucket(live_seconds: int) -> int:
    """行情快取的時間格：盤中 live_seconds 秒換一格，盤外 5 分鐘換一格。
    Fugle 每分鐘只有 60 次額度，自選股 20 多檔若每 2 秒重抓一次，一定會撞到上限而卡住頁面。"""
    return int(time.time() // (live_seconds if is_market_open_now() else 300))


def watch_rows(watchlist_key):
    return _watch_rows(watchlist_key, _quote_bucket(15))


@st.cache_data(ttl=600, show_spinner=False, max_entries=64)
def _watch_rows(watchlist_key, _bucket):
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

    def fetch_one(code):
        # 盤後是本次問題的關鍵：不要使用可能停留在前一交易日的 snapshot。
        # 直接走「今日 Quote -> 今日分K -> 今日日K」的嚴格日期驗證流程。
        if after_close:
            return get_quote(code) or {}
        if not trading_day or now.time() < MARKET_OPEN:
            # 盤前／週末／休市日只顯示最近完成交易日，不拿舊 snapshot 冒充即時行情。
            return previous_day_daily_quote(code)
        x = maps.get(code, {})
        if not x:
            try:
                x = client.quote(code)
            except Exception as e:
                errors.append(f"{code}: {e}")
        return x

    # 逐檔查詢改成並行：20 多檔的等待時間從「加總」變成「最慢一檔」。Fugle 額度由 stock_api 共用計數控管。
    watchlist = list(watchlist_key)
    with ThreadPoolExecutor(max_workers=8) as pool:
        quotes = list(pool.map(fetch_one, watchlist))
    for code, x in zip(watchlist, quotes):
        x = x or {}

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

        name = x.get("name") or x.get("stockName") or x.get("securityName")
        if not name or name == code:
            name = stock_name(code)
        limit_up = pd.to_numeric(x.get("limitUpPrice", x.get("limitUp", np.nan)), errors="coerce")
        limit_down = pd.to_numeric(x.get("limitDownPrice", x.get("limitDown", np.nan)), errors="coerce")
        # Fugle 的 quote／snapshot 沒有漲跌停價：依參考價推算，並以 Fugle 的漲跌停旗標為準。
        # 漲跌停因檔位取整，漲跌幅常只有 9.8x%，只看「≥9.95%」會漏掉（2026-10-27 台燿跌停、上銀漲停都沒亮）。
        if (pd.isna(limit_up) or pd.isna(limit_down)) and pd.notna(prev):
            calc_up, calc_down = intraday_signals.limit_prices(float(prev))
            limit_up = calc_up if pd.isna(limit_up) else limit_up
            limit_down = calc_down if pd.isna(limit_down) else limit_down
        if pd.notna(p) and x.get("isLimitUpPrice") is True:
            limit_up = p
        if pd.notna(p) and x.get("isLimitDownPrice") is True:
            limit_down = p
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


def get_quote(code: str) -> dict:
    return _get_quote(code, _quote_bucket(LIVE_REFRESH_SECONDS))


@st.cache_data(ttl=600, show_spinner=False, max_entries=512)
def _get_quote(code: str, _bucket: int) -> dict:
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

class _NoData(Exception):
    """取不到資料時丟出，讓 st.cache_data 不快取失敗結果（例如 Fugle 429）。"""


def get_intraday(code: str, timeframe: str) -> pd.DataFrame:
    try:
        return _get_intraday_cached(clean_symbol(code), str(timeframe), _quote_bucket(LIVE_REFRESH_SECONDS))
    except _NoData:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False, max_entries=256)
def _get_intraday_cached(code: str, timeframe: str, _bucket: int) -> pd.DataFrame:
    """盤中依 LIVE_REFRESH_SECONDS 更新；盤外分 K 不會變，快取 5 分鐘。失敗不快取。"""
    k = _fetch_intraday(code, timeframe)
    if k is None or k.empty:
        raise _NoData(code)
    return k


def _fetch_intraday(code: str, timeframe: str) -> pd.DataFrame:
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

def get_trades(code: str) -> pd.DataFrame:
    try:
        return _get_trades_cached(clean_symbol(code), _quote_bucket(LIVE_REFRESH_SECONDS))
    except _NoData:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False, max_entries=128)
def _get_trades_cached(code: str, _bucket: int) -> pd.DataFrame:
    try:
        payload = client._get(f"intraday/trades/{code}")
    except Exception as exc:
        raise _NoData(code) from exc
    return pd.DataFrame(payload.get("data", []) or [])

@st.cache_data(ttl=5, show_spinner=False)
def get_taiex() -> pd.DataFrame:
    return fetch_taiex_intraday()

@st.cache_data(ttl=60, show_spinner=False)
def get_ticker(code: str) -> dict:
    return client.ticker(code)

@st.cache_data(ttl=1800, show_spinner=False)
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

.brand-block{text-align:center;padding:0 0 0}.brand-mark{width:54px;height:54px;margin:0 auto 8px}.brand-mark svg{display:block;width:54px;height:54px}.brand-name{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:20px;letter-spacing:1.5px;color:#555}.site-status-line{text-align:center;font-size:10px;color:#989898;margin:4px 0 10px;letter-spacing:.4px}.site-nav{display:flex;justify-content:center;align-items:center;gap:28px;flex-wrap:wrap;padding:8px 4px 12px}.site-nav-link{font-family:"Noto Sans TC","Microsoft JhengHei",sans-serif;color:#777!important;text-decoration:none!important;font-size:11px;letter-spacing:.6px;padding:3px 0;border-bottom:1px solid transparent;transition:.18s ease}.site-nav-link:hover{color:var(--site-red)!important}.site-nav-link.active{color:var(--site-red)!important;border-bottom-color:var(--site-red);font-weight:800}.site-rule{border-top:1px dashed #a9a1a1;margin:1px 0 20px}
.page-image-banner{position:relative;width:100%;height:180px;border-radius:10px;overflow:hidden;margin:0 0 24px;box-shadow:0 8px 24px rgba(0,0,0,.08)}.page-image-banner>img{display:block;width:100%;height:100%;object-fit:cover}.page-image-shade{position:absolute;inset:0;background:rgba(15,15,20,.42)}.page-image-title{position:absolute;left:34px;bottom:24px;color:#fff;font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:28px;font-weight:500;letter-spacing:1px}.contact-page-banner{margin-bottom:20px}.hero-visual{width:100%;height:350px;border-radius:10px;overflow:hidden;margin:0 0 0;box-shadow:0 8px 25px rgba(0,0,0,.08)}
.home-band{padding:48px 42px;margin:0}.home-band.blush{background:var(--site-blush)}.home-band.white-band{background:#fff}.home-two-col{width:100%;margin:0;display:grid;grid-template-columns:1.1fr .9fr;gap:60px;align-items:center}.home-two-col.reverse-on-mobile{grid-template-columns:.92fr 1.08fr}.home-copy{max-width:540px}.eyebrow{color:var(--site-red);font-size:10px;letter-spacing:2px;font-weight:800;margin-bottom:7px}.home-copy h2,.content-heading h1{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;color:var(--site-red);font-size:32px;line-height:1.2;margin:0 0 14px;font-weight:500}.home-copy p,.content-heading p{font-size:13px;line-height:1.9;color:#5f5a5a;margin:0 0 10px}.home-link,.contact-link{display:inline-block;margin-top:8px;color:var(--site-red)!important;text-decoration:none!important;font-size:12px;font-weight:800;letter-spacing:.5px;border-bottom:1px solid rgba(142,43,47,.45);padding-bottom:3px}.home-image-stack{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:end}.small-visual{height:170px;border-radius:8px;overflow:hidden;box-shadow:0 5px 14px rgba(0,0,0,.12)}.home-image-grid{display:grid;gap:10px}.home-image-grid.three-grid{grid-template-columns:repeat(3,1fr)}.home-image-grid.six-grid{grid-template-columns:repeat(3,1fr)}.home-image-grid>div,.focus-mini{min-width:0;border-radius:8px;overflow:hidden;box-shadow:0 5px 12px rgba(0,0,0,.09)}.home-image-grid.three-grid>div{height:142px}.home-image-grid.six-grid>div{height:118px}.home-accordion-hints{margin:14px 0 8px;border-top:1px solid #ded4d4}.home-accordion-hints>div{display:flex;justify-content:space-between;gap:16px;padding:12px 2px;border-bottom:1px solid #ded4d4;font-size:12px}.home-accordion-hints span{color:#8b8282}
.contact-band{position:relative;min-height:280px;overflow:hidden;margin:0}.contact-overlay{position:absolute;inset:0}.contact-overlay svg,.contact-overlay img{width:100%;height:100%;display:block;object-fit:cover;}.contact-shade{position:absolute;inset:0;background:rgba(20,20,20,.76)}.contact-content{position:relative;z-index:2;max-width:780px;margin:0 auto;padding:44px 42px;color:#fff}.contact-content .eyebrow{color:#cf9494}.contact-content h2{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:30px;font-weight:500;margin:0 0 14px;color:#fff}.contact-content p{font-size:13px;line-height:1.9;color:#ddd;margin:0;max-width:660px}.contact-link{color:#e8b2b2!important;border-color:rgba(232,178,178,.4)}
.content-heading{padding:8px 0 22px;max-width:760px}.content-heading h1{font-size:34px;margin-bottom:8px}.feature-section-title{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:22px;color:var(--site-red);font-weight:500;border-bottom:1px solid var(--site-line);padding:8px 0 10px;margin:36px 0 18px}.selected-strip{display:flex;gap:7px;align-items:center;background:var(--site-blush);border-left:3px solid var(--site-red);padding:10px 13px;font-size:12px;color:#766f6f;margin:12px 0 20px}.selected-strip strong{color:var(--site-red);font-size:15px}.data-panel-heading,.finance-card-title,.form-title{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;color:var(--site-red);font-size:20px;margin:20px 0 8px}.search-code-site{color:var(--site-red);font-weight:900;font-size:16px}.search-name-site{font-size:13px;color:#3e3a3a;padding:9px 0}.search-name-site span{color:#8d8585;margin-left:4px}.finance-card-sub{color:#8c8585;font-size:11px;margin-bottom:12px}.finance-refresh-time{font-size:11px;color:#7f6d6d;margin:0 0 12px;text-align:right}
[data-testid="stDataFrame"],[data-testid="stTable"],[data-testid="stExpander"]{border-color:#ddd4d4!important}[data-testid="stExpander"] summary{font-size:13px!important;color:#5b5555!important}.stButton>button{border:1px solid #c9bbbb!important;background:#fff!important;color:#6d5b5b!important;border-radius:3px!important;box-shadow:none!important}.stButton>button:hover{border-color:var(--site-red)!important;color:var(--site-red)!important}[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{border-radius:2px!important;border:0!important;border-bottom:1px solid #cfc7c7!important;background:#fff!important;box-shadow:none!important}[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-bottom-color:var(--site-red)!important;box-shadow:none!important}
.contact-info-card{background:#fff;color:#4a4545;padding:6px 0;min-height:360px}.contact-info-card h2{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:28px;color:var(--site-red);font-weight:500;margin:0 0 10px}.contact-info-card p{color:#777;font-size:12px;line-height:1.8;max-width:470px}.contact-item{display:flex;gap:12px;align-items:flex-start;padding:12px 0;border-top:1px solid #e2dada}.contact-item span{color:var(--site-red);font-size:18px;line-height:1}.contact-item b{display:block;font-size:12px;margin-bottom:3px;color:#4a4545}.contact-item small{display:block;color:#8c8585;font-size:11px;line-height:1.6}.form-title{margin-top:0;font-size:21px}.subscribe-note{margin:8px 0 12px;color:var(--site-red);font-size:12px;line-height:1.6}.stFormSubmitButton>button{background:#252525!important;color:#fff!important;border-color:#252525!important;border-radius:16px!important;padding:6px 22px!important}.site-footer{background:var(--site-footer);color:#eee;margin:34px 0 0;padding:0 0 0;width:100%}.footer-inner{width:100%;margin:0;display:grid;grid-template-columns:1fr 1fr;gap:50px;padding:38px 42px 30px}.footer-contact,.footer-form-note{padding:4px 0}.footer-title{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;color:#d68f8f;font-size:16px;margin-bottom:8px}.footer-contact p,.footer-form-note p{color:#bdbdbd;font-size:11px;line-height:1.8;margin:0 0 10px}.footer-small{color:#8f8f8f;font-size:10px;line-height:1.8}.footer-form-note a{color:#d7a2a2!important;text-decoration:none!important;font-size:11px}.footer-bottom{border-top:1px solid rgba(255,255,255,.08);text-align:center;padding:14px 12px 18px;color:#777;font-size:9px;letter-spacing:.5px}
@media(max-width:900px){.main .block-container{padding-left:16px!important;padding-right:16px!important}.site-nav{gap:18px}.home-band,.site-footer,.contact-band{margin-left:0;margin-right:0}.home-band{padding:42px 22px}.home-two-col,.home-two-col.reverse-on-mobile{grid-template-columns:1fr;gap:28px}.home-copy{max-width:none}.footer-inner{padding:30px 22px;grid-template-columns:1fr;gap:24px}}
@media(max-width:600px){.main .block-container{padding-left:10px!important;padding-right:10px!important}.brand-name{font-size:17px}.site-nav{gap:14px 18px;padding-bottom:10px}.site-nav-link{font-size:11px}.hero-visual{height:220px;margin-bottom:0}.page-image-banner{height:150px;margin-bottom:18px}.page-image-title{left:18px;bottom:18px;font-size:24px}.home-band,.site-footer,.contact-band{margin-left:0;margin-right:0}.home-band{padding:28px 16px}.home-copy h2,.content-heading h1{font-size:26px}.home-image-grid.three-grid{grid-template-columns:1fr 1fr 1fr}.home-image-grid.six-grid{grid-template-columns:1fr 1fr}.home-image-grid.three-grid>div{height:108px}.home-image-grid.six-grid>div{height:96px}.small-visual{height:128px}.feature-section-title{font-size:18px}.footer-inner{padding:26px 18px}}
/* 聯絡資訊：字放大、每一項做成卡片，信箱與電話可直接點 */
.contact-info-card h2{font-size:32px!important;margin-bottom:12px!important}
.contact-info-card p{font-size:15.5px!important;color:#5f5757!important;line-height:1.9!important;max-width:520px!important;margin-bottom:14px!important}
.contact-item{gap:16px!important;align-items:center!important;padding:14px 16px!important;margin:0 0 10px;border:1px solid #eedcdc!important;border-radius:12px;background:#fdf8f8}
.contact-item span{flex:none;width:42px;height:42px;border-radius:50%;background:var(--site-red);color:#fff!important;font-size:20px!important;
  display:inline-flex;align-items:center;justify-content:center;box-shadow:0 4px 10px -5px rgba(142,43,47,.7)}
.contact-item b{font-size:13.5px!important;color:#8c7f7f!important;font-weight:600;letter-spacing:.5px;margin-bottom:2px!important}
.contact-item small{font-size:17px!important;color:#2f2a2a!important;font-weight:700;line-height:1.5!important;overflow-wrap:anywhere}
.contact-item small a{color:var(--site-red)!important;text-decoration:none!important;border-bottom:1px solid rgba(142,43,47,.3)}
.contact-item small a:hover{border-bottom-color:var(--site-red)}
.contact-content h2{font-size:34px!important}
.contact-content p{font-size:16.5px!important;line-height:1.9!important;color:#eee!important}
.contact-link{font-size:16px!important;font-weight:700}
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

def _site_image_html(slot: int, wide: bool = False, dark: bool = False) -> str:
    """網站圖片走 Streamlit 靜態檔（static/img，config.toml 啟用 enableStaticServing）。
    瀏覽器會快取圖片，不再把 base64 圖片塞進每次重跑的頁面；檔案不存在時改用 SVG 示意圖。"""
    name = SITE_IMAGE_FILES[int(slot) % len(SITE_IMAGE_FILES)]
    if (SITE_IMAGE_DIR / name).exists():
        return (f'<img src="app/static/img/{name}" alt="台股市場視覺素材" loading="lazy" '
                'style="display:block;width:100%;height:100%;object-fit:cover;" />')
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
    return f'''
    <div class="brand-block" aria-label="台股即時互動式分析系統">
      <div class="brand-mark">{site_logo.svg_markup()}</div>
      <div class="brand-name">台股即時互動式分析系統</div>
    </div>
    '''


_UX_CSS = """<style>
/* 重跑時的頂端進度條：超過 0.35 秒才出現，快速更新不會閃 */
.stApp[data-test-script-state="running"]::before{content:"";position:fixed;top:0;left:0;right:0;height:3px;z-index:1000000;pointer-events:none;
  background:linear-gradient(90deg,transparent 0,#8e2b2f 30%,#c9837a 50%,#58758e 70%,transparent 100%);background-size:200% 100%;
  animation:uxBarIn .2s .35s both,uxBar 1.1s linear infinite}
@keyframes uxBarIn{from{opacity:0}to{opacity:1}}
@keyframes uxBar{from{background-position:200% 0}to{background-position:-200% 0}}
/* 重跑時舊內容不要整片變灰閃爍（進度條已經告知正在更新） */
[data-stale="true"]{opacity:1!important;transition:none!important}

/* 導覽列固定在頂端，捲到哪都能換頁 */
/* Streamlit 1.5x 起容器外面還包一層 stLayoutWrapper，sticky 要加在那一層 */
[data-testid="stLayoutWrapper"]:has(> .st-key-site_nav_sticky),.st-key-site_nav_sticky{position:sticky;top:0;z-index:990}
/* 實心白底延伸到整個主畫面寬（不只內容欄），捲動時底下的大標題不會透出來；捲動後只加一條細線與淡陰影 */
/* 用 box-shadow 向左右延伸（陰影不算可捲動範圍，不會出現橫向捲動），再用 clip-path 裁掉上下 */
.st-key-site_nav_sticky{position:relative;padding:4px 0 14px}
.st-key-site_nav_sticky::before,.st-key-site_nav_sticky::after{content:"";position:absolute;left:0;right:0;z-index:-1;clip-path:inset(0 -100vw)}
.st-key-site_nav_sticky::before{top:0;bottom:0;background:#fff;box-shadow:0 0 0 100vw #fff}
.st-key-site_nav_sticky::after{bottom:0;height:1px;background:#e9dcdc;box-shadow:0 0 0 100vw #e9dcdc;opacity:0;transition:opacity .25s ease}
.ux-scrolled .st-key-site_nav_sticky::after{opacity:1}
/* 捲動後整條（含延伸的白底）投一層淡陰影，內容像是滑進導覽列下方 */
[data-testid="stLayoutWrapper"]:has(> .st-key-site_nav_sticky){transition:filter .25s ease,transform .32s cubic-bezier(.2,.7,.2,1)}
/* 往下捲後收起，滑鼠移到頁首附近再彈出（JS 切換 body.ux-navhide） */
.ux-navhide [data-testid="stLayoutWrapper"]:has(> .st-key-site_nav_sticky){transform:translateY(calc(-100% - 14px));filter:none}
#ux-navtab{position:fixed;top:0;left:50%;z-index:991;transform:translate(-50%,-110%);padding:3px 16px 5px;border:1px solid #e9dcdc;border-top:0;
  border-radius:0 0 12px 12px;background:#fff;color:#8e2b2f;font-size:12px;line-height:1.5;letter-spacing:.5px;white-space:nowrap;cursor:pointer;
  box-shadow:0 4px 10px -4px rgba(60,20,20,.18);transition:transform .32s cubic-bezier(.2,.7,.2,1)}
.ux-navhide #ux-navtab{transform:translate(-50%,0)}
#ux-navtab:hover{color:#fff;background:#8e2b2f;border-color:#8e2b2f}

/* 導覽下拉選單：滑到主題上展開該頁功能，點了直接前往該區塊 */
.site-nav .nav-item{position:relative;display:flex}
.nav-drop{position:absolute;top:100%;left:50%;z-index:5;min-width:300px;padding:0;display:block;text-align:left;overflow:hidden;
  background:#fff;border:1px solid #ecdcdc;border-radius:16px;
  box-shadow:0 24px 48px -18px rgba(70,20,24,.35),0 2px 6px rgba(70,20,24,.06);
  opacity:0;visibility:hidden;transform:translate(-50%,14px);transition:opacity .2s ease,transform .2s ease,visibility .2s}
.nav-drop::before{content:"";position:absolute;left:0;right:0;top:-14px;height:14px}
.nav-item:hover .nav-drop,.nav-item:focus-within .nav-drop{opacity:1;visibility:visible;transform:translate(-50%,8px)}
.nav-item.nav-closed .nav-drop{opacity:0!important;visibility:hidden!important}
/* 標頭：主題名稱＋說明，酒紅漸層底 */
.nav-drop-head{position:relative;padding:16px 20px 14px;background:linear-gradient(135deg,#8e2b2f 0%,#b0464b 100%);color:#fff}
.nav-drop-head:after{content:"";position:absolute;right:-28px;top:-34px;width:110px;height:110px;border-radius:50%;
  background:rgba(255,255,255,.08)}
.nav-drop-title{display:flex;align-items:center;gap:9px;font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;
  font-size:21px;font-weight:700;letter-spacing:1.5px;line-height:1.2;white-space:nowrap}
.nav-drop-step{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;
  background:#fff;color:#8e2b2f;font-family:Arial,sans-serif;font-size:13px;font-weight:800;letter-spacing:0;flex:none}
.nav-drop-hint{margin-top:6px;font-size:12.5px;color:rgba(255,255,255,.86);letter-spacing:.4px;white-space:nowrap}
/* 功能清單：編號＋名稱，項目多時排兩欄 */
.nav-drop-list{display:grid;grid-template-columns:1fr;gap:2px;padding:10px}
.nav-drop-list.two-col{grid-template-columns:1fr 1fr;min-width:430px}
.nav-drop-link{position:relative;display:flex;align-items:center;gap:11px;padding:10px 14px;border-radius:10px;
  font-size:15.5px;font-weight:600;line-height:1.35;color:#3b3434!important;text-decoration:none!important;white-space:nowrap;
  transition:background .15s,color .15s,transform .15s}
.nav-drop-num{flex:none;min-width:26px;height:22px;border-radius:6px;background:#f6eaea;color:#a0484c;font-family:Arial,sans-serif;
  font-size:11.5px;font-weight:800;display:inline-flex;align-items:center;justify-content:center;transition:background .15s,color .15s}
.nav-drop-text{flex:1}
.nav-drop-arrow{color:#8e2b2f;font-size:14px;opacity:0;transform:translateX(-6px);transition:opacity .15s,transform .15s}
.nav-drop-link:hover{background:#fbf1f1;color:#8e2b2f!important;transform:translateX(2px)}
.nav-drop-link:hover .nav-drop-num{background:#8e2b2f;color:#fff}
.nav-drop-link:hover .nav-drop-arrow{opacity:1;transform:none}
/* 上方主題列：膠囊式分段按鈕，目前頁面為酒紅實心，滑過浮起 */
.st-key-site_nav_sticky .site-nav{width:fit-content;max-width:100%;margin:6px auto 0;gap:4px!important;padding:5px!important;
  background:linear-gradient(180deg,#fdf8f8,#f7eeee);border:1px solid #eedfdf;border-radius:999px;
  box-shadow:inset 0 1px 0 #fff,0 6px 18px -12px rgba(110,30,36,.35)}
.site-nav .site-nav-link{display:inline-flex;align-items:center;gap:7px;padding:9px 19px!important;border-radius:999px;
  font-family:"Noto Sans TC","Microsoft JhengHei",sans-serif!important;font-size:16px!important;font-weight:700;letter-spacing:1.2px!important;
  color:#5a4f4f!important;transition:background .18s ease,color .18s ease,box-shadow .18s ease,transform .18s ease}
.site-nav .site-nav-link::after{display:none}
.site-nav .nav-ico{width:17px;height:17px;flex:none;stroke:currentColor;fill:none;stroke-width:1.9;stroke-linecap:round;stroke-linejoin:round;opacity:.75}
.site-nav .site-nav-link:hover{background:#fff;color:var(--site-red)!important;box-shadow:0 4px 12px -6px rgba(110,30,36,.45);transform:translateY(-1px)}
.site-nav .site-nav-link.active{background:linear-gradient(135deg,#8e2b2f,#b0464b);color:#fff!important;font-weight:800;
  box-shadow:0 6px 14px -6px rgba(142,43,47,.75)}
.site-nav .site-nav-link.active .nav-ico{opacity:1}
.site-nav .site-nav-link.active:hover{transform:none}
@media(max-width:900px){.site-nav .site-nav-link{font-size:14.5px!important;padding:8px 13px!important}.site-nav .nav-ico{width:15px;height:15px}}
@media(max-width:600px){.st-key-site_nav_sticky .site-nav{border-radius:16px;margin-top:2px}
  .site-nav .site-nav-link{font-size:13.5px!important;padding:7px 11px!important;gap:5px}.site-nav .nav-ico{display:none}}
/* 最左、最右的主題靠邊對齊，避免中等寬度時下拉選單超出內容欄被裁掉 */
.site-nav .nav-item:first-child .nav-drop{left:0;transform:translate(0,14px)}
.site-nav .nav-item:last-child .nav-drop{left:auto;right:0;transform:translate(0,14px)}
.site-nav .nav-item:first-child:hover .nav-drop,.site-nav .nav-item:first-child:focus-within .nav-drop,
.site-nav .nav-item:last-child:hover .nav-drop,.site-nav .nav-item:last-child:focus-within .nav-drop{transform:translate(0,8px)}
@media (hover:none),(max-width:600px){.nav-drop{display:none}}
/* 跳轉找不到區塊時的提示 */
#ux-toast{position:fixed;left:50%;bottom:28px;z-index:10000;transform:translate(-50%,16px);opacity:0;pointer-events:none;
  background:#2b2324;color:#fff;font-size:13.5px;padding:9px 18px;border-radius:999px;box-shadow:0 10px 24px -8px rgba(0,0,0,.35);
  transition:opacity .25s ease,transform .25s ease}
#ux-toast.show{opacity:1;transform:translate(-50%,0)}
.ux-scrolled [data-testid="stLayoutWrapper"]:has(> .st-key-site_nav_sticky){filter:drop-shadow(0 5px 8px rgba(60,20,20,.09))}

/* 換頁淡入 */
[class*="st-key-page_"]{animation:uxPageIn .5s cubic-bezier(.2,.7,.2,1) backwards}
@keyframes uxPageIn{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
/* 換頁等待：點下主題到新頁面出現之間，舊頁面先藏起來（JS 切換 body.ux-nav-pending） */
.ux-nav-pending [class*="st-key-page_"],.ux-nav-pending .site-footer{visibility:hidden}
.st-key-site_pages{gap:0!important}
#ux-nav-loading{display:none;position:fixed;left:50%;top:44%;z-index:900;transform:translate(-50%,-50%);flex-direction:column;align-items:center;gap:14px;pointer-events:none}
.ux-nav-pending #ux-nav-loading{display:flex;animation:uxNavShow .25s ease}
@keyframes uxNavShow{from{opacity:0}}
.ux-nav-bars{display:flex;align-items:flex-end;gap:6px;height:34px}
.ux-nav-bars i{width:9px;height:100%;border-radius:2px;background:#8e2b2f;transform-origin:bottom;animation:uxNavBar 1s ease-in-out infinite alternate}
.ux-nav-bars i:nth-child(1){animation-delay:-.6s}
.ux-nav-bars i:nth-child(2){animation-delay:-.3s}
@keyframes uxNavBar{from{transform:scaleY(.35);opacity:.45}to{transform:none;opacity:1}}
.ux-nav-text{font-size:13.5px;letter-spacing:.08em;color:#8a7676}

/* 首頁：五步驟快速入口 */
.home-flow{padding:44px 0 30px;text-align:center}
.home-flow h2{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;color:var(--site-red,#8e2b2f);font-size:30px;margin:6px 0 8px}
.home-flow>p{color:#6f6767;font-size:14px;line-height:1.8;margin:0 auto 24px;max-width:640px}
.flow-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}
.flow-card{position:relative;display:flex;flex-direction:column;gap:7px;text-align:left;padding:18px 16px 42px;border:1px solid #eee3e3;border-radius:14px;
  background:#fff;color:#3d3838!important;text-decoration:none!important;box-shadow:0 2px 8px rgba(0,0,0,.03);
  transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease;animation:uxRise .6s cubic-bezier(.2,.7,.2,1) backwards;animation-delay:calc(var(--i) * 90ms + .15s)}
.flow-card:hover{transform:translateY(-6px);box-shadow:0 16px 32px -10px rgba(142,43,47,.28);border-color:#dcbcbc}
.flow-step{width:32px;height:32px;border-radius:50%;display:grid;place-items:center;background:#f8eeee;color:var(--site-red,#8e2b2f);font-weight:800;font-size:16px;transition:background .25s,color .25s}
.flow-card:hover .flow-step{background:var(--site-red,#8e2b2f);color:#fff}
.flow-card b{font-size:17px;color:var(--site-red,#8e2b2f);letter-spacing:.5px}
.flow-card small{font-size:12.5px;color:#7a7171;line-height:1.65}
.flow-go{position:absolute;right:16px;bottom:13px;color:var(--site-red,#8e2b2f);font-weight:800;transition:transform .25s ease}
.flow-card:hover .flow-go{transform:translateX(6px)}
@keyframes uxRise{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}
@media(max-width:900px){.flow-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){
  .flow-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.flow-card{padding:13px 12px 32px;gap:5px}
  .flow-card b{font-size:15px}.flow-card small{font-size:11.5px}.flow-step{width:26px;height:26px;font-size:13px}.flow-go{bottom:9px;right:12px}
  .home-flow{padding:28px 0 18px}.home-flow h2{font-size:24px}
  /* 手機：導覽改成單列、可左右滑，固定在頂端時不佔兩行高度 */
  .st-key-site_nav_sticky .site-nav{flex-wrap:nowrap!important;justify-content:flex-start!important;overflow-x:auto;gap:20px!important;
    padding:8px 14px 10px!important;scrollbar-width:none;-webkit-mask-image:linear-gradient(90deg,#000 88%,transparent);mask-image:linear-gradient(90deg,#000 88%,transparent)}
  .st-key-site_nav_sticky .site-nav::-webkit-scrollbar{display:none}
  .st-key-site_nav_sticky .site-nav-link{white-space:nowrap;flex:0 0 auto}
  .jump-bar{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none;padding-bottom:4px}.jump-chip,.jump-label{flex:0 0 auto}
  #ux-top{right:14px;bottom:18px;width:40px;height:40px}
}

/* 操作策略：即時狀態與區塊快速跳轉 */
.jump-bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:-4px 0 6px}
.jump-label{font-size:12px;color:#8a8080;margin-left:4px}
.jump-chip{cursor:pointer;user-select:none;font-size:12.5px;padding:4px 12px;border-radius:999px;border:1px solid #e6d6d6;background:#fff;color:#5b4f4f!important;
  text-decoration:none!important;transition:background .2s,color .2s,border-color .2s,transform .2s}
.jump-chip:hover{background:var(--site-red,#8e2b2f);color:#fff!important;border-color:var(--site-red,#8e2b2f);transform:translateY(-2px)}
.live-badge{font-size:12px;padding:4px 12px;border-radius:999px;background:#f1f3f5;color:#6b7280;white-space:nowrap}
.live-badge.on{background:#fdecec;color:#b42318;font-weight:700}
.live-badge.on::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#e5484d;margin-right:7px;vertical-align:1px;animation:uxPulse 1.6s infinite}
@keyframes uxPulse{0%{box-shadow:0 0 0 0 rgba(229,72,77,.55)}70%{box-shadow:0 0 0 8px rgba(229,72,77,0)}100%{box-shadow:0 0 0 0 rgba(229,72,77,0)}}
[class*="st-key-strategy-0"]{scroll-margin-top:78px}
@keyframes uxFlash{0%{box-shadow:0 0 0 0 rgba(142,43,47,.35)}100%{box-shadow:0 0 0 14px rgba(142,43,47,0)}}
.ux-flash{animation:uxFlash .9s ease-out}

/* 載入骨架 */
.skeleton{display:grid;gap:10px;padding:4px 0 14px}
.skeleton span{display:block;height:14px;border-radius:7px;background:linear-gradient(90deg,#f1ecec 25%,#fbf8f8 50%,#f1ecec 75%);background-size:200% 100%;animation:uxShimmer 1.2s linear infinite}
.skeleton span:first-child{height:170px;border-radius:12px}.skeleton .short{width:45%}
@keyframes uxShimmer{from{background-position:200% 0}to{background-position:-200% 0}}

/* 圖表出現時輕微淡入；指標卡與按鈕的互動回饋 */
[data-testid="stPlotlyChart"]{animation:uxFade .45s ease backwards}
@keyframes uxFade{from{opacity:0}to{opacity:1}}
[data-testid="stMetric"]{transition:transform .2s ease}
[data-testid="stMetric"]:hover{transform:translateY(-2px)}
.stButton button,.stFormSubmitButton button{transition:transform .12s ease,box-shadow .2s ease}
.stButton button:active,.stFormSubmitButton button:active{transform:scale(.97)}

/* 籌碼訊號：看多紅、看空綠 */
.sig-row{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 12px}
.sig{display:inline-block;font-size:13px;font-weight:700;line-height:1.5;padding:2px 11px;border-radius:999px;border:1px solid;white-space:nowrap}
.sig-bull{color:#d23c3c;background:#fdeeee;border-color:#f2c3c3}
.sig-bear{color:#14874f;background:#e8f6ee;border-color:#bde3cd}
.sig.sig-top.sig-bull{color:#fff!important;background:#b33636;border-color:#b33636}
.sig.sig-top.sig-bear{color:#fff!important;background:#1d7a4a;border-color:#1d7a4a}
.sig-flat{color:#6b6464;background:#f3f1f1;border-color:#e2dcdc}
.sig-none,.sig-wait{font-size:12.5px;color:#8a8080}
.sig-wait{animation:uxFade 1s ease-in-out infinite alternate}
.wc-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px;position:relative;z-index:1;pointer-events:none}
/* Streamlit 的 markdown 區塊底部有 -1rem 邊距；自選股卡片多了一列訊號後會被吃進卡片底邊，這裡歸零讓卡片依內容長高 */
body .stApp.stApp [class*="st-key-wc_"] [data-testid="stMarkdownContainer"]{margin-bottom:0!important}
.wc-chips:empty{display:none}
.wc-chips .sig{font-size:10.5px;padding:0 6px}
.wc-chips .sig-wait{font-size:10.5px}
.sig-table-wrap{overflow-x:auto}
.sig-table{width:100%;border-collapse:collapse;font-size:14px}
.sig-table th{text-align:left;font-weight:700;color:#6b6464;background:#faf6f6;padding:8px 10px;border-bottom:1px solid #eadede;white-space:nowrap}
.sig-table td{padding:8px 10px;border-bottom:1px solid #f1eaea;font-variant-numeric:tabular-nums}
.sig-t-bull{color:#d23c3c;font-weight:700}.sig-t-bear{color:#14874f;font-weight:700}.sig-t-flat,.sig-muted{color:#8a8080}

/* 個股符合特徵：依類別上色（籌碼橘、價量黃、基本藍、技術紫、消息紅），圓點＝偏多紅／偏空綠 */
.feat-grid{display:grid;grid-template-columns:72px 1fr;gap:14px 16px;align-items:start;margin:4px 0 10px}
.feat-cat{font-size:14px;font-weight:700;color:#7a7171;padding-top:5px}
.feat-tags{display:flex;flex-wrap:wrap;gap:8px}.feat.feat-top{font-weight:800;color:#fff!important;border-color:transparent!important}.feat.feat-top.feat-t-bull{background:#b33636!important}.feat.feat-top.feat-t-bear{background:#1d7a4a!important}
.feat{display:inline-flex;align-items:center;gap:5px;font-size:14px;font-weight:700;line-height:1.5;padding:3px 13px;border-radius:999px;border:1.5px solid;white-space:nowrap;cursor:help}
.feat-news{color:#b4372f;background:#fdf0ef;border-color:#efc4c0}
.feat-chip{color:#b5651d;background:#fdf4ea;border-color:#ecc9a3}
.feat-pv{color:#94750a;background:#fcf8e6;border-color:#e6d68f}
.feat-fund{color:#1f6aa5;background:#eef5fb;border-color:#b5d2ea}
.feat-tech{color:#6a4fb3;background:#f3f0fb;border-color:#cfc3ee}
.feat-t-bull::before,.feat-t-bear::before,.feat-dot{content:"";display:inline-block;width:7px;height:7px;border-radius:50%}
.feat-t-bull::before,.feat-dot.bull{background:#d23c3c}
.feat-t-bear::before,.feat-dot.bear{background:#14874f}
.feat-none{font-size:14px;color:#8a8080;padding-top:5px}
.feat-why td:nth-child(-n+2){white-space:nowrap}
.feat-legend{font-size:12.5px;color:#8a8080;margin:2px 0 8px;display:flex;flex-wrap:wrap;align-items:center;gap:4px 14px}
@media (max-width:640px){.feat-grid{grid-template-columns:1fr;gap:6px}.feat-cat{padding-top:8px}.feat{font-size:13px;padding:2px 11px}}

/* 國際風向球卡片 */
.gm-card{border:1px solid #eee3e3;border-radius:12px;padding:14px 16px 8px;background:#fff}
.gm-head{display:flex;align-items:center;gap:8px;justify-content:space-between}
.gm-head b{font-size:15px;color:#3d3838}
.gm-val{font-size:28px;font-weight:800;color:#2b2626;font-variant-numeric:tabular-nums;margin-top:4px}
.gm-val small{font-size:12px;font-weight:500;color:#8a8080;margin-left:6px}
.gm-chg{font-size:14px;font-weight:700;font-variant-numeric:tabular-nums}
.gm-note{font-size:12.5px;color:#7a7171;margin-top:6px;line-height:1.6}
.gm-asof{font-size:11.5px;color:#a29999;margin-top:2px}

/* 回到頂端 */
#ux-top{position:fixed;right:22px;bottom:26px;width:44px;height:44px;border-radius:50%;border:0;background:#8e2b2f;color:#fff;font-size:20px;
  box-shadow:0 8px 20px rgba(142,43,47,.35);cursor:pointer;opacity:0;transform:translateY(16px) scale(.9);pointer-events:none;transition:opacity .3s,transform .3s;z-index:9999}
#ux-top.show{opacity:1;transform:none;pointer-events:auto}
#ux-top.show:hover{transform:translateY(-3px)}

@media (prefers-reduced-motion: reduce){
  [class*="st-key-page_"],.flow-card,[data-testid="stPlotlyChart"],.skeleton span,.live-badge.on::before,.ux-flash{animation:none!important}
  .flow-card:hover,.jump-chip:hover,[data-testid="stMetric"]:hover{transform:none}
}
</style>"""


def _inject_ux_layer() -> None:
    """全站互動層：載入進度條、固定導覽、換頁淡入、卡片與骨架動畫、回到頂端。"""
    st.markdown(_UX_CSS, unsafe_allow_html=True)


def render_header_and_search() -> None:
    """依設計文件統一呈現所有頁面的頁首品牌區。"""
    now = taiwan_now()
    st.markdown(_site_header_svg(), unsafe_allow_html=True)

def render_strategy_search() -> None:
    """操作策略頁的股票查詢；選取後同步整個操作策略與未來分析。"""
    st.markdown('<div class="content-heading"><div class="eyebrow">STOCK OPERATION</div><h1>操作策略</h1><p>從自選股即時行情到 K 線、量能、籌碼與法人資料，集中處理單一分析標的。</p></div>', unsafe_allow_html=True)
    left, right = st.container(key="stock_search_box").columns([8.2, 1.35], gap="small", vertical_alignment="bottom")
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
        if is_authenticated():
            if st.button("加入自選股", key=f"add_watch_site_{code}"):
                st.session_state.watchlist.append(code)
                st.session_state.watchlist = list(dict.fromkeys(st.session_state.watchlist))
                if not save_watchlist(st.session_state.watchlist):
                    st.session_state.watchlist_save_failed = True
                st.rerun()
        else:
            st.caption("登入後才能新增自選股。")

    if st.session_state.pop("watchlist_save_failed", False):
        st.warning("自選股已更新，但沒有存進雲端資料庫；網站重新部署後可能會還原。")
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


def _recent_earnings_items(days: int = 5):
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
    name = str(item.get("name") or item.get("company") or item.get("company_name") or "").strip()
    if not name or name == code:
        # Fugle 法說會備忘錄只有代號，名稱在標題裡：「【裕山法說會重點內容備忘錄】…」
        m = re.search(r"【([^】]+?)法說會", str(item.get("title") or ""))
        name = m.group(1).strip() if m else (stock_name(code) if code else "")
    name = name or code
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


def _digest_author(digest: dict) -> str:
    by = digest.get("generated_by")
    if by == "claude":
        at = str(digest.get("agent_generated_at", ""))[11:16]
        return f"Claude Agent {at} " if at else "Claude Agent "
    return "Qwen3 Agent（舊資料）" if by == "qwen3" else "規則摘要"


def _render_news_digest(digest: dict, total_news: int) -> None:
    if not digest:
        return
    stats = digest.get("stats", {}) or {}
    headline = digest.get("agent_headline") or digest.get("headline") or ""
    by = _digest_author(digest)
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


def _render_earnings_digest(digest: dict, days: int = 5) -> None:
    """法說會 AI 重點（版型與新聞的 AI 今日重點一致；新聞的函式不動）。"""
    if not digest or not digest.get("total"):
        return
    headline = digest.get("agent_headline") or digest.get("headline") or ""
    by = _digest_author(digest)
    parts = [f'<div class="fin-digest"><h3>📌 法說會 AI 重點 <small style="font-weight:400;font-size:12px">（最近 {days} 天・{by}整理）</small></h3>']
    if headline:
        parts.append(f'<div class="fin-headline">{_fin_hl(headline, [])}</div>')
    points = digest.get("agent_key_points") or []
    if points:
        parts.append('<ul class="fin-list">' + "".join(f"<li>{_fin_hl(p, [])}</li>" for p in points) + "</ul>")
    else:
        rows = digest.get("key_points", []) or []
        parts.append(
            '<ul class="fin-list">'
            + "".join(
                f'<li>{_fin_badge(x.get("sentiment"))}<b>{_fin_hl(x.get("title"), [])}</b>：{_fin_hl(x.get("point"), x.get("highlights"))}</li>'
                for x in rows[:6]
            )
            + "</ul>"
        )
    heat = digest.get("sector_heat", []) or []
    if heat:
        chips = "".join(
            f'<span class="fin-chip {"hot" if i < 3 else ""}">{x["sector"]} {x["count"]}'
            f'{"　▲" if x.get("tone") == "偏多" else "　▼" if x.get("tone") == "偏空" else ""}</span>'
            for i, x in enumerate(heat[:8])
        )
        parts.append(f'<div class="fin-sub" style="margin-top:12px">產業熱度（場數）</div>{chips}')
    watch, risks = digest.get("watch_items", []) or [], digest.get("risks", []) or []
    if watch or risks:
        def col(title, rows):
            body = "".join(f"<li>{_fin_hl(x, [])}</li>" for x in rows) or "<li>—</li>"
            return f'<div><div class="fin-sub">{title}</div><ul class="fin-list">{body}</ul></div>'
        parts.append('<div class="fin-two">' + col("👀 待追蹤", watch) + col("⚠️ 風險", risks) + "</div>")
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
    """法說會卡片：以新聞摘要相同的螢光重點呈現，並使用法說會 Agent 的結構化分析。"""
    from html import escape as _esc

    highlights = item.get("highlights") or []
    summary = str(
        item.get("one_line_summary")
        or item.get("ai_summary")
        or item.get("summary")
        or _record_summary(item)
        or ""
    ).strip()
    points = item.get("ai_points") or item.get("key_points") or []
    points = [str(point).strip() for point in points if str(point).strip()] if isinstance(points, (list, tuple)) else []
    signal = (
        item.get("sentiment")
        or item.get("impact")
        or item.get("signal")
        or item.get("judgement")
    )
    badge = _fin_badge(str(signal)) if signal else ""
    body = []

    # 與財經新聞一致：先放一行摘要，再以條列列出可掃讀的 AI 重點。
    if summary:
        body.append(
            f'<div class="fin-why"><b>法說會摘要：</b>{_fin_hl(summary, highlights)}</div>'
        )
    if points:
        body.append(
            '<div class="fin-sub">📌 AI 分析重點</div>'
            '<ul class="fin-list">'
            + "".join(f'<li>{_fin_hl(point, highlights)}</li>' for point in points[:5])
            + "</ul>"
        )
    elif not summary:
        body.append(f'<div>{_fin_hl(str(_record_summary(item)), highlights)}</div>')

    if item.get("why"):
        body.append(f'<div class="fin-why">💡 {_fin_hl(item["why"], highlights)}</div>')

    # 這些陣列由 Fugle 法說會 Agent 直接產生；用巢狀展開區保留詳細資訊，
    # 預設不拉長整個摘要清單。新聞摘要的函式與呈現保持不動。
    detail_sections = [
        ("財務表現", item.get("financial_highlights")),
        ("營運重點", item.get("operating_highlights")),
        ("展望與指引", item.get("guidance")),
        ("利多因素", item.get("positive_factors")),
        ("利空因素", item.get("negative_factors")),
        ("關鍵風險", item.get("key_risks")),
        ("Q&A 重點", item.get("qa_highlights")),
    ]
    detail_parts = []
    for heading, values in detail_sections:
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)):
            continue
        clean_values = [str(value).strip() for value in values if str(value).strip()]
        if not clean_values:
            continue
        rows = "".join(
            f'<li>{_fin_hl(value[:420], highlights)}</li>'
            for value in clean_values[:3]
        )
        detail_parts.append(
            f'<div><div class="fin-sub">{_esc(heading)}</div><ul class="fin-list">{rows}</ul></div>'
        )

    reasoning = str(item.get("reasoning") or "").strip()
    confidence = item.get("confidence")
    try:
        confidence_text = f"分析信心度：{float(confidence):.0f}%" if confidence not in (None, "") else ""
    except (TypeError, ValueError):
        confidence_text = ""
    analysis_source = str(
        item.get("analysis_source")
        or ("AI 法說會 Agent" if item.get("agent_tool_used") else "正文規則備援")
    ).strip()

    if detail_parts or reasoning or confidence_text:
        inner = []
        if detail_parts:
            inner.append('<div class="fin-two">' + "".join(detail_parts) + "</div>")
        if reasoning:
            inner.append(f'<div class="fin-why"><b>判斷依據：</b>{_fin_hl(reasoning[:900], highlights)}</div>')
        meta_bits = [x for x in (analysis_source, confidence_text) if x]
        if meta_bits:
            inner.append(f'<div class="fin-meta">{"　｜　".join(_esc(x) for x in meta_bits)}</div>')
        body.append(
            '<details class="fin-item"><summary>查看法說會 Agent 詳細分析</summary>'
            f'<div class="fin-body">{"".join(inner)}</div></details>'
        )

    meta = []
    published = str(item.get("published_date") or item.get("event_date") or item.get("date") or "")[:16]
    if published:
        meta.append(_esc(published))
    if item.get("industry_name"):
        meta.append(_esc(str(item.get("industry_name"))))
    if meta:
        body.append(f'<div class="fin-meta">{"　｜　".join(meta)}</div>')

    chips = _fin_chips(item.get("sectors") or []) + _fin_chips(item.get("stocks") or [], "stock")
    if chips:
        body.append(f"<div>{chips}</div>")

    url = _safe_url(item.get("source_url") or item.get("url"))
    if url:
        body.append(
            f'<div style="margin-top:6px"><a href="{url}" target="_blank" rel="noopener noreferrer">查看 Fugle 原文 ↗</a></div>'
        )

    return (
        f'<details class="fin-item"{" open" if is_open else ""}>'
        f'<summary>{badge}{_esc(label)}</summary><div class="fin-body">{"".join(body)}</div></details>'
    )


def render_finance_workspace() -> None:
    st.markdown(
        '<div class="content-heading"><div class="eyebrow">DAILY FINANCE</div><h1>財經資訊</h1></div>',
        unsafe_allow_html=True,
    )
    with _sec_box("fin-global"):
        render_global_markets()
    _render_finance_body()


# 財經內容在瀏覽器開啟期間每分鐘自動更新；只重跑這個 fragment，不會整頁重畫或打斷名人持股區的操作。
# 180 秒資料快取到期後會重新向 GitHub 取最新 JSON。
@st.fragment(run_every=60)
def _render_finance_body() -> None:
    cache = _load_finance_info_cache()
    earnings = _recent_earnings_items(days=5)
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
        f"更新排程：每天 {_FIN_SCHEDULE_TEXT}｜法說會摘要範圍：最近 5 天｜財經新聞範圍：最近 2 天。晨報 08:40、盤後分析 15:45（Claude 研究 Agent）。</div>",
        unsafe_allow_html=True,
    )

    with _sec_box("fin-digest", tone=1):
        _render_news_digest(cache.get("news_digest") or {}, len(news))

    left, right = st.columns([1, 1.45], gap="large")

    # ---------------- 法說會 ----------------
    with left, _sec_box("fin-earnings", tone=2):
        st.markdown('<div class="finance-card-title">法說會摘要</div>', unsafe_allow_html=True)
        # 摘要標題以實際顯示的五日清單重算，避免舊快取仍寫著「最近 2 天」或顯示過期筆數。
        earnings_counts = {"利多": 0, "利空": 0, "中性": 0, "混合": 0}
        for item in earnings:
            sentiment = str(
                item.get("sentiment") or item.get("impact") or item.get("judgement") or "混合"
            ).strip()
            if sentiment not in earnings_counts:
                sentiment = "混合"
            earnings_counts[sentiment] += 1
        earnings_headline = (
            f"最近 5 天共 {len(earnings)} 場法說會備忘錄："
            f"利多 {earnings_counts['利多']}、利空 {earnings_counts['利空']}、"
            f"中性 {earnings_counts['中性']}、混合 {earnings_counts['混合']}。"
            if earnings else "最近 5 天沒有可用的法說會備忘錄。"
        )
        st.markdown(
            f'<div class="fin-meta">{_fin_hl(earnings_headline, [])}</div>',
            unsafe_allow_html=True,
        )
        if not earnings:
            st.info("最近 5 天目前沒有發現新的可用 Fugle 法說會摘要；這不代表財經資訊快取未更新。")
        else:
            _render_earnings_digest(cache.get("earnings_digest") or {}, int(cache.get("earnings_lookback_days") or 5))
            html_items = "".join(
                _earning_item_html(item, f"{item['_date'].strftime('%Y-%m-%d')}｜{_record_company(item)}", i == 0)
                for i, item in enumerate(earnings[:20])
            )
            st.markdown(f'<div translate="no" class="notranslate">{html_items}</div>', unsafe_allow_html=True)

    # ---------------- 新聞 ----------------
    with right, _sec_box("fin-news", tone=3):
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
                st.rerun(scope="fragment")


@st.cache_data(ttl=43200, show_spinner=False)
def _cached_guru_holdings() -> dict:
    """13F 每季才更新一次；快取 12 小時，SEC 連不上時讀 data/cache 的上一次結果。"""
    return fetch_all_gurus(cache_path=DATA / "cache" / "celebrity_13f.json")


def _fmt_usd(v: float) -> str:
    return f"${v / 1e9:,.1f}B" if v >= 1e9 else f"${v / 1e6:,.0f}M"


def _render_politician_card(p: dict) -> None:
    color = {"買進": "#c62828", "賣出": "#2e7d32"}
    rows = "".join(
        f'<tr><td><span style="color:{color.get(a, "#555")};font-weight:700">{_esc(a)}</span></td>'
        f'<td>{_esc(t)}</td><td>{_esc(amt)}</td><td>{_esc(d)}</td></tr>'
        for a, t, amt, d in p["highlights"]
    )
    links = "｜".join(
        f'<a href="{_safe_url(u)}" target="_blank" rel="noopener noreferrer">{_esc(label)} ↗</a>'
        for label, u in p["sources"]
    )
    st.markdown(
        f'''<div translate="no" class="notranslate celeb-card">
        <div class="celeb-name">{_esc(p["name"])}<span>{_esc(p["role"])}</span></div>
        <div class="celeb-asof">資料期間：{_esc(p["as_of"])}</div>
        <p>{_esc(p["summary"])}</p>
        <table class="celeb-table"><thead><tr><th>動作</th><th>標的</th><th>申報金額區間</th><th>日期</th></tr></thead>
        <tbody>{rows}</tbody></table>
        <div class="celeb-src">原始資料：{links}</div></div>''',
        unsafe_allow_html=True,
    )


def _render_guru_portfolios() -> None:
    with st.spinner("讀取 SEC 13F 申報…"):
        data = _cached_guru_holdings()
    gurus = [g for g in data.get("gurus", []) if not g.get("error")]
    failed = [g for g in data.get("gurus", []) if g.get("error")]
    if not gurus:
        st.warning("目前無法取得 SEC 13F 資料，請稍後再試。")
        return
    if data.get("stale"):
        st.caption("⚠️ SEC 暫時連不上，顯示上一次取得的資料。")

    tw = [(g, r) for g in gurus for r in g.get("taiwan_rows", [])]
    if tw:
        chips = "".join(
            f'<span class="celeb-chip">{_esc(g["person"].split(" ")[0])}{"（" + r["put_call"] + " 選擇權）" if r["put_call"] else ""}：'
            f'{r["weight"]:.1f}%（{_esc(r["change"])}）</span>'
            for g, r in sorted(tw, key=lambda x: x[1]["weight"], reverse=True)
        )
        st.markdown(f'<div class="celeb-tw"><b>台積電 ADR（TSM）持有者</b>{chips}</div>', unsafe_allow_html=True)

    names = [g["person"] for g in gurus]
    pick = st.radio("投資人", names, horizontal=True, key="celeb_guru", label_visibility="collapsed")
    g = next(x for x in gurus if x["person"] == pick)

    st.caption(f'{g["firm"]}（SEC CIK {g["cik"]}）')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("前十大集中度", f'{sum(r["weight"] for r in g["top"]):.0f}%')
    c2.metric("申報持股市值", _fmt_usd(g["total_value"]))
    c3.metric("持股檔數", f'{g["positions"]}')
    c4.metric("資料季度", g["period"], help=f'申報日 {g["filed"]}；與前一季 {g["prev_period"]} 比較股數變化')

    top = g["top"]
    labels = [r["issuer"].title()[:30] + (f' ({r["put_call"]})' if r["put_call"] else "") for r in top]
    bar_colors = ["#c62828" if r["taiwan"] else "#8e2b2f" if r["change"].startswith(("加碼", "新進")) else "#58758e" for r in top]
    fig = go.Figure(go.Bar(
        x=[r["weight"] for r in top][::-1], y=labels[::-1], orientation="h",
        marker_color=bar_colors[::-1],
        text=[f'{r["weight"]:.1f}%｜{r["change"]}' for r in top][::-1], textposition="outside",
        hovertemplate="%{y}<br>占比 %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=36 * len(top) + 70, margin=dict(l=10, r=40, t=30, b=10),
        title=dict(text="前十大持股占比（紅：加碼／新進，藍：減碼／持平）", font=dict(size=13)),
        xaxis=dict(ticksuffix="%", range=[0, max(r["weight"] for r in top) * 1.35]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    table = pd.DataFrame([{
        "公司": r["issuer"] + (f' ({r["put_call"]})' if r["put_call"] else ""),
        "占比 %": round(r["weight"], 2),
        "市值": _fmt_usd(r["value"]),
        "股數": f'{r["shares"]:,.0f}',
        "較前一季": r["change"],
        "台股相關": r["taiwan"],
    } for r in top])
    st.dataframe(table, hide_index=True, use_container_width=True)

    buys = "、".join(r["issuer"].title() for r in g["new_buys"]) or "無"
    sells = "、".join(x["issuer"].title() for x in g["sold_out"]) or "無"
    st.markdown(
        f'<div class="info-strip">本季新進：{_esc(buys)}<br>本季出清：{_esc(sells)}<br>'
        f'<a href="{_safe_url(g["url"])}" target="_blank" rel="noopener noreferrer">SEC EDGAR 原始申報 ↗</a></div>',
        unsafe_allow_html=True,
    )
    if failed:
        st.caption("暫時無法取得：" + "、".join(f["person"] for f in failed))


def render_celebrity_holdings() -> None:
    """財經資訊頁：名人投資人持股（政治人物財產申報＋投資大師 13F）。"""
    st.markdown('''<style>
    .celeb-card{background:#fff;border:1px solid #e7dede;border-left:3px solid var(--site-red,#8e2b2f);border-radius:10px;padding:14px 18px;margin:10px 0 16px}
    .celeb-name{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:19px;color:var(--site-red,#8e2b2f);font-weight:700}
    .celeb-name span{font-size:12px;color:#7f7676;font-weight:400;margin-left:10px}
    .celeb-asof,.celeb-src{font-size:12px;color:#7f7676;margin:4px 0 8px}.celeb-card p{font-size:14px;line-height:1.75;margin:6px 0 10px}
    .celeb-table{width:100%;border-collapse:collapse;font-size:13px}.celeb-table th{background:#f6efef;text-align:left;padding:6px 8px;color:#5b5353}
    .celeb-table td{border-bottom:1px solid #eee;padding:6px 8px;vertical-align:top}
    .celeb-src a{color:#0072c6!important;font-weight:700}
    .celeb-tw{background:#fbf4f4;border:1px solid #ecd6d6;border-radius:10px;padding:10px 13px;margin:6px 0 12px;font-size:13px;line-height:2}
    .celeb-tw b{color:#8e2b2f;margin-right:10px}.celeb-chip{display:inline-block;background:#fff;border:1px solid #e3cfcf;border-radius:999px;padding:0 10px;margin:0 6px 0 0;white-space:nowrap}
    @media(max-width:600px){.celeb-table{font-size:12px}.celeb-table th,.celeb-table td{padding:5px 4px}}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="feature-section-title">名人投資人持股</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='info-strip'>政治人物依美國 OGE／國會 STOCK Act 財產申報整理（只公布金額區間）；"
        "投資大師依 SEC 13F 季報即時讀取（季底後 45 天內申報，約落後一季，僅含美股多方部位與選擇權）。"
        "資料僅供參考，不構成投資建議。</div>",
        unsafe_allow_html=True,
    )
    tab_pol, tab_guru = st.tabs(["🏛️ 政治人物（川普、裴洛西）", "📈 投資大師 13F"])
    with tab_pol:
        for p in POLITICIANS:
            _render_politician_card(p)
    with tab_guru:
        _render_guru_portfolios()


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


@st.cache_data(ttl=21600, show_spinner=False)
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


def _render_home_flow() -> None:
    """首頁快速入口：照研究流程排成五步，點卡片直接進對應頁面。"""
    cards = "".join(
        f'<a class="flow-card" href="?section={key}" target="_self" style="--i:{i}">'
        f'<span class="flow-step">{i + 1}</span><b>{label}</b><small>{hint[2:]}</small><span class="flow-go">→</span></a>'
        for i, (key, label, hint) in enumerate(SITE_NAV[1:6])
    )
    st.markdown(
        '<section translate="no" class="home-flow"><div class="eyebrow">HOW TO USE</div>'
        '<h2>五步驟看懂台股</h2><p>照著順序走：從市場消息找到強勢族群，再深入個股與型態，最後參考 AI 預測。</p>'
        f'<div class="flow-grid">{cards}</div></section>',
        unsafe_allow_html=True,
    )


def render_home_workspace() -> None:
    """首頁：依 PDF 的節點節奏，以實際提供的六張台股圖片做視覺主體。"""
    _render_hero_carousel()
    _render_home_flow()

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
      <p>集中查看最近 5 天的法說會摘要與最近 2 天的財經新聞摘要，先理解市場事件，再回到操作策略與未來分析。</p>
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

@st.cache_data(ttl=300, show_spinner=False)
def _cached_subscription_status() -> dict:
    """訂閱人數要向 Google Apps Script 下載整份名單（約 2 秒）；只用來顯示人數，快取 5 分鐘。"""
    return subscription_status(BASE)


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
              <p>歡迎針對網站功能、股票資料、AI 分析或每日財經資訊提出問題。</p>
              <div class="contact-item"><span>◎</span><div><b>台灣股市研究</b><small>AI 台股即時互動式分析系統</small></div></div>
              <div class="contact-item"><span>⌕</span><div><b>研究服務</b><small>股票操作策略／未來分析／財經資訊</small></div></div>
              <div class="contact-item"><span>✉</span><div><b>電子郵件</b><small><a href="mailto:a1113359@mail.nuk.edu.tw">a1113359@mail.nuk.edu.tw</a></small></div></div>
              <div class="contact-item"><span>☏</span><div><b>聯絡電話</b><small><a href="tel:+886907611728">+886-907-611-728</a></small></div></div>
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
                        _cached_subscription_status.clear()
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
        sub_status = _cached_subscription_status()
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
    allowed = {"home", "strategy", "trend", "future", "finance", "rotation", "subscribe"}
    return value if value in allowed else "home"


def _set_nav_section(section: str) -> None:
    # 按鈕的 on_click 在重跑之前執行：同一次重跑就直接畫新頁面，不必再 st.rerun() 多跑一整輪。
    st.session_state["nav_section"] = section
    try:
        st.query_params["section"] = section
    except Exception:
        pass


# 導覽順序＝建議的使用流程：先看市場消息，再找強勢族群，接著研究個股、型態，最後看預測。
SITE_NAV = [
    ("home", "首頁", "網站導覽"),
    ("finance", "財經資訊", "① 看市場：今日重點、法說會、新聞、名人持股"),
    ("rotation", "產業分析", "② 找族群：產業輪動與資金流向"),
    ("strategy", "操作策略", "③ 選個股：行情、K 線、籌碼、法人"),
    ("trend", "趨勢型態", "④ 看型態：K 線型態掃描"),
    ("future", "未來分析", "⑤ 看未來：AI 預測與研究 Agent（需登入）"),
    ("subscribe", "訂閱系統", "訂閱晨報與聯絡我們"),
]


# 導覽下拉選單：每個主題底下的功能。(顯示名稱, 頁面上區塊標題包含的文字, 要切換的分頁標籤)
# 點選後先切到該頁，等頁面畫完再捲到標題文字相符的區塊（JS 依 data-target 比對標題）。
SITE_NAV_SUB = {
    "home": [("五步驟看懂台股", "五步驟看懂台股"), ("關於台股", "關於台股"), ("每日財經資訊", "每日財經資訊"),
             ("專注台股", "專注台股"), ("聯絡資訊", "聯絡資訊")],
    "finance": [("國際風向球：美股、原油、VIX", "國際風向球"), ("AI 今日重點", "AI 今日重點"), ("法說會摘要", "法說會摘要"), ("財經新聞摘要", "財經新聞摘要"),
                ("名人持股：政治人物", "名人投資人持股", "政治人物"), ("名人持股：投資大師 13F", "名人投資人持股", "投資大師")],
    "rotation": [("產業族群走勢", "產業族群"), ("資金輪動熱力圖", "資金輪動熱力圖"), ("產業輪動圖", "產業輪動圖"), ("產業報酬與輪動狀態", "產業報酬與輪動狀態"),
                 ("美股產業指標股", "美股產業指標股"), ("本週象限變化", "本週象限變化"), ("成分股漲跌前十", "成分股漲跌前十"),
                 ("輪動解讀與資料限制", "輪動解讀與資料限制")],
    "strategy": [("自選股盤中焦點", "自選股盤中焦點"), ("個股符合特徵", "個股符合特徵"), ("股價趨勢", "股價趨勢"), ("技術 K 線", "技術 K 線"), ("成交量", "成交量"), ("進場建議分析", "進場建議分析系統"),
                 ("大戶 VS 散戶", "大戶 VS 散戶"), ("交易明細", "交易明細"), ("三大法人", "三大法人")],
    "trend": [("期間統計", "期間統計")] + [(c, c) for c in TREND_CATEGORY_ORDER],
    "future": [("最近系統推薦股票", "最近系統推薦股票"), ("AI 隔日預測", "AI 隔日預測"), ("市場情報與利多／利空", "市場情報"), ("模型研究 Agent", "模型研究 Agent"),
               ("AI 個股研究筆記", "AI 個股研究筆記")],
    "subscribe": [("聯絡資訊", "聯絡資訊"), ("訂閱／聯絡表單", "訂閱／聯絡表單")],
}


# 主題列小圖示（線條 SVG，顏色跟著文字）
_NAV_ICONS = {
    "home": '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
    "finance": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9h10M7 13h10M7 17h6"/>',
    "rotation": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
    "strategy": '<path d="M4 20V4"/><path d="M4 20h16"/><path d="M8 16v-5M12 16V8M16 16v-3"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "future": '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 17l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/>',
    "subscribe": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/>',
}


def render_site_navigation() -> str:
    """Restore the original HTML navigation appearance while retaining hidden controls for session-safe navigation."""
    current = _get_nav_section()
    links = []
    for key, label, hint in SITE_NAV:
        active = " active" if current == key else ""
        items = SITE_NAV_SUB.get(key, [])
        subs = "".join(
            f'<a class="nav-drop-link" href="?section={key}" target="_self" data-target="{_esc(sub[1])}"'
            + (f' data-tab="{_esc(sub[2])}"' if len(sub) > 2 else "")
            + f'><span class="nav-drop-num">{i:02d}</span><span class="nav-drop-text">{_esc(sub[0])}</span>'
              '<span class="nav-drop-arrow">→</span></a>'
            for i, sub in enumerate(items, 1)
        )
        # 說明開頭的 ①～⑤ 拆成標題旁的步驟圓章
        step = "①②③④⑤".find(hint[:1]) + 1
        badge = f'<span class="nav-drop-step">{step}</span>' if step else ""
        text = hint[1:].lstrip() if step else hint
        drop = (f'<div class="nav-drop"><div class="nav-drop-head"><div class="nav-drop-title">{badge}{_esc(label)}</div>'
                f'<div class="nav-drop-hint">{_esc(text)}</div></div>'
                f'<div class="nav-drop-list{" two-col" if len(items) > 6 else ""}">{subs}</div></div>') if subs else ""
        icon = f'<svg class="nav-ico" viewBox="0 0 24 24">{_NAV_ICONS.get(key, "")}</svg>'
        links.append(f'<div class="nav-item"><a class="site-nav-link{active}" href="?section={key}" target="_self">{icon}{label}</a>{drop}</div>')
    # 包在有 key 的容器裡才能 position:sticky（捲到下方也能直接換頁）
    with st.container(key="site_nav_sticky"):
        st.markdown('<nav translate="no" class="site-nav">' + ''.join(links) + '</nav>', unsafe_allow_html=True)
    nav = [(key, label) for key, label, _ in SITE_NAV]
    st.markdown(
        '<style>.st-key-site_nav_control{position:fixed!important;left:-10000px!important;top:0!important;width:1px!important;height:1px!important;overflow:hidden!important;opacity:0!important;pointer-events:none!important;z-index:-1!important;}</style>',
        unsafe_allow_html=True,
    )
    with st.container(key="site_nav_control"):
        for section, label in nav:
            st.button(label, key=f"nav_section_{section}", on_click=_set_nav_section, args=(section,))
    st.markdown('<div class="site-rule"></div>', unsafe_allow_html=True)
    return current


def _render_footer() -> None:
    """每一頁固定放頁尾，並顯示資料來源、新聞著作權與投資風險聲明。"""
    st.markdown('''
    <footer translate="no" class="site-footer"><div class="footer-inner"><div class="footer-contact">
      <div class="footer-title">台股即時互動式分析系統</div><p>股票即時行情、操作策略、趨勢型態、未來分析與每日財經資訊。</p>
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

_LIVE_SECTIONS = {"strategy-01-focus", "strategy-03-trend", "strategy-05-volume", "strategy-08-trades"}
_SKELETON_HTML = '<div class="skeleton"><span></span><span></span><span class="short"></span></div>'


def _render_sidebar_watch() -> None:
    render_pinned_selected()
    st.markdown('<div class="feature-section-title side-watch-title">自選股即時行情</div>', unsafe_allow_html=True)
    render_watchlist(compact=True)


def render_strategy_workspace() -> None:
    """操作策略頁：維持文件的功能順序；各功能節點使用淺色交錯背景，形成同一套網站視覺。"""
    st.markdown(f'<div class="page-image-banner">{_site_image_html(7, wide=True)}</div>', unsafe_allow_html=True)
    render_strategy_search()

    sections = [
        ("strategy-01-focus", "00｜自選股盤中焦點", render_watch_focus),
        ("strategy-02-chips", "01｜個股符合特徵", render_stock_features),
        ("strategy-03-trend", "02｜股價趨勢", render_trend),
        ("strategy-04-kline", "03｜技術 K 線", render_kline),
        ("strategy-05-volume", "04｜成交量", render_volume_snapshot),
        ("strategy-06-entry", "05｜進場建議分析系統", render_entry_prediction),
        ("strategy-07-tdcc", "06｜大戶 VS 散戶持有股比率", render_tdcc),
        ("strategy-08-trades", "07｜交易明細（台灣時間）", render_trades),
        ("strategy-09-institutions", "08｜三大法人 10 交易日買賣超", render_institutions),
    ]
    # 兩段式：剛切換到這頁或換股票時，先送出全部區塊標題與「載入中」，舊頁面會立刻被替換；
    # 接著才依序抓資料、畫圖，填回各自的位置。盤中自動刷新不顯示「載入中」，避免圖表閃爍。
    selected = st.session_state.get("selected")
    fresh = st.session_state.get("_strategy_view") != selected
    st.session_state["_strategy_view"] = selected
    slots = []
    for key, title, render in sections:
        box = st.container(key=key)
        box.markdown(f'<div class="feature-section-title">{title}</div>', unsafe_allow_html=True)
        body = box.empty()
        if fresh:
            body.markdown(_SKELETON_HTML, unsafe_allow_html=True)
        slots.append((key, body, render))

    # 每個區塊各自是 fragment：切換期間、勾選法人等互動只重跑該區塊，不會整頁重畫。
    # 盤中只有即時區塊依秒數自動更新；K 線、籌碼、法人等慢資料不跟著每 5 秒重送圖表。
    live_every = REFRESH_INTERVAL if not st.session_state.get("ai_busy", False) else None

    # 自選股固定在左側欄（Streamlit 側邊欄本身固定、可獨立捲動），不會蓋到右側的分析圖表。
    # 自選股的籌碼訊號在背景計算；還有沒算完的就每 5 秒重畫側欄一次，算好就自動出現
    chips_pending = any(v is None for v in chip_signals.get_nonblocking(st.session_state.watchlist).values())
    side_every = max(live_every, 15) if live_every else (5 if chips_pending else None)
    with st.sidebar:
        st.fragment(_render_sidebar_watch, run_every=side_every)()

    for key, body, render in slots:
        with body.container():
            # 盤中焦點每分鐘重畫（資料 2 分鐘換一次）；非交易時段也每分鐘重畫，補抓撞到 Fugle 額度而缺的股票
            every = 60 if key == "strategy-01-focus" else live_every if key in _LIVE_SECTIONS else None
            st.fragment(render, run_every=every)()

def render_future_workspace() -> None:
    st.markdown(f'<div class="page-image-banner">{_site_image_html(8, wide=True)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-heading"><div class="eyebrow">FUTURE ANALYSIS</div><h1>未來分析</h1><p>把預測、市場情報與研究 Agent 集中在同一頁，讓操作策略之後的判斷有明確出口。</p></div>', unsafe_allow_html=True)

    with st.container(key="future-00-picks"):
        st.markdown('<div class="feature-section-title">01｜最近系統推薦股票</div>', unsafe_allow_html=True)
        render_system_picks(BASE)

    with st.container(key="future-01-ai"):
        st.markdown('<div class="feature-section-title">02｜AI 隔日預測</div>', unsafe_allow_html=True)
        render_ai()

    with st.container(key="future-02-intelligence"):
        st.markdown('<div class="feature-section-title">03｜市場情報與利多／利空分析</div>', unsafe_allow_html=True)
        selected = clean_symbol(st.session_state.get("selected", ""))
        industry_code = str(INDUSTRY_OVERRIDES.get(selected, WATCHLIST_INDUSTRY_FALLBACK.get(selected, "00"))).zfill(2)
        render_market_intelligence(BASE, OUTPUT, selected_symbol=selected, selected_industry=industry_code, blocked=st.session_state.get("ai_busy", False))

    with st.container(key="future-03-agent"):
        st.markdown('<div class="feature-section-title">04｜模型研究 Agent</div>', unsafe_allow_html=True)
        render_model_research_agent()

    with st.container(key="future-04-stock-notes"):
        st.markdown('<div class="feature-section-title">05｜AI 個股研究筆記</div>', unsafe_allow_html=True)
        render_stock_notes_workspace(BASE, selected_symbol=clean_symbol(st.session_state.get("selected", "")))


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


def _render_flow_heatmap(payload: dict) -> None:
    """02｜資金輪動熱力圖：金額較前一交易日增減，或占全市場成交比例較前 20 日平均的變化（可切換）。"""
    _sec('02｜資金輪動熱力圖')
    c1, c2 = st.columns([1, 1.4])
    with c1:
        kind_label = st.segmented_control("分類", list(stock_features.FLOW_KINDS.values()), default="產業主題",
                                          key="flow_heatmap_kind") or "產業主題"
    labels_by_metric = stock_features.FLOW_METRICS
    with c2:
        metric_label = st.segmented_control("計算方式", list(labels_by_metric.values()),
                                            default=labels_by_metric[_flow_metric()], key="flow_metric_label") or labels_by_metric["share20"]
    # 計算方式要帶到自選股卡片與個股特徵（其他頁面沒有這個元件），另存一份非元件的狀態
    st.session_state["flow_metric"] = next(k for k, v in labels_by_metric.items() if v == metric_label)
    metric = _flow_metric()
    kind = next(k for k, v in stock_features.FLOW_KINDS.items() if v == kind_label)
    ranks = stock_features.rotation_flow_ranks(payload, kind, metric)
    asof, prev = str(payload.get("data_asof") or ""), str(payload.get("previous_data_asof") or "")
    if not ranks:
        st.info("這份產業資料還沒有這種計算方式的資金增減；下一次產業分析更新（交易日 15:10）後會出現。")
        return
    if metric == "share20":
        span = f"{asof[5:].replace('-', '/')} 對前 {payload.get('share_base_days') or 20} 個交易日平均"
    else:
        span = f"{prev[5:].replace('-', '/')} → {asof[5:].replace('-', '/')}" if asof and prev else "較前一交易日"
    peak = max(abs(r["change"]) for r in ranks) or 1.0
    # 紅、綠各自以該側最大增減為最深色，資金增加較少的日子紅色也看得出深淺
    peak_up = max((r["change"] for r in ranks if r["change"] > 0), default=1.0)
    peak_down = max((-r["change"] for r in ranks if r["change"] < 0), default=1.0)

    def depth(v: float) -> float:
        return min(1.0, abs(v) / (peak_up if v > 0 else peak_down)) ** 0.6

    def color(v: float) -> str:
        # 台股慣例：資金增加紅、減少綠；增減越多越深（深紅→淺紅、深綠→淺綠）
        t = depth(v)
        base = (139, 26, 26) if v > 0 else (14, 90, 50)
        light = (244, 178, 178) if v > 0 else (168, 220, 188)
        return "rgb({},{},{})".format(*(round(light[i] + (base[i] - light[i]) * t) for i in range(3)))

    labels, values, colors, texts, hovers = [], [], [], [], []
    for r in ranks:
        star = "★ " if r["top"] else ""
        pct = f"{r['pct']:+.1f}%" if r.get("pct") is not None else ""
        labels.append(r["name"])
        values.append(max(abs(r["change"]), peak * 0.02))   # 增減很小的主題仍留一小格
        colors.append(color(r["change"]))
        rank = f"{'增加' if r['side'] == 'up' else '減少'}第 {r['rank']} 名" if r.get("rank") else "持平"
        if metric == "share20":
            texts.append(f"<b>{star}{r['name']}</b><br>{r['change']:+.1f} pp<br>占 {r['share']:.1f}%")
            hovers.append(f"{r['name']}<br>今天占全市場成交 {r['share']:.2f}%，前 20 日平均 {r['share_avg']:.2f}%<br>"
                          f"{r['change']:+.2f} 個百分點，{rank}")
        else:
            texts.append(f"<b>{star}{r['name']}</b><br>{r['change'] / 1e8:+,.1f} 億<br>{pct}")
            hovers.append(f"{r['name']}<br>成交金額 {(r.get('turnover') or 0) / 1e8:,.1f} 億（{r['change'] / 1e8:+,.1f} 億，{pct}）<br>{rank}")
    text_colors = ["#ffffff" if depth(r["change"]) > 0.45 else "#2b2424" for r in ranks]
    ids, parents = [f"g{i}" for i in range(len(ranks))], [""] * len(ranks)
    if kind == "value_chains":
        # 兩層：外框是 TPEx 產業（半導體、電腦及週邊設備…），裡面是子鏈；點產業可放大
        industries: dict[str, float] = {}
        for i, r in enumerate(ranks):
            ind = r.get("industry") or "其他"
            parents[i] = f"ind:{ind}"
            labels[i] = r["name"].split("／", 1)[-1]
            industries[ind] = industries.get(ind, 0.0) + values[i]
        for ind, total in industries.items():
            ids.append(f"ind:{ind}"); parents.append(""); labels.append(ind); values.append(total)
            colors.append("#f3eeee"); texts.append(f"<b>{ind}</b>"); hovers.append(ind); text_colors.append("#5a4f4f")
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, text=texts, textinfo="text",
        branchvalues="total" if kind == "value_chains" else "remainder",
        hovertext=hovers, hoverinfo="text", marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textfont=dict(size=13, color=text_colors), tiling=dict(pad=2), sort=True,
        maxdepth=2, pathbar=dict(visible=kind == "value_chains"),
    ))
    fig.update_layout(height=560 if kind == "value_chains" else 460, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True, key="rotation_flow_heatmap", config={"displaylogo": False, "responsive": True})

    names = dict(_local_stock_names())
    for g in payload.get("groups") or []:
        for c in g.get("components") or []:
            names[str(c.get("symbol"))] = c.get("name") or str(c.get("symbol"))

    def top_list(side: str, title: str) -> str:
        rows = [r for r in ranks if r["side"] == side and r["top"]]
        rows.sort(key=lambda r: r["rank"])
        items = []
        for r in rows:
            shown = r["members"] if kind == "groups" else _top_members_by_value(r["members"], payload)
            where = f'<div class="flow-path">{_esc(r["path"])}</div>' if r.get("path") else ""
            members = "、".join(f"{names.get(c, c)}（{c}）" for c in shown if c)
            if kind != "groups" and len(r["members"]) > len(shown):
                members += f" 等 {len(r['members'])} 檔"
            items.append(f'<li><div class="flow-top-head"><span class="flow-rank">{r["rank"]}</span><b>{_esc(r["name"])}</b>'
                         f'<span class="flow-amt">{stock_features.flow_value_text(r)}</span></div>'
                         f'{where}<div class="flow-members">{_esc(members)}</div></li>')
        body = "".join(items) or '<li class="flow-empty">無</li>'
        return f'<div class="flow-top flow-{side}"><div class="flow-top-title">{title}</div><ol>{body}</ol></div>'

    st.markdown(_FLOW_CSS, unsafe_allow_html=True)   # CSS 要單獨一個 markdown，含 <style> 的區塊會被整個隱藏
    st.markdown('<div class="flow-top-grid">'
                + top_list("up", f"產業市場輪動資金增加 前 {stock_features.FLOW_TOP}")
                + top_list("down", f"產業市場輪動資金減少 前 {stock_features.FLOW_TOP}")
                + "</div>", unsafe_allow_html=True)
    how = ("今天占全市場成交金額的比例減掉前 20 個交易日的平均占比（百分點，pp；ToAlpha 資金流向同法，大盤量縮放大時不受影響）"
           if metric == "share20" else "今天成交金額減前一交易日成交金額，直接比較金額增減、不除以全市場總成交金額（大盤整體量縮時多數產業會一起變少）")
    unit = {"groups": "產業主題（代表股）", "official_industries": "官方產業別（上市櫃全部股票，不含 ETF）",
            "value_chains": "TPEx 產業價值鏈子鏈（上市櫃全部股票，同一檔可屬於多個子鏈；外框是產業，點產業可放大、點上方路徑返回）"}[kind]
    st.caption(f"資料日期 {span}。每格是一個{unit}，"
               f"大小與顏色＝{how}；紅＝增加、綠＝減少，越深越多；★＝增加或減少前 {stock_features.FLOW_TOP} 名。"
               "同一股票可能屬於多個主題；代表股只是研究樣本，不等於整個產業。這些主題的個股在「操作策略 → 個股符合特徵 → 價量面」也會標出資金增減。")


def _top_members_by_value(members: list[str], payload: dict, n: int = 8) -> list[str]:
    """官方產業成分股太多，只列今天成交金額最大的 n 檔（金額來自發布資料的 member_values）。"""
    values = payload.get("official_member_values") or {}
    return sorted(members, key=lambda c: -(values.get(c) or 0))[:n]


_FLOW_CSS = """<style>
.flow-top-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:6px 0 10px}
@media (max-width:760px){.flow-top-grid{grid-template-columns:1fr}}
.flow-top{border-radius:12px;padding:12px 14px;border:1px solid}
.flow-up{background:#fdf1f1;border-color:#efcfcf}.flow-down{background:#eef7f2;border-color:#cfe6d9}
.flow-top-title{font-weight:800;font-size:16px;margin-bottom:6px}
.flow-up .flow-top-title{color:#a52a2a}.flow-down .flow-top-title{color:#17693f}
.flow-top ol{list-style:none;margin:0;padding:0}
.flow-top li{padding:7px 0;border-top:1px dashed rgba(0,0,0,.08)}.flow-top li:first-child{border-top:0}
.flow-top-head{display:flex;align-items:center;gap:8px;font-size:14.5px}
.flow-rank{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;color:#fff;font-size:12px;font-weight:800}
.flow-up .flow-rank{background:#b33636}.flow-down .flow-rank{background:#1d7a4a}
.flow-amt{margin-left:auto;font-weight:800}.flow-up .flow-amt{color:#b33636}.flow-down .flow-amt{color:#1d7a4a}
.flow-path{font-size:12px;color:#8a7d7d;margin:2px 0 0 30px}
.flow-members{font-size:12.5px;color:#6b6262;margin:3px 0 0 30px;line-height:1.6}
.flow-empty{color:#8d8585;font-size:13px}
</style>"""


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
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
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
        """依正負值上色，兼容新舊版 Pandas Styler API。"""
        valid = [col for col in columns if col in frame.columns]
        if not valid:
            return frame

        styler = frame.style
        # Pandas 新版使用 Styler.map；舊版才提供 Styler.applymap。
        cell_map = getattr(styler, "map", None)
        if callable(cell_map):
            return cell_map(sign_style, subset=valid)

        cell_applymap = getattr(styler, "applymap", None)
        if callable(cell_applymap):
            return cell_applymap(sign_style, subset=valid)

        # 若執行環境的 Styler API 更精簡，仍要讓頁面正常顯示，不因顏色功能而整頁失敗。
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
        period = st.radio("輪動軌跡", ["1 個月", "3 個月", "半年", "1 年"], index=0, key="rotation_published_period_v3")
    display_to_internal = {rotation_label(name): name for name in by_name}
    all_labels = list(display_to_internal.keys())
    selection_key = "rotation_published_groups_all_v3"
    # 新工作階段預設選取全部 27 群；獨立新 key 避免沿用舊版只選 10 群的狀態。
    if selection_key not in st.session_state:
        st.session_state[selection_key] = all_labels.copy()
    with b:
        select_actions = st.columns(2)
        with select_actions[0]:
            if st.button(f"全選 {len(all_labels)} 群", key="rotation_select_all_v3", use_container_width=True):
                st.session_state[selection_key] = all_labels.copy()
        with select_actions[1]:
            if st.button("清除選取", key="rotation_clear_selection_v3", use_container_width=True):
                st.session_state[selection_key] = []
        selected_labels = st.multiselect(
            "觀察主題（預設全部顯示）",
            all_labels,
            key=selection_key,
            help="預設選取全部 27 個子產業。可按「清除選取」後重新挑選，或按「全選」恢復全部群組。",
        )
        selected = [display_to_internal[label] for label in selected_labels if label in display_to_internal]
    with c:
        sort_by = st.selectbox(
            "排行依據",
            ["族群平均漲跌幅", "前一日平均漲跌幅", "近 20 日", "近 5 日", "相對強弱", "相對動能"],
            index=0,
            key="rotation_published_sort_v2",
        )

    quadrants = {q: sum(1 for g in groups if g.get("quadrant") == q) for q in ["領先", "改善", "轉弱", "落後"]}
    kpis = st.columns(5)
    kpis[0].metric("追蹤主題", f"{len(groups)} 群")
    kpis[1].metric("領先", f"{quadrants['領先']} 群")
    kpis[2].metric("改善", f"{quadrants['改善']} 群")
    kpis[3].metric("轉弱", f"{quadrants['轉弱']} 群")
    kpis[4].metric("落後", f"{quadrants['落後']} 群")

    # 族群平均漲跌幅：依最新完成交易日各主題代表股的日漲跌幅算術平均。
    previous_date = str(payload.get("previous_data_asof") or "")
    latest_label = f"最新交易日 {asof[5:]}" if len(asof) >= 10 else "最新交易日"
    previous_label = f"前一交易日 {previous_date[5:]}" if len(previous_date) >= 10 else "前一交易日"

    # 「排行依據」同時控制下方台股產業圖的數值、排序、標籤與基準線。
    chart_metric_fields = {
        "族群平均漲跌幅": "today_return",
        "前一日平均漲跌幅": "yesterday_return",
        "近 20 日": "return_20d",
        "近 5 日": "return_5d",
        "相對強弱": "relative_strength",
        "相對動能": "relative_momentum",
    }
    chart_metric_labels = {
        "族群平均漲跌幅": "平均漲跌幅",
        "前一日平均漲跌幅": "前一日平均漲跌幅",
        "近 20 日": "近 20 日報酬",
        "近 5 日": "近 5 日報酬",
        "相對強弱": "相對強弱指數",
        "相對動能": "相對動能指數",
    }
    chart_metric_label = chart_metric_labels[sort_by]
    _sec(f'01｜產業族群{chart_metric_label}')
    metric_is_index = sort_by in ("相對強弱", "相對動能")

    if sort_by == "族群平均漲跌幅":
        average_day = st.radio(
            "資料日期（使用最新已完成交易日，不以日曆日判斷）",
            [latest_label, previous_label],
            horizontal=True,
            key="rotation_published_avg_return_day_v1",
        )
        metric_key = "yesterday_return" if average_day == previous_label else "today_return"
        average_date = previous_date if average_day == previous_label else asof
    elif sort_by == "前一日平均漲跌幅":
        average_day = previous_label
        metric_key = "yesterday_return"
        average_date = previous_date
        st.caption(f"目前排行依據：前一日平均漲跌幅（資料日期：{average_date or '未知'}）")
    else:
        average_day = chart_metric_label
        metric_key = chart_metric_fields[sort_by]
        average_date = asof

    # 漲跌報酬以 0 為基準；相對強弱／動能以 100 為中性基準。
    axis_baseline = 100.0 if metric_is_index else 0.0

    average_rows = []
    for group in groups:
        raw_value = group.get(metric_key)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        average_rows.append({
            "name": rotation_label(group.get("name", "")),
            "metric_value": value,
            "rising": int(group.get("rising", 0) or 0) if metric_key == "today_return" else None,
            "falling": int(group.get("falling", 0) or 0) if metric_key == "today_return" else None,
        })
    # 讓台股圖表順序和「排行依據」一致；美股對照圖沿用此產業順序。
    average_rows.sort(key=lambda row: row["metric_value"], reverse=True)
    if not average_rows:
        st.info(f"目前快照沒有可用的「{sort_by}」資料，請確認產業分析快照包含此指標。")
    else:
        # 取得美股指標與 SPY 一年日線；SPY 僅作為相對強弱／動能的市場基準，不會當成產業顯示。
        try:
            us_quotes_df = _cached_us_industry_quotes()
        except Exception:
            us_quotes_df = pd.DataFrame()
        us_quote_lookup = {}
        if not us_quotes_df.empty:
            for _, quote_row in us_quotes_df.iterrows():
                us_quote_lookup[str(quote_row.get("symbol", "")).upper()] = quote_row.to_dict()

        # 以每日收盤／最新可取得價建立美股歷史序列，使排行依據選單會同時改變右圖的指標數值。
        us_close_map = {}
        for symbol, quote in us_quote_lookup.items():
            history = quote.get("history") or []
            history_rows = []
            for point in (history if isinstance(history, list) else []):
                try:
                    date_text = str(point.get("date") or "")[:10]
                    close_value = float(point.get("close"))
                    if date_text and np.isfinite(close_value) and close_value > 0:
                        history_rows.append((date_text, close_value))
                except (AttributeError, TypeError, ValueError):
                    continue
            if history_rows:
                history_frame = pd.DataFrame(history_rows, columns=["date", "close"])
                history_frame = history_frame.drop_duplicates("date", keep="last").sort_values("date")
                close_series = pd.Series(
                    history_frame["close"].to_numpy(dtype=float),
                    index=pd.to_datetime(history_frame["date"], errors="coerce"),
                    name=symbol,
                )
                close_series = close_series[~close_series.index.isna()]
                if not close_series.empty:
                    us_close_map[symbol] = close_series

        us_daily_return_columns = {
            symbol: close_series.pct_change() * 100.0
            for symbol, close_series in us_close_map.items()
            if len(close_series) >= 2
        }
        us_daily_returns_df = (
            pd.concat(us_daily_return_columns, axis=1).sort_index()
            if us_daily_return_columns else pd.DataFrame()
        )
        us_metric_for_sort = {
            "族群平均漲跌幅": "today_return",
            "前一日平均漲跌幅": "yesterday_return",
            "近 20 日": "return_20d",
            "近 5 日": "return_5d",
            "相對強弱": "relative_strength",
            "相對動能": "relative_momentum",
        }
        # 若台股單日報酬圖切換到前一交易日，美股也改用美股前一個有效交易日。
        us_metric_selected = us_metric_for_sort[sort_by]
        us_chart_metric_label = chart_metric_label
        if sort_by == "族群平均漲跌幅" and average_day == previous_label:
            us_metric_selected = "yesterday_return"
            us_chart_metric_label = "前一日平均漲跌幅"

        def _us_theme_metrics(internal_name: str) -> dict:
            """計算美股主題等權平均報酬、相對強弱與相對動能。"""
            indicators = US_INDUSTRY_INDICATORS.get(internal_name, [])
            mapped_symbols = [str(item.get("symbol", "")).upper() for item in indicators]
            usable_symbols = [
                symbol for symbol in mapped_symbols
                if symbol in us_daily_returns_df.columns
                and symbol in us_quote_lookup
                and us_quote_lookup[symbol].get("status") == "OK"
            ]
            result = {
                "today_return": None,
                "yesterday_return": None,
                "return_5d": None,
                "return_20d": None,
                "relative_strength": None,
                "relative_momentum": None,
                "symbols": ", ".join(mapped_symbols) if mapped_symbols else "未設定對照",
                "valid_symbols": ", ".join(usable_symbols) if usable_symbols else "無可用行情",
                "valid_count": len(usable_symbols),
                "total_count": len(mapped_symbols),
                "data_dates": ", ".join(sorted({
                    str(us_quote_lookup[symbol].get("data_date") or "")
                    for symbol in usable_symbols
                    if us_quote_lookup[symbol].get("data_date")
                })) or "無可用日期",
            }
            if not usable_symbols:
                return result

            # 每個主題以有效美股指標股的每日報酬算術平均建立等權觀察序列。
            theme_daily = us_daily_returns_df[usable_symbols].mean(axis=1, skipna=True).dropna()
            if theme_daily.empty:
                return result
            theme_index = (1.0 + theme_daily / 100.0).cumprod() * 100.0
            result["today_return"] = float(theme_daily.iloc[-1])
            if len(theme_daily) >= 2:
                result["yesterday_return"] = float(theme_daily.iloc[-2])
            if len(theme_index) >= 6:
                result["return_5d"] = float((theme_index.iloc[-1] / theme_index.iloc[-6] - 1.0) * 100.0)
            if len(theme_index) >= 21:
                result["return_20d"] = float((theme_index.iloc[-1] / theme_index.iloc[-21] - 1.0) * 100.0)

            # 相對強弱：主題等權指數相對 SPY，除以 50 日相對比值均值；
            # 相對動能：RS 相較 10 個交易日前的變化。指標 100 代表中性。
            benchmark_returns = us_daily_returns_df.get("SPY")
            if benchmark_returns is not None:
                aligned = pd.concat([
                    theme_daily.rename("theme"),
                    benchmark_returns.rename("benchmark"),
                ], axis=1).dropna()
                if len(aligned) >= 61:
                    theme_path = (1.0 + aligned["theme"] / 100.0).cumprod()
                    benchmark_path = (1.0 + aligned["benchmark"] / 100.0).cumprod()
                    relative = theme_path / benchmark_path * 100.0
                    relative_strength = relative / relative.rolling(50, min_periods=50).mean() * 100.0
                    relative_momentum = relative_strength / relative_strength.shift(10) * 100.0
                    if not relative_strength.dropna().empty:
                        result["relative_strength"] = float(relative_strength.dropna().iloc[-1])
                    if not relative_momentum.dropna().empty:
                        result["relative_momentum"] = float(relative_momentum.dropna().iloc[-1])
            return result

        # 所選「排行依據」直接決定美股圖顯示的數值；兩張圖各自按所選指標由高至低排序。
        us_average_rows = []
        for row in average_rows:
            internal_name = next(
                (name for name in by_name if rotation_label(name) == row["name"]),
                row["name"],
            )
            theme_metrics = _us_theme_metrics(internal_name)
            us_value = theme_metrics.get(us_metric_selected)
            if us_value is not None:
                try:
                    us_value = float(us_value)
                    if not np.isfinite(us_value):
                        us_value = None
                except (TypeError, ValueError):
                    us_value = None
            us_average_rows.append({
                "name": row["name"],
                "return_pct": us_value,
                **theme_metrics,
            })

        us_average_rows.sort(
            key=lambda row: (
                row["return_pct"] is None,
                -(float(row["return_pct"])
                  if row["return_pct"] is not None and np.isfinite(float(row["return_pct"]))
                  else 0.0),
            )
        )

        us_returns = [
            float(row["return_pct"]) for row in us_average_rows
            if row["return_pct"] is not None and np.isfinite(float(row["return_pct"]))
        ]
        tw_values = [float(row["metric_value"]) for row in average_rows]
        if metric_is_index:
            combined_values = [100.0] + tw_values + us_returns
            low, high = min(combined_values), max(combined_values)
            pad = max(0.5, (high - low) * 0.15)
            tw_xrange = [low - pad, high + pad]
            us_xrange = list(tw_xrange)
        elif sort_by in ("族群平均漲跌幅", "前一日平均漲跌幅"):
            # 同為單日百分比時，台股與美股共用百分比尺度，方便直接比較。
            daily_values = tw_values + us_returns
            max_abs = max([abs(value) for value in daily_values] or [1.0])
            pad = max(0.25, max_abs * 0.15)
            tw_xrange = [-max_abs - pad, max_abs + pad]
            us_xrange = list(tw_xrange)
        else:
            # 近 5／20 日為區間累積報酬，各市場以百分比單位顯示並保留合適刻度。
            tw_max_abs = max([abs(value) for value in tw_values] or [1.0])
            tw_pad = max(0.25, tw_max_abs * 0.15)
            tw_xrange = [-tw_max_abs - tw_pad, tw_max_abs + tw_pad]
            us_max_abs = max([abs(value) for value in us_returns] or [1.0])
            us_pad = max(0.25, us_max_abs * 0.15)
            us_xrange = [-us_max_abs - us_pad, us_max_abs + us_pad]
        chart_height = max(620, len(average_rows) * 25)

        def build_industry_return_chart(rows, *, us_market: bool):
            values = [
                row["return_pct"] if us_market else row["metric_value"]
                for row in rows
            ]
            threshold = 100.0 if metric_is_index else 0.0
            colors = [
                "#c62828" if value is not None and value > threshold
                else "#16803c" if value is not None and value < threshold
                else "#94a3b8"
                for value in values
            ]
            if us_market:
                chart_customdata = [
                    [row["symbols"], row["valid_symbols"], row["valid_count"], row["total_count"], row["data_dates"]]
                    for row in rows
                ]
                us_value_format = ".2f" if metric_is_index else "+.2f"
                us_value_suffix = "" if metric_is_index else "%"
                us_metric_title = "相對強弱指數" if sort_by == "相對強弱" else "相對動能指數" if sort_by == "相對動能" else us_chart_metric_label
                hovertemplate = (
                    f"%{{y}}<br>美股對應指標{us_metric_title}：%{{x:{us_value_format}}}{us_value_suffix}"
                    "<br>對照代號：%{customdata[0]}"
                    "<br>有效行情：%{customdata[2]}/%{customdata[3]}"
                    "<br>有效代號：%{customdata[1]}"
                    "<br>美東資料日期：%{customdata[4]}<extra></extra>"
                )
                labels = [
                    (f"{float(value):.2f}" if metric_is_index else f"{float(value):+.2f}%")
                    if value is not None and np.isfinite(float(value)) else "—"
                    for value in values
                ]
                x_values = [
                    float(value) if value is not None and np.isfinite(float(value)) else None
                    for value in values
                ]
            else:
                tw_value_format = ".2f" if metric_is_index else "+.2f"
                tw_suffix = "" if metric_is_index else "%"
                if metric_key == "today_return":
                    chart_customdata = [[row["rising"], row["falling"]] for row in rows]
                    hovertemplate = (
                        f"%{{y}}<br>{chart_metric_label}：%{{x:{tw_value_format}}}{tw_suffix}"
                        "<br>上漲／下跌代表股：%{customdata[0]}／%{customdata[1]}<extra></extra>"
                    )
                else:
                    chart_customdata = None
                    hovertemplate = (
                        f"%{{y}}<br>{chart_metric_label}：%{{x:{tw_value_format}}}{tw_suffix}<extra></extra>"
                    )
                labels = [
                    f"{float(value):.2f}" if metric_is_index else f"{float(value):+.2f}%"
                    for value in values
                ]
                x_values = [float(value) for value in values]

            chart = go.Figure(go.Bar(
                x=x_values,
                y=[row["name"] for row in rows],
                orientation="h",
                marker=dict(color=colors),
                text=labels,
                textposition="outside",
                cliponaxis=False,
                customdata=chart_customdata,
                hovertemplate=hovertemplate,
            ))
            if us_market:
                missing_names = [
                    row["name"] for row in rows
                    if row["return_pct"] is None
                    or not np.isfinite(float(row["return_pct"]))
                ]
                if missing_names:
                    # 即使某個產業暫時抓不到美股報價，仍在該列標註「—」，不要誤認為 0%。
                    chart.add_trace(go.Scatter(
                        x=[100.0 if metric_is_index else 0.0] * len(missing_names),
                        y=missing_names,
                        mode="text",
                        text=["—"] * len(missing_names),
                        textposition="middle right",
                        textfont=dict(size=12, color="#64748b"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
            chart.add_vline(
                x=(100.0 if metric_is_index else 0.0) if us_market else axis_baseline,
                line_color="#8793a1",
                line_width=1,
            )
            if us_market and metric_is_index:
                xaxis_title = f"對應美股指標{us_metric_title}（100＝中性基準）"
            elif us_market:
                xaxis_title = f"對應美股指標{us_chart_metric_label}（%）"
            elif metric_is_index:
                xaxis_title = f"{chart_metric_label}（100＝中性基準）"
            else:
                xaxis_title = f"{chart_metric_label}（%）"
            chart.update_layout(
                height=chart_height, autosize=True,
                margin=dict(l=165, r=55, t=14, b=48),
                showlegend=False,
                xaxis_title=xaxis_title,
                yaxis_title=None,
                hovermode="closest",
                bargap=0.24,
            )
            chart.update_xaxes(
                range=us_xrange if us_market else tw_xrange,
                zeroline=False,
                gridcolor="rgba(130,145,160,0.18)",
            )
            chart.update_yaxes(
                autorange="reversed",
                categoryorder="array",
                categoryarray=[row["name"] for row in rows],
                automargin=True,
                showticklabels=True,
            )
            return chart

        # 台股與美股均依所選排行依據排序，兩側均顯示產業族群名稱。
        tw_chart = build_industry_return_chart(average_rows, us_market=False)
        us_chart = build_industry_return_chart(us_average_rows, us_market=True)
        tw_col, us_col = st.columns(2, gap="medium")
        with tw_col:
            st.markdown(
                f'<span style="color:#9ca3af; font-size:0.85rem;">盤後看</span> '
                f'<strong>台股｜產業族群{chart_metric_label}</strong>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                tw_chart,
                use_container_width=True,
                key=f"rotation_published_avg_return_tw_{metric_key}_{average_day}_{len(average_rows)}",
                config={"displaylogo": False, "responsive": True},
            )
        with us_col:
            if sort_by == "相對強弱":
                us_heading = "美股｜產業族群相對強弱"
            elif sort_by == "相對動能":
                us_heading = "美股｜產業族群相對動能"
            else:
                us_heading = f"美股｜產業族群{us_chart_metric_label}（%）"
            st.markdown(
                '<span style="color:#9ca3af; font-size:0.85rem;">盤前看</span> '
                f'<strong>{us_heading}</strong>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                us_chart,
                use_container_width=True,
                key=f"rotation_published_avg_return_us_{metric_key}_{average_day}_{len(us_average_rows)}",
                config={"displaylogo": False, "responsive": True},
            )
        st.caption(
            f"目前排行依據：{sort_by}；台股資料日期：{average_date or '未知'}。"
            f"台股與美股均依所選指標重新計算並由高至低排序；美股值為該主題對照指標股的等權平均。"
            "單日／區間報酬以百分比呈現；相對強弱與相對動能以 100 為中性基準，美股相對指標使用主題等權報酬對照 SPY，RS 採 50 日均值、動能比較 10 個交易日前。"
            "兩張圖左側均顯示產業族群名稱；紅色代表高於基準、綠色代表低於基準。美股行情快取 15 分鐘，且部分美股為供應鏈代理，並非完全相同的純標的。"
        )

    _render_flow_heatmap(payload)

    _sec('03｜產業輪動圖')
    st.caption("每條線代表主題相對加權指數的軌跡，每 5 個交易日取一點。RS 高於 100 代表相對強弱高於自身 50 日均值；動能高於 100 代表較 10 個交易日前增強。預設顯示 1 個月與「領先」象限。固定 1400×760 畫布：滑鼠滾輪縮放、雙擊重設；放大後可切換工具列平移，原始大小時不能平移出界，縮小不會超出原始視圖。")
    quadrant_options = ["領先", "轉弱", "改善", "落後", "全選"]
    quadrant_filter = st.radio(
        "當前所在的象限",
        quadrant_options,
        index=0,
        horizontal=True,
        key="rotation_published_quadrant_filter_v1",
        help="只顯示目前位於指定象限的產業軌跡；「全選」會顯示所有象限。",
    )
    plot_selected = [
        name for name in selected
        if quadrant_filter == "全選"
        or str(by_name.get(name, {}).get("quadrant", "")) == quadrant_filter
    ]

    n_points = {"1 個月": 6, "3 個月": 15, "半年": 28, "1 年": 53}[period]
    fig = go.Figure()
    colors = {"領先": "#c65b62", "轉弱": "#d9a441", "落後": "#64748b", "改善": "#319879"}
    all_x, all_y = [], []
    rrg_plot_groups = 0
    for name in plot_selected:
        g = by_name.get(name, {})
        display_label = rotation_label(name)
        pts = [x for x in g.get("rrg", []) if isinstance(x, dict)][-n_points:]
        if not pts:
            continue
        rrg_plot_groups += 1
        xs = [float(x["rs"]) for x in pts]
        ys = [float(x["momentum"]) for x in pts]
        labels = [str(x.get("date", ""))[5:] for x in pts]
        color = colors.get(str(g.get("quadrant")), "#64748b")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=display_label,
            line=dict(width=2, color=color),
            marker=dict(size=[6] * max(0, len(pts) - 1) + [12], color=color),
            text=labels,
            hovertemplate=display_label + "<br>日期 %{text}<br>相對強弱 %{x:.1f}<br>相對動能 %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]], mode="text", text=[f"<b>{display_label}</b>"],
            textposition="top center",
            textfont=dict(size=14, color=color, family="Arial Black, Arial, sans-serif"),
            showlegend=False, hoverinfo="skip",
        ))
        all_x.extend(xs)
        all_y.extend(ys)
    if all_x and all_y:
        xpad = max(2.5, (max(all_x) - min(all_x)) * .12)
        ypad = max(2.5, (max(all_y) - min(all_y)) * .12)
        xr = [min(min(all_x) - xpad, 97), max(max(all_x) + xpad, 103)]
        yr = [min(min(all_y) - ypad, 97), max(max(all_y) + ypad, 103)]
        # 以半透明淡色填滿四個 RRG 象限；區塊放在資料線後方，避免遮住軌跡。
        quadrant_fill = {
            "領先": "rgba(252, 232, 232, 0.58)",
            "轉弱": "rgba(255, 241, 223, 0.58)",
            "落後": "rgba(237, 241, 245, 0.58)",
            "改善": "rgba(229, 244, 237, 0.58)",
        }
        quadrant_rects = [
            ("改善", xr[0], 100, 100, yr[1]),
            ("領先", 100, 100, xr[1], yr[1]),
            ("落後", xr[0], yr[0], 100, 100),
            ("轉弱", 100, yr[0], xr[1], 100),
        ]
        for quadrant_name, x0, y0, x1, y1 in quadrant_rects:
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x0, y0=y0, x1=x1, y1=y1,
                fillcolor=quadrant_fill[quadrant_name],
                line=dict(width=0, color="rgba(80, 90, 100, 0)"),
                layer="below",
            )
        fig.add_shape(type="line", x0=100, x1=100, y0=yr[0], y1=yr[1], line=dict(color="#9aa7b5", dash="dash"))
        fig.add_shape(type="line", x0=xr[0], x1=xr[1], y0=100, y1=100, line=dict(color="#9aa7b5", dash="dash"))
        # 象限文字以半透明深灰色顯示，並避開中心交界附近的軌跡密集區。
        quadrant_labels = [
            ("改善", xr[0] + (100 - xr[0]) * 0.18, 100 + (yr[1] - 100) * 0.86),
            ("領先", 100 + (xr[1] - 100) * 0.82, 100 + (yr[1] - 100) * 0.86),
            ("落後", xr[0] + (100 - xr[0]) * 0.18, yr[0] + (100 - yr[0]) * 0.14),
            ("轉弱", 100 + (xr[1] - 100) * 0.82, yr[0] + (100 - yr[0]) * 0.14),
        ]
        for quadrant_name, label_x, label_y in quadrant_labels:
            fig.add_annotation(
                x=label_x, y=label_y, xref="x", yref="y",
                text=f"<b>{quadrant_name}</b>",
                showarrow=False, xanchor="center", yanchor="middle",
                font=dict(size=16, color="rgba(55, 65, 81, 0.76)", family="Arial Black, Arial, sans-serif"),
                opacity=0.92,
            )
        fig.update_xaxes(range=xr, minallowed=xr[0], maxallowed=xr[1], title_text="相對強弱", zeroline=False, fixedrange=False)
        fig.update_yaxes(range=yr, minallowed=yr[0], maxallowed=yr[1], title_text="相對動能", zeroline=False, fixedrange=False)
    # 固定畫布尺寸；滾輪可縮放、雙擊重設。軸範圍限制於原始視圖邊界內，未放大時不能平移出界。
    fig.update_layout(
        width=1400, height=760, autosize=False,
        margin=dict(l=45, r=35, t=28, b=40),
        hovermode="closest", dragmode="zoom",
        uirevision=f"rotation_rrg_{period}_{quadrant_filter}",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            font=dict(size=11, family="Arial Black, Arial, sans-serif"),
        ),
        showlegend=True,
    )
    if all_x and all_y:
        st.plotly_chart(
            fig,
            use_container_width=False,
            key=f"rotation_published_rrg_v4_{period}_{quadrant_filter}_{len(plot_selected)}",
            config={
                "scrollZoom": True,
                "doubleClick": "reset",
                "displaylogo": False,
                "responsive": False,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["zoomOut2d", "autoScale2d", "resetScale2d"],
            },
        )
    elif not plot_selected:
        if not selected:
            st.info("目前沒有選取任何產業群。請按「全選 27 群」或在上方選取要觀察的產業。")
        else:
            st.info(f"目前「{quadrant_filter}」象限沒有符合條件的已選產業。請切換到其他象限或「全選」。")
    else:
        st.info("符合篩選的產業目前缺少足夠的 RRG 歷史座標；請查看下方報酬表與成分股分類。")

    st.caption(
        f"產業篩選 {len(selected)} / {len(groups)} 群；符合「{quadrant_filter}」象限條件 {len(plot_selected)} 群，其中 {rrg_plot_groups} 群具備可繪製的 RRG 歷史軌跡。"
        "框選圖表可放大；放大後請切換工具列的平移工具移動視窗。縮小、重設與滑鼠滾輪縮放已關閉。"
        "若某群沒有軌跡，代表快照中的相對強弱／動能歷史點不足，不代表該產業分類不存在；"
        "該群仍會列在下方報酬表及成分股分類名單。"
    )

    _sec('04｜產業報酬與輪動狀態')
    fields = {"族群平均漲跌幅": "today_return", "前一日平均漲跌幅": "yesterday_return",
              "近 20 日": "return_20d", "近 5 日": "return_5d", "今日": "today_return",
              "相對強弱": "relative_strength", "相對動能": "relative_momentum"}
    field = fields[sort_by]
    table_groups = [g for g in groups if g.get("name") in selected]
    table_groups.sort(key=lambda g: g.get(field) if g.get(field) is not None else -999999, reverse=True)
    def pct(g, k):
        value = g.get(k)
        return f"{float(value):+.2f}%" if value is not None and np.isfinite(float(value)) else "—"
    table_rows = []
    for g in table_groups:
        rs, momentum = g.get("relative_strength"), g.get("relative_momentum")
        table_rows.append({
            "產業主題": rotation_label(g.get("name", "")),
            "有效／樣本": f"{g.get('members_fresh', 0)}/{g.get('members_total', 0)}",
            "資料覆蓋率": f"{g.get('data_coverage_pct', 0):.0f}%",
            "族群平均漲跌幅": pct(g, "today_return"),
            "前一日平均漲跌幅": pct(g, "yesterday_return"),
            "5 日": pct(g, "return_5d"), "20 日": pct(g, "return_20d"),
            "60 日": pct(g, "return_60d"), "今年以來": pct(g, "return_ytd"),
            "上漲／下跌": f"{g.get('rising', 0)}/{g.get('falling', 0)}",
            "52 週新高": g.get("year_high_count", 0),
            "RS": f"{float(rs):.1f}" if rs is not None else "—",
            "動能": f"{float(momentum):.1f}" if momentum is not None else "—",
            "象限": g.get("quadrant", "資料不足"),
        })
    metrics_df = pd.DataFrame(table_rows)
    st.dataframe(
        styled_frame(metrics_df, ["族群平均漲跌幅", "前一日平均漲跌幅", "5 日", "20 日", "60 日", "今年以來"]),
        use_container_width=True,
        hide_index=True,
    )

    # 每個產業主題各自提供可展開的成分股清單，行情直接沿用排程快照。
    st.markdown("**各產業主題概念股明細**")
    st.caption(
        "展開任一主題，即可查看納入該主題的代表概念股、個股收盤價、當日漲跌幅、"
        "5／20／60 日與今年以來報酬、成交量、成交金額及行情日期。"
    )
    for detail_group in table_groups:
        group_name = str(detail_group.get("name") or "")
        symbols = ROTATION_THEMES.get(group_name, [])
        group_components = {
            str(item.get("symbol", "")).strip(): item
            for item in (detail_group.get("components") or [])
            if isinstance(item, dict) and item.get("symbol") is not None
        }
        valid_count = int(detail_group.get("members_fresh") or 0)
        total_count = int(detail_group.get("members_total") or len(symbols))
        coverage = detail_group.get("data_coverage_pct")
        coverage_text = f"{float(coverage):.0f}%" if coverage is not None and pd.notna(pd.to_numeric(coverage, errors="coerce")) else "—"
        expander_title = (
            f"{rotation_label(group_name)}｜今日 {pct(detail_group, 'today_return')}｜"
            f"{detail_group.get('quadrant') or '資料不足'}｜有效行情 {valid_count}/{total_count}"
        )
        with st.expander(expander_title, expanded=False):
            st.caption(
                f"快照日期：{asof}｜資料覆蓋率：{coverage_text}｜"
                "族群平均報酬為有效代表股等權平均，並非官方產業指數。"
            )
            component_rows = []
            for symbol in symbols:
                item = group_components.get(str(symbol), {})
                close_value = pd.to_numeric(item.get("close"), errors="coerce")
                volume_value = pd.to_numeric(item.get("volume"), errors="coerce")
                turnover_value = pd.to_numeric(item.get("traded_value"), errors="coerce")
                has_quote = pd.notna(close_value)
                if not item:
                    status_text = "尚無行情"
                elif item.get("fresh"):
                    status_text = "最新有效行情"
                elif has_quote:
                    status_text = "非最新行情"
                else:
                    status_text = "尚無有效收盤價"
                component_rows.append({
                    "股票代號": str(symbol),
                    "概念股名稱": item.get("name") or NAME_FALLBACKS.get(str(symbol), str(symbol)),
                    "行情日期": item.get("data_date") or "尚無行情",
                    "收盤價": f"{float(close_value):,.2f}" if has_quote else "—",
                    "今日漲跌幅": pct(item, "change_1d"),
                    "5 日報酬": pct(item, "return_5d"),
                    "20 日報酬": pct(item, "return_20d"),
                    "60 日報酬": pct(item, "return_60d"),
                    "今年以來": pct(item, "return_ytd"),
                    "成交量（股）": f"{int(volume_value):,}" if pd.notna(volume_value) else "—",
                    "成交金額（元）": f"{float(turnover_value):,.0f}" if pd.notna(turnover_value) else "—",
                    "52 週新高": ("是" if item.get("year_high") else "否") if item else "—",
                    "行情狀態": status_text,
                })
            detail_df = pd.DataFrame(component_rows)
            if not detail_df.empty:
                st.dataframe(
                    styled_frame(
                        detail_df,
                        ["今日漲跌幅", "5 日報酬", "20 日報酬", "60 日報酬", "今年以來"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=min(560, 42 + len(detail_df) * 35),
                )
            else:
                st.info("此主題目前沒有設定代表概念股。")

    _sec('05｜美股產業指標股資訊')
    st.caption(
        "依台股供應鏈子產業整理美股上市指標公司，顯示最新可取得價格與當日漲跌幅。"
        "美股價格與日期來自 Yahoo Finance；每一筆以來源回傳的美東交易日期為準。"
    )
    us_display_to_internal = {rotation_label(name): name for name in US_INDUSTRY_INDICATORS}
    us_category_options = ["全部產業"] + list(us_display_to_internal.keys())
    chosen_us_category = st.selectbox(
        "美股指標股產業篩選",
        us_category_options,
        index=0,
        key="rotation_us_industry_category_v1",
        help="選擇一個台股子產業，即可查看對應的美股上市指標公司；全部產業會顯示完整對照。",
    )
    if chosen_us_category == "全部產業":
        us_selected_categories = list(US_INDUSTRY_INDICATORS.keys())
    else:
        us_selected_categories = [us_display_to_internal[chosen_us_category]]

    us_symbols = tuple(get_us_indicator_symbols(tuple(us_selected_categories)))
    us_quote_df = pd.DataFrame()
    with st.spinner(f"正在取得 {len(us_symbols)} 檔美股指標股的最新可用行情…"):
        try:
            us_quote_df = _cached_us_industry_quotes()
        except Exception as exc:
            st.warning(f"美股行情服務目前無法完成查詢：{type(exc).__name__}。仍會列出產業對照清單。")

    us_quote_lookup = {}
    if not us_quote_df.empty and "symbol" in us_quote_df.columns:
        us_quote_lookup = us_quote_df.set_index("symbol").to_dict(orient="index")

    us_rows = []
    for us_category in us_selected_categories:
        for indicator in US_INDUSTRY_INDICATORS.get(us_category, []):
            quote = us_quote_lookup.get(indicator["symbol"], {})
            price = quote.get("price")
            change = quote.get("change")
            change_pct = quote.get("change_pct")
            currency = quote.get("currency") or "USD"
            status = quote.get("status") or "行情暫不可用"
            us_rows.append({
                "對應台股產業": rotation_label(us_category),
                "美股代號": indicator["symbol"],
                "公司名稱": indicator["name"],
                "產業對照角色": indicator["role"],
                "最新價（USD）": f"{float(price):,.2f}" if price is not None and pd.notna(pd.to_numeric(price, errors="coerce")) else "—",
                "漲跌額（USD）": f"{float(change):+,.2f}" if change is not None and pd.notna(pd.to_numeric(change, errors="coerce")) else "—",
                "漲跌幅": f"{float(change_pct):+.2f}%" if change_pct is not None and pd.notna(pd.to_numeric(change_pct, errors="coerce")) else "—",
                "資料日期（美東）": quote.get("data_date") or "—",
                "行情狀態": "正常" if status == "OK" else status,
            })

    us_table = pd.DataFrame(us_rows)
    if not us_table.empty:
        success_count = sum(1 for row in us_rows if row["行情狀態"] == "正常")
        us_kpis = st.columns(3)
        us_kpis[0].metric("產業分類", f"{len(us_selected_categories)} 群")
        us_kpis[1].metric("指標股分類列", f"{len(us_rows)} 筆")
        us_kpis[2].metric("成功取得行情", f"{success_count} / {len(us_symbols)} 檔")
        st.dataframe(
            styled_frame(us_table, ["漲跌額（USD）", "漲跌幅"]),
            use_container_width=True,
            hide_index=True,
            height=min(640, 42 + len(us_table) * 35),
        )
        st.caption(
            "備註：美股指標股是對照各台股供應鏈主題的參考標的，並非每一類都有一對一的美國上市純標的。"
            "表中已在「產業對照角色」標明直接對照或供應鏈代理；同一美股可出現在多個子產業。"
            "最新價以美元（USD）呈現，紅色表示上漲、綠色表示下跌。"
        )
    else:
        st.info("所選產業目前沒有設定美股對照指標。")

    _sec('06｜本週象限變化')
    changes = payload.get("quadrant_changes") or []
    if changes:
        changes_df = pd.DataFrame(changes).rename(columns={"name": "產業主題", "from": "前一象限", "to": "目前象限"})
        if "產業主題" in changes_df.columns:
            changes_df["產業主題"] = changes_df["產業主題"].map(rotation_label)
        st.dataframe(changes_df, use_container_width=True, hide_index=True)
    else:
        st.caption("目前沒有可用的象限轉換紀錄，或輪動方向尚未跨越象限。")

    _sec('07｜成分股漲跌前十')
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

    _sec('08｜輪動解讀與資料限制')
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
        '<br>族群平均漲跌幅＝該主題有有效行情的代表股日漲跌幅算術平均，代表股等權計算；同一股票可同時屬於不同主題，因此請同時參考有效樣本數與資料覆蓋率。</div>',
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
            index=0,
            horizontal=False,
            key="rotation_period_choice",
        )
    with col_groups:
        selected_groups = st.multiselect(
            "分析主題（可自行增減）",
            list(ROTATION_THEMES.keys()),
            default=list(ROTATION_THEMES.keys()),
            key="rotation_selected_themes_all_v2",
            help="預設選取全部 27 個子產業；每一群都有明確的代表股票代號清單，同一檔股票可出現在多個主題。",
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

    _sec('01｜產業輪動圖')
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
            textfont=dict(size=14, color=color, family="Arial Black, Arial, sans-serif"),
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
        fig.update_xaxes(range=x_range, minallowed=x_range[0], maxallowed=x_range[1], title_text="相對強弱（100＝近 50 日平均）", zeroline=False)
        fig.update_yaxes(range=y_range, minallowed=y_range[0], maxallowed=y_range[1], title_text="相對動能（100＝10 日前水準）", zeroline=False)
    fig.update_layout(
        width=1400, height=760, autosize=False,
        margin=dict(l=45, r=35, t=24, b=34),
        hovermode="closest", dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True,
    )
    st.caption("操作：滑鼠滾輪放大／縮小；雙擊重設回完整原圖。放大後切換工具列平移工具可移動視窗；原始大小時軸範圍鎖定，不能位移出界或縮小到原圖以外。")
    st.plotly_chart(
        fig, use_container_width=False, key=f"rotation_rrg_v2_{signature}",
        config={
            "scrollZoom": True,
            "doubleClick": "reset",
            "displaylogo": False,
            "responsive": False,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["zoomOut2d", "autoScale2d", "resetScale2d"],
        },
    )

    _sec('02｜族群績效與輪動狀態')
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

    _sec('03｜四象限判讀')
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
    if not is_authenticated():
        st.warning("請先登入，才能刪除自選股。")
        return
    new_watchlist = [x for x in st.session_state.watchlist if x != code]
    st.session_state.watchlist = new_watchlist
    if not save_watchlist(new_watchlist):
        st.session_state.watchlist_save_failed = True
    if st.session_state.get('selected') == code:
        st.session_state.selected = new_watchlist[0] if new_watchlist else WATCHLIST_DEFAULT[0]
    st.session_state.ai_selected = None
    st.session_state.ai_result = None
    st.rerun()


def render_watchlist(compact: bool = False):
    rows, errors = watch_rows(tuple(st.session_state.watchlist))
    chip_map = chip_signals.get_nonblocking(st.session_state.watchlist) if compact else {}
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

                    limit_up = pd.notna(price) and pd.notna(lup) and abs(float(price)-float(lup)) < 1e-6
                    limit_down = pd.notna(price) and pd.notna(ldn) and abs(float(price)-float(ldn)) < 1e-6
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
                        # 一檔股票 = 一張完整卡片；點選卡片任一處即選取（透明按鈕覆蓋整張卡片）
                        is_sel = str(r['code']) == str(st.session_state.get('selected'))
                        # 漲跌停、選取的外框靠 key 上的標記上色：不要用 :has()，Plotly 量文字時每次都要重算整頁樣式（實測 35ms 對 3ms）
                        card_state = {"limit-up": "_lu", "limit-down": "_ld"}.get(name_class, "") + ("_sel" if is_sel else "")
                        with st.container(key=f"wc_{r['code']}_{idx}{card_state}"):
                            st.markdown(
                                f"<div class='wc-info {name_class}{' wc-selected' if is_sel else ''}{' wc-has-del' if is_authenticated() else ''}'>"
                                f"<div class='wc-name'>{r['name']}<span>（{r['code']}）</span></div>"
                                "<div class='wc-vals'>"
                                f"<span class='wc-price' style='color:{vcolor}'>{price_txt}</span>"
                                f"<span style='color:{vcolor}'>{ch_txt}</span>"
                                f"<span style='color:{vcolor}'>{cp_txt}</span>"
                                "</div>"
                                f"<div class='wc-chips'>{_flow_chips_html(r['code'])}{_signal_chips_html(chip_map.get(clean_symbol(r['code'])), empty_text='', pending_text='籌碼計算中…')}</div>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                            if st.button("\u200b", key=f"watch_select_{r['code']}_{idx}", help=f"選取 {r['name']}（{r['code']}）", use_container_width=True):
                                _watch_select(r['code'])
                            if is_authenticated() and st.button("刪除", key=f"wcdel_{r['code']}_{idx}", help=f"從自選股移除 {r['name']}（{r['code']}）"):
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
                            if is_authenticated() and st.button("\u200b", key=f"watch_remove_{r['code']}_{idx}", icon=":material/delete:", use_container_width=True, type="secondary", help=f"從自選股移除 {r['code']}"):
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
def render_pinned_selected() -> None:
    """自選股側欄最上方的「分析標的」精簡卡（淺色、置頂常駐）。"""
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
    biz = ""
    try:
        bp = get_business_profiles_cached((selected,)).get(selected, {})
        biz = bp.get("business_group") or ""
    except Exception:
        pass

    live = {}
    if is_market_open_now():
        try: live = get_quote(selected)
        except Exception: live = {}
    elif sr is not None:
        live = {"tradePrice": sr.get("price", np.nan), "change": sr.get("change", np.nan), "changePercent": sr.get("change_pct", np.nan)}
    lp = pd.to_numeric(live.get("tradePrice", live.get("closePrice", np.nan)), errors="coerce")
    lch = pd.to_numeric(live.get("change", np.nan), errors="coerce")
    lcp = pd.to_numeric(live.get("changePercent", np.nan), errors="coerce")
    ref = pd.to_numeric(live.get("referencePrice", live.get("previousClose", np.nan)), errors="coerce")
    if pd.isna(lch) and pd.notna(lp) and pd.notna(ref): lch = lp - ref
    if pd.isna(lcp) and pd.notna(lch) and pd.notna(ref) and ref != 0: lcp = lch / ref * 100
    color = "#c62828" if pd.notna(lch) and lch > 0 else "#0f8a5f" if pd.notna(lch) and lch < 0 else "#475569"

    try:
        taiex = get_taiex()
        tx = taiex.iloc[-1] if not taiex.empty else None
    except Exception:
        tx = None
    if tx is not None:
        ixch = float(tx["change"])
        ixcolor = "#c62828" if ixch > 0 else "#0f8a5f" if ixch < 0 else "#475569"
        ix_html = (f"<div class='pin-ix'><span class='pin-ix-label'>加權指數</span>"
                   f"<b>{float(tx['index']):,.2f}</b>"
                   f"<span style='color:{ixcolor}'>{ixch:+,.2f}　{float(tx['change_pct']):+.2f}%</span></div>")
    else:
        ix_html = "<div class='pin-ix'><span class='pin-ix-label'>加權指數</span><b>--</b></div>"

    price_txt = "--" if pd.isna(lp) else f"{lp:,.2f}"
    ch_txt = "--" if pd.isna(lch) else f"{lch:+,.2f}"
    cp_txt = "--" if pd.isna(lcp) else f"{lcp:+.2f}%"
    meta = f"{market}｜{industry_code} {industry_name}" + (f"｜{biz}" if biz else "")
    with st.container(key="side_pinned"):
        st.markdown(
            "<div class='pin-card'>"
            "<div class='pin-tag'>分析標的</div>"
            f"<div class='pin-name'>{name}<span>（{selected}）</span></div>"
            f"<div class='pin-vals' style='color:{color}'><b>{price_txt}</b><span>{ch_txt}</span><span>{cp_txt}</span></div>"
            f"<div class='pin-meta'>{meta}</div>"
            f"{ix_html}"
            "</div>",
            unsafe_allow_html=True,
        )


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

# Canvas 繪圖的 K 線（TradingView Lightweight Charts）：十字游標移動不需重畫整張 SVG，大量 K 棒也順。
_LWC_URL = "https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"


def _kline_chart_html(d: pd.DataFrame, daily: bool, mas: list, reset_version: int = 0) -> str:
    stamps = pd.to_datetime(d["date"], errors="coerce")
    if daily:
        times = stamps.dt.strftime("%Y-%m-%d").tolist()
        vol_div, vol_unit = 1000, "張"     # 日 K 成交量單位是股
    else:
        # 圖表時間以 UTC 顯示，直接把台北時間當成 UTC 秒數傳入，畫面上就是台北時間
        times = [int((t.tz_localize(None) if t.tzinfo else t).timestamp()) for t in stamps]
        vol_div, vol_unit = 1, "張"
    o, h, l, c, v = (d[k].astype(float).tolist() for k in ("open", "high", "low", "close", "volume"))
    prev = [np.nan] + c[:-1]
    bars = [{"time": t, "open": a, "high": b, "low": e, "close": f, "prev": None if np.isnan(pc) else pc,
             "vol": round(x / vol_div)} for t, a, b, e, f, pc, x in zip(times, o, h, l, c, prev, v)]
    lines = [{"label": label, "color": color,
              "data": [{"time": t, "value": round(float(y), 4)} for t, y in zip(times, ys) if pd.notna(y)]}
             for label, color, ys in mas]
    payload = json.dumps({"bars": bars, "mas": lines, "daily": daily, "unit": vol_unit, "v": reset_version},
                         ensure_ascii=False)
    return """<!doctype html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;height:100%;background:#fff;font-family:Arial,"Microsoft JhengHei",sans-serif}
#wrap{position:relative;height:100%}#chart{position:absolute;inset:0}
#legend{position:absolute;left:8px;top:6px;z-index:3;font-size:12.5px;line-height:1.55;color:#3e3a3a;
  background:rgba(255,255,255,.86);padding:3px 8px;border-radius:6px;pointer-events:none;font-variant-numeric:tabular-nums}
#legend b{font-weight:700}#legend .up{color:#e04545}#legend .dn{color:#14935a}#legend .ma{margin-right:10px}
#err{display:none;padding:20px;color:#a33;font-size:13px}
</style></head><body><div id="wrap"><div id="chart"></div><div id="legend"></div><div id="err">K 線圖元件載入失敗，請重新整理頁面。</div></div>
<script src="__LWC__"></script>
<script>
const P = __DATA__;
if (!window.LightweightCharts) { document.getElementById('err').style.display = 'block'; }
else {
const el = document.getElementById('chart'), lg = document.getElementById('legend');
const chart = LightweightCharts.createChart(el, {
  autoSize: true,
  layout: {background: {color: '#ffffff'}, textColor: '#5b5555', fontSize: 11},
  grid: {vertLines: {color: '#eef1f4'}, horzLines: {color: '#eef1f4'}},
  crosshair: {mode: LightweightCharts.CrosshairMode.Normal},
  rightPriceScale: {borderColor: '#e2dcdc', scaleMargins: {top: 0.08, bottom: 0.26}},
  // 左右邊界固定在第一根與最後一根 K 棒：不能縮小到比完整原圖還小，原圖大小時也無法平移，放大後才能左右移動
  timeScale: {borderColor: '#e2dcdc', timeVisible: !P.daily, secondsVisible: false, rightOffset: 0,
              fixLeftEdge: true, fixRightEdge: true, minBarSpacing: 0.01},
  // 價格軸跟著可見範圍自動縮放，不讓拖曳價格軸把圖壓扁
  handleScale: {mouseWheel: true, pinch: true, axisPressedMouseMove: {time: true, price: false}, axisDoubleClickReset: true},
  handleScroll: {mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false},
  localization: {locale: 'zh-TW',
    timeFormatter: t => typeof t === 'number'
      ? new Date(t * 1000).toISOString().slice(P.daily ? 0 : 5, 16).replace('T', ' ')
      : (typeof t === 'string' ? t : `${t.year}-${String(t.month).padStart(2,'0')}-${String(t.day).padStart(2,'0')}`)},
});
const candle = chart.addCandlestickSeries({upColor: '#ef4444', downColor: '#10b981', borderUpColor: '#ef4444',
  borderDownColor: '#10b981', wickUpColor: '#ef4444', wickDownColor: '#10b981', priceLineVisible: false});
candle.setData(P.bars.map(b => ({time: b.time, open: b.open, high: b.high, low: b.low, close: b.close})));
const vol = chart.addHistogramSeries({priceScaleId: 'vol', priceFormat: {type: 'volume'}, lastValueVisible: false,
  priceLineVisible: false});
chart.priceScale('vol').applyOptions({scaleMargins: {top: 0.78, bottom: 0}});
vol.setData(P.bars.map(b => ({time: b.time, value: b.vol, color: b.close >= b.open ? 'rgba(239,68,68,.55)' : 'rgba(16,185,129,.55)'})));
const maSeries = P.mas.map(m => {
  const s = chart.addLineSeries({color: m.color, lineWidth: 1.4, priceLineVisible: false, lastValueVisible: false,
    crosshairMarkerVisible: false});
  s.setData(m.data);
  return {m, s, map: new Map(m.data.map(x => [String(x.time), x.value]))};
});
const byTime = new Map(P.bars.map(b => [String(b.time), b]));
const fmt = x => x == null ? '--' : x.toLocaleString('zh-TW', {maximumFractionDigits: 2});
const key = t => typeof t === 'object' && t ? `${t.year}-${String(t.month).padStart(2,'0')}-${String(t.day).padStart(2,'0')}` : String(t);
function show(b) {
  if (!b) return;
  const ch = b.prev == null ? null : b.close - b.prev, pct = ch == null ? null : ch / b.prev * 100;
  const cls = ch > 0 ? 'up' : ch < 0 ? 'dn' : '';
  const when = typeof b.time === 'number' ? new Date(b.time * 1000).toISOString().slice(5, 16).replace('T', ' ') : b.time;
  const mas = maSeries.map(x => { const v = x.map.get(String(b.time));
    return v == null ? '' : `<span class="ma" style="color:${x.m.color}">${x.m.label} ${fmt(v)}</span>`; }).join('');
  lg.innerHTML = `<b>${when}</b>　開 ${fmt(b.open)}　高 ${fmt(b.high)}　低 ${fmt(b.low)}　收 <b class="${cls}">${fmt(b.close)}</b>`
    + (ch == null ? '' : `　<span class="${cls}">${ch > 0 ? '▲' : ch < 0 ? '▼' : ''}${fmt(Math.abs(ch))}（${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%）</span>`)
    + `　量 ${fmt(b.vol)} ${P.unit}` + (mas ? `<br>${mas}` : '');
}
const last = P.bars[P.bars.length - 1];
show(last);
chart.subscribeCrosshairMove(p => show(p && p.time !== undefined ? byTime.get(key(p.time)) : last));
chart.timeScale().fitContent();
}
</script></body></html>""".replace("__LWC__", _LWC_URL).replace("__DATA__", payload)


def render_kline():
    """互動式技術 K 線：日期簡寫、交易日等距排列，價格與成交量共用時間軸。"""
    selected = st.session_state.selected
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
            div[class*="st-key-technical-reset-button"] button {
                background:#fbf1f1 !important;
                background-color:#fbf1f1 !important;
                color:#8e2b2f !important;
                border:1px solid #e6d3d3 !important;
                border-radius:8px !important;
                box-shadow:none !important;
                font-weight:700 !important;
                white-space:nowrap !important;
            }
            div[class*="st-key-technical-reset-button"] button:hover {
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

        ma_colors = {5: "#2563eb", 10: "#f59e0b", 20: "#8b5cf6", 60: "#0891b2", 120: "#64748b", 240: "#a16207"}
        mas = [(label, ma_colors[window], d["close"].rolling(window=window, min_periods=window).mean())
               for label, window, _default_on in ma_specs if enabled_ma.get(window)]
        st.caption("滑鼠移到 K 棒上，左上角顯示當天開高低收、漲跌與成交量；滾輪放大／縮小（最小就是完整原圖），放大後按住拖曳左右平移，按「回到最新」重設。非交易日不占用圖表空間。")
        st.iframe(_kline_chart_html(d, daily=(k_period == "日K"), mas=mas, reset_version=reset_version), height=640)
    except Exception as e:
        st.error(f"{k_period} 資料取得失敗：{e}")

# ============================================================
# 成交量概況
# ============================================================
def render_volume_snapshot():
    selected = st.session_state.selected
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
    from entry_model import fit_model, load_latest_saved_model, load_saved_model, load_training_intraday, save_saved_model
    try:
        saved = load_saved_model(selected, training_end_key)
        if saved is not None and hasattr(saved, "reg") and hasattr(saved, "clf"):
            return saved
    except Exception:
        pass
    try:
        # 訓練一天只做一次：Fugle 每分鐘額度滿時等待空檔，而不是直接放棄
        hist = load_training_intraday(FugleClient(wait_on_limit=True), selected, days=140)
        model = fit_model(hist, horizon_bars=6)
        save_saved_model(selected, training_end_key, model)
        return model
    except Exception as exc:
        fallback = load_latest_saved_model(selected)
        if fallback is not None:
            return fallback
        raise RuntimeError(f"無法建立 5 分 K 模型：{exc}") from exc


def render_entry_prediction():
    selected = st.session_state.selected

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
        from entry_model import predict_entry
        try:
            daily_k = get_history(selected, 150)
        except Exception:
            daily_k = None
        result = predict_entry(model, k5, float(live_price), daily=daily_k, market_open=is_market_open_now())
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
    c6.metric("K線型態（5分／日）", f"{result['pattern_name']}／{result['daily_pattern_name']}")
    c7.metric("未來30分回檔機率" + ("" if result["model_has_skill"] else "（歷史頻率）"), f"{result['opportunity_probability']*100:.1f}%")
    c8.metric("VWAP", f"{result['vwap']:.2f}")

    st.markdown("#### K 線型態與趨勢分析")
    if not result.get("market_open", True):
        st.caption("目前非盤中：以下依最近一個交易日的收盤結構判斷，作為下一個交易日開盤的參考。")
    a1, a2, a3, a4 = st.columns(4)
    a1.markdown(f"**目前趨勢**  \n### {result['trend_state']}  \n日線：{result['daily_trend_state']}")
    a2.markdown(f"**5 分 K 型態**  \n### {result['pattern_name']}  \n{result['pattern_desc']}")
    a3.markdown(f"**日 K 型態**  \n### {result['daily_pattern_name']}  \n{result['daily_pattern_desc']}")
    a4.markdown(f"**型態偏多分數（5分／日）**  \n### {result['pattern_score']:+.1f}／{result['daily_pattern_score']:+.1f}  \n-1 偏空 ～ +1 偏多")

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
    d4.metric("6K量能比", "資料不足" if result.get("volume_ratio_6") is None else f"{result['volume_ratio_6']:.2f}x")

    # 這一區把決策理由直接翻成使用者可讀的分析。
    st.markdown("#### 進場判斷理由")
    reasons = result["decision_reason"].split("；") if result["decision_reason"] else ["等待價格結構進一步確認"]
    # 固定成單一 Markdown 元件，避免盤中 2 秒更新時因理由數量變動造成 React DOM
    # insertBefore / removeChild reconciliation 問題。
    reason_text = "\n".join(f"- {str(reason).strip()}" for reason in reasons if str(reason).strip())
    st.markdown(reason_text or "- 等待價格結構進一步確認")

    bss = getattr(model, "brier_skill", float("nan"))
    st.markdown(
        f"模型：**Random Forest 5 分 K 回檔模型 v3**｜訓練樣本 **{model.train_samples:,}**（{model.train_days} 個交易日）｜"
        f"前推驗證（最後 {getattr(model, 'oof_days', 0)} 個交易日，樣本外）：回檔機率 Brier skill **{bss:+.3f}**"
        f"（> 0 代表比歷史頻率準）、準確率 **{model.oof_opportunity_accuracy*100:.1f}%**（基準 {model.baseline_accuracy*100:.1f}%）、"
        f"最低價誤差 **{model.oof_mae*100:.2f}%**（基準 {model.naive_mae*100:.2f}%）｜"
        f"評分採用：{'模型機率' if result['model_has_skill'] else '歷史回檔頻率'}、"
        f"{'模型預測最低價' if result['low_model_has_skill'] else '現價 − 0.5 ATR'}"
    )
    st.caption("技術面量化參考，不構成投資建議。")

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
        render_stock_note_for_symbol(BASE, selected)
    except Exception as e:
        st.caption(f"AI Agent 協作層暫不可用，不影響原本進場模型：{e}")


# ============================================================
# Big vs retail
# ============================================================
# ============================================================
# Chip signals - 外資／投信／主力連買連賣、融資融券增減、大戶散戶（看多紅、看空綠）
# ============================================================
_SIG_WORDS = {"foreign": ("買超", "賣超"), "trust": ("買超", "賣超"), "main": ("買超", "賣超"),
              "margin": ("增加", "減少"), "short": ("增加", "減少")}
_SIG_BULL_WHEN_UP = {"foreign": True, "trust": True, "main": True, "margin": True, "short": False}


def _flow_metric() -> str:
    """資金輪動的計算方式（產業分析頁的切換）；自選股卡片與個股特徵跟著用。"""
    return st.session_state.get("flow_metric", "share20")


@st.cache_data(ttl=600, show_spinner=False, max_entries=512)
def _flow_rows(code: str, _generated_at: str, metric: str = "share20") -> list[dict]:
    """個股的產業資金增減（產業主題 → 產業鏈細分 → 官方產業別），前 5 名排前面；依產業分析快照的產生時間快取。"""
    rows = stock_features.symbol_flow_rows(clean_symbol(code), load_published_rotation(BASE), metric)
    return sorted(rows, key=lambda r: (not r["top"], -abs(r["change"])))


def _flow_chips_html(code: str, limit: int = 2) -> str:
    """自選股卡片上的產業資金增減標籤：前 5 名實心，其餘一般；一檔屬於多個主題時最多顯示 limit 個。"""
    payload = load_published_rotation(BASE)
    rows = _flow_rows(code, str(payload.get("generated_at") or ""), _flow_metric())
    chips = []
    for r in rows[:limit]:
        up = r["side"] == "up"
        text = f"{r['name']} 資金{'流入' if up else '流出'}#{r['rank']}" if r["top"] else f"{r['name']} 資金{'增' if up else '減'}"
        why = f"{stock_features.flow_why_text(r)}（{stock_features.FLOW_METRICS[r['metric']]}）"
        chips.append(f'<span class="sig sig-{"bull" if up else "bear"}{" sig-top" if r["top"] else ""}" title="{_esc(why)}">{_esc(text)}</span>')
    return "".join(chips)


def _signal_chips_html(res: dict | None, empty_text: str = "目前沒有連續 3 日以上的籌碼訊號", pending_text: str = "籌碼計算中…") -> str:
    if res is None:
        return f'<span class="sig-wait">{pending_text}</span>' if pending_text else ""
    sigs = res.get("signals") or []
    if not sigs:
        return f'<span class="sig-none">{empty_text}</span>' if empty_text else ""
    return "".join(f'<span class="sig sig-{s["tone"]}">{_esc(s["text"])}</span>' for s in sigs)


# ============================================================
# 自選股盤中焦點：自選股當日 1 分 K 的價量事件，依時間軸列出
# ============================================================
FOCUS_REFRESH_SECONDS = 120   # 23 檔 × 每 2 分鐘一次 ≈ 每分鐘 12 次 Fugle，留額度給其他區塊
FOCUS_MAX_CARDS = 40
FOCUS_NEW_MINUTES = 10


@st.cache_data(ttl=21600, show_spinner=False, max_entries=256)
def _focus_daily(code: str, day_iso: str) -> pd.DataFrame:
    """事件當天以前的日 K（昨收、昨高低、20／60 日高低、20 日均量）。"""
    day = pd.Timestamp(day_iso)
    stale = lambda d: d.empty or (day - d["date"].max()).days > 7
    h = get_history(code, 130)
    h = h[h["date"] < day] if not h.empty else h
    if stale(h):
        # 本機日 K 快取可能停在很久以前（之前補資料時撞到 Fugle 額度），直接抓這段期間
        try:
            h = client.historical_candles(code, (day - pd.Timedelta(days=130)).date(), (day - pd.Timedelta(days=1)).date(), "D")
        except Exception as exc:
            raise _NoData(code) from exc
        if not h.empty:
            h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.tz_localize(None)
    if stale(h):
        raise _NoData(code)
    return h[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False, max_entries=256)
def _focus_minutes(code: str, _bucket: int) -> pd.DataFrame:
    k = _fetch_intraday(code, "1")
    if k is None or k.empty:
        raise _NoData(code)
    return k


def _focus_one(code: str, bucket: int, meta: dict) -> dict | None:
    try:
        k = _focus_minutes(code, bucket).copy()
        k["date"] = k["date"].map(taipei_time)
        k = k.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        day = k["date"].iloc[-1].normalize()
        k = k[k["date"] >= day].reset_index(drop=True)
        daily = _focus_daily(code, day.date().isoformat())
    except Exception:
        return None
    # 昨收優先用行情 API 的參考價（同一天的資料才用），日 K 收盤價只是備援
    m = meta.get(code, {})
    ref = m["ref"] if m.get("day") == day.date().isoformat() and pd.notna(m.get("ref")) else float(daily["close"].iloc[-1])
    lu, ld = m.get("limit_up", np.nan), m.get("limit_down", np.nan)
    events = intraday_signals.scan(k, ref, daily, lu, ld)
    last = float(k["close"].iloc[-1])
    return {"code": code, "name": (m.get("name") if m.get("name") not in (None, "", code) else None) or _local_stock_names().get(code) or code, "day": day.date(),
            "ref": ref, "last": last, "pct": (last - ref) / ref * 100 if ref else np.nan,
            "times": k["date"].tolist(), "closes": k["close"].astype(float).tolist(), "events": events}


def _focus_feed(watchlist: tuple, meta: dict) -> tuple[list[dict], list[str]]:
    bucket = int(time.time() // (FOCUS_REFRESH_SECONDS if is_market_open_now() else 900))
    codes = [clean_symbol(c) for c in watchlist]
    with ThreadPoolExecutor(max_workers=6) as pool:
        got = list(pool.map(lambda c: _focus_one(c, bucket, meta), codes))
    return [g for g in got if g], [c for c, g in zip(codes, got) if not g]


def _focus_spark(stock: dict, upto: pd.Timestamp) -> str:
    """當日走勢小圖：灰色虛線＝昨收，黃色虛線＝事件發生時間；平盤上紅、平盤下綠。"""
    times, closes, ref = stock["times"], stock["closes"], stock["ref"]
    w, h, pad = 132, 46, 3
    t0 = times[0].replace(hour=9, minute=0)
    span = 270 * 60
    lo, hi = min(closes + [ref]), max(closes + [ref])
    rng = (hi - lo) or 1
    xy = lambda t, v: ((t - t0).total_seconds() / span * w, pad + (hi - v) / rng * (h - 2 * pad))
    ry = xy(t0, ref)[1]
    segs, cur, cur_up = [], [], None
    for t, v in zip(times, closes):
        x, y = xy(t, v)
        up = v >= ref
        if cur_up is not None and up != cur_up:
            segs.append((cur_up, cur + [(x, y)]))
            cur = []
        cur.append((x, y))
        cur_up = up
    if cur:
        segs.append((cur_up, cur))
    lines = "".join(
        f'<polyline fill="none" stroke="{"#e04545" if up else "#1a9b5c"}" stroke-width="1.3" '
        f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>' for up, pts in segs)
    mx = xy(upto, ref)[0]
    return (f'<svg class="fx-spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<line x1="0" x2="{w}" y1="{ry:.1f}" y2="{ry:.1f}" stroke="#b9b0b0" stroke-dasharray="2 2" stroke-width=".8"/>'
            f'{lines}<line x1="{mx:.1f}" x2="{mx:.1f}" y1="0" y2="{h}" stroke="#e0a100" stroke-dasharray="2 2" stroke-width="1.2"/></svg>')


_FOCUS_CSS = """<style>
.fx-head{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;font-size:12px;color:#7d7474;margin:-6px 0 10px}
.fx-count b{color:#d23c3c}.fx-count i{font-style:normal;color:#14874f;font-weight:700}
.fx-feed{position:relative;padding-left:18px;max-height:720px;overflow-y:auto}
.fx-feed:before{content:"";position:absolute;left:5px;top:6px;bottom:6px;border-left:2px solid #e6caca}
.fx-item{position:relative;margin:0 0 12px}
.fx-time{display:inline-flex;align-items:center;gap:6px;margin:0 0 6px;padding:3px 11px 3px 8px;border-radius:7px;
  background:#8e2b2f;color:#fff;font-size:16px;font-weight:800;letter-spacing:.5px;font-variant-numeric:tabular-nums;
  box-shadow:0 3px 8px -4px rgba(142,43,47,.7)}
.fx-time svg{width:15px;height:15px;stroke:#fff;fill:none;stroke-width:2.2;stroke-linecap:round}
.fx-time:before{content:"";position:absolute;left:-18px;top:9px;width:12px;height:12px;border-radius:50%;background:#8e2b2f;box-shadow:0 0 0 3px #fbe9e9}
.fx-new{display:inline-block;margin-left:2px;font-size:10.5px;font-weight:800;color:#8e2b2f;background:#fff;border-radius:4px;padding:0 5px}
.fx-card{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr) minmax(0,1fr);gap:10px;align-items:center;
  background:#fff;border:1px solid #eadfdf;border-radius:8px;padding:10px 12px}
.fx-name{font-size:17px;font-weight:700;color:#3b3535;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fx-name span{font-size:12px;color:#8d8585;font-weight:400;margin-left:6px}
.fx-row{display:flex;gap:10px;align-items:center;margin-top:4px}
.fx-spark{width:132px;height:46px;flex:none}
.fx-px{font-size:15px;font-weight:800;line-height:1.25;font-variant-numeric:tabular-nums}
.fx-px small{display:block;font-size:12px;font-weight:700}
.fx-now{font-size:10.5px;color:#9a9090;margin-top:2px;font-weight:400}
.fx-up{color:#d23c3c}.fx-down{color:#14874f}.fx-flat{color:#6b6464}
.fx-tags,.fx-chips{display:flex;flex-wrap:wrap;gap:5px;align-content:center}
.fx-chips{border-left:1px solid #eee3e3;padding-left:10px;min-height:40px}
.fx-tags .sig,.fx-chips .sig{font-size:12px;padding:1px 9px}
.fx-chips .sig-none,.fx-chips .sig-wait{font-size:11px}
.fx-empty{color:#8a8080;font-size:13px;padding:18px 4px}
/* 稍早事件：卡片底部一整列小時間軸，最新的在左；超過 4 則收合 */
.fx-hist{grid-column:1/-1;display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;margin-top:2px;padding-top:9px;border-top:1px dashed #eadede}
.fx-hist-label{font-size:11.5px;font-weight:700;color:#8e2b2f;background:#f6eaea;border-radius:4px;padding:1px 7px;flex:none}
.fx-ev{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:#5b5252;background:#fff;border:1px solid #eee3e3;border-radius:999px;padding:1px 9px 1px 4px;white-space:nowrap}
.fx-ev time{font-size:12px;font-weight:800;color:#fff;background:#6b5d5d;border-radius:999px;padding:1px 7px;font-variant-numeric:tabular-nums}
.fx-ev.bull time{background:#c94848}.fx-ev.bear time{background:#1f8a57}
.fx-ev.bull{border-color:#f2c9c9}.fx-ev.bull b{color:#d23c3c}.fx-ev.bear{border-color:#c4e5d2}.fx-ev.bear b{color:#14874f}
.fx-ev b{font-weight:700}.fx-ev i{font-style:normal;color:#b3a9a9}
.fx-more{display:contents}.fx-more summary{list-style:none;cursor:pointer;font-size:11.5px;color:#8e2b2f;font-weight:700;padding:1px 4px}
.fx-more summary::-webkit-details-marker{display:none}.fx-more[open] summary{display:none}
@media(max-width:700px){.fx-card{grid-template-columns:1fr 1fr}.fx-card>div:first-child{grid-column:1/-1}.fx-spark{width:110px}}
</style>"""


_CLOCK_SVG = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>'


def render_watch_focus():
    """自選股盤中焦點：每 2 分鐘掃描一次自選股的 1 分 K，把剛發生的價量事件依時間列出。"""
    watchlist = tuple(st.session_state.watchlist)
    if not watchlist:
        st.info("自選股是空的，先在左側加入股票。")
        return
    meta = {}
    try:
        rows, _ = watch_rows(watchlist)
        for r in rows.to_dict("records"):
            ref = pd.to_numeric(r.get("price"), errors="coerce") - pd.to_numeric(r.get("change"), errors="coerce")
            meta[clean_symbol(r["code"])] = {"name": r.get("name"), "day": r.get("data_date"), "ref": float(ref) if pd.notna(ref) else np.nan,
                                             "limit_up": r.get("limit_up"), "limit_down": r.get("limit_down")}
    except Exception:
        pass
    with st.spinner("掃描自選股盤中事件…"):
        stocks, failed = _focus_feed(watchlist, meta)
    chip_map = chip_signals.get_nonblocking(list(watchlist))

    # 同一檔、同一分鐘的事件合成一張卡片
    cards: dict[tuple, dict] = {}
    for s in stocks:
        for e in s["events"]:
            c = cards.setdefault((e["time"], s["code"]), {"time": e["time"], "stock": s, "price": e["price"], "events": []})
            c["events"].append(e)
    cards = sorted(cards.values(), key=lambda c: c["time"], reverse=True)
    for c in cards:
        sides = {e["side"] for e in c["events"]}
        c["side"] = "both" if len(sides) > 1 else sides.pop()

    pick = st.segmented_control("方向", ["全部", "多方", "空方"], default="全部", key="focus_side",
                                label_visibility="collapsed") or "全部"
    want = {"全部": None, "多方": "bull", "空方": "bear"}[pick]
    # 每檔只留最新一張卡片：新事件出現時覆蓋同一檔較早的卡片，較早的事件收在「稍早」提示裡
    shown, earlier = [], {}
    for c in cards:
        if want is not None and c["side"] not in (want, "both"):
            continue
        code = c["stock"]["code"]
        if code in earlier:
            earlier[code].append(c)
        else:
            earlier[code] = []
            shown.append(c)

    latest = {}
    for c in cards:
        latest.setdefault(c["stock"]["code"], c)
    n_bull = sum(1 for c in latest.values() if c["side"] in ("bull", "both"))
    n_bear = sum(1 for c in latest.values() if c["side"] in ("bear", "both"))
    day = max((s["day"] for s in stocks), default=None)
    live = is_market_open_now()
    when = (f"盤中・每 {FOCUS_REFRESH_SECONDS // 60} 分鐘更新" if live
            else f"非交易時段・顯示 {day:%m/%d} 的盤中事件" if day else "目前沒有資料")
    miss = f'<span>暫時取不到：{"、".join(failed[:6])}{"…" if len(failed) > 6 else ""}</span>' if failed else ""
    st.markdown(_FOCUS_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="fx-head"><span>{when}</span><span class="fx-count">自選股 {len(stocks)} 檔・'
                f'最新偏多 <b>{n_bull}</b> 檔・偏空 <i>{n_bear}</i> 檔</span>{miss}</div>', unsafe_allow_html=True)

    now = taiwan_now().tz_localize(None)
    cls = lambda v: "fx-up" if v > 0 else "fx-down" if v < 0 else "fx-flat"
    items = []
    for c in shown[:FOCUS_MAX_CARDS]:
        s, t = c["stock"], c["time"]
        pct = (c["price"] - s["ref"]) / s["ref"] * 100
        new = '<span class="fx-new">NEW</span>' if live and (now - t).total_seconds() <= FOCUS_NEW_MINUTES * 60 else ""
        tags = "".join(f'<span class="sig sig-{e["side"]}">{_esc(e["text"])}</span>' for e in c["events"])
        chips = _flow_chips_html(s["code"]) + _signal_chips_html(chip_map.get(s["code"]), empty_text="無連續籌碼訊號", pending_text="籌碼計算中…")
        prior = earlier.get(s["code"], [])
        hist = ""
        if prior:
            def ev_html(p):
                side = p["side"] if p["side"] in ("bull", "bear") else ""
                txt = "<i>・</i>".join(f'<b>{_esc(e["text"])}</b>' for e in p["events"])
                return f'<span class="fx-ev {side}"><time>{p["time"]:%H:%M}</time>{txt}</span>'
            shown_prior = "".join(ev_html(p) for p in prior[:4])
            rest = "".join(ev_html(p) for p in prior[4:])
            more = (f'<details class="fx-more"><summary>＋展開其餘 {len(prior) - 4} 則</summary>{rest}</details>'
                    if rest else "")
            hist = f'<div class="fx-hist"><span class="fx-hist-label">稍早 {len(prior)} 則</span>{shown_prior}{more}</div>'
        items.append(
            f'<div class="fx-item"><div class="fx-time">{_CLOCK_SVG}{t:%H:%M}{new}</div><div class="fx-card">'
            f'<div><div class="fx-name">{_esc(s["name"])}<span>{s["code"]}</span></div><div class="fx-row">{_focus_spark(s, t)}'
            f'<div class="fx-px {cls(pct)}">{c["price"]:,.2f}<small>{"▲" if pct > 0 else "▼" if pct < 0 else ""}{pct:+.2f}%</small>'
            f'<div class="fx-now">現 {s["last"]:,.2f}（{s["pct"]:+.2f}%）</div></div></div></div>'
            f'<div class="fx-tags">{tags}</div><div class="fx-chips">{chips}</div>{hist}</div></div>')
    if items:
        more = len(shown) - FOCUS_MAX_CARDS
        tail = f'<div class="fx-empty">還有 {more} 檔較早有事件的股票未顯示</div>' if more > 0 else ""
        st.markdown(f'<div class="fx-feed">{"".join(items)}{tail}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="fx-empty">目前還沒有符合條件的事件。</div>', unsafe_allow_html=True)

    with st.expander("判斷依據與門檻"):
        st.markdown('<div class="sig-table-wrap"><table class="sig-table"><thead><tr><th>事件</th><th>條件</th></tr></thead><tbody>'
                    + "".join(f"<tr><td><b>{_esc(a)}</b></td><td>{_esc(b)}</td></tr>" for a, b in intraday_signals.RULES)
                    + "</tbody></table></div>", unsafe_allow_html=True)
        st.caption("價量事件用 Fugle 當日 1 分 K 計算，百分比以昨收為分母；右側籌碼標籤是前一交易日為止的連續買賣超（外資、投信、主力、融資券）與集保大戶散戶變化。"
                   "時間是事件第一次成立的那一分鐘，價格是當時價格。只是提醒「正在發生什麼」，不構成投資建議。")


_FEAT_CLASS = {"消息面": "news", "籌碼面": "chip", "價量面": "pv", "基本面": "fund", "技術面": "tech"}


@st.cache_data(ttl=900, show_spinner=False, max_entries=128)
def _cached_stock_features(code: str, flow_metric: str = "share20") -> dict:
    tickers = get_all_tickers()
    hit = tickers.loc[tickers["symbol"] == code, "name"] if "symbol" in tickers.columns else pd.Series(dtype=str)
    return stock_features.compute(
        code, get_history(code, 420), name=str(hit.iloc[0]) if len(hit) else "",
        news=_recent_news_items(days=2), earnings=_recent_earnings_items(days=5),
        chips=chip_signals.get_blocking(code, timeout=90), rotation=load_published_rotation(BASE), flow_metric=flow_metric,
    )


def render_stock_features():
    """個股符合特徵：消息面／籌碼面／價量面／基本面／技術面標籤；滑過標籤看判斷依據，下方附籌碼明細。"""
    selected = st.session_state.selected
    with st.spinner("整理個股特徵…（第一次約 10～30 秒）"):
        try:
            res = _cached_stock_features(selected, _flow_metric())
        except Exception as exc:
            st.warning(f"個股特徵暫時取不到：{type(exc).__name__}")
            return
    rows = []
    for cat, tags in res["groups"].items():
        cls = _FEAT_CLASS[cat]
        chips = "".join(f'<span class="feat feat-{cls} feat-t-{t["tone"]}{" feat-top" if t.get("top") else ""}" title="{_esc(t["why"])}">{_esc(t["text"])}</span>'
                        for t in tags) or '<span class="feat-none">無</span>'
        rows.append(f'<div class="feat-cat">{cat}</div><div class="feat-tags">{chips}</div>')
    asof = res.get("asof") or "—"
    st.markdown(f'<div class="feat-grid">{"".join(rows)}</div>'
                f'<div class="feat-legend"><span><span class="feat-dot bull"></span> 偏多</span><span><span class="feat-dot bear"></span> 偏空</span>'
                f'<span>沒有圓點＝中性</span><span>全市場資料：{asof} 收盤</span><span>滑鼠移到標籤上可看判斷依據</span></div>', unsafe_allow_html=True)
    with st.expander("判斷依據與門檻"):
        lines = [f'<tr><td>{cat}</td><td><b>{_esc(t["text"])}</b></td><td>{_esc(t["why"])}</td></tr>'
                 for cat, tags in res["groups"].items() for t in tags]
        if lines:
            st.markdown('<div class="sig-table-wrap"><table class="sig-table feat-why"><thead><tr><th>類別</th><th>特徵</th><th>依據</th></tr></thead>'
                        f'<tbody>{"".join(lines)}</tbody></table></div>', unsafe_allow_html=True)
        st.caption(f"熱門＝買賣超金額、成交金額或當沖金額為全市場股票前 {stock_features.HOT_RANK} 名（不含 ETF）。"
                   f"高當沖比 ≥ {stock_features.DAYTRADE_RATIO:.0%}、高融資＝融資使用率 ≥ {stock_features.MARGIN_USAGE:.0%}、"
                   f"外資高持股 ≥ {stock_features.FOREIGN_HOLD:.0f}%、主力大買賣超 ≥ 成交量 {stock_features.MAIN_FORCE_SHARE:.0%}、"
                   f"大戶（≥400 張）／散戶（≤50 張）持股週變化 ≥ {stock_features.HOLDER_CHANGE} 個百分點。"
                   f"週／月強勢＝5 日／20 日漲跌 ≥ {stock_features.WEEK_STRONG:.0f}%／{stock_features.MONTH_STRONG:.0f}%；"
                   f"營收高成長＝最新月營收年增 ≥ {stock_features.REVENUE_GROWTH:.0f}%；高本益比 ≥ {stock_features.PE_HIGH:.0f} 倍、"
                   f"高股息＝殖利率 ≥ {stock_features.YIELD_HIGH:.0f}%；連續配息＝最近 {stock_features.DIVIDEND_YEARS} 年每年配息；"
                   "大股本 ≥ 100 億。週線／月線／季線＝5／20／60 日均線。"
                   "資料：TWSE／TPEx 收盤行情、三大法人、當沖、融資、外資持股、本益比、月營收、公司基本資料、處置注意公告，"
                   "富邦 e-broker 主力進出、TDCC 集保、Yahoo Finance 配息、鉅亨新聞與 Fugle 法說會。不構成投資建議。")
    with st.expander("籌碼明細：外資／投信／主力／融資融券連續天數"):
        render_chip_signals(show_chips=False)


def render_chip_signals(show_chips: bool = True):
    selected = st.session_state.selected
    with st.spinner("計算籌碼訊號…（第一次約 5～20 秒）"):
        res = chip_signals.get_blocking(selected, timeout=90)
    if res is None:
        st.warning("籌碼資料暫時取不到，請稍後再試。")
        return
    if show_chips:
        st.markdown(f'<div class="sig-row">{_signal_chips_html(res)}</div>', unsafe_allow_html=True)

    trs = []
    for r in res["rows"]:
        up_word, down_word = _SIG_WORDS[r["key"]]
        v, s = r["latest"], r["streak"]
        if v is None or v != v:
            trs.append(f'<tr><td>{r["label"]}</td><td colspan="2" class="sig-muted">沒有資料</td></tr>')
            continue
        tone = "flat" if v == 0 else ("bull" if (v > 0) == _SIG_BULL_WHEN_UP[r["key"]] else "bear")
        latest = "持平" if v == 0 else f'{up_word if v > 0 else down_word} {abs(v):,.0f} {r["unit"]}'
        streak = "—" if s == 0 else f'連續 {abs(s)} 日{up_word if s > 0 else down_word}'
        trs.append(f'<tr><td>{r["label"]}</td><td class="sig-t-{tone}">{latest}</td><td class="sig-t-{tone}">{streak}</td></tr>')
    h = res.get("holder")
    if h:
        tone = "bull" if h["d_big"] > 0 > h["d_retail"] else "bear" if h["d_big"] < 0 < h["d_retail"] else "flat"
        trs.append(
            f'<tr><td>大戶／散戶</td><td class="sig-t-{tone}">大戶 {h["big_pct"]:.2f}%（{h["d_big"]:+.2f}）・散戶 {h["retail_pct"]:.2f}%（{h["d_retail"]:+.2f}）</td>'
            f'<td class="sig-muted">集保 {h["prev"][4:6]}/{h["prev"][6:]} → {h["asof"][4:6]}/{h["asof"][6:]}</td></tr>')
        if h.get("since"):
            text, why, tone = stock_features.holder_since_tag(h["since"])
            trs.append(f'<tr><td>集保後（每日估計）</td><td class="sig-t-{tone}">{text}</td><td class="sig-muted">{why}</td></tr>')
    asof = next((r["asof"] for r in res["rows"] if r.get("asof")), "")
    st.markdown(
        f'<div class="sig-table-wrap"><table class="sig-table"><thead><tr><th>項目</th><th>最新一日（{asof}）</th><th>連續天數</th></tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    st.caption("紅字＝看多、綠字＝看空；連續 3 日以上才列為上方訊號（最多往回看 10 個交易日）。融資增視為散戶看多、融券增視為看空。"
               "主力＝前 15 大買超券商合計減前 15 大賣超券商合計；大戶 ≥400 張、散戶 ≤50 張，比較最近兩週集保資料。"
               "資料：TWSE／TPEx 三大法人與融資融券、富邦 e-broker 主力進出、TDCC 集保。")


@st.cache_data(ttl=900, show_spinner=False)
def _cached_global_indicator(symbol: str) -> dict:
    return fetch_indicator(symbol)


def render_global_markets() -> None:
    """財經資訊頁：國際風向球（道瓊、那斯達克、費城半導體、布蘭特原油、VIX），每列 3 張。"""
    st.markdown('<div class="feature-section-title">國際風向球</div>', unsafe_allow_html=True)
    cols = []
    for start in range(0, len(GLOBAL_INDICATORS), 3):
        cols += st.columns(3, gap="medium")
    for col, ind in zip(cols, GLOBAL_INDICATORS):
        with col:
            try:
                d = _cached_global_indicator(ind["symbol"])
            except Exception as exc:
                st.warning(f'{ind["name"]}資料暫時取不到：{type(exc).__name__}')
                continue
            chg, pct = d.get("change"), d.get("change_pct")
            color = "#d23c3c" if (chg or 0) > 0 else "#14874f" if (chg or 0) < 0 else "#6b6464"
            chg_txt = "—" if chg is None else f"{chg:+.2f}（{pct:+.2f}%）"
            badge = ""
            if ind["symbol"] == "^VIX":
                word, tone = vix_regime(d["last"])
                badge = f'<span class="sig sig-{tone}">{word}</span>'
            asof = pd.Timestamp(d["asof"], unit="s", tz="UTC").tz_convert("Asia/Taipei").strftime("%m/%d %H:%M")
            st.markdown(
                f'<div class="gm-card"><div class="gm-head"><b>{ind["name"]}</b>{badge}</div>'
                f'<div class="gm-val">{d["last"]:,.2f}<small>{ind["unit"]}</small></div>'
                f'<div class="gm-chg" style="color:{color}">{chg_txt}</div>'
                f'<div class="gm-note">{ind["note"]}</div><div class="gm-asof">更新：{asof}（台灣時間）・Yahoo Finance</div></div>',
                unsafe_allow_html=True,
            )
            hist = d.get("history") or []
            if hist:
                xs = [pd.Timestamp(t, unit="s") for t, _ in hist]
                ys = [v for _, v in hist]
                fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2, color="#8e2b2f"),
                                           hovertemplate="%{x|%m/%d}　%{y:,.2f}<extra></extra>"))
                fig.update_layout(height=150, margin=dict(l=8, r=8, t=6, b=6), showlegend=False,
                                  xaxis=dict(showgrid=False, tickformat="%m/%d"), yaxis=dict(showgrid=True, gridcolor="#eee"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"gm_{ind['symbol']}")


def render_tdcc():
    selected = st.session_state.selected
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
    from ai_engine import add_features, train, save_model_bundle, load_model_bundle, bundle_is_usable, predict_from_bundle
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
    with c1:
        st.caption(f"盤後研究由 {AGENT_LABEL} 於交易日 15:45 自動執行並寄出。")
    with c2:
        load_latest = st.button("載入最新研究報告", use_container_width=True, key="load_latest_research")
    with c3:
        if not is_twse_trading_day(now, BASE):
            st.info(f"{market_status_text(now, BASE)}")
        elif now.time() < dt_time(15, 45):
            st.info("今日盤後研究尚未產生，以下顯示最近一次的研究報告。")
        else:
            st.success("今日盤後研究完成後會自動出現在這裡。")

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
        if is_agent_status(agent_notes.get("agent_status")):
            generated = str(agent_notes.get("generated_at", ""))[:16].replace("T", " ")
            st.success(f"{status_label(agent_notes.get('agent_status'))} 已完成自主研究｜產生時間 {generated}｜工具動作 {len(agent_notes.get('agent_actions', []))} 次")
        else:
            st.warning("本次未完成 AI 自主研究，以下可能為保守 fallback。")

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
                            st.markdown(f"**🤖 {x.get('analysis_source') or 'AI 法說會 Agent'} 已閱讀富果『法說會備忘錄』詳細文章並完成摘要**")
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

    llm = report.get("ai_analysis") or report.get("ollama_analysis") or {}
    with st.expander(f"{AGENT_LABEL}研究摘要", expanded=True):
        if llm.get("available"):
            st.markdown(llm.get("content") or "未產生摘要。")
        else:
            st.warning(llm.get("error", "AI 研究摘要尚未產生；仍可查看規則式研究結果。"))



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
.footer-small{font-size:14px!important;line-height:1.9!important;color:#c4b8b8!important;}
.footer-form-note a{font-size:13px!important;}
.footer-bottom{font-size:11px!important;padding:16px 12px 20px!important;}

/* 每一個功能節點皆用淺色背景，避免只有首頁有完整視覺語言 */
.st-key-strategy-01-watchlist,.st-key-strategy-02-selected,.st-key-strategy-03-trend,
.st-key-strategy-04-kline,.st-key-strategy-05-volume,.st-key-strategy-06-entry,
.st-key-strategy-07-tdcc,.st-key-strategy-08-trades,.st-key-strategy-09-institutions,
.st-key-strategy-01-focus,.st-key-strategy-02-chips,
.st-key-future-00-picks,.st-key-future-01-ai,.st-key-future-02-intelligence,.st-key-future-03-agent,.st-key-future-04-stock-notes{
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
.st-key-strategy-01-focus{background:#fbf1f1!important;}
.st-key-strategy-02-chips{background:#fffaf7!important;}
.st-key-future-04-stock-notes{background:#f9f5ef!important;}
.st-key-future-00-picks{background:#fbf1f1!important;}
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
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] {
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
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"]{column-gap:6px !important;}

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
[data-testid="stCheckbox"] label[data-selected="true"] > span:first-of-type{
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
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"]{
  column-gap:0 !important;gap:0 !important;align-items:stretch !important;
}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-baseweb="input"],
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-baseweb="base-input"]{
  background:#fff !important;
  border:4px solid #b3b3b3 !important;
  border-right:0 !important;
  border-radius:999px 0 0 999px !important;
  height:52px !important;
  box-shadow:none !important;
}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] input{
  background:transparent !important;border:0 !important;border-radius:0 !important;
  height:44px !important;padding:0 20px !important;color:#263746 !important;
}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-testid="stButton"] > button,
.st-key-do_stock_search_site button{
  height:52px !important;min-height:52px !important;width:100% !important;
  background:#0072c6 !important;border:0 !important;
  border-radius:0 999px 999px 0 !important;
  font-size:0 !important;color:transparent !important;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'><circle cx='20' cy='20' r='11' fill='none' stroke='white' stroke-width='5'/><line x1='28.5' y1='28.5' x2='40' y2='40' stroke='white' stroke-width='6' stroke-linecap='round'/></svg>") !important;
  background-repeat:no-repeat !important;background-position:center !important;background-size:26px 26px !important;
  box-shadow:none !important;
}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-testid="stButton"] > button:hover,
.st-key-do_stock_search_site button:hover{background-color:#005fa8 !important;}
/* 按鈕內任何文字／段落一律不顯示，只留放大鏡圖示 */
.st-key-do_stock_search_site button *,
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-testid="stButton"] > button *{
  display:none !important;font-size:0 !important;color:transparent !important;
}
.st-key-do_stock_search_site button::before,
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-testid="stButton"] > button::before{display:none !important;}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-baseweb="input"]:focus-within{
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
[data-testid="stCheckbox"] label[data-selected="true"] > span:first-child{
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
[data-testid="stSidebar"]:is([aria-expanded="true"],[data-collapsed="false"]){min-width:360px !important;max-width:360px !important;width:360px !important;}
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

/* ---------- V19：卡片整張可點、置頂分析標的、側欄收放 ---------- */
.stApp [class*="st-key-wc_"]{position:relative;cursor:pointer;}
/* 整張卡片的透明點選層（用最高優先序蓋過舊的淺藍按鈕樣式） */
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-watch_select_"]{position:absolute !important;inset:0 !important;width:100% !important;height:100% !important;margin:0 !important;z-index:5;}
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-watch_select_"] [data-testid="stButton"]{position:absolute !important;inset:0 !important;width:100% !important;height:100% !important;margin:0 !important;display:block !important;}
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-watch_select_"] button,
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-watch_select_"] [data-testid="stButton"] > button[kind]{
  position:absolute !important;inset:0 !important;width:100% !important;max-width:none !important;min-width:0 !important;
  height:100% !important;max-height:none !important;min-height:0 !important;margin:0 !important;padding:0 !important;
  border:0 !important;border-radius:12px !important;background:transparent !important;box-shadow:none !important;
  opacity:0 !important;cursor:pointer !important;transform:none !important;}
/* 刪除：純文字，放在股票欄位右上側 */
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-wcdel_"]{position:absolute !important;top:10px;right:12px;width:auto !important;margin:0 !important;z-index:6;}
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-wcdel_"] [data-testid="stButton"]{position:static !important;width:auto !important;height:auto !important;display:block !important;}
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-wcdel_"] button,
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-wcdel_"] [data-testid="stButton"] > button[kind]{
  width:auto !important;min-width:0 !important;max-width:none !important;height:auto !important;min-height:0 !important;
  padding:2px 4px !important;margin:0 !important;border:0 !important;border-radius:4px !important;
  background:transparent !important;box-shadow:none !important;transform:none !important;cursor:pointer !important;}
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-wcdel_"] button p{
  margin:0 !important;font-size:12.5px !important;font-weight:700 !important;letter-spacing:.08em !important;line-height:1.2 !important;
  color:#b3121a !important;text-decoration:none;transform:none !important;font-family:"Noto Serif TC","Microsoft JhengHei",serif !important;}
body .stApp.stApp [class*="st-key-wc_"] [class*="st-key-wcdel_"] button:hover p{text-decoration:underline;color:#8e0f15 !important;}
body .stApp.stApp [class*="st-key-wc_"][class*="_lu"] [class*="st-key-wcdel_"] button p,
body .stApp.stApp [class*="st-key-wc_"][class*="_ld"] [class*="st-key-wcdel_"] button p{color:#fff !important;}
.wc-info.wc-has-del .wc-name{padding-right:46px;}
.stApp [class*="st-key-wc_"][class*="_sel"]{border:2px solid #8e2b2f;background:#fff5f4;box-shadow:0 4px 14px rgba(142,43,47,.20);}
.stApp [class*="st-key-wc_"][class*="_lu"][class*="_sel"]{background:#7f1d1d;}
.stApp [class*="st-key-wc_"][class*="_ld"][class*="_sel"]{background:#14532d;}

/* 置頂分析標的（淺色、精簡） */
.stApp [data-testid="stSidebar"] .st-key-side_pinned{position:sticky;top:0;z-index:50;background:#fbf1f1;padding:2px 0 8px;margin:0 0 4px;box-shadow:0 6px 8px -6px rgba(142,43,47,.25);}
.pin-card{background:#fff;border:1px solid #e6d3d3;border-left:4px solid #8e2b2f;border-radius:10px;padding:8px 12px;font-family:Arial,"Microsoft JhengHei",sans-serif;}
.pin-tag{font-size:11px;font-weight:800;color:#8e2b2f;letter-spacing:1px;}
.pin-name{font-size:16px;font-weight:900;color:#1f2937;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pin-name span{font-size:12.5px;font-weight:700;color:#64748b;margin-left:2px;}
.pin-vals{display:flex;align-items:baseline;gap:10px;font-variant-numeric:tabular-nums;font-weight:800;font-size:13px;}
.pin-vals b{font-size:21px;font-weight:900;}
.pin-meta{font-size:11.5px;color:#7a6a6a;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pin-ix{display:flex;align-items:baseline;gap:8px;margin-top:5px;padding-top:5px;border-top:1px dashed #e6d3d3;font-size:12px;font-variant-numeric:tabular-nums;}
.pin-ix b{font-size:13px;color:#1f2937;}.pin-ix-label{color:#7a6a6a;font-weight:700;}

/* 側欄收放：貼在畫面左緣／側欄右緣的把手（網站主題紅＋明體，一眼可見） */
#sb-toggle{position:fixed;top:150px;left:0;z-index:1000002;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;
  width:36px;padding:14px 0;border:0;cursor:pointer;
  font-family:"Noto Serif TC","Source Han Serif TC","Songti TC","PMingLiU","MingLiU",serif;
  transition:left .2s ease,width .15s ease,background .15s ease;}
#sb-toggle .sbt-arrow{font-size:20px;font-weight:900;line-height:1;}
#sb-toggle .sbt-label{writing-mode:vertical-rl;letter-spacing:.35em;font-size:14px;font-weight:900;line-height:1;}
#sb-toggle[data-open="0"]{background:#8e2b2f;color:#fff;border-radius:0 14px 14px 0;box-shadow:3px 4px 14px rgba(142,43,47,.35);animation:sbtPulse 1.8s ease-in-out 3;}
#sb-toggle[data-open="0"]:hover{width:44px;background:#a3343a;}
#sb-toggle[data-open="1"]{background:#fff;color:#8e2b2f;border:1px solid #d9b9b9;border-left:0;border-radius:0 14px 14px 0;box-shadow:3px 3px 10px rgba(142,43,47,.15);padding:12px 0;}
#sb-toggle[data-open="1"]:hover{background:#f6e3e3;}
@keyframes sbtPulse{0%,100%{box-shadow:3px 4px 14px rgba(142,43,47,.35);}50%{box-shadow:3px 4px 22px rgba(142,43,47,.75);}}
@media(max-width:600px){#sb-toggle{top:120px;width:32px;}#sb-toggle .sbt-label{font-size:13px;}}

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
[class*="st-key-wc_"][class*="_lu"]{background:#7f1d1d;border-color:#5f1414;}
[class*="st-key-wc_"][class*="_ld"]{background:#14532d;border-color:#0b3a1f;}
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
[class*="st-key-wc_"][class*="_lu"] .wc-vals span,
[class*="st-key-wc_"][class*="_ld"] .wc-vals span{color:#fff !important;}
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
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-testid="stTextInputRootElement"]{
  background:#fff !important;background-color:#fff !important;
  border:3px solid #c3ccd5 !important;border-right:0 !important;
  border-radius:999px 0 0 999px !important;height:52px !important;
  box-shadow:none !important;overflow:hidden;
}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-testid="stTextInputRootElement"]:focus-within{
  border-color:#0072c6 !important;box-shadow:0 0 0 3px rgba(0,114,198,.14) !important;
}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-baseweb="input"],
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] [data-baseweb="base-input"]{
  background:transparent !important;background-color:transparent !important;border:0 !important;border-radius:0 !important;height:100% !important;
}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] input{
  background:transparent !important;color:#263746 !important;-webkit-text-fill-color:#263746 !important;
  caret-color:#0072c6;padding:0 22px !important;font-size:16px !important;
}
.stApp .st-key-stock_search_box div[data-testid="stHorizontalBlock"] input::placeholder{color:#8a97a3 !important;-webkit-text-fill-color:#8a97a3 !important;}

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
.stApp [data-testid="stRadio"] label p{color:#3a2f2f !important;}

/* 財經新聞篩選列：淺膚色面板 */
[class*="st-key-fin_filters"]{
  background:#fbf1f1;border:1px solid #e6d3d3;border-radius:12px;padding:10px 14px 4px !important;margin:6px 0 10px;
}
</style>''', unsafe_allow_html=True)

# 毛筆書法字體（頁首標題用；Google Fonts，需連網。沒載入時依序退回 標楷體 等系統字體）
st.markdown(r'''<style>@import url("https://fonts.googleapis.com/css2?family=Yuji+Boku&family=Yuji+Syuku&display=swap");</style>''', unsafe_allow_html=True)

# ============================================================
# UI 美化（只改外觀，不動系統架構）
#   1. 自選股側欄：產業／類別標題與股票卡片拉開距離
#   2. 技術 K 線「回到最新」按鈕：淺色系（單選／複選框維持原樣）
#   3. 往下捲動時，裝飾圖片與文字由下往上浮現（CSS + 小段 JS；分析圖表不套用）
#   4. 頁尾不再被自選股側欄蓋住  5. 頁首標題：主題紅＋毛筆字體
# ============================================================
st.markdown(r'''<style>
/* ---------- 1) 自選股側欄間距 ---------- */
.stApp [data-testid="stSidebar"] .watch-industry-header{margin:16px 0 14px !important;padding:9px 12px !important;}
.stApp [data-testid="stSidebar"] .watch-business-header{margin:12px 0 14px !important;padding:7px 12px !important;}
.stApp [data-testid="stSidebar"] .watch-industry-separator{margin:22px 0 12px !important;}
.stApp [data-testid="stSidebar"] .watch-row-divider{margin:6px 0 !important;}
.stApp [class*="st-key-wc_"]{margin:0 0 14px !important;padding:14px 14px !important;}
.stApp [class*="st-key-wc_"] [data-testid="stHorizontalBlock"]{gap:10px !important;}
.stApp .wc-info{gap:8px !important;min-height:64px !important;}
.stApp .wc-vals{gap:14px !important;}
.stApp [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.5rem !important;}

/* ---------- 2) 技術 K 線「回到最新」按鈕：淺色系 ---------- */
/* 技術 K 線「回到最新」：淡粉底＋紅字（按鈕外層有提示文字時 DOM 會多一層，所以用後代選擇器） */
.stApp [class*="st-key-technical-reset-button"] button,
.stApp [class*="st-key-technical-reset-button"] button:focus,
.stApp [class*="st-key-technical-reset-button"] button:active{
  background:#fbf1f1 !important;background-color:#fbf1f1 !important;background-image:none !important;
  color:#8e2b2f !important;-webkit-text-fill-color:#8e2b2f !important;
  border:1px solid #e6d3d3 !important;border-radius:10px !important;box-shadow:none !important;
  font-weight:800 !important;white-space:nowrap !important;
}
.stApp [class*="st-key-technical-reset-button"] button:hover{
  background:#f5e4e4 !important;background-color:#f5e4e4 !important;border-color:#d8b9b9 !important;
  color:#6f2024 !important;-webkit-text-fill-color:#6f2024 !important;
}
.stApp [class*="st-key-technical-reset-button"] button *{
  color:inherit !important;-webkit-text-fill-color:inherit !important;background:transparent !important;
}
/* ---------- 3) 捲動動態（只套在裝飾圖片與文字，分析圖表／表格不受影響） ----------
   由 _inject_scroll_reveal() 的小段 JS 偵測「元素捲進畫面」並加上 rv-in：
   文字由下往上浮起（由模糊變清晰、同區塊依序延後），裝飾圖片以遮罩由下往上展開。
   JS 沒載入時不會加上 .rv，所以內容永遠不會被藏起來。 */
.rv{opacity:0;transform:translateY(46px);filter:blur(5px);will-change:opacity,transform;
  transition:opacity .9s cubic-bezier(.2,.7,.2,1),transform .9s cubic-bezier(.2,.7,.2,1),
             filter .9s ease,clip-path 1s cubic-bezier(.2,.7,.2,1);}
.rv.rv-in{opacity:1;transform:none;filter:blur(0);}
.rv.rv-img{clip-path:inset(22% 0 0 0 round 10px);}
.rv.rv-img.rv-in{clip-path:inset(-40px round 10px);}

@keyframes siteImgSettle{from{transform:scale(1.2)}to{transform:scale(1)}}
@keyframes siteParallax{from{transform:scale(1.16) translateY(-5%)}to{transform:scale(1.16) translateY(5%)}}
@keyframes siteTitleIn{from{opacity:0;transform:translateY(28px);letter-spacing:.28em}to{opacity:1;transform:none;letter-spacing:inherit}}
.page-image-title{animation:siteTitleIn 1.1s .25s cubic-bezier(.2,.7,.2,1) both;}
/* 支援捲動驅動動畫的瀏覽器：區塊內圖片進場時由放大緩緩縮回；橫幅圖片隨捲動視差位移 */
@supports (animation-timeline: view()){
  .home-image-grid>* img,.home-image-stack>* img,.contact-band img{
    animation:siteImgSettle linear both;animation-timeline:view();animation-range:entry 0% cover 55%;
  }
  .page-image-banner>img{animation:siteParallax linear both;animation-timeline:view();animation-range:cover 0% cover 100%;}
}
/* 連結：滑過時底線由左向右畫出 */
.home-link,.contact-link{background-image:linear-gradient(currentColor,currentColor);background-repeat:no-repeat;
  background-position:0 100%;background-size:0 1.5px;transition:background-size .4s cubic-bezier(.2,.7,.2,1);}
.home-link:hover,.contact-link:hover{background-size:100% 1.5px;}

/* ---------- 4) 頁尾：自選股側欄展開時，頁尾改成縮在主內容區（不再橫跨到側欄底下被蓋住） ---------- */
[data-testid="stSidebar"]:is([aria-expanded="true"],[data-collapsed="false"]) ~ div [data-testid="stMain"] .site-footer{
  width:100% !important;margin-left:0 !important;margin-right:0 !important;border-radius:14px 14px 0 0;
}

/* ---------- 5) 頁首標題：網站主題紅＋毛筆書法字體（寫字般由左向右顯現） ---------- */
.stApp .brand-name{
  font-family:"Yuji Boku","Yuji Syuku","Zhi Mang Xing","DFKai-SB","BiauKai","標楷體","Noto Serif TC",serif !important;
  font-size:42px !important;font-weight:400 !important;letter-spacing:.14em !important;line-height:1.25 !important;
  color:#8e2b2f !important;-webkit-text-fill-color:#8e2b2f !important;
  animation:brushWrite 1.6s cubic-bezier(.3,.6,.2,1) both;
}
@keyframes brushWrite{from{clip-path:inset(-14px 100% -14px 0)}to{clip-path:inset(-14px -14px -14px -14px)}}
@media(max-width:900px){.stApp .brand-name{font-size:34px !important;}}
@media(max-width:600px){.stApp .brand-name{font-size:26px !important;letter-spacing:.08em !important;}}

@media (prefers-reduced-motion: reduce){
  .rv,.rv.rv-img{opacity:1 !important;transform:none !important;filter:none !important;clip-path:none !important;transition:none !important;}
  .page-image-banner>img,.home-image-grid>* img,.home-image-stack>* img,.contact-band img,
  .page-image-title,.stApp .brand-name{animation:none !important;}
}
</style>''', unsafe_allow_html=True)


def _inject_scroll_reveal() -> None:
    """安裝「捲動浮現」：元素捲進畫面時由下往上浮起。腳本裝在父頁面一次，Streamlit 重新執行時不會重複安裝。"""
    js = r"""
(function(){
  if (window.__siteRevealOn) return; window.__siteRevealOn = true;
  var IMG = ['.page-image-banner','.contact-band','.home-image-grid > *','.home-image-stack > *'];
  var TXT = ['.home-copy > *','.content-heading > *','.feature-section-title','.contact-info-card > *','.footer-inner > *',
    '[data-testid="stMain"] [data-testid="stHeading"]','[data-testid="stMain"] [data-testid="stCaptionContainer"]',
    '[data-testid="stMain"] [data-testid="stMarkdownContainer"] > h1','[data-testid="stMain"] [data-testid="stMarkdownContainer"] > h2',
    '[data-testid="stMain"] [data-testid="stMarkdownContainer"] > h3','[data-testid="stMain"] [data-testid="stMarkdownContainer"] > h4',
    '[data-testid="stMain"] [data-testid="stMarkdownContainer"] > p'];
  var SKIP = '[data-testid="stSidebar"],[data-testid="stPlotlyChart"],[data-testid="stDataFrame"],[data-testid="stTable"],' +
    '[data-testid="stVegaLiteChart"],[data-testid="stWidgetLabel"],label,button,summary,[role="tab"],[role="radiogroup"],' +
    '[data-baseweb="select"],[data-baseweb="tab-list"],.hero-carousel,[class*="st-key-auth_"],.site-nav';
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if (!e.isIntersecting) return;
      var el = e.target; el.classList.add('rv-in'); io.unobserve(el);
      setTimeout(function(){ el.classList.remove('rv','rv-in','rv-img'); el.style.transitionDelay=''; el.style.willChange='auto'; }, 1800);
    });
  }, {threshold: 0.08, rootMargin: '0px 0px -8% 0px'});
  function mark(sel, isImg){
    var list; try { list = document.querySelectorAll(sel); } catch(err){ return; }
    list.forEach(function(el){
      if (el.__rv || el.closest(SKIP)) return;
      var r = el.getBoundingClientRect();
      if (r.height === 0 || r.width === 0) return;           // 還沒顯示（例如別的分頁），之後再試
      el.__rv = 1;
      if (r.bottom <= 0 || r.top < window.innerHeight * 0.96) return;   // 已經在畫面內或已捲過：不動它，避免閃爍
      el.classList.add('rv'); if (isImg) el.classList.add('rv-img');
      var i = Array.prototype.indexOf.call(el.parentNode.children, el);
      el.style.transitionDelay = (Math.min(Math.max(i,0), 4) * 80) + 'ms';
      io.observe(el);
    });
  }
  function scan(){ IMG.forEach(function(q){ mark(q, true); }); TXT.forEach(function(q){ mark(q, false); }); }
  var timer = null;
  new MutationObserver(function(){ if (timer) return; timer = setTimeout(function(){ timer = null; scan(); }, 250); })
    .observe(document.body, {childList: true, subtree: true});
  scan();
})();
"""
    _components.html(
        "<script>(function(){try{var w=window.parent;if(w.__siteRevealInstalled)return;w.__siteRevealInstalled=true;"
        "var s=w.document.createElement('script');s.textContent=" + json.dumps(js) + ";w.document.head.appendChild(s);}catch(e){}})();</script>",
        height=0,
    )


def is_authenticated() -> bool:
    return bool(st.session_state.get("auth_user"))


def _restore_auth_from_cookie(cookie_manager=None) -> None:
    """從瀏覽器 Cookie 還原登入者；Streamlit Cloud 使用 CookieManager 讀取自訂 Cookie。"""
    if st.session_state.get("auth_user"):
        return
    if st.session_state.get("auth_logged_out"):
        st.session_state["auth_user"] = ""
        return

    token = None
    if cookie_manager is not None:
        try:
            token = cookie_manager.get(SESSION_COOKIE_NAME)
        except Exception:
            token = None
    # 相容可直接由 Streamlit context 讀到 Cookie 的環境。
    if not token:
        try:
            token = st.context.cookies.get(SESSION_COOKIE_NAME)
        except Exception:
            token = None

    email = verify_session_token(str(token)) if token else None
    st.session_state["auth_user"] = email or ""


def _write_auth_cookie(email: str) -> None:
    """建立簽章 token 並排隊，下一次執行時透過 CookieManager 寫入瀏覽器。"""
    token = create_session_token(email)
    st.session_state["_auth_cookie_op"] = ("set", token)
    st.session_state.pop("auth_logged_out", None)


def _delete_auth_cookie() -> None:
    """排隊刪除登入 Cookie。"""
    st.session_state["_auth_cookie_op"] = ("delete", "")
    st.session_state["auth_logged_out"] = True


def _flush_auth_cookie_op(cookie_manager=None) -> None:
    """使用 CookieManager 寫入／刪除 Cookie，避免直接操作 iframe 的 parent document 失敗。"""
    op = st.session_state.pop("_auth_cookie_op", None)
    if not op:
        return
    if cookie_manager is None:
        # 元件尚不可用時保留工作，避免丟失登入／登出指令。
        st.session_state["_auth_cookie_op"] = op
        return

    action, token = op
    try:
        if action == "set":
            cookie_manager.set(
                SESSION_COOKIE_NAME,
                token,
                key="stock_trading_auth_cookie_set",
                path="/",
                expires_at=datetime.now() + timedelta(seconds=SESSION_TTL_SECONDS),
                max_age=SESSION_TTL_SECONDS,
                same_site="lax",
            )
        else:
            cookie_manager.delete(
                SESSION_COOKIE_NAME,
                key="stock_trading_auth_cookie_delete",
            )
    except Exception:
        # 不影響 App 的其他頁面；保留動作供下一次 rerun 重試。
        st.session_state["_auth_cookie_op"] = op
        raise


def _toggle_auth_panel(mode: str) -> None:
    same_panel = (
        bool(st.session_state.get("auth_panel_open", False))
        and str(st.session_state.get("auth_panel_mode", "login")) == mode
    )
    st.session_state["auth_panel_open"] = not same_panel
    st.session_state["auth_panel_mode"] = mode


def render_auth_header_actions() -> None:
    """頁首右上角的「登入／申請帳號」：粗體紅字超連結樣式（不是按鈕）；點擊後才展開左側帳號面板。"""
    st.markdown(
        """
        <style>
        /* ---- 右上角入口：貼在頁首最右上角，登入／申請帳號（或已登入／登出）排成一列 ---- */
        [data-testid="stMainBlockContainer"], .stMainBlockContainer, .block-container {position:relative;}
        .stApp .st-key-auth_header_actions{
          position:absolute!important;top:10px;right:6px;z-index:60;
          width:auto!important;min-width:0!important;
        }
        .stApp .st-key-auth_header_actions [data-testid="stHorizontalBlock"]{
          display:flex!important;flex-direction:row!important;flex-wrap:nowrap!important;
          width:auto!important;gap:22px!important;align-items:center!important;justify-content:flex-end!important;
        }
        .stApp .st-key-auth_header_actions [data-testid="stColumn"],
        .stApp .st-key-auth_header_actions [data-testid="column"]{
          width:auto!important;min-width:0!important;flex:0 0 auto!important;
        }
        .stApp .st-key-auth_header_actions .auth-status{
          display:flex;align-items:center;height:30px;line-height:30px;margin:0;
          color:#8e2b2f;font-weight:700;font-size:15px;white-space:nowrap;
        }
        @media(max-width:700px){
          .stApp .st-key-auth_header_actions{position:static!important;width:100%!important;}
          .stApp .st-key-auth_header_actions [data-testid="stHorizontalBlock"]{justify-content:flex-end!important;gap:14px!important;}
        }
        /* ---- 入口按鈕：全站有多條 !important 的按鈕樣式，這裡用更高的權重蓋回超連結外觀 ---- */
        .stApp .st-key-auth_header_actions div[data-testid="stButton"]{
          display:flex!important;justify-content:flex-end!important;
        }
        .stApp .st-key-auth_header_actions div[data-testid="stButton"] > button[kind],
        .stApp .st-key-auth_header_actions div[data-testid="stButton"] > button[kind]:hover,
        .stApp .st-key-auth_header_actions div[data-testid="stButton"] > button[kind]:focus,
        .stApp .st-key-auth_header_actions div[data-testid="stButton"] > button[kind]:active {
          background:transparent!important;background-color:transparent!important;
          border:0!important;border-radius:0!important;box-shadow:none!important;outline:0!important;
          transform:none!important;width:auto!important;min-width:0!important;
          min-height:30px!important;height:30px!important;padding:0 6px!important;
          color:#8e2b2f!important;-webkit-text-fill-color:#8e2b2f!important;
          font-weight:800!important;white-space:nowrap!important;
          display:flex!important;align-items:center!important;justify-content:flex-end!important;
        }
        .stApp .st-key-auth_header_actions div[data-testid="stButton"] > button[kind]::before{display:none!important;}
        .stApp .st-key-auth_header_actions div[data-testid="stButton"] > button[kind] p{
          color:#8e2b2f!important;-webkit-text-fill-color:#8e2b2f!important;
          font-size:15px!important;font-weight:800!important;margin:0!important;
        }
        .stApp .st-key-auth_header_actions div[data-testid="stButton"] > button[kind]:hover p{
          color:#6f2024!important;-webkit-text-fill-color:#6f2024!important;
          text-decoration:underline!important;text-underline-offset:5px!important;
        }
        .stApp .st-key-auth_header_actions [data-testid="stCaptionContainer"] p{
          color:#8e2b2f!important;font-weight:700!important;font-size:13px!important;text-align:right;margin:0!important;
        }

        /* ---- 左側帳號面板：按鈕一律淺色系（淡粉底＋紅字），不用黑色 ---- */
        .st-key-auth_drawer {border:1px solid #e6d3d3;border-radius:12px;
          background:#fffaf9;padding:12px 12px 16px!important;margin-bottom:12px;}
        .stApp .st-key-auth_drawer div[data-testid="stButton"] > button[kind],
        .stApp .st-key-auth_drawer [data-testid="stFormSubmitButton"] > button {
          background:#f8e9e9!important;background-color:#f8e9e9!important;
          color:#8e2b2f!important;-webkit-text-fill-color:#8e2b2f!important;
          border:1px solid #e3c6c6!important;border-radius:10px!important;
          box-shadow:none!important;font-weight:800!important;opacity:1!important;
        }
        .stApp .st-key-auth_drawer div[data-testid="stButton"] > button[kind] p,
        .stApp .st-key-auth_drawer [data-testid="stFormSubmitButton"] > button p {
          color:#8e2b2f!important;-webkit-text-fill-color:#8e2b2f!important;font-weight:800!important;
        }
        .stApp .st-key-auth_drawer div[data-testid="stButton"] > button[kind]:hover,
        .stApp .st-key-auth_drawer [data-testid="stFormSubmitButton"] > button:hover {
          background:#f1d9d9!important;background-color:#f1d9d9!important;
          border-color:#cf9d9d!important;color:#6f2024!important;-webkit-text-fill-color:#6f2024!important;
        }
        .stApp .st-key-auth_drawer .st-key-auth_close_panel button,
        .stApp .st-key-auth_drawer .st-key-auth_close_panel button:hover,
        .stApp .st-key-auth_drawer .st-key-auth_close_panel button:focus,
        .stApp .st-key-auth_drawer .st-key-auth_close_panel button:active{
          background:#f8e9e9!important;background-color:#f8e9e9!important;background-image:none!important;
          color:#8e2b2f!important;-webkit-text-fill-color:#8e2b2f!important;
          border:1px solid #e3c6c6!important;border-radius:10px!important;box-shadow:none!important;
          min-width:38px!important;padding:0 !important;font-size:16px!important;font-weight:800!important;
        }
        .stApp .st-key-auth_drawer .st-key-auth_close_panel button:hover{
          background:#f1d9d9!important;background-color:#f1d9d9!important;border-color:#cf9d9d!important;
        }
        .stApp .st-key-auth_drawer .st-key-auth_close_panel button *{
          color:#8e2b2f!important;-webkit-text-fill-color:#8e2b2f!important;background:transparent!important;
        }
        .stApp .st-key-auth_drawer .st-key-auth_close_panel button::before{display:none!important;}
        @media(max-width:600px) {
          .stApp .st-key-auth_header_actions div[data-testid="stButton"] > button[kind] p{font-size:13px!important;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="auth_header_actions"):
        if is_authenticated():
            (logout_col,) = st.columns(1)
            with logout_col:
                if st.button("登出", key="auth_top_logout"):
                    _delete_auth_cookie()
                    st.session_state.pop("auth_user", None)
                    st.session_state["auth_panel_open"] = False
                    st.rerun()
        else:
            login_col, request_col = st.columns([1, 1], gap="small", vertical_alignment="center")
            with login_col:
                if st.button("登入", key="auth_top_login"):
                    _toggle_auth_panel("login")
                    st.rerun()
            with request_col:
                if st.button("申請帳號", key="auth_top_request"):
                    _toggle_auth_panel("request")
                    st.rerun()


def render_auth_panel() -> None:
    """使用者點擊頁首入口後才顯示側邊登入／申請帳號面板。"""
    if not st.session_state.get("auth_panel_open", False):
        return

    with st.sidebar:
        # 只在面板開啟時才送這段，不要改回 body:has(.st-key-auth_drawer)：單這一條就讓 Plotly 每次量文字慢 3 倍以上
        st.markdown('<style>[data-testid="stSidebarCollapseButton"]{display:none!important}</style>', unsafe_allow_html=True)
        with st.container(key="auth_drawer"):
            top_col, close_col = st.columns([4, 1], vertical_alignment="center")
            mode = str(st.session_state.get("auth_panel_mode", "login"))
            with top_col:
                st.markdown("### " + ("帳號登入" if mode == "login" else "申請帳號"))
            with close_col:
                if st.button("✕", key="auth_close_panel", help="收回帳號選單", use_container_width=True):
                    st.session_state["auth_panel_open"] = False
                    st.rerun()

            if is_authenticated():
                st.success("已登入：" + str(st.session_state.get("auth_user", "")))
                if st.button("登出", key="auth_panel_logout", use_container_width=True):
                    _delete_auth_cookie()
                    st.session_state.pop("auth_user", None)
                    st.session_state["auth_panel_open"] = False
                    st.rerun()
                return

            if mode == "login":
                with st.form("account_login_form"):
                    email = st.text_input("電子郵件", key="auth_login_email").strip().lower()
                    password = st.text_input("密碼", type="password", key="auth_login_password")
                    login_submit = st.form_submit_button("登入", use_container_width=True)
                if login_submit:
                    if authenticate(email, password):
                        try:
                            _write_auth_cookie(email)
                        except Exception as exc:
                            st.error("建立登入工作階段失敗：" + type(exc).__name__)
                        else:
                            st.session_state["auth_user"] = email
                            st.session_state["auth_panel_open"] = False
                            st.rerun()
                    else:
                        st.error("帳號或密碼錯誤，或尚未建立此帳號。")
                if st.button("還沒有帳號？申請帳號", key="auth_switch_to_request", use_container_width=True):
                    st.session_state["auth_panel_mode"] = "request"
                    st.rerun()
            else:
                st.caption("輸入欲註冊的電子郵件；管理員將透過 Email 審核，核准後系統會寄送初始密碼。")
                with st.form("account_request_form"):
                    request_email = st.text_input("欲註冊的電子郵件", key="auth_request_email")
                    request_submit = st.form_submit_button("寄送帳號申請", use_container_width=True)
                if request_submit:
                    result = request_account(request_email, BASE)
                    if result.get("sent"):
                        st.success("帳號申請已送出，管理員將收到同意／拒絕審核連結。")
                        if result.get("confirmation_sent"):
                            st.info("確認信已寄到 " + str(request_email).strip() + "，請留意收件匣及垃圾郵件匣。")
                        else:
                            st.warning("管理員已收到申請通知，但寄給你的確認信未成功送出。請確認電子郵件地址，或直接聯絡管理員。")
                    else:
                        st.error(result.get("error", "帳號申請寄送失敗。"))
                if st.button("返回登入", key="auth_switch_to_login", use_container_width=True):
                    st.session_state["auth_panel_mode"] = "login"
                    st.rerun()


def render_account_review_action() -> None:
    """處理管理員從審核 Email 開啟的連結；GET 僅預覽，按下確認才會提交審核。"""
    decision = str(st.query_params.get("account_review", "") or "").strip().lower()
    token = str(st.query_params.get("token", "") or "").strip()
    if decision not in {"approve", "reject"}:
        return
    st.title("網站帳號申請審核")
    st.caption("AI 台股即時互動式分析系統")
    if not token:
        st.error("審核連結缺少驗證資訊，請回到原始通知信重新開啟。")
        st.stop()

    import hashlib as _hashlib
    result_key = "account_review_result_" + _hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    cached = st.session_state.get(result_key)
    if cached:
        if cached.get("decision") == "approve":
            st.success("此帳號申請已完成核准處理。")
            st.write("申請者：" + str(cached.get("email", "")))
            st.write("申請者密碼郵件：" + ("已寄出" if cached.get("credential_sent") else "寄送失敗"))
            st.write("登入憑證：" + ("已自動儲存至 Supabase，不需手動更新 AUTH_USERS_JSON。" if cached.get("database_credential_saved") else "資料庫儲存狀態需再確認。"))
            if not cached.get("credential_sent"):
                st.warning("密碼郵件未成功送出。請使用下方一次性處理資訊聯絡申請者，勿公開分享此頁面。")
                if cached.get("fallback_password"):
                    st.code("初始密碼：" + str(cached["fallback_password"]))
        else:
            st.success("此帳號申請已拒絕處理。")
            st.write("申請者：" + str(cached.get("email", "")))
            st.write("拒絕通知郵件：" + ("已寄出" if cached.get("applicant_sent") else "寄送失敗"))
            st.write("管理員結案通知：" + ("已寄出" if cached.get("admin_notice_sent") else "寄送失敗"))
        st.stop()

    try:
        import account_provisioning
        review_state = account_provisioning.get_review_state_for_token(token, decision)
    except Exception as exc:
        st.error("無法連線帳號審核資料庫：" + type(exc).__name__)
        st.stop()

    if not review_state:
        st.info(
            "此審核連結已失效或已逾期，不能再執行同意或拒絕。"
            "若這筆申請先前已經核准或拒絕，原審核結果不會被改變。"
        )
        st.caption("原始 Email 是已寄出的歷史郵件，網站無法遠端刪除或改寫；請在信箱自行封存或刪除舊信。")
        st.stop()

    review_status = str(review_state.get("status", "")).strip().lower()
    if review_status != "pending":
        status_labels = {
            "approved": "此帳號申請已核准，原信件中的同意／拒絕連結均已失效。",
            "rejected": "此帳號申請已拒絕，原信件中的同意／拒絕連結均已失效。",
            "notification_failed": "此帳號申請未成功送出管理員通知，原審核連結已失效。",
        }
        message = status_labels.get(review_status, "此帳號申請已結案或不再接受審核，原連結已失效。")
        st.info(message)
        st.write("申請者：" + str(review_state.get("email", "")))
        if review_state.get("decided_at"):
            st.write("處理時間：" + str(review_state.get("decided_at", "")))
        st.caption("原始 Email 是已寄出的歷史郵件，無法從收件匣遠端刪除；但此連結不會再執行任何審核動作。")
        st.stop()

    request = account_provisioning.get_request_for_token(token, decision)
    if not request:
        st.error("此審核連結已失效、已逾期，或已被使用。請勿重複操作。")
        st.stop()

    action_label = "同意建立帳號" if decision == "approve" else "拒絕帳號申請"
    st.subheader(action_label)
    st.write("申請者電子郵件：" + str(request.get("email", "")))
    st.write("申請時間：" + str(request.get("created_at", "")))
    if decision == "approve":
        st.info("確認後系統會自動產生初始密碼並將密碼雜湊儲存至 Supabase。帳號於核准後 5 分鐘啟用，無須管理員手動更新 AUTH_USERS_JSON。")
    else:
        st.warning("確認後系統會將申請標記為拒絕，並寄送審核結果給申請者。此操作無法撤回。")
    with st.form("account_review_confirm_form"):
        confirmed = st.checkbox("我已確認申請者與審核動作正確。", key="account_review_confirmed")
        # Streamlit 表單內的元件不會在變更時單獨 rerun，因此不能用
        # disabled=not confirmed 控制同一個表單的提交按鈕，否則按鈕可能永遠停用。
        submit = st.form_submit_button(action_label, type="primary", use_container_width=True)
    if submit and not confirmed:
        st.warning("請先勾選「我已確認申請者與審核動作正確」，再提交審核。")
        st.stop()
    if submit:
        try:
            result = account_provisioning.decide_account_request(token, decision)
            from email_agent import EmailAgent
            agent = EmailAgent(Path(__file__).resolve().parent)
            if decision == "approve":
                try:
                    credential_mail = agent.send_account_approved_credentials(
                        email=result["email"],
                        password=result["password"],
                        activate_at=result["activate_at"],
                    )
                except Exception as mail_exc:
                    credential_mail = {"sent": False, "error": type(mail_exc).__name__}
                cached_result = {
                    "decision": decision,
                    "email": result["email"],
                    "credential_sent": bool(credential_mail.get("sent")),
                    "database_credential_saved": bool(result.get("password_hash")),
                }
                if not credential_mail.get("sent"):
                    cached_result["fallback_password"] = result["password"]
                st.session_state[result_key] = cached_result
            else:
                try:
                    applicant_mail = agent.send_account_rejected(email=result["email"])
                except Exception as mail_exc:
                    applicant_mail = {"sent": False, "error": type(mail_exc).__name__}
                try:
                    admin_notice = agent.send_account_review_decision_notice(
                        email=result["email"],
                        decision="rejected",
                        decided_at=str(result.get("decided_at", "")),
                    )
                except Exception as mail_exc:
                    admin_notice = {"sent": False, "error": type(mail_exc).__name__}
                st.session_state[result_key] = {
                    "decision": decision,
                    "email": result["email"],
                    "applicant_sent": bool(applicant_mail.get("sent")),
                    "admin_notice_sent": bool(admin_notice.get("sent")),
                }
            st.rerun()
        except Exception as exc:
            st.error("審核處理失敗：" + str(exc))
    st.stop()


def sync_sidebar_visibility(section: str) -> None:
    """側欄可自由收放。
    - 只有「頁面／登入面板狀態改變」時才自動展開或收起一次；自動刷新與一般互動不會再把使用者收起的側欄強制打開。
    - 頁首（含原生展開鈕）被網站樣式隱藏，因此另外放一顆固定的「收放」按鈕，點一下即可拉出／收回。
    - 收放邏輯以 <script> 安裝在主頁面本身：components.html 的 iframe 每次 rerun／自動刷新都會被重建，
      放在 iframe 裡的計時器會跟著消失，按鈕就會卡住。
    """
    auth_open = bool(st.session_state.get("auth_panel_open", False))
    should_open = (section == "strategy") or auth_open
    state_key = (section, auth_open)
    apply_auto = st.session_state.get("_sb_last_state") != state_key
    st.session_state["_sb_last_state"] = state_key
    js_open = "true" if should_open else "false"
    js_apply = "true" if apply_auto else "false"
    _components.html(
        f"""
        <script>
        (function() {{
          let w, d;
          try {{ w = window.parent; d = w.document; }} catch (e) {{ return; }}
          if (!w.__sbToggleV3) {{
            const s = d.createElement('script');
            s.id = 'sb-toggle-runtime';
            s.textContent = `
              (function() {{
                window.__sbToggleV3 = true;
                const d = document;
                const sidebar = () => d.querySelector('[data-testid="stSidebar"]');
                // 舊版 Streamlit 用 aria-expanded，新版（1.5x 以後）改用 data-collapsed
                const isOpen = () => {{ const sb = sidebar(); return !!sb && (sb.getAttribute('aria-expanded') === 'true' || sb.getAttribute('data-collapsed') === 'false'); }};
                function nativeButton() {{
                  const sels = isOpen()
                    ? ['[data-testid="stSidebarCollapseButton"]']
                    : ['[data-testid="stExpandSidebarButton"]', '[data-testid="stSidebarCollapsedControl"]', '[data-testid="stSidebarCollapseButton"]'];
                  for (const sel of sels) {{
                    const node = d.querySelector(sel);
                    if (node) return node.tagName === 'BUTTON' ? node : (node.querySelector('button') || node);
                  }}
                  return null;
                }}
                function toggle() {{ const b = nativeButton(); if (b) b.click(); setTimeout(refresh, 60); }}
                function button() {{
                  let b = d.getElementById('sb-toggle');
                  if (!b) {{
                    b = d.createElement('button');
                    b.id = 'sb-toggle';
                    b.type = 'button';
                    b.addEventListener('click', toggle);
                    d.body.appendChild(b);
                  }}
                  return b;
                }}
                function refresh() {{
                  const sb = sidebar();
                  const b = button();
                  if (!sb) {{ b.style.display = 'none'; return; }}
                  const open = isOpen();
                  b.style.display = 'flex';
                  const flag = open ? '1' : '0';
                  if (b.getAttribute('data-open') !== flag) {{
                    b.setAttribute('data-open', flag);
                    b.innerHTML = open
                      ? '<span class="sbt-arrow">\\u2039</span><span class="sbt-label">收合</span>'
                      : '<span class="sbt-arrow">\\u203a</span><span class="sbt-label">自選股</span>';
                    b.title = open ? '收回自選股側欄' : '拉出自選股側欄';
                  }}
                  b.style.left = open ? Math.round(sb.getBoundingClientRect().right) + 'px' : '0px';
                }}
                // 頁面切換或登入面板開關時，自動展開／收起一次；側欄還沒渲染出來就稍後再試（最多約 3 秒）。
                window.__sbApply = function(shouldOpen) {{
                  let tries = 0;
                  (function attempt() {{
                    const b = sidebar() && nativeButton();
                    if (b) {{ if (isOpen() !== shouldOpen) toggle(); return; }}
                    if (++tries < 15) setTimeout(attempt, 200);
                  }})();
                }};
                setInterval(refresh, 300);
                refresh();
              }})();
            `;
            d.head.appendChild(s);
          }}
          if ({js_apply} && w.__sbApply) w.__sbApply({js_open});
        }})();
        </script>
        """,
        height=0,
    )

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
      const w = window.parent;
      const d = w.document;
      d.documentElement.setAttribute('lang', 'zh-TW');
      d.documentElement.setAttribute('translate', 'no');
      if (!d.querySelector('meta[name="google"]')) {
        const m = d.createElement('meta'); m.name = 'google'; m.content = 'notranslate'; d.head.appendChild(m);
      }
      d.body.classList.add('notranslate');

      // 將首頁卡片與頁尾的 section 連結導向同一組 Streamlit 原生導覽按鈕，
      // 避免 href="?section=..." 讓瀏覽器整頁載入並重建 session。
      if (!w.__stockTradingSectionNavInstalled) {
        w.__stockTradingSectionNavInstalled = true;
        d.addEventListener("click", function(event) {
          const anchor = event.target && event.target.closest
            ? event.target.closest('a[href*="section="]') : null;
          if (!anchor) return;
          let target;
          try { target = new URL(anchor.getAttribute("href"), w.location.href); }
          catch (err) { return; }
          const section = target.searchParams.get("section");
          const allowed = ["home", "strategy", "trend", "future", "finance", "rotation", "subscribe"];
          if (!allowed.includes(section)) return;
          const button = d.querySelector(".st-key-nav_section_" + section + " button");
          if (!button) return;
          event.preventDefault();
          event.stopPropagation();
          // 下拉選單的功能項目：記下要捲到的區塊標題；同一頁直接捲，換頁則等新頁面畫完再捲
          const jumpText = anchor.getAttribute("data-target");
          const active = d.querySelector(".site-nav-link.active");
          const samePage = !!active && (active.getAttribute("href") || "").indexOf("section=" + section) >= 0;
          if (!samePage) w.__uxNavPending(section, active);
          if (!samePage || !jumpText) button.click();      // 點目前所在頁的主題本身：照舊重新整理該頁
          if (jumpText) w.__uxJumpTo(jumpText, anchor.getAttribute("data-tab"), samePage ? null : section);
          // 點完收起下拉；同時移開焦點，否則 :focus-within 會讓選單重新打開、擋住剛捲到的區塊
          if (anchor.blur) anchor.blur();
          const item = anchor.closest(".nav-item");
          if (item) { item.classList.add("nav-closed"); setTimeout(function() { item.classList.remove("nav-closed"); }, 400); }
        }, true);

        // 一點主題就先換畫面：導覽列標到新主題、舊頁面藏起來，不必等伺服器；新頁面有內容（或這輪跑完）才收掉提示
        let pendingTimer = null;
        const clearPending = () => {
          clearInterval(pendingTimer); pendingTimer = null;
          d.body.classList.remove("ux-nav-pending");
        };
        w.__uxNavPending = function(section, active) {
          const link = d.querySelector('.site-nav-link[href*="section=' + section + '"]');
          if (active) active.classList.remove("active");
          if (link) link.classList.add("active");
          let box = d.getElementById("ux-nav-loading");
          if (!box) {
            box = d.createElement("div"); box.id = "ux-nav-loading"; box.setAttribute("role", "status");
            box.innerHTML = '<div class="ux-nav-bars"><i></i><i></i><i></i></div><div class="ux-nav-text"></div>';
            d.body.appendChild(box);
          }
          box.querySelector(".ux-nav-text").textContent = "正在切換到「" + (link ? link.textContent.trim() : "") + "」";
          d.body.classList.add("ux-nav-pending");
          clearInterval(pendingTimer);
          const started = Date.now();
          let sawRunning = false;
          pendingTimer = setInterval(function() {
            const app = d.querySelector(".stApp");
            const running = !!app && app.getAttribute("data-test-script-state") === "running";
            sawRunning = sawRunning || running;
            const page = d.querySelector(".st-key-page_" + section);
            const ready = page && page.scrollHeight > 160;
            if (ready || (page && sawRunning && !running) || Date.now() - started > 45000) clearPending();
          }, 100);
        };

        // 依標題文字找頁面上的區塊（排除導覽列、頁尾、頁首大標與橫幅），捲到它並閃一下提示位置
        const TITLES = '.feature-section-title,.finance-card-title,.form-title,.fin-digest h3,.home-copy h2,.home-flow h2,' +
          '.contact-content h2,.contact-info-card h2,[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3';
        const findTitle = (text) => {
          for (const el of d.querySelectorAll(TITLES)) {
            if (el.closest(".site-nav,.site-footer,.content-heading,.page-image-banner")) continue;
            if ((el.textContent || "").indexOf(text) >= 0 && el.getBoundingClientRect().height > 0) return el;
          }
          return null;
        };
        // 實際在捲動的容器：通常是 stMain，若它沒有捲軸就用整頁
        const pageScroller = () => {
          const m = d.querySelector('[data-testid="stMain"]');
          return m && m.scrollHeight > m.clientHeight + 1 ? m : d.scrollingElement;
        };
        const toast = (msg) => {
          let t = d.getElementById("ux-toast");
          if (!t) { t = d.createElement("div"); t.id = "ux-toast"; t.setAttribute("role", "status"); d.body.appendChild(t); }
          t.textContent = msg; t.classList.add("show");
          clearTimeout(t.__h); t.__h = setTimeout(function() { t.classList.remove("show"); }, 2600);
        };
        let jumpTimer = null, settleTimers = [];
        const stopJump = () => { clearInterval(jumpTimer); jumpTimer = null; };
        const stopSettle = () => { settleTimers.forEach(clearTimeout); settleTimers = []; };
        // 使用者自己捲動或按鍵，就放棄等待中的跳轉與後續對齊，避免之後突然被拉走
        ["wheel", "touchmove", "keydown"].forEach(function(type) {
          d.addEventListener(type, function() { stopJump(); stopSettle(); }, {passive: true});
        });
        w.__uxJumpTo = function(text, tab, targetSection) {
          stopJump(); stopSettle();
          const started = Date.now();
          let idle = 0;
          const go = (el) => {
            stopJump();
            const main = pageScroller();
            const top = main.scrollTop + el.getBoundingClientRect().top - (main === d.scrollingElement ? 0 : main.getBoundingClientRect().top) - 70;
            const start = main.scrollTop;
            main.scrollTo({top: top, behavior: "smooth"});
            setTimeout(function() { if (Math.abs(main.scrollTop - start) < 2 && Math.abs(top - start) > 2) main.scrollTo({top: top}); }, 700);
            // 上方的圖表在捲動後才畫完、把區塊往下推：之後幾秒內位置偏掉就再對齊（使用者自己捲動則停止）
            [900, 1800, 3000, 4500, 6500].forEach(function(ms) {
              settleTimers.push(setTimeout(function() {
                const off = el.getBoundingClientRect().top - (main === d.scrollingElement ? 0 : main.getBoundingClientRect().top) - 70;
                if (Math.abs(off) > 12) main.scrollTo({top: main.scrollTop + off});
              }, ms));
            });
            const box = el.closest('[class*="st-key-"]:not([class*="st-key-page_"])') || el;
            box.classList.remove("ux-flash"); void box.offsetWidth; box.classList.add("ux-flash");
            if (tab) setTimeout(function() {
              const t = Array.prototype.find.call(d.querySelectorAll('[role="tab"]'), function(b) { return (b.textContent || "").indexOf(tab) >= 0; });
              if (t) t.click();
            }, 250);
          };
          jumpTimer = setInterval(function() {
            const app = d.querySelector(".stApp");
            const running = app && app.getAttribute("data-test-script-state") === "running";
            if (running) { idle = 0; return; }
            idle += 1;
            if (targetSection) {
              // 換頁：目標頁的容器要先出現，再連續兩次閒置，畫面才算穩定，避免捲到舊頁面的同名標題
              // （導覽列的目前頁一點就先換了，不能用來判斷）
              if (!d.querySelector(".st-key-page_" + targetSection)) { if (Date.now() - started > 30000) stopJump(); return; }
              if (idle < 2) return;
            }
            const el = findTitle(text);
            if (el) return go(el);
            // 頁面已畫完仍找不到（例如該分類目前 0 檔、或需要登入）：停在頁首並提示，不在背景一直等
            if (idle >= 6) { stopJump(); toast("「" + text + "」目前沒有內容可顯示"); }
          }, 250);
        };
      }
      // 回到頂端按鈕、導覽列陰影、操作策略區塊跳轉（只安裝一次）
      if (!w.__siteUxV1) {
        w.__siteUxV1 = true;
        let scroller = null;
        const mainScroller = () => scroller || d.querySelector('[data-testid="stMain"]') || d.scrollingElement;
        // 平滑捲動；瀏覽器沒有動（例如分頁在背景、不畫影格）就直接跳過去
        const smoothTo = (el, top) => {
          const start = el.scrollTop;
          el.scrollTo({top: top, behavior: "smooth"});
          setTimeout(function() { if (Math.abs(el.scrollTop - start) < 2 && Math.abs(top - start) > 2) el.scrollTo({top: top}); }, 700);
        };
        const topBtn = d.createElement("button");
        topBtn.id = "ux-top"; topBtn.type = "button"; topBtn.title = "回到頂端"; topBtn.textContent = "↑";
        topBtn.addEventListener("click", function() {
          smoothTo(mainScroller(), 0);
        });
        d.body.appendChild(topBtn);
        d.addEventListener("scroll", function(event) {
          const el = event.target === d ? d.scrollingElement : event.target;
          if (!el || !el.closest || el.closest('[data-testid="stSidebar"]')) return;
          if (el.scrollHeight <= el.clientHeight + 50) return;
          scroller = el;
          topBtn.classList.toggle("show", el.scrollTop > 600);
          d.body.classList.toggle("ux-scrolled", el.scrollTop > 80);
        }, true);
        d.addEventListener("click", function(event) {
          const chip = event.target && event.target.closest ? event.target.closest("[data-jump]") : null;
          if (!chip) return;
          const box = d.querySelector(".st-key-" + chip.getAttribute("data-jump"));
          if (!box) return;
          event.preventDefault();
          const el = mainScroller();
          smoothTo(el, el.scrollTop + box.getBoundingClientRect().top - el.getBoundingClientRect().top - 78);
          box.classList.remove("ux-flash"); void box.offsetWidth; box.classList.add("ux-flash");
        }, true);
      }
      // 往下捲後導覽列自動收起；滑鼠移到頁首附近（或點頂端小標籤、往上捲）才彈出，離開後再收回
      if (!w.__siteNavPeekV1) {
        w.__siteNavPeekV1 = true;
        const HIDE_AFTER = 160, PEEK_ZONE = 70;
        let lastY = 0, scrollY = 0, hovering = false, hideTimer = null;
        const body = d.body;
        const tab = d.createElement("button");
        tab.id = "ux-navtab"; tab.type = "button"; tab.innerHTML = "&#9776;&nbsp;選單"; tab.title = "顯示導覽列";
        body.appendChild(tab);
        const show = () => { clearTimeout(hideTimer); body.classList.remove("ux-navhide"); };
        const hideSoon = (ms) => {
          clearTimeout(hideTimer);
          hideTimer = setTimeout(function() { if (!hovering && scrollY > HIDE_AFTER) body.classList.add("ux-navhide"); }, ms);
        };
        tab.addEventListener("click", show);
        tab.addEventListener("mouseenter", show);
        d.addEventListener("scroll", function(event) {
          const el = event.target === d ? d.scrollingElement : event.target;
          if (!el || !el.closest || el.closest('[data-testid="stSidebar"]')) return;
          if (el.scrollHeight <= el.clientHeight + 50) return;
          scrollY = el.scrollTop;
          if (scrollY <= HIDE_AFTER) show();
          else if (scrollY < lastY - 6) show();          // 往上捲（手機沒有滑鼠時也能叫出來）
          else if (scrollY > lastY + 6 && !hovering) body.classList.add("ux-navhide");
          lastY = scrollY;
        }, true);
        d.addEventListener("mousemove", function(event) {
          const nav = d.querySelector(".st-key-site_nav_sticky");
          const navBottom = nav ? nav.getBoundingClientRect().bottom : 0;
          const inMenu = event.target && event.target.closest && event.target.closest(".st-key-site_nav_sticky");
          const near = inMenu || event.clientY <= PEEK_ZONE || (!body.classList.contains("ux-navhide") && event.clientY <= navBottom + 24);
          if (near && !hovering) { hovering = true; show(); }
          else if (!near && hovering) { hovering = false; hideSoon(450); }
        }, {passive: true});
        d.addEventListener("mouseleave", function() { if (hovering) { hovering = false; hideSoon(450); } });
      }
    } catch (e) {}
    </script>""",
    height=0,
)
render_account_review_action()
auth_cookie_manager = stx.CookieManager(key="stock_trading_auth_cookie_manager")
_restore_auth_from_cookie(auth_cookie_manager)
_flush_auth_cookie_op(auth_cookie_manager)
_inject_scroll_reveal()
_inject_ux_layer()
render_header_and_search()
render_auth_header_actions()
current_section = render_site_navigation()
render_auth_panel()
sync_sidebar_visibility(current_section)

if current_section != "strategy":
    st.session_state["_strategy_view"] = None  # 回到操作策略時重新以兩段式顯示
def _render_page(section: str) -> None:
    section_ui.reset()
    st.markdown(section_ui.CSS, unsafe_allow_html=True)
    try:
        _render_page_body(section)
    finally:
        section_ui.section_end()


def _render_page_body(section: str) -> None:
    if section == "home":
        render_home_workspace()
    elif section == "strategy":
        render_strategy_workspace()
    elif section == "trend":
        render_trend_patterns_workspace(BASE)
    elif section == "future":
        if is_authenticated():
            render_future_workspace()
        else:
            st.markdown('<div class="content-heading"><h1>未來分析</h1><p>此功能僅限登入使用者。</p></div>', unsafe_allow_html=True)
            st.info("請點擊頁面右上角的「登入」或「申請帳號」，開啟左側選單。")
    elif section == "finance":
        render_finance_workspace()
        with _sec_box("fin-celebrity"):
            render_celebrity_holdings()
    elif section == "rotation":
        render_rotation_workspace()
    else:
        render_subscription_workspace()


# 頁面內容包一層有 key 的容器：換頁時容器重建、播放淡入動畫；同一頁內重跑不會重播。
try:
    # 同一位置的新容器會繼承舊容器的子元素，舊主題要等新內容一格格蓋掉、整輪跑完才消失，所以每個主題各有一個位置。
    # 換頁時先在舊主題的位置寫入 empty 清掉它。不能在同一位置先 empty 再放容器：送出前會被合併掉，等於沒清。
    with st.container(key="site_pages"):
        _page_slots = {key: st.container() for key, _label, _hint in SITE_NAV}
    _previous_section = st.session_state.get("_rendered_section")
    if _previous_section in _page_slots and _previous_section != current_section:
        _page_slots[_previous_section].empty()
    st.session_state["_rendered_section"] = current_section
    with _page_slots[current_section].container(key=f"page_{current_section}"):
        _render_page(current_section)

    # 盤中自動更新改由「操作策略」各即時區塊的 fragment（run_every）負責，不再整頁重跑。

    # Footer 必須是整個網頁最後一個可見內容。
    _render_footer()
finally:
    _finish_boot_loader()   # 頁面出錯也要收掉載入動畫，錯誤訊息才看得到