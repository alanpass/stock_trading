# -*- coding: utf-8 -*-
"""台股產業輪動資料引擎。

每日排程會更新代表股日 K、官方加權指數歷史、產業報酬、相對強弱與資金聚焦代理，
輸出精簡 JSON 給 Streamlit Cloud 顯示。資料缺少當日基準時，不覆蓋上一份成功資料。
"""
from __future__ import annotations

import json
import os
import time
import base64
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import pandas as pd
import requests

from stock_api import FugleClient, clean_symbol
from market_calendar import is_twse_trading_day

TAIPEI = ZoneInfo("Asia/Taipei")
GITHUB_REPO_DEFAULT = "alanpass/stock_trading"
PUBLIC_REPO_PATH = "output/research_reports/industry_rotation_public.json"
TAIEX_MONTHLY_URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
USER_AGENT = "Mozilla/5.0 TaiwanStockIndustryRotation/1.0"

# 主題按產品／供應鏈分類，與證交所產業別不同；部分成分股會出現在多個主題。
# 為避免全市場 API 呼叫過量，初版聚焦於流動性與主題辨識度較高的代表股。
ROTATION_THEMES: dict[str, list[str]] = {
    "晶圓代工": ["2330", "2303", "6770"],
    "ASIC／矽智財": ["3443", "3661", "3035", "3529", "6533", "2454", "5269", "6531"],
    "BMC／伺服器管理晶片": ["5274"],
    "高速傳輸 IC": ["5269", "4966"],
    "先進封裝設備": ["3583", "3131", "6187", "6640", "6223", "6510"],
    "封裝測試": ["3264", "6257", "2449", "3711", "6239", "3374"],
    "測試介面／設備": ["6510", "6223", "6515", "3583", "6640", "3653"],
    "檢測分析": ["3289", "6830", "3587"],
    "記憶體／儲存": ["2408", "2344", "8299", "3260", "6239", "2342", "4967", "2337", "5351", "2451"],
    "ABF／IC 載板": ["3037", "3189", "8046"],
    "CCL／銅箔基板": ["6213", "6274", "2383", "1303"],
    "玻纖布／銅箔": ["5475", "5340", "1303", "1802", "1815"],
    "PCB／伺服器板／HDI": ["2368", "3044", "2313", "3037", "8046", "6274", "4908", "5469"],
    "散熱／液冷": ["3653", "8996", "3483", "6230", "3324", "2421", "3338", "3017", "6805"],
    "電源／BBU": ["2308", "6412", "6121", "6781", "3323", "2301", "3015", "6282"],
    "重電／電網": ["1519", "1513", "1503", "1504", "1609", "1505"],
    "滑軌": ["2059", "6805"],
    "機殼／機櫃": ["3693", "3013", "8210", "6117"],
    "高速連接器／線材": ["3533", "6197", "3665", "3023", "3526", "3217", "2392", "6290", "6205", "3605"],
    "被動元件": ["8042", "3624", "3026", "6449", "2375", "3357", "2472", "2478", "3090", "2327", "2492", "6173"],
    "組裝代工／ODM": ["3231", "6669", "4938", "3706", "2376", "2356", "2317", "2324", "2382"],
    "網通／交換器": ["2345", "3596", "6285", "4906", "5388", "3380"],
    "矽光子／光通訊": ["3363", "3081", "3234", "4908", "4979", "6442", "3450", "4977"],
    "磊晶／化合物半導體": ["8086", "2455", "3105", "4991", "3707", "3016"],
    "低軌衛星": ["7717", "4908", "6285", "2314", "3491"],
    "機器人／自動化": ["6215", "2049", "1590", "4583", "4576", "2464", "2231"],
    "廠務工程": ["6691", "6196", "6139", "6613", "5536", "2404"],
}
ROTATION_DISPLAY_NAMES: dict[str, str] = {
    "晶圓代工": "晶圓代工(台積電)",
    "ASIC／矽智財": "ASIC/矽智財",
    "BMC／伺服器管理晶片": "BMC(信驊)",
    "高速傳輸 IC": "高速傳輸IC",
    "先進封裝設備": "先進封裝設備",
    "封裝測試": "封裝測試",
    "測試介面／設備": "測試介面/設備",
    "檢測分析": "檢測分析",
    "記憶體／儲存": "記憶體",
    "ABF／IC 載板": "ABF載板",
    "CCL／銅箔基板": "CCL銅箔基板",
    "玻纖布／銅箔": "玻纖布/銅箔",
    "PCB／伺服器板／HDI": "PCB(伺服器板/HDI)",
    "散熱／液冷": "散熱",
    "電源／BBU": "電源/BBU",
    "重電／電網": "重電/電網",
    "滑軌": "滑軌",
    "機殼／機櫃": "機殼",
    "高速連接器／線材": "高速連接器/線材",
    "被動元件": "被動元件",
    "組裝代工／ODM": "組裝代工(ODM)",
    "網通／交換器": "網通設備",
    "矽光子／光通訊": "矽光子/光通訊",
    "磊晶／化合物半導體": "磊晶/化合物半導體",
    "低軌衛星": "低軌衛星",
    "機器人／自動化": "機器人",
    "廠務工程": "廠務工程",
}
ROTATION_DEFAULT_THEMES = [
    "晶圓代工", "ASIC／矽智財", "記憶體／儲存", "先進封裝設備", "封裝測試",
    "ABF／IC 載板", "CCL／銅箔基板", "散熱／液冷", "電源／BBU", "矽光子／光通訊",
]

NAME_FALLBACKS = {
    "2330": "台積電", "2303": "聯電", "6770": "力積電", "3443": "創意",
    "3661": "世芯-KY", "3035": "智原", "3529": "力旺", "6533": "晶心科",
    "2454": "聯發科", "5269": "祥碩", "6531": "愛普*", "5274": "信驊",
    "4966": "譜瑞-KY", "3583": "辛耘", "3131": "弘塑", "6187": "萬潤",
    "6640": "均華", "6223": "旺矽", "6510": "精測", "3264": "欣銓",
    "6257": "矽格", "2449": "京元電子", "3711": "日月光投控", "6239": "力成",
    "3374": "精材", "6515": "穎崴", "3289": "宜特", "6830": "汎銓",
    "3587": "閎康", "2408": "南亞科", "2344": "華邦電", "8299": "群聯",
    "3260": "威剛", "2342": "茂矽", "4967": "十銓", "2337": "旺宏",
    "5351": "鈺創", "2451": "創見", "3037": "欣興", "3189": "景碩",
    "8046": "南電", "6213": "聯茂", "6274": "台燿", "2383": "台光電",
    "1303": "南亞", "5475": "德宏", "5340": "建榮", "1802": "台玻",
    "1815": "富喬", "2368": "金像電", "3044": "健鼎", "2313": "華通",
    "4908": "前鼎", "5469": "瀚宇博", "3653": "健策", "8996": "高力",
    "3483": "力致", "6230": "尼得科超眾", "3324": "雙鴻", "2421": "建準",
    "3338": "泰碩", "3017": "奇鋐", "6805": "富世達", "2308": "台達電",
    "6412": "群電", "6121": "新普", "6781": "AES-KY", "3323": "加百裕",
    "2301": "光寶科", "3015": "全漢", "6282": "康舒", "1519": "華城",
    "1513": "中興電", "1503": "士電", "1504": "東元", "1609": "大亞",
    "1505": "力特", "2059": "川湖", "3693": "營邦", "3013": "晟銘電",
    "8210": "勤誠", "6117": "迎廣", "3533": "嘉澤", "6197": "佳必琪",
    "3665": "貿聯-KY", "3023": "信邦", "3526": "凡甲", "3217": "優群",
    "2392": "正崴", "6290": "良維", "6205": "詮欣", "3605": "宏致",
    "8042": "金山電", "3624": "光頡", "3026": "禾伸堂", "6449": "鈺邦",
    "2375": "凱美", "3357": "臺慶科", "2472": "立隆電", "2478": "大毅",
    "3090": "日電貿", "2327": "國巨", "2492": "華新科", "6173": "信昌電",
    "3231": "緯創", "6669": "緯穎", "4938": "和碩", "3706": "神達",
    "2376": "技嘉", "2356": "英業達", "2317": "鴻海", "2324": "仁寶",
    "2382": "廣達", "2345": "智邦", "3596": "智易", "6285": "啟碁",
    "4906": "正文", "5388": "中磊", "3380": "明泰", "3363": "上詮",
    "3081": "聯亞", "3234": "光環", "4979": "華星光", "6442": "光聖",
    "3450": "聯鈞", "4977": "眾達-KY", "8086": "宏捷科", "2455": "全新",
    "3105": "穩懋", "4991": "環宇-KY", "3707": "漢磊", "3016": "嘉晶",
    "7717": "萊德光電-KY", "2314": "台揚", "3491": "昇達科", "6215": "和椿",
    "2049": "上銀", "1590": "亞德客-KY", "4583": "台灣精銳", "4576": "大銀微系統",
    "2464": "盟立", "2231": "為升", "6691": "洋基工程", "6196": "帆宣",
    "6139": "亞翔", "6613": "朋億*", "5536": "聖暉*", "2404": "漢唐",
    "2881": "富邦金", "2882": "國泰金", "2891": "中信金", "2886": "兆豐金",
    "0050": "元大台灣50",
}


def now_taipei() -> datetime:
    return datetime.now(TAIPEI)


def _parse_number(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "").replace("%", "")
    if text.lower() in {"", "-", "--", "—", "nan", "none", "null"}:
        return np.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def _parse_twse_date(value: Any) -> pd.Timestamp:
    text = str(value or "").strip()
    if not text:
        return pd.NaT
    # TWSE 回傳的日期通常是民國年，例如 115/10/08。
    import re
    match = re.match(r"^(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})$", text)
    if match:
        year, month, day = map(int, match.groups())
        if year < 1911:
            year += 1911
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return pd.NaT
    return pd.to_datetime(text, errors="coerce")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _fetch_taiex_month(session: requests.Session, month: pd.Timestamp) -> pd.DataFrame:
    params = {"date": month.strftime("%Y%m") + "01", "response": "json"}
    response = session.get(TAIEX_MONTHLY_URL, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    fields = payload.get("fields", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame(columns=["date", "close"])

    frame = pd.DataFrame(rows, columns=fields if fields and len(fields) == len(rows[0]) else None)
    if frame.empty:
        return pd.DataFrame(columns=["date", "close"])

    date_col = next((c for c in frame.columns if "日期" in str(c) or str(c).lower() == "date"), frame.columns[0])
    close_col = next((c for c in frame.columns if "收盤指數" in str(c) or "收盤" in str(c)), None)
    if close_col is None:
        close_col = frame.columns[4] if len(frame.columns) >= 5 else frame.columns[-1]
    out = pd.DataFrame({
        "date": frame[date_col].map(_parse_twse_date),
        "close": frame[close_col].map(_parse_number),
    })
    out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.reset_index(drop=True)


def _load_taiex_history(base: Path, end_date: pd.Timestamp) -> pd.DataFrame:
    """讀取／增量更新 TWSE 官方發行量加權指數歷史資料。"""
    folder = base / "data" / "rotation"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "taiex_daily.csv"
    try:
        cache = pd.read_csv(path, parse_dates=["date"])
        cache["date"] = pd.to_datetime(cache["date"], errors="coerce").dt.normalize()
        cache["close"] = pd.to_numeric(cache["close"], errors="coerce")
        cache = cache.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    except Exception:
        cache = pd.DataFrame(columns=["date", "close"])

    start_date = end_date.normalize() - pd.Timedelta(days=560)
    first_month = start_date.replace(day=1)
    all_months = list(pd.date_range(first_month, end_date.normalize().replace(day=1), freq="MS"))
    cached_months = set(cache["date"].dt.strftime("%Y-%m").tolist()) if not cache.empty else set()
    # 每次重抓最近兩個月份，涵蓋當月尚未完整發布的日資料及最近一期修訂。
    refresh_months = {x.strftime("%Y-%m") for x in all_months[-2:]}
    to_fetch = [m for m in all_months if m.strftime("%Y-%m") not in cached_months or m.strftime("%Y-%m") in refresh_months]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "zh-TW,zh;q=0.9"})
    frames = [cache] if not cache.empty else []
    errors = []
    for month in to_fetch:
        try:
            part = _fetch_taiex_month(session, month)
            if not part.empty:
                frames.append(part)
        except Exception as exc:
            errors.append(f"{month.strftime('%Y-%m')} TAIEX: {type(exc).__name__}: {exc}")
        time.sleep(3.0)  # 官方歷史指數端點降低請求頻率，避免短時間連續查詢遭拒。

    if not frames:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
    out = out[out["date"] >= start_date].tail(450).reset_index(drop=True)
    if not out.empty:
        out.to_csv(path, index=False, encoding="utf-8-sig")
    if errors and out.empty:
        raise RuntimeError("; ".join(errors[:4]))
    return out


def _load_symbol_history(client: FugleClient, base: Path, symbol: str, end_date: pd.Timestamp) -> pd.DataFrame:
    """逐檔增量更新歷史日 K；保留約 560 日曆日，足夠一年 RRG 與暖身窗。"""
    folder = base / "data" / "rotation" / "history"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{clean_symbol(symbol)}.csv"
    try:
        cache = pd.read_csv(path, parse_dates=["date"])
        cache["date"] = pd.to_datetime(cache["date"], errors="coerce").dt.normalize()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in cache.columns:
                cache[col] = pd.to_numeric(cache[col], errors="coerce")
        cache = cache.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
    except Exception:
        cache = pd.DataFrame()

    start_date = end_date.normalize() - pd.Timedelta(days=560)
    if not cache.empty:
        latest = cache["date"].max()
        fetch_start = max(start_date, latest - pd.Timedelta(days=7))
        frames = [cache]
    else:
        fetch_start = start_date
        frames = []

    current = fetch_start
    while current <= end_date.normalize():
        chunk_end = min(current + pd.Timedelta(days=320), end_date.normalize())
        try:
            part = client.historical_candles(
                clean_symbol(symbol), current.date(), chunk_end.date(), "D", adjusted=False
            )
            if part is not None and not part.empty:
                frames.append(part)
        except Exception:
            # 快取存在時保留上次可用資料；資料品質會在公開輸出明確顯示日期。
            if not frames:
                return pd.DataFrame()
        current = chunk_end + pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
    out = out[out["date"] >= start_date].reset_index(drop=True)
    try:
        out.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return out


def _ticker_names(client: FugleClient, symbols: set[str]) -> dict[str, str]:
    names = {code: NAME_FALLBACKS.get(code, code) for code in symbols}
    for exchange in ("TWSE", "TPEx"):
        try:
            rows = client.tickers(exchange)
        except Exception:
            continue
        for row in rows:
            code = clean_symbol(row.get("symbol", ""))
            if code in symbols and str(row.get("name") or "").strip():
                names[code] = str(row["name"]).strip()
    return names


def _return_between(close: pd.Series, current_date: pd.Timestamp, base_date: pd.Timestamp) -> float:
    if current_date not in close.index or base_date not in close.index:
        return np.nan
    current, base_value = _parse_number(close.loc[current_date]), _parse_number(close.loc[base_date])
    if not np.isfinite(current) or not np.isfinite(base_value) or base_value == 0:
        return np.nan
    return (current / base_value - 1.0) * 100.0


def _quadrant(rs: float, momentum: float) -> str:
    if not np.isfinite(rs) or not np.isfinite(momentum):
        return "資料不足"
    if rs >= 100 and momentum >= 100:
        return "領先"
    if rs >= 100 and momentum < 100:
        return "轉弱"
    if rs < 100 and momentum < 100:
        return "落後"
    return "改善"


def _mean_pct(values: list[float]) -> float | None:
    good = [float(v) for v in values if np.isfinite(v)]
    return round(float(np.mean(good)), 3) if good else None


def build_rotation_payload(history_map: dict[str, pd.DataFrame], taiex: pd.DataFrame,
                           names: dict[str, str], generated_at: datetime | None = None) -> dict[str, Any]:
    """將歷史日 K 轉成頁面可直接使用的精簡輪動資訊。"""
    generated_at = generated_at or now_taipei()
    taiex = taiex.copy()
    taiex["date"] = pd.to_datetime(taiex["date"], errors="coerce").dt.normalize()
    taiex["close"] = pd.to_numeric(taiex["close"], errors="coerce")
    taiex = taiex.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
    if taiex.empty:
        raise RuntimeError("沒有 TWSE 加權指數歷史資料")

    trade_dates = pd.DatetimeIndex(taiex["date"].drop_duplicates().sort_values())
    asof = trade_dates[-1]
    if len(trade_dates) < 320:
        raise RuntimeError(f"加權指數資料不足以計算一年輪動（目前 {len(trade_dates)} 個交易日，至少需要 320 日）")

    # 基準採用證交所加權指數；只有官方指數來源無法取得時，排程才會在上層退回 0050。
    benchmark_close = taiex.set_index("date")["close"].sort_index()
    benchmark_return = benchmark_close.pct_change()
    benchmark_index = (1.0 + benchmark_return.fillna(0.0)).cumprod() * 100.0

    all_symbols = sorted(set(code for codes in ROTATION_THEMES.values() for code in codes))
    member_data: dict[str, dict[str, Any]] = {}
    member_returns: dict[str, pd.Series] = {}
    member_close: dict[str, pd.Series] = {}
    member_value: dict[str, pd.Series] = {}

    for code in all_symbols:
        h = history_map.get(code)
        if h is None or h.empty or "date" not in h or "close" not in h:
            continue
        d = h.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        for col in ["close", "volume", "high", "open"]:
            if col in d:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
        close = d.set_index("date")["close"].astype(float)
        close = close[close > 0]
        if close.empty:
            continue
        daily_return = close.pct_change()
        volume = d.set_index("date")["volume"] if "volume" in d else pd.Series(index=close.index, dtype=float)
        value = (close * volume.reindex(close.index)).replace([np.inf, -np.inf], np.nan)
        member_close[code] = close
        member_returns[code] = daily_return
        member_value[code] = value

        latest_date = close.index.max()
        at_asof = float(close.loc[asof]) if asof in close.index else np.nan
        previous_date = trade_dates[-2] if len(trade_dates) >= 2 else pd.NaT
        previous_previous_date = trade_dates[-3] if len(trade_dates) >= 3 else pd.NaT
        one_day = _return_between(close, asof, previous_date) if pd.notna(previous_date) else np.nan
        yesterday = _return_between(close, previous_date, previous_previous_date) if pd.notna(previous_date) and pd.notna(previous_previous_date) else np.nan

        period_results = {}
        for label, bars in [("5d", 5), ("20d", 20), ("60d", 60), ("252d", 252)]:
            idx = trade_dates.get_indexer([asof])[0]
            base_pos = idx - bars
            base_date = trade_dates[base_pos] if base_pos >= 0 else pd.NaT
            period_results[label] = _return_between(close, asof, base_date) if pd.notna(base_date) else np.nan

        this_year_start = pd.Timestamp(year=asof.year, month=1, day=1)
        ytd_base = close[close.index < this_year_start]
        if ytd_base.empty:
            ytd = np.nan
        else:
            ytd = _return_between(close, asof, ytd_base.index[-1])

        last_252 = close.loc[close.index <= asof].tail(252)
        year_high = bool(len(last_252) >= 250 and np.isfinite(at_asof) and at_asof >= float(last_252.max()) * 0.9995)
        fresh = pd.notna(at_asof)
        member_data[code] = {
            "symbol": code,
            "name": names.get(code, NAME_FALLBACKS.get(code, code)),
            "data_date": latest_date.strftime("%Y-%m-%d"),
            "close": round(at_asof, 3) if np.isfinite(at_asof) else None,
            "change_1d": round(float(one_day), 3) if np.isfinite(one_day) else None,
            "return_5d": round(float(period_results["5d"]), 3) if np.isfinite(period_results["5d"]) else None,
            "return_20d": round(float(period_results["20d"]), 3) if np.isfinite(period_results["20d"]) else None,
            "return_60d": round(float(period_results["60d"]), 3) if np.isfinite(period_results["60d"]) else None,
            "return_ytd": round(float(ytd), 3) if np.isfinite(ytd) else None,
            "volume": float(volume.loc[asof]) if fresh and asof in volume.index and np.isfinite(_parse_number(volume.loc[asof])) else None,
            "traded_value": float(value.loc[asof]) if fresh and asof in value.index and np.isfinite(_parse_number(value.loc[asof])) else None,
            "year_high": year_high,
            "fresh": fresh,
            "daily_return_series": daily_return,
        }

    tracked_turnover = sum(
        float(v.loc[asof]) for v in member_value.values()
        if asof in v.index and np.isfinite(_parse_number(v.loc[asof])) and float(v.loc[asof]) > 0
    )
    prev_date = trade_dates[-2] if len(trade_dates) >= 2 else pd.NaT
    prev_turnover = sum(
        float(v.loc[prev_date]) for v in member_value.values()
        if pd.notna(prev_date) and prev_date in v.index and np.isfinite(_parse_number(v.loc[prev_date])) and float(v.loc[prev_date]) > 0
    )
    prev_prev_date = trade_dates[-3] if len(trade_dates) >= 3 else pd.NaT
    prev_prev_turnover = sum(
        float(v.loc[prev_prev_date]) for v in member_value.values()
        if pd.notna(prev_prev_date) and prev_prev_date in v.index
        and np.isfinite(_parse_number(v.loc[prev_prev_date])) and float(v.loc[prev_prev_date]) > 0
    )

    groups: list[dict[str, Any]] = []
    for group_name, codes in ROTATION_THEMES.items():
        available = [code for code in codes if code in member_close]
        return_frame = pd.concat(
            [member_returns[code].rename(code) for code in available], axis=1
        ).sort_index() if available else pd.DataFrame()
        group_daily_return = return_frame.mean(axis=1, skipna=True).dropna() if not return_frame.empty else pd.Series(dtype=float)
        group_index = (1.0 + group_daily_return).cumprod() * 100.0 if not group_daily_return.empty else pd.Series(dtype=float)

        # 相對強弱：族群指數 / 加權指數，再用近 50 日相對強弱均值正規化；動能比較 10 日前。
        if not group_index.empty:
            aligned = pd.concat([group_index.rename("group"), benchmark_index.rename("benchmark")], axis=1).dropna()
            if not aligned.empty:
                relative_raw = aligned["group"] / aligned["benchmark"]
                first = float(relative_raw.iloc[0])
                relative = relative_raw / first * 100.0 if first else relative_raw * np.nan
                rs_series = relative / relative.rolling(50, min_periods=50).mean() * 100.0
                momentum_series = rs_series / rs_series.shift(10) * 100.0
                rrg = pd.DataFrame({"rs": rs_series, "momentum": momentum_series}).replace([np.inf, -np.inf], np.nan).dropna()
            else:
                rrg = pd.DataFrame(columns=["rs", "momentum"])
        else:
            rrg = pd.DataFrame(columns=["rs", "momentum"])

        valid_latest = [
            code for code in codes
            if code in member_data and member_data[code]["data_date"] == asof.strftime("%Y-%m-%d")
            and member_data[code]["close"] is not None
        ]
        current_changes = [member_data[code]["change_1d"] for code in valid_latest if member_data[code]["change_1d"] is not None]
        yesterday_changes = []
        for code in codes:
            if code in member_close and pd.notna(prev_date) and len(trade_dates) >= 3:
                val = _return_between(member_close[code], prev_date, trade_dates[-3])
                if np.isfinite(val):
                    yesterday_changes.append(val)
        p5 = [member_data[code]["return_5d"] for code in valid_latest if member_data[code]["return_5d"] is not None]
        p20 = [member_data[code]["return_20d"] for code in valid_latest if member_data[code]["return_20d"] is not None]
        p60 = [member_data[code]["return_60d"] for code in valid_latest if member_data[code]["return_60d"] is not None]
        pytd = [member_data[code]["return_ytd"] for code in valid_latest if member_data[code]["return_ytd"] is not None]
        year_high_count = sum(1 for code in valid_latest if member_data[code]["year_high"])
        rising = sum(1 for val in current_changes if val > 0)
        falling = sum(1 for val in current_changes if val < 0)
        flat = sum(1 for val in current_changes if val == 0)

        turnover = sum(
            float(member_data[code]["traded_value"]) for code in valid_latest
            if member_data[code]["traded_value"] is not None and float(member_data[code]["traded_value"]) > 0
        )
        previous_turnover = 0.0
        previous_previous_group_turnover = 0.0
        for code in codes:
            if code in member_value and pd.notna(prev_date) and prev_date in member_value[code].index:
                val = _parse_number(member_value[code].loc[prev_date])
                if np.isfinite(val) and val > 0:
                    previous_turnover += float(val)
            if code in member_value and pd.notna(prev_prev_date) and prev_prev_date in member_value[code].index:
                val_prev_prev = _parse_number(member_value[code].loc[prev_prev_date])
                if np.isfinite(val_prev_prev) and val_prev_prev > 0:
                    previous_previous_group_turnover += float(val_prev_prev)
        share = turnover / tracked_turnover * 100.0 if tracked_turnover else np.nan
        previous_share = previous_turnover / prev_turnover * 100.0 if prev_turnover else np.nan
        previous_previous_share = (
            previous_previous_group_turnover / prev_prev_turnover * 100.0
            if prev_prev_turnover else np.nan
        )

        if not rrg.empty:
            last = rrg.iloc[-1]
            rs_now, mom_now = float(last["rs"]), float(last["momentum"])
            quadrant = _quadrant(rs_now, mom_now)
            rrg_points = []
            sampled = rrg.iloc[::5].copy()
            if sampled.empty or sampled.index[-1] != rrg.index[-1]:
                sampled = pd.concat([sampled, rrg.tail(1)])
            # 限制公開資料大小：一年以內、每 5 個交易日一點。
            sampled = sampled.tail(53)
            for dt, point in sampled.iterrows():
                rrg_points.append({
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "rs": round(float(point["rs"]), 3),
                    "momentum": round(float(point["momentum"]), 3),
                })
            prior_q = "資料不足"
            if len(rrg) > 5:
                prior = rrg.iloc[-6]
                prior_q = _quadrant(float(prior["rs"]), float(prior["momentum"]))
        else:
            rs_now, mom_now, quadrant, rrg_points, prior_q = np.nan, np.nan, "資料不足", [], "資料不足"

        share_delta = share - previous_share if np.isfinite(share) and np.isfinite(previous_share) else np.nan
        yesterday_share_delta = (
            previous_share - previous_previous_share
            if np.isfinite(previous_share) and np.isfinite(previous_previous_share) else np.nan
        )
        group = {
            "name": group_name,
            "members_total": len(codes),
            "members_valid": len(available),
            "members_fresh": len(valid_latest),
            "data_coverage_pct": round(len(valid_latest) / len(codes) * 100.0, 1) if codes else 0.0,
            "today_return": _mean_pct(current_changes),
            "yesterday_return": _mean_pct(yesterday_changes),
            "return_5d": _mean_pct(p5),
            "return_20d": _mean_pct(p20),
            "return_60d": _mean_pct(p60),
            "return_ytd": _mean_pct(pytd),
            "rising": int(rising),
            "falling": int(falling),
            "flat": int(flat),
            "year_high_count": int(year_high_count),
            "sample_turnover_share_pct": round(float(share), 3) if np.isfinite(share) else None,
            "sample_turnover_share_delta_pp": round(float(share_delta), 3) if np.isfinite(share_delta) else None,
            "yesterday_sample_turnover_share_pct": round(float(previous_share), 3) if np.isfinite(previous_share) else None,
            "yesterday_sample_turnover_share_delta_pp": round(float(yesterday_share_delta), 3) if np.isfinite(yesterday_share_delta) else None,
            "relative_strength": round(rs_now, 3) if np.isfinite(rs_now) else None,
            "relative_momentum": round(mom_now, 3) if np.isfinite(mom_now) else None,
            "quadrant": quadrant,
            "previous_quadrant": prior_q,
            "quadrant_changed": prior_q not in {"資料不足", quadrant},
            "rrg": rrg_points,
            "components": [
                {k: v for k, v in member_data[code].items() if k != "daily_return_series"}
                for code in codes if code in member_data
            ],
            "leaders": [
                {k: v for k, v in member_data[code].items() if k != "daily_return_series"}
                for code in sorted(valid_latest, key=lambda c: member_data[c]["change_1d"] if member_data[c]["change_1d"] is not None else -999, reverse=True)[:3]
            ],
            "laggards": [
                {k: v for k, v in member_data[code].items() if k != "daily_return_series"}
                for code in sorted(valid_latest, key=lambda c: member_data[c]["change_1d"] if member_data[c]["change_1d"] is not None else 999)[:3]
            ],
        }
        groups.append(group)

    fresh_members = [
        {k: v for k, v in data.items() if k != "daily_return_series"}
        for data in member_data.values()
        if data["data_date"] == asof.strftime("%Y-%m-%d") and data["close"] is not None
    ]
    gainers = sorted(
        [x for x in fresh_members if x.get("change_1d") is not None],
        key=lambda x: x["change_1d"], reverse=True
    )[:10]
    losers = sorted(
        [x for x in fresh_members if x.get("change_1d") is not None],
        key=lambda x: x["change_1d"]
    )[:10]
    benchmark_today = float(benchmark_close.loc[asof]) if asof in benchmark_close.index else np.nan
    benchmark_yesterday = float(benchmark_close.loc[prev_date]) if pd.notna(prev_date) and prev_date in benchmark_close.index else np.nan
    benchmark_change = (benchmark_today / benchmark_yesterday - 1.0) * 100.0 if benchmark_yesterday else np.nan

    return {
        "schema_version": 1,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "data_asof": asof.strftime("%Y-%m-%d"),
        "previous_data_asof": prev_date.strftime("%Y-%m-%d") if pd.notna(prev_date) else None,
        "previous_previous_data_asof": prev_prev_date.strftime("%Y-%m-%d") if pd.notna(prev_prev_date) else None,
        "benchmark": "臺灣加權股價指數（TWSE 官方歷史資料）",
        "benchmark_close": round(benchmark_today, 2) if np.isfinite(benchmark_today) else None,
        "benchmark_change_1d": round(float(benchmark_change), 3) if np.isfinite(benchmark_change) else None,
        "is_trading_day": True,
        "data_source": ["Fugle 歷史日 K", "臺灣證券交易所加權指數歷史資料"],
        "method": "子產業代表股每日報酬等權平均；RRG 以族群指數相對加權指數正規化，RS 使用 50 日均值，動能比較 10 個交易日前。",
        "turnover_method": "樣本成交值占比變化僅是本頁追蹤股票的聚焦代理，不代表全市場實際資金流；同一成分股可屬多個主題。",
        "tracked_unique_stocks": len(all_symbols),
        "fresh_stock_count": len(fresh_members),
        "up_group_count": sum(1 for g in groups if g["today_return"] is not None and g["today_return"] > 0),
        "down_group_count": sum(1 for g in groups if g["today_return"] is not None and g["today_return"] < 0),
        "leaders": [g["name"] for g in sorted(groups, key=lambda x: x["relative_strength"] if x["relative_strength"] is not None else -1, reverse=True) if g["quadrant"] == "領先"][:10],
        "quadrant_changes": [
            {"name": g["name"], "from": g["previous_quadrant"], "to": g["quadrant"]}
            for g in groups if g["quadrant_changed"]
        ],
        "gainers": gainers,
        "losers": losers,
        "groups": sorted(groups, key=lambda x: x["return_20d"] if x["return_20d"] is not None else -999, reverse=True),
    }


def publish_to_github(payload: dict[str, Any], repo: str | None = None) -> dict[str, Any]:
    """用既有 GITHUB_TOKEN 將不含私密資料的輪動摘要發布至 repository。"""
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError("找不到 GITHUB_TOKEN 環境變數。請在 Windows 使用者環境變數設定有 repo contents write 權限的 Token。")

    repo = repo or os.getenv("GITHUB_REPO", GITHUB_REPO_DEFAULT)
    api_url = f"https://api.github.com/repos/{repo}/contents/{PUBLIC_REPO_PATH}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    session = requests.Session()
    session.headers.update(headers)
    current_sha = None

    for attempt in range(2):
        existing = session.get(api_url, params={"ref": "main"}, timeout=25)
        if existing.status_code == 200:
            current_sha = existing.json().get("sha")
        elif existing.status_code == 404:
            current_sha = None
        else:
            raise RuntimeError(f"GitHub 讀取現有輪動檔失敗 HTTP {existing.status_code}: {existing.text[:500]}")

        body = json.dumps(payload, ensure_ascii=False, indent=2, separators=(",", ": "), allow_nan=False)
        content_b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
        commit_date = str(payload.get("data_asof", "unknown"))
        put_body: dict[str, Any] = {
            "message": f"Update industry analysis data {commit_date}",
            "content": content_b64,
            "branch": "main",
        }
        if current_sha:
            put_body["sha"] = current_sha
        pushed = session.put(api_url, json=put_body, timeout=45)
        if pushed.status_code in (200, 201):
            return {
                "ok": True,
                "repo": repo,
                "path": PUBLIC_REPO_PATH,
                "commit_sha": pushed.json().get("commit", {}).get("sha"),
            }
        if pushed.status_code == 409 and attempt == 0:
            continue
        raise RuntimeError(f"GitHub 發布輪動資料失敗 HTTP {pushed.status_code}: {pushed.text[:800]}")
    raise RuntimeError("GitHub 發布失敗：重試後仍有版本衝突。")


def update_industry_rotation(base_dir: str | Path, force: bool = False, publish: bool = True) -> dict[str, Any]:
    """更新一輪資料；若官方基準未更新到今天，保留上一份發布資料。"""
    base = Path(base_dir).resolve()
    now = now_taipei()
    today = pd.Timestamp(now.date())
    trading_today = is_twse_trading_day(now, base)
    if not force:
        if not trading_today:
            return {"ok": True, "skipped": True, "reason": "今天不是台股交易日", "date": today.strftime("%Y-%m-%d")}
        if now.time() < dt_time(15, 10):
            return {"ok": True, "skipped": True, "reason": "尚未到排程時間 15:10", "date": today.strftime("%Y-%m-%d")}

    client = FugleClient()
    errors: list[str] = []
    taiex = pd.DataFrame()

    # 排程執行時只接受今日官方資料；手動 --force 在週末／休市日可用最近交易日資料初始化。
    if force and not trading_today:
        taiex = _load_taiex_history(base, today)
        if taiex.empty:
            raise RuntimeError("無法取得 TWSE 加權指數歷史資料；沒有覆蓋上一份成功資料。")
        target_date = pd.to_datetime(taiex["date"], errors="coerce").max().normalize()
    else:
        target_date = today
        for attempt in range(6):
            try:
                taiex = _load_taiex_history(base, today)
            except Exception as exc:
                errors.append(f"TAIEX 更新失敗：{type(exc).__name__}: {exc}")
                taiex = pd.DataFrame()
            if not taiex.empty and pd.to_datetime(taiex["date"], errors="coerce").max().normalize() == target_date:
                break
            if attempt < 5:
                print(f"官方加權指數尚未更新到 {target_date.date()}，60 秒後重試（{attempt + 1}/6）", flush=True)
                time.sleep(60)

        if taiex.empty or pd.to_datetime(taiex["date"], errors="coerce").max().normalize() != target_date:
            raise RuntimeError(
                f"無法確認官方加權指數已更新至 {target_date.date()}；不覆蓋上一份成功資料。"
                + ("；" + "；".join(errors[-3:]) if errors else "")
            )

    symbols = sorted(set(code for codes in ROTATION_THEMES.values() for code in codes))
    history_map: dict[str, pd.DataFrame] = {}
    names = {code: NAME_FALLBACKS.get(code, code) for code in symbols}

    try:
        names.update(_ticker_names(client, set(symbols)))
    except Exception as exc:
        errors.append(f"股票名稱補齊失敗，使用備援名稱：{type(exc).__name__}: {exc}")

    log_path = base / "logs" / "industry_rotation_update.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{now.isoformat(timespec='seconds')}] start symbols={len(symbols)}\n")
        for index, symbol in enumerate(symbols, start=1):
            try:
                history = _load_symbol_history(client, base, symbol, target_date)
                if not history.empty:
                    history_map[symbol] = history
            except Exception as exc:
                errors.append(f"{symbol} 歷史日 K 更新失敗：{type(exc).__name__}: {exc}")
            if index % 10 == 0 or index == len(symbols):
                log.write(f"[{now_taipei().isoformat(timespec='seconds')}] progress={index}/{len(symbols)} valid={len(history_map)}\n")
                log.flush()
            time.sleep(0.15)

    payload = build_rotation_payload(history_map, taiex, names, generated_at=now_taipei())
    payload["errors"] = errors[:30]
    # 至少需要有足夠主題提供今日有效資料，才發布這次快照。
    usable_groups = [g for g in payload["groups"] if g["members_fresh"] > 0 and g["quadrant"] != "資料不足"]
    if len(usable_groups) < 8:
        raise RuntimeError(f"可用主題只有 {len(usable_groups)} 群，品質檢查未通過；保留上一份成功資料。")

    public_path = base / "output" / "research_reports" / "industry_rotation_public.json"
    _atomic_write(public_path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    publish_result = None
    if publish:
        publish_result = publish_to_github(payload)
    return {
        "ok": True,
        "data_asof": payload["data_asof"],
        "generated_at": payload["generated_at"],
        "groups": len(payload["groups"]),
        "fresh_stock_count": payload["fresh_stock_count"],
        "public_file": str(public_path),
        "github": publish_result,
        "errors": errors[:20],
    }


def load_published_rotation(base_dir: str | Path) -> dict[str, Any]:
    """載入已發布的產業輪動 JSON；沒有檔案時回傳空 dict。"""
    path = Path(base_dir).resolve() / "output" / "research_reports" / "industry_rotation_public.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
