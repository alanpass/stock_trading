# -*- coding: utf-8 -*-
"""趨勢型態分析引擎。

以產業分析 ROTATION_THEMES 中的所有概念股為範圍。每天讀取日 K，
使用可解釋的規則辨識常見圖表型態，並只在型態／趨勢狀態有重大變化時更新公開快照。
此為技術型態啟發式辨識，不是保證預測或投資建議。
"""
from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from industry_rotation import ROTATION_THEMES, NAME_FALLBACKS
from market_calendar import is_twse_trading_day
from stock_api import FugleClient, clean_symbol

TAIPEI = ZoneInfo("Asia/Taipei")
GITHUB_REPO_DEFAULT = "alanpass/stock_trading"
PUBLIC_REPO_PATH = "output/research_reports/trend_patterns_public.json"
USER_AGENT = "Mozilla/5.0 TaiwanStockTrendPatterns/1.0"

CATEGORY_ORDER = [
    "資料調整檢查", "底部反轉候選", "多頭延續／持續創高",
    "整理待突破", "高檔轉弱／反轉警示", "下降趨勢／弱勢整理", "中性觀察",
]
ADJUSTMENT_METHOD = "Fugle 還原股價（adjusted=True）"
SUSPICIOUS_GAP_THRESHOLD = 0.35
ANALYSIS_VERSION = "pattern_window_40_trading_days_v1"
CONSOLIDATION_PATTERNS = {"杯柄", "多頭旗形", "空頭旗形", "上升三角形", "下降三角形", "對稱三角形", "箱型整理", "盤整待突破"}
MATERIAL_EVENTS = {"箱頂突破", "近20日壓力突破", "連跌後轉穩（3日）", "上漲轉弱（3日）", "向下跌破支撐"}

THEME_BY_SYMBOL: dict[str, list[str]] = {}
for _theme, _codes in ROTATION_THEMES.items():
    for _code in _codes:
        THEME_BY_SYMBOL.setdefault(clean_symbol(_code), []).append(_theme)


def _now() -> datetime:
    return datetime.now(TAIPEI)


def _normalise_history(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
    out = out[out["close"] > 0].reset_index(drop=True)
    if out.empty:
        return out
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])
    return out


def load_symbol_history(base: Path, client: FugleClient, symbol: str, asof: pd.Timestamp) -> pd.DataFrame:
    """取得近半年還原日 K，避免股票股利或分割被誤判成單日暴跌。

    使用獨立快取，不混用產業輪動頁的 adjusted=False 資料。每次排程重新查近 260 天，
    因為新除權息或股票股利可能追溯調整之前的歷史 OHLC。
    """
    clean = clean_symbol(symbol)
    folder = base / "data" / "trend_patterns" / "history_adjusted"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{clean}.csv"
    start = (asof.normalize() - pd.Timedelta(days=260)).date()
    end = asof.normalize().date()
    cached = pd.DataFrame()
    try:
        cached = _normalise_history(pd.read_csv(path, parse_dates=["date"]))
    except Exception:
        pass

    try:
        # Fugle adjusted=True 使用還原股價序列；每次抓回完整分析窗，確保新公司行動能追溯調整歷史價格。
        refreshed = client.historical_candles(clean, start, end, "D", adjusted=True)
        refreshed = _normalise_history(refreshed)
        refreshed = refreshed[refreshed["date"] >= pd.Timestamp(start)].tail(200).reset_index(drop=True)
        if len(refreshed) < 45:
            raise RuntimeError(f"Fugle adjusted=True 回傳日 K 不足 45 根（{len(refreshed)} 根）")
        temp = path.with_suffix(".csv.tmp")
        refreshed.to_csv(temp, index=False, encoding="utf-8-sig")
        temp.replace(path)
        return refreshed
    except Exception as exc:
        # 僅在快取足夠新、且資料本身已由 adjusted=True 取得時才容許沿用。
        if not cached.empty and len(cached) >= 45:
            latest = pd.Timestamp(cached["date"].max()).normalize()
            if latest >= asof.normalize() - pd.Timedelta(days=5):
                return cached
        raise RuntimeError(f"{clean} 還原股價歷史資料取得失敗：{type(exc).__name__}: {exc}") from exc


def _pivot_indices(values: np.ndarray, kind: str, order: int = 3) -> list[int]:
    points: list[int] = []
    if len(values) < order * 2 + 3:
        return points
    for i in range(order, len(values) - order):
        window = values[i - order:i + order + 1]
        if not np.isfinite(values[i]) or not np.isfinite(window).all():
            continue
        centre = values[i]
        if kind == "high" and centre >= np.max(window) and int(np.argmax(window)) == order:
            points.append(i)
        elif kind == "low" and centre <= np.min(window) and int(np.argmin(window)) == order:
            points.append(i)
    cleaned: list[int] = []
    for idx in points:
        if cleaned and idx - cleaned[-1] < order + 2:
            prev = cleaned[-1]
            better = values[idx] >= values[prev] if kind == "high" else values[idx] <= values[prev]
            if better:
                cleaned[-1] = idx
        else:
            cleaned.append(idx)
    return cleaned


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 5:
        return 0.0
    x = np.arange(len(values), dtype=float)[good]
    y = values[good]
    return float(np.polyfit(x, y, 1)[0] / max(float(np.nanmean(np.abs(y))), 1e-9))


def _find_double_pattern(highs: np.ndarray, lows: np.ndarray, close: np.ndarray, mode: str):
    if len(close) < 30:
        return None
    if mode == "bottom":
        points = _pivot_indices(lows, "low", 3)
        for p in reversed(points):
            for q in reversed(points[:-1]):
                if not 8 <= p - q <= 38:
                    continue
                a, b = float(lows[q]), float(lows[p])
                if abs(a - b) / max((a + b) / 2, 1e-9) > 0.045:
                    continue
                middle = float(np.max(highs[q:p + 1]))
                if middle / max(a, b) - 1 < 0.045:
                    continue
                crossed = float(close[-1]) > middle * 1.005
                return ("雙底突破" if crossed else "雙底", 0.82 if crossed else 0.68, middle,
                        "雙底已站上中間反彈高點。" if crossed else "兩個相近低點構成潛在雙底，仍需觀察中間壓力是否突破。")
    else:
        points = _pivot_indices(highs, "high", 3)
        for p in reversed(points):
            for q in reversed(points[:-1]):
                if not 8 <= p - q <= 38:
                    continue
                a, b = float(highs[q]), float(highs[p])
                if abs(a - b) / max((a + b) / 2, 1e-9) > 0.04:
                    continue
                middle = float(np.min(lows[q:p + 1]))
                if min(a, b) / max(middle, 1e-9) - 1 < 0.04:
                    continue
                crossed = float(close[-1]) < middle * 0.995
                return ("雙頂跌破" if crossed else "雙頂", 0.82 if crossed else 0.67, middle,
                        "雙頂已跌破中間回檔低點。" if crossed else "兩個相近高點顯示壓力可能存在，仍需觀察支撐是否跌破。")
    return None


def _find_head_shoulders(highs: np.ndarray, lows: np.ndarray, close: np.ndarray):
    ph = _pivot_indices(highs, "high", 3)
    for k in range(len(ph) - 1, 1, -1):
        left, head, right = ph[k - 2:k + 1]
        if not (4 <= head - left <= 22 and 4 <= right - head <= 22):
            continue
        a, b, c = float(highs[left]), float(highs[head]), float(highs[right])
        if b >= max(a, c) * 1.035 and abs(a - c) / max((a + c) / 2, 1e-9) <= 0.075:
            neck = min(float(np.min(lows[left:head + 1])), float(np.min(lows[head:right + 1])))
            if close[-1] < neck * 0.995:
                return "頭肩頂跌破", 0.84, neck, "左肩、較高頭部與右肩成形，且已跌破頸線。"
            if close[-1] < neck * 1.04:
                return "頭肩頂", 0.68, neck, "左肩、頭部與右肩近似成形；頸線尚未確認跌破。"
    pl = _pivot_indices(lows, "low", 3)
    for k in range(len(pl) - 1, 1, -1):
        left, head, right = pl[k - 2:k + 1]
        if not (4 <= head - left <= 22 and 4 <= right - head <= 22):
            continue
        a, b, c = float(lows[left]), float(lows[head]), float(lows[right])
        if b <= min(a, c) * 0.965 and abs(a - c) / max((a + c) / 2, 1e-9) <= 0.075:
            neck = max(float(np.max(highs[left:head + 1])), float(np.max(highs[head:right + 1])))
            if close[-1] > neck * 1.005:
                return "倒頭肩突破", 0.84, neck, "倒頭肩結構已收盤突破頸線。"
            if close[-1] > neck * 0.96:
                return "倒頭肩", 0.68, neck, "倒頭肩可能成形，仍需觀察頸線突破。"
    return None


def detect_pattern(history: pd.DataFrame) -> dict[str, Any]:
    """使用最近 40 個交易日辨識型態，並以約 6 個月資料判斷背景趨勢。"""
    data = _normalise_history(history)
    if len(data) < 45:
        raise ValueError(f"可用日 K 不足 45 根，目前 {len(data)} 根")
    data = data.tail(260).reset_index(drop=True)
    all_close = data["close"].to_numpy(dtype=float)
    all_high = data["high"].to_numpy(dtype=float)
    all_low = data["low"].to_numpy(dtype=float)
    n = len(data)
    width = min(40, n)
    close, highs, lows = all_close[-width:], all_high[-width:], all_low[-width:]
    last = float(close[-1])
    ret3 = last / float(close[-4]) - 1
    ret10 = last / float(close[-11]) - 1 if width >= 11 else ret3
    ret20 = last / float(close[-21]) - 1 if width >= 21 else ret3
    ret40 = last / float(close[0]) - 1
    day_ret = pd.Series(close).pct_change().tail(3).dropna().to_numpy()
    ma20 = float(pd.Series(all_close).rolling(20).mean().iloc[-1])
    ma50 = float(pd.Series(all_close).rolling(50).mean().iloc[-1]) if n >= 50 else ma20

    six_n = min(126, n)
    s_high, s_low, s_close = all_high[-six_n:], all_low[-six_n:], all_close[-six_n:]
    peak, trough = float(np.max(s_high)), float(np.min(s_low))
    location = (last - trough) / max(peak - trough, 1e-9)
    ret6m = last / float(s_close[0]) - 1
    peak_idx, trough_idx = int(np.argmax(s_high)), int(np.argmin(s_low))
    if peak_idx < int(six_n * 0.58) and last <= peak * 0.88 and ret40 < -0.02:
        direction6m = "近半年由高點回落"
    elif trough_idx < int(six_n * 0.72) and last >= trough * 1.15 and ret40 > 0.02:
        direction6m = "近半年由低點回升"
    elif location >= 0.72:
        direction6m = "半年區間偏高"
    elif location <= 0.28:
        direction6m = "半年區間偏低"
    else:
        direction6m = "半年區間中段"

    # 還原股價後若仍出現超出正常台股單日波動範圍的價格斷層，先做資料檢查。
    recent_returns = data["close"].pct_change()
    review_returns = recent_returns.iloc[max(1, n - 150):]
    suspicious = review_returns[review_returns.abs() >= SUSPICIOUS_GAP_THRESHOLD]
    if not suspicious.empty:
        gap_idx = int(suspicious.index[-1])
        gap_pct = float(suspicious.iloc[-1]) * 100
        gap_date = pd.Timestamp(data["date"].iloc[gap_idx]).strftime("%Y-%m-%d")
        return {
            "analysis_version": ANALYSIS_VERSION,
            "pattern_name": "價格基準待確認",
            "classification": "資料調整檢查",
            "trend_40d": "暫不判定",
            "market_direction_6m": "暫不判定（價格基準檢查中）",
            "return_3d_pct": round(ret3 * 100, 2),
            "return_10d_pct": round(ret10 * 100, 2),
            "return_20d_pct": round(ret20 * 100, 2),
            "return_40d_pct": round(ret40 * 100, 2),
            "return_6m_pct": round(ret6m * 100, 2),
            "close_at_assessment": round(last, 2),
            "support": round(float(np.min(lows[-20:])), 2),
            "resistance": round(float(np.max(highs[-20:])), 2),
            "confidence": 0.0,
            "event": "疑似公司行動／價格斷層",
            "corporate_action_review": True,
            "suspected_gap_date": gap_date,
            "suspected_gap_pct": round(gap_pct, 2),
            "adjustment_method": ADJUSTMENT_METHOD,
            "reason": (
                f"{gap_date} 的還原收盤價仍出現 {gap_pct:+.1f}% 單日跳空；"
                "可能是股票分割、股票股利、除權息或異常行情。已暫停型態方向判斷，需先核對公司行動與資料來源。"
            ),
            "last_bar_date": pd.Timestamp(data["date"].iloc[-1]).strftime("%Y-%m-%d"),
        }

    pattern, confidence = "箱型整理", 0.48
    resistance, note = float(np.max(highs[-25:])), "最近 40 個交易日價格在區間內反覆震盪，尚未符合更明確的典型形態。"
    hs = _find_head_shoulders(highs, lows, close)
    db = _find_double_pattern(highs, lows, close, "bottom")
    dt = _find_double_pattern(highs, lows, close, "top")

    # 杯柄：兩側接近杯沿、中段形成杯底，最後約兩週出現較淺回檔。
    if width >= 40:
        rim_l, rim_r = float(np.mean(close[:8])), float(np.mean(close[-20:-10]))
        rim = (rim_l + rim_r) / 2
        cup_low, handle_low = float(np.min(close[12:-15])), float(np.min(close[-10:]))
        depth = 1 - cup_low / max(rim, 1e-9)
        handle = 1 - handle_low / max(rim, 1e-9)
        if 0.07 <= depth <= 0.32 and abs(rim_l - rim_r) / max(rim, 1e-9) <= 0.07 and 0.015 <= handle <= 0.10 and last >= rim * 0.94:
            pattern, confidence, resistance, note = "杯柄", 0.67, rim, "價格回到杯沿附近後形成較淺回檔柄部；杯沿突破才算確認。"
    if hs and hs[0].startswith("倒頭肩"):
        pattern, confidence, resistance, note = hs
    elif hs and hs[0].startswith("頭肩頂"):
        pattern, confidence, resistance, note = hs
    elif db:
        pattern, confidence, resistance, note = db
    elif dt:
        pattern, confidence, resistance, note = dt

    if pattern == "箱型整理" and width >= 45:
        left_avg, mid_avg, right_avg = float(np.mean(close[:15])), float(np.mean(close[18:43])), float(np.mean(close[-12:]))
        trough_pos = int(np.argmin(close)) / max(len(close) - 1, 1)
        if left_avg > mid_avg * 1.025 and right_avg > mid_avg * 1.035 and 0.28 <= trough_pos <= 0.78 and ret10 > 0:
            pattern, confidence, resistance, note = "圓底", 0.64, float(np.max(highs[-30:-10])), "中段形成弧形低點、右側均價回升；仍需觀察前方壓力。"

    # 三角形以價格區間前後半段的邊界變化近似辨識。
    if pattern == "箱型整理" and width >= 30:
        eh, lh = float(np.max(highs[-30:-15])), float(np.max(highs[-15:]))
        el, ll = float(np.min(lows[-30:-15])), float(np.min(lows[-15:]))
        top_flat = abs(lh / max(eh, 1e-9) - 1) <= 0.03
        bottom_flat = abs(ll / max(el, 1e-9) - 1) <= 0.03
        top_falling, bottom_rising = lh < eh * 0.97, ll > el * 1.02
        if top_flat and bottom_rising:
            pattern, confidence, resistance, note = "上升三角形", 0.66, (eh + lh) / 2, "高點大致水平、低點逐步墊高；收盤突破壓力可確認。"
        elif bottom_flat and top_falling:
            pattern, confidence, resistance, note = "下降三角形", 0.65, ll, "低點大致水平、反彈高點逐步降低；跌破支撐偏弱。"
        elif top_falling and bottom_rising:
            pattern, confidence, resistance, note = "對稱三角形", 0.63, lh, "高點下移、低點上移，波動收斂；方向需等有效突破。"

    # 旗形：前段快速推進，最近十個交易日的整理相對收斂。
    if pattern == "箱型整理" and width >= 30:
        pole = float(close[-11] / close[-26] - 1)
        flag = float(close[-1] / close[-11] - 1)
        flag_range = float((np.max(highs[-10:]) - np.min(lows[-10:])) / max(last, 1e-9))
        if pole >= 0.09 and -0.08 <= flag <= 0.035 and flag_range <= 0.13:
            pattern, confidence, resistance, note = "多頭旗形", 0.62, float(np.max(highs[-10:])), "先有明顯上漲，近期以較小區間整理；向上突破才確認延續。"
        elif pole <= -0.09 and -0.035 <= flag <= 0.08 and flag_range <= 0.13:
            pattern, confidence, resistance, note = "空頭旗形", 0.62, float(np.min(lows[-10:])), "先有明顯下跌，近期反彈有限；跌破整理區才確認延續。"

    if pattern == "箱型整理" and width >= 30:
        h_slope, l_slope = _slope(highs[-30:]), _slope(lows[-30:])
        if h_slope > 0.0005 and l_slope > h_slope * 1.20:
            pattern, confidence, resistance, note = "上升楔形", 0.63, float(np.min(lows[-10:])), "高低點同步上移但下方支撐斜率更陡，留意上漲動能收斂。"
        elif h_slope < -0.0005 and h_slope < l_slope - 0.00045:
            pattern, confidence, resistance, note = "下降楔形", 0.63, float(np.max(highs[-10:])), "高低點同步下移但上方壓力下降更快，留意向上突破。"

    if pattern == "箱型整理":
        high40, low40 = float(np.max(highs)), float(np.min(lows))
        if last >= high40 * 0.975 and ret40 > 0.06 and last > ma20 and ma20 >= ma50 * 0.97:
            pattern, confidence, resistance, note = "持續創高", 0.72, high40, "收盤接近最近 40 個交易日高點，且中短期均線與報酬偏多。"
        elif last < ma20 and ma20 < ma50 and ret20 < -0.05:
            pattern, confidence, resistance, note = "下降趨勢", 0.70, float(np.max(highs[-20:])), "收盤低於中短期均線，近 20 日報酬偏弱。"
        elif last > ma20 and ma20 > ma50 and ret20 > 0.04:
            pattern, confidence, resistance, note = "上升趨勢", 0.68, float(np.max(highs[-20:])), "收盤高於上行的中短期均線，近期趨勢偏多。"

    # 使用者指定的三日轉折與區間突破。
    up_days, down_days = int(np.sum(day_ret > 0)), int(np.sum(day_ret < 0))
    turning_up = ret10 <= -0.055 and ret3 >= 0.015 and up_days >= 2
    turning_down = ret10 >= 0.055 and ret3 <= -0.015 and down_days >= 2
    event = "連跌後轉穩（3日）" if turning_up else "上漲轉弱（3日）" if turning_down else ""
    previous_close = float(all_close[-2])
    prior20_high = float(np.max(all_high[-21:-1])) if n >= 22 else float(np.max(all_high[:-1]))
    prior20_low = float(np.min(all_low[-21:-1])) if n >= 22 else float(np.min(all_low[:-1]))
    crossed_up = previous_close <= prior20_high * 1.005 and last > prior20_high * 1.005
    crossed_down = previous_close >= prior20_low * 0.995 and last < prior20_low * 0.995
    if not event and crossed_up:
        event = "箱頂突破" if pattern in CONSOLIDATION_PATTERNS else "近20日壓力突破"
    elif not event and crossed_down:
        event = "向下跌破支撐"

    if event == "連跌後轉穩（3日）" or pattern in {"雙底", "雙底突破", "倒頭肩", "倒頭肩突破", "圓底", "下降楔形"}:
        classification = "底部反轉候選"
    elif event == "上漲轉弱（3日）" or pattern in {"頭肩頂", "頭肩頂跌破", "雙頂", "雙頂跌破", "上升楔形"}:
        classification = "高檔轉弱／反轉警示"
    elif event in {"箱頂突破", "近20日壓力突破"} or pattern in {"持續創高", "上升趨勢", "多頭旗形"}:
        classification = "多頭延續／持續創高"
    elif event == "向下跌破支撐" or pattern in {"下降趨勢", "下降三角形", "空頭旗形"} or (ret20 < -0.06 and last < ma20):
        classification = "下降趨勢／弱勢整理"
    elif pattern in CONSOLIDATION_PATTERNS:
        classification = "整理待突破"
    else:
        classification = "中性觀察"

    trend40d = "上升" if ret40 >= 0.06 and last >= ma20 else "下降" if ret40 <= -0.06 and last <= ma20 else "區間／整理"
    return {
        "analysis_version": ANALYSIS_VERSION,
        "pattern_name": pattern, "classification": classification, "trend_40d": trend40d,
        "market_direction_6m": direction6m,
        "return_3d_pct": round(ret3 * 100, 2), "return_10d_pct": round(ret10 * 100, 2),
        "return_20d_pct": round(ret20 * 100, 2), "return_40d_pct": round(ret40 * 100, 2),
        "return_6m_pct": round(ret6m * 100, 2), "close_at_assessment": round(last, 2),
        "support": round(float(np.min(lows[-20:])), 2),
        "resistance": round(float(resistance if resistance is not None and np.isfinite(resistance) else np.max(highs[-20:])), 2),
        "confidence": round(float(confidence), 2), "event": event,
        "corporate_action_review": False,
        "adjustment_method": ADJUSTMENT_METHOD,
        "reason": note + f" 近 40 個交易日報酬 {ret40 * 100:+.1f}%，近 3 日 {ret3 * 100:+.1f}%。",
        "last_bar_date": pd.Timestamp(data["date"].iloc[-1]).strftime("%Y-%m-%d"),
    }


def _event_for_change(old: dict[str, Any] | None, new: dict[str, Any]) -> str:
    raw = str(new.get("event") or "")
    if raw in MATERIAL_EVENTS:
        return raw
    if old is None:
        return "初始型態建檔"
    if old.get("classification") != new.get("classification"):
        return f"趨勢狀態改變：{old.get('classification')} → {new.get('classification')}"
    return f"型態改變：{old.get('pattern_name')} → {new.get('pattern_name')}"


def _is_material_change(old: dict[str, Any] | None, new: dict[str, Any]) -> tuple[bool, str]:
    if old is None:
        return True, "初始型態建檔"
    if old.get("adjustment_method") != ADJUSTMENT_METHOD:
        return True, "改用還原股價重新校正型態"
    if old.get("analysis_version") != ANALYSIS_VERSION:
        return True, "型態判斷區間改為最近 40 個交易日，重新校正型態"
    if old.get("classification") != new.get("classification"):
        return True, _event_for_change(old, new)
    if old.get("pattern_name") != new.get("pattern_name") and float(new.get("confidence") or 0) >= 0.64:
        return True, _event_for_change(old, new)
    raw, old_event = str(new.get("event") or ""), str(old.get("event") or "")
    if raw in MATERIAL_EVENTS and raw != old_event and old.get("last_bar_date") != new.get("last_bar_date"):
        if raw in {"連跌後轉穩（3日）", "上漲轉弱（3日）", "向下跌破支撐"}:
            return True, raw
        if raw in {"箱頂突破", "近20日壓力突破"} and old.get("classification") != "多頭延續／持續創高":
            return True, raw
    return False, ""


def _build_symbol_record(symbol: str, name: str, data: dict[str, Any], old: dict[str, Any] | None, checked_at: str):
    changed, event = _is_material_change(old, data)
    if old and not changed:
        # 保留上次重大變化的型態判讀；個股圖在使用者展開時即時查詢。
        return old, False
    row = {
        "symbol": clean_symbol(symbol),
        "name": str(name or NAME_FALLBACKS.get(clean_symbol(symbol), clean_symbol(symbol))),
        "themes": sorted(THEME_BY_SYMBOL.get(clean_symbol(symbol), [])),
        **data,
        "last_major_change_at": checked_at,
        "major_change": event,
        "assessment_basis": "最近 40 個交易日日 K 型態 + 近 6 個月背景趨勢；規則式啟發判讀",
    }
    return row, True


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def _publish_to_github(payload: dict[str, Any]) -> dict[str, Any]:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError("找不到 GITHUB_TOKEN。請設定具有 Repository Contents 讀寫權限的 Token。")
    repo = os.getenv("GITHUB_REPO", GITHUB_REPO_DEFAULT)
    api_url = f"https://api.github.com/repos/{repo}/contents/{PUBLIC_REPO_PATH}"
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": USER_AGENT,
    })
    body_text = json.dumps(payload, ensure_ascii=False, indent=2, separators=(",", ": "), allow_nan=False) + "\n"
    encoded = base64.b64encode(body_text.encode("utf-8")).decode("ascii")
    for attempt in range(2):
        existing = session.get(api_url, params={"ref": "main"}, timeout=25)
        sha = None
        if existing.status_code == 200:
            detail = existing.json()
            sha = detail.get("sha")
            try:
                old_text = base64.b64decode(str(detail.get("content") or "").encode("ascii")).decode("utf-8", errors="replace")
                if old_text.strip() == body_text.strip():
                    return {"ok": True, "no_change": True, "path": PUBLIC_REPO_PATH, "repo": repo}
            except Exception:
                pass
        elif existing.status_code != 404:
            raise RuntimeError(f"GitHub 讀取趨勢型態快照失敗 HTTP {existing.status_code}: {existing.text[:500]}")
        put_body = {"message": f"Update trend pattern analysis {payload.get('updated_at', '')}", "content": encoded, "branch": "main"}
        if sha:
            put_body["sha"] = sha
        pushed = session.put(api_url, json=put_body, timeout=45)
        if pushed.status_code in (200, 201):
            return {"ok": True, "repo": repo, "path": PUBLIC_REPO_PATH, "commit_sha": pushed.json().get("commit", {}).get("sha")}
        if pushed.status_code == 409 and attempt == 0:
            continue
        raise RuntimeError(f"GitHub 發布趨勢型態快照失敗 HTTP {pushed.status_code}: {pushed.text[:800]}")
    raise RuntimeError("GitHub 發布趨勢型態快照失敗：重試後仍有版本衝突。")


def update_trend_patterns(base_dir: str | Path, force: bool = False, publish: bool = True) -> dict[str, Any]:
    """每日檢查所有概念股；沒有重大型態變化時不改寫／發布公開快照。"""
    base = Path(base_dir).resolve()
    now = _now()
    asof = pd.Timestamp(now.date())
    if not force and not is_twse_trading_day(now, base):
        return {"ok": True, "skipped": True, "reason": "非台股交易日，不更新趨勢型態。"}

    public_path = base / PUBLIC_REPO_PATH
    try:
        prior_payload = json.loads(public_path.read_text(encoding="utf-8"))
        if not isinstance(prior_payload, dict):
            prior_payload = {}
    except Exception:
        prior_payload = {}
    prior_rows = prior_payload.get("stocks", [])
    # --no-publish 產生的本機預覽不能當作「已發布快照」；
    # 下次以正常模式執行時，仍須把初始資料發布到 GitHub。
    is_local_preview = prior_payload.get("publication_status") == "local_preview"
    prior = {
        clean_symbol(str(row.get("symbol") or "")): row
        for row in prior_rows if isinstance(row, dict) and row.get("symbol")
    } if isinstance(prior_rows, list) and not is_local_preview else {}

    symbols = sorted({clean_symbol(code) for codes in ROTATION_THEMES.values() for code in codes})
    client = FugleClient()
    updated_rows: dict[str, dict[str, Any]] = dict(prior)
    changes: list[dict[str, Any]] = []
    errors: list[str] = []
    latest_dates: list[str] = []
    checked_at = now.isoformat(timespec="seconds")
    log_path = base / "output" / "scheduler_logs" / f"trend_patterns_{now.date().isoformat()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{checked_at}] START symbols={len(symbols)} previous={len(prior)}\n")
        for index, symbol in enumerate(symbols, start=1):
            try:
                history = load_symbol_history(base, client, symbol, asof)
                if len(history) < 45:
                    errors.append(f"{symbol}: usable history bars < 45 ({len(history)})")
                    continue
                latest_dates.append(pd.Timestamp(history["date"].iloc[-1]).strftime("%Y-%m-%d"))
                signals = detect_pattern(history)
                old = prior.get(symbol)
                row, changed = _build_symbol_record(symbol, NAME_FALLBACKS.get(symbol, symbol), signals, old, checked_at)
                updated_rows[symbol] = row
                if changed:
                    changes.append({
                        "symbol": symbol, "name": row["name"], "themes": row.get("themes", []),
                        "pattern_name": row["pattern_name"], "classification": row["classification"],
                        "event": row.get("major_change", ""), "last_bar_date": row.get("last_bar_date", ""),
                    })
                if index % 20 == 0 or index == len(symbols):
                    log.write(f"[{_now().isoformat(timespec='seconds')}] progress={index}/{len(symbols)} usable={len(updated_rows)} changes={len(changes)} errors={len(errors)}\n")
                time.sleep(0.06)
            except Exception as exc:
                errors.append(f"{symbol}: {type(exc).__name__}: {exc}")

    if not prior and updated_rows:
        changes = [{
            "symbol": row["symbol"], "name": row["name"], "themes": row.get("themes", []),
            "pattern_name": row.get("pattern_name", ""), "classification": row.get("classification", ""),
            "event": row.get("major_change", "初始型態建檔"), "last_bar_date": row.get("last_bar_date", ""),
        } for row in updated_rows.values()]

    if not changes:
        result = {
            "ok": True, "skipped": True,
            "reason": "沒有偵測到重大型態／趨勢狀態變化，保留既有公開快照，不建立 GitHub commit。",
            "checked_at": checked_at, "tracked_symbols": len(symbols), "usable_symbols": len(updated_rows),
            "errors_count": len(errors), "errors": errors[:20],
        }
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_now().isoformat(timespec='seconds')}] NO_SIGNIFICANT_CHANGE {json.dumps(result, ensure_ascii=False)}\n")
        return result

    data_asof = max(latest_dates) if latest_dates else now.date().isoformat()
    rows = sorted(updated_rows.values(), key=lambda row: (
        CATEGORY_ORDER.index(row.get("classification", "中性觀察")) if row.get("classification", "中性觀察") in CATEGORY_ORDER else 99,
        str(row.get("pattern_name", "")), str(row.get("symbol", "")),
    ))
    payload = {
        "schema_version": 1, "report_type": "trend_patterns_public",
        "updated_at": checked_at, "data_asof": data_asof,
        "check_mode": "daily_check_publish_on_material_change",
        "publication_status": "published" if publish else "local_preview",
        "tracked_symbols": len(symbols), "usable_symbols": len(updated_rows), "changed_count": len(changes),
        "category_order": CATEGORY_ORDER,
        "methodology": (
            "使用產業分析 ROTATION_THEMES 的全部概念股；所有分析 OHLC 都取 Fugle adjusted=True 還原股價，"
            "以避免除權息、股票股利與分割造成的價格基準斷層。最近 40 個交易日日 K 辨識頭肩、雙頂／雙底、圓底、杯柄、"
            "楔形、旗形與三角形，近 6 個月位置辨識由高點回落或由低點回升。若還原後仍出現巨大單日跳空，"
            "先歸入資料調整檢查、不直接判斷趨勢；只在型態／趨勢狀態重大變化時更新快照。"
        ),
        "pattern_guide_url": "https://www.ig.com/cn/trading-strategies/10-chart-patterns-every-trader-needs-to-know-220907",
        "changes": changes[:160], "errors_count": len(errors), "errors": errors[:30], "stocks": rows,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # 發布成功後才更新本機「上次已發布」快照。
    # 若 GitHub API／Token 暫時失敗，保留前一份本機快照，讓下次排程可以重新判定並重試發布。
    github = _publish_to_github(payload) if publish else None
    _atomic_write(public_path, serialized)
    return {
        "ok": True, "skipped": False, "updated_at": checked_at, "data_asof": data_asof,
        "tracked_symbols": len(symbols), "usable_symbols": len(updated_rows), "changed_count": len(changes),
        "public_file": str(public_path), "github": github, "errors_count": len(errors), "errors": errors[:20],
    }


def load_published_trend_patterns(base_dir: str | Path) -> dict[str, Any]:
    path = Path(base_dir).resolve() / PUBLIC_REPO_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
