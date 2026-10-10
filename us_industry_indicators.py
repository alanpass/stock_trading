# -*- coding: utf-8 -*-
"""美股產業指標股：台股供應鏈主題對照表與免 API Key 行情查詢。

行情使用 Yahoo Finance chart endpoint 的最新可取得資料；美股節假日／盤後時，
日期以來源回傳的美東 regularMarketTime 為準。部分台股子產業沒有完全相同的美股上市純標的，
因此明確標註為供應鏈代理股，不把它當成一對一同業。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import math
import time

import pandas as pd
import requests

US_MARKET_TZ = ZoneInfo("America/New_York")
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# 每個台股子產業對應 1～4 檔美股指標股。
# role 說明對照理由；「供應鏈代理」表示美國沒有完全一對一的上市純標的。
US_INDUSTRY_INDICATORS: dict[str, list[dict[str, str]]] = {
    "晶圓代工": [
        {"symbol": "TSM", "name": "台積電 ADR", "role": "晶圓代工直接對照"},
    ],
    "ASIC／矽智財": [
        {"symbol": "AVGO", "name": "Broadcom", "role": "客製化 ASIC／網路晶片"},
        {"symbol": "MRVL", "name": "Marvell Technology", "role": "資料中心 ASIC／高速互連"},
        {"symbol": "ARM", "name": "Arm Holdings", "role": "CPU IP／矽智財"},
        {"symbol": "NVDA", "name": "NVIDIA", "role": "AI 加速器需求端指標；非 ASIC／IP 純標的"},
    ],
    "BMC／伺服器管理晶片": [
        {"symbol": "SMCI", "name": "Super Micro Computer", "role": "伺服器系統供應鏈代理；非 BMC 純標的"},
        {"symbol": "DELL", "name": "Dell Technologies", "role": "伺服器硬體供應鏈代理；非 BMC 純標的"},
    ],
    "高速傳輸 IC": [
        {"symbol": "MRVL", "name": "Marvell Technology", "role": "高速 SerDes／資料中心互連"},
        {"symbol": "AVGO", "name": "Broadcom", "role": "交換器 ASIC／高速網路晶片"},
    ],
    "先進封裝設備": [
        {"symbol": "AMAT", "name": "Applied Materials", "role": "半導體製程設備"},
        {"symbol": "LRCX", "name": "Lam Research", "role": "蝕刻／沉積設備"},
        {"symbol": "KLAC", "name": "KLA", "role": "製程控制與量測設備"},
    ],
    "封裝測試": [
        {"symbol": "AMKR", "name": "Amkor Technology", "role": "封裝測試直接對照"},
        {"symbol": "ASX", "name": "ASE Technology ADR", "role": "日月光投控 ADR／封裝測試"},
    ],
    "測試介面／設備": [
        {"symbol": "TER", "name": "Teradyne", "role": "半導體自動測試設備"},
        {"symbol": "FORM", "name": "FormFactor", "role": "探針卡與晶圓測試介面"},
        {"symbol": "AEHR", "name": "Aehr Test Systems", "role": "半導體測試設備"},
    ],
    "檢測分析": [
        {"symbol": "KLAC", "name": "KLA", "role": "晶圓檢測／製程控制"},
        {"symbol": "CAMT", "name": "Camtek", "role": "先進封裝檢測與量測"},
        {"symbol": "ONTO", "name": "Onto Innovation", "role": "檢測／量測設備"},
    ],
    "記憶體／儲存": [
        {"symbol": "MU", "name": "Micron Technology", "role": "DRAM／HBM／NAND"},
        {"symbol": "WDC", "name": "Western Digital", "role": "資料儲存"},
        {"symbol": "STX", "name": "Seagate Technology", "role": "硬碟／儲存設備"},
    ],
    "ABF／IC 載板": [
        {"symbol": "AMKR", "name": "Amkor Technology", "role": "先進封裝供應鏈代理；非 ABF 載板純標的"},
        {"symbol": "TSM", "name": "台積電 ADR", "role": "先進封裝需求端代理；非載板純標的"},
    ],
    "CCL／銅箔基板": [
        {"symbol": "ROG", "name": "Rogers Corporation", "role": "高頻電路材料／層壓板"},
        {"symbol": "GLW", "name": "Corning", "role": "特種玻璃與連接材料供應鏈代理"},
    ],
    "玻纖布／銅箔": [
        {"symbol": "GLW", "name": "Corning", "role": "玻璃材料供應鏈代理"},
        {"symbol": "OC", "name": "Owens Corning", "role": "玻纖／複合材料供應鏈代理"},
    ],
    "PCB／伺服器板／HDI": [
        {"symbol": "CLS", "name": "Celestica", "role": "資料中心硬體與電子製造服務"},
        {"symbol": "JBL", "name": "Jabil", "role": "電子製造服務／伺服器硬體"},
        {"symbol": "SANM", "name": "Sanmina", "role": "高複雜度電子製造服務"},
    ],
    "散熱／液冷": [
        {"symbol": "VRT", "name": "Vertiv", "role": "資料中心冷卻與基礎設施"},
        {"symbol": "MOD", "name": "Modine Manufacturing", "role": "熱管理／資料中心冷卻"},
        {"symbol": "TT", "name": "Trane Technologies", "role": "暖通空調與冷卻系統"},
    ],
    "電源／BBU": [
        {"symbol": "ETN", "name": "Eaton", "role": "電力管理／備援電力"},
        {"symbol": "VRT", "name": "Vertiv", "role": "資料中心電源與 UPS"},
        {"symbol": "POWL", "name": "Powell Industries", "role": "電力配電設備"},
    ],
    "重電／電網": [
        {"symbol": "GEV", "name": "GE Vernova", "role": "電網／電力設備"},
        {"symbol": "ETN", "name": "Eaton", "role": "電力管理與配電"},
        {"symbol": "PWR", "name": "Quanta Services", "role": "電網工程與基礎建設"},
    ],
    "滑軌": [
        {"symbol": "DELL", "name": "Dell Technologies", "role": "伺服器機構件供應鏈代理；非滑軌純標的"},
        {"symbol": "SMCI", "name": "Super Micro Computer", "role": "伺服器機櫃／滑軌需求端代理"},
    ],
    "機殼／機櫃": [
        {"symbol": "SMCI", "name": "Super Micro Computer", "role": "伺服器機櫃與系統"},
        {"symbol": "DELL", "name": "Dell Technologies", "role": "伺服器系統與機構件需求端"},
        {"symbol": "HPE", "name": "Hewlett Packard Enterprise", "role": "企業伺服器／機櫃需求端"},
    ],
    "高速連接器／線材": [
        {"symbol": "APH", "name": "Amphenol", "role": "高速連接器與互連"},
        {"symbol": "TEL", "name": "TE Connectivity", "role": "連接器與線束"},
        {"symbol": "GLW", "name": "Corning", "role": "光纖互連"},
    ],
    "被動元件": [
        {"symbol": "VSH", "name": "Vishay Intertechnology", "role": "電阻／電容／分立元件"},
        {"symbol": "BELFB", "name": "Bel Fuse", "role": "磁性元件／電源與電子元件"},
        {"symbol": "CTS", "name": "CTS Corporation", "role": "電子元件與感測元件"},
    ],
    "組裝代工／ODM": [
        {"symbol": "CLS", "name": "Celestica", "role": "高階伺服器與網通設備製造"},
        {"symbol": "JBL", "name": "Jabil", "role": "電子製造服務"},
        {"symbol": "SANM", "name": "Sanmina", "role": "電子製造服務"},
    ],
    "網通／交換器": [
        {"symbol": "ANET", "name": "Arista Networks", "role": "資料中心交換器"},
        {"symbol": "CSCO", "name": "Cisco Systems", "role": "企業網路與交換器"},
        {"symbol": "CIEN", "name": "Ciena", "role": "光傳輸與網路設備"},
    ],
    "矽光子／光通訊": [
        {"symbol": "LITE", "name": "Lumentum", "role": "光通訊元件／雷射"},
        {"symbol": "COHR", "name": "Coherent", "role": "光學／光通訊元件"},
        {"symbol": "FN", "name": "Fabrinet", "role": "光電與光通訊製造服務"},
    ],
    "磊晶／化合物半導體": [
        {"symbol": "ON", "name": "onsemi", "role": "功率與感測半導體"},
        {"symbol": "MTSI", "name": "MACOM Technology Solutions", "role": "高頻／微波／光通訊半導體"},
        {"symbol": "COHR", "name": "Coherent", "role": "化合物半導體與光電材料"},
    ],
    "低軌衛星": [
        {"symbol": "RKLB", "name": "Rocket Lab", "role": "太空系統與發射服務"},
        {"symbol": "ASTS", "name": "AST SpaceMobile", "role": "衛星直連行動通訊"},
        {"symbol": "IRDM", "name": "Iridium Communications", "role": "衛星通訊服務代理"},
    ],
    "機器人／自動化": [
        {"symbol": "TER", "name": "Teradyne", "role": "工業協作機器人／自動化測試"},
        {"symbol": "ROK", "name": "Rockwell Automation", "role": "工業自動化"},
        {"symbol": "SYM", "name": "Symbotic", "role": "倉儲機器人與自動化"},
    ],
    "廠務工程": [
        {"symbol": "EME", "name": "EMCOR Group", "role": "機電工程與廠務服務"},
        {"symbol": "PWR", "name": "Quanta Services", "role": "電力與基礎建設工程"},
        {"symbol": "FIX", "name": "Comfort Systems USA", "role": "機電／暖通工程"},
    ],
}

def get_us_indicator_symbols(industry_names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """依所選台股產業列出不重複的美股代號；None 表示全部產業。"""
    names = industry_names if industry_names is not None else tuple(US_INDUSTRY_INDICATORS)
    return sorted({
        item["symbol"]
        for industry in names
        for item in US_INDUSTRY_INDICATORS.get(industry, [])
    })


def _num(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _fetch_one(symbol: str) -> dict[str, Any]:
    """查詢美股一年日線，供單日、5／20日報酬與相對強弱計算共用。"""
    from datetime import timezone

    url = YAHOO_CHART_URL.format(symbol=symbol)
    # 需要約 60 個交易日計算 50 日相對強弱均線與 10 日動能，取一年歷史。
    params = {"range": "1y", "interval": "1d", "events": "history"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    base: dict[str, Any] = {
        "symbol": symbol,
        "price": None,
        "change": None,
        "change_pct": None,
        "currency": "USD",
        "data_date": None,
        "volume": None,
        "history": [],
        "status": "行情暫不可用",
        "source": "Yahoo Finance chart",
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=(4, 12))
        response.raise_for_status()
        payload = response.json()
        chart = payload.get("chart", {}) if isinstance(payload, dict) else {}
        error = chart.get("error")
        result_list = chart.get("result") or []
        if error or not result_list:
            base["status"] = "查無行情"
            return base

        result = result_list[0]
        meta = result.get("meta") or {}
        quote_list = (result.get("indicators") or {}).get("quote") or [{}]
        quote = quote_list[0] if quote_list else {}
        timestamps = result.get("timestamp") or []
        close_values = quote.get("close") or []
        volume_values = quote.get("volume") or []

        # Yahoo 的日線時間戳以 UTC 日期為鍵，方便不同美股與 SPY 基準對齊。
        history_by_date: dict[str, dict[str, Any]] = {}
        for idx, timestamp in enumerate(timestamps):
            if idx >= len(close_values):
                continue
            ts = int(timestamp)
            close = _num(close_values[idx])
            if close is None or close <= 0:
                continue
            date_key = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            volume = _num(volume_values[idx]) if idx < len(volume_values) else None
            history_by_date[date_key] = {"date": date_key, "close": float(close), "volume": volume}

        price = _num(meta.get("regularMarketPrice"))
        latest_ts = meta.get("regularMarketTime")
        latest_valid = next(
            ((int(ts), _num(close)) for ts, close in reversed(list(zip(timestamps, close_values)))
             if _num(close) is not None and _num(close) > 0),
            None,
        )
        if price is None or price <= 0:
            if latest_valid is None:
                base["status"] = "沒有有效收盤價"
                return base
            latest_ts, price = latest_valid

        if latest_ts is None:
            latest_ts = latest_valid[0] if latest_valid else None
        if latest_ts is not None:
            latest_date = datetime.fromtimestamp(int(latest_ts), tz=US_MARKET_TZ).strftime("%Y-%m-%d")
            quote_session_date = datetime.fromtimestamp(int(latest_ts), tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            latest_date = None
            quote_session_date = None

        # 以最新可取得報價更新對應交易日；盤中尚未形成收盤 K 時也能納入最新價格。
        if quote_session_date and price is not None:
            history_by_date[quote_session_date] = {
                "date": quote_session_date,
                "close": float(price),
                "volume": _num(meta.get("regularMarketVolume")) or (
                    _num(volume_values[-1]) if volume_values else None
                ),
            }
        history = [history_by_date[key] for key in sorted(history_by_date)]
        closes = [float(item["close"]) for item in history if _num(item.get("close")) is not None]

        # 用最近一筆與前一筆交易日收盤／最新價格計算漲跌，避免盤中誤拿兩日前收盤價。
        prior_close = closes[-2] if len(closes) >= 2 else _num(meta.get("chartPreviousClose"))
        change = float(price) - prior_close if prior_close is not None and prior_close > 0 else None
        change_pct = (float(price) / prior_close - 1.0) * 100.0 if prior_close is not None and prior_close > 0 else None

        volumes = quote.get("volume") or []
        volume = _num(meta.get("regularMarketVolume"))
        if volume is None:
            volume = next((_num(v) for v in reversed(volumes) if _num(v) is not None), None)

        base.update({
            "price": float(price),
            "change": change,
            "change_pct": change_pct,
            "currency": str(meta.get("currency") or "USD"),
            "data_date": latest_date or (history[-1]["date"] if history else None),
            "volume": volume,
            "history": history,
            "status": "OK",
            "source": "Yahoo Finance chart",
        })
        return base
    except Exception as exc:
        base["status"] = f"取得失敗（{type(exc).__name__}）"
        return base

def fetch_us_indicators(symbols: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """平行取得多檔美股最新可用行情；單檔失敗不會中斷整個產業頁面。"""
    unique_symbols = list(dict.fromkeys(str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()))
    if not unique_symbols:
        return []
    records: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(unique_symbols))) as pool:
        futures = {pool.submit(_fetch_one, symbol): symbol for symbol in unique_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                records[symbol] = future.result()
            except Exception as exc:
                records[symbol] = {
                    "symbol": symbol, "price": None, "change": None, "change_pct": None,
                    "currency": "USD", "data_date": None, "volume": None, "history": [],
                    "status": f"取得失敗（{type(exc).__name__}）", "source": "Yahoo Finance chart",
                }
    return [records[symbol] for symbol in unique_symbols]
