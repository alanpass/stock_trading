# -*- coding: utf-8 -*-
"""台股市場交易日曆。

優先使用臺灣證券交易所官方「市場開休市日期」資料，並快取至本機。
目的：
1. 區分「平日」與「實際交易日」。
2. 讓週末、國定休市日仍可執行盤後研究。
3. 休市日不誤啟動即時行情 2 秒輪詢。
4. 休市日的價格資料明確回退至最近完成交易日。
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

TWSE_HOLIDAY_URL = "https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=json"
TAIPEI_TZ = ZoneInfo("Asia/Taipei")

# 2026 官方日曆的安全 fallback；若官方 API 暫時不可用，至少不會把已知休市日誤判成交易日。
# 程式啟動後仍會優先抓取官方資料並快取。
KNOWN_TWSE_CLOSED_2026 = {
    "2026-01-01": "中華民國開國紀念日",
    "2026-02-12": "市場無交易，僅辦理結算交割作業",
    "2026-02-13": "市場無交易，僅辦理結算交割作業",
    "2026-02-15": "農曆除夕及春節",
    "2026-02-16": "農曆除夕及春節",
    "2026-02-17": "農曆除夕及春節",
    "2026-02-18": "農曆除夕及春節",
    "2026-02-19": "農曆除夕及春節",
    "2026-02-20": "農曆春節補假",
    "2026-02-27": "和平紀念日補假",
    "2026-02-28": "和平紀念日",
    "2026-04-03": "兒童節及民族掃墓節補假",
    "2026-04-04": "兒童節及民族掃墓節",
    "2026-04-05": "兒童節及民族掃墓節",
    "2026-04-06": "兒童節及民族掃墓節補假",
    "2026-05-01": "勞動節",
    "2026-06-19": "端午節",
    "2026-09-25": "中秋節",
    "2026-09-28": "孔子誕辰紀念日/教師節",
    "2026-10-09": "國慶日補假",
    "2026-10-10": "國慶日",
    "2026-10-25": "臺灣光復暨金門古寧頭大捷紀念日",
    "2026-10-26": "臺灣光復暨金門古寧頭大捷紀念日補假",
    "2026-12-25": "行憲紀念日",
}


def _cache_path(base_dir: str | Path, year: int) -> Path:
    p = Path(base_dir).resolve() / "data" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"twse_holidays_{year}.json"


def _parse_rows(payload) -> dict[str, str]:
    out: dict[str, str] = {}
    rows = []
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("rows") or []
    elif isinstance(payload, list):
        rows = payload
    for row in rows:
        if not isinstance(row, (list, tuple, dict)):
            continue
        if isinstance(row, dict):
            values = list(row.values())
        else:
            values = list(row)
        if not values:
            continue
        raw_date = str(values[0]).strip()
        m = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", raw_date)
        if not m:
            continue
        d = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        name = str(values[1]).strip() if len(values) >= 2 else ""
        note = str(values[2]).strip() if len(values) >= 3 else ""
        # 「開始交易日」明確視為交易日，其餘資料列按官方名稱/說明判定為休市。
        if "開始交易" in name or "恢復交易" in name:
            continue
        out[d] = name or note or "市場休市"
    return out


def get_twse_closed_dates(base_dir: str | Path = ".", year: int | None = None, force: bool = False) -> dict[str, str]:
    year = int(year or datetime.now(TAIPEI_TZ).year)
    path = _cache_path(base_dir, year)
    if path.exists() and not force:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("closed_dates"):
                return {str(k): str(v) for k, v in payload["closed_dates"].items()}
        except Exception:
            pass

    closed = {}
    try:
        r = requests.get(TWSE_HOLIDAY_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0 TaiwanStockResearchAgent/1.0"})
        r.raise_for_status()
        closed = _parse_rows(r.json())
        # 保留指定年份資料，避免端點回傳其他年份。
        closed = {k: v for k, v in closed.items() if k.startswith(f"{year}-")}
        path.write_text(json.dumps({"year": year, "closed_dates": closed, "source": TWSE_HOLIDAY_URL}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        closed = {}

    if not closed and year == 2026:
        closed = dict(KNOWN_TWSE_CLOSED_2026)
        try:
            path.write_text(json.dumps({"year": year, "closed_dates": closed, "source": "official-fallback-2026"}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return closed


def is_twse_trading_day(value=None, base_dir: str | Path = ".") -> bool:
    if value is None:
        ts = pd.Timestamp.now(tz=TAIPEI_TZ)
    else:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(TAIPEI_TZ)
        else:
            ts = ts.tz_convert(TAIPEI_TZ)
    if ts.weekday() >= 5:
        return False
    closed = get_twse_closed_dates(base_dir, ts.year)
    return ts.strftime("%Y-%m-%d") not in closed


def market_status_text(value=None, base_dir: str | Path = ".") -> str:
    if value is None:
        ts = pd.Timestamp.now(tz=TAIPEI_TZ)
    else:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(TAIPEI_TZ)
        else:
            ts = ts.tz_convert(TAIPEI_TZ)
    ds = ts.strftime("%Y-%m-%d")
    if ts.weekday() >= 5:
        return f"{ds}｜非交易日：週末｜可執行研究；行情使用最近完成交易日資料"
    closed = get_twse_closed_dates(base_dir, ts.year)
    if ds in closed:
        return f"{ds}｜非交易日：{closed[ds]}｜可執行研究；行情使用最近完成交易日資料"
    t = ts.time()
    if t < pd.Timestamp("09:00").time():
        return f"{ds}｜交易日／盤前｜可執行研究；即時行情尚未開盤"
    if t <= pd.Timestamp("13:30").time():
        return f"{ds}｜交易日／交易時段｜即時行情可用"
    return f"{ds}｜交易日／盤後｜可執行研究；行情以今日收盤為主"
