"""個股籌碼訊號：外資／投信／主力連買連賣、融資融券增減、大戶與散戶持股變化。

資料來源（皆為公開資料，過去日期存到 data/cache/chips 或沿用三大法人日報快取，不會重複下載）：
- 外資、投信：TWSE T86（上市）、TPEx 三大法人日報（上櫃），一天一份全市場
- 融資融券：TWSE MI_MARGN（上市）、TPEx 融資融券餘額（上櫃），一天一份全市場
- 主力：富邦 e-broker 主力進出（前 15 大買超券商合計 − 前 15 大賣超券商合計，單位張），每檔每天一筆
- 大戶／散戶：TDCC 集保股權分散表每週查詢，本週對上週（大戶 ≥400 張、散戶 ≤50 張）

訊號顏色依台股慣例：看多紅、看空綠。融資增視為散戶看多、融券增視為看空。
"""
from __future__ import annotations

import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from market_calendar import is_twse_trading_day
from stock_api import _daily_report_text, clean_symbol

BASE_DIR = Path(__file__).resolve().parent
CACHE = BASE_DIR / "data" / "cache" / "chips"
CACHE.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
LOOKBACK_DAYS = 10          # 連續天數最多往回看 10 個交易日
STREAK_MIN = 3              # 連續 3 日以上才列為訊號
TDCC_MIN_CHANGE = 0.05      # 大戶／散戶持股比例週變化至少 0.05 個百分點才算

_lock = threading.Lock()
_mem: dict[str, object] = {}


def _num(v) -> float:
    try:
        return float(str(v).replace(",", "").strip() or "nan")
    except Exception:
        return float("nan")


def recent_trading_days(n: int = LOOKBACK_DAYS, today: date | None = None) -> list[date]:
    """最近 n 個交易日（新到舊）。今天的資料若還沒公布，各資料源會自動略過。"""
    d = today or pd.Timestamp.now(tz="Asia/Taipei").date()
    out = []
    while len(out) < n:
        if is_twse_trading_day(pd.Timestamp(d), BASE_DIR):
            out.append(d)
        d -= timedelta(days=1)
    return out


# ---------------- 全市場日報（一天一份，所有股票共用） ----------------
def _market_day(kind: str, day: date) -> dict[str, dict] | None:
    """回傳 {代號: {...}}；當天沒資料（未公布或休市）回傳 None。結果放記憶體，原始文字由 _daily_report_text 存磁碟。"""
    key = f"{kind}:{day:%Y%m%d}"
    with _lock:
        if key in _mem and (day < date.today() or time.time() - _mem[key][0] < 600):
            return _mem[key][1]
    rows: dict[str, dict] = {}
    try:
        if kind == "twse_insti":
            text = _daily_report_text("twse_t86", day, "https://www.twse.com.tw/rwd/zh/fund/T86",
                                      {"date": f"{day:%Y%m%d}", "selectType": "ALLBUT0999", "response": "json"}, HEADERS, 15)
            js = json.loads(text or "{}")
            f = js.get("fields") or []
            ix = {name: i for i, name in enumerate(f)}
            for r in js.get("data") or []:
                core = _num(r[ix["外陸資買賣超股數(不含外資自營商)"]]); fd = _num(r[ix.get("外資自營商買賣超股數", 0)])
                rows[clean_symbol(r[0])] = {"foreign": core + (fd if fd == fd else 0), "trust": _num(r[ix["投信買賣超股數"]])}
        elif kind == "tpex_insti":
            text = _daily_report_text("tpex_insti_v2", day, "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
                                      {"type": "Daily", "sect": "EW", "date": f"{day:%Y/%m/%d}", "response": "json"}, HEADERS, 15)
            js = json.loads(text or "{}")
            for r in ((js.get("tables") or [{}])[0].get("data") or []):
                # 欄位：外資(不含自營) 2-4、外資自營 5-7、外資合計 8-10、投信 11-13、自營商 14-22、三大法人合計 23
                rows[clean_symbol(r[0])] = {"foreign": _num(r[10]), "trust": _num(r[13])}
        elif kind == "twse_margin":
            text = _daily_report_text("twse_margin", day, "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
                                      {"date": f"{day:%Y%m%d}", "selectType": "ALL", "response": "json"}, HEADERS, 15)
            js = json.loads(text or "{}")
            tables = js.get("tables") or []
            table = next((t for t in tables if "彙總" in str(t.get("title", "")) and len(t.get("data") or []) > 50), None)
            for r in (table or {}).get("data") or []:
                # 融資：前日餘額 5、今日餘額 6；融券：前日餘額 11、今日餘額 12（單位張）
                rows[clean_symbol(r[0])] = {"margin": _num(r[6]) - _num(r[5]), "short": _num(r[12]) - _num(r[11]),
                                            "margin_bal": _num(r[6]), "short_bal": _num(r[12])}
        elif kind == "tpex_margin":
            text = _daily_report_text("tpex_margin", day, "https://www.tpex.org.tw/www/zh-tw/margin/balance",
                                      {"date": f"{day:%Y/%m/%d}", "response": "json"}, HEADERS, 15)
            js = json.loads(text or "{}")
            for r in ((js.get("tables") or [{}])[0].get("data") or []):
                # 前資餘額 2、資餘額 6、前券餘額 10、券餘額 14（單位張）
                rows[clean_symbol(r[0])] = {"margin": _num(r[6]) - _num(r[2]), "short": _num(r[14]) - _num(r[10]),
                                            "margin_bal": _num(r[6]), "short_bal": _num(r[14])}
    except Exception:
        rows = {}
    result = rows or None
    with _lock:
        _mem[key] = (time.time(), result)
    return result


def _series(symbol: str, kinds: tuple[str, str], fields: tuple[str, ...], days: list[date]) -> dict[str, list[tuple[date, float]]]:
    out: dict[str, list[tuple[date, float]]] = {f: [] for f in fields}
    for day in days:
        rec = None
        for kind in kinds:
            data = _market_day(kind, day)
            if data and symbol in data:
                rec = data[symbol]
                break
        if rec is None:
            continue
        for f in fields:
            out[f].append((day, rec.get(f, float("nan"))))
    return out


# ---------------- 主力進出（富邦 e-broker） ----------------
def _main_force_day(symbol: str, day: date) -> float | None:
    path = CACHE / f"main_{symbol}.json"
    with _lock:
        store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    k = f"{day:%Y-%m-%d}"
    if k in store and day < date.today():
        return store[k]
    try:
        r = requests.get("https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco.djhtm",
                         params={"a": symbol, "e": k, "f": k}, headers=HEADERS, timeout=15)
        html = r.content.decode("big5", "ignore")
    except Exception:
        return None
    m = re.search(r"最後更新日[:：]\s*([\d/]+)", html)
    if not m or m.group(1).replace("/", "-") != k:
        value = None                     # 當天沒有資料（未公布或休市）
    else:
        buy = sell = 0.0
        for row in re.findall(r"<TR>(.*?)</tr>", html, re.S | re.I):
            cells = [re.sub(r"<[^>]+>|,|&nbsp;", "", c).strip() for c in re.findall(r"<TD[^>]*>(.*?)</TD>", row, re.S | re.I)]
            if len(cells) == 10:
                b, s = _num(cells[3]), _num(cells[8])
                buy += b if b == b else 0
                sell += s if s == s else 0
        value = buy - sell
    if day < date.today() and value is not None:
        with _lock:
            store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            store[k] = value
            path.write_text(json.dumps(store), encoding="utf-8")
    return value


def _main_force_series(symbol: str, days: list[date]) -> list[tuple[date, float]]:
    out = []
    for day in days:                    # 從最新一天往回，方向一變就停（只需要算連續天數）
        v = _main_force_day(symbol, day)
        if v is None:
            continue
        out.append((day, v))
        if len(out) >= 2 and (v > 0) != (out[0][1] > 0):
            break
    return out


# ---------------- 集保大戶／散戶（TDCC 每週） ----------------
def _tdcc_week(symbol: str, sca_date: str, session: requests.Session, token: str) -> dict | None:
    path = CACHE / f"tdcc_{symbol}_{sca_date}.json"
    r = session.post("https://www.tdcc.com.tw/portal/zh/smWeb/qryStock", timeout=20, data={
        "SYNCHRONIZER_TOKEN": token, "SYNCHRONIZER_URI": "/portal/zh/smWeb/qryStock", "method": "submit",
        "firDate": sca_date, "scaDate": sca_date, "sqlMethod": "StockNo", "stockNo": symbol, "stockName": ""})
    table = next((t for t in pd.read_html(io.StringIO(r.text)) if t.shape[0] >= 15), None)
    if table is None:
        return None
    table.columns = ["level", "range", "holders", "shares", "pct"][: len(table.columns)]
    table["level"] = pd.to_numeric(table["level"], errors="coerce")
    pct = pd.to_numeric(table["pct"], errors="coerce")
    rec = {"date": sca_date, "big_pct": float(pct[table["level"].between(12, 15)].sum()),
           "retail_pct": float(pct[table["level"].between(1, 8)].sum())}
    path.write_text(json.dumps(rec), encoding="utf-8")
    return rec


def _tdcc_dates(session: requests.Session) -> tuple[list[str], str | None]:
    """TDCC 可查的週資料日期（新到舊）與查詢 token；日期清單記憶體快取 6 小時。"""
    page = session.get("https://www.tdcc.com.tw/portal/zh/smWeb/qryStock", timeout=20).text
    token = re.search(r'name="SYNCHRONIZER_TOKEN"[^>]*value="([^"]+)"', page)
    dates = re.findall(r'<option value="(\d{8})"', page)
    with _lock:
        _mem["tdcc_dates"] = (time.time(), dates)
    return dates, token.group(1) if token else None


def _tdcc_two_weeks(symbol: str) -> list[dict]:
    try:
        s = requests.Session(); s.headers.update(HEADERS)
        hit = _mem.get("tdcc_dates")
        dates = hit[1] if hit and time.time() - hit[0] < 6 * 3600 else _tdcc_dates(s)[0]
        weeks = []
        for d in dates[:2]:
            path = CACHE / f"tdcc_{symbol}_{d}.json"
            if path.exists():
                weeks.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            _, token = _tdcc_dates(s)            # 每次送出查詢都要新的 token
            rec = _tdcc_week(symbol, d, s, token) if token else None
            if rec:
                weeks.append(rec)
        return weeks if len(weeks) == 2 else []
    except Exception:
        return []


# ---------------- 組合訊號 ----------------
def _streak(series: list[tuple[date, float]]) -> int:
    """從最新一天起算同方向連續天數；正數＝連買／連增，負數＝連賣／連減。"""
    vals = [v for _, v in series if v == v]
    if not vals or vals[0] == 0:
        return 0
    sign = 1 if vals[0] > 0 else -1
    n = 0
    for v in vals:
        if v * sign > 0:
            n += 1
        else:
            break
    return n * sign


def compute(symbol: str) -> dict:
    symbol = clean_symbol(symbol)
    days = recent_trading_days()
    insti = _series(symbol, ("twse_insti", "tpex_insti"), ("foreign", "trust"), days)
    margin = _series(symbol, ("twse_margin", "tpex_margin"), ("margin", "short"), days)
    main = _main_force_series(symbol, days)
    weeks = _tdcc_two_weeks(symbol)

    rows = []       # 給個股頁的完整表格
    signals = []    # 達門檻的訊號（自選股旁與個股頁的標籤）

    def add_flow(key: str, label: str, series, unit_div: float, unit: str, up_word: str, down_word: str, bull_when_up: bool):
        s = _streak(series)
        latest = series[0][1] / unit_div if series else None
        rows.append({"key": key, "label": label, "streak": s, "latest": latest, "unit": unit,
                     "asof": series[0][0].isoformat() if series else None})
        if abs(s) >= STREAK_MIN:
            up = s > 0
            more = "以上" if abs(s) >= len([v for _, v in series if v == v]) else ""   # 整段回看期間都同方向，實際可能更久
            signals.append({"key": key, "text": f"{label}{'連' + (up_word if up else down_word)}{abs(s)}日{more}",
                            "tone": "bull" if up == bull_when_up else "bear"})

    add_flow("foreign", "外資", insti["foreign"], 1000, "張", "買", "賣", True)
    add_flow("trust", "投信", insti["trust"], 1000, "張", "買", "賣", True)
    add_flow("main", "主力", main, 1, "張", "買", "賣", True)
    add_flow("margin", "融資", margin["margin"], 1, "張", "增", "減", True)
    add_flow("short", "融券", margin["short"], 1, "張", "增", "減", False)

    holder = None
    if len(weeks) == 2:
        now, prev = weeks
        d_big, d_retail = now["big_pct"] - prev["big_pct"], now["retail_pct"] - prev["retail_pct"]
        holder = {"asof": now["date"], "prev": prev["date"], "big_pct": now["big_pct"], "retail_pct": now["retail_pct"],
                  "d_big": d_big, "d_retail": d_retail}
        if d_big >= TDCC_MIN_CHANGE and d_retail <= -TDCC_MIN_CHANGE:
            signals.append({"key": "holder", "text": "大戶買散戶賣", "tone": "bull"})
        elif d_big <= -TDCC_MIN_CHANGE and d_retail >= TDCC_MIN_CHANGE:
            signals.append({"key": "holder", "text": "大戶賣散戶買", "tone": "bear"})
    return {"symbol": symbol, "rows": rows, "holder": holder, "signals": signals, "computed_at": time.time()}


# ---------------- 背景計算（自選股不擋畫面） ----------------
_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="chips")
_jobs: dict[str, object] = {}
_results: dict[str, dict] = {}
RESULT_TTL = 30 * 60


def get_nonblocking(symbols: list[str]) -> dict[str, dict | None]:
    """已算好的直接回傳；沒算過或超過 30 分鐘的丟到背景計算，先回 None。"""
    fresh = lambda r: r is not None and time.time() - r["computed_at"] < RESULT_TTL
    out = {}
    for sym in map(clean_symbol, symbols):
        res = _results.get(sym)
        job = _jobs.get(sym)
        if not fresh(res) and job is not None and job.done():
            _jobs.pop(sym, None)
            if job.exception() is None:
                res = _results[sym] = job.result()
        if not fresh(res) and sym not in _jobs:
            _jobs[sym] = _pool.submit(compute, sym)
        out[sym] = res          # 背景還在算時回傳舊結果或 None
    return out


def get_blocking(symbol: str, timeout: float = 60) -> dict | None:
    sym = clean_symbol(symbol)
    res = get_nonblocking([sym])[sym]
    if res is not None and time.time() - res["computed_at"] < RESULT_TTL:
        return res
    try:
        _results[sym] = _jobs[sym].result(timeout=timeout)
        return _results[sym]
    except Exception:
        return res
