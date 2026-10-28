"""個股符合特徵：把一檔股票目前符合的條件整理成「消息面／籌碼面／價量面／基本面／技術面」五類標籤。

全市場資料一天一份、所有股票共用（過去日期由 stock_api._daily_report_text 存到磁碟，不會重複下載）：
- 收盤行情：TWSE MI_INDEX、TPEx dailyQuotes → 熱門股（成交值排名）
- 三大法人：TWSE T86、TPEx 三大法人日報 → 外資／投信／自營商熱門買賣超（買賣超金額排名）
- 當沖：TWSE TWTB4U、TPEx 現股當沖統計 → 當沖熱門、高當沖比
- 融資：TWSE MI_MARGN（餘額÷限額）、TPEx 融資使用率 → 高融資
- 外資持股：TWSE MI_QFIIS、TPEx 外資持股比率 → 外資高持股
- 基本面（TWSE／TPEx OpenAPI 最新一期）：本益比、殖利率、股價淨值比、月營收年增率、實收資本額
- 處置股、注意股公告（TWSE／TPEx OpenAPI）
個股自己的資料：日 K（由呼叫端傳入）、Yahoo Finance 歷年配息、chip_signals 的連買連賣與集保大戶散戶。

規則都是固定門檻，寫在下面的常數與每個標籤的「判斷依據」裡，網站上可以直接看到。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from chip_signals import recent_trading_days
from stock_api import _daily_report_text, clean_symbol

BASE_DIR = Path(__file__).resolve().parent
CACHE = BASE_DIR / "data" / "cache" / "features"
CACHE.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
TWSE_OPENAPI = "https://openapi.twse.com.tw/v1"
TPEX_OPENAPI = "https://www.tpex.org.tw/openapi/v1"

CATEGORIES = ["消息面", "籌碼面", "價量面", "基本面", "技術面"]

HOT_RANK = 30               # 全市場排名前 30 名才算「熱門」
DAYTRADE_RATIO = 0.40       # 當沖成交股數占成交量 ≥ 40%
MARGIN_USAGE = 0.25         # 融資使用率 ≥ 25%
FOREIGN_HOLD = 30.0         # 外資持股比率 ≥ 30%
MAIN_FORCE_SHARE = 0.10     # 主力買賣超 ≥ 當日成交量 10%
HOLDER_CHANGE = 0.20        # 集保大戶／散戶持股比例週變化 ≥ 0.2 個百分點
WEEK_STRONG = 7.0           # 5 日漲跌幅 ≥ ±7%
MONTH_STRONG = 15.0         # 20 日漲跌幅 ≥ ±15%
VOLUME_SURGE = 2.0          # 成交量 ≥ 20 日均量 2 倍
VOLUME_DRY = 0.5            # 成交量 ≤ 20 日均量一半
REVENUE_GROWTH = 20.0       # 月營收年增率 ≥ ±20%
PE_HIGH, PE_LOW = 30.0, 12.0
YIELD_HIGH = 5.0
BIG_CAPITAL = 100e8         # 實收資本額 ≥ 100 億
SMALL_CAPITAL = 10e8        # 實收資本額 ≤ 10 億
DIVIDEND_YEARS = 5          # 最近 5 個完整年度每年都有配息

_lock = threading.Lock()
_mem: dict[str, tuple[float, object]] = {}


def _num(v) -> float:
    try:
        s = str(v).replace(",", "").replace("%", "").strip()
        return float(s) if s not in ("", "--", "-", "N/A") else float("nan")
    except Exception:
        return float("nan")


def _is_stock(code: str) -> bool:
    """一般股票（四碼、非 0 開頭的 ETF）；熱門排名只在股票之間比較。"""
    return len(code) == 4 and code.isdigit() and not code.startswith("0")


def _memo(key: str, ttl: float, fn):
    with _lock:
        hit = _mem.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _lock:
        _mem[key] = (time.time(), value)
    return value


def _rwd_tables(kind: str, day: date, url: str, params: dict) -> list[dict]:
    text = _daily_report_text(kind, day, url, params, HEADERS, 30)
    try:
        js = json.loads(text or "{}")
    except Exception:
        return []
    if "tables" in js:
        return js.get("tables") or []
    return [{"fields": js.get("fields"), "data": js.get("data")}] if js.get("data") else []


def _openapi(name: str, url: str) -> list[dict]:
    """OpenAPI 只提供最新一期；成功就存一份到磁碟，連不上時沿用上次的資料。"""
    def load():
        path = CACHE / f"openapi_{name}.json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
        except Exception:
            pass
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return _memo(f"openapi:{name}", 3600, load)


# ---------------- 全市場日資料 ----------------
def _quotes(day: date) -> dict[str, dict]:
    out = {}
    for t in _rwd_tables("twse_mi_index", day, "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
                         {"date": f"{day:%Y%m%d}", "type": "ALLBUT0999", "response": "json"}):
        f = t.get("fields") or []
        if "證券代號" in f and "收盤價" in f:
            ix = {n: i for i, n in enumerate(f)}
            for r in t.get("data") or []:
                out[clean_symbol(r[0])] = {"close": _num(r[ix["收盤價"]]), "volume": _num(r[ix["成交股數"]]),
                                           "value": _num(r[ix["成交金額"]])}
    for t in _rwd_tables("tpex_quotes", day, "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
                         {"date": f"{day:%Y/%m/%d}", "response": "json"}):
        f = t.get("fields") or []
        if "代號" in f and "收盤" in f:
            ix = {n: i for i, n in enumerate(f)}
            for r in t.get("data") or []:
                out[clean_symbol(r[0])] = {"close": _num(r[ix["收盤"]]), "volume": _num(r[ix["成交股數"]]),
                                           "value": _num(r[ix["成交金額(元)"]])}
    return out


def _institutions(day: date) -> dict[str, dict]:
    out = {}
    for t in _rwd_tables("twse_t86", day, "https://www.twse.com.tw/rwd/zh/fund/T86",
                         {"date": f"{day:%Y%m%d}", "selectType": "ALLBUT0999", "response": "json"}):
        f = t.get("fields") or []
        if "投信買賣超股數" not in f:
            continue
        ix = {n: i for i, n in enumerate(f)}
        for r in t.get("data") or []:
            fd = _num(r[ix["外資自營商買賣超股數"]])
            out[clean_symbol(r[0])] = {"foreign": _num(r[ix["外陸資買賣超股數(不含外資自營商)"]]) + (fd if fd == fd else 0),
                                       "trust": _num(r[ix["投信買賣超股數"]]), "dealer": _num(r[ix["自營商買賣超股數"]])}
    for t in _rwd_tables("tpex_insti_v2", day, "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
                         {"type": "Daily", "sect": "EW", "date": f"{day:%Y/%m/%d}", "response": "json"})[:1]:
        for r in t.get("data") or []:
            # 外資合計 10、投信 13、自營商合計 22（與 chip_signals 相同的欄位位置）
            out[clean_symbol(r[0])] = {"foreign": _num(r[10]), "trust": _num(r[13]), "dealer": _num(r[22])}
    return out


def _daytrade(day: date) -> dict[str, float]:
    out = {}
    tables = _rwd_tables("twse_twtb4u", day, "https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U",
                         {"date": f"{day:%Y%m%d}", "selectType": "All", "response": "json"})
    try:   # 上櫃的查詢要用 POST，自己存檔
        path = CACHE / f"tpex_daytrade_{day:%Y%m%d}.json"
        if path.exists():
            tp = json.loads(path.read_text(encoding="utf-8"))
        else:
            tp = requests.post("https://www.tpex.org.tw/www/zh-tw/intraday/stat", headers=HEADERS, timeout=30,
                               data={"date": f"{day:%Y/%m/%d}", "type": "Daily", "response": "json"}).json()
            if tp.get("tables") and day < date.today():
                path.write_text(json.dumps(tp, ensure_ascii=False), encoding="utf-8")
        tables = tables + (tp.get("tables") or [])
    except Exception:
        pass
    for t in tables:
        f = t.get("fields") or []
        if "當日沖銷交易成交股數" in f:
            i = f.index("當日沖銷交易成交股數")
            for r in t.get("data") or []:
                out[clean_symbol(r[0])] = _num(r[i])
    return out


def _margin_usage(day: date) -> dict[str, float]:
    """融資使用率（%）。上市＝今日餘額÷次一營業日限額；上櫃直接用櫃買中心公布的使用率（最新一期）。"""
    out = {}
    for t in _rwd_tables("twse_margin", day, "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
                         {"date": f"{day:%Y%m%d}", "selectType": "ALL", "response": "json"}):
        if "彙總" not in str(t.get("title", "")) or len(t.get("data") or []) < 50:
            continue
        for r in t["data"]:
            bal, quota = _num(r[6]), _num(r[7])      # 融資今日餘額、次一營業日限額（張）
            if quota and quota == quota and quota > 0:
                out[clean_symbol(r[0])] = bal / quota * 100
    for r in _openapi("tpex_margin", f"{TPEX_OPENAPI}/tpex_mainboard_margin_balance"):
        out[clean_symbol(r.get("SecuritiesCompanyCode", ""))] = _num(r.get("MarginPurchaseUtilizationRate"))
    return out


def _foreign_hold(day: date) -> dict[str, float]:
    out = {}
    for t in _rwd_tables("twse_qfiis", day, "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS",
                         {"date": f"{day:%Y%m%d}", "selectType": "ALLBUT0999", "response": "json"}):
        f = t.get("fields") or []
        if "全體外資及陸資持股比率" in f:
            i = f.index("全體外資及陸資持股比率")
            for r in t.get("data") or []:
                out[clean_symbol(r[0])] = _num(r[i])
    for r in _openapi("tpex_qfii", f"{TPEX_OPENAPI}/tpex_3insti_qfii"):
        out[clean_symbol(r.get("SecuritiesCompanyCode", ""))] = _num(r.get("PercentageOfSharesOC/FMIHeld"))
    return out


def _rank_sets(values: dict[str, float], n: int = HOT_RANK) -> tuple[set, set]:
    """前 n 大（正）與前 n 小（負）的股票代號。"""
    s = pd.Series({k: v for k, v in values.items() if _is_stock(k) and v == v})
    return set(s[s > 0].nlargest(n).index), set(s[s < 0].nsmallest(n).index)


def market_snapshot() -> dict:
    """最近一個有收盤資料的交易日的全市場特徵資料；同一天只算一次，記憶體保留 30 分鐘。"""
    def build():
        day = None
        quotes = {}
        for d in recent_trading_days(6):
            quotes = _quotes(d)
            if len(quotes) > 500:
                day = d
                break
        if day is None:
            return {"day": None}
        insti = _institutions(day)
        dt = _daytrade(day)
        snap = {"day": day, "quotes": quotes, "insti": insti, "daytrade": dt,
                "margin_usage": _margin_usage(day), "foreign_hold": _foreign_hold(day)}
        snap["hot_value"] = _rank_sets({k: v["value"] for k, v in quotes.items()})[0]
        for who in ("foreign", "trust", "dealer"):
            amount = {k: v[who] * quotes[k]["close"] for k, v in insti.items() if k in quotes and v[who] == v[who]}
            snap[f"hot_{who}"] = _rank_sets(amount)
        snap["hot_daytrade"] = _rank_sets({k: v * quotes[k]["close"] for k, v in dt.items() if k in quotes})[0]
        return snap
    return _memo("snapshot", 1800, build)


def fundamentals() -> dict[str, dict]:
    """本益比、殖利率、股價淨值比、月營收年增率、實收資本額、處置／注意（OpenAPI 最新一期）。"""
    def build():
        out: dict[str, dict] = {}
        put = lambda code, **kw: out.setdefault(clean_symbol(code), {}).update(kw)
        for r in _openapi("twse_bwibbu", f"{TWSE_OPENAPI}/exchangeReport/BWIBBU_ALL"):
            put(r.get("Code", ""), pe=_num(r.get("PEratio")), yld=_num(r.get("DividendYield")), pb=_num(r.get("PBratio")))
        for r in _openapi("tpex_pe", f"{TPEX_OPENAPI}/tpex_mainboard_peratio_analysis"):
            put(r.get("SecuritiesCompanyCode", ""), pe=_num(r.get("PriceEarningRatio")), yld=_num(r.get("YieldRatio")),
                pb=_num(r.get("PriceBookRatio")))
        for name, url in (("twse_rev", f"{TWSE_OPENAPI}/opendata/t187ap05_L"), ("tpex_rev", f"{TPEX_OPENAPI}/mopsfin_t187ap05_O")):
            for r in _openapi(name, url):
                ym = str(r.get("資料年月", ""))
                put(r.get("公司代號", ""), rev_yoy=_num(r.get("營業收入-去年同月增減(%)")),
                    rev_month=f"{int(ym[:-2]) + 1911}/{ym[-2:]}" if len(ym) >= 5 else "")
        for r in _openapi("twse_basic", f"{TWSE_OPENAPI}/opendata/t187ap03_L"):
            put(r.get("公司代號", ""), capital=_num(r.get("實收資本額")))
        for r in _openapi("tpex_basic", f"{TPEX_OPENAPI}/mopsfin_t187ap03_O"):
            put(r.get("SecuritiesCompanyCode", ""), capital=_num(r.get("Paidin.Capital.NTDollars")))
        for r in _openapi("twse_punish", f"{TWSE_OPENAPI}/announcement/punish"):
            if r.get("Code"):
                put(r["Code"], disposal=f'{r.get("DispositionPeriod", "")}（{r.get("ReasonsOfDisposition", "")}）')
        for r in _openapi("tpex_punish", f"{TPEX_OPENAPI}/tpex_disposal_information"):
            if r.get("SecuritiesCompanyCode"):
                put(r["SecuritiesCompanyCode"], disposal=f'{r.get("DispositionPeriod", "")}（{r.get("DispositionReasons", "")}）')
        for r in _openapi("twse_notice", f"{TWSE_OPENAPI}/announcement/notice"):
            if r.get("Code"):
                put(r["Code"], notice=str(r.get("TradingInfoForAttention", ""))[:80])
        for r in _openapi("tpex_notice", f"{TPEX_OPENAPI}/tpex_trading_warning_information"):
            if r.get("SecuritiesCompanyCode"):
                put(r["SecuritiesCompanyCode"], notice=str(r.get("TradingInformation", ""))[:80])
        return out
    return _memo("fundamentals", 3600, build)


def dividend_years(symbol: str) -> list[int]:
    """Yahoo Finance 歷年有配息的年度；磁碟快取 7 天。"""
    path = CACHE / f"div_{symbol}.json"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - rec["t"] < 7 * 86400:
            return rec["years"]
    except Exception:
        pass
    years: set[int] = set()
    for suffix in (".TW", ".TWO"):
        try:
            js = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}{suffix}", headers=HEADERS, timeout=15,
                              params={"range": "10y", "interval": "1mo", "events": "div"}).json()
            res = (js.get("chart", {}).get("result") or [None])[0]
        except Exception:
            continue
        if res:
            for ev in (res.get("events", {}).get("dividends") or {}).values():
                years.add(datetime.fromtimestamp(ev["date"]).year)
            break
    out = sorted(years)
    path.write_text(json.dumps({"t": time.time(), "years": out}), encoding="utf-8")
    return out


# ---------------- 個股標籤 ----------------
def _tag(text: str, why: str, tone: str = "flat") -> dict:
    """tone：bull 偏多、bear 偏空、flat 中性（網站上看多紅、看空綠）。"""
    return {"text": text, "why": why, "tone": tone}


FLOW_TOP = 5                # 產業資金增加／減少前 5 名特別顯示


def rotation_flow_ranks(rotation: dict | None, kind: str = "groups") -> list[dict]:
    """產業依「成交金額較前一交易日增減」排序（大到小）；同一批股票比較，不除以全市場總額。
    kind：groups＝產業輪動主題（代表股）、official_industries＝官方產業別（上市櫃全部股票）。"""
    rows = [g for g in (rotation or {}).get(kind) or [] if g.get("turnover_change") is not None]
    rows.sort(key=lambda g: g["turnover_change"], reverse=True)
    up = [g for g in rows if g["turnover_change"] > 0]
    down = sorted((g for g in rows if g["turnover_change"] < 0), key=lambda g: g["turnover_change"])
    out = []
    for g in rows:
        side = "up" if g["turnover_change"] > 0 else "down" if g["turnover_change"] < 0 else "flat"
        rank = (up.index(g) + 1) if side == "up" else (down.index(g) + 1) if side == "down" else None
        out.append({"name": g["name"], "change": g["turnover_change"], "pct": g.get("turnover_change_pct"),
                    "turnover": g.get("turnover"), "side": side, "rank": rank,
                    "top": rank is not None and rank <= FLOW_TOP,
                    "members": [str(c.get("symbol") or c.get("code") or "") for c in g.get("components") or []]
                    if "components" in g else [str(c) for c in g.get("members") or []]})
    return out


def _rotation_flow(symbol: str, rotation: dict | None) -> list[dict]:
    """個股所屬產業主題的資金增減標籤；前 5 名加 top 標記（網站上實心粗體）。"""
    ranks = rotation_flow_ranks(rotation)
    if not any(symbol in r["members"] for r in ranks):
        # 不在任何主題代表股裡：改用官方產業別（排名也是在官方產業之間比）
        ranks = rotation_flow_ranks(rotation, "official_industries")
    asof = (rotation or {}).get("data_asof") or ""
    prev = (rotation or {}).get("previous_data_asof") or ""
    span = f"{prev[5:].replace('-', '/')} → {asof[5:].replace('-', '/')}" if asof and prev else "較前一交易日"
    tags = []
    for r in ranks:
        if symbol not in r["members"] or r["side"] == "flat":
            continue
        up = r["side"] == "up"
        pct = f"（{r['pct']:+.1f}%）" if r.get("pct") is not None else ""
        why = (f"{r['name']}成分股成交金額{'增加' if up else '減少'} {abs(r['change']) / 1e8:,.1f} 億{pct}，"
               f"資金{'增加' if up else '減少'}第 {r['rank']} 名（{span}）")
        if r["top"]:
            text = f"{r['name']} 資金{'流入' if up else '流出'}前{FLOW_TOP}"
        else:
            text = f"{r['name']} 資金{'增加' if up else '減少'}"
        tag = _tag(text, why, "bull" if up else "bear")
        tag["top"] = r["top"]
        tags.append(tag)
    return tags


def _technical(hist: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """回傳（價量面、技術面）。hist 需含 date、high、low、close、volume，舊到新。"""
    pv, tech = [], []
    if hist is None or len(hist) < 25:
        return pv, tech
    h = hist.sort_values("date").reset_index(drop=True)
    c, v = h["close"].astype(float), h["volume"].astype(float)
    last = c.iloc[-1]
    if len(c) > 5:
        r5 = (last / c.iloc[-6] - 1) * 100
        if r5 >= WEEK_STRONG:
            pv.append(_tag("週強勢", f"近 5 日上漲 {r5:.1f}%（門檻 +{WEEK_STRONG:.0f}%）", "bull"))
        elif r5 <= -WEEK_STRONG:
            pv.append(_tag("週弱勢", f"近 5 日下跌 {abs(r5):.1f}%（門檻 −{WEEK_STRONG:.0f}%）", "bear"))
    if len(c) > 20:
        r20 = (last / c.iloc[-21] - 1) * 100
        if r20 >= MONTH_STRONG:
            pv.append(_tag("月強勢", f"近 20 日上漲 {r20:.1f}%（門檻 +{MONTH_STRONG:.0f}%）", "bull"))
        elif r20 <= -MONTH_STRONG:
            pv.append(_tag("月弱勢", f"近 20 日下跌 {abs(r20):.1f}%（門檻 −{MONTH_STRONG:.0f}%）", "bear"))
    for n, word in ((240, "一年"), (60, "季")):
        if len(c) >= n:
            if h["high"].iloc[-1] >= h["high"].iloc[-n:].max():
                pv.append(_tag(f"價創{word}新高", f"今日最高價 {h['high'].iloc[-1]:,.2f} 是近 {n} 個交易日最高", "bull"))
                break
            if h["low"].iloc[-1] <= h["low"].iloc[-n:].min():
                pv.append(_tag(f"價創{word}新低", f"今日最低價 {h['low'].iloc[-1]:,.2f} 是近 {n} 個交易日最低", "bear"))
                break
    avg20 = v.iloc[-21:-1].mean()
    if avg20 > 0:
        ratio = v.iloc[-1] / avg20
        if ratio >= VOLUME_SURGE:
            pv.append(_tag("爆量", f"成交量是 20 日均量的 {ratio:.1f} 倍（門檻 {VOLUME_SURGE:.0f} 倍）"))
        elif ratio <= VOLUME_DRY:
            pv.append(_tag("量縮", f"成交量只有 20 日均量的 {ratio:.0%}（門檻 {VOLUME_DRY:.0%}）"))

    mas = {n: c.rolling(n).mean().iloc[-1] for n in (5, 20, 60) if len(c) >= n}
    for n, word in ((5, "週線"), (20, "月線"), (60, "季線")):
        if n in mas:
            above = last >= mas[n]
            tech.append(_tag(f"{word}{'上' if above else '下'}", f"收盤 {last:,.2f} {'高於' if above else '低於'} {n} 日均線 {mas[n]:,.2f}",
                             "bull" if above else "bear"))
    if len(mas) == 3:
        if mas[5] > mas[20] > mas[60]:
            tech.append(_tag("均線多頭排列", f"5 日 {mas[5]:,.2f} > 20 日 {mas[20]:,.2f} > 60 日 {mas[60]:,.2f}", "bull"))
        elif mas[5] < mas[20] < mas[60]:
            tech.append(_tag("均線空頭排列", f"5 日 {mas[5]:,.2f} < 20 日 {mas[20]:,.2f} < 60 日 {mas[60]:,.2f}", "bear"))
    # KD（9 日 RSV，K、D 以 1/3 平滑）
    low9, high9 = h["low"].rolling(9).min(), h["high"].rolling(9).max()
    rsv = ((c - low9) / (high9 - low9).replace(0, float("nan")) * 100).fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    if k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] > d.iloc[-1]:
        tech.append(_tag("KD 黃金交叉", f"K {k.iloc[-1]:.1f} 由下往上穿過 D {d.iloc[-1]:.1f}", "bull"))
    elif k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] < d.iloc[-1]:
        tech.append(_tag("KD 死亡交叉", f"K {k.iloc[-1]:.1f} 由上往下穿過 D {d.iloc[-1]:.1f}", "bear"))
    if k.iloc[-1] >= 80:
        tech.append(_tag("KD 高檔", f"K 值 {k.iloc[-1]:.1f}（≥ 80）"))
    elif k.iloc[-1] <= 20:
        tech.append(_tag("KD 低檔", f"K 值 {k.iloc[-1]:.1f}（≤ 20）"))
    return pv, tech


def holder_since_tag(since: dict) -> tuple[str, str, str]:
    """集保公布後到最新交易日的每日估計：主力買賣超（大戶）、融資增減（散戶）。回傳 (標籤, 依據, 色調)。"""
    main, margin = since.get("main"), since.get("margin")
    span = f"{since['start'][5:].replace('-', '/')}–{since['end'][5:].replace('-', '/')}，{since['days']} 日"
    parts = []
    if main is not None:
        parts.append(f"主力{'買' if main >= 0 else '賣'}超 {abs(main):,.0f} 張")
    if margin is not None:
        parts.append(f"融資{'增' if margin >= 0 else '減'} {abs(margin):,.0f} 張")
    big_buy = main is not None and main > 0
    big_sell = main is not None and main < 0
    retail_in = margin is not None and margin > 0
    retail_out = margin is not None and margin < 0
    if big_buy and not retail_in:
        text, tone = "集保後大戶續買", "bull"
    elif big_sell and not retail_out:
        text, tone = "集保後大戶續賣", "bear"
    else:
        text, tone = "集保後籌碼（每日估計）", "flat"
    return text, f"{'、'.join(parts)}（{span}；集保下次公布前以每日資料估計）", tone


def _news(symbol: str, name: str, news: list[dict], earnings: list[dict], fund: dict) -> list[dict]:
    out = []
    if fund.get("disposal"):
        out.append(_tag("處置股", f"處置期間 {fund['disposal']}", "bear"))
    if fund.get("notice"):
        out.append(_tag("注意股", fund["notice"], "bear"))
    for e in earnings or []:
        if clean_symbol(str(e.get("symbol", ""))) == symbol:
            out.append(_tag("近期法說會", f"{e.get('published_date') or e.get('event_date') or ''} Fugle 法說會備忘錄"))
            break
    keys = [f"({symbol}-TW)", f"({symbol}-TWO)", f"（{symbol}）"] + ([name] if name and len(name) >= 2 else [])
    hits = [n for n in news or [] if any(k in f"{n.get('title', '')} {n.get('summary', '')}" for k in keys)]
    if hits:
        out.append(_tag("新聞焦點", f"近 2 日鉅亨新聞提到 {len(hits)} 則：{hits[0].get('title', '')[:40]}"))
    return out


def compute(symbol: str, hist: pd.DataFrame | None = None, name: str = "", news: list | None = None,
            earnings: list | None = None, chips: dict | None = None, rotation: dict | None = None) -> dict:
    """hist：日 K（呼叫端提供，避免重複下載）；chips：chip_signals.compute 的結果（可省略）。"""
    symbol = clean_symbol(symbol)
    snap = market_snapshot()
    fund = fundamentals().get(symbol, {})
    groups: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    groups["消息面"] = _news(symbol, name, news, earnings, fund)

    # ---- 籌碼面 ----
    chip = groups["籌碼面"]
    day = snap.get("day")
    q = (snap.get("quotes") or {}).get(symbol, {})
    ins = (snap.get("insti") or {}).get(symbol, {})
    for who, label in (("foreign", "外資"), ("trust", "投信"), ("dealer", "自營商")):
        buy, sell = snap.get(f"hot_{who}", (set(), set()))
        net = ins.get(who, float("nan")) / 1000
        if symbol in buy:
            chip.append(_tag(f"{label}熱門買超", f"買超 {net:,.0f} 張，買超金額全市場前 {HOT_RANK} 名", "bull"))
        elif symbol in sell:
            chip.append(_tag(f"{label}熱門賣超", f"賣超 {abs(net):,.0f} 張，賣超金額全市場前 {HOT_RANK} 名", "bear"))
    for r in (chips or {}).get("rows") or []:
        if r["key"] == "main" and r.get("latest") is not None and r["latest"] == r["latest"] and q.get("volume"):
            share = r["latest"] * 1000 / q["volume"]
            if abs(share) >= MAIN_FORCE_SHARE:
                chip.append(_tag("主力大買超" if share > 0 else "主力大賣超",
                                 f"主力{'買' if share > 0 else '賣'}超 {abs(r['latest']):,.0f} 張，占成交量 {abs(share):.0%}"
                                 f"（門檻 {MAIN_FORCE_SHARE:.0%}，{r.get('asof') or ''}）", "bull" if share > 0 else "bear"))
    for s in (chips or {}).get("signals") or []:
        if s["key"] != "holder":
            chip.append(_tag(s["text"], "連續 3 日以上同方向（最多往回看 10 個交易日）", s["tone"]))
    dt = (snap.get("daytrade") or {}).get(symbol)
    if symbol in snap.get("hot_daytrade", set()):
        chip.append(_tag("當沖熱門", f"當沖 {dt / 1000:,.0f} 張，當沖金額全市場前 {HOT_RANK} 名"))
    if dt and q.get("volume"):
        ratio = dt / q["volume"]
        if ratio >= DAYTRADE_RATIO:
            chip.append(_tag("高當沖比", f"當沖占成交量 {ratio:.0%}（門檻 {DAYTRADE_RATIO:.0%}）"))
    mu = (snap.get("margin_usage") or {}).get(symbol, float("nan"))
    if mu == mu and mu >= MARGIN_USAGE * 100:
        chip.append(_tag("高融資", f"融資使用率 {mu:.1f}%（門檻 {MARGIN_USAGE:.0%}）"))
    h = (chips or {}).get("holder")
    if h:
        span = f"集保 {h['prev'][4:6]}/{h['prev'][6:]} → {h['asof'][4:6]}/{h['asof'][6:]}"
        if h["d_big"] >= HOLDER_CHANGE:
            chip.append(_tag("大戶加碼", f"≥400 張大戶持股 {h['big_pct']:.2f}%（{h['d_big']:+.2f}），{span}", "bull"))
        elif h["d_big"] <= -HOLDER_CHANGE:
            chip.append(_tag("大戶減碼", f"≥400 張大戶持股 {h['big_pct']:.2f}%（{h['d_big']:+.2f}），{span}", "bear"))
        if h["d_retail"] <= -HOLDER_CHANGE:
            chip.append(_tag("散戶離場", f"≤50 張散戶持股 {h['retail_pct']:.2f}%（{h['d_retail']:+.2f}），{span}", "bull"))
        elif h["d_retail"] >= HOLDER_CHANGE:
            chip.append(_tag("散戶進場", f"≤50 張散戶持股 {h['retail_pct']:.2f}%（{h['d_retail']:+.2f}），{span}", "bear"))
        since = h.get("since")
        if since:
            chip.append(_tag(*holder_since_tag(since)))
    fh = (snap.get("foreign_hold") or {}).get(symbol, float("nan"))
    if fh == fh and fh >= FOREIGN_HOLD:
        chip.append(_tag("外資高持股", f"外資持股比率 {fh:.1f}%（門檻 {FOREIGN_HOLD:.0f}%）"))

    # ---- 價量面、技術面 ----
    if symbol in snap.get("hot_value", set()):
        groups["價量面"].append(_tag("熱門股", f"成交金額 {q['value'] / 1e8:,.1f} 億，全市場前 {HOT_RANK} 名"))
    pv, tech = _technical(hist)
    groups["價量面"] += pv
    groups["價量面"] += _rotation_flow(symbol, rotation)
    groups["技術面"] = tech

    # ---- 基本面 ----
    fd = groups["基本面"]
    yoy = fund.get("rev_yoy", float("nan"))
    if yoy == yoy and yoy >= REVENUE_GROWTH:
        fd.append(_tag("營收高成長", f"{fund.get('rev_month', '')} 營收年增 {yoy:.1f}%（門檻 {REVENUE_GROWTH:.0f}%）", "bull"))
    elif yoy == yoy and yoy <= -REVENUE_GROWTH:
        fd.append(_tag("營收衰退", f"{fund.get('rev_month', '')} 營收年減 {abs(yoy):.1f}%（門檻 −{REVENUE_GROWTH:.0f}%）", "bear"))
    pe = fund.get("pe", float("nan"))
    if pe == pe and pe >= PE_HIGH:
        fd.append(_tag("高本益比", f"本益比 {pe:.1f} 倍（≥ {PE_HIGH:.0f}）"))
    elif pe == pe and 0 < pe <= PE_LOW:
        fd.append(_tag("低本益比", f"本益比 {pe:.1f} 倍（≤ {PE_LOW:.0f}）"))
    yld = fund.get("yld", float("nan"))
    if yld == yld and yld >= YIELD_HIGH:
        fd.append(_tag("高股息", f"殖利率 {yld:.2f}%（≥ {YIELD_HIGH:.0f}%）"))
    pb = fund.get("pb", float("nan"))
    if pb == pb and 0 < pb < 1:
        fd.append(_tag("股價低於淨值", f"股價淨值比 {pb:.2f}"))
    years = dividend_years(symbol) if _is_stock(symbol) else []
    this_year = date.today().year
    need = set(range(this_year - DIVIDEND_YEARS, this_year))
    if years and need <= set(years):
        streak = 0
        for y in range(this_year - 1, min(years) - 1, -1):
            if y not in years:
                break
            streak += 1
        fd.append(_tag("連續配息", f"{this_year - streak}–{this_year - 1} 連續 {streak} 年配息（Yahoo Finance）", "bull"))
    cap = fund.get("capital", float("nan"))
    if cap == cap and cap >= BIG_CAPITAL:
        fd.append(_tag("大股本", f"實收資本額 {cap / 1e8:,.0f} 億（≥ {BIG_CAPITAL / 1e8:.0f} 億）"))
    elif cap == cap and 0 < cap <= SMALL_CAPITAL:
        fd.append(_tag("小股本", f"實收資本額 {cap / 1e8:,.1f} 億（≤ {SMALL_CAPITAL / 1e8:.0f} 億）"))

    return {"symbol": symbol, "asof": day.isoformat() if day else None, "groups": groups, "computed_at": time.time()}
