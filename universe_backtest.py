"""全市場（非事後選股）波段回測。

目的：theme_backtest.py 用「今天的 175 檔題材股」回測過去，有事後選股偏差。這支程式改用
「目前仍上市櫃的全部普通股（4 碼、非 ETF）」＋每天只用當天以前的資料決定流動性與動能，
重新測試動能波段的各種變形，並輸出逐筆 CSV 給 strategy-gate。

python universe_backtest.py fetch      # 下載 3 年還原日 K 到 data/universe_bt/（Fugle，約 55 次／分）
python universe_backtest.py run_theme  # 同樣規則套在舊的 175 檔題材股（事後選股，只當對照）
python universe_backtest.py run        # 回測，逐筆寫到 output/strategy_research_v2/，摘要寫 summary.json

仍存在的偏差（報告要寫清楚）：
- 名單是「今天仍掛牌」的股票，過去 3 年下市／被併購的股票不在裡面（輕度倖存者偏差）。
- 產業別是今天的 TWSE/TPEx 產業代碼（只用在分散與產業動能，產業別很少變動）。
研究用，不下單、不構成投資建議。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
os.chdir(BASE); sys.path.insert(0, str(BASE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE = BASE / "data" / "universe_bt"
OLD_CACHE = BASE / "data" / "theme_bt"
OUT = BASE / "output" / "strategy_research_v2"
INDEX = "IX0001"
END = date(2026, 10, 23)


def universe() -> pd.DataFrame:
    d = pd.read_csv(BASE / "data" / "industry_master.csv", dtype=str)
    d = d[d.symbol.str.fullmatch(r"[1-9]\d{3}") & (d.industry != "00")]
    return d.drop_duplicates("symbol").set_index("symbol")


def fetch() -> None:
    from stock_api import FugleClient
    c = FugleClient(wait_on_limit=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    syms = list(universe().index) + [INDEX]
    for n, s in enumerate(syms, 1):
        p = CACHE / f"{s}.csv"
        if p.exists():
            continue
        old = OLD_CACHE / f"{s}.csv"
        if old.exists():
            o = pd.read_csv(old, parse_dates=["date"])
            if len(o) and o.date.min() <= pd.Timestamp("2023-11-01") and o.date.max() >= pd.Timestamp("2026-10-20"):
                shutil.copy(old, p)
                continue
        parts = []
        for y in range(3):
            e = END - timedelta(days=365 * y)
            for attempt in range(4):
                try:
                    parts.append(c.historical_candles(s, e - timedelta(days=364), e, "D", adjusted=True))
                    break
                except Exception as exc:
                    if "429" in str(exc) and attempt < 3:
                        print(f"{s} 429，等 61 秒", flush=True)
                        time.sleep(61)
                        continue
                    print(f"{n}/{len(syms)} {s} 失敗：{str(exc)[:120]}", flush=True)
                    break
        parts = [x for x in parts if len(x)]
        if not parts:
            pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]).to_csv(p, index=False)
            continue
        d = pd.concat(parts).drop_duplicates("date").sort_values("date")
        d.to_csv(p, index=False, encoding="utf-8-sig")
        if n % 50 == 0:
            print(f"{n}/{len(syms)} {s} {len(d)} 筆", flush=True)
    print("fetch 完成", flush=True)


RAW_CACHE = BASE / "data" / "universe_bt_raw"


def fetch_raw(years: int = 6) -> None:
    """未還原日 K（當時的真實股價），只抓曾經進入流動名單的股票（data/universe_bt_liquid_syms.txt）。
    用途：判斷一張買不買得起／股價上限、計算股數與 tick 滑價。報酬仍用還原價（含除權息）。"""
    from stock_api import FugleClient
    c = FugleClient(wait_on_limit=True)
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    syms = (BASE / "data" / "universe_bt_liquid_syms.txt").read_text(encoding="utf-8").split()
    for n, s in enumerate(syms, 1):
        p = RAW_CACHE / f"{s}.csv"
        if p.exists():
            continue
        parts = []
        for y in range(years):
            e = END - timedelta(days=365 * y)
            for attempt in range(4):
                try:
                    parts.append(c.historical_candles(s, e - timedelta(days=364), e, "D", adjusted=False))
                    break
                except Exception as exc:
                    if "429" in str(exc) and attempt < 3:
                        print(f"{s} 429，等 61 秒", flush=True); time.sleep(61); continue
                    break   # 404＝那一年還沒上市
        parts = [x for x in parts if len(x)]
        d = pd.concat(parts).drop_duplicates("date").sort_values("date") if parts else pd.DataFrame(columns=["date", "close"])
        d.to_csv(p, index=False, encoding="utf-8-sig")
        if n % 50 == 0:
            print(f"raw {n}/{len(syms)} {s}", flush=True)
    print("fetch_raw 完成", flush=True)


def fetch_ext(years: int = 6) -> None:
    """把快取往前延長到 years 年（另外補 0050）。已延長過的檔案（最早日期 ≤ 起點＋30 天）略過。"""
    from stock_api import FugleClient
    c = FugleClient(wait_on_limit=True)
    start = END - timedelta(days=365 * years)
    syms = list(universe().index) + [INDEX, "0050"]
    for n, s in enumerate(syms, 1):
        p = CACHE / f"{s}.csv"
        old = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame(columns=["date"])
        if len(old) and old.date.min() <= pd.Timestamp(start) + pd.Timedelta(days=30):
            continue
        done = (CACHE / ".ext_done.txt")
        if done.exists() and s in done.read_text().split():
            continue
        parts = [old] if len(old) else []
        first_y = 0 if not len(old) else 3
        for y in range(first_y, years):
            e = END - timedelta(days=365 * y)
            for attempt in range(4):
                try:
                    parts.append(c.historical_candles(s, e - timedelta(days=364), e, "D", adjusted=True))
                    break
                except Exception as exc:
                    if "429" in str(exc) and attempt < 3:
                        print(f"{s} 429，等 61 秒", flush=True); time.sleep(61); continue
                    break   # 404＝那一年還沒上市
        parts = [x for x in parts if len(x)]
        if parts:
            d = pd.concat(parts).drop_duplicates("date").sort_values("date")
            d.to_csv(p, index=False, encoding="utf-8-sig")
        with done.open("a") as fh:
            fh.write(s + "\n")
        if n % 100 == 0:
            print(f"ext {n}/{len(syms)} {s}", flush=True)
    print("fetch_ext 完成", flush=True)


# ---------------------------------------------------------------- 時點正確的法人與月營收

INSTI_DIR = BASE / "data" / "universe_bt_insti"
REV_DIR = BASE / "data" / "universe_bt_revenue"
_H = {"User-Agent": "Mozilla/5.0"}


def _num_s(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan


def fetch_insti(market: str, start: str | None = None, end: str | None = None) -> None:
    """三大法人每日買賣超（股），TWSE T86／TPEx dailyTrade 官方日報，每個交易日一個檔。只存 {代號: 三大法人合計}。
    給 start/end 時用平日日曆（假日會存成空檔）。"""
    import requests
    INSTI_DIR.mkdir(parents=True, exist_ok=True)
    if start:
        days = list(pd.bdate_range(start, end))
    else:
        days = pd.read_csv(CACHE / f"{INDEX}.csv", parse_dates=["date"]).date
        days = [d for d in days if d <= DATA_END]
    for n, d in enumerate(days, 1):
        p = INSTI_DIR / f"{market}_{d:%Y%m%d}.json"
        if p.exists():
            continue
        out = None
        for attempt in range(4):
            try:
                if market == "twse":
                    r = requests.get("https://www.twse.com.tw/rwd/zh/fund/T86", headers=_H, timeout=30,
                                     params={"date": f"{d:%Y%m%d}", "selectType": "ALLBUT0999", "response": "json"})
                    j = r.json()
                    rows = j.get("data") or []
                    out = {str(x[0]).strip(): _num_s(x[-1]) for x in rows}
                else:
                    r = requests.get("https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade", headers=_H, timeout=30,
                                     params={"type": "Daily", "sect": "AL", "date": f"{d:%Y/%m/%d}", "response": "json"})
                    t = (r.json().get("tables") or [{}])[0]
                    out = {str(x[0]).strip(): _num_s(x[-1]) for x in t.get("data") or []}
                break
            except Exception as exc:
                print(f"{market} {d.date()} 第 {attempt + 1} 次失敗：{str(exc)[:80]}", flush=True)
                time.sleep(30 * (attempt + 1))
        if out is not None:
            p.write_text(json.dumps(out), encoding="utf-8")
        if n % 50 == 0:
            print(f"{market} {n}/{len(days)} {d.date()} {len(out or {})} 檔", flush=True)
        time.sleep(3.2 if market == "twse" else 2.0)
    print(f"{market} 法人 fetch 完成", flush=True)


def fetch_revenue(start_year: int = 2020) -> None:
    """MOPS 每月營收彙總表（上市 sii、上櫃 otc），start_year-01 起到「已過公告期限」的最新月份。
    第 m 月要到 m+1 月 11 日才算公告完整，在那之前不下載（避免把只公告一半的月份存進快取）。"""
    import io
    import requests
    REV_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    last = (today.replace(day=1) - timedelta(days=1)) if today.day >= 11 else         ((today.replace(day=1) - timedelta(days=1)).replace(day=1) - timedelta(days=1))
    for kind in ("sii", "otc"):
        for y in range(start_year, today.year + 1):
            for m in range(1, 13):
                if (y, m) > (last.year, last.month):
                    break
                p = REV_DIR / f"{kind}_{y}{m:02d}.csv"
                if p.exists():
                    continue
                url = f"https://mopsov.twse.com.tw/nas/t21/{kind}/t21sc03_{y - 1911}_{m}_0.html"
                try:
                    r = requests.get(url, headers=_H, timeout=60)
                    r.encoding = "big5"
                    tabs = pd.read_html(io.StringIO(r.text))
                except Exception as exc:
                    print(f"{kind} {y}-{m} 失敗：{str(exc)[:80]}", flush=True)
                    time.sleep(5); continue
                rows = []
                for t in tabs:
                    if isinstance(t.columns, pd.MultiIndex):
                        t.columns = ["".join(str(x) for x in c) for c in t.columns]
                    t.columns = [str(c).replace(" ", "") for c in t.columns]
                    cols = list(t.columns)
                    code = next((c for c in cols if "公司代號" in c), None)
                    cur = next((c for c in cols if "當月營收" in c), None)
                    ly = next((c for c in cols if "去年當月營收" in c), None)
                    if code and cur and ly:
                        x = t[[code, cur, ly]].copy(); x.columns = ["symbol", "rev", "rev_ly"]
                        rows.append(x)
                if not rows:
                    print(f"{kind} {y}-{m} 沒有表格", flush=True); continue
                d = pd.concat(rows)
                d["symbol"] = d.symbol.astype(str).str.strip()
                d = d[d.symbol.str.fullmatch(r"\d{4}")]
                d["rev"] = d.rev.map(_num_s); d["rev_ly"] = d.rev_ly.map(_num_s)
                d.to_csv(p, index=False)
                print(f"{kind} {y}-{m} {len(d)} 家", flush=True)
                time.sleep(3)
    print("營收 fetch 完成", flush=True)


def load_insti(dates: pd.DatetimeIndex, cols) -> pd.DataFrame | None:
    """三大法人買賣超（股）寬表；沒有資料的日期為 NaN。"""
    if not INSTI_DIR.exists():
        return None
    rec = {}
    for d in dates:
        row = {}
        for m in ("twse", "tpex"):
            p = INSTI_DIR / f"{m}_{d:%Y%m%d}.json"
            if p.exists():
                row.update(json.loads(p.read_text(encoding="utf-8")))
        if row:
            rec[d] = row
    if not rec:
        return None
    return pd.DataFrame.from_dict(rec, orient="index").reindex(index=dates, columns=cols)


def load_revenue_yoy(dates: pd.DatetimeIndex, cols) -> pd.DataFrame | None:
    """月營收 YoY（近 3 個月合計 vs 去年同期），**第 m 月資料從 m+1 月 11 日起才可用**（法定 10 日前公告）。"""
    if not REV_DIR.exists():
        return None
    mon = {}
    for p in REV_DIR.glob("*.csv"):
        ym = p.stem.split("_")[1]
        d = pd.read_csv(p, dtype={"symbol": str}).drop_duplicates("symbol").set_index("symbol")
        mon.setdefault(ym, []).append(d)
    if not mon:
        return None
    months = sorted(mon)
    mm = {m: (lambda x: x[~x.index.duplicated()])(pd.concat(mon[m])) for m in months}
    rev = pd.DataFrame({m: mm[m]["rev"] for m in months}).T
    rly = pd.DataFrame({m: mm[m]["rev_ly"] for m in months}).T
    yoy3 = rev.rolling(3).sum() / rly.rolling(3).sum() - 1
    avail = [pd.Timestamp(int(m[:4]), int(m[4:]), 1) + pd.DateOffset(months=1, days=10) for m in months]
    yoy3.index = pd.DatetimeIndex(avail)
    yoy3 = yoy3[~yoy3.index.duplicated()].sort_index()
    return yoy3.reindex(columns=cols).reindex(dates, method="ffill")


# ---------------------------------------------------------------- 回測

DATA_END = pd.Timestamp("2026-10-22")       # 舊快取只到 10/22，全部統一截在這天
CAPITAL = 300_000
FEE, FEE_MIN, TAX, SLIP = 0.001425, 1.0, 0.003, 2
WARMUP = 60


def tick_size(p):
    p = np.asarray(p, dtype=float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def load(min_rows: int = 60) -> dict:
    uni = universe()
    fr = {}
    for p in CACHE.glob("*.csv"):
        s = p.stem
        try:
            d = pd.read_csv(p, parse_dates=["date"])
        except Exception:
            continue
        d = d[d.date <= DATA_END]
        if len(d) >= min_rows:
            fr[s] = d.drop_duplicates("date").set_index("date")
    px = {k: pd.DataFrame({s: f[k] for s, f in fr.items()}).sort_index() for k in ("open", "high", "low", "close", "volume")}
    idx = px["close"].pop(INDEX) if INDEX in px["close"] else None
    for k in ("open", "high", "low", "volume"):
        px[k] = px[k].drop(columns=[INDEX], errors="ignore")
    syms = [s for s in px["close"].columns if s in uni.index]
    for k in px:
        px[k] = px[k][syms]
    px["index"] = idx
    px["industry"] = uni.loc[syms, "industry"]
    return px


def prepare(px: dict, min_turnover: float) -> dict:
    c, v = px["close"], px["volume"]
    f = dict(px)
    f["ma20"] = c.rolling(20, min_periods=20).mean()
    f["r20"] = c.pct_change(20, fill_method=None)
    # 流動性：過去 20 天平均成交值（只用當天以前），而且過去 20 天都有成交（排除剛上市、停牌）
    f["turnover"] = (c * v).rolling(20, min_periods=20).mean()
    f["liquid"] = (f["turnover"] > min_turnover) & c.notna()
    # 產業動能：當天流動股票裡，同一產業代碼 ≥3 檔的 20 日等權報酬
    r = f["r20"].where(f["liquid"])
    ind = px["industry"]
    g = r.T.groupby(ind.reindex(r.columns).values)
    f["ind_mom"] = g.mean().T.where(g.count().T >= 3)
    f["ind_mom"].index = c.index
    # v3：ATR(14)（佔收盤 %）、60 日動能、法人 20 日累計買賣超、月營收近 3 月 YoY（公告延遲）
    pc = c.shift(1)
    tr = pd.concat([px["high"] - px["low"], (px["high"] - pc).abs(), (px["low"] - pc).abs()]).groupby(level=0).max()
    f["atr_pct"] = (tr.rolling(14, min_periods=14).mean() / c).reindex(index=c.index, columns=c.columns)
    f["r60"] = c.pct_change(60, fill_method=None)
    ins = load_insti(c.index, c.columns)
    f["insti20"] = ins.rolling(20, min_periods=15).sum() if ins is not None else None
    f["insti_cover"] = ins.notna().any(axis=1) if ins is not None else None
    f["rev_yoy3"] = load_revenue_yoy(c.index, c.columns)
    # 未還原價（當時真實股價）：股價上限、一張買不買得起、股數、tick 滑價都用它；沒有資料時退回還原價
    f["raw_open"], f["raw_close"] = load_raw(c.index, c.columns)
    f["raw_ok"] = f["raw_close"] is not None
    return f


def _trade(f, s, a, last, stop, tp, gap, value, stop_close=False, slip=SLIP, fee_min=FEE_MIN, lot=1):
    """d+1（索引 a）開盤買，持有到索引 last 收盤；回傳 (dict 或 None, 每日市值 list)。
    lot=1000：整股（股數取 1000 的倍數）；slip：買賣各幾個 tick；fee_min：手續費最低金額。"""
    o, h, l, c = (f[k][s].values for k in ("open", "high", "low", "close"))
    prev = c[a - 1]
    op = o[a]
    if not (op > 0 and prev > 0):
        return None, "no_open"
    if gap is not None and op > prev * (1 + gap):
        return None, "gap"
    if op >= prev * 1.09 and h[a] == l[a]:
        return None, "limit_locked"
    e = op + slip * float(tick_size(op))
    sh = int(value / (e * (1 + FEE)) / lot) * lot
    if sh < 1:
        return None, "too_expensive"
    buy_fee = max(fee_min, e * sh * FEE)
    cost_in = e * sh + buy_fee
    st = e * (1 - stop) if stop else None
    tgt = e * (1 + tp) if tp else None
    path, xr, reason, xi, pend_exit = [], None, "time", last, False
    for i in range(a, last + 1):
        if np.isnan(c[i]):
            path.append(path[-1] if path else e * sh); continue
        if pend_exit:                       # 前一天收盤跌破停損 → 今天開盤賣
            xr, reason, xi = o[i] if o[i] > 0 else c[i], "stop", i; break
        if st is not None and not stop_close and (l[i] <= st):
            xr = min(o[i], st) if i > a else st
            reason, xi = "stop", i; break
        if tgt is not None and h[i] >= tgt:
            xr = max(o[i], tgt) if i > a else tgt
            reason, xi = "target", i; break
        path.append(c[i] * sh)
        if st is not None and stop_close and c[i] <= st and i < last:
            pend_exit = True
    if xr is None:
        xr = c[last] if not np.isnan(c[last]) else (path[-1] / sh)
        path = path[:-1]
    x = max(xr - slip * float(tick_size(xr)), 0.01)
    proceeds = x * sh - max(fee_min, x * sh * FEE) - x * sh * TAX
    path.append(proceeds)
    leftover = value - cost_in
    return {"symbol": s, "entry_i": a, "exit_i": xi, "entry": round(e, 2), "exit": round(x, 2), "shares": sh,
            "gross": xr / op - 1, "net": proceeds / cost_in - 1, "reason": reason,
            "path": [p + leftover for p in path]}, None


def load_raw(dates, cols):
    if not RAW_CACHE.exists():
        return None, None
    o, c = {}, {}
    for p in RAW_CACHE.glob("*.csv"):
        try:
            d = pd.read_csv(p, parse_dates=["date"])
        except Exception:
            continue
        if len(d) and p.stem in cols:
            d = d.drop_duplicates("date").set_index("date")
            o[p.stem], c[p.stem] = d["open"], d["close"]
    if not c:
        return None, None
    ro = pd.DataFrame(o).reindex(index=dates, columns=cols)
    rc = pd.DataFrame(c).reindex(index=dates, columns=cols)
    return ro, rc


def _trade_raw(f, s, a, last, stop, gap, value, slip, fee_min, lot):
    """用未還原價算股數、tick 滑價、手續費；用還原價算持有期間的報酬（包含除權息）。停損只看收盤（隔天開盤賣）。
    做法：進場時以真實價買 sh 股（名目 = 真實價 × sh），換算成「還原價單位」= 名目 ÷ 還原進場價，
    之後市值 = 單位 × 還原收盤；出場時依當天真實價的 tick 扣滑價，再扣手續費與證交稅。"""
    o, h, l, c = (f[k][s].values for k in ("open", "high", "low", "close"))
    ro, rc = f["raw_open"][s].values, f["raw_close"][s].values
    prev, op = c[a - 1], o[a]
    if not (op > 0 and prev > 0):
        return None, "no_open"
    if gap is not None and op > prev * (1 + gap):
        return None, "gap"
    if op >= prev * 1.09 and h[a] == l[a]:
        return None, "limit_locked"
    rop = ro[a] if ro[a] > 0 else op
    slip_in = slip * float(tick_size(rop)) / rop
    e_raw = rop * (1 + slip_in)
    sh = int(value / (e_raw * (1 + FEE)) / lot) * lot
    if sh < 1:
        return None, "too_expensive"
    notional = e_raw * sh
    buy_fee = max(fee_min, notional * FEE)
    cost_in = notional + buy_fee
    e_adj = op * (1 + slip_in)
    units = notional / e_adj
    st = e_adj * (1 - stop) if stop else None
    path, xr, xi, reason, pend = [], None, last, "time", False
    for i in range(a, last + 1):
        if np.isnan(c[i]):
            path.append(path[-1] if path else notional); continue
        if pend:
            xr, xi, reason = (o[i] if o[i] > 0 else c[i]), i, "stop"
            break
        path.append(units * c[i])
        if st is not None and c[i] <= st and i < last:
            pend = True
    if xr is None:
        xi = last
        xr = c[last] if not np.isnan(c[last]) else path[-1] / units
        path = path[:-1]
    ratio = (rc[xi] / c[xi]) if (rc[xi] > 0 and c[xi] > 0) else 1.0
    rx = xr * ratio
    x_adj = xr * (1 - slip * float(tick_size(rx)) / rx)
    val = units * x_adj
    proceeds = val - max(fee_min, val * FEE) - val * TAX
    path.append(proceeds)
    leftover = value - cost_in
    return {"symbol": s, "entry_i": a, "exit_i": xi, "entry": round(e_raw, 2), "exit": round(rx, 2), "shares": sh,
            "gross": xr / op - 1, "net": proceeds / cost_in - 1, "reason": reason,
            "path": [p + leftover for p in path]}, None


def simulate(f: dict, cfg: dict, offset: int = 0, seed: int = 0):
    """每 hold 個交易日一批：訊號日 i 收盤選股 → i+1 開盤買（每檔 NAV/K）→ 停損/停利或第 hold 天收盤賣。"""
    rng = np.random.default_rng(seed)
    dates = f["close"].index
    hold, k = cfg["hold"], cfg.get("k", 6)
    ind = f["industry"]
    nav, curve, trades, skips = float(CAPITAL), [], [], {}
    idx = f["index"]
    ima60 = idx.rolling(60).mean() if idx is not None else None
    i = WARMUP + offset
    while i + hold < len(dates):
        d = dates[i]
        a, last = i + 1, i + hold
        picks = []
        if not (cfg.get("mkt") and ima60 is not None and idx.iloc[i] < ima60.iloc[i]):
            liq = f["liquid"].loc[d]
            base = liq[liq].index
            sel = cfg["sel"]
            if cfg.get("lot", 1) > 1 or cfg.get("price_cap"):   # 整股：一張買得起才列入（用訊號日收盤估）
                cap_px = cfg.get("price_cap") or nav / k / (cfg["lot"] * (1 + FEE) * 1.03)
                pxs = (f["raw_close"] if f.get("raw_ok") and cfg.get("raw") else f["close"]).loc[d, base]
                base = base[(pxs <= cap_px).values]
            if cfg.get("insti"):          # 三大法人近 20 日累計買超 > 0（當天收盤後公布，隔天開盤才用）
                v = f["insti20"].loc[d, base] if f.get("insti20") is not None else None
                base = base[(v > 0).values] if v is not None else base[:0]
            if cfg.get("rev") is not None:  # 月營收近 3 月 YoY > 門檻（只用已公告的月份）
                v = f["rev_yoy3"].loc[d, base] if f.get("rev_yoy3") is not None else None
                base = base[(v > cfg["rev"]).values] if v is not None else base[:0]
            if sel in ("mom", "ind", "mom60", "revrank"):
                key = {"mom60": "r60", "revrank": "rev_yoy3"}.get(sel, "r20")
                ok = (f["close"].loc[d, base] > f["ma20"].loc[d, base]) & f[key].loc[d, base].notna()
                cand = list(f[key].loc[d, base][ok].sort_values(ascending=False).index)
                if sel == "ind":
                    im = f["ind_mom"].loc[d].dropna().sort_values(ascending=False)
                    hot = set(im.index[:cfg.get("top_ind", 3)])
                    cand = [s for s in cand if ind[s] in hot]
            elif sel == "rand_trend":
                ok = f["close"].loc[d, base] > f["ma20"].loc[d, base]
                cand = list(rng.permutation(list(base[ok.values])))
            else:  # rand：流動股票隨機
                cand = list(rng.permutation(list(base)))
            cnt: dict = {}
            for s in cand:
                if cfg.get("cap") and cnt.get(ind[s], 0) >= cfg["cap"]:
                    continue
                picks.append(s); cnt[ind[s]] = cnt.get(ind[s], 0) + 1
                if len(picks) >= k:
                    break
        per = nav / k
        slots = []
        for s in picks:
            stop = cfg.get("stop")
            if cfg.get("atr_k"):
                ap = f["atr_pct"].at[d, s]
                stop = min(cfg["atr_k"] * (ap if ap > 0 else 0.03), 0.5)
            if cfg.get("raw") and f.get("raw_ok"):
                t, why = _trade_raw(f, s, a, last, stop, cfg.get("gap"), per,
                                    cfg.get("slip", SLIP), cfg.get("fee_min", FEE_MIN), cfg.get("lot", 1))
            else:
                t, why = _trade(f, s, a, last, stop, cfg.get("tp"), cfg.get("gap"), per,
                                cfg.get("stop_close", False) or bool(cfg.get("atr_k")),
                                cfg.get("slip", SLIP), cfg.get("fee_min", FEE_MIN), cfg.get("lot", 1))
            if t is None:
                skips[why] = skips.get(why, 0) + 1
                continue
            t["day"] = str(dates[a].date()); t["signal_day"] = str(d.date())
            trades.append(t); slots.append(t)
        n_days = last - a + 1
        cash = nav - per * len(slots)
        for j in range(n_days):
            v = cash
            for t in slots:
                p = t["path"]
                v += p[min(j, len(p) - 1)]
            curve.append((dates[a + j], v))
        nav = curve[-1][1]
        i = last
    c = pd.Series(dict(curve)).sort_index()
    return trades, c, skips


def stats(curve: pd.Series) -> dict:
    if curve.empty:
        return {}
    eq = curve / CAPITAL
    yrs = max((curve.index[-1] - curve.index[0]).days / 365.25, 0.1)
    dd = (eq / eq.cummax().clip(lower=1.0) - 1).min()
    return {"total": round((eq.iloc[-1] - 1) * 100, 1), "cagr": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
            "maxdd": round(dd * 100, 1)}


def fname(name: str) -> str:
    return name.replace(">", "超過").replace("<", "低於")


def _gate():
    import importlib.util
    p = BASE / ".claude" / "skills" / "strategy-gate" / "scripts" / "gate_check.py"
    spec = importlib.util.spec_from_file_location("gate_check", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def tdf(trades) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["day", "symbol", "net", "gross", "reason"])
    t = pd.DataFrame([{k: v for k, v in x.items() if k != "path"} for x in trades])
    t["net"] = t["net"] * 100; t["gross"] = t["gross"] * 100
    return t


def variants() -> dict:
    v = {}
    for sel, sn in (("mom", "個股動能"), ("ind", "產業動能")):
        for hold in (5, 10, 20):
            for cap in (None, 2):
                for gap in (None, 0.03):
                    name = f"{sn}_持{hold}天" + (f"_同產業≤{cap}" if cap else "") + ("_跳空>3%不追" if gap else "")
                    v[name] = {"sel": sel, "hold": hold, "cap": cap, "gap": gap, "stop": 0.05}
    for hold in (5, 10, 20):
        v[f"個股動能_持{hold}天_同產業≤2_跳空>3%不追_大盤60日線濾網"] = {"sel": "mom", "hold": hold, "cap": 2, "gap": 0.03, "stop": 0.05, "mkt": True}
        v[f"個股動能_持{hold}天_同產業≤2_跳空>3%不追_停利+5%"] = {"sel": "mom", "hold": hold, "cap": 2, "gap": 0.03, "stop": 0.05, "tp": 0.05}
        v[f"個股動能_持{hold}天_同產業≤2_跳空>3%不追_收盤跌破-5%才停損"] = {"sel": "mom", "hold": hold, "cap": 2, "gap": 0.03, "stop": 0.05, "stop_close": True}
        v[f"個股動能_持{hold}天_同產業≤2_跳空>3%不追_無停損"] = {"sel": "mom", "hold": hold, "cap": 2, "gap": 0.03, "stop": None}
    return v


def run(min_turnover: float = 5e8, n_seeds: int = 20, tag: str = "") -> int:
    """tag="theme"：改用舊的 175 檔題材股快取（事後選股，只當對照）。"""
    global CACHE, OUT
    if tag == "theme":
        CACHE, OUT = OLD_CACHE, OUT.parent / "strategy_research_v2_theme_hindsight"
    elif min_turnover != 5e8:
        OUT = OUT.parent / f"strategy_research_v2_turnover{min_turnover / 1e8:g}e8"
    gate = _gate()
    px = load()
    f = prepare(px, min_turnover)
    dates = f["close"].index
    nliq = f["liquid"].sum(axis=1)
    print(f"股票 {f['close'].shape[1]} 檔，{dates[0].date()}～{dates[-1].date()}，{len(dates)} 天；"
          f"每天流動股票數 中位 {int(nliq.iloc[WARMUP:].median())}（最少 {int(nliq.iloc[WARMUP:].min())}）", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trades").mkdir(exist_ok=True)
    res = {"meta": {"symbols": int(f["close"].shape[1]), "start": str(dates[0].date()), "end": str(dates[-1].date()),
                    "days": len(dates), "min_turnover": min_turnover,
                    "liquid_median": int(nliq.iloc[WARMUP:].median()), "liquid_min": int(nliq.iloc[WARMUP:].min())}}
    idx = f["index"]
    if idx is not None:
        ib = idx.iloc[WARMUP:]
        res["加權指數買進持有"] = {"total": round((ib.iloc[-1] / ib.iloc[0] - 1) * 100, 1),
                             "maxdd": round(((ib / ib.cummax()) - 1).min() * 100, 1)}

    def evaluate(name, cfg, seeds=(0,)):
        out = {"cfg": {k: v for k, v in cfg.items()}, "offsets": []}
        hold = cfg["hold"]
        offs = range(hold)
        for off in offs:
            per_seed = []
            for sd in seeds:
                tr, cv, sk = simulate(f, cfg, off, sd)
                t = tdf(tr)
                g = gate.check(t[["day", "net"]].copy(), False) if len(t) else {"verdict": "FAIL", "checks": []}
                per_seed.append({"off": off, "seed": sd, **stats(cv), "n": len(t),
                                 "avg_net": round(float(t.net.mean()), 3) if len(t) else None,
                                 "win": round(float((t.net > 0).mean() * 100), 1) if len(t) else None,
                                 "stop_rate": round(float((t.reason == "stop").mean() * 100), 1) if len(t) else None,
                                 "gate": g["verdict"], "gate_fail": [c["name"] for c in g["checks"] if not c["ok"]],
                                 "skips": sk})
                if off == 0 and sd == seeds[0]:
                    t.to_csv(OUT / "trades" / f"{fname(name)}.csv", index=False, encoding="utf-8-sig")
                    out["off0_curve_end"] = float(cv.iloc[-1]) if len(cv) else None
            out["offsets"] += per_seed
        o = pd.DataFrame(out["offsets"])
        out["summary"] = {"off0": {k: o.iloc[0][k] for k in ("total", "cagr", "maxdd", "n", "avg_net", "win", "stop_rate", "gate")},
                          "off0_gate_fail": o.iloc[0]["gate_fail"],
                          "total_median": float(o.total.median()), "total_min": float(o.total.min()), "total_max": float(o.total.max()),
                          "maxdd_median": float(o.maxdd.median()), "maxdd_worst": float(o.maxdd.min()),
                          "avg_net_median": float(o.avg_net.median()),
                          "gate_pass_share": round(float((o.gate == "PASS").mean() * 100), 0), "runs": len(o)}
        s = out["summary"]
        print(f"{name:44s} 起點0：總 {s['off0']['total']:+7.1f}% 回落 {s['off0']['maxdd']:6.1f}% 筆 {s['off0']['n']:4d} "
              f"均 {s['off0']['avg_net']:+.2f}% 勝 {s['off0']['win']}% 停損 {s['off0']['stop_rate']}% gate {s['off0']['gate']}"
              f" | 各起點總報酬 中位 {s['total_median']:+.1f}%（{s['total_min']:+.1f}～{s['total_max']:+.1f}） 回落最差 {s['maxdd_worst']:.1f}%"
              f" PASS 比例 {s['gate_pass_share']:.0f}%", flush=True)
        return out

    # 隨機對照：每種持有天數，流動股票隨機 6 檔（同樣的停損、跳空、同產業 ≤2 規則）
    for hold in (5, 10, 20):
        for sel, sn in (("rand", "隨機_流動股"), ("rand_trend", "隨機_站上20日線")):
            name = f"{sn}_持{hold}天"
            cfg = {"sel": sel, "hold": hold, "cap": 2, "gap": 0.03, "stop": 0.05}
            res[name] = evaluate(name, cfg, seeds=tuple(range(n_seeds)) if hold == 20 else tuple(range(max(3, n_seeds // 4))))
    for name, cfg in variants().items():
        res[name] = evaluate(name, cfg)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


ODD = {"slip": 2, "fee_min": 1.0, "lot": 1, "k": 6}            # 零股：買賣各 2 tick、手續費最低 1 元
LOT6 = {"slip": 1, "fee_min": 20.0, "lot": 1000, "k": 6}       # 整股 6 檔：每檔約 5 萬 → 只能買股價約 48 元以下
LOT3 = {"slip": 1, "fee_min": 20.0, "lot": 1000, "k": 3}       # 整股 3 檔：每檔約 10 萬 → 股價約 97 元以下


def variants_v3() -> dict:
    v = {}
    sels = {"個股動能20日": {"sel": "mom"}, "個股動能60日": {"sel": "mom60"}, "產業動能": {"sel": "ind"},
            "動能20日+法人買超": {"sel": "mom", "insti": True}, "動能20日+營收YoY>0": {"sel": "mom", "rev": 0.0},
            "動能20日+法人+營收": {"sel": "mom", "insti": True, "rev": 0.0}, "營收YoY排名": {"sel": "revrank"},
            "隨機(對照)": {"sel": "rand"}, "隨機+法人+營收(對照)": {"sel": "rand", "insti": True, "rev": 0.0}}
    stops = {"不停損": {}, "收盤跌破2ATR": {"atr_k": 2}, "收盤跌破3ATR": {"atr_k": 3}}
    costs = {"零股6檔": ODD, "整股6檔": LOT6, "整股3檔": LOT3}
    for hold in (20, 40, 60):
        for sn, sc in sels.items():
            for tn, tc in stops.items():
                for cn, cc in costs.items():
                    if cn != "零股6檔" and sn not in ("個股動能20日", "動能20日+法人+營收", "隨機(對照)", "營收YoY排名"):
                        continue
                    v[f"{sn}_持{hold}天_{tn}_{cn}"] = {"hold": hold, "cap": 2, "gap": 0.03, **sc, **tc, **cc}
    return v


def variants_v4() -> dict:
    """v4：未還原價版（股數、滑價、股價限制用當時真實股價）。只重跑 v3 的領先候選＋對照，避免再擴大多重檢定。"""
    v = {}
    sels = {"營收YoY排名": {"sel": "revrank"}, "動能20日+法人+營收": {"sel": "mom", "insti": True, "rev": 0.0},
            "隨機+法人+營收(對照)": {"sel": "rand", "insti": True, "rev": 0.0}, "隨機(對照)": {"sel": "rand"}}
    stops = {"不停損": {}, "收盤跌破3ATR": {"atr_k": 3}, "收盤跌破2ATR": {"atr_k": 2}}
    costs = {"整股6檔": LOT6, "零股6檔": ODD}
    for hold in (40, 60):
        for sn, sc in sels.items():
            for tn, tc in stops.items():
                for cn, cc in costs.items():
                    v[f"{sn}_持{hold}天_{tn}_{cn}"] = {"hold": hold, "cap": 2, "gap": 0.03, "raw": True, **sc, **tc, **cc}
    return v


def _bench(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, cost: float = 0.006) -> float | None:
    if series is None:
        return None
    s = series.dropna()
    s = s[(s.index >= start) & (s.index <= end)]
    if len(s) < 2:
        return None
    return round((s.iloc[-1] / s.iloc[0] * (1 - cost) - 1) * 100, 1)


def run_v3(min_turnover: float = 5e8, n_seeds: int = 5, only: str | None = None, shard: str | None = None,
           version: str = "v3") -> int:
    """only：名稱包含這段文字的變形；shard="i/n"：只跑第 i 份（共 n 份）。每個變形的結果另存 parts/，已存在就略過（可續跑）。"""
    global OUT
    OUT = BASE / "output" / f"strategy_research_{version}"
    (OUT / "trades").mkdir(parents=True, exist_ok=True)
    gate = _gate()
    px = load()
    f = prepare(px, min_turnover)
    dates = f["close"].index
    e50 = None
    p50 = CACHE / "0050.csv"
    if p50.exists():
        e50 = pd.read_csv(p50, parse_dates=["date"]).set_index("date")["close"]
    nliq = f["liquid"].sum(axis=1)
    cov = f["insti_cover"]
    res = {"meta": {"symbols": int(f["close"].shape[1]), "start": str(dates[0].date()), "end": str(dates[-1].date()),
                    "days": len(dates), "liquid_median": int(nliq.iloc[WARMUP:].median()),
                    "insti_days": int(cov.sum()) if cov is not None else 0,
                    "insti_first": str(cov[cov].index[0].date()) if cov is not None and cov.any() else None,
                    "rev_first": str(f["rev_yoy3"].dropna(how="all").index[0].date()) if f["rev_yoy3"] is not None else None}}
    print(json.dumps(res["meta"], ensure_ascii=False), flush=True)
    (OUT / "parts").mkdir(exist_ok=True)
    vs = variants_v4() if version == "v4" else variants_v3()
    todo = [(n, c) for n, c in vs.items() if not only or only in n]
    if shard:
        si, sn = (int(x) for x in shard.split("/"))
        todo = todo[si::sn]
    for name, cfg in todo:
        part = OUT / "parts" / f"{fname(name)}.json"
        if part.exists():
            res[name] = json.loads(part.read_text(encoding="utf-8"))
            continue
        hold = cfg["hold"]
        offs = list(range(0, hold, max(1, hold // 20)))
        seeds = tuple(range(n_seeds)) if cfg["sel"] == "rand" else (0,)
        runs = []
        for off in offs:
            for sd in seeds:
                tr, cv, sk = simulate(f, cfg, off, sd)
                t = tdf(tr)
                if len(t) and len(cv):
                    g = gate.check(t[["day", "net"]].copy(), False, None)
                    st = stats(cv)
                    b_ix = _bench(f["index"], cv.index[0], cv.index[-1])
                    b50 = _bench(e50, cv.index[0], cv.index[-1])
                else:
                    g, st, b_ix, b50 = {"verdict": "FAIL", "checks": []}, {"total": 0.0, "cagr": 0.0, "maxdd": 0.0}, None, None
                runs.append({"off": off, "seed": sd, **st, "n": len(t), "days": int(t.day.nunique()) if len(t) else 0,
                             "avg_net": round(float(t.net.mean()), 3) if len(t) else None,
                             "win": round(float((t.net > 0).mean() * 100), 1) if len(t) else None,
                             "stop_rate": round(float((t.reason == "stop").mean() * 100), 1) if len(t) else None,
                             "gate": g["verdict"], "gate_fail": [c["name"] for c in g["checks"] if not c["ok"]],
                             "bench_taiex": b_ix, "bench_0050": b50,
                             "start": str(cv.index[0].date()) if len(cv) else None})
                if off == 0 and sd == seeds[0]:
                    t.to_csv(OUT / "trades" / f"{fname(name)}.csv", index=False, encoding="utf-8-sig")
        o = pd.DataFrame(runs)
        beat = (o.total > o.bench_0050).mean() * 100 if o.bench_0050.notna().any() else None
        s = {"total_median": float(o.total.median()), "total_min": float(o.total.min()), "total_max": float(o.total.max()),
             "cagr_median": float(o.cagr.median()), "maxdd_median": float(o.maxdd.median()), "maxdd_worst": float(o.maxdd.min()),
             "avg_net_median": float(o.avg_net.median()) if o.avg_net.notna().any() else None,
             "win_median": float(o.win.median()) if o.win.notna().any() else None,
             "stop_rate_median": float(o.stop_rate.median()) if o.stop_rate.notna().any() else None,
             "n_median": float(o.n.median()), "days_median": float(o.days.median()),
             "gate_off0": o.iloc[0]["gate"], "gate_off0_fail": o.iloc[0]["gate_fail"],
             "gate_pass_share": round(float((o.gate == "PASS").mean() * 100), 0), "runs": len(o),
             "bench_0050_median": float(o.bench_0050.median()) if o.bench_0050.notna().any() else None,
             "bench_taiex_median": float(o.bench_taiex.median()) if o.bench_taiex.notna().any() else None,
             "beat_0050_share": round(float(beat), 0) if beat is not None else None}
        res[name] = {"cfg": cfg, "summary": s, "runs": runs}
        part.write_text(json.dumps(res[name], ensure_ascii=False, default=str), encoding="utf-8")
        print(f"{name:40s} 總報酬中位 {s['total_median']:+7.1f}%（{s['total_min']:+.0f}～{s['total_max']:+.0f}） 年化 {s['cagr_median']:+.1f}% "
              f"回落中位 {s['maxdd_median']:.1f}% 最差 {s['maxdd_worst']:.1f}% 每筆 {s['avg_net_median']}% 筆數 {s['n_median']:.0f} "
              f"| 0050 {s['bench_0050_median']}% 贏0050 {s['beat_0050_share']}% | gate0 {s['gate_off0']} PASS {s['gate_pass_share']:.0f}%", flush=True)
    out = OUT / (f"summary_{only or 'all'}{'_' + shard.replace('/', 'of') if shard else ''}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


def table(path: Path) -> str:
    """把 summary.json 轉成 Markdown 表格（報告用）。"""
    r = json.loads(path.read_text(encoding="utf-8"))
    rows = ["| 策略 | 起點0 總報酬 | 起點0 最大回落 | 筆數 | 每筆平均(淨) | 勝率 | 停損比例 | 起點0 gate | 不同起點總報酬 中位（最差～最好） | 最差回落 | 不同起點 PASS 比例 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in r.items():
        if name == "meta" or "summary" not in v:
            continue
        s, o = v["summary"], v["summary"]["off0"]
        rows.append(f"| {name} | {o['total']:+.1f}% | {o['maxdd']:.1f}% | {o['n']} | {o['avg_net']:+.2f}% | {o['win']}% | {o['stop_rate']}% | "
                    f"{o['gate']} | {s['total_median']:+.1f}%（{s['total_min']:+.1f}～{s['total_max']:+.1f}） | {s['maxdd_worst']:.1f}% | {s['gate_pass_share']:.0f}%（{s['runs']} 次） |")
    return "\n".join(rows)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        fetch()
    elif cmd == "run":
        mt = float(sys.argv[2]) if len(sys.argv) > 2 else 5e8
        sys.exit(run(mt))
    elif cmd == "run_v3":
        sys.exit(run_v3(only=sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "all" else None,
                        shard=sys.argv[3] if len(sys.argv) > 3 else None))
    elif cmd == "run_v4":
        sys.exit(run_v3(only=sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "all" else None,
                        shard=sys.argv[3] if len(sys.argv) > 3 else None, version="v4"))
    elif cmd in ("merge_v3", "merge_v4"):
        ver = cmd[-2:]
        allr = {}
        for p in sorted((BASE / "output" / f"strategy_research_{ver}").glob("summary_*.json")):
            allr.update(json.loads(p.read_text(encoding="utf-8")))
        (BASE / "output" / f"strategy_research_{ver}" / "summary.json").write_text(
            json.dumps(allr, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(len(allr) - 1, "個變形")
    elif cmd == "fetch_raw":
        fetch_raw()
    elif cmd == "fetch_ext":
        fetch_ext()
    elif cmd == "fetch_insti":
        fetch_insti(sys.argv[2], *(sys.argv[3:5]))
    elif cmd == "fetch_revenue":
        fetch_revenue()
    elif cmd == "table":
        print(table(Path(sys.argv[2])))
    elif cmd == "run_theme":
        sys.exit(run(5e8, tag="theme"))
