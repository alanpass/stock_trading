"""題材強弱回測：專案「趨勢型態」頁的 27 個題材（175 檔），用每天以前的資料算題材動能，
測試「強勢題材股」的盤中當沖與 1～5 日持有，和全部候選股比較。

python theme_backtest.py fetch   # 下載 3 年還原日 K 到 data/theme_bt/（Fugle 每分鐘 60 次，約 10 分鐘）
python theme_backtest.py         # 回測，結果印出並寫入 output/daytrade/theme_backtest.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
os.chdir(BASE); sys.path.insert(0, str(BASE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade_engine import tick_size  # noqa: E402

CACHE = BASE / "data" / "theme_bt"
INDEX = "IX0001"
FEE, TAX_DT, TAX = 0.001425, 0.003, 0.003   # 零股當沖：證交稅以 0.3% 保守計；非當沖 0.3%
SLIP = 2                                    # 零股買賣各 2 個 tick


def _themes() -> dict[str, list[str]]:
    rows = json.loads((BASE / "output" / "research_reports" / "trend_patterns_public.json").read_text(encoding="utf-8"))["stocks"]
    return {r["symbol"]: list(r.get("themes") or []) for r in rows if r.get("themes")}


def fetch() -> None:
    from stock_api import FugleClient
    c = FugleClient(wait_on_limit=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    end = date.today()
    syms = list(_themes()) + [INDEX]
    for n, s in enumerate(syms, 1):
        if (CACHE / f"{s}.csv").exists():
            continue
        parts = []
        try:
            for y in range(3):
                e = end - timedelta(days=365 * y)
                parts.append(c.historical_candles(s, e - timedelta(days=364), e, "D", adjusted=True))
        except Exception as exc:
            print(f"{n}/{len(syms)} {s} 失敗：{exc}", flush=True)
            continue
        d = pd.concat(parts).drop_duplicates("date").sort_values("date")
        d.to_csv(CACHE / f"{s}.csv", index=False, encoding="utf-8-sig")
        print(f"{n}/{len(syms)} {s} {len(d)} 筆", flush=True)


def load() -> tuple[pd.DataFrame, dict]:
    th = _themes()
    frames = {}
    for s in list(th) + [INDEX]:
        p = CACHE / f"{s}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, parse_dates=["date"])
        if len(d) > 100:
            frames[s] = d.set_index("date")
    close = pd.DataFrame({s: f["close"] for s, f in frames.items()}).sort_index()
    return close, {"open": pd.DataFrame({s: f["open"] for s, f in frames.items()}).sort_index(),
                   "high": pd.DataFrame({s: f["high"] for s, f in frames.items()}).sort_index(),
                   "low": pd.DataFrame({s: f["low"] for s, f in frames.items()}).sort_index(),
                   "volume": pd.DataFrame({s: f["volume"] for s, f in frames.items()}).sort_index(),
                   "themes": th}


def theme_scores(close: pd.DataFrame, th: dict, look: int) -> pd.DataFrame:
    """每天收盤時各題材的 look 日平均報酬（等權）。"""
    r = close.pct_change(look, fill_method=None)
    names = sorted({t for ts in th.values() for t in ts})
    out = {}
    for t in names:
        mem = [s for s, ts in th.items() if t in ts and s in close]
        if len(mem) >= 3:
            out[t] = r[mem].mean(axis=1)
    return pd.DataFrame(out)


def _cost(entry: float, exit_: float, tax: float) -> float:
    e2 = entry + SLIP * tick_size(entry); x2 = exit_ - SLIP * tick_size(exit_)
    return x2 / e2 - 1 - (e2 * FEE + x2 * FEE + x2 * tax) / e2


def swing(close, px, look=10, top=3, hold=3, require_trend=True, label=""):
    """訊號日 d 收盤：題材 look 日動能前 top 名的成員（且個股站上 20 日線），d+1 開盤買、持有 hold 天後收盤賣。"""
    th = px["themes"]
    ts = theme_scores(close, th, look)
    ma20 = close.rolling(20).mean()
    turnover = (close * px["volume"]).rolling(20).mean()
    o = px["open"]
    idx = close.index
    rows = []
    for i in range(60, len(idx) - hold - 1):
        d = idx[i]
        sc = ts.loc[d].dropna()
        if sc.empty:
            continue
        hot = set(sc.sort_values(ascending=False).index[:top])
        cold = set(sc.sort_values().index[:top])
        for s, tl in th.items():
            if s not in close or pd.isna(close.at[d, s]) or pd.isna(o.iat[i + 1, o.columns.get_loc(s)]):
                continue
            if turnover.at[d, s] < 5e8 / 1000 * 1:   # 日 K 量是股；20 日平均成交值 > 5 億
                pass
            grp = "hot" if hot & set(tl) else ("cold" if cold & set(tl) else "mid")
            trend_ok = close.at[d, s] > ma20.at[d, s]
            e = float(o.iat[i + 1, o.columns.get_loc(s)])
            x = float(close.iat[i + hold, close.columns.get_loc(s)])
            if not (e > 0 and x > 0):
                continue
            rows.append({"day": d, "s": s, "grp": grp, "trend": trend_ok, "gross": x / e - 1,
                         "net": _cost(e, x, TAX)})
    t = pd.DataFrame(rows)
    return t


def summarize(t: pd.DataFrame, name: str) -> dict:
    if t.empty:
        print(name, "無資料"); return {}
    days = sorted(t.day.unique()); cut1, cut2 = days[len(days) // 3], days[2 * len(days) // 3]
    def seg(x):
        return {"n": int(len(x)), "gross": round(x.gross.mean() * 100, 2), "net": round(x.net.mean() * 100, 2),
                "win": round((x.net > 0).mean() * 100, 0)}
    res = {"all": seg(t), "p1": seg(t[t.day < cut1]), "p2": seg(t[(t.day >= cut1) & (t.day < cut2)]), "p3": seg(t[t.day >= cut2])}
    f = lambda r: f"n{r['n']:5d} 毛{r['gross']:+.2f}% 淨{r['net']:+.2f}% 勝{r['win']:.0f}%"
    print(f"{name:34s} | 全 {f(res['all'])} | 前 {f(res['p1'])} | 中 {f(res['p2'])} | 後 {f(res['p3'])}", flush=True)
    return res


def portfolio(close, px, idx_close, look=10, top=3, k=6, hold=5, slip_pct=0.002, market_filter=True,
              pick="momentum") -> dict:
    """投資組合模擬：每 hold 個交易日換股一次（收盤決定、隔天開盤成交）。
    選股：題材 look 日動能前 top 名 → 成員中站上 20 日線、20 日平均成交值 > 5 億 → 依個股 look 日報酬取前 k 檔，等權。
    market_filter：加權指數收盤在 60 日線之下時空手。成本：換掉的部位收買賣手續費、證交稅 0.3%、滑價 slip_pct（單邊）。"""
    th = px["themes"]
    ts = theme_scores(close, th, look)
    ma20 = close.rolling(20).mean()
    turnover = (close * px["volume"]).rolling(20).mean()
    mom = close.pct_change(look, fill_method=None)
    o = px["open"]
    ima60 = idx_close.rolling(60).mean()
    dates = close.index
    eq, cur, curve = 1.0, {}, []
    for i in range(60, len(dates) - 1, hold):
        d = dates[i]
        j = min(i + hold, len(dates) - 1)
        new: dict = {}
        if not market_filter or idx_close.get(d, np.nan) >= ima60.get(d, np.nan):
            sc = ts.loc[d].dropna().sort_values(ascending=False)
            hot = set(sc.index[:top]) if pick != "all" else set(sc.index)
            cand = [s for s, tl in th.items() if s in close and hot & set(tl)
                    and close.at[d, s] > ma20.at[d, s] and turnover.at[d, s] > 5e8]
            cand = sorted(cand, key=lambda s: -(mom.at[d, s] if pd.notna(mom.at[d, s]) else -9))[:k]
            new = {s: 1 / len(cand) for s in cand} if cand else {}
        # 換股成本
        traded = sum(abs(new.get(s, 0) - cur.get(s, 0)) for s in set(new) | set(cur))
        sells = sum(max(cur.get(s, 0) - new.get(s, 0), 0) for s in set(cur) | set(new))
        cost = traded * (FEE + slip_pct) + sells * TAX
        # d+1 開盤到 j+1 開盤（最後一段用收盤）
        a, z = dates[i + 1], dates[min(j + 1, len(dates) - 1)]
        r = 0.0
        for s, w in new.items():
            pa, pz = o.at[a, s], (o.at[z, s] if z != dates[-1] else close.at[z, s])
            if pd.notna(pa) and pd.notna(pz) and pa > 0:
                r += w * (pz / pa - 1)
        eq *= (1 + r) * (1 - cost)
        curve.append((z, eq, len(new)))
        cur = new
    c = pd.DataFrame(curve, columns=["date", "equity", "n"]).set_index("date")
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    dd = (c.equity / c.equity.cummax() - 1).min()
    per = c.equity.pct_change().dropna()
    thirds = np.array_split(c.equity, 3)
    seg = [round((t.iloc[-1] / t.iloc[0] - 1) * 100, 1) for t in thirds]
    return {"total": round((c.equity.iloc[-1] - 1) * 100, 1), "cagr": round((c.equity.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
            "maxdd": round(dd * 100, 1), "sharpe": round(per.mean() / per.std() * np.sqrt(252 / hold), 2) if per.std() > 0 else None,
            "thirds": seg, "invested": round((c.n > 0).mean() * 100), "start": str(c.index[0].date()), "end": str(c.index[-1].date())}


def run_portfolios() -> dict:
    close, px = load()
    idx_close = close[INDEX] if INDEX in close else pd.Series(dtype=float)
    close = close.drop(columns=[INDEX], errors="ignore")
    px = {k: (v.drop(columns=[INDEX], errors="ignore") if isinstance(v, pd.DataFrame) else v) for k, v in px.items()}
    print(f"股票 {close.shape[1]} 檔，{close.index[0].date()} ～ {close.index[-1].date()}", flush=True)
    # 基準：加權指數買進持有、全部候選股等權（每 5 天再平衡，同樣成本）
    yrs = (close.index[-1] - close.index[60]).days / 365.25
    ib = idx_close.iloc[-1] / idx_close.iloc[60]
    res = {"加權指數買進持有": {"total": round((ib - 1) * 100, 1), "cagr": round((ib ** (1 / yrs) - 1) * 100, 1),
                          "maxdd": round(((idx_close.iloc[60:] / idx_close.iloc[60:].cummax()) - 1).min() * 100, 1)}}
    res["全部題材股等權（無大盤濾網）"] = portfolio(close, px, idx_close, pick="all", k=999, market_filter=False)
    for look in (5, 10, 20):
        for hold in (5, 10, 20):
            for k in (3, 6):
                for mf in (True, False):
                    name = f"題材{look}日前3名 取{k}檔 每{hold}天換 {'大盤濾網' if mf else '無濾網'}"
                    res[name] = portfolio(close, px, idx_close, look=look, hold=hold, k=k, market_filter=mf)
    for name, r in res.items():
        print(f"{name:34s} 總報酬 {r.get('total'):+7.1f}%  年化 {r.get('cagr'):+6.1f}%  最大回落 {r.get('maxdd'):6.1f}%"
              f"  夏普 {r.get('sharpe')}  三段 {r.get('thirds')}  持股時間 {r.get('invested')}%", flush=True)
    return res


def intraday(look: int = 10, top: int = 3) -> dict:
    """盤中當沖（5 分 K 快取，約 71 天）：只在強勢／弱勢題材股裡，比較幾種進場方式。
    訊號在 5 分 K 收盤 → 下一根開盤買，13:15 收盤或 −5% 停損出場，每天每組最多 3 筆。"""
    import daytrade_backtest as b
    th = _themes()
    k5, dly = {}, {}
    for s in th:
        k = b._load_5m(None, s)
        if not k.empty:
            k5[s], dly[s] = k, b._daily_from_5m(k)
    close = pd.DataFrame({s: d["close"] for s, d in dly.items()}).sort_index()
    ts = theme_scores(close, th, look)
    ma20 = close.rolling(20).mean()
    days = [d for d in close.index[40:]]

    def trade(g, i, stop=0.05):
        if i + 1 >= len(g) or g.time.iloc[i + 1] >= "13:15":
            return None
        e = float(g.open.iloc[i + 1]); st = e * (1 - stop)
        for j in range(i + 1, len(g)):
            if g.low.iloc[j] <= st:
                return e, st
            if g.time.iloc[j] >= "13:15":
                return e, float(g.close.iloc[j])
        return e, float(g.close.iloc[-1])

    def first(mask):
        w = np.flatnonzero(mask.values)
        return int(w[0]) if len(w) else None

    sigs = {
        "開盤第一根收紅就買（09:05）": lambda g: 0 if g.close.iloc[0] > g.open.iloc[0] else None,
        "ORB 15 分突破＋VWAP": lambda g: first((g.time >= "09:15") & (g.close > g[g.time < "09:15"].high.max()) & (g.close > g.vwap)),
        "回測 VWAP 後收回（拉回買）": None,
        "跌深 >3% 後收回 VWAP": None,
        "跌深反彈 09:30（現行）": lambda g: (lambda i: int(i[0]) if len(i) and g.close.iloc[i[0]] / g.open.iloc[0] - 1 < -0.04 and g.close.iloc[i[0]] < g.vwap.iloc[i[0]] * 0.99 else None)(g.index[g.time == "09:25"]),
    }

    def pullback(g):           # 曾在 VWAP 之上 0.5% 以上，回到 VWAP 下方，再收回 VWAP 之上（強勢股拉回）
        above = dipped = False
        for i in range(len(g)):
            c, v = g.close.iloc[i], g.vwap.iloc[i]
            if g.time.iloc[i] > "11:30":
                return None
            if c > v * 1.005: above = True
            if above and c < v: dipped = True
            elif dipped and c > v: return i
        return None

    def deep_reclaim(g):
        armed = False
        for i in range(len(g)):
            c, v = g.close.iloc[i], g.vwap.iloc[i]
            if g.time.iloc[i] > "11:30":
                return None
            if c / g.open.iloc[0] - 1 < -0.03 and c < v * 0.985: armed = True
            elif armed and c > v: return i
        return None
    sigs["回測 VWAP 後收回（拉回買）"] = pullback
    sigs["跌深 >3% 後收回 VWAP"] = deep_reclaim

    res = {}
    for name, sig in sigs.items():
        rows = []
        for d in days:
            sc = ts.loc[d].dropna() if d in ts.index else pd.Series(dtype=float)
            if sc.empty:
                continue
            # 題材分數用前一天收盤（避免偷看當天）
            prev = ts.index[ts.index.get_loc(d) - 1]
            sc = ts.loc[prev].dropna()
            hot = set(sc.sort_values(ascending=False).index[:top]); cold = set(sc.sort_values().index[:top])
            per = {"hot": [], "cold": [], "mid": []}
            for s, tl in th.items():
                if s not in k5 or d not in dly[s].index:
                    continue
                r = dly[s].loc[d]
                if pd.isna(r.atr_pct) or r.atr_pct < 0.03 or r.turnover20 < 5e8:
                    continue
                g = k5[s][k5[s].day == d].reset_index(drop=True)
                if len(g) < 50:
                    continue
                tp = (g.high + g.low + g.close) / 3
                g["vwap"] = ((tp * g.volume).cumsum() / g.volume.cumsum().replace(0, np.nan)).fillna(tp)
                i = sig(g)
                if i is None or g.time.iloc[i] > "11:30":
                    continue
                grp = "hot" if hot & set(tl) else ("cold" if cold & set(tl) else "mid")
                trend = close.at[prev, s] > ma20.at[prev, s] if s in close else False
                per[grp].append((g.time.iloc[i], s, g, i, trend))
            for grp, lst in per.items():
                for tm, s, g, i, trend in sorted(lst, key=lambda x: x[0])[:3]:
                    r = trade(g, i)
                    if r:
                        rows.append({"day": d, "s": s, "grp": grp, "trend": trend, "gross": r[1] / r[0] - 1,
                                     "net": _cost(r[0], r[1], TAX_DT)})
        t = pd.DataFrame(rows)
        print(f"\n--- 盤中：{name}（題材動能 {look} 日） ---")
        for g in ("hot", "mid", "cold"):
            res[f"{name}_{g}"] = summarize(t[t.grp == g] if not t.empty else t, g)
    return res


def main() -> int:
    close, px = load()
    print(f"股票 {close.shape[1] - 1} 檔，{close.index[0].date()} ～ {close.index[-1].date()}，{len(close)} 天", flush=True)
    close = close.drop(columns=[INDEX], errors="ignore")
    px = {k: (v.drop(columns=[INDEX], errors="ignore") if isinstance(v, pd.DataFrame) else v) for k, v in px.items()}
    out = {}
    for look in (5, 10, 20):
        for hold in (1, 3, 5):
            t = swing(close, px, look=look, hold=hold)
            print(f"\n--- 題材動能 {look} 日、持有 {hold} 天（d+1 開盤買、第 {hold} 天收盤賣，零股成本） ---")
            for g in ("hot", "mid", "cold"):
                x = t[t.grp == g]
                out[f"look{look}_hold{hold}_{g}"] = summarize(x, f"{g}")
                out[f"look{look}_hold{hold}_{g}_trend"] = summarize(x[x.trend], f"{g}＋站上20日線")
    (BASE / "output" / "daytrade").mkdir(parents=True, exist_ok=True)
    (BASE / "output" / "daytrade" / "theme_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    elif len(sys.argv) > 1 and sys.argv[1] == "portfolio":
        r = run_portfolios()
        (BASE / "output" / "daytrade" / "theme_portfolio.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    elif len(sys.argv) > 1 and sys.argv[1] == "intraday":
        r = intraday()
        (BASE / "output" / "daytrade" / "theme_intraday.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        sys.exit(main())
