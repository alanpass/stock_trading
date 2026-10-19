"""當沖進場時機研究：等反轉確認（第二隻腳、收回 VWAP、錘子線）、跳空回補、開盤八法、當日高低點時間分布。
用 daytrade_backtest.py 的 5 分 K 快取；結果見 output/daytrade/strategy_research.md 第五節。"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from datetime import timedelta
import numpy as np, pandas as pd
import daytrade_backtest as b
from daytrade_engine import _trend_patterns, _watchlist, tick_size
from stock_api import clean_symbol
pats = _trend_patterns()
uni = list(dict.fromkeys([clean_symbol(x) for x in _watchlist()] + list(pats)))
idx = b._load_5m(None, "IX0001"); ix = {d: g.reset_index(drop=True) for d, g in idx.groupby("day")}
data = {}
for s in uni:
    k = b._load_5m(None, s)
    if not k.empty: data[s] = (k, b._daily_from_5m(k))
alldays = sorted(ix)
alldays = [d for d in alldays if d >= alldays[0] + timedelta(days=35)]
half = alldays[len(alldays) // 2]
EXIT = "13:15"


def day_universe(d):
    c = []
    for s, (k, dd) in data.items():
        if d not in dd.index: continue
        r = dd.loc[d]
        if pd.isna(r.atr_pct) or r.atr_pct < 0.03 or r.turnover20 < 5e8: continue
        c.append((r.turnover20, s))
    return [s for _, s in sorted(c, reverse=True)[:40]]


days = {}
for d in alldays:
    u = {}
    for s in day_universe(d):
        k, dd = data[s]
        g = k[k.day == d].reset_index(drop=True)
        if len(g) < 50: continue
        tp = (g.high + g.low + g.close) / 3
        g["vwap"] = (tp * g.volume).cumsum() / g.volume.cumsum().replace(0, np.nan)
        g["vwap"] = g["vwap"].fillna(tp)
        g["prev_close"] = float(dd.loc[d, "prev_close"])
        u[s] = g
    days[d] = u
print("days", len(alldays), "first", alldays[0], "half", half, "last", alldays[-1], flush=True)


def trade(g, i, stop=0.05):
    """訊號在第 i 根收盤 → 下一根開盤買進；13:15 那根收盤或 −5% 停損出場。"""
    if i + 1 >= len(g) or g.time.iloc[i + 1] >= EXIT: return None
    e = float(g.open.iloc[i + 1]); st = e * (1 - stop)
    for j in range(i + 1, len(g)):
        if g.low.iloc[j] <= st:
            return e, st, "stop"
        if g.time.iloc[j] >= EXIT:
            return e, float(g.close.iloc[j]), "time"
    return e, float(g.close.iloc[-1]), "end"


def net(e, x, slip, tax):
    e2 = e + slip * tick_size(e); x2 = x - slip * tick_size(x)
    return (x2 / e2 - 1 - (e2 * 0.001425 + x2 * 0.001425 + x2 * tax) / e2) * 100


def evaluate(name, trades):
    if not trades:
        print(f"{name} | 無交易"); return
    t = pd.DataFrame(trades)
    t["gross"] = (t.x / t.e - 1) * 100
    t["odd2"] = [net(a, c, 2, 0.003) for a, c in zip(t.e, t.x)]
    t["whole1"] = [net(a, c, 1, 0.0015) for a, c in zip(t.e, t.x)]
    t["excess"] = t.gross - t.bench

    def seg(x):
        if x.empty: return "n0"
        return f"n{len(x)} 毛{x.gross.mean():+.2f}% 超額{x.excess.mean():+.2f}% 勝{(x.gross > 0).mean() * 100:.0f}%"
    a, bb = t[t.day < half], t[t.day >= half]
    print(f"{name} | 全 {seg(t)} 零股淨{t.odd2.mean():+.2f}% 整股淨{t.whole1.mean():+.2f}% | 前半 {seg(a)} | 後半 {seg(bb)}", flush=True)
    return t


def bench_from(u, tm):
    """同一天所有候選股從 tm 下一根開盤到 13:15 的平均（隨機進場的基準）。"""
    r = []
    for g in u.values():
        a = g[g.time > tm]; z = g[g.time == EXIT]
        if a.empty or z.empty: continue
        r.append(float(z.close.iloc[0]) / float(a.open.iloc[0]) - 1)
    return np.mean(r) * 100 if r else 0.0


def run(name, signal, last="11:30", idx_cond=None, maxper=3):
    trades = []
    for d, u in days.items():
        I = ix[d]
        cands = []
        for s, g in u.items():
            i = signal(g, I)
            if i is None or g.time.iloc[i] > last: continue
            cands.append((g.time.iloc[i], s, i))
        cands.sort()
        n = 0
        for tm, s, i in cands:
            if n >= maxper: break
            if idx_cond and not idx_cond(I, tm): continue
            r = trade(u[s], i)
            if r:
                n += 1
                trades.append({"day": d, "s": s, "t": tm, "e": r[0], "x": r[1], "why": r[2], "bench": bench_from(u, tm)})
    return evaluate(name, trades)


def first(mask):
    w = np.flatnonzero(mask.values)
    return int(w[0]) if len(w) else None


def sig_meanrev(g, I):
    i = g.index[g.time == "09:25"]
    if not len(i): return None
    i = i[0]; o = g.open.iloc[0]; c = g.close.iloc[i]
    return i if c / o - 1 < -0.04 and c / g.vwap.iloc[i] - 1 < -0.01 else None


def sig_second_leg(th):
    # 第二隻腳：先從開盤跌超過 th 形成低點 L，反彈 ≥1.5% 後回測不破 L（且在 L 的 2% 內），收盤突破前一根高點的紅 K
    def f(g, I):
        o = g.open.iloc[0]; L = np.inf; bounced = False; pl = np.inf
        for i in range(len(g)):
            if g.time.iloc[i] > "11:30": return None
            lo, c = g.low.iloc[i], g.close.iloc[i]
            if lo < L:
                L, bounced, pl = lo, False, np.inf
                continue
            if L / o - 1 > -th: continue
            if not bounced:
                if g.high.iloc[i] >= L * 1.015: bounced = True
                continue
            pl = min(pl, lo)
            if pl <= L * 1.02 and c > g.high.iloc[i - 1] and c > g.open.iloc[i]:
                return i
        return None
    return f


def sig_vwap_reclaim(th):
    def f(g, I):
        o = g.open.iloc[0]; armed = False
        for i in range(len(g)):
            c = g.close.iloc[i]
            if c / o - 1 < -th and c / g.vwap.iloc[i] - 1 < -0.015: armed = True
            elif armed and c > g.vwap.iloc[i]: return i
        return None
    return f


def sig_hammer(th):
    def f(g, I):
        o = g.open.iloc[0]; L = np.inf
        for i in range(len(g)):
            lo = g.low.iloc[i]; newlow = lo < L; L = min(L, lo)
            if not newlow or L / o - 1 > -th: continue
            op, c, h = g.open.iloc[i], g.close.iloc[i], g.high.iloc[i]
            body = abs(c - op); rng = h - lo; lower = min(c, op) - lo
            if rng > 0 and lower >= 2 * max(body, rng * 0.05) and (c - lo) / rng > 0.6: return i
        return None
    return f


def sig_orb(g, I):
    orb = g[g.time < "09:15"]
    if len(orb) < 3: return None
    h = orb.high.max()
    return first((g.time >= "09:15") & (g.close > h) & (g.close > g.vwap))


def sig_gapdown_fill(g, I):
    pc = g.prev_close.iloc[0]; o = g.open.iloc[0]
    if o / pc - 1 > -0.02: return None
    return first(g.close > (o + pc) / 2)


def _ivwap(m):
    tp = (m.high + m.low + m.close) / 3
    return (tp * m.volume).sum() / m.volume.sum() if m.volume.sum() > 0 else tp.mean()


def idx_above_vwap(I, tm):
    m = I[I.time <= tm]
    return True if m.empty else m.close.iloc[-1] >= _ivwap(m)


def idx_higher_low(I, tm):   # 大盤最近 30 分鐘沒有再創當日新低
    m = I[I.time <= tm]
    if len(m) < 7: return False
    return m.low.iloc[-6:].min() > m.low.min()


print("\n=== 現行與對照 ===")
run("跌深反彈 09:30（現行）", sig_meanrev)
run("ORB 15 分＋VWAP", sig_orb)
print("\n=== 等反轉確認再進場 ===")
for th in (0.03, 0.04):
    run(f"第二隻腳 跌>{th:.0%}", sig_second_leg(th))
    run(f"第二隻腳 跌>{th:.0%}＋大盤站上VWAP", sig_second_leg(th), idx_cond=idx_above_vwap)
    run(f"第二隻腳 跌>{th:.0%}＋大盤止跌", sig_second_leg(th), idx_cond=idx_higher_low)
    run(f"收回VWAP 跌>{th:.0%}", sig_vwap_reclaim(th))
    run(f"收回VWAP 跌>{th:.0%}＋大盤站上VWAP", sig_vwap_reclaim(th), idx_cond=idx_above_vwap)
    run(f"錘子線 跌>{th:.0%}", sig_hammer(th))
    run(f"錘子線 跌>{th:.0%}＋大盤止跌", sig_hammer(th), idx_cond=idx_higher_low)
run("跳空跌>2% 回補一半", sig_gapdown_fill)
run("跳空跌>2% 回補一半＋大盤站上VWAP", sig_gapdown_fill, idx_cond=idx_above_vwap)

print("\n=== 個股當日最低點出現時間（候選股，5 分 K 開始時間） ===")
lows, lows_dn, highs = [], [], []
for d, u in days.items():
    for s, g in u.items():
        g2 = g[g.time < "13:25"]
        lows.append(g2.time.iloc[int(g2.low.values.argmin())])
        highs.append(g2.time.iloc[int(g2.high.values.argmax())])
        r = g2[g2.time == "09:25"]
        if len(r) and r.close.iloc[0] / g2.open.iloc[0] - 1 < -0.03:
            lows_dn.append(g2.time.iloc[int(g2.low.values.argmin())])


def dist(x):
    x = pd.Series(x)
    return " ".join(f"≤{c}:{(x <= c).mean() * 100:.0f}%" for c in ("09:00", "09:25", "09:55", "10:25", "11:00", "12:00", "13:00"))


print("最低點 全部         ", dist(lows), f"(n={len(lows)})")
print("最低點 09:30前跌>3%  ", dist(lows_dn), f"(n={len(lows_dn)})")
print("最高點 全部         ", dist(highs))

print("\n=== 開盤八法（大盤 09:00/09:05/09:10 三根 5 分 K 漲跌）→ 09:15～13:15 候選股平均 ===")
rows = []
for d, u in days.items():
    I = ix[d]
    if len(I) < 4: continue
    key = "".join("漲" if I.close.iloc[i] > (I.open.iloc[0] if i == 0 else I.close.iloc[i - 1]) else "跌" for i in range(3))
    rows.append({"day": d, "k": key, "ret": bench_from(u, "09:10")})
r = pd.DataFrame(rows)
for k, g in r.groupby("k"):
    a, bb = g[g.day < half], g[g.day >= half]
    print(f"{k} 天數{len(g):3d} 平均 {g.ret.mean():+.2f}% 上漲天 {(g.ret > 0).mean() * 100:3.0f}% | 前半 {a.ret.mean():+.2f}%({len(a)}) 後半 {bb.ret.mean():+.2f}%({len(bb)})")
