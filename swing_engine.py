"""題材波段紙上交易（模擬本金 30 萬，零股，只做多）。規則來自 theme_backtest.py 的投資組合模擬：

  每 20 個交易日換股一次：
  1. 題材動能＝網站「趨勢型態」27 個題材的成員 20 日等權報酬，取前 3 名題材
  2. 成員中收盤站上 20 日線、20 日平均成交值 > 5 億，依個股 20 日報酬取前 6 檔，等權
  3. 排除處置股、注意股、非正常交易股（用當天 Fugle ticker 判斷）；開盤漲停鎖死買不到就跳過

python swing_engine.py signal    # 收盤後（14:30 後）：更新日 K、以收盤價計算淨值；到換股日就產生隔天開盤的委託
python swing_engine.py execute   # 開盤後（09:05 後）：以今天開盤價（含零股滑價）成交待執行委託
python swing_engine.py status    # 列出持股、現金、淨值

狀態寫在 output/swing/：state.json、trades.csv、nav.csv。紙上交易，不連券商、不構成投資建議。
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import theme_backtest as tb  # noqa: E402
from daytrade_engine import eligibility, tick_size, to_tick  # noqa: E402
from stock_api import FugleClient  # noqa: E402

CAPITAL = 300_000
LOOK, TOP_THEMES, K, HOLD_DAYS = 20, 3, 6, 20
MIN_TURNOVER = 5e8
FEE, TAX, FEE_MIN, SLIP = 0.001425, 0.003, 1, 2
OUT = BASE / "output" / "swing"
STATE = OUT / "state.json"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"capital": CAPITAL, "cash": CAPITAL, "positions": {}, "pending": [], "days_since_rebalance": None,
            "last_rebalance": None, "last_signal_date": None, "nav": CAPITAL, "created_at": _now()}


def save_state(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def _append_csv(name: str, row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def update_daily(client: FugleClient, days: int = 120) -> None:
    """把 data/theme_bt/ 的還原日 K 補到今天（每檔 1 次 API）。"""
    tb.CACHE.mkdir(parents=True, exist_ok=True)
    end = date.today()
    syms = list(tb._themes()) + [tb.INDEX]
    for s in syms:
        new = None
        for attempt in range(3):
            try:
                new = client.historical_candles(s, end - timedelta(days=days), end, "D", adjusted=True)
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 2:
                    time.sleep(61)
                    continue
                print(f"{s} 日 K 更新失敗：{exc}", flush=True)
        if new is None:
            continue
        if new.empty:
            continue
        p = tb.CACHE / f"{s}.csv"
        old = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()
        # 還原股價會因除權息整段改變：新抓的區間整段覆蓋舊資料
        if not old.empty:
            old = old[old["date"] < new["date"].min()]
        pd.concat([old, new]).sort_values("date").to_csv(p, index=False, encoding="utf-8-sig")


def pick_targets(client: FugleClient) -> tuple[list[dict], list[str], pd.Timestamp]:
    close, px = tb.load()
    close = close.drop(columns=[tb.INDEX], errors="ignore")
    vol = px["volume"].drop(columns=[tb.INDEX], errors="ignore")
    th = px["themes"]
    d = close.index[-1]
    ts = tb.theme_scores(close, th, LOOK).loc[d].dropna().sort_values(ascending=False)
    ma20 = close.rolling(20).mean().loc[d]
    turnover = (close * vol).rolling(20).mean().loc[d]
    mom = close.pct_change(LOOK, fill_method=None).loc[d]

    def members(themes):
        c = [s for s, tl in th.items() if s in close and set(themes) & set(tl)
             and close.at[d, s] > ma20[s] and turnover[s] > MIN_TURNOVER and pd.notna(mom[s])]
        return sorted(c, key=lambda s: -mom[s])

    # 前 3 名題材；處置／注意股排除後不足 K 檔時，依序加入下一名題材（只用動能為正的題材）
    hot = list(ts.index[:TOP_THEMES])
    cand = members(hot)
    extra = [t for t in ts.index[TOP_THEMES:] if ts[t] > 0]
    picks, checked = [], set()
    while True:
        picks += _eligible(client, [s for s in cand if s not in checked], th, mom, close, d, K - len(picks))
        checked |= set(cand)
        if len(picks) >= K or not extra:
            break
        hot.append(extra.pop(0))
        cand = members(hot)
    return picks, [f"{t}（{ts[t] * 100:+.1f}%）" for t in hot], d


def _eligible(client, cand, th, mom, close, d, need):
    picks = []
    for s in cand:
        if need <= 0:
            break
        try:
            el = eligibility(client.ticker(s))
        except Exception as exc:
            print(f"{s} ticker 失敗：{exc}", flush=True)
            continue
        bad = [r for r in el["reasons"] if "當沖" not in r]   # 波段不需要當沖資格
        if bad:
            print(f"排除 {s} {el['name']}：{'、'.join(bad)}", flush=True)
            continue
        picks.append({"symbol": s, "name": el["name"], "themes": th[s], "mom20": round(float(mom[s]) * 100, 1),
                      "close": float(close.at[d, s])})
        need -= 1
    return picks


def signal() -> int:
    client = FugleClient(wait_on_limit=True)
    st = load_state()
    print("更新日 K…", flush=True)
    update_daily(client)
    close, _ = tb.load()
    d = close.index[-1]
    today = date.today()
    if d.date() != today:
        print(f"日 K 最新為 {d.date()}，今天 {today} 不是交易日或資料尚未更新，結束。")
        return 0
    if st.get("last_signal_date") == str(today):
        print("今天已經計算過，略過。")
        return 0
    # 以收盤價計算淨值
    mv = 0.0
    for s, p in st["positions"].items():
        c = float(close.at[d, s]) if s in close and pd.notna(close.at[d, s]) else p["last"]
        p["last"] = c
        mv += c * p["shares"]
    st["nav"] = round(st["cash"] + mv)
    if st["days_since_rebalance"] is not None:
        st["days_since_rebalance"] += 1
    _append_csv("nav.csv", {"date": str(today), "nav": st["nav"], "cash": round(st["cash"]), "positions": len(st["positions"]),
                            "return_pct": round((st["nav"] / st["capital"] - 1) * 100, 2)})
    due = st["days_since_rebalance"] is None or st["days_since_rebalance"] >= HOLD_DAYS
    msg = f"{today} 淨值 {st['nav']:,}（{(st['nav'] / st['capital'] - 1) * 100:+.2f}%），現金 {st['cash']:,.0f}，持股 {len(st['positions'])} 檔"
    if due:
        picks, hot, _ = pick_targets(client)
        keep = {p["symbol"] for p in picks}
        orders = [{"side": "sell", "symbol": s, "shares": p["shares"], "name": p.get("name", ""), "reason": "換股：不在新名單"}
                  for s, p in st["positions"].items() if s not in keep]
        per = st["nav"] / K
        for p in picks:
            held = st["positions"].get(p["symbol"])
            have = held["shares"] * held["last"] if held else 0
            if per - have > per * 0.2:      # 差距超過 20% 才補，避免小額來回交易
                orders.append({"side": "buy", "symbol": p["symbol"], "name": p["name"], "value": round(per - have),
                               "reason": f"題材 {'、'.join(p['themes'])}，20 日 {p['mom20']:+.1f}%"})
        st["pending"] = orders
        st["days_since_rebalance"] = 0
        st["last_rebalance"] = str(today)
        st["last_picks"] = picks
        st["last_hot_themes"] = hot
        msg += f"\n換股日：強勢題材 {'、'.join(hot)}\n名單：" + "、".join(f"{p['symbol']}{p['name']}({p['mom20']:+.1f}%)" for p in picks)
        msg += f"\n明天開盤委託 {len(orders)} 筆：" + "；".join(f"{o['side']} {o['symbol']}{o.get('name', '')}" for o in orders)
    else:
        msg += f"\n距下次換股還有 {HOLD_DAYS - st['days_since_rebalance']} 個交易日"
    st["last_signal_date"] = str(today)
    save_state(st)
    print(msg)
    return 0


def _cost(value: float, sell: bool) -> float:
    return max(FEE_MIN, value * FEE) + (value * TAX if sell else 0)


def execute() -> int:
    st = load_state()
    if not st["pending"]:
        print("沒有待執行委託。")
        return 0
    client = FugleClient(wait_on_limit=True)
    done, left = [], []
    for o in sorted(st["pending"], key=lambda o: o["side"] != "sell"):   # 先賣後買
        s = o["symbol"]
        q = client.quote(s)
        op = float(q.get("openPrice") or 0)
        if not op:
            left.append(o); print(f"{s} 尚未開盤，保留委託"); continue
        t = client.ticker(s)
        if o["side"] == "sell":
            p = st["positions"].get(s)
            if not p:
                continue
            fill = to_tick(op - SLIP * tick_size(op), up=False)
            value = fill * p["shares"]
            cost = _cost(value, True)
            st["cash"] += value - cost
            pnl = value - cost - p["cost_total"]
            row = {"time": _now(), "side": "賣", "symbol": s, "name": p.get("name", ""), "shares": p["shares"], "price": fill,
                   "value": round(value), "cost": round(cost, 1), "pnl": round(pnl), "reason": o["reason"]}
            del st["positions"][s]
        else:
            el = eligibility(t)
            bad = [r for r in el["reasons"] if "當沖" not in r]
            if bad:
                print(f"{s} 今天{'、'.join(bad)}，不買"); continue
            if el["limit_up"] and op >= el["limit_up"]:
                print(f"{s} 開盤漲停鎖死，買不到，不買"); continue
            fill = to_tick(op + SLIP * tick_size(op), up=True)
            budget = min(o["value"], st["cash"])
            shares = int(budget / (fill * (1 + FEE)))
            if shares < 1:
                print(f"{s} 現金不足，不買"); continue
            value = fill * shares
            cost = _cost(value, False)
            st["cash"] -= value + cost
            p = st["positions"].setdefault(s, {"shares": 0, "cost_total": 0.0, "name": el["name"], "entry_date": str(date.today())})
            p["shares"] += shares
            p["cost_total"] += value + cost
            p["avg"] = round(p["cost_total"] / p["shares"], 2)
            p["last"] = fill
            row = {"time": _now(), "side": "買", "symbol": s, "name": el["name"], "shares": shares, "price": fill,
                   "value": round(value), "cost": round(cost, 1), "pnl": "", "reason": o["reason"]}
        _append_csv("trades.csv", row)
        done.append(row)
        print(f"{row['side']} {s}{row['name']} {row['shares']} 股 @ {row['price']}（{row['reason']}）")
    st["pending"] = left
    save_state(st)
    print(f"完成 {len(done)} 筆，現金 {st['cash']:,.0f}")
    return 0


def status() -> int:
    st = load_state()
    print(f"本金 {st['capital']:,}，淨值 {st['nav']:,}（{(st['nav'] / st['capital'] - 1) * 100:+.2f}%），現金 {st['cash']:,.0f}")
    print(f"上次換股 {st.get('last_rebalance')}，之後已過 {st.get('days_since_rebalance')} 個交易日；強勢題材 {st.get('last_hot_themes')}")
    for s, p in st["positions"].items():
        print(f"  {s}{p.get('name', '')} {p['shares']} 股，成本 {p['avg']}，最新 {p['last']}，損益 {p['last'] * p['shares'] - p['cost_total']:+,.0f}")
    for o in st["pending"]:
        print(f"  待執行：{o['side']} {o['symbol']}{o.get('name', '')}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"signal": signal, "execute": execute, "status": status}[cmd]())
