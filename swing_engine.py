"""波段紙上交易引擎 v2（模擬本金 30 萬，只做多）。不連券商、不下真實單、不構成投資建議。

策略（2026-10-24 起，來源 universe_backtest.py v4 未還原價回測，詳見 output/strategy_final.md）：設定在 STRATEGY。
  1. 股票池：全部上市櫃普通股；訊號日 20 日平均成交值 > 5 億、真實股價 ≤ 80 元（只用當天以前的資料）
  2. 選股：收盤站上 20 日線，依「近 3 個月營收合計年增率」由高到低（只用已過公告期限的月份）
  3. 同產業最多 2 檔、等權 K 檔；每 HOLD 個交易日整批換股
  4. 停損：STRATEGY["atr_k"]＝None 表示不停損（v4 回測 3×ATR 讓年化從 16.1% 降到 12.4%、回落沒有變小）；
     設成數字時：收盤跌破「進場價 − k × ATR14」→ 隔天開盤賣，空出的資金留到下次換股
  5. 進場（隔天開盤）：處置股／注意股／非正常交易排除；開盤一價漲停鎖死不買；開盤跳空 > 3% 不追；
     只用手上現金買（不會違約交割）；整股模式股數取 1000 的倍數

python swing_engine.py signal [--dry-run] [--out DIR]   # 收盤後 17:30 之後：更新資料、算淨值、停損檢查、到期換股
python swing_engine.py execute                           # 開盤後 09:05 之後：以今天開盤價成交待執行委託
python swing_engine.py status

狀態寫在 output/swing/：state.json、trades.csv、nav.csv（格式與 v1 相同，trading-review/perf_summary.py 可直接讀）。
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

import universe_backtest as ub  # noqa: E402
from daytrade_engine import eligibility, tick_size, to_tick  # noqa: E402
from stock_api import FugleClient  # noqa: E402

CAPITAL = 300_000
STRATEGY = {
    "name": "營收YoY排名_持40天_不停損_零股6檔_股價≤80",
    "sel": "revrank",          # revrank：站上 20 日線、依近 3 月營收 YoY 排序；mom_insti_rev：20 日動能＋法人買超＋營收 YoY>0
    "hold": 40, "k": 6, "atr_k": None, "lot": 1, "price_cap": 80.0,   # price_cap：當時真實股價（未還原）上限
    "min_turnover": 5e8, "cap_per_industry": 2, "gap": 0.03,
    "status": "PASS",          # strategy-gate（加嚴：贏隨機 ≥80%、起始日通過 ≥50%）2026-10-24
    "gate_file": "output/strategy_gate/v4_final.json",
}
FEE, TAX = 0.001425, 0.003
OUT = BASE / "output" / "swing"
LIVE = BASE / "data" / "swing_live"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _paths(out: Path) -> dict:
    return {"state": out / "state.json", "trades": out / "trades.csv", "nav": out / "nav.csv"}


def load_state(out: Path = OUT) -> dict:
    p = _paths(out)["state"]
    if p.exists():
        st = json.loads(p.read_text(encoding="utf-8"))
    else:
        st = {"capital": CAPITAL, "cash": CAPITAL, "positions": {}, "pending": [], "days_since_rebalance": None,
              "last_rebalance": None, "last_signal_date": None, "nav": CAPITAL, "created_at": _now()}
    st["strategy"] = STRATEGY["name"]
    st["strategy_status"] = STRATEGY["status"]
    st.setdefault("dividends", 0.0)
    return st


def save_state(st: dict, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _paths(out)["state"].write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def _append_csv(out: Path, name: str, row: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- 資料

def _candles(client: FugleClient, s: str, days: int, adjusted: bool) -> pd.DataFrame:
    end = date.today()
    for attempt in range(3):
        try:
            return client.historical_candles(s, end - timedelta(days=days), end, "D", adjusted=adjusted)
        except Exception as exc:
            if "429" in str(exc) and attempt < 2:
                time.sleep(61); continue
            print(f"{s} 日 K 失敗：{str(exc)[:80]}", flush=True)
            return pd.DataFrame()
    return pd.DataFrame()


def _openapi_values() -> dict[str, float]:
    """TWSE／TPEx OpenAPI 最新一天全市場成交值（元），只用來把「最近才變熱」的股票加入候選。"""
    import requests
    out = {}
    try:
        for r in requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=30,
                              headers={"User-Agent": "Mozilla/5.0"}).json():
            out[str(r.get("Code", "")).strip()] = ub._num_s(r.get("TradeValue"))
        for r in requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes", timeout=30,
                              headers={"User-Agent": "Mozilla/5.0"}).json():
            out[str(r.get("SecuritiesCompanyCode", "")).strip()] = ub._num_s(r.get("TransactionAmount"))
    except Exception as exc:
        print(f"OpenAPI 成交值失敗（只用快取名單）：{str(exc)[:80]}", flush=True)
    return out


def _superset() -> list[str]:
    """候選股：普通股中，快取的 20 日均成交值 > 2 億，或最新一天成交值 > 3 億。"""
    uni = ub.universe()
    keep = set()
    for s in uni.index:
        for p in (LIVE / "raw" / f"{s}.csv", ub.RAW_CACHE / f"{s}.csv", ub.CACHE / f"{s}.csv"):
            if p.exists():
                try:
                    d = pd.read_csv(p).tail(20)
                    if len(d) == 20 and float((d.close * d.volume).mean()) > 2e8:
                        keep.add(s)
                except Exception:
                    pass
                break
    for s, v in _openapi_values().items():
        if s in uni.index and v > 3e8:
            keep.add(s)
    return sorted(keep)


def refresh(client: FugleClient, syms: list[str], days: int = 130, adjusted_too: bool = False) -> dict:
    """抓未還原日 K（必要時也抓還原），存到 data/swing_live/。回傳 {symbol: (raw_df, adj_df|None)}。"""
    (LIVE / "raw").mkdir(parents=True, exist_ok=True)
    (LIVE / "adj").mkdir(parents=True, exist_ok=True)
    out = {}
    for n, s in enumerate(syms, 1):
        raw = _candles(client, s, days, False)
        adj = _candles(client, s, days, True) if adjusted_too else None
        if not raw.empty:
            raw.to_csv(LIVE / "raw" / f"{s}.csv", index=False, encoding="utf-8-sig")
        if adj is not None and not adj.empty:
            adj.to_csv(LIVE / "adj" / f"{s}.csv", index=False, encoding="utf-8-sig")
        out[s] = (raw, adj)
        if n % 50 == 0:
            print(f"  日 K {n}/{len(syms)}", flush=True)
    return out


def _atr_pct(d: pd.DataFrame) -> float:
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(14).mean().iloc[-1] / d.close.iloc[-1]
    return float(v) if np.isfinite(v) and v > 0 else 0.03


# ---------------------------------------------------------------- 選股

def pick_targets(client: FugleClient, nav: float) -> tuple[list[dict], pd.Timestamp | None, dict]:
    uni = ub.universe()
    print("更新月營收（MOPS）…", flush=True)
    ub.fetch_revenue(start_year=date.today().year - 2)
    sup = _superset()
    print(f"候選股 {len(sup)} 檔，更新日 K…", flush=True)
    data = refresh(client, sup, adjusted_too=STRATEGY["sel"] != "revrank")
    rows = []
    for s, (raw, adj) in data.items():
        if raw is None or len(raw) < 60:
            continue
        d = adj if adj is not None and len(adj) >= 60 else raw
        last = raw.iloc[-1]
        to20 = float((raw.close * raw.volume).tail(20).mean())
        rows.append({"symbol": s, "date": pd.Timestamp(last.date), "raw_close": float(last.close), "turnover20": to20,
                     "ma20": float(d.close.tail(20).mean()), "close": float(d.close.iloc[-1]),
                     "r20": float(d.close.iloc[-1] / d.close.iloc[-21] - 1), "atr_pct": _atr_pct(d),
                     "industry": uni.at[s, "industry"] if s in uni.index else "?"})
    t = pd.DataFrame(rows)
    if t.empty:
        return [], None, {}
    d0 = t.date.max()
    t = t[(t.date == d0) & (t.turnover20 > STRATEGY["min_turnover"]) & (t.close > t.ma20)]
    yoy = ub.load_revenue_yoy(pd.DatetimeIndex([d0]), list(t.symbol))
    t["rev_yoy3"] = t.symbol.map(yoy.iloc[-1]) if yoy is not None else np.nan
    if STRATEGY.get("price_cap"):
        t = t[t.raw_close <= STRATEGY["price_cap"]]
    if STRATEGY["lot"] > 1:
        t = t[t.raw_close * STRATEGY["lot"] * (1 + FEE) * 1.03 <= nav / STRATEGY["k"]]
    if STRATEGY["sel"] == "revrank":
        t = t[t.rev_yoy3.notna()].sort_values("rev_yoy3", ascending=False)
    else:
        start = (d0 - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        for m in ("twse", "tpex"):
            ub.fetch_insti(m, start, d0.strftime("%Y-%m-%d"))
        days = pd.bdate_range(start, d0)
        ins = ub.load_insti(days, list(t.symbol))
        t["insti20"] = t.symbol.map(ins.tail(20).sum(min_count=15)) if ins is not None else np.nan
        t = t[(t.insti20 > 0) & (t.rev_yoy3 > 0)].sort_values("r20", ascending=False)
    picks, cnt, info = [], {}, {"screened": int(len(t)), "as_of": str(d0.date())}
    for r in t.itertuples():
        if len(picks) >= STRATEGY["k"]:
            break
        if cnt.get(r.industry, 0) >= STRATEGY["cap_per_industry"]:
            continue
        try:
            el = eligibility(client.ticker(r.symbol))
        except Exception as exc:
            print(f"{r.symbol} ticker 失敗：{exc}", flush=True); continue
        bad = [x for x in el["reasons"] if "當沖" not in x]
        if bad:
            print(f"排除 {r.symbol} {el['name']}：{'、'.join(bad)}", flush=True); continue
        cnt[r.industry] = cnt.get(r.industry, 0) + 1
        picks.append({"symbol": r.symbol, "name": el["name"], "industry": r.industry, "close": r.raw_close,
                      "rev_yoy3": round(float(r.rev_yoy3) * 100, 1) if pd.notna(r.rev_yoy3) else None,
                      "r20": round(r.r20 * 100, 1), "atr_pct": round(r.atr_pct * 100, 2)})
    return picks, d0, info


# ---------------------------------------------------------------- 每日收盤後

def _update_positions(client: FugleClient, st: dict) -> pd.Timestamp | None:
    """持股：用未還原收盤算市值；用還原／未還原比例的變化估算除權息（現金股利記入現金）；檢查停損。"""
    latest = None
    for s, p in st["positions"].items():
        raw, adj = _candles(client, s, 40, False), _candles(client, s, 40, True)
        if raw.empty:
            continue
        m = raw.set_index("date")[["close"]].join(adj.set_index("date")[["close"]].rename(columns={"close": "adj"}), how="left")
        m["ratio"] = m.adj / m.close
        last_d = pd.Timestamp(m.index[-1])
        latest = last_d if latest is None or last_d > latest else latest
        since = pd.Timestamp(p.get("last_div_check") or p["entry_date"])
        r = m[m.index > since]
        prev_ratio = m[m.index <= since].ratio.iloc[-1] if (m.index <= since).any() else None
        for d, row in r.iterrows():
            if prev_ratio and np.isfinite(row.ratio) and row.ratio > prev_ratio * 1.002:
                prev_close = m.close.shift(1).loc[d]
                div = p["shares"] * prev_close * (1 - prev_ratio / row.ratio)
                st["cash"] += div
                st["dividends"] = st.get("dividends", 0.0) + div
                p["dividends"] = round(p.get("dividends", 0.0) + div, 1)
                print(f"{s} {d.date()} 除權息，估計股利 {div:,.0f} 元記入現金", flush=True)
            if np.isfinite(row.ratio):
                prev_ratio = row.ratio
        p["last_div_check"] = str(last_d.date())
        p["last"] = float(m.close.iloc[-1])
        # 停損：用「還原報酬」比較，避免除權息造成誤判
        ent = pd.Timestamp(p["entry_date"])
        er = m.ratio[m.index >= ent].iloc[0] if (m.index >= ent).any() else m.ratio.iloc[-1]
        tr_px = p["last"] * (m.ratio.iloc[-1] / er) if np.isfinite(er) and er > 0 else p["last"]
        p["stop_check"] = {"date": str(last_d.date()), "tr_price": round(float(tr_px), 2), "stop": p.get("stop")}
        if p.get("stop") and tr_px <= p["stop"] and not any(o["symbol"] == s and o["side"] == "sell" for o in st["pending"]):
            st["pending"].append({"side": "sell", "symbol": s, "shares": p["shares"], "name": p.get("name", ""),
                                  "reason": f"停損：收盤 {p['last']}（還原 {tr_px:.2f}）跌破 {p['stop']}"})
            print(f"{s} 觸發停損，明天開盤賣出", flush=True)
    return latest


def signal(dry_run: bool = False, out: Path = OUT) -> int:
    client = FugleClient(wait_on_limit=True)
    st = load_state(out)
    today = date.today()
    if st.get("last_signal_date") == str(today) and not dry_run:
        print("今天已經計算過，略過。"); return 0
    latest = _update_positions(client, st) if st["positions"] else None
    if latest is None:
        q = _candles(client, ub.INDEX, 10, False)
        latest = pd.Timestamp(q.date.iloc[-1]) if not q.empty else None
    if latest is None or (latest.date() != today and not dry_run):
        print(f"日 K 最新為 {latest.date() if latest is not None else '無'}，今天 {today} 不是交易日或資料尚未更新，結束。")
        return 0
    mv = sum(p["last"] * p["shares"] for p in st["positions"].values())
    st["nav"] = round(st["cash"] + mv)
    if st["days_since_rebalance"] is not None and not dry_run:
        st["days_since_rebalance"] += 1
    if not dry_run:
        _append_csv(out, "nav.csv", {"date": str(today), "nav": st["nav"], "cash": round(st["cash"]),
                                     "positions": len(st["positions"]), "return_pct": round((st["nav"] / st["capital"] - 1) * 100, 2)})
    due = st["days_since_rebalance"] is None or st["days_since_rebalance"] >= STRATEGY["hold"] or dry_run
    msg = (f"{today} 淨值 {st['nav']:,}（{(st['nav'] / st['capital'] - 1) * 100:+.2f}%），現金 {st['cash']:,.0f}，"
           f"持股 {len(st['positions'])} 檔｜策略 {STRATEGY['name']}（{STRATEGY['status']}）")
    if due:
        picks, d0, info = pick_targets(client, st["nav"])
        keep = {p["symbol"] for p in picks}
        orders = [o for o in st["pending"] if o["side"] == "sell"]
        orders += [{"side": "sell", "symbol": s, "shares": p["shares"], "name": p.get("name", ""), "reason": "換股：不在新名單"}
                   for s, p in st["positions"].items() if s not in keep and not any(o["symbol"] == s for o in orders)]
        per = st["nav"] / STRATEGY["k"]
        for p in picks:
            held = st["positions"].get(p["symbol"])
            have = held["shares"] * held["last"] if held else 0
            if per - have > per * 0.2:
                orders.append({"side": "buy", "symbol": p["symbol"], "name": p["name"], "value": round(per - have),
                               "atr_pct": p["atr_pct"],
                               "reason": f"營收近3月YoY {p['rev_yoy3']}%、20日 {p['r20']:+.1f}%、ATR {p['atr_pct']}%"})
        msg += f"\n換股日（資料日 {info.get('as_of')}，通過篩選 {info.get('screened')} 檔）：\n名單：" + "、".join(
            f"{p['symbol']}{p['name']}(收{p['close']}、營收YoY {p['rev_yoy3']}%)" for p in picks)
        msg += f"\n明天開盤委託 {len(orders)} 筆：" + "；".join(f"{o['side']} {o['symbol']}{o.get('name', '')}" for o in orders)
        if dry_run:
            print(msg + "\n（dry-run：不寫入狀態）")
            (out / "dry_run_picks.json").parent.mkdir(parents=True, exist_ok=True)
            (out / "dry_run_picks.json").write_text(json.dumps({"as_of": info.get("as_of"), "picks": picks, "orders": orders,
                                                                 "strategy": STRATEGY}, ensure_ascii=False, indent=1), encoding="utf-8")
            return 0
        st["pending"] = orders
        st["days_since_rebalance"] = 0
        st["last_rebalance"] = str(today)
        st["last_picks"] = picks
    else:
        msg += f"\n距下次換股還有 {STRATEGY['hold'] - st['days_since_rebalance']} 個交易日"
    st["last_signal_date"] = str(today)
    save_state(st, out)
    print(msg)
    return 0


def _cost(value: float, sell: bool) -> float:
    fee_min = 20 if STRATEGY["lot"] > 1 else 1
    return max(fee_min, value * FEE) + (value * TAX if sell else 0)


def execute(out: Path = OUT) -> int:
    st = load_state(out)
    if not st["pending"]:
        print("沒有待執行委託。"); return 0
    client = FugleClient(wait_on_limit=True)
    slip = 1 if STRATEGY["lot"] > 1 else 2
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
            fill = to_tick(op - slip * tick_size(op), up=False)
            value = fill * p["shares"]
            cost = _cost(value, True)
            st["cash"] += value - cost
            pnl = value - cost - p["cost_total"] + p.get("dividends", 0.0)
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
            prev = el["prev_close"]
            if prev and op > prev * (1 + STRATEGY["gap"]):
                print(f"{s} 開盤跳空 {op / prev - 1:+.1%} > {STRATEGY['gap']:.0%}，不追（資金留到下次換股）"); continue
            fill = to_tick(op + slip * tick_size(op), up=True)
            budget = min(o["value"], st["cash"])          # 只用現金，避免違約交割
            lot = STRATEGY["lot"]
            shares = int(budget / (fill * (1 + FEE)) / lot) * lot
            if shares < 1:
                print(f"{s} 現金不足，不買"); continue
            value = fill * shares
            cost = _cost(value, False)
            if value + cost > st["cash"]:
                print(f"{s} 現金不足，不買"); continue
            st["cash"] -= value + cost
            stop = round(fill * (1 - STRATEGY["atr_k"] * o.get("atr_pct", 3.0) / 100), 2) if STRATEGY.get("atr_k") else None
            p = st["positions"].setdefault(s, {"shares": 0, "cost_total": 0.0, "name": el["name"], "entry_date": str(date.today())})
            p["shares"] += shares
            p["cost_total"] += value + cost
            p["avg"] = round(p["cost_total"] / p["shares"], 2)
            p["last"] = fill
            p["stop"] = stop
            row = {"time": _now(), "side": "買", "symbol": s, "name": el["name"], "shares": shares, "price": fill,
                   "value": round(value), "cost": round(cost, 1), "pnl": "", "reason": o["reason"] + (f"；停損 {stop}" if stop else "")}
        _append_csv(out, "trades.csv", row)
        done.append(row)
        print(f"{row['side']} {s}{row['name']} {row['shares']} 股 @ {row['price']}（{row['reason']}）")
    st["pending"] = left
    save_state(st, out)
    print(f"完成 {len(done)} 筆，現金 {st['cash']:,.0f}")
    return 0


def status(out: Path = OUT) -> int:
    st = load_state(out)
    print(f"策略 {STRATEGY['name']}（{STRATEGY['status']}）")
    print(f"本金 {st['capital']:,}，淨值 {st['nav']:,}（{(st['nav'] / st['capital'] - 1) * 100:+.2f}%），現金 {st['cash']:,.0f}，"
          f"累計股利 {st.get('dividends', 0):,.0f}")
    print(f"上次換股 {st.get('last_rebalance')}，之後已過 {st.get('days_since_rebalance')} 個交易日")
    for s, p in st["positions"].items():
        print(f"  {s}{p.get('name', '')} {p['shares']} 股，成本 {p['avg']}，最新 {p['last']}，停損 {p.get('stop')}，"
              f"損益 {p['last'] * p['shares'] - p['cost_total'] + p.get('dividends', 0):+,.0f}")
    for o in st["pending"]:
        print(f"  待執行：{o['side']} {o['symbol']}{o.get('name', '')}（{o['reason']}）")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "status"
    out = Path(args[args.index("--out") + 1]) if "--out" in args else OUT
    if cmd == "signal":
        sys.exit(signal(dry_run="--dry-run" in args, out=out))
    sys.exit({"execute": execute, "status": status}[cmd](out=out))
