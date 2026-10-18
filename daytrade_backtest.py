# -*- coding: utf-8 -*-
"""當沖策略回測研究（盤中零股、只做多、含成本與滑價）。

  python daytrade_backtest.py            # 下載／更新 5 分 K 快取並比較各策略
  python daytrade_backtest.py --no-fetch # 只用快取

方法：
- 候選股：自選股＋產業概念股；每天只用「前一天以前」的資料挑 20 日成交值前 40 名、14 日 ATR ≥ 3% 的股票（不偷看未來）。
- 成本：手續費 0.1425%×2、證交稅 0.3%（零股不適用當沖減半）、進出各 2 個 tick 滑價。
- 同一根 5 分 K 同時碰到停損與停利時，一律當作先停損（保守）。
- 交易日依時間切成兩段：前 2/3 選參數（train）、後 1/3 驗證（test）。只有 test 也賺錢的設定才採用。
僅供研究，不構成投資建議。
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from daytrade_engine import FEE, TAX, ODD_LOT_SLIPPAGE_TICKS, INDEX, tick_size, _trend_patterns, _watchlist
from stock_api import FugleClient, clean_symbol

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data" / "daytrade_bt"
DAYS = 140


def _load_5m(client: FugleClient | None, s: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{s}.csv"
    if client is not None:
        end = date.today()
        d = client.historical_candles(s, end - timedelta(days=DAYS), end, "5")
        if not d.empty:
            d.to_csv(path, index=False, encoding="utf-8-sig")
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    d["day"] = d["date"].dt.date
    d["time"] = d["date"].dt.strftime("%H:%M")
    return d[(d["time"] >= "09:00") & (d["time"] <= "13:25")].reset_index(drop=True)


def _daily_from_5m(k: pd.DataFrame) -> pd.DataFrame:
    g = k.groupby("day")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out["prev_close"] = out["close"].shift()
    tr = np.maximum(out["high"] - out["low"], np.maximum((out["high"] - out["prev_close"]).abs(), (out["low"] - out["prev_close"]).abs()))
    out["atr_pct"] = (tr.rolling(14).mean() / out["close"]).shift()            # 只用前一天以前
    out["turnover20"] = (out["close"] * out["volume"] * 1000).rolling(20).mean().shift()   # 5 分 K 量的單位是張
    first = k[k["time"] == "09:00"].set_index("day")["volume"]
    out["first_vol"] = first
    out["first_vol_avg"] = first.rolling(14).mean().shift()
    return out


@dataclass
class Cfg:
    strategy: str          # orb15 / sip5 / vwap_pull
    stop: float            # 停損 %（0.025 = 2.5%）
    target: float | None   # 停利 %；None＝不設停利
    trail: float | None    # 移動停損：賺到 +stop 後，從最高點回落 trail 出場
    exit_time: str = "13:20"
    min_rv: float = 2.0    # sip5：開盤第一根量能倍數門檻
    no_chase: float = 0.01 # 進場價最多高於 VWAP 的比例
    max_up: float = 0.04   # 進場時比昨收最多漲幅
    index_filter: bool = True


def _cost_pct(entry: float, exit_: float) -> float:
    return (entry * FEE + exit_ * FEE + exit_ * TAX) / entry


def simulate_day(cfg: Cfg, day, bars: dict[str, pd.DataFrame], info: dict, idx: pd.DataFrame, themes: dict) -> list[dict]:
    """單日模擬：依時間順序處理訊號，最多 3 檔同時持有、同族群 1 檔。回傳每筆交易（淨報酬 %）。"""
    cands = []
    for s, k in bars.items():
        k = k.reset_index(drop=True)
        pc = info[s]["prev_close"]
        if not pc or len(k) < 10:
            continue
        vw = ((k["high"] + k["low"] + k["close"]) / 3 * k["volume"]).cumsum() / k["volume"].cumsum().replace(0, np.nan)
        sig = None
        if cfg.strategy == "orb15":
            orb = k[k["time"] < "09:15"]
            if len(orb) < 3:
                continue
            orh = orb["high"].max()
            for i in range(3, len(k)):
                t = k.at[i, "time"]
                if t >= "12:30":
                    break
                c = k.at[i, "close"]
                if c > orh and c > vw[i]:
                    sig = (i + 1, c, f"ORB15 突破 {orh}")
                    break
        elif cfg.strategy == "sip5":
            b0 = k.iloc[0]
            rv = info[s]["first_vol"] / info[s]["first_vol_avg"] if info[s]["first_vol_avg"] else 0
            if rv < cfg.min_rv or b0["close"] <= b0["open"]:
                continue
            for i in range(1, len(k)):
                if k.at[i, "time"] >= "12:30":
                    break
                if k.at[i, "high"] > b0["high"]:
                    sig = (i, max(b0["high"] + tick_size(b0["high"]), k.at[i, "open"]), f"SIP 第一根量 {rv:.1f} 倍")
                    break
        elif cfg.strategy == "vwap_pull":
            for i in range(6, len(k)):          # 09:30 之後
                t = k.at[i, "time"]
                if t >= "12:30":
                    break
                strong = k.at[i - 1, "close"] / pc - 1 >= 0.02 and k.at[i - 1, "close"] > vw[i - 1]
                touch = k.at[i, "low"] <= vw[i - 1] * 1.003 and k.at[i, "close"] > vw[i - 1]
                if strong and touch:
                    sig = (i + 1, k.at[i, "close"], "強勢股回測 VWAP")
                    break
        if sig and sig[0] < len(k):
            cands.append((k.at[sig[0] if cfg.strategy != "sip5" else sig[0], "time"], s, sig, k, vw))
    cands.sort(key=lambda x: x[0])

    trades, open_until, held_themes = [], [], []
    for t0, s, (i0, px, why), k, vw in cands:
        # 同時持倉與族群限制
        live = [(end, th) for end, th in zip(open_until, held_themes) if end > t0]
        if len(live) >= 3 or any(set(th) & set(themes.get(s, [])) for _, th in live if themes.get(s)):
            continue
        pc = info[s]["prev_close"]
        if cfg.index_filter and not idx.empty:
            ib = idx[idx["time"] < t0]
            if not ib.empty:
                iv = ((ib["high"] + ib["low"] + ib["close"]) / 3 * ib["volume"]).sum() / ib["volume"].sum()
                if ib["close"].iloc[-1] < iv:
                    continue
        ref_vw = vw[max(i0 - 1, 0)]
        if px > ref_vw * (1 + cfg.no_chase) or px > pc * (1 + cfg.max_up):
            continue
        entry = px + ODD_LOT_SLIPPAGE_TICKS * tick_size(px)
        stop = entry * (1 - cfg.stop)
        target = entry * (1 + cfg.target) if cfg.target else None
        peak, exit_px, reason, t_exit = entry, None, "", None
        for j in range(i0, len(k)):
            hi, lo, t = k.at[j, "high"], k.at[j, "low"], k.at[j, "time"]
            if lo <= stop:
                exit_px, reason, t_exit = stop - ODD_LOT_SLIPPAGE_TICKS * tick_size(stop), "停損", t
                break
            if target and hi >= target:
                exit_px, reason, t_exit = target, "停利", t
                break
            peak = max(peak, hi)
            if cfg.trail and peak >= entry * (1 + cfg.stop):
                stop = max(stop, entry * (1 + 2 * FEE + TAX), peak * (1 - cfg.trail))
            if t >= cfg.exit_time:
                exit_px, reason, t_exit = k.at[j, "close"] - ODD_LOT_SLIPPAGE_TICKS * tick_size(k.at[j, "close"]), "時間出場", t
                break
        if exit_px is None:
            exit_px, reason, t_exit = k["close"].iloc[-1], "收盤", k["time"].iloc[-1]
        net = (exit_px / entry - 1) - _cost_pct(entry, exit_px)
        trades.append({"day": str(day), "symbol": s, "time": t0, "why": why, "entry": round(entry, 2),
                       "exit": round(exit_px, 2), "reason": reason, "exit_time": t_exit, "net_pct": net * 100})
        open_until.append(t_exit)
        held_themes.append(themes.get(s, []))
    return trades


def evaluate(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    r = pd.Series([t["net_pct"] for t in trades])
    daily = pd.DataFrame(trades).groupby("day")["net_pct"].sum()   # 每筆相同風險下的每日合計（%）
    eq = daily.cumsum()
    win, loss = r[r > 0].sum(), -r[r <= 0].sum()
    return {"n": int(len(r)), "win_rate": round(float((r > 0).mean() * 100), 1), "avg": round(float(r.mean()), 3),
            "total": round(float(r.sum()), 2), "pf": round(float(win / loss), 2) if loss else None,
            "max_dd": round(float((eq - eq.cummax()).min()), 2), "days": int(len(daily)),
            "pos_days": round(float((daily > 0).mean() * 100), 1)}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ns = ap.parse_args()
    client = None if ns.no_fetch else FugleClient(wait_on_limit=True)
    pats = _trend_patterns()
    universe = list(dict.fromkeys([clean_symbol(x) for x in _watchlist()] + list(pats.keys())))
    themes = {s: list((pats.get(s) or {}).get("themes") or []) for s in universe}

    # 先用日 K 快取挑出曾經進入「成交值前 40 名」的股票，只下載這些股票的 5 分 K
    k5, dly = {}, {}
    for n, s in enumerate(universe, 1):
        try:
            k = _load_5m(client, s)
        except Exception as exc:
            print(f"{s} 下載失敗：{exc}")
            continue
        if k.empty:
            continue
        k5[s], dly[s] = k, _daily_from_5m(k)
        if n % 25 == 0:
            print(f"已載入 {n}/{len(universe)}", flush=True)
    idx = _load_5m(client, INDEX)
    days = sorted(set.intersection(*[set(d.index) for d in list(dly.values())[:5]]))
    days = [d for d in days if d >= days[0] + timedelta(days=35)]   # 前 35 天留給 ATR／成交值暖身
    split = days[int(len(days) * 2 / 3)]
    print(f"回測期間 {days[0]} ～ {days[-1]}，共 {len(days)} 個交易日；train < {split} ≤ test", flush=True)

    grid = []
    for strat in ("orb15", "sip5", "vwap_pull"):
        for stop, target, trail in [(0.025, 0.05, None), (0.025, 0.05, 0.02), (0.015, 0.03, None),
                                    (0.015, None, 0.015), (0.02, None, 0.02), (0.01, 0.02, None)]:
            for idxf in (True, False):
                grid.append(Cfg(strat, stop, target, trail, index_filter=idxf))

    results = []
    for cfg in grid:
        tr_train, tr_test = [], []
        for d in days:
            cands = []
            for s, dd in dly.items():
                if d not in dd.index:
                    continue
                row = dd.loc[d]
                if not (row["atr_pct"] >= 0.03 and row["turnover20"] > 5e8):
                    continue
                cands.append((row["turnover20"], s))
            top = [s for _, s in sorted(cands, reverse=True)[:40]]
            bars = {s: k5[s][k5[s]["day"] == d] for s in top}
            info = {s: {"prev_close": float(dly[s].loc[d, "prev_close"]),
                        "first_vol": float(dly[s].loc[d, "first_vol"] or 0),
                        "first_vol_avg": float(dly[s].loc[d, "first_vol_avg"] or 0)} for s in top}
            ib = idx[idx["day"] == d] if not idx.empty else idx
            t = simulate_day(cfg, d, bars, info, ib, themes)
            (tr_train if d < split else tr_test).extend(t)
        results.append((cfg, evaluate(tr_train), evaluate(tr_test), tr_test))
        tr = results[-1]
        print(f"{cfg.strategy:9s} 停損{cfg.stop:.1%} 停利{(cfg.target or 0):.0%} 移動{(cfg.trail or 0):.1%} 大盤濾網{'有' if cfg.index_filter else '無'}"
              f" | train {tr[1]} | test {tr[2]}", flush=True)

    out = BASE / "output" / "daytrade" / "backtest_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{"cfg": c.__dict__, "train": a, "test": b} for c, a, b, _ in results],
                              ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    best = sorted([r for r in results if r[1].get("n", 0) >= 20 and r[2].get("n", 0) >= 10],
                  key=lambda r: -(r[1].get("avg") or -9))[:5]
    print("\n=== train 平均每筆最好的 5 組，與其 test 表現 ===")
    for c, a, b, _ in best:
        print(c, "\n   train", a, "\n   test ", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
