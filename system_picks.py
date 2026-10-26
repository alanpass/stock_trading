# -*- coding: utf-8 -*-
"""系統推薦股票（波段、長期）與看多原因，給網站「未來分析 → 最近系統推薦股票」。

選股範圍：趨勢型態追蹤的概念股（約 175 檔）＋自選股。每天盤後用固定規則篩一次：

波段（約 2～8 週）── 每符合一項記一個看多原因，至少 3 項、且型態或均線至少一項成立：
  題材位於輪動「領先／改善」象限、K 線型態偏多（多頭延續、底部反轉已突破、整理待突破）、
  收盤站上月線且月線高於季線、近 20 日上漲 5～40%（有動能但未過熱）、月營收年增 ≥ 20%、題材波段引擎入選。
  排除：型態高檔轉弱／下降趨勢、跌破季線、處置股。
長期（數月以上）── 必須營收年增 ≥ 20% 且站上年線、季線高於年線，再加上本益比 ≤ 35 或連續 5 年配息其一。

入選股票再補上「個股符合特徵」的偏多籌碼／價量標籤與 Claude 研究筆記的利多重點。
每天的名單存進 system_picks_history.json，之後用最新收盤追蹤推薦後漲跌。
規則都還沒通過 strategy-gate，網站上標示為觀察名單，不是買賣建議。

  python system_picks.py [--no-publish] [--limit 8]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
REPORTS = OUT / "research_reports"
PUBLIC_PATH = REPORTS / "system_picks_public.json"
HISTORY_PATH = REPORTS / "system_picks_history.json"
REMOTE_PATH = "output/research_reports/system_picks_public.json"
TAIPEI = ZoneInfo("Asia/Taipei")

HORIZONS = {"swing": "波段（約 2～8 週）", "long": "長期（數月以上）"}
SWING_MIN_REASONS = 3
R20_MIN, R20_MAX = 5.0, 40.0
REV_GROWTH = 20.0
PE_MAX = 35.0
DIVIDEND_YEARS = 5
HISTORY_DAYS = 60
BULL_PATTERNS = {"多頭延續／持續創高", "整理待突破"}
BEAR_PATTERNS = {"高檔轉弱／反轉警示", "下降趨勢／弱勢整理"}
RULES = [
    ("波段", f"至少 {SWING_MIN_REASONS} 項看多原因，且 K 線型態或均線至少一項成立：題材在輪動領先／改善象限、型態偏多、"
             f"站上月線且月線高於季線、近 20 日上漲 {R20_MIN:.0f}～{R20_MAX:.0f}%、月營收年增 ≥ {REV_GROWTH:.0f}%、題材波段引擎入選；"
             "排除高檔轉弱、下降趨勢、跌破季線與處置股"),
    ("長期", f"月營收年增 ≥ {REV_GROWTH:.0f}%，收盤站上年線且季線高於年線，再加上本益比 ≤ {PE_MAX:.0f} 倍或連續 {DIVIDEND_YEARS} 年配息"),
]


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _universe() -> dict[str, dict]:
    trend = _read(REPORTS / "trend_patterns_public.json", {}) or {}
    uni = {str(s["symbol"]): dict(s) for s in trend.get("stocks") or [] if s.get("symbol")}
    for code in _read(BASE / "user_watchlist.json", []) or []:
        uni.setdefault(str(code), {"symbol": str(code), "themes": []})
    return uni


def _theme_state() -> dict[str, dict]:
    rot = _read(REPORTS / "industry_rotation_public.json", {}) or {}
    return {g["name"]: g for g in rot.get("groups") or [] if g.get("name")}


def _swing_engine_symbols() -> set[str]:
    out = set()
    for path in OUT.glob("swing/**/state.json"):
        out |= {str(p.get("symbol")) for p in (_read(path, {}) or {}).get("last_picks") or []}
    return out


def _note_points() -> dict[str, dict]:
    notes = (_read(REPORTS / "stock_notes_public.json", {}) or {}).get("notes") or []
    return {str(n.get("symbol")): n for n in notes}


def _history(client, symbol: str) -> pd.DataFrame:
    end = date.today()
    h = client.historical_candles(symbol, end - timedelta(days=364), end, "D")   # Fugle 一次最多一年（約 245 根）
    return h.sort_values("date").reset_index(drop=True) if not h.empty else h


def _tech(h: pd.DataFrame) -> dict:
    c = h["close"].astype(float)
    ma = lambda n: c.rolling(n).mean().iloc[-1] if len(c) >= n else math.nan
    ret = lambda n: (c.iloc[-1] / c.iloc[-n - 1] - 1) * 100 if len(c) > n else math.nan
    return {"close": float(c.iloc[-1]), "date": pd.Timestamp(h["date"].iloc[-1]).strftime("%Y-%m-%d"),
            "ma20": ma(20), "ma60": ma(60), "ma240": ma(240), "r20": ret(20), "r60": ret(60), "r240": ret(240)}


def _swing_reasons(s: dict, t: dict, fund: dict, themes: dict, engine: set) -> tuple[list[dict], bool]:
    reasons, core = [], False
    for th in s.get("themes") or []:
        g = themes.get(th) or {}
        if g.get("quadrant") in ("領先", "改善"):
            reasons.append({"cat": "題材", "text": f"「{th}」位於產業輪動{g['quadrant']}象限，族群近 20 日 {_f(g.get('return_20d')):+.1f}%"})
            break
    cls, pat = s.get("classification") or "", s.get("pattern_name") or ""
    if cls in BULL_PATTERNS or (cls == "底部反轉候選" and "突破" in pat):
        reasons.append({"cat": "型態", "text": f"K 線型態「{pat}」（{cls}）"})
        core = True
    if t["close"] > t["ma20"] > t["ma60"]:
        reasons.append({"cat": "技術", "text": f"收盤 {t['close']:,.2f} 站上月線 {t['ma20']:,.2f}，月線高於季線 {t['ma60']:,.2f}"})
        core = True
    if R20_MIN <= t["r20"] <= R20_MAX:
        reasons.append({"cat": "動能", "text": f"近 20 日上漲 {t['r20']:.1f}%，動能延續但未過熱"})
    yoy = _f(fund.get("rev_yoy"))
    if yoy >= REV_GROWTH:
        reasons.append({"cat": "基本面", "text": f"{fund.get('rev_month', '')} 月營收年增 {yoy:.1f}%"})
    if s["symbol"] in engine:
        reasons.append({"cat": "系統", "text": "題材波段引擎（前 3 強題材 × 20 日動能）入選"})
    return reasons, core


def _long_reasons(t: dict, fund: dict, div_years) -> list[dict] | None:
    yoy, pe, yld = _f(fund.get("rev_yoy")), _f(fund.get("pe")), _f(fund.get("yld"))
    if not (yoy >= REV_GROWTH and t["close"] > t["ma240"] and t["ma60"] > t["ma240"]):
        return None
    reasons = [{"cat": "基本面", "text": f"{fund.get('rev_month', '')} 月營收年增 {yoy:.1f}%"},
               {"cat": "技術", "text": f"站上年線 {t['ma240']:,.2f}，季線高於年線，長期趨勢向上"}]
    ok = False
    if 0 < pe <= PE_MAX:
        reasons.append({"cat": "評價", "text": f"本益比 {pe:.1f} 倍（≤ {PE_MAX:.0f}）"})
        ok = True
    years = div_years()
    this_year = date.today().year
    if years and set(range(this_year - DIVIDEND_YEARS, this_year)) <= set(years):
        reasons.append({"cat": "股利", "text": f"最近 {DIVIDEND_YEARS} 年每年配息"})
        ok = True
    if not ok:
        return None
    if yld >= 3:
        reasons.append({"cat": "股利", "text": f"殖利率 {yld:.2f}%"})
    if t["r240"] == t["r240"] and t["r240"] > 0:
        reasons.append({"cat": "動能", "text": f"近一年上漲 {t['r240']:.1f}%"})
    return reasons


def _extra_reasons(symbol: str, name: str, hist: pd.DataFrame, notes: dict) -> list[dict]:
    """個股符合特徵的偏多籌碼／價量標籤＋研究筆記利多重點。"""
    import chip_signals
    import stock_features

    out = []
    try:
        chips = chip_signals.compute(symbol)
    except Exception:
        chips = None
    try:
        feat = stock_features.compute(symbol, hist, name=name, chips=chips)
        for cat, label in (("籌碼面", "籌碼"), ("價量面", "價量")):
            # 連買連賣的依據都是同一句「連續 3 日以上…」，只留標籤本身
            out += [{"cat": label, "text": t["text"] if t["why"].startswith("連續 3 日") else f"{t['text']}：{t['why']}"}
                    for t in feat["groups"][cat] if t["tone"] == "bull"]
    except Exception:
        pass
    note = notes.get(symbol)
    for p in (note or {}).get("positives") or []:
        out.append({"cat": "研究筆記", "text": f"{p}（{(note.get('headline') or '')[:30]}）"})
    return out


def screen(limit: int = 8) -> dict:
    from stock_api import FugleClient
    import stock_features

    client = FugleClient(wait_on_limit=True)
    uni, themes, engine, notes = _universe(), _theme_state(), _swing_engine_symbols(), _note_points()
    fund_all = stock_features.fundamentals()
    swing, long_, hists, asof = [], [], {}, ""
    for sym, s in uni.items():
        try:
            h = _history(client, sym)
        except Exception as exc:
            print(f"  {sym} 日K取不到：{type(exc).__name__}", file=sys.stderr)
            continue
        if len(h) < 60:
            continue
        hists[sym] = h
        t = _tech(h)
        asof = max(asof, t["date"])
        fund = fund_all.get(sym, {})
        name = s.get("name") or ""
        base = {"symbol": sym, "name": name, "themes": s.get("themes") or [], "close": round(t["close"], 2),
                "r20": round(t["r20"], 2) if t["r20"] == t["r20"] else None}
        if fund.get("disposal"):
            continue
        if (s.get("classification") or "") not in BEAR_PATTERNS and t["close"] >= t["ma60"]:
            reasons, core = _swing_reasons(s, t, fund, themes, engine)
            if core and len(reasons) >= SWING_MIN_REASONS:
                swing.append({**base, "horizon": "swing", "reasons": reasons, "score": len(reasons),
                              "pattern": s.get("pattern_name") or "", "classification": s.get("classification") or ""})
        if sym.isdigit() and len(sym) == 4:
            lr = _long_reasons(t, fund, lambda: stock_features.dividend_years(sym))
            if lr:
                long_.append({**base, "horizon": "long", "reasons": lr, "score": len(lr), "rev_yoy": _f(fund.get("rev_yoy"))})

    swing.sort(key=lambda p: (p["score"], p["r20"] or 0), reverse=True)
    long_.sort(key=lambda p: (p["score"], p["rev_yoy"]), reverse=True)
    picks = swing[:limit] + long_[:limit]
    for p in picks:
        p["reasons"] += _extra_reasons(p["symbol"], p["name"], hists.get(p["symbol"]), notes)
        p.pop("rev_yoy", None)
    return {"date": asof, "picks": picks, "candidates": {"swing": len(swing), "long": len(long_)}}


def _with_returns(history: list[dict], closes: dict[str, tuple[str, float]]) -> list[dict]:
    rows = []
    for day in history:
        for p in day["picks"]:
            last_d, last = closes.get(p["symbol"], ("", None))
            ret = round((last / p["close"] - 1) * 100, 2) if last and p.get("close") and last_d > day["date"] else None
            rows.append({"date": day["date"], "symbol": p["symbol"], "name": p["name"], "horizon": p["horizon"],
                         "ref_price": p["close"], "last_close": last, "last_date": last_d, "ret_pct": ret})
    return rows


def update(limit: int = 8, publish: bool = True) -> dict:
    today = screen(limit)
    history = [d for d in _read(HISTORY_PATH, []) or [] if d.get("date") != today["date"]]
    history = sorted(history + [{"date": today["date"], "picks": today["picks"]}], key=lambda d: d["date"])[-HISTORY_DAYS:]
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")

    # 追蹤用的最新收盤：今天篩選時抓過的日 K 就是最新收盤
    closes = {p["symbol"]: (today["date"], p["close"]) for p in today["picks"]}
    need = {p["symbol"] for d in history for p in d["picks"]} - set(closes)
    if need:
        from stock_api import FugleClient
        client = FugleClient(wait_on_limit=True)
        for sym in need:
            try:
                h = _history(client, sym)
                closes[sym] = (pd.Timestamp(h["date"].iloc[-1]).strftime("%Y-%m-%d"), float(h["close"].iloc[-1]))
            except Exception:
                pass
    tracked = _with_returns(history[-10:], closes)
    summary = {}
    for key, label in HORIZONS.items():
        rs = [r["ret_pct"] for r in tracked if r["horizon"] == key and r["ret_pct"] is not None]
        summary[key] = {"label": label, "count": len(rs), "avg_ret": round(sum(rs) / len(rs), 2) if rs else None,
                        "win_rate": round(sum(1 for x in rs if x > 0) / len(rs) * 100, 1) if rs else None}
    data = {
        "schema_version": 2,
        "report_type": "system_picks_public",
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "data_asof": today["date"],
        "latest": today,
        "tracked": tracked,
        "summary": summary,
        "rules": RULES,
        "disclaimer": "規則式篩選的觀察名單，目前沒有任何策略通過上線驗證（strategy-gate），不構成投資建議。",
    }
    text = json.dumps(data, ensure_ascii=False, indent=1, default=str)
    PUBLIC_PATH.write_text(text, encoding="utf-8")
    result = {"ok": True, "date": today["date"], "swing": sum(p["horizon"] == "swing" for p in today["picks"]),
              "long": sum(p["horizon"] == "long" for p in today["picks"]), "candidates": today["candidates"]}
    if publish:
        from github_publish import publish_json
        result["publish"] = publish_json(REMOTE_PATH, text, f"data: system picks {data['generated_at']}")
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="波段／長期系統推薦股票與看多原因，發布到網站")
    ap.add_argument("--limit", type=int, default=8, help="波段、長期各列幾檔")
    ap.add_argument("--no-publish", action="store_true")
    ns = ap.parse_args()
    print(json.dumps(update(ns.limit, publish=not ns.no_publish), ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
