# -*- coding: utf-8 -*-
"""最近系統推薦（篩選）過的股票，整理成網站「未來分析」頁用的公開 JSON。

來源（都是本機排程已經產生的檔案，這裡只彙整、不重新選股）：
  - 題材波段：output/swing/**/state.json 的 last_picks（題材前 3 強 → 個股 20 日動能）
  - 當沖候選：output/daytrade/<日期>/plan.json 中 tradable 且可做多的股票
  - 盤後利多題材：output/research_reports/research_<日期>.json 的 bullish_themes 前 2 個題材成分股

每筆記下推薦日的參考價（推薦日收盤，取不到用來源檔的價格），再用最新日 K 收盤算出推薦後漲跌，
方便事後檢討「系統挑的股票後來怎麼走」。這些名單都還沒通過 strategy-gate，只是觀察名單，不是買賣建議。

  python system_picks.py [--days 10] [--no-publish]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
REPORTS = OUT / "research_reports"
PUBLIC_PATH = REPORTS / "system_picks_public.json"
REMOTE_PATH = "output/research_reports/system_picks_public.json"
TAIPEI = ZoneInfo("Asia/Taipei")

SOURCES = {
    "swing": "題材波段選股",
    "daytrade": "當沖候選（可做多）",
    "theme": "盤後利多題材",
}


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _swing_picks() -> list[dict]:
    rows = []
    for path in sorted(OUT.glob("swing/**/state.json")):
        st = _read(path) or {}
        day = str(st.get("last_signal_date") or st.get("last_rebalance") or "")[:10]
        for p in st.get("last_picks") or []:
            themes = p.get("themes") or []
            rows.append({"date": day, "symbol": str(p.get("symbol")), "name": p.get("name") or "", "source": "swing",
                         "reason": f"題材 {'、'.join(themes)}，20 日動能 {p.get('mom20', 0):+.1f}%",
                         "themes": themes, "src_price": p.get("close")})
    return rows


def _daytrade_picks() -> list[dict]:
    rows = []
    for path in sorted(OUT.glob("daytrade/20*/plan.json")):
        plan = _read(path) or {}
        day = str(plan.get("date") or path.parent.name)[:10]
        for s in plan.get("stocks") or []:
            if not (s.get("tradable") and s.get("allow_long")):
                continue
            bits = [b for b in (s.get("classification"), s.get("pattern"), f"日線{s['daily_trend']}" if s.get("daily_trend") else "") if b]
            rows.append({"date": day, "symbol": str(s.get("symbol")), "name": s.get("name") or "", "source": "daytrade",
                         "reason": "、".join(bits), "themes": s.get("themes") or [], "src_price": s.get("prev_close")})
    return rows


def _theme_picks() -> list[dict]:
    rows = []
    for path in sorted(REPORTS.glob("research_20*.json")):
        rep = _read(path) or {}
        day = str(rep.get("report_date") or path.stem[-10:])[:10]
        themes = sorted(rep.get("bullish_themes") or [], key=lambda t: -float(t.get("strength") or 0))[:2]
        for t in themes:
            for c in t.get("affected_companies") or []:
                rows.append({"date": day, "symbol": str(c.get("symbol")), "name": c.get("name") or "", "source": "theme",
                             "reason": f"{t.get('theme')}（強度 {float(t.get('strength') or 0):.0f}，{t.get('status') or ''}）",
                             "themes": [t.get("theme")], "src_price": None})
    return rows


def _closes(symbols: set[str], start: date) -> dict[str, list[tuple[str, float]]]:
    from stock_api import FugleClient

    client = FugleClient(wait_on_limit=True)
    out = {}
    for sym in sorted(symbols):
        try:
            h = client.historical_candles(sym, start, date.today(), "D")
            out[sym] = [(d.strftime("%Y-%m-%d"), float(c)) for d, c in zip(h["date"], h["close"])]
        except Exception as exc:  # 單一檔失敗不影響整份名單
            print(f"  {sym} 日K取不到：{type(exc).__name__}", file=sys.stderr)
            out[sym] = []
    return out


def build(days: int = 10) -> dict:
    rows = _swing_picks() + _daytrade_picks() + _theme_picks()
    rows = [r for r in rows if r["date"] and r["symbol"]]
    pick_days = sorted({r["date"] for r in rows}, reverse=True)[:days]
    rows = [r for r in rows if r["date"] in pick_days]
    # 同一天、同來源、同一檔只留一筆
    seen, uniq = set(), []
    for r in rows:
        k = (r["date"], r["source"], r["symbol"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    start = date.fromisoformat(min(pick_days)) - timedelta(days=7) if pick_days else date.today()
    closes = _closes({r["symbol"] for r in uniq}, start)

    asof = ""
    for r in uniq:
        series = closes.get(r["symbol"]) or []
        on_day = [c for d, c in series if d <= r["date"]]
        ref = on_day[-1] if on_day else r.pop("src_price", None)
        r.pop("src_price", None)
        last_d, last = (series[-1] if series else ("", None))
        r["ref_price"] = round(ref, 2) if ref else None
        r["last_close"] = round(last, 2) if last else None
        r["last_date"] = last_d
        # 推薦日當天或之後還沒有新的收盤（例如假日前一晚產生的名單）就不算漲跌
        r["ret_pct"] = round((last / ref - 1) * 100, 2) if ref and last and last_d > r["date"] else None
        r["source_label"] = SOURCES[r["source"]]
        asof = max(asof, last_d)

    summary = {}
    for key, label in SOURCES.items():
        rs = [r["ret_pct"] for r in uniq if r["source"] == key and r["ret_pct"] is not None]
        summary[key] = {"label": label, "count": len(rs),
                        "avg_ret": round(sum(rs) / len(rs), 2) if rs else None,
                        "win_rate": round(sum(1 for x in rs if x > 0) / len(rs) * 100, 1) if rs else None}
    uniq.sort(key=lambda r: (r["date"], r["source"], -(r["ret_pct"] or -999)), reverse=True)
    return {
        "schema_version": 1,
        "report_type": "system_picks_public",
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "data_asof": asof,
        "pick_days": pick_days,
        "summary": summary,
        "picks": uniq,
        "disclaimer": "系統篩選的觀察名單，目前沒有任何策略通過上線驗證（strategy-gate），不構成投資建議。",
    }


def update(days: int = 10, publish: bool = True) -> dict:
    data = build(days)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    PUBLIC_PATH.write_text(text, encoding="utf-8")
    result = {"ok": True, "path": str(PUBLIC_PATH), "count": len(data["picks"])}
    if publish:
        from github_publish import publish_json
        result["publish"] = publish_json(REMOTE_PATH, text, f"data: system picks {data['generated_at']}")
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="彙整最近系統推薦股票並發布到網站")
    ap.add_argument("--days", type=int, default=10, help="保留最近幾個有推薦的日子")
    ap.add_argument("--no-publish", action="store_true")
    ns = ap.parse_args()
    print(json.dumps(update(ns.days, publish=not ns.no_publish), ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
