# -*- coding: utf-8 -*-
"""進場分析與模型稽核（MCP server、Skill 與命令列共用）。

  python entry_analysis.py analyze 2330
  python entry_analysis.py audit [2330 2408 ...]   # 不給代號＝全部自選股；結果寫入 output/research_reports/entry_model_audit.json

analyze：取最近交易日 5 分 K＋日 K，用當日快取的 v3 模型（沒有就訓練）產生進場判斷。
audit：每檔重新訓練並做前推驗證，回報回檔機率是否比歷史頻率準（Brier skill）、最低價是否比基準準，
       並和上一次稽核比較，列出失去／恢復預測力的股票。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import entry_model as em
from market_calendar import is_twse_trading_day
from stock_api import FugleClient, clean_symbol

BASE = Path(__file__).resolve().parent
AUDIT_PATH = BASE / "output" / "research_reports" / "entry_model_audit.json"
AUDIT_HISTORY = BASE / "output" / "research_reports" / "entry_model_audit_history.jsonl"
TAIPEI = ZoneInfo("Asia/Taipei")
_client: FugleClient | None = None


def _fugle() -> FugleClient:
    global _client
    if _client is None:
        _client = FugleClient(wait_on_limit=True)
    return _client


def _num(v):
    try:
        v = float(v)
        return round(v, 4) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _model_for(symbol: str, hist: pd.DataFrame):
    key = str(pd.to_datetime(hist["date"]).dt.date.max())
    model = em.load_saved_model(symbol, key)
    if model is None:
        model = em.fit_model(hist)
        em.save_saved_model(symbol, key, model)
    return model


def _model_metrics(model) -> dict:
    return {
        "brier_skill": _num(getattr(model, "brier_skill", None)),
        "prob_has_skill": bool(getattr(model, "has_skill", False)),
        "oof_accuracy": _num(model.oof_opportunity_accuracy), "baseline_accuracy": _num(model.baseline_accuracy),
        "low_mae": _num(model.oof_mae), "low_naive_mae": _num(model.naive_mae),
        "low_has_skill": bool(getattr(model, "reg_has_skill", False)),
        "train_days": int(model.train_days), "oof_days": int(getattr(model, "oof_days", 0)),
    }


def analyze(symbol: str) -> dict:
    symbol = clean_symbol(symbol)
    client = _fugle()
    hist = em.load_training_intraday(client, symbol, days=140)
    hist["date"] = pd.to_datetime(hist["date"])
    last_day = hist["date"].dt.date.max()
    k5 = hist[hist["date"].dt.date == last_day]
    daily = client.historical_candles(symbol, date.today() - timedelta(days=150), date.today(), "D")
    model = _model_for(symbol, hist)
    now = datetime.now(TAIPEI)
    market_open = bool(is_twse_trading_day(now, BASE)) and (9, 0) <= (now.hour, now.minute) <= (13, 30)
    if market_open:
        # 歷史 5 分 K 不含今天；盤中改用今天的即時 5 分 K
        try:
            live = client.intraday_candles(symbol, "5")
            if live is not None and not live.empty:
                k5, last_day = live, now.date()
        except Exception:
            pass
    r = em.predict_entry(model, k5, float(k5["close"].iloc[-1]), daily=daily, market_open=market_open)
    keep = ("action", "decision_reason", "reference_price", "best_entry_price", "entry_lower", "entry_upper", "entry_score",
            "confidence", "trend_score", "trend_state", "intraday_trend_score", "daily_trend_score", "daily_trend_state",
            "pattern_name", "pattern_score", "pattern_desc", "daily_pattern_name", "daily_pattern_score", "daily_pattern_desc",
            "opportunity_probability", "model_has_skill", "low_model_has_skill", "vwap", "atr_price", "support_price",
            "overextended", "volume_ratio_6", "market_open")
    out = {k: (_num(r[k]) if isinstance(r[k], (float, np.floating)) else r[k]) for k in keep}
    return {"symbol": symbol, "data_date": str(last_day), "bar_time": str(r["feature_time"]),
            **out, "model": _model_metrics(model), "note": "技術面量化參考，不構成投資建議。"}


def audit(symbols: list[str] | None = None, save: bool = True) -> dict:
    if not symbols:
        symbols = [str(s) for s in json.loads((BASE / "user_watchlist.json").read_text(encoding="utf-8"))]
    previous, previous_at = {}, None
    if AUDIT_PATH.exists():
        try:
            last = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
            previous = {r["symbol"]: r for r in last.get("results", [])}
            previous_at = last.get("audited_at")
        except Exception:
            previous = {}
    results, errors = [], []
    for s in symbols:
        s = clean_symbol(s)
        try:
            hist = em.load_training_intraday(_fugle(), s, days=140)
            results.append({"symbol": s, **_model_metrics(em.fit_model(hist))})
        except Exception as exc:
            errors.append(f"{s}: {type(exc).__name__}: {exc}")
    lost = [r["symbol"] for r in results if previous.get(r["symbol"], {}).get("prob_has_skill") and not r["prob_has_skill"]]
    gained = [r["symbol"] for r in results if r["symbol"] in previous and not previous[r["symbol"]].get("prob_has_skill") and r["prob_has_skill"]]
    skills = [r["brier_skill"] for r in results if r["brier_skill"] is not None]
    report = {
        "audited_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "summary": {
            "symbols": len(results),
            "prob_model_useful": sum(r["prob_has_skill"] for r in results),
            "low_model_useful": sum(r["low_has_skill"] for r in results),
            "median_brier_skill": _num(float(np.median(skills))) if skills else None,
            "compared_with": previous_at or "（沒有上一次稽核）",
            "lost_skill_since_last": lost, "gained_skill_since_last": gained,
        },
        "results": results, "errors": errors,
    }
    if save:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        # 每次稽核的摘要與各檔 Brier skill 另存一行，之後可以看長期趨勢
        line = {"audited_at": report["audited_at"], **report["summary"],
                "brier_skill": {r["symbol"]: r["brier_skill"] for r in results}}
        with AUDIT_HISTORY.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="進場分析與模型稽核")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("symbol")
    b = sub.add_parser("audit")
    b.add_argument("symbols", nargs="*")
    ns = ap.parse_args()
    result = analyze(ns.symbol) if ns.cmd == "analyze" else audit(ns.symbols)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
