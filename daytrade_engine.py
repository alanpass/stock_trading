# -*- coding: utf-8 -*-
"""即時當沖判斷引擎（紙上交易，不連券商、不下真實委託）。

  python daytrade_engine.py plan                 # 盤前：資格檢查（處置／注意／不可當沖）＋方向偏好＋關鍵價位
  python daytrade_engine.py live                 # 盤中：每 60 秒判斷一次，模擬成交寫入 output/daytrade/<日期>/
  python daytrade_engine.py replay [--date D]    # 用當天 5 分 K 重播一次（檢查規則、盤後檢討）

策略：開盤區間（09:00～09:15）突破＋當日 VWAP，並用網站的資訊確認方向：
  日線趨勢（entry_model.daily_trend）、趨勢型態分類（trend_patterns_public.json）、
  進場模型的日內趨勢分數（entry_model.predict_entry）。

安全規則（避免日後用真實帳戶時無法沖銷、違約交割）：
  1. 排除處置股（含分盤撮合）、注意股、全額交割／非正常交易、異常推介、特殊異常股。
  2. 先買後賣需 canBuyDayTrade；先賣後買需 canDayTrade，平盤下放空另需平盤下可券賣。
  3. 距漲停／跌停 3% 內不進場；停利目標不得超過漲跌停前 1.5%。
  4. 持倉方向不利且距漲停（空單）／跌停（多單）1.5% 內立即沖銷，不等停損。
  5. 所有持倉總金額不超過資金：萬一被鎖住無法當沖，仍付得出交割款。
僅供模擬與研究，不構成投資建議。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import entry_model as em
from market_calendar import is_twse_trading_day
from stock_api import FugleClient, clean_symbol

BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "daytrade"
TAIPEI = ZoneInfo("Asia/Taipei")

CAPITAL = 1_000_000
RISK_PER_TRADE = 0.005        # 單筆最多虧資金 0.5%
MAX_POSITIONS = 3
DAILY_LOSS_LIMIT = 0.015      # 當日虧損達 1.5% 停止進場
MAX_STOP_PCT = 0.012          # 停損最遠 1.2%
REWARD_RISK = 2.0
LIMIT_ENTRY_BUFFER = 0.03     # 距漲跌停 3% 內不進場
LIMIT_EXIT_BUFFER = 0.015     # 持倉不利方向距漲跌停 1.5% 內立即沖銷
MAX_GAP = 0.05                # 開盤跳空超過 ±5% 當天不做
MIN_TURNOVER = 5e8            # 20 日平均成交值至少 5 億
FEE, TAX = 0.001425, 0.0015   # 手續費（單邊）、當沖證交稅
OR_END, LAST_ENTRY, FORCE_EXIT = "09:15", "12:30", "13:20"


def tick_size(p: float) -> float:
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0


def now_tpe() -> datetime:
    return datetime.now(TAIPEI)


# ---------------------------------------------------------------- 盤前資格與方向
def eligibility(t: dict) -> dict:
    """依 Fugle ticker 判斷能不能做、能做哪個方向。"""
    reasons = []
    if str(t.get("securityStatus", "NORMAL")) != "NORMAL":
        reasons.append(f"非正常交易（{t.get('securityStatus')}，可能為全額交割／停止交易）")
    if t.get("isDisposition") or int(t.get("matchingInterval") or 0) > 0:
        reasons.append("處置股（分盤撮合／預收款券）")
    if t.get("isAttention"):
        reasons.append("注意股（隨時可能被處置）")
    if t.get("isUnusuallyRecommended"):
        reasons.append("異常推介股")
    if t.get("isSpecificAbnormally"):
        reasons.append("特殊異常有價證券")
    blocked = bool(reasons)
    can_long = not blocked and bool(t.get("canBuyDayTrade"))
    can_short = not blocked and bool(t.get("canDayTrade"))
    if not blocked and not can_long:
        reasons.append("不可先買後賣當沖")
    if not blocked and not can_short:
        reasons.append("不可先賣後買當沖")
    return {"can_long": can_long, "can_short": can_short,
            "short_below_flat": bool(t.get("canBelowFlatMarginShortSell") or t.get("canBelowFlatSBLShortSell")),
            "reasons": reasons, "limit_up": float(t.get("limitUpPrice") or 0), "limit_down": float(t.get("limitDownPrice") or 0),
            "prev_close": float(t.get("previousClose") or t.get("referencePrice") or 0), "name": t.get("name", "")}


def _trend_patterns() -> dict:
    try:
        rows = json.loads((BASE / "output" / "research_reports" / "trend_patterns_public.json").read_text(encoding="utf-8"))["stocks"]
        return {clean_symbol(r["symbol"]): r for r in rows}
    except Exception:
        return {}


def build_plan(client: FugleClient, symbols: list[str]) -> dict:
    patterns = _trend_patterns()
    rows = []
    for s in symbols:
        s = clean_symbol(s)
        try:
            el = eligibility(client.ticker(s))
            daily = client.historical_candles(s, date.today() - timedelta(days=150), date.today(), "D")
        except Exception as exc:
            rows.append({"symbol": s, "tradable": False, "reasons": [f"資料取得失敗：{type(exc).__name__}"]})
            continue
        daily = daily[pd.to_datetime(daily["date"]).dt.date < date.today()].tail(120).reset_index(drop=True)
        if len(daily) < 30:
            rows.append({"symbol": s, "tradable": False, "reasons": ["日 K 不足"]})
            continue
        close = float(daily["close"].iloc[-1])
        tr = np.maximum(daily["high"] - daily["low"], np.maximum((daily["high"] - daily["close"].shift()).abs(), (daily["low"] - daily["close"].shift()).abs()))
        atr_pct = float(tr.tail(14).mean() / close)
        turnover = float((daily["close"] * daily["volume"]).tail(20).mean())
        trend = em.daily_trend(daily, close)
        pat = patterns.get(s, {})
        cls = str(pat.get("classification", ""))
        bull = trend.get("state") in ("多頭排列", "偏多") or cls in ("多頭延續／持續創高", "底部反轉候選")
        bear = trend.get("state") in ("空頭排列", "偏空") or cls in ("下降趨勢／弱勢整理", "高檔轉弱／反轉警示")
        allow_long, allow_short = el["can_long"], el["can_short"]
        if bull and not bear:
            allow_short = False
        elif bear and not bull:
            allow_long = False
        reasons = list(el["reasons"])
        if turnover < MIN_TURNOVER:
            reasons.append(f"成交值不足（20 日均 {turnover / 1e8:.1f} 億）")
        if close * 1000 > CAPITAL * 0.6:
            reasons.append(f"一張 {close * 1000 / 1e4:.0f} 萬，超過資金 60%")
        tradable = not reasons and (allow_long or allow_short)
        rows.append({
            "symbol": s, "name": el["name"], "tradable": tradable, "reasons": reasons,
            "allow_long": allow_long and tradable, "allow_short": allow_short and tradable,
            "short_below_flat": el["short_below_flat"],
            "prev_close": el["prev_close"] or close, "limit_up": el["limit_up"], "limit_down": el["limit_down"],
            "atr_pct": round(atr_pct * 100, 2), "turnover_e": round(turnover / 1e8, 1),
            "daily_trend": trend.get("state"), "pattern": pat.get("pattern_name"), "classification": cls,
        })
    return {"date": date.today().isoformat(), "generated_at": now_tpe().isoformat(timespec="seconds"),
            "capital": CAPITAL, "stocks": rows}


# ---------------------------------------------------------------- 盤中判斷與紙上成交
@dataclass
class Position:
    symbol: str
    side: int                 # +1 多、-1 空
    lots: int
    entry: float
    stop: float
    target: float
    opened_at: str
    exit: float | None = None
    closed_at: str | None = None
    exit_reason: str = ""
    pnl: float = 0.0


@dataclass
class Engine:
    plan: dict
    day: str
    positions: list[Position] = field(default_factory=list)
    traded: set = field(default_factory=set)
    skipped: dict = field(default_factory=dict)
    last_reason: dict = field(default_factory=dict)
    realized: float = 0.0
    events: list[dict] = field(default_factory=list)
    models: dict = field(default_factory=dict)
    dailies: dict = field(default_factory=dict)

    def log(self, t: str, symbol: str, kind: str, msg: str, **extra):
        if kind == "不進場":  # 同一檔同一個理由只記一次（風險金額這類數字會小幅變動，只比對理由前段）
            key = re.sub(r"[\d,.\s]+", "", msg.split("（")[0].split("，")[0])
            if self.last_reason.get(symbol) == key:
                return
            self.last_reason[symbol] = key
        e = {"time": t, "symbol": symbol, "type": kind, "msg": msg, **extra}
        self.events.append(e)
        print(f"[{t}] {symbol} {kind}：{msg}", flush=True)

    def open_positions(self):
        return [p for p in self.positions if p.exit is None]

    def _close(self, p: Position, price: float, t: str, reason: str):
        tk = tick_size(price)
        fill = price - tk if p.side > 0 else price + tk   # 市價沖銷多付 1 個 tick
        gross = (fill - p.entry) * p.side * p.lots * 1000
        cost = (p.entry + fill) * p.lots * 1000 * FEE + (fill if p.side > 0 else p.entry) * p.lots * 1000 * TAX
        p.exit, p.closed_at, p.exit_reason, p.pnl = fill, t, reason, round(gross - cost)
        self.realized += p.pnl
        self.log(t, p.symbol, "出場", f"{'多' if p.side > 0 else '空'}單 {reason}，{fill} 沖銷，損益 {p.pnl:+,} 元（含成本）",
                 pnl=p.pnl, price=fill)

    def on_bar(self, s: str, t: str, bars: pd.DataFrame, last: float, vwap: float, locked: str = ""):
        """bars：今天已完成的 5 分 K（含 time 欄 HH:MM）；last：最新價；vwap：當日均價；locked：'up'/'down'/''。"""
        info = next((r for r in self.plan["stocks"] if r["symbol"] == s), None)
        if not info or not info.get("tradable"):
            return
        lu, ld, pc = info["limit_up"], info["limit_down"], info["prev_close"]

        # 1) 持倉管理：漲跌停保護優先於停損
        for p in [p for p in self.open_positions() if p.symbol == s]:
            if p.side < 0 and lu and last >= lu * (1 - LIMIT_EXIT_BUFFER):
                if locked == "up":
                    self.log(t, s, "警示", "空單遇漲停鎖死，無法回補（真實帳戶需借券／違約風險）")
                    continue
                self._close(p, last, t, "逼近漲停緊急回補")
            elif p.side > 0 and ld and last <= ld * (1 + LIMIT_EXIT_BUFFER):
                if locked == "down":
                    self.log(t, s, "警示", "多單遇跌停鎖死，無法賣出（需全額交割）")
                    continue
                self._close(p, last, t, "逼近跌停緊急賣出")
            elif (p.side > 0 and last <= p.stop) or (p.side < 0 and last >= p.stop):
                self._close(p, last, t, "停損")
            elif (p.side > 0 and last >= p.target) or (p.side < 0 and last <= p.target):
                self._close(p, last, t, "停利")
            elif t >= FORCE_EXIT:
                self._close(p, last, t, "13:20 時間出場")

        # 2) 進場條件
        if s in self.traded or s in self.skipped or t < OR_END or t >= LAST_ENTRY:
            return
        if self.realized <= -CAPITAL * DAILY_LOSS_LIMIT or len(self.open_positions()) >= MAX_POSITIONS:
            return
        today_open = float(bars["open"].iloc[0])
        if pc and abs(today_open / pc - 1) > MAX_GAP:
            self.skipped[s] = f"開盤跳空 {today_open / pc - 1:+.1%}，超過 ±5%"
            self.log(t, s, "略過", self.skipped[s])
            return
        orb = bars[bars["time"] < OR_END]
        if len(orb) < 3:
            return
        orh, orl = float(orb["high"].max()), float(orb["low"].min())
        close = float(bars["close"].iloc[-1])

        side = 0
        if info["allow_long"] and close > orh and close > vwap:
            side = 1
        elif info["allow_short"] and close < orl and close < vwap:
            side = -1
        if not side:
            return

        # 網站的進場模型：日內趨勢分數確認方向
        model_note = ""
        model = self.models.get(s)
        if model is not None:
            try:
                r = em.predict_entry(model, bars, last, daily=self.dailies.get(s), market_open=True)
                its = float(r.get("intraday_trend_score", 50))
                model_note = f"日內趨勢 {its:.0f} 分、K 線 {r.get('pattern_name')}"
                if (side > 0 and its < 55) or (side < 0 and its > 45):
                    self.log(t, s, "不進場", f"突破成立但進場模型不支持（{model_note}）")
                    return
            except Exception:
                pass

        # 漲跌停距離與平盤下放空
        if lu and last >= lu * (1 - LIMIT_ENTRY_BUFFER):
            self.log(t, s, "不進場", f"距漲停 {lu} 不到 3%")
            return
        if ld and last <= ld * (1 + LIMIT_ENTRY_BUFFER):
            self.log(t, s, "不進場", f"距跌停 {ld} 不到 3%")
            return
        if side < 0 and last < pc and not info.get("short_below_flat"):
            self.log(t, s, "不進場", "平盤下不可放空")
            return

        tk = tick_size(last)
        entry = last + tk if side > 0 else last - tk           # 市價多付 1 個 tick
        stop = max(orl, entry * (1 - MAX_STOP_PCT)) if side > 0 else min(orh, entry * (1 + MAX_STOP_PCT))
        risk = abs(entry - stop)
        if risk <= 0:
            return
        target = entry + side * REWARD_RISK * risk
        if side > 0 and lu:
            target = min(target, lu * (1 - LIMIT_EXIT_BUFFER))
        if side < 0 and ld:
            target = max(target, ld * (1 + LIMIT_EXIT_BUFFER))
        if abs(target - entry) / entry < 2 * (FEE * 2 + TAX):  # 空間不夠付成本的兩倍
            self.log(t, s, "不進場", "停利空間太小，不夠支付交易成本")
            return
        lots = int(CAPITAL * RISK_PER_TRADE // (risk * 1000))
        if lots < 1:
            self.log(t, s, "不進場", f"一張風險 {risk * 1000:,.0f} 元，超過單筆上限 {CAPITAL * RISK_PER_TRADE:,.0f}")
            return
        notional = sum(p.entry * p.lots * 1000 for p in self.open_positions())
        lots = min(lots, int((CAPITAL - notional) // (entry * 1000)))
        if lots < 1:
            self.log(t, s, "不進場", "持倉總金額已達資金上限（避免違約交割）")
            return
        p = Position(s, side, lots, round(entry, 2), round(stop, 2), round(target, 2), t)
        self.positions.append(p)
        self.traded.add(s)
        self.log(t, s, "進場", f"{'做多' if side > 0 else '做空'} {lots} 張 @ {p.entry}，停損 {p.stop}、停利 {p.target}"
                 f"（區間 {orl}～{orh}、VWAP {vwap:.2f}{'、' + model_note if model_note else ''}）",
                 side=side, lots=lots, price=p.entry)

    def summary(self) -> dict:
        closed = [p for p in self.positions if p.exit is not None]
        wins = [p for p in closed if p.pnl > 0]
        return {"date": self.day, "trades": len(closed), "wins": len(wins),
                "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
                "pnl": round(self.realized), "pnl_pct": round(self.realized / CAPITAL * 100, 3),
                "open_positions": len(self.open_positions()),
                "positions": [asdict(p) for p in self.positions], "skipped": self.skipped}

    def save(self):
        folder = OUT / self.day
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plan.json").write_text(json.dumps(self.plan, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "events.json").write_text(json.dumps(self.events, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _prepare(engine: Engine, client: FugleClient):
    for r in engine.plan["stocks"]:
        if not r.get("tradable"):
            continue
        s = r["symbol"]
        engine.models[s] = em.load_latest_saved_model(s)
        d = client.historical_candles(s, date.today() - timedelta(days=150), date.today(), "D")
        engine.dailies[s] = d[pd.to_datetime(d["date"]).dt.date < date.fromisoformat(engine.day)].reset_index(drop=True)


def _bars_today(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if d["date"].dt.tz is not None:
        d["date"] = d["date"].dt.tz_convert(TAIPEI).dt.tz_localize(None)
    d["time"] = d["date"].dt.strftime("%H:%M")
    return d[(d["time"] >= "09:00") & (d["time"] <= "13:30")].reset_index(drop=True)


def replay(day: str | None = None) -> dict:
    client = FugleClient(wait_on_limit=True)
    plan = build_plan(client, _watchlist())
    day = day or date.today().isoformat()
    engine = Engine(plan, day)
    _prepare(engine, client)
    data = {}
    for r in plan["stocks"]:
        if r.get("tradable"):
            d = date.fromisoformat(day)
            data[r["symbol"]] = _bars_today(client.historical_candles(r["symbol"], d, d, "5"))
    times = sorted({t for b in data.values() for t in b["time"]})
    for t in times:
        bar_end = (datetime.strptime(t, "%H:%M") + timedelta(minutes=5)).strftime("%H:%M")
        for s, bars in data.items():
            done = bars[bars["time"] <= t]
            if done.empty:
                continue
            pv = ((done["high"] + done["low"] + done["close"]) / 3 * done["volume"]).sum()
            vwap = float(pv / done["volume"].sum()) if done["volume"].sum() else float(done["close"].iloc[-1])
            engine.on_bar(s, bar_end, done, float(done["close"].iloc[-1]), vwap)
    for p in engine.open_positions():
        engine._close(p, float(data[p.symbol]["close"].iloc[-1]), "13:30", "收盤強制出場")
    engine.save()
    return engine.summary()


def live() -> dict:
    client = FugleClient(wait_on_limit=True)
    now = now_tpe()
    if not is_twse_trading_day(now, BASE):
        print("今天不是交易日。")
        return {}
    engine = Engine(build_plan(client, _watchlist()), now.date().isoformat())
    _prepare(engine, client)
    engine.save()
    print(json.dumps([{k: r.get(k) for k in ("symbol", "name", "tradable", "allow_long", "allow_short", "reasons")}
                      for r in engine.plan["stocks"]], ensure_ascii=False, indent=1))
    while now_tpe().strftime("%H:%M") <= "13:25":
        t = now_tpe().strftime("%H:%M")
        if t < "09:00":
            time.sleep(20)
            continue
        for r in [r for r in engine.plan["stocks"] if r.get("tradable")]:
            s = r["symbol"]
            try:
                q = client.quote(s)
                bars = _bars_today(client.intraday_candles(s, "5"))
                if bars.empty:
                    continue
                # 只用已收完的 5 分 K 判斷突破；最新價用即時成交價
                done = bars[pd.to_datetime(bars["date"]) + pd.Timedelta(minutes=5) <= pd.Timestamp(now_tpe().replace(tzinfo=None))]
                last = float(q.get("lastPrice") or q.get("closePrice") or bars["close"].iloc[-1])
                vwap = float(q.get("avgPrice") or last)
                asks, bids = q.get("asks") or [], q.get("bids") or []
                locked = ("up" if r["limit_up"] and last >= r["limit_up"] and not asks else
                          "down" if r["limit_down"] and last <= r["limit_down"] and not bids else "")
                engine.on_bar(s, t, done if not done.empty else bars.iloc[:0], last, vwap, locked)
            except Exception as exc:
                engine.log(t, s, "錯誤", f"{type(exc).__name__}: {exc}")
        engine.save()
        time.sleep(max(5, 60 - now_tpe().second))
    for p in engine.open_positions():
        try:
            engine._close(p, float(client.quote(p.symbol).get("lastPrice")), now_tpe().strftime("%H:%M"), "收盤前強制出場")
        except Exception:
            pass
    engine.save()
    return engine.summary()


def _watchlist() -> list[str]:
    return [str(s) for s in json.loads((BASE / "user_watchlist.json").read_text(encoding="utf-8"))]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="即時當沖判斷引擎（紙上交易）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("live")
    rp = sub.add_parser("replay")
    rp.add_argument("--date")
    ns = ap.parse_args()
    if ns.cmd == "plan":
        plan = build_plan(FugleClient(wait_on_limit=True), _watchlist())
        folder = OUT / plan["date"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        for r in plan["stocks"]:
            side = "多空" if r.get("allow_long") and r.get("allow_short") else "只做多" if r.get("allow_long") else "只做空" if r.get("allow_short") else "不做"
            print(f"{r['symbol']} {r.get('name', ''):6s} {side:4s} {'；'.join(r.get('reasons', []))}")
        return 0
    result = live() if ns.cmd == "live" else replay(ns.date)
    print(json.dumps({k: v for k, v in result.items() if k != "positions"}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
