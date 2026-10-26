# -*- coding: utf-8 -*-
"""自選股盤中焦點：從當日 1 分 K 找出「正在發生」的價量事件。

輸入一檔股票的當日 1 分 K、昨收、近期日 K（不含今天），由早到晚逐根掃描，
每個事件記下觸發的時間點，網站依時間軸列出（類似看盤軟體的「即時焦點股」）。
純計算、不打 API；資料由 stock_dashboard 負責抓取與快取。

事件分多方（bull）／空方（bear）；同一種事件有冷卻時間，避免盤整時一直重複觸發。
"""
from __future__ import annotations

import math

import pandas as pd

# ---- 門檻（百分比都以昨收為分母） ----
SURGE_BARS = 5          # 短線急拉／急殺：5 分鐘內
SURGE_PCT = 1.5         #   漲跌超過 1.5%
SURGE_COOLDOWN = 15     #   同方向 15 分鐘內不重複
BODY_PCT = 5.0          # 長紅棒／長黑棒：今日開盤到現價 ≥ 5%
SHADOW_PCT = 3.0        # 下影線／上影線：今日低（高）點與實體相距 ≥ 3%
SPIKE_MULT = 8.0        # 爆量：單分鐘量 ≥ 前 30 分鐘中位數 8 倍
SPIKE_DAILY_SHARE = 0.02  #   且 ≥ 20 日均量的 2%（排除冷門股零星幾張）
SPIKE_COOLDOWN = 20
CROSS_COOLDOWN = 60     # 翻紅／翻黑
LIMIT_COOLDOWN = 10     # 漲跌停打開後再觸及
OPEN_SKIP_BARS = 3      # 開盤前 3 分鐘價格跳動大，不判斷急拉／急殺與翻紅黑

# 說明文字（網站「判斷依據」展開區塊用）
RULES = [
    ("短線急拉／急殺", f"{SURGE_BARS} 分鐘內漲跌 ≥ 昨收的 {SURGE_PCT}%（同方向 {SURGE_COOLDOWN} 分鐘內不重複）"),
    ("翻紅／翻黑", f"價格由平盤下穿越到平盤上（或相反），{CROSS_COOLDOWN} 分鐘內不重複"),
    ("突破昨高／跌破昨低", "盤中第一次站上昨日最高價、跌破昨日最低價"),
    ("創 20／60 日新高、新低", "盤中價格第一次超過前 20（60）個交易日的最高價或最低價"),
    ("長紅棒／長黑棒 5%", f"今日開盤到現價漲跌 ≥ 昨收的 {BODY_PCT:.0f}%"),
    ("下影線／上影線 3%", f"今日低點到實體下緣（或高點到實體上緣）≥ 昨收的 {SHADOW_PCT:.0f}%，代表低檔有承接／高檔有賣壓"),
    ("爆量上攻／爆量下殺", f"單分鐘成交量 ≥ 前 30 分鐘中位數的 {SPIKE_MULT:.0f} 倍，且 ≥ 20 日均量的 {SPIKE_DAILY_SHARE:.0%}；依該分鐘紅黑判斷方向"),
    ("量增：達均量／爆量兩倍", "今日累計成交量第一次達到 20 日均量、兩倍；方向依當下漲跌"),
    ("觸及漲停／跌停、打開", "第一次觸及漲跌停；觸及後又離開漲跌停價（漲停打開偏空、跌停打開偏多）"),
]


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _tick(price: float) -> float:
    for bound, tick in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)):
        if price < bound:
            return tick
    return 5.0


def limit_prices(ref: float) -> tuple[float, float]:
    """依昨收推算漲跌停價（±10%，漲停向下、跌停向上取到檔位）。"""
    ref = _f(ref)
    if math.isnan(ref) or ref <= 0:
        return math.nan, math.nan
    up, down = ref * 1.1, ref * 0.9
    up = math.floor(round(up / _tick(up), 6)) * _tick(up)
    down = math.ceil(round(down / _tick(down), 6)) * _tick(down)
    return round(up, 2), round(down, 2)


def scan(k1: pd.DataFrame, prev_close: float, daily: pd.DataFrame | None = None,
         limit_up: float = math.nan, limit_down: float = math.nan) -> list[dict]:
    """回傳事件列表：[{time, key, text, side, price, pct}]，依時間先後排序。

    k1：當日 1 分 K（date/open/high/low/close/volume，volume 單位＝張）。
    daily：今天以前的日 K（high/low/volume，volume 單位＝股）；沒有就略過日線類事件。
    """
    ref = _f(prev_close)
    if k1 is None or k1.empty or not ref or math.isnan(ref):
        return []
    k = k1.sort_values("date").reset_index(drop=True)
    lu, ld = _f(limit_up), _f(limit_down)
    if math.isnan(lu) or math.isnan(ld):
        calc_up, calc_down = limit_prices(ref)
        lu = calc_up if math.isnan(lu) else lu
        ld = calc_down if math.isnan(ld) else ld

    prev_high = prev_low = math.nan
    hi20 = lo20 = hi60 = lo60 = avg_vol = math.nan
    if daily is not None and not daily.empty:
        d = daily.sort_values("date")
        prev_high, prev_low = _f(d["high"].iloc[-1]), _f(d["low"].iloc[-1])
        hi20, lo20 = _f(d["high"].tail(20).max()), _f(d["low"].tail(20).min())
        if len(d) >= 60:
            hi60, lo60 = _f(d["high"].tail(60).max()), _f(d["low"].tail(60).min())
        avg_vol = _f(d["volume"].tail(20).mean()) / 1000   # 股 → 張

    events: list[dict] = []
    last_fire: dict[str, int] = {}
    once: set[str] = set()

    def fire(i: int, key: str, text: str, side: str, price: float, cooldown: int | None = None) -> None:
        if cooldown is None:
            if key in once:
                return
            once.add(key)
        elif i - last_fire.get(key, -10**9) < cooldown:
            return
        last_fire[key] = i
        events.append({"time": k.at[i, "date"], "key": key, "text": text, "side": side,
                       "price": price, "pct": (price - ref) / ref * 100})

    day_open = _f(k.at[0, "open"])
    day_high, day_low = -math.inf, math.inf
    cum_vol = 0.0
    at_up = at_down = False
    closes = k["close"].astype(float).tolist()
    vols = k["volume"].astype(float).fillna(0).tolist()

    for i in range(len(k)):
        h, l, c = _f(k.at[i, "high"]), _f(k.at[i, "low"]), closes[i]
        o = _f(k.at[i, "open"])
        new_high, new_low = h > day_high, l < day_low
        day_high, day_low = max(day_high, h), min(day_low, l)
        cum_vol += vols[i]

        # 漲跌停
        if not math.isnan(lu):
            if h >= lu - 1e-9 and not at_up:
                fire(i, "limit_up", "觸及漲停", "bull", lu, LIMIT_COOLDOWN)
                at_up = True
            elif at_up and c < lu - 1e-9:
                fire(i, "limit_up_open", "漲停打開", "bear", c, LIMIT_COOLDOWN)
                at_up = False
        if not math.isnan(ld):
            if l <= ld + 1e-9 and not at_down:
                fire(i, "limit_down", "觸及跌停", "bear", ld, LIMIT_COOLDOWN)
                at_down = True
            elif at_down and c > ld + 1e-9:
                fire(i, "limit_down_open", "跌停打開", "bull", c, LIMIT_COOLDOWN)
                at_down = False

        # 日線關卡：60 日優先於 20 日，同一根只報較大的那個
        if new_high:
            if not math.isnan(hi60) and h > hi60 and hi60 > hi20 - 1e-9:
                fire(i, "hi60", "創 60 日新高", "bull", h)
                once.add("hi20")
            elif not math.isnan(hi20) and h > hi20:
                fire(i, "hi20", "創 20 日新高", "bull", h)
            elif not math.isnan(prev_high) and h > prev_high:
                fire(i, "prev_high", "突破昨高", "bull", h)
        if new_low:
            if not math.isnan(lo60) and l < lo60 and lo60 < lo20 + 1e-9:
                fire(i, "lo60", "創 60 日新低", "bear", l)
                once.add("lo20")
            elif not math.isnan(lo20) and l < lo20:
                fire(i, "lo20", "創 20 日新低", "bear", l)
            elif not math.isnan(prev_low) and l < prev_low:
                fire(i, "prev_low", "跌破昨低", "bear", l)

        if i >= OPEN_SKIP_BARS:
            # 短線急拉／急殺
            if i >= SURGE_BARS:
                move = (c - closes[i - SURGE_BARS]) / ref * 100
                if move >= SURGE_PCT:
                    fire(i, "surge_up", "短線急拉", "bull", c, SURGE_COOLDOWN)
                elif move <= -SURGE_PCT:
                    fire(i, "surge_down", "短線急殺", "bear", c, SURGE_COOLDOWN)
            # 翻紅／翻黑
            pc = closes[i - 1]
            if pc <= ref < c:
                fire(i, "turn_up", "翻紅", "bull", c, CROSS_COOLDOWN)
            elif pc >= ref > c:
                fire(i, "turn_down", "翻黑", "bear", c, CROSS_COOLDOWN)

        # 長紅／長黑棒、上下影線（以今日開盤到現價為實體）
        body = (c - day_open) / ref * 100
        if body >= BODY_PCT:
            fire(i, "long_red", f"長紅棒 {BODY_PCT:.0f}%", "bull", c)
        elif body <= -BODY_PCT:
            fire(i, "long_black", f"長黑棒 {BODY_PCT:.0f}%", "bear", c)
        if (min(day_open, c) - day_low) / ref * 100 >= SHADOW_PCT:
            fire(i, "low_shadow", f"下影線 {SHADOW_PCT:.0f}%", "bull", c)
        if (day_high - max(day_open, c)) / ref * 100 >= SHADOW_PCT:
            fire(i, "up_shadow", f"上影線 {SHADOW_PCT:.0f}%", "bear", c)

        # 單分鐘爆量（13:30 收盤集合競價那一根量本來就大，不算）
        if i >= 10 and k.at[i, "date"].strftime("%H:%M") < "13:25":
            base = pd.Series(vols[max(0, i - 30):i]).median()
            big_enough = math.isnan(avg_vol) or vols[i] >= avg_vol * SPIKE_DAILY_SHARE
            if base > 0 and vols[i] >= base * SPIKE_MULT and big_enough:
                if c >= o:
                    fire(i, "spike_up", "爆量上攻", "bull", c, SPIKE_COOLDOWN)
                else:
                    fire(i, "spike_down", "爆量下殺", "bear", c, SPIKE_COOLDOWN)

        # 累計量：達 20 日均量、兩倍
        if not math.isnan(avg_vol) and avg_vol > 0:
            side = "bull" if c >= ref else "bear"
            if cum_vol >= avg_vol * 2:
                fire(i, "vol_x2", "爆量兩倍", side, c)
                once.add("vol_x1")
            elif cum_vol >= avg_vol:
                fire(i, "vol_x1", "量達均量", side, c)

    return events
