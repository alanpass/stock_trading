# -*- coding: utf-8 -*-
"""5 分鐘 K 線進場分析（v3）。

組成：
  1. K 線型態：略過收盤集合競價等「零波動」K 棒，在 5 分 K 與日 K 上辨識單根／雙根／三根型態，
     型態分數 -1（偏空）～ +1（偏多），不會再出現 nan。
  2. 趨勢：日內（相對當日 VWAP、均線、動能）＋日線（MA20／MA60 位置與斜率）。
  3. 模型：預測「未來 6 根 5 分 K（同一交易日）內的最低價」與「出現 0.2%～2% 回檔的機率」。
     以「依交易日切分的前推驗證」計算真實樣本外表現；模型沒有贏過基準時，
     評分改用歷史回檔頻率，並在結果中標示，不拿訓練資料自評的數字當作準確率。

只提供技術面量化參考，不構成投資建議。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_VERSION = "v3"
FEATURES = ["body", "upper", "lower", "ret1", "ret3", "ret6", "ma3", "ma6", "ma12",
            "vol6", "vwap_gap", "position12", "day_progress"]
HORIZON_BARS = 6
DIP_MIN, DIP_MAX = -0.02, -0.002
SKILL_MARGIN = 0.02  # 樣本外 Brier skill score 至少 0.02（機率比「歷史回檔頻率」準 2% 以上）才採用模型機率


class EntryModel:
    """Joblib-safe container for the intraday entry model."""

    def __init__(self, reg, clf, features, horizon_bars, train_samples, train_days, oof_mae,
                 oof_direction_accuracy, oof_opportunity_accuracy, **extra):
        self.reg = reg
        self.clf = clf
        self.features = features
        self.horizon_bars = horizon_bars
        self.train_samples = train_samples
        self.train_days = train_days
        self.oof_mae = oof_mae
        self.oof_direction_accuracy = oof_direction_accuracy
        self.oof_opportunity_accuracy = oof_opportunity_accuracy
        self.version = extra.get("version", MODEL_VERSION)
        self.baseline_accuracy = extra.get("baseline_accuracy", np.nan)
        self.naive_mae = extra.get("naive_mae", np.nan)
        self.base_rate = extra.get("base_rate", 0.5)
        self.has_skill = bool(extra.get("has_skill", False))
        self.oof_days = extra.get("oof_days", 0)
        self.brier_skill = extra.get("brier_skill", np.nan)
        self.reg_has_skill = bool(extra.get("reg_has_skill", False))


def _symbol_path(base, code, key):
    p = Path(base) / "entry"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{code}_{key}_{MODEL_VERSION}.joblib"


def load_training_intraday(client, code, days=140):
    end = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
    start = end - pd.Timedelta(days=days)
    parts, cur, last_error = [], start, None
    while cur <= end:
        ce = min(cur + pd.Timedelta(days=329), end)
        try:
            p = client.historical_candles(code, cur.date(), ce.date(), "5")
            if not p.empty:
                parts.append(p)
        except Exception as exc:
            last_error = exc
        cur = ce + pd.Timedelta(days=1)
    if not parts:
        raise RuntimeError(f"無法取得5分鐘歷史K線（{last_error}）" if last_error else "無法取得5分鐘歷史K線")
    return pd.concat(parts, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 資料整理與特徵
# ---------------------------------------------------------------------------
def _clean(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        x[col] = pd.to_numeric(x.get(col), errors="coerce")
    x["volume"] = x["volume"].fillna(0)
    x = x.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    x["day"] = x["date"].dt.date
    return x


def _valid(x: pd.DataFrame) -> pd.Series:
    """有實際波動的 K 棒（排除收盤集合競價、零成交的平盤 K 棒）。"""
    return (x["high"] > x["low"]) & (x["volume"] > 0)


def _features(d: pd.DataFrame, horizon: int = HORIZON_BARS) -> pd.DataFrame:
    x = _clean(d)
    c, o, h, l, v = x["close"], x["open"], x["high"], x["low"], x["volume"]
    rng = h - l
    safe = rng.where(rng > 0)
    # 零波動 K 棒視為十字線：實體與影線都是 0，而不是 NaN
    x["body"] = ((c - o) / safe).fillna(0.0)
    x["upper"] = ((h - np.maximum(o, c)) / safe).fillna(0.0)
    x["lower"] = ((np.minimum(o, c) - l) / safe).fillna(0.0)
    x["ret1"], x["ret3"], x["ret6"] = c.pct_change(), c.pct_change(3), c.pct_change(6)
    for n in (3, 6, 12):
        x[f"ma{n}"] = c / c.rolling(n).mean() - 1
    x["vol6"] = v / v.rolling(6).mean().replace(0, np.nan)
    by_day = x.groupby("day", sort=False)
    typical = (h + l + c) / 3
    # VWAP 每個交易日重新累計
    x["vwap"] = (typical * v).groupby(x["day"]).cumsum() / v.groupby(x["day"]).cumsum().replace(0, np.nan)
    x["vwap"] = x["vwap"].fillna(c)
    x["vwap_gap"] = c / x["vwap"] - 1
    span = h.rolling(12).max() - l.rolling(12).min()
    x["position12"] = (c - l.rolling(12).min()) / span.replace(0, np.nan)
    x["day_progress"] = by_day.cumcount() / by_day["close"].transform("size").clip(lower=1)
    # 標籤：同一交易日內「接下來 horizon 根」的最低價（用 low，不跨日）
    future_low = l.groupby(x["day"]).transform(
        lambda s: s.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1))
    x["future_low_return"] = future_low / c - 1
    x["opportunity"] = ((x["future_low_return"] <= DIP_MAX) & (x["future_low_return"] >= DIP_MIN)).astype(float)
    x.loc[x["future_low_return"].isna(), "opportunity"] = np.nan
    return x


# ---------------------------------------------------------------------------
# K 線型態（5 分 K 與日 K 共用）
# ---------------------------------------------------------------------------
def detect_pattern(bars: pd.DataFrame) -> dict:
    """辨識最近 1～3 根有效 K 棒的型態。回傳 name、score（-1 偏空～+1 偏多）、desc。"""
    x = _clean(bars) if "day" not in bars.columns else bars
    x = x[_valid(x)] if "volume" in x.columns else x[x["high"] > x["low"]]
    if len(x) < 2:
        return {"name": "資料不足", "score": 0.0, "desc": "有效 K 棒不足，無法判斷型態"}
    rng = (x["high"] - x["low"]).astype(float)
    atr = float(rng.tail(14).mean()) or 1e-9
    o, h, l, c = (x[k].astype(float).to_numpy() for k in ("open", "high", "low", "close"))
    body = c - o
    size = np.abs(body)
    r = rng.to_numpy()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    prior = c[-1] / c[-6] - 1 if len(c) >= 6 else c[-1] / c[0] - 1  # 型態出現前的走勢
    rising, falling = prior > 0.003, prior < -0.003

    def big(i):
        if size[i] >= 0.6 * r[i] and size[i] >= 0.8 * atr:
            return True
        # 實體比例不夠，但漲跌 ≥ 1 ATR 或 ≥ 3%、且收在振幅上（下）1/3，也算長紅（長黑）
        if len(c) + i < 1:
            return False
        move_abs = abs(c[i] - c[i - 1])
        pos = (c[i] - l[i]) / r[i] if r[i] > 0 else 0.5
        large = move_abs >= atr or move_abs / c[i - 1] >= 0.03
        return large and ((body[i] > 0 and pos >= 2 / 3) or (body[i] < 0 and pos <= 1 / 3))

    if len(c) >= 3:
        if all(body[i] > 0 and size[i] >= 0.3 * atr for i in (-3, -2, -1)) and c[-1] > c[-2] > c[-3]:
            return {"name": "紅三兵", "score": 0.8, "desc": "連續三根上漲實體 K，買盤延續"}
        if all(body[i] < 0 and size[i] >= 0.3 * atr for i in (-3, -2, -1)) and c[-1] < c[-2] < c[-3]:
            return {"name": "黑三兵", "score": -0.8, "desc": "連續三根下跌實體 K，賣壓延續"}
        if body[-3] < 0 and big(-3) and size[-2] <= 0.3 * atr and body[-1] > 0 and c[-1] > (o[-3] + c[-3]) / 2:
            return {"name": "晨星", "score": 0.7, "desc": "長黑後出現小實體，再以紅 K 收復一半以上跌幅，可能止跌"}
        if body[-3] > 0 and big(-3) and size[-2] <= 0.3 * atr and body[-1] < 0 and c[-1] < (o[-3] + c[-3]) / 2:
            return {"name": "夜星", "score": -0.7, "desc": "長紅後出現小實體，再以黑 K 吃掉一半以上漲幅，可能轉弱"}
    if body[-2] < 0 < body[-1] and o[-1] <= c[-2] and c[-1] >= o[-2] and size[-1] > size[-2]:
        return {"name": "多頭吞噬", "score": 0.7, "desc": "紅 K 實體完全包覆前一根黑 K，買方轉強"}
    if body[-2] > 0 > body[-1] and o[-1] >= c[-2] and c[-1] <= o[-2] and size[-1] > size[-2]:
        return {"name": "空頭吞噬", "score": -0.7, "desc": "黑 K 實體完全包覆前一根紅 K，賣方轉強"}
    if big(-1):
        return ({"name": "長紅K", "score": 0.6, "desc": "實體大或大漲、收在高檔，買盤強勁"} if body[-1] > 0
                else {"name": "長黑K", "score": -0.6, "desc": "實體大或大跌、收在低檔，賣壓沉重"})
    if lower[-1] >= 2 * max(size[-1], 1e-9) and upper[-1] <= 0.3 * r[-1]:
        return ({"name": "錘子線", "score": 0.5, "desc": "下跌後長下影線，低檔有買盤承接"} if falling
                else {"name": "上吊線", "score": -0.3, "desc": "上漲後長下影線，留意高檔賣壓"} if rising
                else {"name": "長下影線", "score": 0.2, "desc": "盤中下探後拉回，有支撐"})
    if upper[-1] >= 2 * max(size[-1], 1e-9) and lower[-1] <= 0.3 * r[-1]:
        # 黑 K 收在最低附近＝衝高後被壓回，不算「倒錘線」的反攻訊號
        closed_at_low = body[-1] < 0 and lower[-1] <= 0.05 * r[-1]
        return ({"name": "流星線", "score": -0.5, "desc": "上漲後長上影線，高檔遇壓"} if rising
                else {"name": "倒錘線", "score": 0.3, "desc": "下跌後長上影線，買方嘗試反攻"} if falling and not closed_at_low
                else {"name": "長上影線", "score": -0.3 if closed_at_low else -0.2,
                      "desc": "衝高後被壓回並收在最低附近，上方賣壓重" if closed_at_low else "盤中衝高後拉回，上方有賣壓"})
    move = c[-1] / c[-2] - 1  # 相對前一根收盤的漲跌
    move_text = f"（較前一根 {move * 100:+.1f}%）"
    if size[-1] <= 0.1 * r[-1]:
        return {"name": "十字線", "score": -0.1 if rising else 0.1 if falling else 0.0, "desc": "多空拉鋸，等待方向" + move_text}
    return ({"name": "小紅K", "score": 0.2, "desc": "收紅但實體未達長紅標準" + move_text} if body[-1] > 0
            else {"name": "小黑K", "score": -0.2, "desc": "收黑但實體未達長黑標準" + move_text})


def _gap_note(k5: pd.DataFrame, daily: pd.DataFrame | None) -> str:
    """今天開盤相對前一日收盤跳空超過 1 個日 ATR 時提醒（日 K 型態只看得到前一天）。"""
    if daily is None or len(daily) < 15 or k5.empty:
        return ""
    d = _clean(daily)
    today = k5["day"].iloc[-1]
    prev = d[d["day"] < today]
    if prev.empty:
        return ""
    prev_close = float(prev["close"].iloc[-1])
    first_open = float(k5[k5["day"] == today]["open"].iloc[0])
    daily_atr = float((prev["high"] - prev["low"]).tail(14).mean())
    gap = first_open - prev_close
    if not np.isfinite(daily_atr) or daily_atr <= 0 or abs(gap) < daily_atr:
        return ""
    return f"今日跳空{'開高' if gap > 0 else '開低'} {gap / prev_close * 100:+.1f}%（超過 1 個日 ATR），日 K 型態為前一日結構，參考性降低"


def daily_trend(daily: pd.DataFrame | None, price: float) -> dict:
    """日線趨勢：價格相對 MA20／MA60 與 MA20 斜率，0～100。"""
    if daily is None or len(daily) < 25:
        return {"score": np.nan, "state": "日線資料不足", "ma20": np.nan, "ma60": np.nan}
    c = pd.to_numeric(daily["close"], errors="coerce").dropna()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean() if len(c) >= 60 else pd.Series(np.nan, index=c.index)
    m20, m60 = float(ma20.iloc[-1]), float(ma60.iloc[-1])
    slope = float(ma20.iloc[-1] / ma20.iloc[-6] - 1) if len(ma20.dropna()) >= 6 else 0.0

    def vs(ma, full):
        """價格在均線 ±0.5% 緩衝內只給一半分數，避免剛好壓在均線上就判成多或空。"""
        if not np.isfinite(ma):
            return full / 2
        gap = price / ma - 1
        return full if gap > 0.005 else 0 if gap < -0.005 else full / 2

    score = vs(m20, 40) + (30 if slope > 0.001 else 0 if slope < -0.001 else 15) + vs(m60, 30)
    if np.isfinite(m60) and price > m20 * 1.005 and m20 > m60 and slope > 0:
        state = "多頭排列"
    elif np.isfinite(m60) and price < m20 * 0.995 and m20 < m60 and slope < 0:
        state = "空頭排列"
    else:
        state = "偏多" if score >= 70 else "偏空" if score < 40 else "整理"
    return {"score": float(score), "state": state, "ma20": m20, "ma60": m60, "ma20_slope": slope}


# ---------------------------------------------------------------------------
# 模型訓練（含前推驗證）
# ---------------------------------------------------------------------------
def _rf(kind):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    common = dict(n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
    # 不用 class_weight="balanced"：評分使用的是機率，需要保持校準
    return RandomForestClassifier(**common) if kind == "clf" else RandomForestRegressor(**common)


def fit_model(hist, horizon_bars=HORIZON_BARS):
    x = _features(hist, horizon_bars)
    data = x[_valid(x)].dropna(subset=["future_low_return", "ret6", "ma12", "position12"]).copy()
    if len(data) < 200:
        raise RuntimeError("可用於訓練的 5 分鐘 K 線不足")
    X, y, yc = data[FEATURES].fillna(0), data["future_low_return"], data["opportunity"].astype(int)
    days = sorted(data["day"].unique())

    # 前推驗證：後 40% 的交易日分 4 段，每段只用之前的交易日訓練
    test_days = days[int(len(days) * 0.6):]
    folds = [f for f in np.array_split(np.array(test_days, dtype=object), 4) if len(f)]
    hits, total, abs_err, naive_err, base_hits, brier, brier_base = 0, 0, [], [], 0, [], []
    for fold in folds:
        tr = data["day"] < fold[0]
        te = data["day"].isin(set(fold))
        if tr.sum() < 200 or te.sum() == 0:
            continue
        clf, reg = _rf("clf").fit(X[tr], yc[tr]), _rf("reg").fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        prob = clf.predict_proba(X[te])[:, 1] if len(clf.classes_) == 2 else np.full(int(te.sum()), float(yc[tr].mean()))
        brier.extend((prob - yc[te]) ** 2)
        brier_base.extend((float(yc[tr].mean()) - yc[te]) ** 2)
        majority = int(yc[tr].mean() >= 0.5)
        hits += int((pred == yc[te]).sum())
        base_hits += int((yc[te] == majority).sum())
        total += int(te.sum())
        abs_err.extend(np.abs(reg.predict(X[te]) - y[te]))
        naive_err.extend(np.abs(float(y[tr].median()) - y[te]))
    acc = hits / total if total else np.nan
    base_acc = base_hits / total if total else np.nan
    mae = float(np.mean(abs_err)) if abs_err else np.nan
    naive = float(np.mean(naive_err)) if naive_err else np.nan
    bss = 1 - float(np.mean(brier)) / float(np.mean(brier_base)) if brier and np.mean(brier_base) > 0 else np.nan
    has_skill = bool(np.isfinite(bss) and bss >= SKILL_MARGIN)
    reg_has_skill = bool(np.isfinite(mae) and np.isfinite(naive) and mae <= naive * 0.98)

    clf, reg = _rf("clf").fit(X, yc), _rf("reg").fit(X, y)
    return EntryModel(reg, clf, FEATURES, horizon_bars, len(data), len(days), mae, acc, acc,
                      version=MODEL_VERSION, baseline_accuracy=base_acc, naive_mae=naive,
                      base_rate=float(yc.mean()), has_skill=has_skill, oof_days=len(test_days), brier_skill=bss,
                      reg_has_skill=reg_has_skill)


def save_saved_model(code, key, model):
    import joblib
    joblib.dump(model, _symbol_path(Path(__file__).resolve().parent / "models", code, key))


def load_latest_saved_model(code):
    """這檔股票最近一次訓練好的 v3 模型（訓練資料暫時取不到時備援）。"""
    import joblib
    files = sorted((Path(__file__).resolve().parent / "models" / "entry").glob(f"{code}_*_{MODEL_VERSION}.joblib"))
    for p in reversed(files):
        try:
            m = joblib.load(p)
            if getattr(m, "version", None) == MODEL_VERSION:
                return m
        except Exception:
            continue
    return None


def load_saved_model(code, key):
    import joblib
    p = _symbol_path(Path(__file__).resolve().parent / "models", code, key)
    m = joblib.load(p) if p.exists() else None
    return m if getattr(m, "version", None) == MODEL_VERSION else None


# ---------------------------------------------------------------------------
# 進場判斷
# ---------------------------------------------------------------------------
def predict_entry(model, k5, live_price, daily=None, market_open=True):
    x = _features(k5, getattr(model, "horizon_bars", HORIZON_BARS))
    valid = x[_valid(x)]
    row = (valid if not valid.empty else x).iloc[-1]  # 最後一根「有實際波動」的 K 棒
    c = float(live_price)
    vwap = float(x["vwap"].iloc[-1]) if np.isfinite(x["vwap"].iloc[-1]) else c
    ranges = (valid["high"] - valid["low"]) if not valid.empty else (x["high"] - x["low"])
    atr = float(ranges.tail(12).mean()) if len(ranges) else max(c * 0.003, 0.01)
    atr = atr if np.isfinite(atr) and atr > 0 else max(c * 0.003, 0.01)

    X = row[model.features].to_frame().T.astype(float).fillna(0)
    low_ret = float(model.reg.predict(X)[0])
    model_prob = float(model.clf.predict_proba(X)[0, 1]) if hasattr(model.clf, "predict_proba") else 0.5
    skill = bool(getattr(model, "has_skill", False))
    opp = model_prob if skill else float(getattr(model, "base_rate", 0.5))

    pattern = detect_pattern(x.tail(40))
    if pattern["name"] == "資料不足":
        pattern = {"name": "開盤初期", "score": 0.0, "desc": "今日有效 5 分 K 不足 2 根，先參考日 K 型態"}
    day_pat = detect_pattern(daily.tail(40)) if daily is not None and len(daily) >= 3 else {"name": "—", "score": 0.0, "desc": ""}
    dtrend = daily_trend(daily, c)

    def flag(v):
        return float(v) > 0 if np.isfinite(v) else False
    parts = {"VWAP": c > vwap, "MA6": flag(row["ma6"]), "MA12": flag(row["ma12"]), "動能": flag(row["ret6"])}
    intraday = 30 * parts["VWAP"] + 25 * parts["MA6"] + 20 * parts["MA12"] + 25 * parts["動能"]
    trend = intraday if not np.isfinite(dtrend["score"]) else 0.55 * intraday + 0.45 * dtrend["score"]

    pattern_part = (pattern["score"] + 1) * 50  # 0～100
    score = float(np.clip(0.55 * trend + 0.25 * opp * 100 + 0.20 * pattern_part, 0, 100))
    over = bool(c > vwap + 1.5 * atr)
    bearish = pattern["score"] <= -0.5 or day_pat["score"] <= -0.6
    # 動作與趨勢狀態使用同一組門檻：≥68 偏強、45～68 震盪、<45 偏弱
    if trend >= 68 and over:
        action = "上漲趨勢，但不追高"
    elif trend >= 68 and score >= 66 and not bearish:
        action = "適合回檔進場"
    elif trend >= 55 and score >= 52 and not bearish:
        action = "等待回檔確認"
    else:
        action = "不建議進場"

    detail = "／".join(f"{k}{'上' if k != '動能' and v else '下' if k != '動能' else '+' if v else '−'}" for k, v in parts.items())
    reasons = [
        f"日內趨勢{'偏多' if intraday >= 55 else '偏弱'}（{intraday:.0f} 分：{detail}；當日 VWAP {vwap:.2f}）",
        f"日線{dtrend['state']}" + (f"（MA20 {dtrend['ma20']:.2f}）" if np.isfinite(dtrend.get("ma20", np.nan)) else ""),
        f"5 分 K：{pattern['name']}，{pattern['desc']}",
    ]
    if day_pat["name"] != "—":
        reasons.append(f"日 K：{day_pat['name']}，{day_pat['desc']}")
    if over:
        reasons.append("價格高於 VWAP 超過 1.5 個 ATR，不追高")
    if bearish and trend >= 68:
        reasons.append("趨勢偏多但出現偏空 K 線型態，等待確認")
    if not skill:
        reasons.append("回檔機率模型樣本外未優於歷史頻率，改用歷史回檔頻率")
    gap_note = _gap_note(x, daily)
    if gap_note:
        reasons.append(gap_note)
    if not market_open:
        reasons.append("目前非盤中：以上為依最近收盤結構的下一個交易日參考")

    agree = np.sign(pattern["score"]) == np.sign(trend - 50)
    conf = float(np.clip(0.45 + abs(score - 50) / 150 + (0.08 if skill else -0.05) + (0.05 if agree else 0.0), 0.3, 0.85))
    # 最低價模型樣本外沒有贏過「歷史中位數」時，最佳進場價改用 現價 − 0.5 ATR
    reg_skill = bool(getattr(model, "reg_has_skill", False))
    best = max(0.0, c * (1 + low_ret)) if reg_skill else max(0.0, c - 0.5 * atr)
    support = float(min(x["low"].tail(12).min(), c))  # 含收盤競價那一根，支撐不會高於現價
    vol6 = float(row["vol6"]) if np.isfinite(row["vol6"]) else None  # 開盤不到 6 根時沒有量能比
    return {
        "action": action, "decision_reason": "；".join(reasons), "reference_price": c,
        "best_entry_price": best, "entry_score": score, "confidence": conf, "trend_score": float(trend),
        "intraday_trend_score": float(intraday), "daily_trend_score": dtrend["score"], "daily_trend_state": dtrend["state"],
        "pattern_name": pattern["name"], "pattern_score": float(pattern["score"]), "pattern_desc": pattern["desc"],
        "daily_pattern_name": day_pat["name"], "daily_pattern_score": float(day_pat["score"]), "daily_pattern_desc": day_pat["desc"],
        "opportunity_probability": opp, "model_probability": model_prob, "model_has_skill": skill,
        "low_model_has_skill": reg_skill,
        "vwap": vwap, "trend_state": "偏強／上漲" if trend >= 68 else "區間震盪" if trend >= 45 else "偏弱／下跌",
        "entry_lower": best - 0.4 * atr, "entry_upper": best + 0.4 * atr, "predicted_future_low": best,
        "overextended": over, "support_price": support, "predicted_rebound_price": float(c + atr * 0.8),
        "atr_price": atr, "volume_ratio_6": vol6, "feature_time": row["date"], "market_open": bool(market_open),
    }
