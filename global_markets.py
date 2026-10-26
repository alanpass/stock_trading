"""國際風向球：美股三大指標（道瓊、那斯達克、費城半導體）、布蘭特原油、VIX 恐慌指數，資料取自 Yahoo Finance 日線。"""
from __future__ import annotations

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

GLOBAL_INDICATORS = [
    {"symbol": "^DJI", "name": "道瓊工業指數", "unit": "點",
     "note": "美國 30 家大型藍籌股；反映傳產、金融與整體景氣信心。"},
    {"symbol": "^IXIC", "name": "那斯達克指數", "unit": "點",
     "note": "科技股比重高；台股電子權值股常跟著同向波動。"},
    {"symbol": "^SOX", "name": "費城半導體指數", "unit": "點",
     "note": "30 檔半導體股；與台積電、IC 設計、記憶體最連動。"},
    {"symbol": "BZ=F", "name": "布蘭特原油", "unit": "美元／桶",
     "note": "油價上漲推升通膨與運輸成本，對塑化、航運有直接影響。"},
    {"symbol": "^VIX", "name": "VIX 恐慌指數", "unit": "點",
     "note": "標普 500 選擇權隱含波動率；數字越高代表市場越恐慌。"},
]


def vix_regime(v: float) -> tuple[str, str]:
    """VIX 區間判讀（bear＝對股市偏空，以綠色表示）。"""
    if v < 15:
        return "平靜", "bull"
    if v < 20:
        return "正常", "flat"
    if v < 30:
        return "警戒", "bear"
    return "恐慌", "bear"


def fetch_indicator(symbol: str, rng: str = "3mo") -> dict:
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol)}",
                     params={"range": rng, "interval": "1d"}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    pts = [(t, c) for t, c in zip(res["timestamp"], closes) if c is not None]
    meta = res.get("meta", {})
    last = meta.get("regularMarketPrice") or pts[-1][1]
    prev = pts[-2][1] if len(pts) >= 2 else None
    return {
        "symbol": symbol,
        "last": float(last),
        "prev_close": prev,
        "change": float(last) - prev if prev else None,
        "change_pct": (float(last) / prev - 1) * 100 if prev else None,
        "asof": meta.get("regularMarketTime") or pts[-1][0],
        "history": pts,
    }
