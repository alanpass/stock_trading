# -*- coding: utf-8 -*-
"""台股上市/上櫃產業分類主檔。

資料來源：Fugle v1.0 /intraday/tickers 的最新產業別欄位。
Fugle 文件明確提供 TWSE/TPEx、TSE/OTC 股票清單與 industry 代碼。
本模組會將全市場分類保存到 data/industry_master.csv，以避免每次畫面刷新重新查詢。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

INDUSTRY_OVERRIDES = {
    # 使用者目前研究／自選股：若舊快取缺產業碼，優先使用已確認的
    # TWSE/TPEx 產業分類，避免被舊版本的 "00 / ETF" 污染。
    "3481": "26",  # 群創：光電業
    "2327": "28",  # 國巨：電子零組件業
    "2492": "28",  # 華新科：電子零組件業
    "3037": "28",  # 欣興：電子零組件業
    "3044": "28",  # 健鼎：電子零組件業
    "3533": "28",  # 嘉澤：電子零組件業
    "2303": "24",  # 聯電：半導體業
    "2330": "24",  # 台積電：半導體業
    "2344": "24",  # 華邦電：半導體業
    "2408": "24",  # 南亞科：半導體業
    "6515": "24",  # 穎崴：半導體業
    "7769": "24",  # 鴻勁：半導體業
    "6488": "24",  # 環球晶：半導體業
    "3374": "24",  # 精材：半導體業
    "3450": "24",  # 聯鈞：半導體業
    "3006": "24",  # 晶豪科：半導體業
    "3661": "24",  # 世芯-KY：半導體業
    "8299": "24",  # 群聯：半導體業
    "2377": "25",  # 微星：電腦及週邊設備業
    "2308": "28",  # 台達電：電子零組件業
    "6274": "28",  # 台燿：電子零組件業
    "0050": "00",  # 元大台灣50：ETF，不納入股票產業排名
}



def normalize_industry(value) -> str:
    try:
        if pd.isna(value):
            return "00"
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return "00"
    return s.zfill(2)


def canonical_industry_for_symbol(symbol: str, fallback: str = "00") -> str:
    code = str(symbol or "").strip().upper()
    if "." in code:
        code = code.split(".")[0]
    return INDUSTRY_OVERRIDES.get(code, normalize_industry(fallback))


def build_master(client, path: Path, force: bool = False) -> pd.DataFrame:
    """取得 TSE + OTC 全市場一般股票分類，保存成本機快取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        try:
            old = pd.read_csv(path, dtype={"symbol": str, "industry": str})
            if not old.empty:
                old["symbol"] = old["symbol"].astype(str).str.strip().str.zfill(4)
                old["industry"] = old["industry"].map(normalize_industry)
                old["industry"] = old.apply(
                    lambda r: INDUSTRY_OVERRIDES.get(r["symbol"], r["industry"]), axis=1
                )
                old["industry_name"] = old["industry"].map(
                    lambda x: {
                        "01":"水泥工業","02":"食品工業","03":"塑膠工業","04":"紡織纖維","05":"電機機械","06":"電器電纜",
                        "08":"玻璃陶瓷","09":"造紙工業","10":"鋼鐵工業","11":"橡膠工業","12":"汽車工業","14":"建材營造",
                        "15":"航運業","16":"觀光餐旅","17":"金融保險","19":"綜合","20":"其他","21":"化學工業","22":"生技醫療業",
                        "23":"油電燃氣業","24":"半導體業","25":"電腦及週邊設備業","26":"光電業","27":"通信網路業","28":"電子零組件業",
                        "29":"電子通路業","30":"資訊服務業","31":"其他電子業","32":"文化創意業","33":"農業科技業","35":"綠能環保",
                        "36":"數位雲端","37":"運動休閒","38":"居家生活","80":"管理股票","00":"ETF/無產業分類"
                    }.get(str(x).zfill(2), "其他/未分類")
                )
                return old.drop_duplicates("symbol", keep="last")
        except Exception:
            pass

    rows = []
    for exchange in ("TWSE", "TPEx"):
        try:
            data = client.tickers(exchange)
        except Exception:
            data = []
        for x in data:
            code = str(x.get("symbol", "")).strip()
            if not code:
                continue
            rows.append({
                "symbol": code,
                "name": x.get("name", ""),
                "industry": normalize_industry(x.get("industry", "00")),
                "exchange": x.get("exchange", exchange),
                "market": x.get("market", ""),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["industry"] = df["industry"].map(normalize_industry)
    df["industry"] = df.apply(
        lambda r: INDUSTRY_OVERRIDES.get(r["symbol"], r["industry"]), axis=1
    )
    df = df.drop_duplicates("symbol", keep="last").sort_values("symbol")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip()
        if "industry" not in out.columns:
            out["industry"] = "00"
        out["industry"] = out["industry"].map(normalize_industry)
        out["industry"] = out.apply(
            lambda r: INDUSTRY_OVERRIDES.get(r["symbol"], r["industry"]), axis=1
        )
    return out
