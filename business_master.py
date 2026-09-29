# -*- coding: utf-8 -*-
"""公司主要業務／產業鏈細分主檔。

設計原則：
1. 第一層仍使用 TWSE/TPEx 的官方產業代碼（24/25/26/...）。
2. 第二層使用 TPEx「產業價值鏈資訊平台」的公司個體頁面，取得
   「產業 > 子鏈」資訊；該平台同時涵蓋上市、上櫃等公司個體鏈。
3. 若官方價值鏈尚無資料，再使用公司名稱/官方基本資料文字做保守分類。
4. 不把 LLM 猜測當作正式分類；LLM 只能在研究報告中解釋。

主檔：data/business_master.csv
欄位：symbol,name,industry,industry_name,business_group,primary_chain,all_chain_paths,source_url,updated_at
"""
from __future__ import annotations

import html
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

try:
    from stock_api import INDUSTRIES, clean_symbol
    from industry_master import INDUSTRY_OVERRIDES, normalize_industry
except Exception:
    INDUSTRIES = {}
    INDUSTRY_OVERRIDES = {}
    def normalize_industry(v):
        return str(v or "00").strip().zfill(2)
    def clean_symbol(s):
        return str(s).strip().upper().split(".")[0]

TPEX_CHAIN_URL = "https://ic.tpex.org.tw/company_chain.php?stk_code={}"
TAIPEI_TZ = "Asia/Taipei"

KNOWN_INDUSTRY_CODES = {
    "3481":"26","2327":"28","2492":"28","3037":"28","3044":"28","3533":"28",
    "2303":"24","2330":"24","2344":"24","2408":"24","6515":"24","7769":"24",
    "6488":"24","3374":"24","3450":"24","3006":"24","3661":"24","8299":"24",
    "2377":"25","2308":"28","6274":"28","0050":"00",
}

# 使用者目前自選股與近期常查標的的明確分類；正式主檔仍會以官方公司鏈資料優先。
BUSINESS_OVERRIDES: dict[str, dict[str, str]] = {
    "3481": {"business_group": "面板／顯示器", "primary_chain": "平面顯示器 > 面板"},
    "2327": {"business_group": "被動元件／電阻電容電感", "primary_chain": "被動元件 > 電阻器"},
    "2492": {"business_group": "被動元件／電阻電容電感", "primary_chain": "被動元件 > 電容器"},
    "3037": {"business_group": "PCB／IC載板", "primary_chain": "印刷電路板 > 硬板、軟板、IC載板製造"},
    "3044": {"business_group": "PCB／高階電路板", "primary_chain": "印刷電路板 > 硬板、軟板、IC載板製造"},
    "3533": {"business_group": "連接器／高速互連", "primary_chain": "連接器 > 連接器設計、組裝及製造"},
    "2303": {"business_group": "晶圓代工", "primary_chain": "半導體 > 晶圓製造"},
    "2330": {"business_group": "晶圓代工／先進製程", "primary_chain": "半導體 > 晶圓製造"},
    "2344": {"business_group": "記憶體／Flash", "primary_chain": "半導體 > 記憶體"},
    "2408": {"business_group": "記憶體／DRAM", "primary_chain": "半導體 > 記憶體"},
    "6515": {"business_group": "半導體測試／測試介面", "primary_chain": "半導體 > 生產製程及檢測設備"},
    "7769": {"business_group": "半導體設備／測試", "primary_chain": "半導體 > 生產製程及檢測設備"},
    "6488": {"business_group": "半導體材料／矽晶圓", "primary_chain": "半導體 > 矽晶圓／矽晶片"},
    "3374": {"business_group": "封裝測試／晶圓級封裝", "primary_chain": "半導體 > IC封裝測試"},
    "3450": {"business_group": "封裝測試／光通訊", "primary_chain": "半導體 > IC封裝測試"},
    "6274": {"business_group": "CCL／銅箔基板", "primary_chain": "印刷電路板 > 銅箔基板"},
    "3006": {"business_group": "IC設計／記憶體IC", "primary_chain": "半導體 > IC設計"},
    "3661": {"business_group": "ASIC／SoC設計服務", "primary_chain": "半導體 > IP設計／IC設計代工服務"},
    "8299": {"business_group": "儲存控制／NAND方案", "primary_chain": "半導體 > 儲存控制器IC"},
    "2308": {"business_group": "電源供應／AI電源", "primary_chain": "電源與電子零組件 > 電源供應器"},
    "2377": {"business_group": "電腦周邊／電競硬體", "primary_chain": "電腦及週邊設備 > 筆記型電腦／桌上型電腦及周邊"},
    "0050": {"business_group": "ETF／指數投資", "primary_chain": "ETF"},
}

# 已確認的常用標的名稱 fallback。這只用於避免 API 暫時沒有 name 時顯示 nan，
# 不取代官方資料；若 Fugle 能取得最新名稱仍優先使用 Fugle。
NAME_OVERRIDES = {
    "3481": "群創", "2327": "國巨", "2492": "華新科", "3037": "欣興", "3044": "健鼎",
    "3533": "嘉澤", "2303": "聯電", "2330": "台積電", "2344": "華邦電", "2408": "南亞科",
    "6515": "穎崴", "7769": "鴻勁", "6488": "環球晶", "3374": "精材", "3450": "聯鈞",
    "2377": "微星", "3006": "晶豪科", "3661": "世芯-KY", "8299": "群聯", "2308": "台達電",
    "6274": "台燿", "0050": "元大台灣50",
}

# 官方子鏈名稱轉成較容易閱讀的研究分類。


# 研究 Agent 用的「公司主要業務 → 研究主題」規則。
# 這些不是投資結論，而是把官方產業鏈/子鏈轉成更精準的搜尋與查證關鍵字。
BUSINESS_RESEARCH_RULES = [
    (("晶圓代工", "晶圓製造"), ["晶圓代工", "先進製程", "成熟製程", "稼動率", "晶圓價格", "CAPEX", "AI/HPC"]),
    (("記憶體", "DRAM"), ["DRAM", "DDR5", "HBM", "記憶體價格", "庫存", "伺服器需求", "AI server"]),
    (("Flash", "NAND", "儲存控制"), ["NAND", "Flash", "SSD", "NAND價格", "PCIe", "企業SSD", "UFS"]),
    (("封裝", "IC封裝測試", "晶圓級封裝", "先進封裝"), ["封裝測試", "WLCSP", "先進封裝", "CoWoS", "Chiplet", "封裝需求"]),
    (("測試介面", "測試設備", "檢測設備"), ["Test Socket", "Probe Card", "Burn-in", "SLT", "測試設備", "HPC測試"]),
    (("矽晶圓", "晶圓", "半導體材料"), ["矽晶圓", "300mm", "200mm", "晶圓需求", "稼動率", "晶圓廠CAPEX"]),
    (("ASIC", "SoC", "IC設計", "IC設計服務"), ["ASIC", "SoC", "AI accelerator", "tape-out", "design win", "NRE", "HPC"]),
    (("被動元件", "電阻電容電感"), ["MLCC", "電阻", "電容", "電感", "被動元件價格", "稼動率", "AI伺服器", "車用電子"]),
    (("PCB", "IC載板"), ["PCB", "ABF", "BT載板", "HDI", "CCL", "AI伺服器", "稼動率", "接單"]),
    (("連接器", "高速互連"), ["高速連接器", "高速互連", "PCIe", "112G", "224G", "800G", "AI伺服器", "網通"]),
    (("面板", "顯示器", "觸控"), ["面板價格", "LCD", "OLED", "Mini LED", "Micro LED", "IT面板", "TV面板", "稼動率"]),
    (("伺服器", "AI運算", "PC", "電腦整機"), ["AI伺服器", "Server", "ODM", "GPU", "機櫃", "出貨", "CAPEX"]),
    (("電源供應", "電力設備"), ["PSU", "BBU", "800V", "資料中心電源", "AI伺服器電源", "電源供應器", "電力需求"]),
    (("散熱", "熱管理"), ["液冷", "冷板", "熱管", "散熱模組", "資料中心", "AI伺服器散熱"]),
    (("光通訊", "網通", "網路設備"), ["800G", "1.6T", "光模組", "交換器", "資料中心", "AI網路"]),
    (("銀行", "金融"), ["淨利差", "NIM", "放款", "信用成本", "資產品質", "利率", "金融市場"]),
    (("保險",), ["保費", "投資收益", "利率", "理賠", "金融市場"]),
    (("航運", "海運"), ["SCFI", "貨櫃運價", "運能", "燃油", "船隊", "港口"]),
    (("生技", "製藥", "醫療器材"), ["臨床試驗", "FDA", "藥證", "CDMO", "醫材", "產品認證"]),
    (("電機", "自動化", "傳動"), ["自動化", "馬達", "伺服", "機器人", "傳動", "工控", "資本支出"]),
    (("汽車", "汽車零組件"), ["車用電子", "電動車", "ADAS", "OEM", "Tier1", "產量", "零組件"]),
    (("建設", "營建"), ["房價", "餘屋", "預售", "推案", "利率", "土地成本", "建築成本"]),
    (("食品", "飲料"), ["原物料", "通路", "食品價格", "消費", "毛利率"]),
    (("石化", "塑膠", "橡膠"), ["原油", "石化產品價格", "乙烯", "塑化價差", "庫存", "需求"]),
    (("鋼鐵",), ["鋼價", "鐵礦砂", "煤", "產能", "庫存", "基建需求"]),
]

def build_business_research_context(profile: dict | None) -> dict:
    """將公司官方產業鏈/主要業務轉為研究 Agent 可用的查詢上下文。"""
    profile = profile or {}
    group = str(profile.get("business_group") or "").strip()
    primary = str(profile.get("primary_chain") or "").strip()
    all_paths = str(profile.get("all_chain_paths") or "").strip()
    text = " ".join([group, primary, all_paths])
    keywords: list[str] = []
    for needles, vals in BUSINESS_RESEARCH_RULES:
        if any(n.lower() in text.lower() for n in needles):
            for v in vals:
                if v not in keywords:
                    keywords.append(v)
    # 從官方子鏈名稱抽取少量高辨識度詞，避免只靠公司名稱搜尋。
    chain_tokens = []
    for part in re.split(r"[>|｜/、,， ]+", primary):
        part = part.strip()
        if 2 <= len(part) <= 16 and part not in {"半導體", "電子零組件", "電腦及週邊設備"}:
            chain_tokens.append(part)
    for v in chain_tokens:
        if v not in keywords:
            keywords.append(v)
    keywords = keywords[:16]
    return {
        "symbol": clean_symbol(profile.get("symbol", "")),
        "name": str(profile.get("name") or "").strip(),
        "industry": str(profile.get("industry") or "00").zfill(2),
        "industry_name": str(profile.get("industry_name") or ""),
        "business_group": group,
        "primary_chain": primary,
        "all_chain_paths": all_paths,
        "research_keywords": keywords,
        "business_scope_text": f"{group}｜{primary}" if group or primary else "主要業務待建立",
        "source_type": profile.get("source_type"),
        "source_url": profile.get("source_url"),
    }

ALIASES = [
    ("晶圓製造", "晶圓代工／晶圓製造"),
    ("IC封裝測試", "封裝測試／先進封裝"),
    ("生產製程及檢測設備", "半導體設備／測試"),
    ("IC設計代工服務", "ASIC／IC設計服務"),
    ("IP設計/IC設計代工服務", "ASIC／IC設計服務"),
    ("電阻器", "被動元件／電阻電容電感"),
    ("電容器", "被動元件／電阻電容電感"),
    ("電感器", "被動元件／電阻電容電感"),
    ("濾波器、振盪器", "被動元件／電阻電容電感"),
    ("硬板、軟板、IC載板製造", "PCB／IC載板"),
    ("連接器設計、組裝及製造", "連接器／高速互連"),
    ("電源供應器", "電源供應／AI電源"),
    ("散熱片、風扇馬達、散熱模組", "散熱／熱管理"),
    ("記憶體", "記憶體／儲存"),
    ("磁碟儲存控制器IC", "儲存控制／NAND方案"),
    ("輸出入介面IC", "介面IC／SoC"),
    ("消費性IC", "消費性／特殊應用IC"),
    ("影像感測IC", "影像感測／感測IC"),
    ("化學品", "半導體材料／化學品"),
    ("IC通路", "IC通路／電子代理"),
    ("伺服器", "伺服器／AI運算"),
    ("筆記型電腦", "PC／電腦整機"),
    ("桌上型電腦", "PC／電腦整機"),
    ("其他電腦及週邊設備之零組件", "電腦零組件"),
    ("面板", "面板／顯示器"),
    ("觸控面板", "觸控／顯示器"),
    ("光通訊設備", "光通訊／網通"),
    ("網路設備", "網通／網路設備"),
    ("應用/系統軟體設計開發", "軟體／系統整合"),
    ("系統整合服務", "軟體／系統整合"),
    ("銀行", "銀行／金融"),
    ("證券", "證券／金融"),
    ("保險", "保險／金融"),
    ("輪胎", "輪胎／橡膠"),
    ("汽車零組件", "汽車零組件"),
    ("海運", "海運／貨櫃"),
    ("航空", "航空／旅運"),
    ("觀光旅館", "旅館／餐旅"),
    ("營造", "建築／營造"),
    ("建設業", "建築／營造"),
    ("電線電纜", "電線電纜／電力設備"),
    ("變壓器", "電力設備／變壓器"),
    ("電控元件", "電機／自動化"),
    ("傳動元件", "電機／自動化"),
    ("醫療器材", "醫療器材"),
    ("製藥", "製藥／新藥"),
    ("食品", "食品／飲料"),
]


def _normalize_group(chain: str) -> str:
    s = str(chain or "").strip()
    if not s:
        return "主要業務待建立"
    for needle, group in ALIASES:
        if needle in s:
            return group
    return s.split(" > ")[-1].strip() or "主要業務待建立"


def _strip_html(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "\n", text)
    return html.unescape(text)


def fetch_chain_paths(symbol: str, timeout: int = 6) -> tuple[list[str], str]:
    """從 TPEx 產業價值鏈個體公司頁面抓取「► 產業 > 子鏈」。"""
    code = clean_symbol(symbol)
    url = TPEX_CHAIN_URL.format(code)
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 TaiwanStockDashboard/1.0"})
        r.raise_for_status()
        text = _strip_html(r.text)
        # 文字化後可能仍是多行；抓「半導體 > ...」這類鏈路。
        paths = []
        for line in re.split(r"[\r\n]+", text):
            line = re.sub(r"\s+", " ", line).strip(" \t►")
            if " > " in line and len(line) < 120 and not line.startswith("http"):
                if re.search(r"(半導體|電腦及週邊設備|印刷電路板|被動元件|通信網路|平面顯示器|電機機械|金融|汽車|食品|鋼鐵|航運|生技|油電燃氣|建材營造|軟體服務|貿易百貨)", line):
                    paths.append(line)
        # 去重並保留順序
        out = []
        seen = set()
        for p in paths:
            p = p.replace("►", "").strip()
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out, url
    except Exception:
        return [], url


def _fallback_group(industry: str, name: str) -> str:
    text = f"{name}"
    rules = {
        "01": [("水泥", "水泥／建材")],
        "02": [("食品", "食品／飲料")],
        "03": [("塑", "塑膠／石化")],
        "04": [("紡", "紡織／成衣")],
        "05": [("電機", "電機／自動化")],
        "06": [("電纜", "電線電纜／電力設備")],
        "10": [("鋼", "鋼鐵／金屬")],
        "12": [("車", "汽車／零組件")],
        "15": [("航", "航運／物流")],
        "17": [("銀", "銀行／金融"), ("保", "保險／金融")],
        "22": [("藥", "製藥／新藥"), ("醫", "醫療器材")],
        "24": [("晶", "半導體／待細分")],
        "25": [("電", "電腦／伺服器／周邊")],
        "26": [("面", "面板／顯示器")],
        "28": [("連", "連接器／高速互連"), ("PCB", "PCB／IC載板")],
        "29": [("通路", "IC通路／電子代理")],
        "30": [("軟", "軟體／系統整合")],
        "32": [("遊戲", "遊戲／數位內容")],
        "35": [("電池", "儲能／電池"), ("太陽", "太陽能／光電"), ("風", "風電／電力")],
        "36": [("雲", "雲端／SaaS"), ("資安", "資安"), ("AI", "人工智慧／軟體")],
        "37": [("運動", "運動用品")],
        "38": [("家", "居家／零售")],
    }
    for needle, group in rules.get(str(industry).zfill(2), []):
        if needle.lower() in text.lower():
            return group
    return f"{INDUSTRIES.get(str(industry).zfill(2), '其他')}／待細分"


def enrich_profiles(symbols: Iterable[str], path: Path, client=None, force: bool = False, sleep_seconds: float = 0.15) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if path.exists() and not force:
        try:
            existing = pd.read_csv(path, dtype={"symbol": str, "industry": str})
        except Exception:
            existing = pd.DataFrame()

    records = {}
    if not existing.empty:
        existing["symbol"] = existing["symbol"].astype(str).map(clean_symbol)
        for r in existing.to_dict("records"):
            records[clean_symbol(r.get("symbol", ""))] = r

    codes = list(dict.fromkeys(clean_symbol(s) for s in symbols if clean_symbol(s)))

    # 即使 business_master.csv 已經存在，也重新從目前 Fugle ticker 取得
    # symbol/name/industry 的短資料，讓舊版 "00 / nan" 快取可以被修復。
    live_meta: dict[str, dict] = {}
    if client is not None:
        for exchange in ("TWSE", "TPEx"):
            try:
                for row in client.tickers(exchange):
                    code = clean_symbol(row.get("symbol", ""))
                    if code in codes:
                        live_meta[code] = row
            except Exception:
                continue
    fetched: dict[str, tuple[list[str], str]] = {}
    # 目前頁面只對 watchlist/目前研究標的做增量抓取；並行 6 個請求避免啟動等待過久。
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(codes)))) as ex:
        futures = {ex.submit(fetch_chain_paths, symbol): symbol for symbol in codes}
        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                fetched[symbol] = fut.result()
            except Exception:
                fetched[symbol] = ([], TPEX_CHAIN_URL.format(symbol))

    for symbol in codes:
        old = records.get(symbol, {})
        meta = live_meta.get(symbol, {}) or {}
        old_name = str(old.get("name", ""))
        live_name = str(meta.get("name", ""))
        name = live_name if live_name and live_name.lower() != "nan" else (old_name if old_name and old_name.lower() != "nan" else NAME_OVERRIDES.get(symbol, ""))
        raw_industry = meta.get("industry", old.get("industry", "00"))
        industry = INDUSTRY_OVERRIDES.get(symbol, normalize_industry(raw_industry))
        paths, url = fetched.get(symbol, ([], TPEX_CHAIN_URL.format(symbol)))
        override = BUSINESS_OVERRIDES.get(symbol, {})
        if paths:
            # 優先與官方產業碼對得上的鏈路；否則第一條。
            primary = paths[0]
            desired_industry_name = INDUSTRIES.get(industry, "")
            for pth in paths:
                if desired_industry_name and desired_industry_name in pth:
                    primary = pth
                    break
            group = _normalize_group(primary)
        else:
            primary = override.get("primary_chain", old.get("primary_chain", ""))
            group = override.get("business_group", old.get("business_group", "")) or _fallback_group(industry, name)

        if override:
            group = override.get("business_group", group)
            primary = override.get("primary_chain", primary)

        records[symbol] = {
            "symbol": symbol,
            "name": name,
            "industry": industry,
            "industry_name": INDUSTRIES.get(industry, "其他/未分類"),
            "business_group": group,
            "primary_chain": primary,
            "all_chain_paths": " || ".join(paths[:12]),
            "research_keywords": ", ".join(build_business_research_context({"symbol": symbol, "name": name, "industry": industry, "industry_name": INDUSTRIES.get(industry, "其他/未分類"), "business_group": group, "primary_chain": primary, "all_chain_paths": " || ".join(paths[:12])}).get("research_keywords", [])),
            "source_url": url,
            "source_type": "TPEx產業價值鏈（公司個體）" if paths else "fallback/override",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    out = pd.DataFrame(list(records.values()))
    if not out.empty:
        out["symbol"] = out["symbol"].astype(str).map(clean_symbol)
        out.to_csv(path, index=False, encoding="utf-8-sig")
        out = out.sort_values(["industry", "business_group", "symbol"]).reset_index(drop=True)
    return out


def load_business_master(path: Path) -> pd.DataFrame:
    columns = ["symbol", "name", "industry", "industry_name", "business_group", "primary_chain", "all_chain_paths", "source_url", "source_type", "updated_at"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path, dtype={"symbol": str, "industry": str})
        if df.empty:
            return pd.DataFrame(columns=columns)
        df["symbol"] = df["symbol"].astype(str).map(clean_symbol)
        if "industry" not in df.columns:
            df["industry"] = "00"
        df["industry"] = df.apply(
            lambda r: KNOWN_INDUSTRY_CODES.get(
                clean_symbol(r.get("symbol", "")),
                INDUSTRY_OVERRIDES.get(
                    clean_symbol(r.get("symbol", "")),
                    normalize_industry(r.get("industry", "00"))
                )
            ), axis=1
        )
        df["industry_name"] = df["industry"].map(
            lambda x: INDUSTRIES.get(str(x).zfill(2), "其他/未分類")
        )
        return df.drop_duplicates("symbol", keep="last")
    except Exception:
        return pd.DataFrame(columns=columns)


def business_map_for_symbols(symbols: Iterable[str], client=None, path: Path | None = None, force: bool = False) -> dict[str, dict]:
    if path is None:
        raise ValueError("business_master.csv 路徑不可為空")
    codes = list(dict.fromkeys(clean_symbol(s) for s in symbols if clean_symbol(s)))
    df = load_business_master(path)

    # v58 核心修復：舊 business_master 可能把整批股票永久保存成 00 / ETF。
    # 每次載入先套用產業 override；若發現缺公司、錯產業或缺名稱，再做一次增量修復。
    stale = []
    if not df.empty:
        by = df.set_index("symbol", drop=False)
        for code in codes:
            if code not in by.index:
                stale.append(code)
                continue
            ind = normalize_industry(by.loc[code].get("industry", "00"))
            if code in INDUSTRY_OVERRIDES and ind != INDUSTRY_OVERRIDES[code]:
                stale.append(code)
                continue
            name = str(by.loc[code].get("name", ""))
            if not name or name.lower() in {"nan", "none"}:
                stale.append(code)

    if force or stale or df.empty:
        repaired = enrich_profiles(codes, path, client=client, force=False)
        if not repaired.empty:
            df = repaired

    if df.empty:
        return {}

    # 再做一次記憶體層修正，即使網站暫時不可用也不讓已知股票回到 00。
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).map(clean_symbol)
    df["industry"] = df.apply(
        lambda r: KNOWN_INDUSTRY_CODES.get(
            clean_symbol(r.get("symbol", "")),
            INDUSTRY_OVERRIDES.get(
                clean_symbol(r.get("symbol", "")),
                normalize_industry(r.get("industry", "00"))
            )
        ), axis=1
    )
    df["industry_name"] = df["industry"].map(lambda x: INDUSTRIES.get(str(x).zfill(2), "其他/未分類"))

    wanted = df[df["symbol"].isin(codes)]
    return {r["symbol"]: r for r in wanted.to_dict("records")}


def profile_for_symbol(symbol: str, path: Path, client=None, force: bool = False) -> dict:
    symbol = clean_symbol(symbol)
    mapping = business_map_for_symbols([symbol], client=client, path=path, force=force)
    return mapping.get(symbol, {
        "symbol": symbol,
        "industry": "00",
        "industry_name": "其他/未分類",
        "business_group": "主要業務待建立",
        "primary_chain": "",
        "all_chain_paths": "",
        "research_keywords": "",
        "source_url": TPEX_CHAIN_URL.format(symbol),
        "source_type": "unavailable",
    })


def research_context_for_symbol(symbol: str, path: Path) -> dict:
    profile = profile_for_symbol(symbol, path, force=False)
    return build_business_research_context(profile)
