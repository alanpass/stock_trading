# -*- coding: utf-8 -*-
"""公司主要業務／產業鏈細分主檔。

設計原則：
1. 第一層仍使用 TWSE/TPEx 的官方產業代碼（24/25/26/...）。
2. 第二層使用 TPEx「產業價值鏈資訊平台」的全市場分類（value_chain.py）取得「產業 > 子鏈」。
3. 若官方價值鏈尚無資料，只顯示官方產業，不另外猜測。
4. 不把 LLM 猜測當作正式分類；LLM 只能在研究報告中解釋。

主檔：data/business_master.csv
欄位：symbol,name,industry,industry_name,business_group,primary_chain,all_chain_paths,source_url,updated_at
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

import value_chain

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
    # 橫跨多條產業鏈的公司：價值鏈分類列出十幾條，這裡標明主要業務
    "1303": {"business_group": "塑化／銅箔基板材料", "primary_chain": "印刷電路板 > 銅箔基板"},
    "2317": {"business_group": "電子代工／AI伺服器", "primary_chain": "電腦及週邊設備 > 伺服器"},
    "2382": {"business_group": "伺服器／AI運算", "primary_chain": "電腦及週邊設備 > 伺服器"},
    "2376": {"business_group": "主機板／伺服器", "primary_chain": "電腦及週邊設備 > 主機板"},
    "4938": {"business_group": "電子代工／電腦整機", "primary_chain": "電腦及週邊設備 > 筆記型電腦"},
    "2301": {"business_group": "電源供應／光電零組件", "primary_chain": "電腦及週邊設備 > 電源供應器"},
    "2352": {"business_group": "電腦周邊／顯示器", "primary_chain": "電腦及週邊設備 > 顯示器模組"},
    "2360": {"business_group": "半導體設備／測試", "primary_chain": "半導體 > 生產製程及檢測設備"},
    "3653": {"business_group": "散熱／均熱片", "primary_chain": "半導體 > 導線架"},
    "3105": {"business_group": "晶圓代工／化合物半導體", "primary_chain": "半導體 > IC/晶圓製造 > 晶圓製造"},
    "3532": {"business_group": "半導體材料／矽晶圓", "primary_chain": "半導體 > IC/晶圓製造 > 晶圓製造"},
    "5483": {"business_group": "半導體材料／矽晶圓", "primary_chain": "半導體 > IC/晶圓製造 > 晶圓製造"},
    "6182": {"business_group": "半導體材料／矽晶圓", "primary_chain": "半導體 > IC/晶圓製造 > 晶圓製造"},
    "6196": {"business_group": "半導體設備／廠務工程", "primary_chain": "其他 > 其他電子產品及電子服務產業"},
    "6691": {"business_group": "半導體設備／廠務工程", "primary_chain": "其他 > 其他電子產品及電子服務產業"},
    "2454": {"business_group": "IC設計／手機SoC", "primary_chain": "半導體 > IC設計"},
    "3081": {"business_group": "光通訊／光元件", "primary_chain": "通信網路 > 主/被動元件"},
    "3234": {"business_group": "光通訊／光元件", "primary_chain": "通信網路 > 主/被動元件"},
    "4979": {"business_group": "光通訊／光元件", "primary_chain": "通信網路 > 主/被動元件"},
}

# 一檔股票屬於多條產業鏈時，優先選官方產業對應的鏈；都對不上時避開跨產業的應用主題。
OFFICIAL_CHAIN_INDUSTRIES = {
    "01": ("水泥",), "02": ("食品", "食品生技"), "03": ("石化及塑橡膠",), "04": ("紡織",),
    "05": ("電機機械", "自動化"), "10": ("鋼鐵",), "11": ("石化及塑橡膠",), "12": ("汽車", "電動車輛產業"),
    "14": ("建材營造",), "15": ("交通運輸及航運",), "17": ("金融",),
    "22": ("製藥", "醫療器材", "食品生技", "再生醫療"), "23": ("油電燃氣",), "24": ("半導體",),
    "25": ("電腦及週邊設備",), "26": ("平面顯示器", "觸控面板", "LED照明產業", "太陽能產業"),
    "27": ("通信網路",), "28": ("印刷電路板", "被動元件", "連接器", "電腦及週邊設備"),
    "30": ("軟體服務",), "32": ("文化創意業",),
}
THEME_CHAIN_INDUSTRIES = {
    "電動車輛產業", "能源元件", "智慧電網", "區塊鏈", "金融科技", "人工智慧", "雲端運算", "資通訊安全",
    "大數據", "體驗科技", "運動科技", "太空衛星科技",
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
    ("半導體 > 基板", "PCB／IC載板"),
    ("通信網路 > 主/被動元件", "通訊元件／主被動元件"),
    ("銅箔基板", "CCL／銅箔基板"),
    ("玻璃纖維", "玻纖布／玻纖紗"),
    ("印刷電路板 > 銅箔", "銅箔"),
    ("IC模組", "記憶體模組"),
    ("晶圓製造", "晶圓代工／晶圓製造"),
    ("IC封裝測試", "封裝測試／先進封裝"),
    ("半導體 > 生產製程及檢測設備", "半導體設備／測試"),
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
    # 子鏈名稱常帶很長的括號說明，例如「光通訊設備(如光纖電纜、光傳輸設備)」
    return re.split(r"[(（]", s.split(" > ")[-1])[0].strip() or "主要業務待建立"


def chain_paths_by_symbol() -> dict[str, list[str]]:
    """全市場分類檔換成 {股票: [產業 > 子鏈 ...]}；TPEx 的公司個體頁面已抓不到鏈路，改用這份。"""
    out: dict[str, list[str]] = {}
    for leaf in value_chain.leaf_groups(value_chain.load()):
        for code in leaf["members"]:
            paths = out.setdefault(code, [])
            if leaf["path"] not in paths:
                paths.append(leaf["path"])
    return out


def _pick_primary_chain(paths: list[str], industry: str) -> str:
    for chain_industry in OFFICIAL_CHAIN_INDUSTRIES.get(str(industry).zfill(2), ()):
        for path in paths:
            if path.startswith(f"{chain_industry} > "):
                return path
    for path in paths:
        if path.split(" > ")[0] not in THEME_CHAIN_INDUSTRIES:
            return path
    return paths[0]


def _official_industries(path: Path) -> dict[str, dict]:
    """全市場產業主檔（industry_master.csv）：TPEx 沒列的股票也有官方產業與名稱。"""
    try:
        df = pd.read_csv(path.parent / "industry_master.csv", dtype=str)
        df["symbol"] = df["symbol"].map(clean_symbol)
        return {r["symbol"]: r for r in df.to_dict("records")}
    except Exception:
        return {}


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
    # 查不到子鏈就只顯示官方產業
    code = str(industry).zfill(2)
    return INDUSTRIES.get(code, "其他") if code != "00" else "其他／未分類"


def _is_bad_etf_group(symbol: str, group: str) -> bool:
    """不是 ETF（代號 00 開頭）卻被存成 ETF 分類。"""
    return group.startswith("ETF/無產業分類") and not str(symbol).startswith("00")


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

    # 舊版 "00 / nan" 快取先用全市場產業主檔修復，主檔也沒有的股票才查 Fugle ticker。
    official = _official_industries(path)
    live_meta: dict[str, dict] = {c: official[c] for c in codes if c in official}
    if client is not None and any(c not in official for c in codes):
        for exchange in ("TWSE", "TPEx"):
            try:
                for row in client.tickers(exchange):
                    code = clean_symbol(row.get("symbol", ""))
                    if code in codes and code not in live_meta:
                        live_meta[code] = row
            except Exception:
                continue
    chains = chain_paths_by_symbol()

    for symbol in codes:
        old = records.get(symbol, {})
        meta = live_meta.get(symbol, {}) or {}
        old_name = str(old.get("name", ""))
        live_name = str(meta.get("name", ""))
        name = live_name if live_name and live_name.lower() != "nan" else (old_name if old_name and old_name.lower() != "nan" else NAME_OVERRIDES.get(symbol, ""))
        raw_industry = meta.get("industry", old.get("industry", "00"))
        industry = INDUSTRY_OVERRIDES.get(symbol, normalize_industry(raw_industry))
        paths, url = chains.get(symbol, []), TPEX_CHAIN_URL.format(symbol)
        override = BUSINESS_OVERRIDES.get(symbol, {})
        if paths:
            primary = _pick_primary_chain(paths, industry)
            group = _normalize_group(primary)
        else:
            primary = override.get("primary_chain", old.get("primary_chain", ""))
            old_group = str(old.get("business_group", "") or "")
            if _is_bad_etf_group(symbol, old_group) or old_group.lower() == "nan" or "待細分" in old_group:
                old_group = ""   # 舊版把查不到產業的股票存成 ETF 或待細分，不沿用
            etf_group = "ETF／指數投資" if symbol.startswith("00") else ""
            group = override.get("business_group", old_group) or etf_group or _fallback_group(industry, name)

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
            group = str(by.loc[code].get("business_group", ""))
            if _is_bad_etf_group(code, group) or "待細分" in group or (ind == "00" and not code.startswith("00")):
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
    # 先篩出要的股票再逐列修正，不對整份全市場主檔做 apply。
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).map(clean_symbol)
    df = df[df["symbol"].isin(codes)].copy()
    if df.empty:
        return {}
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

    return {r["symbol"]: r for r in df.to_dict("records")}


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

