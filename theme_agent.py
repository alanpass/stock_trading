# -*- coding: utf-8 -*-
"""台股「利多題材雷達」：從公司業務、產業鏈、新聞、營收與接單證據
辨識可能帶動產業發展的正向題材。

設計原則：
- 題材不是單純新聞情緒，而是「題材 -> 產業鏈 -> 公司 -> 基本面」的證據鏈。
- 只把已存在的資料做結構化整合；不存在的證據不補猜。
- Qwen3 可做研究摘要，但不能自行創造題材證據。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Iterable
import re

import numpy as np
import pandas as pd
import plotly.express as px


THEME_CATALOG: list[dict[str, Any]] = [
    {
        "theme": "AI伺服器與資料中心擴建",
        "keywords": ["AI伺服器", "資料中心", "AI server", "機櫃", "CSP", "雲端", "推論", "訓練"],
        "business_tokens": ["伺服器", "AI", "電源", "散熱", "PCB", "連接器", "被動元件", "晶圓代工", "ASIC", "儲存"],
        "industries": {"24", "25", "28", "26"},
    },
    {
        "theme": "先進製程與AI ASIC",
        "keywords": ["先進製程", "2奈米", "A16", "A14", "AI ASIC", "ASIC", "SoC", "tape-out", "design win", "HPC"],
        "business_tokens": ["晶圓代工", "ASIC", "SoC", "IC設計", "IC設計服務", "先進製程"],
        "industries": {"24"},
    },
    {
        "theme": "先進封裝、Chiplet與HBM",
        "keywords": ["CoWoS", "CoPoS", "先進封裝", "Chiplet", "HBM", "HBM3e", "HBM4", "異質整合", "FOPLP", "封裝測試"],
        "business_tokens": ["封裝", "測試", "測試介面", "先進封裝", "晶圓級封裝", "記憶體", "ASIC"],
        "industries": {"24"},
    },
    {
        "theme": "記憶體價格與供需改善",
        "keywords": ["DRAM", "DDR5", "DDR4", "HBM", "NOR Flash", "NAND", "記憶體價格", "漲價", "供需", "缺貨"],
        "business_tokens": ["記憶體", "DRAM", "Flash", "NAND", "儲存控制", "記憶體IC"],
        "industries": {"24"},
    },
    {
        "theme": "CCL、高速材料與高階PCB",
        "keywords": ["CCL", "銅箔基板", "高速材料", "高速高頻", "HDI", "ABF", "BT載板", "PCB", "AI伺服器", "高階PCB"],
        "business_tokens": ["PCB", "IC載板", "被動元件", "高速互連", "連接器"],
        "industries": {"28", "24", "25"},
    },
    {
        "theme": "高速網通與800G／1.6T光通訊",
        "keywords": ["800G", "1.6T", "光模組", "光通訊", "高速網路", "交換器", "Wi-Fi 7", "網通"],
        "business_tokens": ["光通訊", "網通", "高速互連", "連接器", "高速傳輸"],
        "industries": {"24", "26", "28", "27"},
    },
    {
        "theme": "AI伺服器電源、BBU與高壓供電",
        "keywords": ["PSU", "BBU", "800V", "資料中心電力", "電源供應器", "備援電力", "高功率電源", "OCP"],
        "business_tokens": ["電源供應", "電力設備", "電源", "電機"],
        "industries": {"25", "05", "06", "35"},
    },
    {
        "theme": "AI／邊緣AI／Physical AI",
        "keywords": ["Edge AI", "邊緣AI", "Physical AI", "機器人", "自駕", "NPU", "AI PC", "AI手機"],
        "business_tokens": ["ASIC", "SoC", "IC設計", "伺服器", "PC", "電機", "自動化", "感測"],
        "industries": {"24", "25", "05", "28"},
    },
    {
        "theme": "高階顯示器、OLED與車用顯示",
        "keywords": ["OLED", "AMOLED", "Mini LED", "Micro LED", "LTPO", "車用顯示", "面板價格", "顯示器"],
        "business_tokens": ["面板", "顯示器", "觸控", "光電"],
        "industries": {"26"},
    },
    {
        "theme": "車用電子與ADAS",
        "keywords": ["ADAS", "車用電子", "電動車", "EV", "智慧座艙", "車規", "Tier 1", "自駕"],
        "business_tokens": ["車用", "汽車零組件", "感測", "電源", "被動元件", "光電"],
        "industries": {"24", "26", "28", "12"},
    },
]

ORDER_TERMS = [
    "接單", "訂單", "在手訂單", "backlog", "design win", "量產", "放量", "出貨", "供貨",
    "擴產", "產能", "稼動率", "CAPEX", "客戶導入", "認證", "漲價", "報價",
]
POSITIVE_TERMS = [
    "成長", "增加", "上升", "創高", "強勁", "回溫", "改善", "復甦", "突破", "擴產", "放量",
    "漲價", "供不應求", "缺貨", "訂單", "接單", "出貨增加", "需求強", "AI需求", "需求成長",
]
NEGATIVE_TERMS = [
    "下滑", "衰退", "下修", "砍單", "需求弱", "庫存", "供過於求", "降價", "成本上升", "取消",
]


@dataclass
class ThemeResult:
    theme: str
    strength: float
    status: str
    affected_symbols: list[str]
    affected_count: int
    affected_companies: list[dict[str, Any]]
    positive_news_count: int
    negative_news_count: int
    official_evidence_count: int
    corroboration_count: int
    order_evidence_count: int
    revenue_support_count: int
    avg_news_relevance: float
    bullish_drivers: list[str]
    supporting_evidence: list[dict[str, Any]]
    contradiction_flags: list[str]
    research_queries: list[str]
    source_urls: list[str]


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _contains(text: str, token: str) -> bool:
    return _norm(token) in _norm(text)


def _theme_profile_match(profile: dict[str, Any], theme: dict[str, Any]) -> bool:
    blob = " ".join([
        profile.get("business_group", ""),
        profile.get("primary_chain", ""),
        profile.get("all_chain_paths", ""),
        " ".join(profile.get("research_keywords", []) or []),
    ])
    if str(profile.get("industry", "00")).zfill(2) in theme.get("industries", set()):
        for t in theme.get("business_tokens", []):
            if _contains(blob, t):
                return True
    return False


def _news_text(row: dict[str, Any]) -> str:
    return f"{row.get('title','')} {row.get('summary','')} {row.get('query','')}"


def _news_matches_theme(row: dict[str, Any], theme: dict[str, Any]) -> bool:
    text = _news_text(row)
    return any(_contains(text, k) for k in theme.get("keywords", []))


def _score_theme(
    *, affected_count: int, positive_ratio: float, official: int, corroboration: int,
    order_evidence: int, revenue_support: int, contradiction: int,
) -> float:
    breadth = min(1.0, affected_count / 5.0)
    official_score = min(1.0, official / 3.0)
    corroboration_score = min(1.0, corroboration / 6.0)
    order_score = min(1.0, order_evidence / 4.0)
    revenue_score = min(1.0, revenue_support / 4.0)
    contradiction_penalty = min(0.4, contradiction * 0.08)
    raw = (
        0.24 * breadth
        + 0.24 * positive_ratio
        + 0.14 * official_score
        + 0.14 * corroboration_score
        + 0.12 * order_score
        + 0.12 * revenue_score
        - contradiction_penalty
    )
    return float(np.clip(raw * 100.0, 0.0, 100.0))


def _status(score: float, positive_ratio: float, contradiction: int, evidence: int) -> str:
    if evidence == 0:
        return "無足夠證據"
    if contradiction >= 3 and score < 60:
        return "題材存在但基本面待驗證"
    if score >= 72 and positive_ratio >= 0.60:
        return "強勢成長題材"
    if score >= 52 and positive_ratio >= 0.50:
        return "成長題材"
    if score >= 35:
        return "初步題材"
    return "題材證據偏弱"


def discover_bullish_themes(
    *,
    business_profiles: dict[str, dict[str, Any]],
    news_rows: Iterable[dict[str, Any]],
    financial_snapshots: Iterable[dict[str, Any]],
    order_evidence: Iterable[dict[str, Any]],
    source_urls: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """用現有研究資料找出「正在形成、且有基本面證據」的利多題材。"""
    news_rows = list(news_rows or [])
    financial_snapshots = list(financial_snapshots or [])
    order_evidence = list(order_evidence or [])
    source_urls = source_urls or {}
    results: list[dict[str, Any]] = []

    for theme in THEME_CATALOG:
        impacted = [
            sym for sym, profile in business_profiles.items()
            if _theme_profile_match(profile, theme)
        ]
        if not impacted:
            continue

        matched = [r for r in news_rows if _news_matches_theme(r, theme)]
        if not matched and not order_evidence:
            continue

        positive = [
            r for r in matched
            if any(_contains(_news_text(r), t) for t in POSITIVE_TERMS)
            and not any(_contains(_news_text(r), t) for t in NEGATIVE_TERMS)
        ]
        negative = [r for r in matched if any(_contains(_news_text(r), t) for t in NEGATIVE_TERMS)]
        official = sum(1 for r in matched if bool(r.get("official_match")))
        corroboration = sum(int(r.get("corroboration", 0) or 0) for r in matched)
        order_hits = [r for r in order_evidence if _news_matches_theme(r, theme)]

        fin_by_symbol = {str(x.get("symbol")): x for x in financial_snapshots if x.get("symbol")}
        revenue_support = 0
        for sym in impacted:
            row = fin_by_symbol.get(sym, {})
            yoy = row.get("revenue_yoy")
            try:
                if yoy is not None and np.isfinite(float(yoy)) and float(yoy) > 0:
                    revenue_support += 1
            except Exception:
                pass

        contradiction = 0
        if negative:
            contradiction += 1
        if positive and revenue_support == 0:
            contradiction += 1
        if matched and official == 0:
            contradiction += 1

        pos_ratio = len(positive) / max(1, len(matched))
        strength = _score_theme(
            affected_count=len(impacted),
            positive_ratio=pos_ratio,
            official=official,
            corroboration=corroboration,
            order_evidence=len(order_hits),
            revenue_support=revenue_support,
            contradiction=contradiction,
        )

        drivers = []
        if len(impacted) >= 2:
            drivers.append(f"{len(impacted)} 家研究標的業務與題材直接相關")
        if order_hits:
            drivers.append(f"發現 {len(order_hits)} 筆接單／出貨／擴產證據")
        if revenue_support:
            drivers.append(f"{revenue_support} 家相關公司最新營收呈正成長")
        if official:
            drivers.append(f"{official} 筆新聞可對應官方／原始資料")
        if corroboration:
            drivers.append(f"跨來源佐證累計 {corroboration} 次")

        contradictions = []
        if negative:
            contradictions.append("同題材同時存在負面新聞，需區分需求成長與成本/估值風險")
        if matched and official == 0:
            contradictions.append("目前未找到直接官方公告佐證")
        if positive and revenue_support == 0:
            contradictions.append("利多敘事尚未在自選股財務資料中形成明顯營收支撐")

        queries = [
            f'"{theme["theme"]}" 台灣 接單 營收 供應鏈',
            f'"{theme["theme"]}" 訂單 出貨 產能 報價',
        ]
        for sym in impacted[:5]:
            p = business_profiles.get(sym, {})
            name = p.get("name") or sym
            queries.append(f'"{sym}" {name} "{theme["theme"]}"')

        evidence = []
        for row in sorted(matched, key=lambda r: (bool(r.get("official_match")), int(r.get("corroboration", 0) or 0)), reverse=True)[:8]:
            evidence.append({
                "title": row.get("title"),
                "source": row.get("source"),
                "published": row.get("published"),
                "sentiment_score": row.get("sentiment_score", 0),
                "business_relevance": row.get("business_relevance", 0),
                "official_match": row.get("official_match", False),
                "corroboration": row.get("corroboration", 0),
                "verification": row.get("verification", "待查證"),
                "link": row.get("link"),
            })

        affected_companies = []
        for sym in impacted:
            p = business_profiles.get(sym, {}) or {}
            affected_companies.append({
                "symbol": sym,
                "name": p.get("name") or sym,
                "business_group": p.get("business_group", ""),
                "primary_chain": p.get("primary_chain", ""),
            })

        results.append(asdict(ThemeResult(
            theme=theme["theme"],
            strength=strength,
            status=_status(strength, pos_ratio, contradiction, len(matched) + len(order_hits)),
            affected_symbols=impacted,
            affected_count=len(impacted),
            affected_companies=affected_companies,
            positive_news_count=len(positive),
            negative_news_count=len(negative),
            official_evidence_count=official,
            corroboration_count=corroboration,
            order_evidence_count=len(order_hits),
            revenue_support_count=revenue_support,
            avg_news_relevance=float(np.mean([float(r.get("business_relevance", 0) or 0) for r in matched])) if matched else 0.0,
            bullish_drivers=drivers,
            supporting_evidence=evidence,
            contradiction_flags=contradictions,
            research_queries=list(dict.fromkeys(queries))[:8],
            source_urls=list(dict.fromkeys([u for u in source_urls.values() if u]))[:8],
        )))

    results.sort(key=lambda x: float(x.get("strength", 0)), reverse=True)
    return results


def theme_sankey_frame(themes: list[dict[str, Any]], business_profiles: dict[str, dict[str, Any]], top_n: int = 8) -> pd.DataFrame:
    rows = []
    for t in themes[:top_n]:
        for sym in t.get("affected_symbols", [])[:12]:
            p = business_profiles.get(sym, {})
            rows.append({
                "題材": t.get("theme"),
                "公司": f"{sym} {p.get('name','')}".strip(),
                "強度": float(t.get("strength", 0)),
                "業務": p.get("business_group", ""),
            })
    return pd.DataFrame(rows)
