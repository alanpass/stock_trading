# -*- coding: utf-8 -*-
"""AI 台股盤後研究排程入口 v3

這個版本專門修正「研究有跑、但 Email 變成全部 0 筆」的資料鏈問題。

核心流程：
1. ResearchAgent 先建立原始研究包。
2. 立刻做一次「官方資料 recovery」：TWSE / TPEx 公告、月營收、財報。
3. 盤後行情如果缺失，重新向官方日行情端點取得；不把舊日資料冒充今天。
4. CNYES 不重新爬網站，優先讀取今日早報已成功建立的夜間快取。
5. Recovery 完成後，重新交給 Qwen3 8B ResearchOrchestratorAgent。
   這一步非常重要：Agent 必須看到「真正恢復後的市場、財報、新聞資料」再做摘要。
6. 覆寫同一天 research JSON / Markdown，再由 EmailAgent 排版寄送。

EmailAgent 只負責呈現與寄送；研究判讀由 ResearchAgent / Qwen3 Agent 負責。
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests

from business_master import business_map_for_symbols, build_business_research_context
from email_agent import EmailAgent
from market_calendar import is_twse_trading_day, market_status_text
from market_mover_analysis import MarketMoverAnalyzer
from research_agent import ResearchAgent
from stock_api import INDUSTRIES, INDUSTRY_OVERRIDES, clean_symbol

try:
    from agent_research_assistant import ResearchOrchestratorAgent
except Exception:
    ResearchOrchestratorAgent = None

try:
    from cnyes_news_crawler import CnyesNewsCrawler
except Exception:
    CnyesNewsCrawler = None

BASE = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE / "user_watchlist.json"
REPORT_DIR = BASE / "output" / "research_reports"
LOG_DIR = BASE / "output" / "scheduler_logs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WATCHLIST = [
    "3481", "2327", "2492", "3037", "3044", "3533", "2303", "2330", "2344", "2408",
    "6515", "7769", "6488", "3374", "2377", "3450", "0050", "3006", "3661", "8299",
    "2308", "6274",
]

AFTER_CLOSE_HOUR = int(os.getenv("AFTER_CLOSE_HOUR", "14"))
AFTER_CLOSE_MINUTE = int(os.getenv("AFTER_CLOSE_MINUTE", "30"))
TAIPEI_TZ = "Asia/Taipei"

TWSE_OPENAPI = "https://openapi.twse.com.tw/v1"
TPEX_OPENAPI = "https://www.tpex.org.tw/openapi/v1"
TWSE_ANN_URL = f"{TWSE_OPENAPI}/opendata/t187ap04_L"
TWSE_REV_URL = f"{TWSE_OPENAPI}/opendata/t187ap05_L"
TWSE_INCOME_URL = f"{TWSE_OPENAPI}/opendata/t187ap06_L_ci"
TPEX_ANN_URL = f"{TPEX_OPENAPI}/mopsfin_t187ap04_O"
TPEX_REV_URL = f"{TPEX_OPENAPI}/mopsfin_t187ap05_O"
TPEX_INCOME_URL = f"{TPEX_OPENAPI}/mopsfin_t187ap06_O_ci"
TWSE_DAILY_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_DAILY_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"


# ============================================================================
# 基本工具
# ============================================================================
def load_watchlist() -> list[str]:
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                values = [clean_symbol(x) for x in data if clean_symbol(x)]
                if values:
                    return list(dict.fromkeys(values))
    except Exception:
        pass
    return DEFAULT_WATCHLIST.copy()


def now_taipei() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TAIPEI_TZ)


def report_log_path(report_date: str) -> Path:
    return LOG_DIR / f"after_close_research_{report_date or 'unknown'}.log"


def write_log(report_date: str, message: str) -> None:
    line = f"[{now_taipei().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    try:
        with report_log_path(report_date).open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def write_status(report_date: str, status: str, **extra: Any) -> None:
    payload = {
        "report_date": report_date,
        "status": status,
        "updated_at": now_taipei().isoformat(),
        **extra,
    }
    try:
        (LOG_DIR / f"after_close_status_{report_date or 'unknown'}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        if text in {"", "-", "--", "None", "none", "nan", "NaN", "null"}:
            return default
        x = float(text)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _date_text(value: Any) -> str:
    """可處理民國 YYYYMMDD / YYYYMM / 西元日期。"""
    text = str(value or "").strip().replace("/", "-").replace(".", "-")
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    m = re.fullmatch(r"(\d{3})-(\d{1,2})-(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)) + 1911:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.fullmatch(r"(\d{3})(\d{2})(\d{2})", text)
    if m:
        return f"{int(m.group(1)) + 1911:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.fullmatch(r"(\d{3})(\d{2})", text)
    if m:
        return f"{int(m.group(1)) + 1911:04d}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"(\d{4})(\d{2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _row_symbol(row: dict[str, Any]) -> str:
    for key in row:
        k = str(key).strip().lower()
        if k in {"公司代號", "證券代號", "股票代號", "公司代碼", "code", "symbol", "securitiescompanycode"}:
            value = clean_symbol(row.get(key, ""))
            if value:
                return value
    return ""


def _row_name(row: dict[str, Any]) -> str:
    for key in row:
        k = str(key).strip().lower()
        if k in {"公司名稱", "證券名稱", "股票名稱", "name", "companyname"}:
            value = str(row.get(key) or "").strip()
            if value and value.lower() not in {"nan", "none"}:
                return value
    return ""


def _field(row: dict[str, Any], keywords: list[str]) -> Any:
    norm_keys = [(str(k).replace(" ", "").strip().lower(), k) for k in row]
    wanted = [str(x).replace(" ", "").strip().lower() for x in keywords]
    # 先找完全包含關鍵字的欄位，避免 Revenue 與其他欄位誤配。
    for key_norm, key_raw in norm_keys:
        for w in wanted:
            if w and w in key_norm:
                return row.get(key_raw)
    return None


def _field_float(row: dict[str, Any], keywords: list[str]) -> float | None:
    return _safe_float(_field(row, keywords), None)


# ============================================================================
# 官方 OpenAPI recovery
# ============================================================================
def _fetch_json(session: requests.Session, url: str, label: str, attempts: int = 3) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0 TaiwanStockResearchAssistant/3.0",
                    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return [x for x in payload if isinstance(x, dict)], errors
            if isinstance(payload, dict):
                for key in ("data", "rows", "result", "records"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        return [x for x in value if isinstance(x, dict)], errors
            errors.append(f"{label}: 回傳格式不是列表")
        except Exception as exc:
            errors.append(f"{label}: 第 {attempt} 次失敗 {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(attempt)
    return [], errors


def _match_rows(rows: list[dict[str, Any]], symbols: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
    for row in rows:
        symbol = _row_symbol(row)
        if symbol in out:
            out[symbol].append(row)
    return out


def _latest_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    scored = []
    for idx, row in enumerate(rows):
        period = (
            _field(row, ["資料年月", "資料日期", "年月", "年度", "DataPeriod", "YearMonth", "Date"])
        )
        period_text = _date_text(period)
        scored.append((period_text or "", idx, row))
    scored.sort(key=lambda x: (x[0], x[1]))
    return dict(scored[-1][2])


def _official_financial_recovery(symbols: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = requests.Session()
    symbol_set = set(symbols)
    all_errors: list[str] = []

    specs = [
        ("TWSE公告", TWSE_ANN_URL),
        ("TPEx公告", TPEX_ANN_URL),
        ("TWSE營收", TWSE_REV_URL),
        ("TPEx營收", TPEX_REV_URL),
        ("TWSE財報", TWSE_INCOME_URL),
        ("TPEx財報", TPEX_INCOME_URL),
    ]
    fetched: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for label, url in specs:
        rows, errors = _fetch_json(session, url, label)
        fetched[label] = rows
        counts[label] = len(rows)
        all_errors.extend(errors)

    twse_rev_map = _match_rows(fetched["TWSE營收"], symbol_set)
    tpex_rev_map = _match_rows(fetched["TPEx營收"], symbol_set)
    twse_inc_map = _match_rows(fetched["TWSE財報"], symbol_set)
    tpex_inc_map = _match_rows(fetched["TPEx財報"], symbol_set)

    profiles_raw = business_map_for_symbols(
        symbols,
        path=BASE / "data" / "business_master.csv",
        force=False,
    )

    snapshots: list[dict[str, Any]] = []
    for symbol in symbols:
        rev_rows = twse_rev_map.get(symbol, []) or tpex_rev_map.get(symbol, [])
        inc_rows = twse_inc_map.get(symbol, []) or tpex_inc_map.get(symbol, [])
        rev = _latest_row(rev_rows)
        inc = _latest_row(inc_rows)
        profile = profiles_raw.get(symbol, {}) or {}
        name = _row_name(rev) or _row_name(inc) or str(profile.get("name") or symbol)

        revenue = _field_float(inc, ["營業收入淨額", "營業收入", "OperatingRevenue", "Revenue"])
        gross_profit = _field_float(inc, ["營業毛利", "營業毛利(損)", "GrossProfit"])
        op_profit = _field_float(inc, ["營業利益", "營業利益(損失)", "OperatingIncome"])
        net_income = _field_float(inc, ["本期淨利", "本期淨利(淨損)", "NetIncome"])
        eps = _field_float(inc, ["基本每股盈餘", "基本每股盈餘(元)", "EPS"])
        latest_month_revenue = _field_float(rev, ["營業收入-當月營收", "當月營收", "營業收入", "Revenue"])
        revenue_yoy = _field_float(rev, ["營業收入-去年同月增減(%)", "去年同月增減", "年增率", "YoY"])
        rev_period = _date_text(_field(rev, ["資料年月", "資料日期", "年月", "DataPeriod", "YearMonth"]))
        inc_period = _date_text(_field(inc, ["資料年月", "資料日期", "年月", "年度", "DataPeriod", "YearMonth", "Date"]))

        market = "TWSE" if twse_rev_map.get(symbol) or twse_inc_map.get(symbol) else "TPEx" if tpex_rev_map.get(symbol) or tpex_inc_map.get(symbol) else ""
        pctx = build_business_research_context(profile) if profile else {}
        industry_code = (
            INDUSTRY_OVERRIDES.get(symbol)
            or str(profile.get("industry_code") or profile.get("industry") or "00").zfill(2)
        )
        industry_name = INDUSTRIES.get(str(industry_code).zfill(2), profile.get("industry_name") or "其他")

        if not any(v is not None for v in [latest_month_revenue, revenue_yoy, eps, revenue, op_profit, net_income]):
            continue

        snapshots.append({
            "symbol": symbol,
            "name": name,
            "industry_code": str(industry_code).zfill(2),
            "industry_name": industry_name,
            "market": market,
            "revenue_period": rev_period,
            "income_period": inc_period,
            "statement_revenue": revenue,
            "gross_profit": gross_profit,
            "operating_profit": op_profit,
            "net_income": net_income,
            "eps": eps,
            "latest_month_revenue": latest_month_revenue,
            "revenue_yoy": revenue_yoy,
            "business_group": pctx.get("business_group") or profile.get("business_group") or "",
            "primary_chain": pctx.get("primary_chain") or profile.get("primary_chain") or "",
            "raw_financial": inc,
            "raw_revenue": rev,
            "official_data_verified": True,
        })

    official_sources = {
        "twse_announcements_rows": counts["TWSE公告"],
        "tpex_announcements_rows": counts["TPEx公告"],
        "twse_revenue_rows": counts["TWSE營收"],
        "tpex_revenue_rows": counts["TPEx營收"],
        "twse_income_rows": counts["TWSE財報"],
        "tpex_income_rows": counts["TPEx財報"],
    }

    # 將與自選股直接相關的官方公告保存成精簡證據，讓後續 Agent 有可用原始訊號。
    announcement_items: list[dict[str, Any]] = []
    for label in ("TWSE公告", "TPEx公告"):
        for row in fetched[label]:
            symbol = _row_symbol(row)
            if symbol not in symbol_set:
                continue
            date = _date_text(_field(row, ["發佈日期", "發布日期", "日期", "Date", "時間"]))
            title = str(_field(row, ["主旨", "標題", "Subject", "Title", "事由"]) or "").strip()
            detail = str(_field(row, ["內容", "說明", "摘要", "Description"]) or "").strip()
            if not title and not detail:
                # 不把整列所有欄位塞給 Agent，避免 context 過大。
                text_parts = [str(v).strip() for v in row.values() if str(v).strip()]
                detail = "；".join(text_parts[:6])
            announcement_items.append({
                "symbol": symbol,
                "name": _row_name(row) or symbol,
                "date": date,
                "title": title[:300],
                "detail": detail[:700],
                "source": label,
            })

    announcement_items.sort(key=lambda x: str(x.get("date", "")), reverse=True)
    diagnostics = {
        "official_sources": official_sources,
        "matched_financial_snapshots": len(snapshots),
        "matched_symbols": sorted({x["symbol"] for x in snapshots}),
        "official_announcement_evidence": announcement_items[:80],
        "errors": all_errors,
    }
    return snapshots, diagnostics


# ============================================================================
# CNYES cache recovery
# ============================================================================
def _load_cnyes_from_morning(report_date: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = REPORT_DIR / f"morning_{report_date}.json"
    if not path.exists():
        return {}, {}, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}, {}, ""
        corpus = payload.get("cnyes_news", {}) or {}
        digest = payload.get("cnyes_research_digest", {}) or {}
        if isinstance(corpus, dict) and int(corpus.get("article_count", 0) or 0) > 0:
            return corpus, digest if isinstance(digest, dict) else {}, str(path)
    except Exception:
        pass
    return {}, {}, ""


def _load_cnyes_cache(report_date: str) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    corpus, digest, path = _load_cnyes_from_morning(report_date)
    if corpus:
        return corpus, digest, path, []

    errors: list[str] = []
    if CnyesNewsCrawler is None:
        return {}, {}, "", ["CnyesNewsCrawler 模組不可用"]
    try:
        crawler = CnyesNewsCrawler(
            BASE,
            max_articles=int(os.getenv("CNYES_NIGHTLY_MAX_ARTICLES", "600")),
            max_scrolls=int(os.getenv("CNYES_NEWS_MAX_SCROLLS", "18")),
        )
        corpus = crawler.load_recent_cached(
            before_date=report_date,
            days=int(os.getenv("CNYES_RESEARCH_LOOKBACK_DAYS", "2")),
            max_search_days=int(os.getenv("CNYES_RESEARCH_CACHE_SEARCH_DAYS", "7")),
        )
        if isinstance(corpus, dict) and corpus.get("articles"):
            return corpus, {}, "cached_nightly_files", errors
        errors.append("找不到可用 CNYES 夜間日檔")
    except Exception as exc:
        errors.append(f"讀取 CNYES 夜間快取失敗：{type(exc).__name__}: {exc}")
    return {}, {}, "", errors


# ============================================================================
# Market recovery
# ============================================================================
def _market_recovery(base: Path, watchlist: list[str], report_date: str) -> tuple[dict[str, Any], dict[str, Any]]:
    analyzer = MarketMoverAnalyzer(base)
    try:
        direct = analyzer.analyze(watchlist_symbols=watchlist, max_candidates=40)
        if isinstance(direct, dict) and direct.get("movers") and str(direct.get("data_asof", ""))[:10] == report_date:
            return direct, {"method": "MarketMoverAnalyzer_after_fix", "errors": direct.get("errors", [])}
    except Exception as exc:
        direct_error = f"MarketMoverAnalyzer recovery 失敗：{type(exc).__name__}: {exc}"
    else:
        direct_error = "MarketMoverAnalyzer 未取得今日 movers"

    # 第二層：直接打官方日行情 API，避免 MarketMoverAnalyzer 的內部 cache 影響。
    session = requests.Session()
    rows_all: list[dict[str, Any]] = []
    errors: list[str] = [direct_error]
    today_is_trading = is_twse_trading_day(pd.Timestamp(report_date, tz=TAIPEI_TZ), base)
    sources = [("TWSE", TWSE_DAILY_URL), ("TPEx", TPEX_DAILY_URL)]
    for label, url in sources:
        rows, fetch_errors = _fetch_json(session, url, label, attempts=3)
        errors.extend(fetch_errors)
        for row in rows:
            sym = _row_symbol(row)
            close = _field_float(row, ["ClosingPrice", "收盤價", "Close"])
            change = _field_float(row, ["Change", "漲跌價差", "漲跌", "PriceChange"])
            if not sym or len(sym) < 4 or close is None:
                continue
            rows_all.append({
                "symbol": sym,
                "name": _row_name(row) or sym,
                "close": close,
                "change": change or 0.0,
                "source": label,
                "source_date": _date_text(_field(row, ["Date", "date", "資料日期"])),
                "industry": "00",
            })

    if not rows_all:
        return {"generated_at": now_taipei().isoformat(), "movers": [], "industry_summary": [], "errors": errors}, {"errors": errors}

    # TWSE STOCK_DAY_ALL 是最新交易日全市場快照，某些回應沒有 Date。
    # 只有交易日才允許用當天日期做錨定；休市日直接停止。
    for row in rows_all:
        if not row["source_date"] and today_is_trading:
            row["source_date"] = report_date
    if not today_is_trading:
        return {
            "generated_at": now_taipei().isoformat(),
            "data_asof": "",
            "is_trading_day": False,
            "data_source": "TWSE/TPEx官方日行情",
            "movers": [],
            "industry_summary": [],
            "errors": errors,
            "note": "今天非交易日，未將最近交易日資料冒充今日盤後行情。",
        }, {"errors": errors}

    rows_all = [x for x in rows_all if str(x.get("source_date", ""))[:10] == report_date]
    if not rows_all:
        errors.append(f"官方行情沒有可驗證的 {report_date} 資料")
        return {"generated_at": now_taipei().isoformat(), "movers": [], "industry_summary": [], "errors": errors}, {"errors": errors}

    df = pd.DataFrame(rows_all).drop_duplicates("symbol", keep="first")
    df["previous_close"] = df["close"] - df["change"]
    df["change_percent"] = np.where(
        df["previous_close"].abs() > 1e-12,
        df["change"] / df["previous_close"] * 100.0,
        0.0,
    )
    df = analyzer._enrich_industries(df)
    profiles = business_map_for_symbols(
        df["symbol"].astype(str).tolist(),
        path=base / "data" / "business_master.csv",
        force=False,
    )
    watch = set(clean_symbol(x) for x in watchlist)
    up = df.sort_values("change_percent", ascending=False).head(20)
    down = df.sort_values("change_percent", ascending=True).head(20)
    extra = df[df["symbol"].isin(watch)]
    candidates = pd.concat([up, down, extra], ignore_index=True).drop_duplicates("symbol")

    records: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        code = clean_symbol(row["symbol"])
        profile = profiles.get(code, {}) or {}
        try:
            hist = analyzer._history(code)
            r5, r20 = analyzer._returns(hist)
        except Exception:
            r5, r20 = None, None
        direction = "上漲" if float(row["change_percent"]) > 0 else "下跌" if float(row["change_percent"]) < 0 else "平盤"
        business = str(profile.get("business_group") or profile.get("primary_chain") or row.get("industry_name") or "").strip()
        records.append({
            "symbol": code,
            "name": str(row.get("name", code)),
            "industry_code": str(row.get("industry", "00")).zfill(2),
            "industry_name": str(row.get("industry_name", INDUSTRIES.get(str(row.get("industry", "00")).zfill(2), "其他"))),
            "business_group": business,
            "primary_chain": str(profile.get("primary_chain") or ""),
            "today_change_percent": round(float(row["change_percent"]), 3),
            "ret_5d": None if r5 is None else round(r5 * 100, 3),
            "ret_20d": None if r20 is None else round(r20 * 100, 3),
            "direction": direction,
            # Email 不再把「新聞不足」當成公司下跌原因；這裡只記錄為 Agent 查證入口。
            "reason": "待 Agent 根據新聞與公告查證",
            "reason_type": "待查證",
            "reason_confidence": 0.0,
            "news": [],
            "selection_basis": "官方全市場當日漲跌前段＋自選股補充",
            "market_data_source": "TWSE/TPEx官方日行情",
            "market_data_asof": report_date,
        })

    mdf = pd.DataFrame(records)
    industry_rows: list[dict[str, Any]] = []
    if not mdf.empty:
        for (icode, iname), g in mdf.groupby(["industry_code", "industry_name"], dropna=False):
            industry_rows.append({
                "industry_code": str(icode).zfill(2),
                "industry_name": str(iname),
                "candidate_count": int(len(g)),
                "today_avg_change": round(float(g["today_change_percent"].mean()), 3),
                "ret_5d_avg": round(float(pd.to_numeric(g["ret_5d"], errors="coerce").mean()), 3) if g["ret_5d"].notna().any() else 0.0,
                "ret_20d_avg": round(float(pd.to_numeric(g["ret_20d"], errors="coerce").mean()), 3) if g["ret_20d"].notna().any() else 0.0,
                "rising_count": int((g["today_change_percent"] > 0).sum()),
                "falling_count": int((g["today_change_percent"] < 0).sum()),
                "breadth_confidence": "較高" if len(g) >= 5 else "中" if len(g) >= 3 else "低：樣本少",
                "top_risers": [f"{x['name']}({x['symbol']}) {x['today_change_percent']:+.2f}%" for _, x in g.sort_values("today_change_percent", ascending=False).head(3).iterrows()],
                "top_fallers": [f"{x['name']}({x['symbol']}) {x['today_change_percent']:+.2f}%" for _, x in g.sort_values("today_change_percent").head(3).iterrows()],
            })
        industry_rows.sort(key=lambda x: x["today_avg_change"], reverse=True)

    result = {
        "generated_at": now_taipei().isoformat(),
        "universe_count": int(len(df)),
        "candidate_count": int(len(mdf)),
        "movers": mdf.to_dict("records") if not mdf.empty else [],
        "industry_summary": industry_rows,
        "method": "官方 TWSE/TPEx 當日收盤快照＋5/20交易日歷史報酬；Agent 再查原因與證據。",
        "data_source": "TWSE/TPEx官方日行情",
        "data_asof": report_date,
        "is_trading_day": True,
        "data_date_verified": True,
        "errors": errors,
    }
    return result, {"errors": errors, "method": "direct_official_daily_recovery"}


# ============================================================================
# Qwen3 Agent 重新研究
# ============================================================================
def _fallback_afterclose_agent(report: dict[str, Any], error: str) -> dict[str, Any]:
    mm = report.get("market_movers", {}) or {}
    movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]
    ups = sorted([x for x in movers if (_safe_float(x.get("today_change_percent"), 0) or 0) > 0], key=lambda x: _safe_float(x.get("today_change_percent"), 0) or 0, reverse=True)[:5]
    downs = sorted([x for x in movers if (_safe_float(x.get("today_change_percent"), 0) or 0) < 0], key=lambda x: _safe_float(x.get("today_change_percent"), 0) or 0)[:5]
    financials = sorted(
        [x for x in report.get("financial_snapshots", []) if isinstance(x, dict)],
        key=lambda x: _safe_float(x.get("revenue_yoy"), -999) or -999,
        reverse=True,
    )[:8]
    notes: list[dict[str, Any]] = []
    for x in ups:
        notes.append({
            "importance": "high",
            "type": "market",
            "symbol": clean_symbol(x.get("symbol")),
            "title": "今日上漲重點",
            "note": f"{x.get('name', x.get('symbol'))}（{x.get('symbol')}）今日 {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%；產業／業務為 {x.get('business_group') or x.get('industry_name') or '未分類'}。原因需以新聞與公告進一步交叉查證。",
            "evidence": [f"官方當日行情 {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%"],
            "tags": ["市場波動", "利多／原因待查證"],
            "confidence": 0.45,
            "follow_up": ["查詢公司公告與近期產業新聞"],
        })
    for x in downs:
        notes.append({
            "importance": "high",
            "type": "market",
            "symbol": clean_symbol(x.get("symbol")),
            "title": "今日下跌重點",
            "note": f"{x.get('name', x.get('symbol'))}（{x.get('symbol')}）今日 {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%；產業／業務為 {x.get('business_group') or x.get('industry_name') or '未分類'}。原因需以新聞與公告進一步交叉查證。",
            "evidence": [f"官方當日行情 {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%"],
            "tags": ["市場波動", "利空／原因待查證"],
            "confidence": 0.45,
            "follow_up": ["查詢公司公告與近期產業新聞"],
        })
    return {
        "agent_status": "fallback_afterclose_recovery",
        "model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        "generated_at": now_taipei().isoformat(),
        "overview": "官方行情與財務資料已完成 recovery；本次 Qwen3 深度研究未完成，因此以下僅保留證據層摘要。",
        "key_takeaways": [
            f"今日可驗證市場候選 {len(movers)} 筆",
            f"已恢復自選股財務資料 {len(financials)} 筆",
            f"CNYES 快取 {int((report.get('cnyes_news', {}) or {}).get('article_count', 0) or 0)} 篇",
        ],
        "research_notes": notes,
        "positive_signals": [f"{x.get('name', x.get('symbol'))}（{x.get('symbol')}）今日 {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%" for x in ups[:3]],
        "negative_signals": [f"{x.get('name', x.get('symbol'))}（{x.get('symbol')}）今日 {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%" for x in downs[:3]],
        "watch_items": [f"Agent 未完成自主研究：{error}"],
        "earnings_digest": "",
        "financial_focus": [
            f"{x.get('name', x.get('symbol'))}（{x.get('symbol')}）：營收 YoY {(_safe_float(x.get('revenue_yoy'), 0) or 0):+.2f}%；EPS {x.get('eps', '—')}"
            for x in financials[:5]
        ],
        "model_alert": "",
        "agent_actions": [],
        "error": error,
    }


def rerun_qwen_agent(report: dict[str, Any]) -> dict[str, Any]:
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    os.environ["AGENT_MAX_TOOL_ROUNDS"] = os.getenv("AFTER_CLOSE_AGENT_MAX_TOOL_ROUNDS", "10")
    if ResearchOrchestratorAgent is None:
        return _fallback_afterclose_agent(report, "agent_research_assistant.py 無法匯入 ResearchOrchestratorAgent")
    try:
        agent = ResearchOrchestratorAgent(BASE, ollama_host=host, ollama_model=model)
        notes = agent.run(report)
        if not isinstance(notes, dict):
            raise TypeError("ResearchOrchestratorAgent.run() 沒有回傳 dict")
        return notes
    except Exception as exc:
        return _fallback_afterclose_agent(report, f"ResearchOrchestratorAgent 失敗：{type(exc).__name__}: {exc}")


# ============================================================================
# 報告保存
# ============================================================================
def save_recovered_report(report: dict[str, Any]) -> None:
    report_date = str(report.get("report_date", ""))[:10]
    json_path = Path(str(report.get("json_path") or REPORT_DIR / f"research_{report_date}.json"))
    md_path = Path(str(report.get("markdown_path") or REPORT_DIR / f"research_{report_date}.md"))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    mm = report.get("market_movers", {}) or {}
    notes = report.get("agent_research_notes", {}) or {}
    lines = [
        f"# AI 台股盤後研究｜{report_date}",
        "",
        f"## ① 今日 AI 研究重點\n{notes.get('overview', '')}",
        "",
        "## 一、今日市場重點",
    ]
    movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]
    ups = sorted([x for x in movers if (_safe_float(x.get("today_change_percent"), 0) or 0) > 0], key=lambda x: _safe_float(x.get("today_change_percent"), 0) or 0, reverse=True)[:10]
    downs = sorted([x for x in movers if (_safe_float(x.get("today_change_percent"), 0) or 0) < 0], key=lambda x: _safe_float(x.get("today_change_percent"), 0) or 0)[:10]
    lines.append("### 漲幅重點")
    lines.extend([f"- **{x.get('name', x.get('symbol'))}（{x.get('symbol')}）** {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%｜{x.get('business_group') or x.get('industry_name') or '未分類'}" for x in ups] or ["- 本次沒有可用的上漲資料。"])
    lines.append("### 跌幅重點")
    lines.extend([f"- **{x.get('name', x.get('symbol'))}（{x.get('symbol')}）** {(_safe_float(x.get('today_change_percent'), 0) or 0):+.2f}%｜{x.get('business_group') or x.get('industry_name') or '未分類'}" for x in downs] or ["- 本次沒有可用的下跌資料。"])
    lines.append("### 產業強弱 Top 6")
    for x in (mm.get("industry_summary", []) or [])[:6]:
        if isinstance(x, dict):
            lines.append(f"- **{x.get('industry_name','')}**｜今日平均 {(_safe_float(x.get('today_avg_change'),0) or 0):+.2f}%｜漲 {x.get('rising_count',0)} / 跌 {x.get('falling_count',0)}")

    lines += ["", "## 二、最近 5 天法說會"]
    for x in (report.get("earnings_calls", []) or [])[:10]:
        if isinstance(x, dict) and not x.get("error"):
            lines.append(f"- **{x.get('name', x.get('symbol'))}（{x.get('symbol')}）**｜{x.get('event_date','')}｜{x.get('impact','待確認')}｜{x.get('one_line_summary') or x.get('summary','')}")

    lines += ["", "## 三、目前主要利多題材"]
    for x in (report.get("bullish_themes", []) or [])[:6]:
        if isinstance(x, dict):
            lines.append(f"- **{x.get('theme','')}**｜強度 {(_safe_float(x.get('strength'),0) or 0):.1f}/100｜{x.get('status','')}")

    lines += ["", "## 四、營收成長重點"]
    financials = sorted(
        [x for x in report.get("financial_snapshots", []) if isinstance(x, dict)],
        key=lambda x: _safe_float(x.get("revenue_yoy"), -999) or -999,
        reverse=True,
    )[:10]
    for x in financials:
        yoy = _safe_float(x.get("revenue_yoy"), None)
        yoy_text = f"{yoy:+.2f}%" if yoy is not None else "—"
        eps = _safe_float(x.get("eps"), None)
        eps_text = f"{eps:.2f}" if eps is not None else "—"
        lines.append(f"- **{x.get('name', x.get('symbol'))}（{x.get('symbol')}）**｜營收 YoY {yoy_text}｜EPS {eps_text}")

    lines += ["", "## 五、Agent 深度研究"]
    for n in (notes.get("research_notes", []) or [])[:30]:
        if isinstance(n, dict) and n.get("note"):
            lines.append(f"- **{n.get('title','研究發現')}**｜{n.get('note')}")

    src = report.get("official_sources", {}) or {}
    lines += [
        "", "## 六、資料來源與查證",
        f"TWSE公告 {src.get('twse_announcements_rows',0)} 筆、TPEx公告 {src.get('tpex_announcements_rows',0)} 筆、TWSE營收 {src.get('twse_revenue_rows',0)} 筆、TPEx營收 {src.get('tpex_revenue_rows',0)} 筆、TWSE財報 {src.get('twse_income_rows',0)} 筆、TPEx財報 {src.get('tpex_income_rows',0)} 筆、鉅亨近兩日新聞 {(report.get('cnyes_news',{}) or {}).get('article_count',0)} 篇",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# 主流程
# ============================================================================
def main() -> int:
    now = now_taipei()
    today = now.strftime("%Y-%m-%d")
    cutoff = now.normalize() + pd.Timedelta(hours=AFTER_CLOSE_HOUR, minutes=AFTER_CLOSE_MINUTE)
    watchlist = load_watchlist()

    write_log(today, "=" * 72)
    write_log(today, "AI TW STOCK AFTER-CLOSE RESEARCH v3")
    write_log(today, f"BASE={BASE}")
    write_log(today, f"PYTHON={os.sys.executable}")
    write_log(today, f"WATCHLIST_COUNT={len(watchlist)}")
    write_log(today, f"WATCHLIST={','.join(watchlist)}")
    write_log(today, f"OLLAMA_HOST={os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')}")
    write_log(today, f"OLLAMA_MODEL={os.getenv('OLLAMA_MODEL', 'qwen3:8b')}")
    write_log(today, f"MARKET_STATUS={market_status_text(now, BASE)}")
    write_log(today, "CNYES=盤後只讀夜間快取，不重新爬網站")

    if now < cutoff:
        write_log(today, f"尚未到 {AFTER_CLOSE_HOUR:02d}:{AFTER_CLOSE_MINUTE:02d}，略過研究。")
        return 0

    if not is_twse_trading_day(now, BASE):
        write_log(today, "今天非交易日，盤後排程正常結束，不寄送今日行情信件。")
        write_status(today, "skipped_non_trading_day")
        return 0

    os.environ["AFTER_CLOSE_FAST_MODE"] = "true"
    os.environ["EARNINGS_USE_CACHE"] = "true"
    os.environ["AGENT_MAX_TOOL_ROUNDS"] = os.getenv("AFTER_CLOSE_AGENT_MAX_TOOL_ROUNDS", "10")

    # ------------------------------------------------------------------
    # 1. 建立原始研究包
    # ------------------------------------------------------------------
    write_log(today, "[1/4] ResearchAgent 建立原始研究包")
    try:
        base_agent = ResearchAgent(BASE, ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"))
        report = base_agent.run_daily_research(symbols=watchlist, sector_codes=["24", "25", "26", "28"])
    except Exception as exc:
        write_log(today, f"ResearchAgent 執行失敗：{type(exc).__name__}: {exc}")
        write_log(today, traceback.format_exc())
        return 2

    report["report_type"] = "afterclose"
    report["report_date"] = today
    report["research_scope"] = report.get("research_scope", {}) or {}
    report["research_scope"]["watchlist_symbols"] = watchlist
    report.setdefault("report_generated_at", now.isoformat())

    # ------------------------------------------------------------------
    # 2. 官方資料與 CNYES recovery
    # ------------------------------------------------------------------
    write_log(today, "[2/4] 官方 TWSE / TPEx 財務與公告 recovery")
    snapshots, official_diag = _official_financial_recovery(watchlist)
    report["financial_snapshots"] = snapshots or report.get("financial_snapshots", []) or []
    report["official_sources"] = {
        **(report.get("official_sources", {}) or {}),
        **(official_diag.get("official_sources", {}) or {}),
    }
    report["official_announcements"] = official_diag.get("official_announcement_evidence", [])
    report["official_data_recovery"] = official_diag
    write_log(today, "OFFICIAL_COUNTS=" + json.dumps(report["official_sources"], ensure_ascii=False))
    write_log(today, f"FINANCIAL_SNAPSHOTS={len(report.get('financial_snapshots', []) or [])}")
    if official_diag.get("errors"):
        for err in official_diag["errors"][:8]:
            write_log(today, f"OFFICIAL_WARNING={err}")

    write_log(today, "[2/4] 市場行情 recovery")
    market_movers, market_diag = _market_recovery(BASE, watchlist, today)
    report["market_movers"] = market_movers
    report["market_data_recovery"] = market_diag
    write_log(today, f"MARKET_ASOF={market_movers.get('data_asof','')}")
    write_log(today, f"MARKET_UNIVERSE={market_movers.get('universe_count',0)} CANDIDATES={market_movers.get('candidate_count',0)}")
    market_warning = ""
    if not market_movers.get("movers"):
        market_warning = "官方當日行情 recovery 未取得 movers；本次研究不使用前一交易日行情冒充今日，仍繼續產生研究報告並寄送。"
        write_log(today, "WARNING=" + market_warning)
        report["market_data_status"] = "unavailable_today"
        report["market_data_warning"] = market_warning
        # 維持空白 movers，禁止用舊資料補齊。後續 Agent 與 Email 會知道今日行情不可用。
    elif str(market_movers.get("data_asof", ""))[:10] != today:
        market_warning = f"官方行情日期 {market_movers.get('data_asof')} 與報告日期 {today} 不一致；不採用該行情，仍繼續寄送研究報告。"
        write_log(today, "WARNING=" + market_warning)
        report["market_data_status"] = "date_mismatch_rejected"
        report["market_data_warning"] = market_warning
        report["market_movers"] = {
            "movers": [],
            "industry_summary": [],
            "data_asof": "",
            "is_trading_day": True,
            "data_source": "rejected_date_mismatch",
            "errors": list((market_movers.get("errors") or [])) + [market_warning],
        }
    else:
        report["market_data_status"] = "verified_today"

    write_log(today, "[2/4] CNYES 夜間快取 recovery")
    cnyes_news, morning_cnyes_digest, cnyes_path, cnyes_errors = _load_cnyes_cache(today)
    if cnyes_news:
        report["cnyes_news"] = cnyes_news
        report["official_sources"]["cnyes_news_articles"] = int(cnyes_news.get("article_count", 0) or len(cnyes_news.get("articles", []) or []))
        report["cnyes_news_cache_dates"] = cnyes_news.get("cache_dates", [])
        report["cnyes_news_cache_files"] = cnyes_news.get("cache_files", [])
        if morning_cnyes_digest:
            report["cnyes_research_digest"] = {**morning_cnyes_digest, "reused_from_morning_report": True}
        write_log(today, f"CNYES_ARTICLES={report['official_sources']['cnyes_news_articles']}")
        write_log(today, f"CNYES_CACHE_DATES={cnyes_news.get('cache_dates', [])}")
        write_log(today, f"CNYES_SOURCE={cnyes_path}")
    else:
        report["cnyes_news"] = report.get("cnyes_news", {}) or {}
        report["cnyes_news"]["article_count"] = 0
        report["official_sources"]["cnyes_news_articles"] = 0
        for err in cnyes_errors:
            write_log(today, f"CNYES_WARNING={err}")

    # ------------------------------------------------------------------
    # 3. 恢復完資料後，再交給 Qwen3 做真正的 Agent 研究
    # ------------------------------------------------------------------
    write_log(today, "[3/4] Qwen3 8B ResearchOrchestratorAgent 重新研究 recovery 後資料")
    notes = rerun_qwen_agent(report)
    report["agent_research_notes"] = notes
    report["ollama_analysis"] = {
        "available": str(notes.get("agent_status", "")).startswith("qwen3"),
        "model": notes.get("model", os.getenv("OLLAMA_MODEL", "qwen3:8b")),
        "content": notes.get("overview", ""),
        "validation": "post_recovery_agent",
        "agent_status": notes.get("agent_status", ""),
    }
    write_log(today, f"AGENT_STATUS={notes.get('agent_status','')}")
    write_log(today, f"AGENT_RESEARCH_NOTES={len(notes.get('research_notes', []) or [])}")
    write_log(today, f"AGENT_POSITIVE={len(notes.get('positive_signals', []) or [])}")
    write_log(today, f"AGENT_NEGATIVE={len(notes.get('negative_signals', []) or [])}")
    write_log(today, f"AGENT_WATCH={len(notes.get('watch_items', []) or [])}")
    if notes.get("agent_actions"):
        for action in notes.get("agent_actions", [])[-15:]:
            if isinstance(action, dict):
                write_log(today, f"AGENT_ACTION={action.get('tool','')}")

    # Agent 研究結果也可能產生自己的新聞 digest；若沒有則保留晨報 cache。
    if not report.get("cnyes_research_digest") and morning_cnyes_digest:
        report["cnyes_research_digest"] = {**morning_cnyes_digest, "reused_from_morning_report": True}

    report["agent_research_notes"] = report.get("agent_research_notes", {}) or {}
    report["agent_research_notes"]["official_recovery_summary"] = {
        "official_financial_snapshots": len(report.get("financial_snapshots", []) or []),
        "official_announcements": len(report.get("official_announcements", []) or []),
        "market_universe": int((report.get("market_movers", {}) or {}).get("universe_count", 0) or 0),
        "cnyes_articles": int((report.get("cnyes_news", {}) or {}).get("article_count", 0) or 0),
    }

    # ------------------------------------------------------------------
    # 4. 保存後再寄送
    # ------------------------------------------------------------------
    save_recovered_report(report)
    write_log(today, f"REPORT_JSON={report.get('json_path')}")
    write_log(today, f"REPORT_MD={report.get('markdown_path')}")

    write_log(today, "[4/4] Email Agent")
    try:
        email = EmailAgent(BASE)
        status = email.status()
        write_log(today, f"EMAIL_CONFIGURED={status.get('configured')} AUTO_SEND={status.get('auto_send')} RECIPIENT={status.get('recipient')}")
        if not status.get("configured"):
            write_status(today, "error_email_not_configured", missing=status.get("missing", []))
            return 3
        if not email.config.auto_send:
            email_result = {"sent": False, "skipped": True, "reason": "EMAIL_AUTO_SEND=false", "status": status}
        else:
            email_result = email.send_report(report, force=True)
    except Exception as exc:
        email_result = {"sent": False, "skipped": False, "error": f"Email Agent 執行失敗：{type(exc).__name__}: {exc}"}

    write_log(today, "EMAIL_RESULT=" + json.dumps(email_result, ensure_ascii=False, default=str))
    if email_result.get("error"):
        write_status(today, "error_email", email_result=email_result)
        return 3

    write_status(
        today,
        "success",
        market_asof=market_movers.get("data_asof", ""),
        market_data_status=report.get("market_data_status", ""),
        market_warning=report.get("market_data_warning", ""),
        market_universe=market_movers.get("universe_count", 0),
        market_candidates=market_movers.get("candidate_count", 0),
        financial_snapshots=len(report.get("financial_snapshots", []) or []),
        cnyes_articles=int((report.get("cnyes_news", {}) or {}).get("article_count", 0) or 0),
        agent_status=notes.get("agent_status", ""),
        email_sent=bool(email_result.get("sent")),
    )
    write_log(today, "=" * 72)
    write_log(today, "AI AFTER-CLOSE RESEARCH COMPLETED")
    write_log(today, "=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        report_date = now_taipei().strftime("%Y-%m-%d")
        write_status(report_date, "error_unhandled", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(traceback.format_exc())
        raise
