# -*- coding: utf-8 -*-
"""tw-stock-research：台股研究專案的 MCP server（stdio）。

給 Claude（排程任務、互動對話）使用的結構化工具，分四組：
  1. 市場資料    get_quote / get_price_history / get_institutional_flow / get_holder_structure / get_market_status
                 analyze_entry（5 分 K＋日 K 進場分析）／ audit_entry_model（進場模型樣本外稽核）
  2. 研究包查證  research_* （盤後、早報研究包；與 claude_tools.py 相同的工具實作）
  3. 流程維運    pipeline_status / get_job_brief / get_stock_note
  4. 品質把關    validate_afterclose_notes / validate_morning_notes / validate_finance_digest / validate_earnings_memo

寫入動作（寄信、發布）不在這裡：仍由 Skill 透過 run_*.py --finalize 與 *_digest/memo apply 執行，
那些入口會再跑一次同樣的品質檢查，有錯誤就拒絕寄出。

註冊方式：專案根目錄的 .mcp.json。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mcp.server.mcpserver import MCPServer  # noqa: E402

import entry_analysis  # noqa: E402
import pipeline_status as ps  # noqa: E402
import quality_checks as qc  # noqa: E402
from agent_research_assistant import ResearchToolbox  # noqa: E402
from market_calendar import is_twse_trading_day, market_status_text  # noqa: E402
from stock_api import FugleClient, clean_symbol, fetch_institution_daily, fetch_tdcc_big_retail  # noqa: E402

TAIPEI = ZoneInfo("Asia/Taipei")
JOBS = ROOT / "output" / "claude_jobs"
MAX_CHARS = 20000

mcp = MCPServer(
    name="tw-stock-research",
    title="台股研究專案工具",
    instructions=(
        "台股研究專案的資料、研究包與品質檢查工具。研究流程寫在 .claude/skills/；"
        "撰寫研究筆記前用 research_* 查證，寫完後一定要用 validate_* 檢查並修正所有 errors，"
        "再執行 Skill 指定的 finalize／apply 指令。不給買賣建議。"
    ),
)
_client: FugleClient | None = None
_toolbox = ResearchToolbox(ROOT)
_packages: dict[str, tuple[float, dict[str, Any]]] = {}


def _today() -> str:
    return datetime.now(TAIPEI).strftime("%Y-%m-%d")


def _fugle() -> FugleClient:
    global _client
    if _client is None:
        _client = FugleClient(wait_on_limit=True)
    return _client


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _frame(df) -> list[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    return _jsonable(df.to_dict("records"))


def _clip(value: Any, max_chars: int = MAX_CHARS) -> Any:
    """太大的結果截斷成文字，避免一次塞滿 context。"""
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return _jsonable(value)
    return {"truncated": True, "total_chars": len(text), "preview": text[:max_chars],
            "hint": "結果過長已截斷；請縮小查詢（例如加上 symbol、query 或 limit）。"}


def _package(job: str, report_date: str) -> dict[str, Any]:
    path = JOBS / f"{job}_{report_date or _today()}" / "package.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到研究包 {path}；請先執行對應的 --prepare")
    mtime = path.stat().st_mtime
    cached = _packages.get(str(path))
    if not cached or cached[0] != mtime:
        _packages[str(path)] = (mtime, json.loads(path.read_text(encoding="utf-8")))
    return _packages[str(path)][1]


def _research(tool: str, args: dict[str, Any], job: str, report_date: str) -> Any:
    return _clip(_toolbox._tool(tool, args, _package(job, report_date)))


# ---------------------------------------------------------------------------
# 1. 市場資料
# ---------------------------------------------------------------------------
@mcp.tool()
def get_quote(symbol: str) -> dict:
    """台股即時／最近一筆報價（Fugle）。symbol 例：2330。"""
    return _jsonable(_fugle().quote(clean_symbol(symbol)))


@mcp.tool()
def get_price_history(symbol: str, days: int = 60) -> list[dict]:
    """日 K 歷史行情（Fugle），預設最近 60 個日曆日，最多 400 天。"""
    end = date.today()
    start = end - timedelta(days=max(5, min(int(days), 400)))
    return _frame(_fugle().historical_candles(clean_symbol(symbol), start, end, "D"))


@mcp.tool()
def get_institutional_flow(symbol: str) -> list[dict]:
    """最近 10 個交易日三大法人買賣超（TWSE T86／TPEx 日報，股數）。"""
    return _frame(fetch_institution_daily(clean_symbol(symbol)))


@mcp.tool()
def get_holder_structure(symbol: str) -> list[dict]:
    """集保大戶／散戶持股比例（TDCC，每週）。"""
    return _frame(fetch_tdcc_big_retail(clean_symbol(symbol)))


@mcp.tool()
def get_market_status() -> dict:
    """台股今天是否交易日、目前盤中／盤後狀態（台灣時間）。"""
    now = datetime.now(TAIPEI)
    return {"now": now.isoformat(timespec="seconds"), "is_trading_day": bool(is_twse_trading_day(now, ROOT)),
            "status": market_status_text(now, ROOT)}


@mcp.tool()
def analyze_entry(symbol: str) -> dict:
    """單一股票的進場分析：5 分 K／日 K 型態（-1 偏空～+1 偏多）、日內＋日線趨勢、進場評分與區間，
    以及模型的樣本外表現（prob_has_skill=false 時回檔機率是歷史頻率）。技術面參考，不是買賣建議。"""
    return _jsonable(entry_analysis.analyze(symbol))


@mcp.tool()
def audit_entry_model(symbols: list[str] | None = None) -> dict:
    """重新訓練並以前推驗證稽核進場模型（不給 symbols＝全部自選股，約 5 秒／檔），
    回報哪些股票的回檔機率模型比歷史頻率準、與上次稽核相比失去或恢復預測力的股票。"""
    return _jsonable(entry_analysis.audit(symbols))


# ---------------------------------------------------------------------------
# 2. 研究包查證（job = afterclose 或 morning；report_date 預設今天）
# ---------------------------------------------------------------------------
@mcp.tool()
def research_market_overview(focus: str = "", job: str = "afterclose", report_date: str = "") -> Any:
    """研究包的市場強弱：漲跌幅最大的 20 檔與產業彙整。"""
    return _research("get_market_overview", {"focus": focus}, job, report_date)


@mcp.tool()
def research_company_financial(symbol: str, job: str = "afterclose", report_date: str = "") -> Any:
    """研究包中指定公司的營收、YoY、EPS 與業務分類。"""
    return _research("get_company_financial", {"symbol": symbol}, job, report_date)


@mcp.tool()
def research_company_profile(symbol: str, job: str = "afterclose", report_date: str = "") -> Any:
    """公司業務定位、產業鏈與研究關鍵字。"""
    return _research("get_company_research", {"symbol": symbol}, job, report_date)


@mcp.tool()
def research_earnings_events(symbol: str = "", job: str = "afterclose", report_date: str = "") -> Any:
    """近期法說會事件與已有的摘要；symbol 留空列出全部。"""
    return _research("get_earnings_events", {"symbol": symbol}, job, report_date)


@mcp.tool()
def research_read_fugle_memo(url: str) -> Any:
    """開啟 Fugle 法說會備忘錄詳細文章並回傳正文（會連網，較慢）。"""
    return _clip(_toolbox._tool("read_fugle_memo", {"url": url}, {}))


@mcp.tool()
def research_news_batch(offset: int = 0, limit: int = 50, job: str = "afterclose", report_date: str = "") -> Any:
    """批次瀏覽研究包內鉅亨近兩日新聞（標題、時間、短摘要、網址）。"""
    return _research("get_cnyes_news_batch", {"offset": offset, "limit": limit}, job, report_date)


@mcp.tool()
def research_search_news(query: str, limit: int = 12, job: str = "afterclose", report_date: str = "") -> Any:
    """在研究包的鉅亨新聞全文中搜尋（例：「載板 ABF CCL 上漲」「2408 南亞科 DRAM」）。"""
    return _research("search_cnyes_news", {"query": query, "limit": limit}, job, report_date)


@mcp.tool()
def research_search_live_news(query: str, days: int = 3) -> Any:
    """研究包證據不足時，用 Google News RSS 補查近期新聞（會連網）。"""
    return _clip(_toolbox._tool("search_live_news", {"query": query, "days": days}, {}))


@mcp.tool()
def research_theme_evidence(theme: str, job: str = "afterclose", report_date: str = "") -> Any:
    """指定題材的支持證據、營收支撐與矛盾訊號。"""
    return _research("get_theme_evidence", {"theme": theme}, job, report_date)


@mcp.tool()
def research_previous_notes() -> Any:
    """上一期盤後研究筆記（只能當成待驗證的線索）。"""
    return _clip(_toolbox._tool("get_previous_notes", {}, {}))


# ---------------------------------------------------------------------------
# 3. 流程維運
# ---------------------------------------------------------------------------
@mcp.tool()
def pipeline_status(report_date: str = "") -> dict:
    """今天（或指定日期）早報、盤後研究、財經新聞、法說會、Windows 排程的狀態與待處理問題。"""
    return _jsonable(ps.pipeline_status(report_date or None))


@mcp.tool()
def get_job_brief(job: str, report_date: str = "") -> str:
    """讀取研究包摘要 brief.md。job：afterclose、morning、finance、earnings。"""
    if job in ("finance", "earnings"):
        path = JOBS / job / "brief.md"
    else:
        path = JOBS / f"{job}_{report_date or _today()}" / "brief.md"
    if not path.exists():
        return f"找不到 {path}"
    text = path.read_text(encoding="utf-8")
    return text if len(text) <= 60000 else text[:60000] + "\n…（已截斷）"


@mcp.tool()
def get_stock_note(symbol: str) -> dict:
    """網站上目前發布的 Claude 個股研究筆記。"""
    data = json.loads((ROOT / "output" / "research_reports" / "stock_notes_latest.json").read_text(encoding="utf-8")) \
        if (ROOT / "output" / "research_reports" / "stock_notes_latest.json").exists() else {}
    note = next((n for n in data.get("notes", []) if str(n.get("symbol")) == clean_symbol(symbol)), None)
    return {"generated_at": data.get("generated_at"), "report_date": data.get("report_date"), "note": note}


# ---------------------------------------------------------------------------
# 4. 品質把關（finalize／apply 也會跑同樣的檢查；errors 必須為 0 才會寄出／發布）
# ---------------------------------------------------------------------------
@mcp.tool()
def validate_afterclose_notes(report_date: str = "") -> dict:
    """檢查 agent_notes.json：自選股是否都有筆記、今日漲跌幅是否與行情一致、連結是否在研究包、是否含投資建議。"""
    return qc.validate_afterclose(report_date or _today())


@mcp.tool()
def validate_morning_notes(report_date: str = "") -> dict:
    """檢查 morning_notes.json：新聞連結、觀察股票是否為自選股、是否含投資建議或目標價。"""
    return qc.validate_morning(report_date or _today())


@mcp.tool()
def validate_finance_digest(notes_path: str = "output/claude_jobs/finance/digest_notes.json") -> dict:
    """檢查財經 AI 重點：articles 的 id 是否存在、highlights 是否在重點裡。"""
    return qc.validate_finance_digest(ROOT / notes_path)


@mcp.tool()
def validate_earnings_memo(notes_path: str = "output/claude_jobs/earnings/memo_notes.json") -> dict:
    """檢查法說會摘要：url 是否在待分析清單、數字是否出現在備忘錄正文。"""
    return qc.validate_earnings_memo(ROOT / notes_path)


if __name__ == "__main__":
    mcp.run("stdio")
