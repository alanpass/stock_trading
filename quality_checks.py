# -*- coding: utf-8 -*-
"""Claude 研究結果的品質把關（guardrails）。

MCP server（validate_* 工具）、run_*.py --finalize 與 *_digest/memo apply 都會呼叫這裡；
有 error 時 finalize／apply 會拒絕寄信與發布，Claude 依 issues 修正後再試。

每個檢查回傳 {"ok": bool, "errors": [...], "warnings": [...], "stats": {...}}。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from stock_api import clean_symbol

BASE = Path(__file__).resolve().parent
JOB_ROOT = BASE / "output" / "claude_jobs"
REPORT_DIR = BASE / "output" / "research_reports"

# 研究筆記只整理事實與證據；出現這些字眼視為投資建議。
ADVICE_PHRASES = ["建議買進", "建議賣出", "建議加碼", "建議減碼", "逢低買進", "逢高賣出", "可以買進", "應該買", "應該賣", "進場買", "停損賣出"]
# 「今日／收盤」後面緊接的漲跌幅，例如：今日上漲4.67%、今日-1.54%、收盤 +0.35%
DAILY_MOVE = re.compile(r"(?:今日|今天|收盤|盤後)[^，。；%0-9+\-]{0,8}?(上漲|下跌|大漲|大跌|漲|跌)?\s?([+\-－＋]?\d+(?:\.\d+)?)\s?%")
URL = re.compile(r"https?://[^\s\"'）)]+")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _result(errors: list[str], warnings: list[str], **stats: Any) -> dict[str, Any]:
    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(*parts: Any) -> str:
    out = []
    for p in parts:
        if isinstance(p, (list, tuple)):
            out.extend(str(x) for x in p)
        elif p:
            out.append(str(p))
    return " ".join(out)


def _advice_hits(text: str) -> list[str]:
    return [p for p in ADVICE_PHRASES if p in text]


def _package_urls(package: dict[str, Any]) -> set[str]:
    """研究包裡出現過的所有網址（新聞、法說會、官方公告）。"""
    blob = json.dumps({k: package.get(k) for k in ("cnyes_news", "news", "sector_news", "earnings_calls",
                                                   "order_evidence", "official_announcements", "theme_research",
                                                   "bullish_themes", "company_research")}, ensure_ascii=False, default=str)
    return {u.rstrip(".,;，。") for u in URL.findall(blob)}


def _to_float(value: Any) -> float | None:
    try:
        v = float(value)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def quality_gate(check: dict, write_status_fn=None) -> int | None:
    """品質把關：有 error 就拒絕寄信／發布（exit 6），列出要修正的地方。"""
    for w in check.get("warnings", [])[:20]:
        print(f"QUALITY_WARNING: {w}")
    if check.get("ok"):
        print(f"QUALITY_OK {check.get('stats', {})}")
        return None
    for e in check.get("errors", []):
        print(f"QUALITY_ERROR: {e}")
    if write_status_fn:
        write_status_fn(check)
    return 6


# ----------------------------------------------------------------------------
# 盤後研究
# ----------------------------------------------------------------------------
def validate_afterclose(report_date: str) -> dict[str, Any]:
    d = JOB_ROOT / f"afterclose_{report_date}"
    if not (d / "package.json").exists():
        return _result([f"找不到研究包 {d / 'package.json'}，請先執行 --prepare"], [])
    if not (d / "agent_notes.json").exists():
        return _result([f"找不到 {d / 'agent_notes.json'}"], [])
    package = _load(d / "package.json")
    try:
        notes = _load(d / "agent_notes.json")
    except json.JSONDecodeError as exc:
        return _result([f"agent_notes.json 不是合法 JSON：{exc}"], [])
    if not isinstance(notes, dict):
        return _result(["agent_notes.json 必須是 JSON 物件"], [])

    errors: list[str] = []
    warnings: list[str] = []
    if not str(notes.get("overview", "")).strip():
        errors.append("缺少 overview（今日盤後總結）")
    research = [n for n in notes.get("research_notes") or [] if isinstance(n, dict)]
    if not research:
        errors.append("research_notes 是空的")
    for i, n in enumerate(research):
        if not str(n.get("note", "")).strip():
            errors.append(f"research_notes[{i}]「{n.get('title', '')}」沒有 note 內容")
        if not n.get("evidence"):
            warnings.append(f"research_notes[{i}]「{n.get('title', '')}」沒有 evidence")

    known_urls = _package_urls(package)
    movers = {clean_symbol(x.get("symbol")): x for x in (package.get("market_movers") or {}).get("movers", []) if isinstance(x, dict)}
    watchlist = [clean_symbol(s) for s in (package.get("research_scope") or {}).get("watchlist_symbols", []) or []]
    stock_notes = [n for n in notes.get("stock_notes") or [] if isinstance(n, dict)]
    by_symbol = {clean_symbol(n.get("symbol")): n for n in stock_notes}

    for s in watchlist:
        if s not in by_symbol:
            errors.append(f"自選股 {s} 沒有個股筆記")
    hot = [n for n in stock_notes if str(n.get("category", "")).lower() != "watchlist"]
    if len(hot) < 5:
        warnings.append(f"熱門股筆記只有 {len(hot)} 檔（建議 8～10 檔）")

    for n in stock_notes:
        sym = clean_symbol(n.get("symbol"))
        label = f"{n.get('name', '')}（{sym}）"
        body = _text(n.get("headline"), n.get("summary"))
        summary_len = len(str(n.get("summary", "")).strip())
        if summary_len < 20:
            errors.append(f"{label} 的 summary 太短或空白")
        elif summary_len < 60:
            warnings.append(f"{label} 的 summary 只有 {summary_len} 字")
        # 今日漲跌幅：文字裡出現帶正負號的百分比時，至少要有一個與實際行情一致
        actual = _to_float((movers.get(sym) or {}).get("today_change_percent"))
        if actual is not None:
            stated = []
            for word, num in DAILY_MOVE.findall(body):
                v = float(num.replace("－", "-").replace("＋", "+"))
                if word in ("下跌", "大跌", "跌") and v > 0:
                    v = -v
                stated.append(v)
            if stated and not any(abs(v - actual) <= 0.06 for v in stated):
                errors.append(f"{label} 文字中的漲跌幅 {stated[:3]} 與實際今日 {actual:+.2f}% 不符")
            if "漲停" in body and actual < 9.0 and "前一日" not in body and "10/1" not in body:
                warnings.append(f"{label} 提到漲停，但今日實際 {actual:+.2f}%")
            if "跌停" in body and actual > -9.0 and "前一日" not in body:
                warnings.append(f"{label} 提到跌停，但今日實際 {actual:+.2f}%")
        elif sym not in watchlist:
            warnings.append(f"{label} 不在研究包的行情候選中，無法核對漲跌幅")
        for url in n.get("source_links") or []:
            if str(url).rstrip(".,;，。") not in known_urls:
                warnings.append(f"{label} 的來源連結不在研究包內：{url}")
        hits = _advice_hits(_text(n.get("headline"), n.get("summary"), n.get("positives"), n.get("risks"), n.get("watch")))
        if hits:
            errors.append(f"{label} 含投資建議用語：{hits}")

    for i, n in enumerate(research):
        for url in n.get("source_links") or []:
            if str(url).rstrip(".,;，。") not in known_urls:
                warnings.append(f"research_notes[{i}] 的來源連結不在研究包內：{url}")
    hits = _advice_hits(_text(notes.get("overview"), notes.get("key_takeaways"), [n.get("note") for n in research]))
    if hits:
        errors.append(f"研究筆記含投資建議用語：{hits}")
    return _result(errors, warnings, research_notes=len(research), stock_notes=len(stock_notes),
                   watchlist=len(watchlist), hot=len(hot))


# ----------------------------------------------------------------------------
# 早報
# ----------------------------------------------------------------------------
def validate_morning(report_date: str) -> dict[str, Any]:
    d = JOB_ROOT / f"morning_{report_date}"
    if not (d / "package.json").exists():
        return _result([f"找不到研究包 {d / 'package.json'}，請先執行 --prepare"], [])
    if not (d / "morning_notes.json").exists():
        return _result([f"找不到 {d / 'morning_notes.json'}"], [])
    package = _load(d / "package.json")
    try:
        notes = _load(d / "morning_notes.json")
    except json.JSONDecodeError as exc:
        return _result([f"morning_notes.json 不是合法 JSON：{exc}"], [])
    errors: list[str] = []
    warnings: list[str] = []
    if not str(notes.get("overview", "")).strip():
        errors.append("缺少 overview")
    news = [x for x in notes.get("news_summary") or [] if isinstance(x, dict)]
    if not news:
        errors.append("news_summary 是空的")
    urls = {str(a.get("url", "")).strip() for a in (package.get("cnyes_news") or {}).get("articles", []) or [] if isinstance(a, dict)}
    for i, x in enumerate(news):
        if not str(x.get("summary", "")).strip():
            errors.append(f"news_summary[{i}]「{x.get('title', '')}」沒有摘要")
        bad = [u for u in x.get("source_links") or [] if str(u).strip() not in urls]
        if bad:
            warnings.append(f"news_summary[{i}] 的連結不在研究包內，會被移除：{bad}")
        if not x.get("published_at"):
            warnings.append(f"news_summary[{i}]「{x.get('title', '')}」缺少 published_at")
    watch = {clean_symbol(s) for s in (package.get("research_scope") or {}).get("watchlist_symbols", []) or []}
    for x in notes.get("recommended_stocks") or []:
        sym = clean_symbol((x or {}).get("symbol"))
        if watch and sym not in watch:
            warnings.append(f"觀察股票 {sym} 不在自選股內，會被移除")
    hits = _advice_hits(json.dumps(notes, ensure_ascii=False))
    if hits:
        errors.append(f"早報含投資建議用語：{hits}")
    if "目標價" in _text([x.get("reason") for x in notes.get("recommended_stocks") or [] if isinstance(x, dict)]):
        errors.append("觀察股票的理由不可以寫目標價")
    return _result(errors, warnings, news_summary=len(news), recommended_stocks=len(notes.get("recommended_stocks") or []))


# ----------------------------------------------------------------------------
# 財經 AI 重點 / 法說會
# ----------------------------------------------------------------------------
def validate_finance_digest(notes_path: str | Path, latest_path: str | Path | None = None) -> dict[str, Any]:
    notes = _load(Path(notes_path))
    latest = _load(Path(latest_path or REPORT_DIR / "finance_info_latest.json"))
    errors: list[str] = []
    warnings: list[str] = []
    news = notes.get("news") or {}
    if news and not news.get("key_points"):
        errors.append("news.key_points 是空的")
    ids = {str(a.get("article_id") or a.get("url") or a.get("title") or "") for a in latest.get("news", []) or [] if isinstance(a, dict)}
    for a in notes.get("articles") or []:
        if str(a.get("id")) not in ids:
            errors.append(f"articles 的 id {a.get('id')} 不在目前的新聞列表（請照 brief.md 的 id 填寫）")
        if not a.get("points"):
            errors.append(f"articles {a.get('id')} 沒有 points")
        blob = " ".join(str(p) for p in a.get("points") or [])
        missing = [h for h in a.get("highlights") or [] if str(h) not in blob]
        if missing:
            warnings.append(f"articles {a.get('id')} 的 highlights 不在 points 裡，會被移除：{missing}")
        if a.get("sentiment") not in {"利多", "利空", "中性", "混合"}:
            warnings.append(f"articles {a.get('id')} 的 sentiment 不是 利多／利空／中性／混合")
    hits = _advice_hits(json.dumps(notes, ensure_ascii=False))
    if hits:
        errors.append(f"含投資建議用語：{hits}")
    return _result(errors, warnings, articles=len(notes.get("articles") or []), key_points=len(news.get("key_points") or []))


def validate_earnings_memo(notes_path: str | Path, latest_path: str | Path | None = None) -> dict[str, Any]:
    notes = _load(Path(notes_path))
    latest = _load(Path(latest_path or REPORT_DIR / "finance_info_latest.json"))
    memos = {str(x.get("url") or x.get("source_url") or ""): x for x in latest.get("earnings", []) or [] if isinstance(x, dict)}
    errors: list[str] = []
    warnings: list[str] = []
    items = [x for x in notes.get("items") or [] if isinstance(x, dict)]
    if not items:
        errors.append("items 是空的")
    for x in items:
        url = str(x.get("url", ""))
        memo = memos.get(url)
        if memo is None:
            errors.append(f"url 不在待分析清單：{url}")
            continue
        if x.get("impact") not in {"利多", "利空", "中性", "混合"}:
            errors.append(f"{url} 的 impact 必須是 利多／利空／中性／混合")
        if not str(x.get("one_line_summary", "")).strip():
            errors.append(f"{url} 缺少 one_line_summary")
        source = re.sub(r"[,\s]", "", _text(memo.get("memo_text"), list((memo.get("sections") or {}).values())))
        for key in ("financial_highlights", "one_line_summary"):
            for num in NUMBER.findall(_text(x.get(key))):
                plain = num.replace(",", "")
                if len(plain.replace(".", "")) >= 3 and plain not in source:
                    warnings.append(f"{url} 的 {key} 數字 {num} 在備忘錄正文中找不到，請確認沒有推估")
    hits = _advice_hits(json.dumps(notes, ensure_ascii=False))
    if hits:
        errors.append(f"含投資建議用語：{hits}")
    return _result(errors, warnings, items=len(items))
