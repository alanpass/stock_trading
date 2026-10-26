# -*- coding: utf-8 -*-
"""每日研究流程的健康狀態（給 MCP server、pipeline-health skill 與人工檢查使用）。

  python pipeline_status.py [--date YYYY-MM-DD]

把 Windows 排程（資料）與 Claude 排程（研究）的結果放在同一份報表：
每個流程回報 stage（完成到哪一步）與 problems（需要處理的問題）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
LOGS = OUT / "scheduler_logs"
REPORTS = OUT / "research_reports"
JOBS = OUT / "claude_jobs"
TAIPEI = ZoneInfo("Asia/Taipei")


def _now() -> datetime:
    return datetime.now(TAIPEI)


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _age_minutes(stamp: Any) -> float | None:
    try:
        dt = datetime.fromisoformat(str(stamp))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TAIPEI)
        return round((_now() - dt).total_seconds() / 60, 1)
    except Exception:
        return None


def _log_tail(path: Path, lines: int = 6) -> list[str]:
    try:
        return [x[:300] for x in path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]]
    except OSError:
        return []


def _job(name: str, day: str) -> dict[str, Any]:
    d = JOBS / f"{name}_{day}"
    return {"job_dir": str(d), "package": (d / "package.json").exists(),
            "notes": any((d / f).exists() for f in ("agent_notes.json", "morning_notes.json"))}


def afterclose_status(day: str) -> dict[str, Any]:
    st = _read(LOGS / f"after_close_status_{day}.json") or {}
    runner = _read(LOGS / f"afterclose_status_{day}.json") or {}
    job = _job("afterclose", day)
    notes = _read(REPORTS / "stock_notes_latest.json") or {}
    status = str(st.get("status", ""))
    stage = ("sent" if status == "success" else "skipped" if status == "skipped_non_trading_day"
             else "notes_written" if job["notes"] else "awaiting_claude" if job["package"] else status or "not_started")
    problems = []
    if status.startswith("error"):
        problems.append(f"盤後研究狀態 {status}：{st.get('errors') or st.get('email_result') or st.get('diagnostics') or ''}")
    if runner.get("status") in {"child_failed", "timeout", "script_missing"}:
        problems.append(f"Windows 排程 afterclose：{runner.get('status')}（rc={runner.get('child_return_code')}）")
    if status == "success" and not st.get("stock_notes_published", True):
        problems.append("個股筆記沒有發布到 GitHub")
    return {"stage": stage, "status": status, "updated_at": st.get("updated_at"), **job,
            "stock_notes": {"report_date": notes.get("report_date"), "generated_at": notes.get("generated_at"), "count": notes.get("count")},
            "problems": problems, "log_tail": _log_tail(LOGS / f"after_close_research_{day}.log")}


def morning_status(day: str) -> dict[str, Any]:
    st = _read(LOGS / f"morning_entry_status_{day}.json") or {}
    job = _job("morning", day)
    status = str(st.get("status", ""))
    stage = "sent" if status == "success" else "notes_written" if job["notes"] else "awaiting_claude" if job["package"] else status or "not_started"
    problems = []
    if status in {"email_failed", "exception", "report_validation_failed", "email_not_configured", "email_disabled"} or status.startswith("error"):
        problems.append(f"早報狀態 {status}：{st.get('message') or st.get('error') or st.get('errors') or ''}")
    return {"stage": stage, "status": status, "updated_at": st.get("updated_at"), **job, "problems": problems,
            "log_tail": _log_tail(LOGS / f"morning_entry_{day}.log")}


def finance_status() -> dict[str, Any]:
    latest = _read(REPORTS / "finance_info_latest.json") or {}
    public = _read(REPORTS / "finance_info_public.json") or {}
    digest = latest.get("news_digest") or {}
    earnings = [x for x in latest.get("earnings", []) or [] if isinstance(x, dict)]
    pending = [x for x in earnings if x.get("analysis_source") != "Claude 法說會 Agent" and not x.get("agent_tool_used")]
    news = [x for x in latest.get("news", []) or [] if isinstance(x, dict)]
    age = _age_minutes(latest.get("updated_at"))
    problems = []
    if age is None or age > 8 * 60:
        problems.append(f"財經新聞超過 8 小時沒有更新（updated_at={latest.get('updated_at')}）")
    if latest.get("stale"):
        problems.append("最近一次新聞爬取沒有拿到資料（stale）")
    if digest.get("generated_by") != "claude":
        problems.append("AI 今日重點目前不是 Claude 寫的（等待 finance-news-digest）")
    pub_age = _age_minutes(public.get("published_at"))
    if pub_age is None or (age is not None and pub_age > age + 30):
        problems.append("網站上的財經資料比本機舊（publish_research_data.py 可能失敗）")
    return {
        "updated_at": latest.get("updated_at"), "age_minutes": age, "news": len(news),
        "claude_article_points": sum(1 for x in news if x.get("ai_source") == "claude"),
        "digest_by": digest.get("generated_by"), "digest_generated_at": digest.get("agent_generated_at"),
        "earnings": len(earnings), "earnings_pending_claude": len(pending),
        "published_at": public.get("published_at"), "problems": problems,
    }


def windows_jobs(day: str) -> dict[str, Any]:
    out = {}
    for job in ("morning", "afterclose", "finance", "earnings"):
        st = _read(LOGS / f"{job}_status_{day}.json") or {}
        out[job] = {k: st.get(k) for k in ("status", "updated_at", "child_return_code", "message") if st.get(k) is not None}
    return out


def _secret(name: str) -> str:
    """環境變數 → 本機 .streamlit/secrets.toml（帳號資料庫設定只放在 secrets）。"""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(tomllib.loads((BASE / ".streamlit" / "secrets.toml").read_text(encoding="utf-8")).get(name, "")).strip()
    except (OSError, ValueError):
        return ""


def account_db_status() -> dict[str, Any]:
    """查一次網站帳號資料庫（Supabase）。

    Supabase 免費專案閒置約 7 天會自動暫停，暫停後網站登入會失敗（2026-10-26 發生過）；
    健康檢查每天兩次真的查詢一筆資料，順便讓專案保持活躍。
    """
    import requests
    from account_provisioning import _normalize_project_url

    url, key = _secret("ACCOUNT_DB_URL"), _secret("ACCOUNT_DB_SERVICE_ROLE_KEY")
    if not (url and key):
        return {"configured": False, "problems": []}
    url = _normalize_project_url(url)   # secrets 裡的網址可能帶 /rest/v1/
    start = time.monotonic()
    try:
        r = requests.get(f"{url}/rest/v1/account_requests", params={"select": "email", "limit": "1"},
                         headers={"apikey": key, "Authorization": "Bearer " + key}, timeout=15)
        ok, err = r.status_code == 200, None if r.status_code == 200 else f"HTTP {r.status_code}"
    except requests.RequestException as exc:
        ok, err = False, type(exc).__name__
    out = {"configured": True, "ok": ok, "ms": round((time.monotonic() - start) * 1000)}
    if err:
        out["error"] = err
    out["problems"] = [] if ok else [f"帳號資料庫連不上（{err}），網站無法登入新帳號；可能是 Supabase 專案被自動暫停"]
    return out


def pipeline_status(day: str | None = None) -> dict[str, Any]:
    day = day or _now().strftime("%Y-%m-%d")
    report = {
        "date": day,
        "checked_at": _now().isoformat(timespec="seconds"),
        "morning": morning_status(day),
        "afterclose": afterclose_status(day),
        "finance": finance_status(),
        "windows_jobs": windows_jobs(day),
        "account_db": account_db_status(),
    }
    report["problems"] = [f"{k}: {p}" for k in ("morning", "afterclose", "finance", "account_db") for p in report[k]["problems"]]
    report["healthy"] = not report["problems"]
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="研究流程健康狀態")
    ap.add_argument("--date", default="")
    ns = ap.parse_args()
    print(json.dumps(pipeline_status(ns.date or None), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
