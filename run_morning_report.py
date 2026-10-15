# -*- coding: utf-8 -*-
"""AI 台股早報排程入口 v3（Claude 研究 Agent）。

  python run_morning_report.py --prepare             # Windows 排程 08:30：建立研究包
  python run_morning_report.py --finalize [--dry-run]  # Claude 寫完 morning_notes.json 後：驗證、保存、寄信

1. MorningResearchAgent 建立研究包，Claude（morning-report skill）撰寫早報內容。
2. 驗證 CNYES 夜間快取與新聞摘要。
3. 使用 EmailAgent 寄送早報。
4. 所有例外都寫入 scheduler_logs/morning_entry_YYYY-MM-DD.log。
5. 以明確的 exit code 回報 Task Scheduler，而不是讓例外直接變成模糊的 exit 1。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from morning_report_agent import MorningResearchAgent, save_morning_report
from ai_backend import AGENT_LABEL, AGENT_MODEL, AGENT_STATUS
from quality_checks import quality_gate, validate_morning
from email_agent import EmailAgent

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "output" / "scheduler_logs"
REPORT_DIR = BASE / "output" / "research_reports"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TAIPEI = ZoneInfo("Asia/Taipei")
DEFAULT_WATCHLIST = [
    "3481", "2327", "2492", "3037", "3044", "3533", "2303", "2330", "2344", "2408",
    "6515", "7769", "6488", "3374", "2377", "3450", "0050", "3006", "3661", "8299",
    "2308", "6274",
]


def now_taipei() -> datetime:
    return datetime.now(TAIPEI)


# 狀態與 log 以「報告日期」命名；--finalize --date 指定其他日期時，不會寫到今天的狀態檔。
REPORT_DATE = ""


def _report_day() -> str:
    return REPORT_DATE or now_taipei().date().isoformat()


def log_file() -> Path:
    return LOG_DIR / f"morning_entry_{_report_day()}.log"


def write_log(message: str) -> None:
    line = f"[{now_taipei().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with log_file().open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(line + "\n")


def write_status(status: str, **extra: object) -> None:
    path = LOG_DIR / f"morning_entry_status_{_report_day()}.json"
    payload = {
        "status": status,
        "updated_at": now_taipei().isoformat(),
        **extra,
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def load_watchlist() -> list[str]:
    path = BASE / "user_watchlist.json"
    if not path.exists():
        return DEFAULT_WATCHLIST.copy()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            out = []
            for value in raw:
                code = str(value).strip().upper().replace(".TW", "")
                if code and code not in out:
                    out.append(code)
            if out:
                return out
    except Exception as exc:
        write_log(f"讀取 user_watchlist.json 失敗，改用預設自選股：{exc}")
    return DEFAULT_WATCHLIST.copy()


def validate_report(report: dict, expected_date: str = "") -> tuple[bool, str]:
    date_text = str(report.get("report_date", ""))[:10]
    today = expected_date or now_taipei().date().isoformat()
    if date_text != today:
        return False, f"報告日期錯誤：report_date={date_text!r}, today={today!r}"

    cnyes = report.get("cnyes_news", {}) or {}
    article_count = int(cnyes.get("article_count", 0) or 0)
    morning = report.get("morning_agent", {}) or {}
    news_count = len(morning.get("news_summary", []) or [])

    earnings_count = len(report.get("earnings_calls", []) or [])
    overview = str(morning.get("overview") or "").strip()

    # CNYES 原始快取有新聞，但 Agent 摘要可能不完整；
    # 不應因為這個單一欄位為 0 而整份晨報不寄。
    # 真正需要阻止寄送的條件是：整份晨報完全沒有可讀內容。
    if not overview and news_count == 0 and earnings_count == 0:
        return False, "早報沒有任何可閱讀的 overview/news/earnings 內容"

    if not report.get("json_path") or not Path(str(report["json_path"])).exists():
        return False, "早報 JSON 路徑不存在"

    return True, f"report_date={date_text}, CNYES={article_count}, news_summary={news_count}"


def refresh_finance_cache() -> None:
    # 財經資訊由 08:10 的排程先更新；晨報（08:30）若發現資料還很新就直接使用，不重爬，
    # 避免兩個工作同時寫入同一份快取。資料過舊（例如 08:10 排程沒跑）才補更新一次。
    write_log("[0/2] 檢查財經資訊快取")
    fresh_minutes = int(os.getenv("MORNING_FINANCE_FRESH_MINUTES", "90"))
    latest_finance = BASE / "output" / "research_reports" / "finance_info_latest.json"
    cache_age_min = None
    try:
        updated = json.loads(latest_finance.read_text(encoding="utf-8")).get("updated_at", "")
        cache_age_min = (datetime.now(TAIPEI) - datetime.fromisoformat(updated)).total_seconds() / 60
    except Exception:
        pass
    finance_script = BASE / "run_finance_info_update.py"
    if cache_age_min is not None and cache_age_min <= fresh_minutes:
        write_log(f"FINANCE_CACHE_FRESH age={cache_age_min:.0f} 分鐘，直接使用")
    elif finance_script.exists():
        write_log(f"FINANCE_CACHE_STALE age={cache_age_min}，補更新一次")
        try:
            proc = subprocess.run(
                [sys.executable, "-u", str(finance_script)],
                cwd=str(BASE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=int(os.getenv("MORNING_FINANCE_TIMEOUT", "1500")),
                check=False,
            )
            write_log(f"FINANCE_UPDATE_RC={proc.returncode}")
            if proc.stdout:
                write_log(proc.stdout[-10000:])
        except Exception as exc:
            # 財經快取失敗不阻斷晨報，後續仍可使用既有快取與 fallback。
            write_log(f"FINANCE_UPDATE_WARNING={type(exc).__name__}: {exc}")
    else:
        write_log("FINANCE_UPDATE=script_missing，沿用既有快取")


def send_morning(report: dict, expected_date: str = "") -> int:
    ok, message = validate_report(report, expected_date)
    write_log(f"REPORT_VALIDATION={ok}")
    write_log(f"REPORT_VALIDATION_MESSAGE={message}")
    if not ok:
        write_status("report_validation_failed", message=message)
        return 4

    morning = report.get("morning_agent", {}) or {}
    write_log(f"MORNING_AGENT_STATUS={morning.get('agent_status', '')}")
    write_log(f"CNYES_CACHE_DATES={report.get('cnyes_news', {}).get('cache_dates', [])}")
    write_log(f"CNYES_ARTICLES={report.get('cnyes_news', {}).get('article_count', 0)}")
    write_log(f"NEWS_SUMMARY_COUNT={len(morning.get('news_summary', []) or [])}")
    write_log(f"RECOMMENDED_INDUSTRIES={len(morning.get('recommended_industries', []) or [])}")
    write_log(f"RECOMMENDED_STOCKS={len(morning.get('recommended_stocks', []) or [])}")

    write_log("[2/2] Email Agent")
    email = EmailAgent(BASE)
    status = email.status()
    write_log(
        f"EMAIL_CONFIGURED={status.get('configured')} "
        f"AUTO_SEND={status.get('auto_send')} "
        f"RECIPIENT={status.get('recipient')}"
    )

    if not status.get("configured"):
        message = f"Email 尚未完成設定：{status.get('missing', [])}"
        write_log(message)
        write_status("email_not_configured", message=message)
        return 3

    if not email.config.auto_send:
        message = "EMAIL_AUTO_SEND=false，早報已建立但沒有寄送；排程視為失敗以避免誤判為已寄出。"
        write_log(message)
        write_status("email_disabled", message=message)
        return 3

    # 直接寄送本次剛建立的 report，避免 send_latest_morning_report 讀到上一份快取。
    if hasattr(email, "send_report"):
        write_log("EMAIL_METHOD=send_report(fresh_report)")
        result = email.send_report(report, force=True)
    else:
        write_log("EMAIL_METHOD=send_latest_morning_report")
        result = email.send_latest_morning_report(force=True)

    if not isinstance(result, dict):
        result = {"sent": bool(result), "raw_result": result}

    write_log("EMAIL_RESULT=" + json.dumps(result, ensure_ascii=False, default=str))
    if result.get("sent"):
        write_status("success", report_path=report.get("json_path"), email=result)
        write_log("AI TW STOCK MORNING REPORT COMPLETED")
        return 0

    # 一次 SMTP 瞬斷可重試 1 次；第二次仍失敗才讓排程判定失敗。
    if not result.get("skipped"):
        write_log("EMAIL_RETRY=1")
        retry = email.send_report(report, force=True) if hasattr(email, "send_report") else email.send_latest_morning_report(force=True)
        if not isinstance(retry, dict):
            retry = {"sent": bool(retry), "raw_result": retry}
        write_log("EMAIL_RETRY_RESULT=" + json.dumps(retry, ensure_ascii=False, default=str))
        if retry.get("sent"):
            write_status("success", report_path=report.get("json_path"), email=retry)
            write_log("AI TW STOCK MORNING REPORT COMPLETED")
            return 0
        result = retry

    message = str(result.get("error") or result.get("reason") or "Email 未成功寄出")
    write_log(f"EMAIL_FAILED={message}")
    write_status("email_failed", message=message, email=result)
    return 3


# ----------------------------------------------------------------------------
# Claude 模式：prepare → Claude 撰寫早報 → finalize
# ----------------------------------------------------------------------------
def job_dir(report_date: str) -> Path:
    return BASE / "output" / "claude_jobs" / f"morning_{report_date}"


def prepare(watchlist: list[str]) -> int:
    refresh_finance_cache()
    write_log("[1/2] 建立早報研究包（等待 Claude 撰寫）")
    agent = MorningResearchAgent(BASE)
    payload = agent._build_payload(watchlist)
    d = job_dir(str(payload.get("report_date") or now_taipei().date().isoformat()))
    d.mkdir(parents=True, exist_ok=True)
    payload["research_scope"] = {**(payload.get("research_scope", {}) or {}), "watchlist_symbols": watchlist}
    (d / "package.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (d / "brief.md").write_text(agent._prompt(watchlist, payload) + "\n", encoding="utf-8")
    write_status("awaiting_claude", job_dir=str(d))
    write_log(f"PREPARED={d}")
    return 0


def finalize(report_date: str, dry_run: bool = False) -> int:
    d = job_dir(report_date)
    package_path, notes_path = d / "package.json", d / "morning_notes.json"
    for path in (package_path, notes_path):
        if not path.exists():
            write_status("error_missing_claude_input", missing=str(path))
            write_log(f"ERROR: 找不到 {path}")
            return 5
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    try:
        raw = json.loads(notes_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("morning_notes.json 必須是 JSON 物件")
    except Exception as exc:
        write_status("error_invalid_claude_notes", error=str(exc))
        write_log(f"ERROR: morning_notes.json 無法解析：{exc}")
        return 5
    blocked = quality_gate(validate_morning(report_date), lambda c: write_status("error_quality_check", errors=c["errors"][:30]))
    if blocked:
        return blocked
    watchlist = (payload.get("research_scope", {}) or {}).get("watchlist_symbols") or load_watchlist()
    agent = MorningResearchAgent(BASE)
    try:
        morning = agent._normalize_raw(raw, payload, watchlist)
    except Exception as exc:
        write_status("error_invalid_claude_notes", error=str(exc))
        write_log(f"ERROR: 早報內容驗證失敗：{exc}")
        return 5
    morning["agent_status"] = AGENT_STATUS
    morning["model"] = AGENT_MODEL
    morning["generated_at"] = str(raw.get("generated_at") or now_taipei().isoformat())
    if dry_run:
        # 測試模式只寫進 job 資料夾，不覆蓋正式的 morning_<date>.json。
        agent.output_dir = d
    report = save_morning_report(agent, payload, morning)
    if dry_run:
        (d / "email_preview.html").write_text(EmailAgent(BASE).build_morning_html(report), encoding="utf-8")
        ok, message = validate_report(report, report_date)
        print(f"DRY_RUN_{'OK' if ok else 'FAILED'} {message}")
        print(f"EMAIL_PREVIEW={d / 'email_preview.html'}")
        return 0 if ok else 4
    return send_morning(report, report_date)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="AI 台股早報")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="只建立研究包，等待 Claude 撰寫早報")
    mode.add_argument("--finalize", action="store_true", help="讀入 Claude 早報後保存並寄信")
    parser.add_argument("--date", default="", help="finalize 的報告日期 YYYY-MM-DD（預設今天）")
    parser.add_argument("--dry-run", action="store_true", help="finalize 測試：只產生 Email 預覽，不寄信")
    args = parser.parse_args(argv)
    global REPORT_DATE
    REPORT_DATE = args.date if args.finalize and args.date else ""

    write_status("starting", pid=os.getpid(), python=sys.executable)
    write_log("=" * 72)
    write_log("AI TW STOCK MORNING REPORT")
    write_log(f"BASE={BASE}")
    write_log(f"PYTHON={sys.executable}")
    write_log(f"AI_AGENT={AGENT_LABEL}")
    try:
        if args.finalize:
            return finalize(args.date or now_taipei().date().isoformat(), dry_run=args.dry_run)
        watchlist = load_watchlist()
        write_log(f"WATCHLIST_COUNT={len(watchlist)}")
        # 沒有參數時（Windows 排程）等同 --prepare。
        return prepare(watchlist)
    except Exception as exc:
        detail = traceback.format_exc()
        write_log(f"UNHANDLED_EXCEPTION={type(exc).__name__}: {exc}")
        write_log(detail)
        write_status("exception", error=f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
