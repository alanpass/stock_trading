# -*- coding: utf-8 -*-
"""AI 台股早報排程入口 v2。

這個入口專門給 Windows Task Scheduler 使用：
1. 執行 MorningResearchAgent 建立當日早報 JSON/Markdown。
2. 驗證 CNYES 夜間快取與新聞摘要。
3. 使用 EmailAgent 寄送剛產生的早報。
4. 所有例外都寫入 scheduler_logs/morning_entry_YYYY-MM-DD.log。
5. 以明確的 exit code 回報 Task Scheduler，而不是讓例外直接變成模糊的 exit 1。
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from morning_report_agent import build_morning_report
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


def log_file() -> Path:
    return LOG_DIR / f"morning_entry_{now_taipei().date().isoformat()}.log"


def write_log(message: str) -> None:
    line = f"[{now_taipei().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with log_file().open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(line + "\n")


def write_status(status: str, **extra: object) -> None:
    path = LOG_DIR / f"morning_entry_status_{now_taipei().date().isoformat()}.json"
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


def validate_report(report: dict) -> tuple[bool, str]:
    date_text = str(report.get("report_date", ""))[:10]
    today = now_taipei().date().isoformat()
    if date_text != today:
        return False, f"報告日期錯誤：report_date={date_text!r}, today={today!r}"

    cnyes = report.get("cnyes_news", {}) or {}
    article_count = int(cnyes.get("article_count", 0) or 0)
    morning = report.get("morning_agent", {}) or {}
    news_count = len(morning.get("news_summary", []) or [])

    if article_count > 0 and news_count == 0:
        return False, f"CNYES 已有 {article_count} 篇新聞，但早報 news_summary=0"

    if not report.get("json_path") or not Path(str(report["json_path"])).exists():
        return False, "早報 JSON 路徑不存在"

    return True, f"report_date={date_text}, CNYES={article_count}, news_summary={news_count}"


def main() -> int:
    write_status("starting", pid=os.getpid(), python=sys.executable)
    write_log("=" * 72)
    write_log("AI TW STOCK MORNING REPORT")
    write_log(f"BASE={BASE}")
    write_log(f"PYTHON={sys.executable}")
    write_log(f"OLLAMA_HOST={os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')}")
    write_log(f"OLLAMA_MODEL={os.getenv('OLLAMA_MODEL', 'qwen3:8b')}")

    try:
        watchlist = load_watchlist()
        write_log(f"WATCHLIST_COUNT={len(watchlist)}")
        write_log("[1/2] 建立早報研究資料")
        report = build_morning_report(BASE, watchlist=watchlist)
        ok, message = validate_report(report)
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
            message = "EMAIL_AUTO_SEND=false，早報已建立但沒有寄送。"
            write_log(message)
            write_status("email_disabled", message=message)
            return 0

        # 相容不同版本的 EmailAgent：
        # 1. 新版優先使用 send_latest_morning_report()。
        # 2. 舊版若沒有該方法，直接把本次剛建立的 report 傳給 send_report()。
        #    這樣不會因為 EmailAgent 版本落後而讓整個早報排程失敗。
        if hasattr(email, "send_latest_morning_report"):
            write_log("EMAIL_METHOD=send_latest_morning_report")
            result = email.send_latest_morning_report(force=True)
        elif hasattr(email, "send_report"):
            write_log("EMAIL_METHOD=send_report(fresh_report)")
            result = email.send_report(report, force=True)
        else:
            result = {
                "sent": False,
                "error": "目前的 EmailAgent 沒有 send_latest_morning_report() 或 send_report()。",
            }

        if not isinstance(result, dict):
            result = {"sent": bool(result), "raw_result": result}

        write_log("EMAIL_RESULT=" + json.dumps(result, ensure_ascii=False, default=str))
        if result.get("sent"):
            write_status("success", report_path=report.get("json_path"), email=result)
            write_log("AI TW STOCK MORNING REPORT COMPLETED")
            return 0

        message = str(result.get("error") or result.get("reason") or "Email 未成功寄出")
        write_log(f"EMAIL_FAILED={message}")
        write_status("email_failed", message=message, email=result)
        return 3

    except Exception as exc:
        detail = traceback.format_exc()
        write_log(f"UNHANDLED_EXCEPTION={type(exc).__name__}: {exc}")
        write_log(detail)
        write_status("exception", error=f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
