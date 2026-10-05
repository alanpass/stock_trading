# -*- coding: utf-8 -*-
"""Windows Task Scheduler 的統一執行器 v9.6。

所有排程都由這裡進入，避免 Task Scheduler 只顯示模糊的 0/1。
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

TAIPEI = ZoneInfo("Asia/Taipei")
BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "output" / "scheduler_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {
    "morning": "run_morning_report.py",
    "afterclose": "run_after_close_research.py",
    "finance": "run_finance_info_update.py",
    "cnyes": "run_cnyes_news_nightly.py",
}


def today() -> str:
    return datetime.now(TAIPEI).date().isoformat()


def log_path(job: str) -> Path:
    return LOG_DIR / f"{job}_{today()}.log"


def write_line(handle, text: str) -> None:
    handle.write(str(text).rstrip() + "\n")
    handle.flush()


def status_path(job: str) -> Path:
    return LOG_DIR / f"{job}_status_{today()}.json"


def write_status(job: str, status: str, **extra: object) -> None:
    payload = {
        "job": job,
        "status": status,
        "updated_at": datetime.now(TAIPEI).isoformat(),
        "pid": os.getpid(),
        **extra,
    }
    try:
        status_path(job).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def validate_output(job: str) -> tuple[bool, str]:
    d = today()
    research_dir = BASE / "output" / "research_reports"

    if job == "cnyes":
        path = research_dir / f"cnyes_news_{d}.json"
        if not path.exists():
            return False, f"CNYES output missing: {path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            count = int(data.get("article_count", len(data.get("articles", []) or [])) or 0)
            if count <= 0:
                return False, f"CNYES output has no articles: {path}"
            return True, f"CNYES output OK: {count} articles"
        except Exception as exc:
            return False, f"CNYES output invalid: {exc}"

    if job == "morning":
        path = research_dir / f"morning_{d}.json"
        if not path.exists():
            return False, f"Morning report missing: {path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            morning = data.get("morning_agent", {}) or {}
            cache_count = int((data.get("cnyes_news", {}) or {}).get("article_count", 0) or 0)
            news_count = len(morning.get("news_summary", []) or [])
            if cache_count > 0 and news_count == 0:
                return False, "Morning report exists but CNYES cache had articles and news_summary is empty"
            return True, f"Morning report OK: CNYES cache={cache_count}, summaries={news_count}"
        except Exception as exc:
            return False, f"Morning report invalid: {exc}"

    if job == "finance":
        path = research_dir / "finance_info_latest.json"
        if not path.exists():
            return False, f"Finance info cache missing: {path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            updated = str(data.get("updated_at", ""))
            if not updated:
                return False, "Finance info cache has no updated_at"
            return True, (
                f"Finance info OK: earnings={int(data.get('earnings_count', 0) or 0)}, "
                f"news={int(data.get('news_count', 0) or 0)}, updated={updated}"
            )
        except Exception as exc:
            return False, f"Finance info invalid: {exc}"

    if job == "afterclose":
        status_candidates = [
            LOG_DIR / f"after_close_status_{d}.json",
            LOG_DIR / f"afterclose_status_{d}.json",
        ]
        for status_file in status_candidates:
            if status_file.exists():
                try:
                    data = json.loads(status_file.read_text(encoding="utf-8"))
                    state = str(data.get("status", ""))
                    if state == "skipped_non_trading_day":
                        return True, f"After-close status OK: {state}"
                    if state == "success":
                        if bool(data.get("email_sent")):
                            return True, "After-close status OK: success + email_sent=true"
                        return False, "After-close report completed but email_sent is not true"
                    return False, f"After-close status = {state or 'unknown'}"
                except Exception as exc:
                    return False, f"After-close status invalid: {exc}"
        return False, "After-close status missing"

    return False, f"Unknown job: {job}"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1].lower() not in JOBS:
        print("Usage: python scheduled_job_runner.py morning|afterclose|finance|cnyes")
        return 2

    job = sys.argv[1].lower()
    script = BASE / JOBS[job]
    if not script.exists():
        print(f"ERROR: script not found: {script}")
        write_status(job, "script_missing", script=str(script))
        return 2

    lp = log_path(job)
    started = datetime.now(TAIPEI).isoformat(timespec="seconds")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TZ"] = "Asia/Taipei"

    write_status(job, "running", script=str(script), python=sys.executable)
    with lp.open("a", encoding="utf-8", errors="replace") as log:
        write_line(log, "=" * 72)
        write_line(log, f"JOB={job}")
        write_line(log, f"START={started}")
        write_line(log, f"SCRIPT={script}")
        write_line(log, f"PYTHON={sys.executable}")
        write_line(log, f"WORKDIR={BASE}")
        write_line(log, "=" * 72)
        try:
            proc = subprocess.run(
                [sys.executable, "-u", str(script)],
                cwd=str(BASE),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            rc = int(proc.returncode)
            write_line(log, f"CHILD_RETURN_CODE={rc}")
        except Exception as exc:
            write_line(log, f"RUNNER_EXCEPTION={type(exc).__name__}: {exc}")
            write_line(log, traceback.format_exc())
            write_status(job, "runner_exception", error=str(exc))
            return 2

        if rc != 0:
            write_status(job, "child_failed", child_return_code=rc)
            write_line(log, f"FINAL_RETURN_CODE={rc}")
            write_line(log, f"END={datetime.now(TAIPEI).isoformat(timespec='seconds')}")
            return rc

        ok, message = validate_output(job)
        write_line(log, f"OUTPUT_VALIDATION={ok}")
        write_line(log, f"OUTPUT_MESSAGE={message}")
        if not ok:
            write_status(job, "output_validation_failed", message=message)
            write_line(log, "FINAL=FAILED_OUTPUT_VALIDATION")
            write_line(log, f"END={datetime.now(TAIPEI).isoformat(timespec='seconds')}")
            return 6

        write_status(job, "success", message=message)
        write_line(log, "FINAL_RETURN_CODE=0")
        write_line(log, f"END={datetime.now(TAIPEI).isoformat(timespec='seconds')}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
