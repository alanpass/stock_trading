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
    "morning": "run_morning_report.py",              # 08:30 晨報
    "afterclose": "run_after_close_research.py",     # 14:30 盤後分析報導
    "finance": "run_finance_info_update.py",         # 08:10 11:00 13:30 16:00 18:00 23:00 新聞／法說會爬取 + Agent 摘要
    "cnyes": "run_finance_info_update.py",           # 舊排程名稱（相容）：同樣走財經資訊更新
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

    if job == "morning":
        path = research_dir / f"morning_{d}.json"
        if not path.exists():
            return False, f"Morning report missing: {path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            morning = data.get("morning_agent", {}) or {}
            cache_count = int((data.get("cnyes_news", {}) or {}).get("article_count", 0) or 0)
            news_count = len(morning.get("news_summary", []) or [])
            earnings_count = len(data.get("earnings_calls", []) or [])
            overview = str(morning.get("overview") or "").strip()
            if not overview and news_count == 0 and earnings_count == 0:
                return False, "Morning report exists but contains no usable overview/news/earnings content"
            return True, f"Morning report OK: CNYES cache={cache_count}, summaries={news_count}, earnings={earnings_count}"
        except Exception as exc:
            return False, f"Morning report invalid: {exc}"


    if job in {"finance", "cnyes"}:
        path = research_dir / "finance_info_latest.json"
        if not path.exists():
            return False, f"Finance info output missing: {path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            earnings_count = int(data.get("earnings_count", 0) or 0)
            news_count = int(data.get("news_count", 0) or 0)
            if earnings_count <= 0 and news_count <= 0:
                return False, "Finance info exists but both earnings_count/news_count are zero"
            updated = datetime.fromisoformat(str(data.get("updated_at")))
            age_min = (datetime.now(TAIPEI) - updated).total_seconds() / 60
            if age_min > 30:
                return False, f"Finance info was not refreshed by this run (age={age_min:.0f} min)"
            if data.get("stale"):
                return False, f"Finance info is STALE (crawl returned nothing; kept previous data). news={news_count}"
            return True, f"Finance info OK: news={news_count}, earnings={earnings_count}, crawl={data.get('crawl', {}).get('per_category_count', {})}"
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
                    if state in {"success", "skipped_non_trading_day"}:
                        return True, f"After-close status OK: {state}"
                    return False, f"After-close status = {state or 'unknown'}"
                except Exception as exc:
                    return False, f"After-close status invalid: {exc}"
        return False, "After-close status missing"

    return False, f"Unknown job: {job}"


def _finance_cache_age_minutes() -> float | None:
    path = BASE / "output" / "research_reports" / "finance_info_latest.json"
    try:
        updated = datetime.fromisoformat(str(json.loads(path.read_text(encoding="utf-8")).get("updated_at")))
        return (datetime.now(TAIPEI) - updated).total_seconds() / 60
    except Exception:
        return None


def main() -> int:
    args = [a for a in sys.argv[1:]]
    stale_minutes = None
    if "--if-stale-minutes" in args:
        i = args.index("--if-stale-minutes")
        try:
            stale_minutes = float(args[i + 1])
        except Exception:
            stale_minutes = 90.0
        del args[i:i + 2]
    if len(args) != 1 or args[0].lower() not in JOBS:
        print("Usage: python scheduled_job_runner.py morning|afterclose|finance [--if-stale-minutes N]")
        return 2

    job = args[0].lower()
    # 開機／登入後補跑：資料還新鮮就不重爬（避免和整點排程重複）
    if stale_minutes is not None and job in {"finance", "cnyes"}:
        age = _finance_cache_age_minutes()
        if age is not None and age <= stale_minutes:
            print(f"財經資訊 {age:.0f} 分鐘前才更新過（門檻 {stale_minutes:.0f} 分鐘），略過補跑。")
            return 0
        print(f"財經資訊快取年齡={age}，開始補跑。")
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
            # 即時把子程式輸出同時印在終端機和 log（原本只寫進 log，終端機看起來像當掉）
            import threading
            proc = subprocess.Popen(
                [sys.executable, "-u", str(script)],
                cwd=str(BASE),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            timeout_sec = int(os.getenv("SCHEDULED_JOB_TIMEOUT", "7200"))
            timed_out = {"v": False}

            def _kill() -> None:
                timed_out["v"] = True
                proc.kill()

            timer = threading.Timer(timeout_sec, _kill)
            timer.start()
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    log.write(line)
                    log.flush()
                    try:
                        print(line, end="", flush=True)
                    except Exception:
                        pass
                proc.wait()
            finally:
                timer.cancel()
            rc = int(proc.returncode)
            if timed_out["v"]:
                write_line(log, f"CHILD_TIMEOUT after {timeout_sec}s")
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

        # 財經資料驗證成功後，自動發布精簡 JSON 到 GitHub。
        # 只有發布成功才回報整條流程成功，避免誤以為網站已同步。
        if job in {"finance", "cnyes"}:
            publisher = BASE / "publish_research_data.py"
            if not publisher.exists():
                message = f"Finance publisher missing: {publisher}"
                write_status(job, "publish_failed", message=message)
                write_line(log, f"PUBLISH_FAILED={message}")
                write_line(log, "FINAL_RETURN_CODE=7")
                write_line(log, f"END={datetime.now(TAIPEI).isoformat(timespec='seconds')}")
                return 7

            write_line(log, "PUBLISH_START=publish_research_data.py")
            try:
                publish_timeout = int(os.getenv("GITHUB_PUBLISH_TIMEOUT", "180"))
                publish_proc = subprocess.run(
                    [sys.executable, "-u", str(publisher)],
                    cwd=str(BASE),
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=publish_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                message = f"GitHub publish timed out after {publish_timeout}s"
                write_status(job, "publish_failed", message=message)
                write_line(log, f"PUBLISH_FAILED={message}")
                write_line(log, "FINAL_RETURN_CODE=7")
                write_line(log, f"END={datetime.now(TAIPEI).isoformat(timespec='seconds')}")
                return 7
            except Exception as exc:
                message = f"GitHub publish exception: {type(exc).__name__}: {exc}"
                write_status(job, "publish_failed", message=message)
                write_line(log, f"PUBLISH_FAILED={message}")
                write_line(log, "FINAL_RETURN_CODE=7")
                write_line(log, f"END={datetime.now(TAIPEI).isoformat(timespec='seconds')}")
                return 7

            for output_line in (publish_proc.stdout or "").splitlines():
                write_line(log, f"PUBLISH_STDOUT: {output_line}")
                print(f"[PUBLISH] {output_line}", flush=True)
            for error_line in (publish_proc.stderr or "").splitlines():
                write_line(log, f"PUBLISH_STDERR: {error_line}")
                print(f"[PUBLISH-ERR] {error_line}", flush=True)

            if publish_proc.returncode != 0:
                message = f"GitHub publisher exit code {publish_proc.returncode}"
                write_status(
                    job, "publish_failed",
                    message=message,
                    publish_return_code=publish_proc.returncode,
                )
                write_line(log, f"PUBLISH_FAILED={message}")
                write_line(log, "FINAL_RETURN_CODE=7")
                write_line(log, f"END={datetime.now(TAIPEI).isoformat(timespec='seconds')}")
                return 7

            write_line(log, "PUBLISH_SUCCESS=True")
            write_status(job, "success", message=message, published=True)
        else:
            write_status(job, "success", message=message)
        write_line(log, "FINAL_RETURN_CODE=0")
        write_line(log, f"END={datetime.now(TAIPEI).isoformat(timespec='seconds')}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
