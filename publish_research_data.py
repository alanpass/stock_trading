# -*- coding: utf-8 -*-
"""
AI 台股研究中心｜網站財經資料發布器

用途：
    讀取 Windows 本機 Qwen3 / CNYES / Fugle 產生的完整財經研究快取，
    轉成「只給網站使用」的精簡 JSON，再只提交這個公開資料檔到 GitHub。

安全設計：
    1. 不提交 .env、API Key、Ollama 位址或本機快取。
    2. 不提交完整新聞正文，網站只拿到標題、摘要、AI 重點、情緒、產業、個股與原文網址。
    3. 只 git add 白名單檔案：output/research_reports/finance_info_public.json
    4. 使用既有 Git Credential Manager / SSH 認證，不在程式內保存 GitHub Token。

典型流程：
    run_finance_info_update.py
        -> output/research_reports/finance_info_latest.json
        -> publish_research_data.py
        -> Git commit / push
        -> Streamlit Community Cloud 自動更新
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")
BASE = Path(__file__).resolve().parent
REPORT_DIR = BASE / "output" / "research_reports"
SOURCE_PATH = REPORT_DIR / "finance_info_latest.json"
PUBLIC_PATH = REPORT_DIR / "finance_info_public.json"
LOCK_PATH = REPORT_DIR / ".finance_publish.lock"


def now_taipei() -> datetime:
    return datetime.now(TAIPEI)


def log(message: str) -> None:
    print(f"[{now_taipei().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run_git(args: list[str], *, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    cmd = ["git", *args]
    result = subprocess.run(
        cmd,
        cwd=BASE,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Git 指令失敗：git {' '.join(args)}\n{detail}")
    return result


def repo_root() -> Path:
    result = run_git(["rev-parse", "--show-toplevel"])
    root = Path(result.stdout.strip()).resolve()
    if root != BASE.resolve():
        # 專案目前預期 repository root 就是程式所在目錄。
        # 若在子目錄執行，仍允許，但輸出路徑必須位於 repo 內。
        return root
    return root


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到來源資料：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"JSON 讀取失敗：{path} / {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON 格式錯誤（必須是 object）：{path}")
    return data


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any, limit: int = 2000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _clean_public_news(item: dict[str, Any]) -> dict[str, Any]:
    """保留網站實際需要的欄位，刻意移除完整正文與爬蟲內部欄位。"""
    return {
        "article_id": _safe_text(item.get("article_id"), 80),
        "title": _safe_text(item.get("title") or item.get("headline"), 300),
        "url": _safe_text(item.get("url") or item.get("source_url"), 1000),
        "category": _safe_text(item.get("category"), 80),
        "published": _safe_text(item.get("published") or item.get("published_at"), 80),
        "published_ts": _safe_text(item.get("published_ts") or item.get("published_at"), 80),
        "ai_points": [_safe_text(x, 300) for x in _safe_list(item.get("ai_points")) if _safe_text(x, 300)],
        "summary": _safe_text(item.get("summary") or item.get("ai_summary"), 1200),
        "sentiment": _safe_text(item.get("sentiment"), 30),
        "sectors": [_safe_text(x, 50) for x in _safe_list(item.get("sectors")) if _safe_text(x, 50)],
        "stocks": [_safe_text(x, 30) for x in _safe_list(item.get("stocks")) if _safe_text(x, 30)],
        "highlights": [_safe_text(x, 80) for x in _safe_list(item.get("highlights")) if _safe_text(x, 80)],
        "why": _safe_text(item.get("why"), 160),
        "importance": item.get("importance"),
        "ai_source": _safe_text(item.get("ai_source"), 30),
        "source": _safe_text(item.get("source"), 80),
    }


def _clean_public_earnings(item: dict[str, Any]) -> dict[str, Any]:
    """保留法說會摘要欄位，移除 memo 正文。"""
    return {
        "title": _safe_text(item.get("title") or item.get("event_title"), 300),
        "symbol": _safe_text(item.get("symbol") or item.get("code") or item.get("stock_code"), 30),
        "name": _safe_text(item.get("name") or item.get("company") or item.get("company_name"), 120),
        "event_date": _safe_text(item.get("event_date") or item.get("published_date") or item.get("date"), 80),
        "published_at": _safe_text(item.get("published_at") or item.get("published_time") or item.get("published_ts") or item.get("published"), 80),
        "ai_points": [_safe_text(x, 300) for x in _safe_list(item.get("ai_points")) if _safe_text(x, 300)],
        "summary": _safe_text(item.get("one_line_summary") or item.get("summary") or item.get("ai_summary"), 1200),
        "sentiment": _safe_text(item.get("sentiment"), 30),
        "impact": _safe_text(item.get("impact"), 160),
        "signal": _safe_text(item.get("signal"), 160),
        "judgement": _safe_text(item.get("judgement"), 160),
        "sectors": [_safe_text(x, 50) for x in _safe_list(item.get("sectors")) if _safe_text(x, 50)],
        "highlights": [_safe_text(x, 80) for x in _safe_list(item.get("highlights")) if _safe_text(x, 80)],
        "source_url": _safe_text(item.get("source_url") or item.get("url"), 1000),
    }


def build_public_payload(source: dict[str, Any]) -> dict[str, Any]:
    """把本機完整研究結果轉成網站可發布版本。"""
    news = [_clean_public_news(x) for x in _safe_list(source.get("news")) if isinstance(x, dict)]
    earnings = [_clean_public_earnings(x) for x in _safe_list(source.get("earnings")) if isinstance(x, dict)]

    payload = {
        "schema_version": 1,
        "report_type": "finance_info_public",
        "report_date": _safe_text(source.get("report_date"), 30),
        "updated_at": _safe_text(source.get("updated_at"), 80),
        "finished_at": _safe_text(source.get("finished_at"), 80),
        "lookback_days": int(source.get("lookback_days", 2) or 2),
        "schedule": _safe_list(source.get("schedule")),
        "stale": bool(source.get("stale", False)),
        "news_count": len(news),
        "earnings_count": len(earnings),
        "news_digest": source.get("news_digest") if isinstance(source.get("news_digest"), dict) else {},
        "earnings_digest": source.get("earnings_digest") if isinstance(source.get("earnings_digest"), dict) else {},
        "crawl": source.get("crawl") if isinstance(source.get("crawl"), dict) else {},
        "agent_stats": source.get("agent_stats") if isinstance(source.get("agent_stats"), dict) else {},
        "errors_count": len(_safe_list(source.get("errors"))),
        "news": news,
        "earnings": earnings,
        "published_at": now_taipei().isoformat(timespec="seconds"),
    }
    return payload


def write_public_payload(payload: dict[str, Any]) -> bool:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    new_text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    old_text = PUBLIC_PATH.read_text(encoding="utf-8") if PUBLIC_PATH.exists() else ""
    if old_text == new_text:
        log("網站公開資料內容沒有變化，不重新寫檔。")
        return False
    tmp = PUBLIC_PATH.with_suffix(PUBLIC_PATH.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, PUBLIC_PATH)
    log(f"已建立網站公開資料：{PUBLIC_PATH}")
    log(f"公開資料大小：{PUBLIC_PATH.stat().st_size / 1024:.1f} KB")
    log(f"新聞={payload['news_count']}、法說會={payload['earnings_count']}、stale={payload['stale']}")
    return True


def current_branch() -> str:
    branch = run_git(["symbolic-ref", "--quiet", "--short", "HEAD"]).stdout.strip()
    if not branch:
        raise RuntimeError("目前 Git 位於 detached HEAD，無法自動發布。請切回 main/master 或你的工作分支。")
    return branch


def ensure_clean_staging_scope() -> None:
    """發布前只允許白名單檔案進入 index。"""
    # 任何既有 staged 變更都可能被一起 commit，因此直接拒絕，避免誤提交使用者其他修改。
    staged = run_git(["diff", "--cached", "--name-only"]).stdout.splitlines()
    allowed = {str(PUBLIC_PATH.relative_to(BASE.resolve())).replace("\\", "/")}
    unexpected = [x for x in staged if x.strip() and x.strip() not in allowed]
    if unexpected:
        raise RuntimeError(
            "發現其他已經 staged 的檔案，為避免自動程式誤提交而停止：\n" + "\n".join(unexpected)
        )


def git_publish(*, dry_run: bool = False) -> dict[str, Any]:
    public_rel = str(PUBLIC_PATH.relative_to(repo_root())).replace("\\", "/")
    ensure_clean_staging_scope()

    run_git(["add", "--", public_rel])
    staged_diff = run_git(["diff", "--cached", "--name-only"]).stdout.splitlines()
    if public_rel not in staged_diff:
        log("GitHub 不需要更新：公開資料與 repository 版本相同。")
        return {"ok": True, "changed": False, "published": False, "reason": "no_change"}

    if dry_run:
        run_git(["reset", "--", public_rel], check=True)
        log("DRY RUN：資料已驗證，但未 commit / push。")
        return {"ok": True, "changed": True, "published": False, "reason": "dry_run"}

    # 只提交公開資料檔，不會把其他本機修改帶進 commit。
    commit_message = f"chore: update finance research data {now_taipei().strftime('%Y-%m-%d %H:%M')}"
    run_git(["commit", "-m", commit_message], timeout=120)

    branch = current_branch()
    remote = run_git(["remote", "get-url", "origin"], check=False).stdout.strip()
    if not remote:
        raise RuntimeError("找不到 Git remote 'origin'。請先設定 GitHub remote，例如：git remote add origin <repo-url>")
    log(f"GitHub remote：origin（{branch}）")
    push = run_git(["push", "origin", branch], timeout=180)
    out = (push.stdout or "") + (push.stderr or "")
    for line in out.splitlines():
        if line.strip():
            log(line.strip())
    log("✅ GitHub 發布成功。Streamlit Cloud 將依 GitHub repository 更新網站。")
    return {"ok": True, "changed": True, "published": True, "branch": branch, "remote_configured": True}


def acquire_lock() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\nstarted={now_taipei().isoformat()}\n")
    except FileExistsError as exc:
        raise RuntimeError(f"另一個發布程序正在執行：{LOCK_PATH}") from exc


def release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def publish(dry_run: bool = False) -> int:
    acquire_lock()
    try:
        # 確認 repository 與工作目錄設定正常
        root = repo_root()
        log(f"Repository：{root}")
        source = read_json(SOURCE_PATH)
        if not source.get("updated_at"):
            raise RuntimeError("來源財經資料缺少 updated_at，拒絕發布。")
        if source.get("stale"):
            raise RuntimeError("來源資料標記 stale，拒絕把過期研究發布到網站。")
        public = build_public_payload(source)
        write_public_payload(public)
        result = git_publish(dry_run=dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        log(f"❌ 發布失敗：{type(exc).__name__}: {exc}")
        return 7
    finally:
        release_lock()


def main() -> int:
    parser = argparse.ArgumentParser(description="發布財經研究摘要到 GitHub / Streamlit Cloud")
    parser.add_argument("--dry-run", action="store_true", help="只產生公開 JSON 並檢查 Git，不 commit / push")
    args = parser.parse_args()
    return publish(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
