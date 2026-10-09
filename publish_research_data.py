# -*- coding: utf-8 -*-
"""將 Windows 本機財經研究結果安全發布到 GitHub finance-data 分支。

必要環境變數：GITHUB_TOKEN（只需 Contents: write 權限）。
可選：GITHUB_REPOSITORY、FINANCE_DATA_BRANCH。
不需要本機 .git、git remote 或 git push。
"""
from __future__ import annotations
import json, os, sys, urllib.error, urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
REPORT_DIR = BASE / "output" / "research_reports"
SOURCE_PATH = REPORT_DIR / "finance_info_latest.json"
PUBLIC_PATH = REPORT_DIR / "finance_info_public.json"
REPO = os.getenv("GITHUB_REPOSITORY", "alanpass/stock_trading")
BRANCH = os.getenv("FINANCE_DATA_BRANCH", "finance-data")
REMOTE_PATH = "output/research_reports/finance_info_public.json"
API_BASE = f"https://api.github.com/repos/{REPO}/contents/{REMOTE_PATH}"
TAIPEI = ZoneInfo("Asia/Taipei")


def safe_text(value, limit=1200):
    return " ".join(str(value or "").split())[:limit]


def safe_list(value):
    return value if isinstance(value, list) else []


def clean_news(x):
    return {
        "article_id": safe_text(x.get("article_id"),80),
        "title": safe_text(x.get("title") or x.get("headline"),300),
        "url": safe_text(x.get("url") or x.get("source_url"),1000),
        "category": safe_text(x.get("category"),80),
        "published": safe_text(x.get("published") or x.get("published_at"),80),
        "published_ts": safe_text(x.get("published_ts") or x.get("published_at"),80),
        "ai_points": [safe_text(v,300) for v in safe_list(x.get("ai_points")) if safe_text(v,300)],
        "summary": safe_text(x.get("summary") or x.get("ai_summary")),
        "sentiment": safe_text(x.get("sentiment"),30),
        "sectors": [safe_text(v,50) for v in safe_list(x.get("sectors")) if safe_text(v,50)],
        "stocks": [safe_text(v,30) for v in safe_list(x.get("stocks")) if safe_text(v,30)],
        "highlights": [safe_text(v,80) for v in safe_list(x.get("highlights")) if safe_text(v,80)],
        "why": safe_text(x.get("why"),160),
        "importance": x.get("importance"),
        "ai_source": safe_text(x.get("ai_source"),30),
        "source": safe_text(x.get("source"),80),
    }


def clean_earnings(x):
    return {
        "title": safe_text(x.get("title") or x.get("event_title"),300),
        "symbol": safe_text(x.get("symbol") or x.get("code") or x.get("stock_code"),30),
        "name": safe_text(x.get("name") or x.get("company") or x.get("company_name"),120),
        "event_date": safe_text(x.get("event_date") or x.get("published_date") or x.get("date"),80),
        "published_at": safe_text(x.get("published_at") or x.get("published_time") or x.get("published_ts") or x.get("published"),80),
        "ai_points": [safe_text(v,300) for v in safe_list(x.get("ai_points")) if safe_text(v,300)],
        "summary": safe_text(x.get("one_line_summary") or x.get("summary") or x.get("ai_summary")),
        "sentiment": safe_text(x.get("sentiment"),30),
        "impact": safe_text(x.get("impact"),160),
        "signal": safe_text(x.get("signal"),160),
        "judgement": safe_text(x.get("judgement"),160),
        "sectors": [safe_text(v,50) for v in safe_list(x.get("sectors")) if safe_text(v,50)],
        "highlights": [safe_text(v,80) for v in safe_list(x.get("highlights")) if safe_text(v,80)],
        "source_url": safe_text(x.get("source_url") or x.get("url"),1000),
    }


def api_request(method, url, token, body=None):
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method=method, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "stock-trading-finance-publisher",
        **({"Content-Type":"application/json"} if payload is not None else {})
    })
    try:
        with urllib.request.urlopen(req, timeout=25) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"GitHub API HTTP {exc.code}: {detail}") from exc


def main():
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        print("ERROR: 請設定環境變數 GITHUB_TOKEN（GitHub fine-grained token，Contents: Read and write）。")
        return 2
    if not SOURCE_PATH.exists():
        print(f"ERROR: 找不到本機來源：{SOURCE_PATH}")
        return 2
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not source.get("updated_at"):
        print("ERROR: 來源資料缺少 updated_at，拒絕發布。")
        return 2
    if source.get("stale"):
        print("ERROR: 來源資料標記 stale，拒絕發布過期資料。")
        return 2
    news = [clean_news(x) for x in safe_list(source.get("news")) if isinstance(x,dict)]
    earnings = [clean_earnings(x) for x in safe_list(source.get("earnings")) if isinstance(x,dict)]
    now = datetime.now(TAIPEI).isoformat(timespec="seconds")
    public = {
        "schema_version": 1, "report_type": "finance_info_public",
        "report_date": safe_text(source.get("report_date"),30),
        "updated_at": safe_text(source.get("updated_at"),80),
        "finished_at": safe_text(source.get("finished_at"),80), "published_at": now,
        "lookback_days": int(source.get("lookback_days",2) or 2),
        "schedule": safe_list(source.get("schedule")), "stale": False,
        "news_count": len(news), "earnings_count": len(earnings),
        "news_digest": source.get("news_digest") if isinstance(source.get("news_digest"),dict) else {},
        "earnings_digest": source.get("earnings_digest") if isinstance(source.get("earnings_digest"),dict) else {},
        "crawl": source.get("crawl") if isinstance(source.get("crawl"),dict) else {},
        "agent_stats": source.get("agent_stats") if isinstance(source.get("agent_stats"),dict) else {},
        "errors_count": len(safe_list(source.get("errors"))), "news": news, "earnings": earnings,
    }
    encoded = (json.dumps(public,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    PUBLIC_PATH.parent.mkdir(parents=True,exist_ok=True)
    PUBLIC_PATH.write_bytes(encoded)
    url = f"{API_BASE}?ref={BRANCH}"
    try:
        current = api_request("GET",url,token)
        sha = current.get("sha")
        # 避免不必要的 commit；GitHub contents API 回傳的 base64 是完整檔案內容。
        import base64
        old = base64.b64decode(current.get("content", "").encode()).decode("utf-8",errors="replace")
        if old.strip() == encoded.decode("utf-8").strip():
            print("NO_CHANGE: 遠端 JSON 與本機相同，不建立 commit。")
            return 0
    except urllib.error.HTTPError:
        sha = None
    except Exception as exc:
        if "HTTP 404" in str(exc):
            sha = None
        else:
            raise
    body = {"message":f"data: update finance research {now}","content":__import__("base64").b64encode(encoded).decode("ascii"),"branch":BRANCH}
    if sha: body["sha"] = sha
    result = api_request("PUT",API_BASE,token,body)
    commit = result.get("commit",{})
    print(json.dumps({"ok":True,"repo":REPO,"branch":BRANCH,"path":REMOTE_PATH,"updated_at":public["updated_at"],"published_at":now,"news_count":len(news),"earnings_count":len(earnings),"commit":commit.get("sha")},ensure_ascii=False,indent=2))
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as e:
        print(f"PUBLISH_FAILED: {type(e).__name__}: {e}")
        raise SystemExit(7)
