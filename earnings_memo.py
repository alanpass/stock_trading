# -*- coding: utf-8 -*-
"""Fugle 法說會備忘錄：Claude 排程任務的輸入／輸出。

  python earnings_memo.py export                 # 列出尚未由 Agent 分析的法說會 → output/claude_jobs/earnings/brief.md
  python earnings_memo.py apply <notes.json> [--no-publish]

apply 會用與舊版相同的驗證（FugleEarningsCallAgent._validate）寫回：
  1. Fugle 備忘錄資料庫（下次排程不會重複分析）
  2. finance_info_latest.json 的 earnings 與 earnings_digest
然後呼叫 publish_research_data.py 發布到網站。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_backend import AGENT_MODEL, AGENT_STATUS
from earnings_call_agent import EarningsCallAgent, FugleEarningsCallAgent
from earnings_summary_agent import EarningsSummaryAgent

BASE = Path(__file__).resolve().parent
LATEST = BASE / "output" / "research_reports" / "finance_info_latest.json"
JOB_DIR = BASE / "output" / "claude_jobs" / "earnings"
TAIPEI = ZoneInfo("Asia/Taipei")
ANALYSIS_SOURCE = "Claude 法說會 Agent"


def _load() -> dict:
    if not LATEST.exists():
        raise SystemExit(f"ERROR: 找不到 {LATEST}，請先執行 run_finance_info_update.py")
    return json.loads(LATEST.read_text(encoding="utf-8"))


def _url(item: dict) -> str:
    return str(item.get("url") or item.get("source_url") or "")


def _pending(item: dict) -> bool:
    return item.get("analysis_source") != ANALYSIS_SOURCE and not item.get("agent_tool_used")


def export() -> int:
    data = _load()
    items = [x for x in data.get("earnings", []) or [] if isinstance(x, dict) and _pending(x) and _url(x)]
    lines = [f"# 待分析法說會｜{len(items)} 場｜資料更新 {data.get('updated_at', '')}", ""]
    for i, x in enumerate(items, 1):
        sections = x.get("sections") if isinstance(x.get("sections"), dict) else {}
        body = "\n".join(f"### {k}\n{str(v)[:4000]}" for k, v in sections.items()) or str(x.get("memo_text", ""))[:8000]
        lines += [f"## {i}. {x.get('name', '')}（{x.get('symbol', '')}）{x.get('published_date', '')}",
                  f"url: {_url(x)}", f"標題：{x.get('title', '')}", "", body, ""]
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = JOB_DIR / "brief.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"PENDING={len(items)}")
    print(f"BRIEF={path}")
    return 0


def apply(notes_path: Path, publish: bool) -> int:
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    rows = notes.get("items") if isinstance(notes, dict) else None
    if not isinstance(rows, list):
        print("ERROR: notes 必須是 {\"items\": [...]}")
        return 5
    from quality_checks import quality_gate, validate_earnings_memo
    blocked = quality_gate(validate_earnings_memo(notes_path, LATEST))
    if blocked:
        return blocked
    by_url = {str(r.get("url")): r for r in rows if isinstance(r, dict) and r.get("url")}
    data = _load()
    crawler = EarningsCallAgent(BASE).crawler
    stamp = datetime.now(TAIPEI).isoformat(timespec="seconds")
    updated = []
    for item in data.get("earnings", []) or []:
        if not isinstance(item, dict) or _url(item) not in by_url:
            continue
        analysis = FugleEarningsCallAgent._validate({**by_url[_url(item)], "agent_tool_used": True, "memo_read_success": True})
        analysis.update({"analysis_source": ANALYSIS_SOURCE, "agent_status": AGENT_STATUS, "model": AGENT_MODEL,
                         "agent_source": "Claude -> Fugle 備忘錄正文 -> structured analysis", "analyzed_at": stamp})
        item.update(analysis)
        crawler.db_save({"url": _url(item), **{k: item.get(k, "") for k in ("symbol", "title", "published_date", "modified_date", "content_hash", "memo_text")}}, analysis)
        updated.append(_url(item))
    missing = sorted(set(by_url) - set(updated))
    if updated:
        # 重新產生網站用的重點與法說會 AI 重點（規則層），保留 Claude 寫的 analysis。
        summary_agent = EarningsSummaryAgent(BASE)
        earnings, digest = summary_agent.analyze_earnings(data["earnings"], days=int(data.get("earnings_lookback_days", 5) or 5))
        if isinstance(notes.get("digest"), dict) and notes["digest"].get("key_points"):
            d = notes["digest"]
            digest.update({
                "agent_headline": str(d.get("headline", "")).strip()[:120],
                "agent_key_points": [str(x).strip() for x in d.get("key_points", []) if str(x).strip()][:6],
                "watch_items": [str(x).strip() for x in d.get("watch_items", []) if str(x).strip()][:5] or digest.get("watch_items", []),
                "risks": [str(x).strip() for x in d.get("risks", []) if str(x).strip()][:3] or digest.get("risks", []),
                "generated_by": "claude",
                "agent_generated_at": stamp,
            })
        data["earnings"], data["earnings_digest"] = earnings, digest
        tmp = LATEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(LATEST)
    print(f"APPLIED={len(updated)} MISSING_URLS={missing}")
    if not publish or not updated:
        return 0 if updated or not by_url else 5
    proc = subprocess.run([sys.executable, str(BASE / "publish_research_data.py")], cwd=str(BASE), text=True,
                          encoding="utf-8", errors="replace", capture_output=True)
    print((proc.stdout or "")[-1500:])
    return proc.returncode


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="法說會備忘錄（Claude）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    ap = sub.add_parser("apply")
    ap.add_argument("notes")
    ap.add_argument("--no-publish", action="store_true")
    ns = parser.parse_args()
    if ns.cmd == "export":
        return export()
    return apply(Path(ns.notes), publish=not ns.no_publish)


if __name__ == "__main__":
    raise SystemExit(main())
