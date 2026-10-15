# -*- coding: utf-8 -*-
"""財經資訊「AI 今日重點」：Claude 排程任務的輸入／輸出。

  python finance_digest.py export            # 產生 output/claude_jobs/finance/brief.md
  python finance_digest.py apply <notes.json> [--no-publish]

apply 會把 Claude 的重點寫回 finance_info_latest.json 的 news_digest／earnings_digest
（generated_by=claude）與逐篇新聞 AI 重點（ai_source=claude，同時寫入摘要快取，
之後的排程更新會沿用），再呼叫 publish_research_data.py 發布到網站。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
LATEST = BASE / "output" / "research_reports" / "finance_info_latest.json"
JOB_DIR = BASE / "output" / "claude_jobs" / "finance"
TAIPEI = ZoneInfo("Asia/Taipei")
ARTICLES_PER_RUN = 30


def _load() -> dict:
    if not LATEST.exists():
        raise SystemExit(f"ERROR: 找不到 {LATEST}，請先執行 run_finance_info_update.py")
    data = json.loads(LATEST.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("ERROR: finance_info_latest.json 格式錯誤")
    return data


def export() -> int:
    data = _load()
    lines = [f"# 財經資訊研究包｜資料更新 {data.get('updated_at', '')}", "",
             f"## 新聞（最近 {data.get('lookback_days', 2)} 天，共 {len(data.get('news', []) or [])} 則；依重要度排序）"]
    for i, a in enumerate(data.get("news", []) or [], 1):
        if not isinstance(a, dict):
            continue
        points = "；".join(str(x) for x in (a.get("ai_points") or [])[:3])
        sectors = "、".join(str(x) for x in (a.get("sectors") or [])[:4])
        lines.append(f"{i}. [{a.get('sentiment', '中性')}] {a.get('title', '')}｜{str(a.get('published_at') or a.get('date') or '')[:16]}｜{sectors}｜{points}｜{a.get('url', '')}")
    pending = sorted([a for a in data.get("news", []) or [] if isinstance(a, dict) and a.get("ai_source") != "claude"],
                     key=lambda a: -float(a.get("importance") or 0))[:ARTICLES_PER_RUN]
    lines += ["", f"## 待撰寫逐篇 AI 重點（依重要度前 {len(pending)} 則）"]
    for a in pending:
        body = " ".join(str(a.get("content") or a.get("summary_raw") or a.get("summary") or "").split())[:1500]
        lines += [f"### id={_key(a)}｜{a.get('title', '')}", body, ""]
    lines += ["", f"## 法說會（共 {len(data.get('earnings', []) or [])} 場）"]
    for i, e in enumerate(data.get("earnings", []) or [], 1):
        if not isinstance(e, dict):
            continue
        points = "；".join(str(x) for x in (e.get("ai_points") or [])[:3])
        lines.append(f"{i}. [{e.get('sentiment', e.get('impact', '中性'))}] {e.get('name', '')}（{e.get('symbol', '')}）{str(e.get('published_date') or e.get('event_date') or '')[:10]}｜{points}")
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = JOB_DIR / "brief.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"BRIEF={path}")
    return 0


def _key(a: dict) -> str:
    return str(a.get("article_id") or a.get("url") or a.get("title") or "")


def _apply_articles(data: dict, rows: list) -> int:
    """逐篇 AI 重點：寫回新聞列並存入摘要快取（指紋以原始摘要計算，與爬蟲新資料一致）。"""
    from finance_agent import FinanceNewsAgent, _article_key, _dedupe_keep_order, _fingerprint

    by_id = {str(r.get("id")): r for r in rows if isinstance(r, dict) and r.get("id")}
    if not by_id:
        return 0
    agent = FinanceNewsAgent(BASE)
    cache = agent._load_cache()
    stamp = datetime.now(TAIPEI).isoformat(timespec="seconds")
    done = 0
    for a in data.get("news", []) or []:
        obj = by_id.get(_key(a)) if isinstance(a, dict) else None
        if not obj:
            continue
        raw = {**a, "summary": a.get("summary_raw", a.get("summary", ""))}
        ai = agent.normalize_article_ai(obj, raw)
        if not ai:
            continue
        ai["ai_source"] = "claude"
        cache[_article_key(raw)] = {"fp": _fingerprint(raw), "title": a.get("title", ""), "ai": ai, "ts": stamp}
        points = ai["points"]
        a.update({
            "ai_points": points,
            "ai_summary": "；".join(points),
            "summary": "；".join(points),
            "sentiment": ai["sentiment"],
            "sectors": ai["sectors"] or a.get("sectors", []),
            "stocks": _dedupe_keep_order(list(ai["stocks"]) + list(a.get("stocks") or []))[:8],
            "why": ai["why"],
            "highlights": ai["highlights"] or agent.extract_highlights(points, a.get("title", ""), a.get("stocks", []), a.get("sectors", [])),
            "ai_source": "claude",
        })
        done += 1
    agent._save_cache(cache)
    return done


def _clean_list(value, n: int) -> list[str]:
    return [str(x).strip()[:200] for x in (value if isinstance(value, list) else []) if str(x).strip()][:n]


def _merge(digest: dict, part: dict, max_points: int, stamp: str) -> dict:
    if not isinstance(part, dict) or not _clean_list(part.get("key_points"), max_points):
        return digest
    return {
        **(digest if isinstance(digest, dict) else {}),
        "agent_headline": str(part.get("headline", "")).strip()[:120],
        "agent_key_points": _clean_list(part.get("key_points"), max_points),
        "watch_items": _clean_list(part.get("watch_items"), 5),
        "risks": _clean_list(part.get("risks"), 3),
        "generated_by": "claude",
        "agent_generated_at": stamp,
    }


def apply(notes_path: Path, publish: bool) -> int:
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    if not isinstance(notes, dict):
        print("ERROR: notes 必須是 JSON 物件")
        return 5
    from quality_checks import quality_gate, validate_finance_digest
    blocked = quality_gate(validate_finance_digest(notes_path, LATEST))
    if blocked:
        return blocked
    data = _load()
    stamp = datetime.now(TAIPEI).isoformat(timespec="seconds")
    data["news_digest"] = _merge(data.get("news_digest", {}), notes.get("news", {}), 8, stamp)
    if data.get("earnings"):
        data["earnings_digest"] = _merge(data.get("earnings_digest", {}), notes.get("earnings", {}), 6, stamp)
    articles = _apply_articles(data, notes.get("articles") or [])
    tmp = LATEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LATEST)
    print(f"APPLIED news={data['news_digest'].get('generated_by')} earnings={(data.get('earnings_digest') or {}).get('generated_by')} articles={articles}")
    if not publish:
        return 0
    proc = subprocess.run([sys.executable, str(BASE / "publish_research_data.py")], cwd=str(BASE), text=True,
                          encoding="utf-8", errors="replace", capture_output=True)
    print((proc.stdout or "")[-1500:])
    return proc.returncode


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="財經資訊 AI 今日重點（Claude）")
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
