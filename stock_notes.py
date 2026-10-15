# -*- coding: utf-8 -*-
"""單檔個股研究筆記的新增／更新（stock-deep-research skill 使用）。

  python stock_notes.py upsert <note.json> [--no-publish]

note.json 可以是單一筆記物件或筆記陣列（欄位同盤後 agent_notes.json 的 stock_notes）。
同一檔股票會被新筆記取代，其他股票保留；每則筆記帶自己的 updated_at，網站會顯示。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_backend import AGENT_LABEL
from quality_checks import ADVICE_PHRASES
from run_after_close_research import (STOCK_NOTES_LATEST, STOCK_NOTES_PUBLIC, STOCK_NOTES_REMOTE_PATH,
                                      _clean_stock_notes)

TAIPEI = ZoneInfo("Asia/Taipei")


def upsert(path: Path, publish: bool) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]
    notes = _clean_stock_notes(items, {})
    if not notes:
        print("ERROR: 沒有有效的筆記（需要 symbol 與 summary）")
        return 5
    for n in notes:
        hits = [p for p in ADVICE_PHRASES if p in json.dumps(n, ensure_ascii=False)]
        if hits:
            print(f"QUALITY_ERROR: {n['symbol']} 含投資建議用語：{hits}")
            return 6
    stamp = datetime.now(TAIPEI).isoformat(timespec="seconds")
    current = json.loads(STOCK_NOTES_LATEST.read_text(encoding="utf-8")) if STOCK_NOTES_LATEST.exists() else {
        "schema_version": 1, "report_type": "stock_notes_public", "report_date": stamp[:10], "generated_at": stamp,
        "generated_by": AGENT_LABEL, "disclaimer": "研究筆記僅整理公開資訊與證據，不構成投資建議。", "notes": []}
    by_symbol = {n.get("symbol"): n for n in current.get("notes", [])}
    for n in notes:
        old = by_symbol.get(n["symbol"], {})
        n["category"] = old.get("category", "deep")
        n["updated_at"] = stamp
        n["research_type"] = "deep"
        by_symbol[n["symbol"]] = n
    current["notes"] = list(by_symbol.values())
    current["count"] = len(current["notes"])
    encoded = json.dumps(current, ensure_ascii=False, indent=2) + "\n"
    STOCK_NOTES_LATEST.write_text(encoded, encoding="utf-8")
    STOCK_NOTES_PUBLIC.write_text(encoded, encoding="utf-8")
    print(f"UPSERTED={[n['symbol'] for n in notes]} TOTAL={current['count']}")
    if not publish:
        return 0
    from github_publish import publish_json
    result = publish_json(STOCK_NOTES_REMOTE_PATH, encoded, f"data: deep research {','.join(n['symbol'] for n in notes)} {stamp}")
    print("PUBLISH=" + json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 7


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="個股研究筆記新增／更新")
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upsert")
    up.add_argument("path")
    up.add_argument("--no-publish", action="store_true")
    ns = ap.parse_args()
    return upsert(Path(ns.path), publish=not ns.no_publish)


if __name__ == "__main__":
    raise SystemExit(main())
