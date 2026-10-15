# -*- coding: utf-8 -*-
"""Claude 研究工具命令列。

Claude 排程任務用這支程式查詢已備妥的研究包（盤後、早報共用）。
工具實作在 agent_research_assistant.ResearchToolbox._tool，
與研究包使用同一份證據。

用法：
  python claude_tools.py list
  python claude_tools.py call <工具名稱> '{"query": "載板 ABF"}' [--date YYYY-MM-DD] [--max-chars 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent_research_assistant import TOOLS, ResearchToolbox

BASE = Path(__file__).resolve().parent
JOB_ROOT = BASE / "output" / "claude_jobs"


def _today() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")


def _load_package(job: str, date: str) -> dict:
    path = JOB_ROOT / f"{job}_{date}" / "package.json"
    if not path.exists():
        raise SystemExit(f"ERROR: 找不到研究包 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Claude 研究工具")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出可用工具")
    call = sub.add_parser("call", help="呼叫工具")
    call.add_argument("tool")
    call.add_argument("args", nargs="?", default="{}")
    call.add_argument("--job", default="afterclose")
    call.add_argument("--date", default="")
    call.add_argument("--max-chars", type=int, default=20000)
    ns = parser.parse_args()

    if ns.cmd == "list":
        for t in TOOLS:
            fn = t["function"]
            params = ", ".join(fn.get("parameters", {}).get("properties", {}).keys())
            print(f"- {fn['name']}({params})：{fn['description']}")
        return 0

    try:
        args = json.loads(ns.args)
    except json.JSONDecodeError as exc:
        print(f"ERROR: 參數不是合法 JSON：{exc}")
        return 2
    payload = _load_package(ns.job, ns.date or _today())
    result = ResearchToolbox(BASE)._tool(ns.tool, args, payload)
    text = json.dumps(result, ensure_ascii=False, indent=1, default=str)
    if len(text) > ns.max_chars:
        text = text[: ns.max_chars] + f"\n…（已截斷，共 {len(text)} 字；可加 --max-chars 或縮小查詢）"
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
