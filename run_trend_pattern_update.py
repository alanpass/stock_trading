# -*- coding: utf-8 -*-
"""趨勢型態每日檢查與發布入口；預定台灣時間 19:00 執行。"""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trend_pattern_engine import update_trend_patterns

BASE = Path(__file__).resolve().parent
TAIPEI = ZoneInfo("Asia/Taipei")
LOG_FILE = BASE / "output" / "scheduler_logs" / "trend_patterns_runner.log"


def log(message: str) -> None:
    stamp = datetime.now(TAIPEI).isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="每日檢查產業概念股趨勢型態；僅重大變化時發布")
    parser.add_argument("--force", action="store_true", help="忽略非交易日檢查，使用目前可用日 K 建立／檢查快照")
    parser.add_argument("--no-publish", action="store_true", help="僅更新本機快照，不發布到 GitHub")
    args = parser.parse_args()
    log("=" * 68)
    log("趨勢型態檢查開始")
    try:
        result = update_trend_patterns(BASE, force=args.force, publish=not args.no_publish)
        log(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if result.get("skipped"):
            log("本次未發布；可能是非交易日或沒有重大型態變化。")
        else:
            log("趨勢型態快照更新完成。")
        return 0
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
