# -*- coding: utf-8 -*-
"""台股產業分析每日更新入口。Windows 工作排程於交易日 08:20 與 15:10 執行。"""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from industry_rotation import update_industry_rotation

BASE = Path(__file__).resolve().parent
TAIPEI = ZoneInfo("Asia/Taipei")
LOG_FILE = BASE / "logs" / "industry_rotation_task.log"


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
    parser = argparse.ArgumentParser(description="更新並發布台股產業分析輪動資料")
    parser.add_argument("--force", action="store_true", help="忽略 08:20／15:10 時段檢查；仍須取得可驗證的行情資料")
    parser.add_argument("--no-publish", action="store_true", help="只更新本機 JSON，不發布到 GitHub")
    args = parser.parse_args()

    log("=" * 68)
    log("產業分析更新開始")
    try:
        result = update_industry_rotation(BASE, force=args.force, publish=not args.no_publish)
        log(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if result.get("skipped"):
            log("本次略過，不修改已發布的成功快照。")
        else:
            log("產業分析更新完成。")
        return 0
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
