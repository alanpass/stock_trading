# -*- coding: utf-8 -*-
"""把系統推薦股票（波段＋長期）寫進三竹股市（電腦版）自選股「群組四」，群組名稱 AI。

三竹的自選股存在 MitakeGU\\USER\\MK\\<帳號>\\：
  UserDefSet.set   各群組名稱（UTF-16，一行一群組：|>MARK=99|>SUB=<0..4>|>TITLE=<名稱>|>）
  MK_99_<n>.stk    第 n 個群組的股票（UTF-16，一行一檔：|>代號=2330|>商品名稱=台積電|>語音=0|>新聞=0|>警示=0|>）
  StampLog.ini     gsn＝最後一次自選股時間戳記
三竹開著時會隨時改寫這些檔案並上傳雲端，所以只在三竹沒開時寫入；開著就跳過、留待下次。
注意（2026-10-26 實測）：三竹登入時會從雲端下載自選股覆蓋這些檔案，改 gsn 也沒用，所以這個工具實際上
無法讓自選股生效；三竹也不接受程式送出的滑鼠／鍵盤操作。目前改成每天寄 Email（system_picks.send_email）。
每次寫入前把原檔備份到 MitakeGU\\USER\\MK\\backup_<時間>\\。

  python mitake_watchlist.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
MITAKE = Path.home() / "OneDrive" / "Desktop" / "MitakeGU"
MK_DIR = MITAKE / "USER" / "MK"
PICKS = BASE / "output" / "research_reports" / "system_picks_public.json"
STATE = BASE / "output" / "research_reports" / "mitake_sync_state.json"
GROUP_SUB = 3          # 群組四（0 起算）
GROUP_TITLE = "AI"


def mitake_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq 三竹股市.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, encoding="cp950", errors="replace").stdout
    return "三竹股市.exe" in out


def account_dir() -> Path:
    dirs = [d for d in MK_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    if len(dirs) != 1:
        raise RuntimeError(f"找不到唯一的三竹帳號資料夾：{[d.name for d in dirs]}")
    return dirs[0]


def _write_utf16(path: Path, lines: list[str]) -> None:
    # 與三竹原檔相同：UTF-16 LE 加 BOM、每行 CRLF 結尾
    path.write_bytes(b"\xff\xfe" + "".join(line + "\r\n" for line in lines).encode("utf-16-le"))


def picks() -> tuple[str, list[tuple[str, str]]]:
    data = json.loads(PICKS.read_text(encoding="utf-8"))
    latest = data.get("latest") or {}
    seen, out = set(), []
    for horizon in ("swing", "long"):     # 波段在前、長期在後，重複的只留一次
        for p in latest.get("picks") or []:
            if p.get("horizon") == horizon and p["symbol"] not in seen:
                seen.add(p["symbol"])
                out.append((p["symbol"], p["name"]))
    return latest.get("date") or "", out


def sync(dry_run: bool = False) -> dict:
    day, stocks = picks()
    if not stocks:
        return {"ok": False, "reason": "沒有推薦股票"}
    acct = account_dir()
    stk = acct / f"MK_99_{GROUP_SUB}.stk"
    titles_path = acct / "UserDefSet.set"
    titles = titles_path.read_text(encoding="utf-16").splitlines()
    new_titles = [f"|>MARK=99|>SUB={GROUP_SUB}|>TITLE={GROUP_TITLE}|>" if f"|>SUB={GROUP_SUB}|>" in t else t for t in titles]
    lines = [f"|>代號={code}|>商品名稱={name}|>語音=0|>新聞=0|>警示=0|>" for code, name in stocks]
    result = {"date": day, "count": len(stocks), "stocks": [c for c, _ in stocks]}
    if dry_run:
        return {"ok": True, "dry_run": True, **result}
    if mitake_running():
        STATE.write_text(json.dumps({"pending": True, "date": day, "checked_at": datetime.now().isoformat(timespec="seconds")},
                                    ensure_ascii=False), encoding="utf-8")
        return {"ok": False, "pending": True, "reason": "三竹股市開著，等關閉後再寫入", **result}

    backup = MK_DIR / f"backup_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(acct, backup)
    _write_utf16(stk, lines)
    _write_utf16(titles_path, new_titles)
    # 時間戳記改成現在（實測三竹登入仍會用雲端版本覆蓋）
    (acct / "StampLog.ini").write_text(f"[MiscInfo]\r\ngsn={datetime.now():%Y%m%d%H%M%S}\r\n", encoding="ascii", newline="")
    STATE.write_text(json.dumps({"pending": False, "date": day, "synced_at": datetime.now().isoformat(timespec="seconds"),
                                 "backup": str(backup), **result}, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "backup": str(backup), **result}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="系統推薦股票寫入三竹自選股群組四（AI）")
    ap.add_argument("--dry-run", action="store_true")
    print(json.dumps(sync(ap.parse_args().dry_run), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
