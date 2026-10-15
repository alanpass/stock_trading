# -*- coding: utf-8 -*-
"""把本機專案的程式碼同步到 GitHub alanpass/stock_trading 的 main（Streamlit Cloud 會重新部署）。

  python sync_to_github.py            # 預覽：列出會新增／修改／刪除的檔案，不推送
  python sync_to_github.py --push -m "說明"   # 提交並推送到 main

做法：在旁邊的 clone（../stock_trading_git）對齊 origin/main，複製本機 git 追蹤的檔案，
刪除 main 上本機已移除的程式檔。以下內容不會被刪除：output/（排程發布的資料）、.github/、.devcontainer/。
推送前會檢查沒有 .env、secrets.toml、備份檔或寫死的金鑰。
也會檢查上次同步之後，GitHub 上有沒有人直接修改程式檔（例如在網頁上改 README）；
有的話會停下來列出檔案，請先把那些修改帶回本機，避免被覆蓋（確定要覆蓋時加 --overwrite-remote-edits）。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
CLONE = LOCAL.parent / "stock_trading_git"
KEEP_ON_MAIN = ("output/", ".github/", ".devcontainer/")
FORBIDDEN = re.compile(r"(^|/)\.env$|secrets\.toml$|\.bak|backup_|(^|/)__pycache__/")
LAST_SYNC = LOCAL / ".git" / "github_last_sync"
KEY_PATTERN = re.compile(r'FUGLE_API_KEY\s*=\s*["\']?[A-Za-z0-9+/=]{40,}')


def git(*args: str, repo: Path = CLONE, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失敗：{out.stderr.strip()}")
    return out.stdout


def tracked(repo: Path) -> list[str]:
    return [f for f in git("ls-files", "-z", repo=repo).split("\0") if f]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="同步程式碼到 GitHub main")
    ap.add_argument("--push", action="store_true", help="提交並推送（沒有這個參數只預覽）")
    ap.add_argument("-m", "--message", default="Sync code from local project")
    ap.add_argument("--overwrite-remote-edits", action="store_true", help="GitHub 上的直接修改也要覆蓋")
    ns = ap.parse_args()

    if not (CLONE / ".git").exists():
        raise SystemExit(f"找不到 GitHub clone：{CLONE}（git clone https://github.com/alanpass/stock_trading.git {CLONE}）")
    if git("status", "--porcelain").strip():
        raise SystemExit(f"{CLONE} 有未提交的變更，請先處理再同步。")
    if git("status", "--porcelain", repo=LOCAL).strip():
        print("注意：本機專案有未提交的變更；同步的是目前工作目錄中的檔案。")

    git("fetch", "-q", "origin")
    git("checkout", "-q", "main")
    git("reset", "-q", "--hard", "origin/main")

    # 上次同步之後，GitHub 上被直接修改的程式檔（output/ 由排程發布，不算）
    if LAST_SYNC.exists() and not ns.overwrite_remote_edits:
        since = LAST_SYNC.read_text(encoding="utf-8").strip()
        edited = [f for f in git("diff", "--name-only", f"{since}..origin/main", check=False).splitlines()
                  if f and not f.startswith(KEEP_ON_MAIN)]
        if edited:
            raise SystemExit("停止：上次同步後 GitHub 上有人直接修改了這些檔案，請先帶回本機再同步"
                             "（或加 --overwrite-remote-edits 覆蓋）：\n  " + "\n  ".join(edited))

    local = tracked(LOCAL)
    local_set = set(local)
    for f in local:
        dst = CLONE / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LOCAL / f, dst)
    removed = [f for f in tracked(CLONE) if not f.startswith(KEEP_ON_MAIN) and f not in local_set]
    for i in range(0, len(removed), 50):
        git("rm", "-q", "--", *removed[i:i + 50])
    git("add", "-A", "--", ".")

    changes = [line.split("\t") for line in git("diff", "--cached", "--name-status").splitlines() if line]
    if not changes:
        print("main 已經是最新，沒有需要同步的變更。")
        LAST_SYNC.write_text(git("rev-parse", "origin/main").strip(), encoding="utf-8")
        return 0
    bad = [c[-1] for c in changes if c[0] != "D" and FORBIDDEN.search(c[-1])]
    leaked = [c[-1] for c in changes if c[0] != "D" and (CLONE / c[-1]).is_file()
              and KEY_PATTERN.search((CLONE / c[-1]).read_text(encoding="utf-8", errors="ignore"))]
    if bad or leaked:
        git("reset", "-q", "--hard", "origin/main")
        raise SystemExit(f"停止：不能上傳的檔案 {bad}，疑似含金鑰 {leaked}")

    summary: dict[str, int] = {}
    for c in changes:
        summary[c[0][0]] = summary.get(c[0][0], 0) + 1
    print("變更：" + "、".join(f"{k}={v}" for k, v in sorted(summary.items())) + "（A 新增／M 修改／D 刪除／R 改名）")
    for c in changes[:80]:
        print(f"  {c[0][0]} {c[-1]}")
    if not ns.push:
        git("reset", "-q", "--hard", "origin/main")
        print("預覽完成（未推送）。確認後加上 --push。")
        return 0

    git("-c", "user.name=alanpass", "-c", "user.email=10501500978aaa@gmail.com", "commit", "-q", "-m", ns.message)
    for attempt in range(2):
        out = subprocess.run(["git", "-C", str(CLONE), "push", "origin", "main"], capture_output=True, text=True)
        if out.returncode == 0:
            LAST_SYNC.write_text(git("rev-parse", "HEAD").strip(), encoding="utf-8")
            print(f"已推送：{git('log', '--oneline', '-1').strip()}")
            return 0
        # 排程在這段時間發布了資料到 main：rebase 後再推一次
        git("fetch", "-q", "origin")
        git("rebase", "origin/main")
    raise SystemExit(f"推送失敗：{out.stderr.strip()}")


if __name__ == "__main__":
    raise SystemExit(main())
