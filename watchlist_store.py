# -*- coding: utf-8 -*-
"""自選股永久保存在 Supabase（site_settings 資料表，key＝watchlist）。

Streamlit Cloud 的檔案系統每次重新部署都會重置，網站上改的 user_watchlist.json 會被 git 版本蓋回去，
所以網站改自選股時同時寫進資料庫；本機排程（早報、盤後研究等）開跑前用 pull_to_file() 把資料庫版本同步回
user_watchlist.json。資料庫連不上或資料表不存在時一律回傳 None，呼叫端退回本機檔案。

資料表（Supabase SQL Editor 執行一次）：
  create table if not exists public.site_settings (
    key text primary key, value jsonb not null, updated_at timestamptz not null default now());
  alter table public.site_settings enable row level security;

  python watchlist_store.py          # 顯示資料庫版本
  python watchlist_store.py --pull   # 資料庫 → user_watchlist.json
  python watchlist_store.py --push   # user_watchlist.json → 資料庫
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE / "user_watchlist.json"
TABLE = "site_settings"
KEY = "watchlist"
TIMEOUT_SECONDS = 8


def _secret(name: str) -> str:
    """環境變數 → 本機 .streamlit/secrets.toml。"""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(tomllib.loads((BASE / ".streamlit" / "secrets.toml").read_text(encoding="utf-8")).get(name, "")).strip()
    except (OSError, ValueError):
        return ""


def _endpoint() -> tuple[str, dict[str, str]] | None:
    raw_url, key = _secret("ACCOUNT_DB_URL"), _secret("ACCOUNT_DB_SERVICE_ROLE_KEY")
    if not raw_url or not key:
        return None
    from account_provisioning import _normalize_project_url
    headers = {"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json"}
    return f"{_normalize_project_url(raw_url)}/rest/v1/{TABLE}", headers


def _clean(items) -> list[str]:
    out = []
    for value in items or []:
        code = str(value).strip().upper().replace(".TW", "")
        if code and code not in out:
            out.append(code)
    return out


def load() -> list[str] | None:
    """資料庫裡的自選股；沒設定、連不上或還沒存過時回傳 None。"""
    ep = _endpoint()
    if not ep:
        return None
    url, headers = ep
    try:
        r = requests.get(url, params={"key": f"eq.{KEY}", "select": "value"}, headers=headers, timeout=TIMEOUT_SECONDS)
        if r.status_code != 200:
            return None
        rows = r.json()
        items = _clean(rows[0]["value"]) if rows and isinstance(rows[0].get("value"), list) else []
        return items or None
    except (requests.RequestException, ValueError, KeyError):
        return None


def save(items) -> bool:
    """寫入（覆蓋）資料庫版本；成功回傳 True。"""
    ep = _endpoint()
    if not ep:
        return False
    url, headers = ep
    payload = {"key": KEY, "value": _clean(items), "updated_at": datetime.now(timezone.utc).isoformat()}
    try:
        r = requests.post(url, params={"on_conflict": "key"}, json=payload, timeout=TIMEOUT_SECONDS,
                          headers={**headers, "Prefer": "resolution=merge-duplicates,return=minimal"})
        return r.status_code in (200, 201, 204)
    except requests.RequestException:
        return False


def pull_to_file(path: Path = WATCHLIST_FILE) -> list[str] | None:
    """把資料庫版本寫回本機 user_watchlist.json（內容相同就不動）；回傳資料庫版本或 None。"""
    items = load()
    if not items:
        return None
    text = json.dumps(items, ensure_ascii=False, indent=2)
    try:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    except OSError:
        pass
    return items


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--push":
        items = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
        print(json.dumps({"ok": save(items), "count": len(_clean(items))}, ensure_ascii=False))
    elif arg == "--pull":
        items = pull_to_file()
        print(json.dumps({"ok": items is not None, "items": items}, ensure_ascii=False))
    else:
        print(json.dumps({"items": load()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
