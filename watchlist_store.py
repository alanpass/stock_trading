# -*- coding: utf-8 -*-
"""自選股名單永久保存在 Supabase（site_settings 資料表，key＝watchlists）。

可以有多份名單，各有自己的名稱：
  {"active": "l1", "lists": [{"id": "l1", "name": "1", "items": ["2330", ...]}, ...]}
舊版只有一份（key＝watchlist）；還沒存過新格式時，自動把它當成名單「1」。

Streamlit Cloud 的檔案系統每次重新部署都會重置，所以網站改名單時寫進資料庫；本機排程（早報、盤後研究等）開跑前
用 pull_to_file() 把「所有名單的聯集」同步回 user_watchlist.json，報告才會涵蓋每一份名單。
資料庫連不上或資料表不存在時回傳 None，呼叫端退回本機檔案。

資料表（Supabase SQL Editor 執行一次）：
  create table if not exists public.site_settings (
    key text primary key, value jsonb not null, updated_at timestamptz not null default now());
  alter table public.site_settings enable row level security;

  python watchlist_store.py          # 顯示資料庫版本
  python watchlist_store.py --pull   # 資料庫（所有名單的聯集）→ user_watchlist.json
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
KEY = "watchlists"
LEGACY_KEY = "watchlist"
TIMEOUT_SECONDS = 8
MAX_LISTS = 20
MAX_NAME_LENGTH = 20


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


def _read(key: str):
    ep = _endpoint()
    if not ep:
        return None
    url, headers = ep
    try:
        r = requests.get(url, params={"key": f"eq.{key}", "select": "value"}, headers=headers, timeout=TIMEOUT_SECONDS)
        if r.status_code != 200:
            return None
        rows = r.json()
        return rows[0].get("value") if rows else None
    except (requests.RequestException, ValueError, KeyError):
        return None


def single_list_doc(items) -> dict:
    return {"active": "l1", "lists": [{"id": "l1", "name": "1", "items": _clean(items)}]}


def _normalize(doc) -> dict | None:
    """整理資料庫內容；格式不對或一份名單都沒有時回傳 None。"""
    if not isinstance(doc, dict):
        return None
    lists = []
    for entry in doc.get("lists") or []:
        if isinstance(entry, dict) and entry.get("id"):
            lists.append({"id": str(entry["id"]), "name": str(entry.get("name") or entry["id"]), "items": _clean(entry.get("items"))})
    if not lists:
        return None
    active = str(doc.get("active") or "")
    return {"active": active if any(x["id"] == active for x in lists) else lists[0]["id"], "lists": lists}


def load_doc() -> dict | None:
    """所有名單；沒設定、連不上或還沒存過時回傳 None。"""
    doc = _normalize(_read(KEY))
    if doc:
        return doc
    legacy = _read(LEGACY_KEY)
    return single_list_doc(legacy) if isinstance(legacy, list) and _clean(legacy) else None


def save_doc(doc: dict) -> bool:
    """寫入（覆蓋）資料庫版本；成功回傳 True。"""
    ep = _endpoint()
    if not ep:
        return False
    url, headers = ep
    payload = {"key": KEY, "value": doc, "updated_at": datetime.now(timezone.utc).isoformat()}
    try:
        r = requests.post(url, params={"on_conflict": "key"}, json=payload, timeout=TIMEOUT_SECONDS,
                          headers={**headers, "Prefer": "resolution=merge-duplicates,return=minimal"})
        return r.status_code in (200, 201, 204)
    except requests.RequestException:
        return False


def active_list(doc: dict) -> dict:
    return next((x for x in doc["lists"] if x["id"] == doc["active"]), doc["lists"][0])


def all_items(doc: dict) -> list[str]:
    """所有名單的聯集，目前選的名單排最前面。"""
    return _clean(active_list(doc)["items"] + [code for x in doc["lists"] for code in x["items"]])


def clean_name(name) -> str:
    return " ".join(str(name or "").split())[:MAX_NAME_LENGTH]


def next_default_name(doc: dict) -> str:
    """下一個沒人用的數字名稱（1、2、3…）。"""
    used = {x["name"] for x in doc["lists"]}
    number = 1
    while str(number) in used:
        number += 1
    return str(number)


def name_taken(doc: dict, name: str, ignore_id: str = "") -> bool:
    return any(x["name"] == name and x["id"] != ignore_id for x in doc["lists"])


def add_list(doc: dict, name="") -> str | None:
    """新增空名單並切換過去；回傳錯誤訊息，成功回傳 None。"""
    if len(doc["lists"]) >= MAX_LISTS:
        return f"最多 {MAX_LISTS} 份名單。"
    name = clean_name(name) or next_default_name(doc)
    if name_taken(doc, name):
        return f"已經有名為「{name}」的名單。"
    used = {x["id"] for x in doc["lists"]}
    number = len(doc["lists"]) + 1
    while f"l{number}" in used:
        number += 1
    doc["lists"].append({"id": f"l{number}", "name": name, "items": []})
    doc["active"] = f"l{number}"
    return None


def rename_list(doc: dict, list_id: str, name) -> str | None:
    name = clean_name(name)
    if not name:
        return "名稱不能空白。"
    if name_taken(doc, name, ignore_id=list_id):
        return f"已經有名為「{name}」的名單。"
    for entry in doc["lists"]:
        if entry["id"] == list_id:
            entry["name"] = name
            return None
    return "找不到這份名單，可能已被刪除。"


def delete_list(doc: dict, list_id: str) -> str | None:
    if len(doc["lists"]) <= 1:
        return "至少要留一份名單。"
    if not any(x["id"] == list_id for x in doc["lists"]):
        return "找不到這份名單，可能已被刪除。"
    index = next(i for i, x in enumerate(doc["lists"]) if x["id"] == list_id)
    doc["lists"].pop(index)
    if doc["active"] == list_id:
        doc["active"] = doc["lists"][max(0, index - 1)]["id"]
    return None


def pull_to_file(path: Path = WATCHLIST_FILE) -> list[str] | None:
    """把所有名單的聯集寫回本機 user_watchlist.json（內容相同就不動）；回傳聯集或 None。"""
    doc = load_doc()
    items = all_items(doc) if doc else []
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
    if "--pull" in sys.argv[1:]:
        items = pull_to_file()
        print(json.dumps({"ok": items is not None, "items": items}, ensure_ascii=False))
    else:
        print(json.dumps(load_doc(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
