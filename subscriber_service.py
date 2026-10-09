# -*- coding: utf-8 -*-
"""
AI 台股研究中心｜訂閱名單服務

功能：
1. 將網站訂閱者保存到 data/subscribers.json。
2. 若設定 SUBSCRIBER_SYNC_URL / SUBSCRIBER_SYNC_TOKEN，
   同步到外部持久化服務（本專案提供 Google Apps Script 範例）。
3. 本機／雲端都可讀取同一份外部訂閱名單，讓晨報與盤後信件可以
   在寄送當下取得最新訂閱者。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean_email(value: Any) -> str:
    return str(value or "").strip().lower()


def _load_local(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict) and isinstance(data.get("subscribers"), list):
            return [x for x in data["subscribers"] if isinstance(x, dict)]
    except Exception:
        pass
    return []


def _save_local(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_local_env_values(base: Path) -> dict[str, str]:
    """讀取專案 .env 中的設定；系統環境變數仍具有更高優先權。

    Streamlit Cloud 會由 stock_dashboard.py 將 st.secrets 載入環境變數；
    Windows Task Scheduler 則不一定會繼承互動式 PowerShell 的臨時變數，
    因此共享訂閱名單設定也必須能從專案 .env 讀取。
    """
    values: dict[str, str] = {}
    path = base / ".env"
    if not path.exists():
        return values

    try:
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key] = value
    except Exception:
        return {}

    return values


def _setting(base: Path, name: str) -> str:
    """先讀系統環境變數，再回退至專案 .env。"""
    value = os.getenv(name, "").strip()
    if value:
        return value
    return _load_local_env_values(base).get(name, "").strip()


def _remote_url(base: Path) -> str:
    return _setting(base, "SUBSCRIBER_SYNC_URL")


def _remote_token(base: Path) -> str:
    return _setting(base, "SUBSCRIBER_SYNC_TOKEN")


def _remote_headers() -> dict[str, str]:
    return {"Content-Type": "application/json; charset=utf-8"}


def load_remote_subscribers(base: str | Path = ".") -> tuple[list[dict[str, Any]], str]:
    base = Path(base).resolve()
    url = _remote_url(base)
    token = _remote_token(base)
    if not url:
        return [], ""
    try:
        params = {"action": "list"}
        if token:
            params["token"] = token
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return [], "遠端訂閱服務回傳格式錯誤（預期 JSON 物件）"
        if data.get("ok") is False:
            reason = str(data.get("error") or "unknown_error")
            return [], f"遠端訂閱服務拒絕請求：{reason}；請確認 Web App 部署與 Token"
        rows = data.get("subscribers", [])
        if isinstance(rows, list):
            return [x for x in rows if isinstance(x, dict)], ""
        return [], "遠端訂閱服務回傳格式錯誤（subscribers 不是清單）"
    except Exception as exc:
        return [], f"遠端訂閱名單讀取失敗：{type(exc).__name__}: {exc}"


def load_subscribers(base: str | Path = ".", include_remote: bool = True) -> list[dict[str, Any]]:
    base = Path(base).resolve()
    local_path = base / "data" / "subscribers.json"
    local = _load_local(local_path)
    remote, _ = load_remote_subscribers(base) if include_remote else ([], "")

    merged: dict[str, dict[str, Any]] = {}
    for row in local + remote:
        email = _clean_email(row.get("email"))
        if not EMAIL_RE.match(email):
            continue
        old = merged.get(email)
        if old is None:
            merged[email] = {**row, "email": email}
            continue
        # 以 updated_at 較新的來源為準，避免雲端舊紀錄覆蓋本機剛更新的訂閱狀態。
        def stamp(v: Any) -> str:
            return str(v or "")
        if stamp(row.get("updated_at")) >= stamp(old.get("updated_at")):
            merged[email] = {**old, **row, "email": email}
        else:
            merged[email] = {**row, **old, "email": email}

    return [
        row for row in merged.values()
        if bool(row.get("active", row.get("subscribed", True)))
    ]


def report_recipient_emails(base: str | Path = ".") -> list[str]:
    return sorted({
        _clean_email(x.get("email"))
        for x in load_subscribers(base, include_remote=True)
        if EMAIL_RE.match(_clean_email(x.get("email")))
    })


def register_subscriber(
    base: str | Path,
    *,
    name: str,
    email: str,
    message: str,
    subscribed: bool,
) -> dict[str, Any]:
    base = Path(base).resolve()
    clean_name = re.sub(r"[\r\n\t]+", " ", str(name or "")).strip()[:120]  # 姓名會用在信件標題，不能含換行
    clean_email = _clean_email(email)
    clean_message = str(message or "").strip()[:5000] or ("訂閱每日晨報與盤後分析" if subscribed else "（未留下訊息）")

    if not clean_name or not clean_email:
        return {"saved": False, "error": "姓名與電子郵件不可為空白。"}
    if not EMAIL_RE.match(clean_email):
        return {"saved": False, "error": "電子郵件格式不正確。"}

    path = base / "data" / "subscribers.json"
    rows = _load_local(path)
    now = _now_iso()
    requested = bool(subscribed)
    existing = next((r for r in rows if _clean_email(r.get("email")) == clean_email), None)
    was_active = bool(existing and existing.get("active", existing.get("subscribed", False)))
    # 只是留言（沒勾訂閱）不會取消既有訂閱；也不會把沒訂閱的人加進收件名單
    active = requested or was_active

    record = {
        "name": clean_name,
        "email": clean_email,
        "message": clean_message,
        "subscribed": active,
        "active": active,
        "updated_at": now,
        "persistence": "local",
        "persistence_warning": "",
    }

    replaced = False
    for idx, row in enumerate(rows):
        if _clean_email(row.get("email")) == clean_email:
            record["subscribed_at"] = row.get("subscribed_at") or now
            rows[idx] = {**row, **record}
            replaced = True
            break

    if not replaced:
        record["subscribed_at"] = now
        record["created_at"] = now
        rows.append(record)

    _save_local(path, rows)

    result: dict[str, Any] = {
        "saved": True,
        "remote_saved": False,
        "owner_notified": False,
        "owner_notify_error": "",
        "email": clean_email,
        "subscribed": active,
        "requested_subscribe": requested,
        "already_subscribed": bool(requested and was_active),
        "welcome_sent": False,
        "welcome_error": "",
        "updated_at": now,
    }

    # 雲端／本機共同持久化：若有設定 Apps Script URL，送到同一份遠端名單。
    url = _remote_url(base)
    if url:
        payload = {
            "action": "upsert",
            "token": _remote_token(base),
            "name": clean_name,
            "email": clean_email,
            "message": clean_message,
            "subscribed": active,
            "notify_owner": True,
            "source": "streamlit_subscription",
            "updated_at": now,
        }
        try:
            response = requests.post(
                url,
                params={"token": _remote_token(base)} if _remote_token(base) else None,
                headers=_remote_headers(),
                data=json.dumps(payload, ensure_ascii=False),
                timeout=20,
            )
            response.raise_for_status()
            remote_data = response.json()
            result["remote_saved"] = bool(
                isinstance(remote_data, dict) and remote_data.get("ok", remote_data.get("saved", False))
            )
            if result["remote_saved"]:
                result["persistence"] = "local+remote"
            result["owner_notified"] = bool(
                isinstance(remote_data, dict) and remote_data.get("owner_notified", False)
            )
            if isinstance(remote_data, dict) and remote_data.get("error"):
                result["owner_notify_error"] = str(remote_data["error"])
        except Exception as exc:
            result["remote_saved"] = False
            result["persistence_warning"] = "SUBSCRIBER_SYNC_URL 已設定但遠端保存失敗；本機資料仍已保存。"
            result["owner_notify_error"] = f"{type(exc).__name__}: {exc}"

    if not url:
        result["persistence_warning"] = "未設定 SUBSCRIBER_SYNC_URL；若網站部署在 Streamlit Cloud，建議啟用 Google Apps Script 共享名單以避免重新部署後本機檔案遺失。"

    # 沒有遠端服務或遠端失敗時，使用本機 SMTP 通知網站管理者。
    if not result["owner_notified"]:
        try:
            from email_agent import EmailAgent
            notice = EmailAgent(base).send_subscription_notice(
                name=clean_name,
                email=clean_email,
                message=clean_message,
                subscribed=active,
            )
            if notice.get("sent"):
                result["owner_notified"] = True
            else:
                result["owner_notify_error"] = str(
                    notice.get("error") or notice.get("reason") or "管理者通知未寄出"
                )
        except Exception as exc:
            result["owner_notify_error"] = f"{type(exc).__name__}: {exc}"

    # 新訂閱（或重新訂閱）才寄「訂閱成功」確認信；已在名單內的人不會重複收到
    if requested and not was_active:
        try:
            from email_agent import EmailAgent
            welcome = EmailAgent(base).send_subscription_welcome(name=clean_name, email=clean_email)
            result["welcome_sent"] = bool(welcome.get("sent"))
            if not welcome.get("sent"):
                result["welcome_error"] = str(welcome.get("error") or "訂閱成功信未寄出")
        except Exception as exc:
            result["welcome_error"] = f"{type(exc).__name__}: {exc}"

    return result


def subscription_status(base: str | Path = ".") -> dict[str, Any]:
    rows = load_subscribers(base, include_remote=True)
    return {
        "active_count": len(rows),
        "emails": [str(x.get("email")) for x in rows],
    }


def deactivate_subscriber(base: str | Path, email: str) -> dict[str, Any]:
    base = Path(base).resolve()
    clean_email = _clean_email(email)
    path = base / "data" / "subscribers.json"
    rows = _load_local(path)
    changed = False

    for row in rows:
        if _clean_email(row.get("email")) == clean_email:
            row["active"] = False
            row["subscribed"] = False
            row["updated_at"] = _now_iso()
            changed = True

    _save_local(path, rows)
    return {"changed": changed, "email": clean_email}
