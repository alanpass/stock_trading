# -*- coding: utf-8 -*-
"""雲端帳號申請／審核／啟用服務。"""
from __future__ import annotations
import hashlib, os, secrets, string
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote
import requests

TABLE = "account_requests"
PASSWORD_ITERATIONS = 310_000
TOKEN_TTL_HOURS = 48
ACTIVATION_DELAY_MINUTES = 5
TIMEOUT_SECONDS = 15

class AccountProvisioningError(RuntimeError):
    """帳號申請／審核服務錯誤。"""

def _config() -> tuple[str, str]:
    url = os.getenv("ACCOUNT_DB_URL", "").strip().rstrip("/")
    key = os.getenv("ACCOUNT_DB_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise AccountProvisioningError("尚未設定 ACCOUNT_DB_URL 與 ACCOUNT_DB_SERVICE_ROLE_KEY；請先設定 Supabase 永久資料庫。")
    return url, key

def is_configured() -> bool:
    return bool(os.getenv("ACCOUNT_DB_URL", "").strip() and os.getenv("ACCOUNT_DB_SERVICE_ROLE_KEY", "").strip())

def _headers(prefer: str | None = None) -> dict[str, str]:
    _, key = _config()
    result = {"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json"}
    if prefer:
        result["Prefer"] = prefer
    return result

def _request(method: str, params: dict[str, str] | None = None, payload: dict[str, Any] | None = None,
             prefer: str | None = None) -> list[dict[str, Any]]:
    url, _ = _config()
    try:
        response = requests.request(method, f"{url}/rest/v1/{TABLE}", params=params, json=payload,
                                    headers=_headers(prefer), timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise AccountProvisioningError(f"連線 Supabase 失敗：{type(exc).__name__}") from exc
    if response.status_code not in (200, 201, 204):
        raise AccountProvisioningError(f"Supabase 回應 HTTP {response.status_code}：{response.text[:500]}")
    if not response.text.strip():
        return []
    try:
        data = response.json()
    except ValueError as exc:
        raise AccountProvisioningError("Supabase 回傳非預期格式。") from exc
    return data if isinstance(data, list) else []

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return "pbkdf2_sha256$" + str(PASSWORD_ITERATIONS) + "$" + salt.hex() + "$" + digest.hex()

def _generate_password(length: int = 8) -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.isalpha() for c in password) and any(c.isdigit() for c in password):
            return password

def create_account_request(email: str) -> dict[str, Any]:
    """建立待審核申請並回傳兩條一次性審核連結。"""
    clean_email = str(email or "").strip().lower()
    if not clean_email or len(clean_email) > 254 or "@" not in clean_email:
        raise AccountProvisioningError("申請者電子郵件格式不正確。")
    existing = _request("GET", params={"select": "id,email,status,created_at", "email": f"eq.{clean_email}",
                                       "status": "in.(pending,approved)", "limit": "1"})
    if existing:
        if str(existing[0].get("status")) == "pending":
            raise AccountProvisioningError("此電子郵件已有待審核申請，請勿重複送出。")
        raise AccountProvisioningError("此電子郵件已有已核准帳號紀錄。")
    approve_token, reject_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    now = _utc_now()
    row = {"email": clean_email, "status": "pending",
           "approve_token_hash": _hash_token(approve_token), "reject_token_hash": _hash_token(reject_token),
           "token_expires_at": _iso(now + timedelta(hours=TOKEN_TTL_HOURS)), "created_at": _iso(now)}
    created = _request("POST", payload=row, prefer="return=representation")
    if not created:
        raise AccountProvisioningError("資料庫未回傳新申請紀錄，請檢查資料表設定。")
    return {"request_id": created[0].get("id"), "email": clean_email, "approve_token": approve_token,
            "reject_token": reject_token, "expires_at": row["token_expires_at"]}

def get_request_for_token(token: str, decision: str) -> dict[str, Any] | None:
    """只查詢有效待審核 token；不修改資料庫。"""
    if decision not in {"approve", "reject"} or not token or len(token) > 200:
        return None
    field = "approve_token_hash" if decision == "approve" else "reject_token_hash"
    rows = _request("GET", params={"select": "id,email,status,token_expires_at,created_at",
        field: f"eq.{_hash_token(token)}", "status": "eq.pending",
        "token_expires_at": f"gt.{_iso(_utc_now())}", "limit": "1"})
    return rows[0] if rows else None

def decide_account_request(token: str, decision: str) -> dict[str, Any]:
    """一次性核准或拒絕；條件更新避免重複審核。"""
    if decision not in {"approve", "reject"} or not token or len(token) > 200:
        raise AccountProvisioningError("審核連結無效。")
    now = _utc_now()
    token_field = "approve_token_hash" if decision == "approve" else "reject_token_hash"
    payload: dict[str, Any] = {"status": "approved" if decision == "approve" else "rejected",
        "decided_at": _iso(now), "approve_token_hash": None, "reject_token_hash": None}
    password, password_hash, activation_at = "", "", None
    if decision == "approve":
        password = _generate_password(8)
        password_hash = _hash_password(password)
        activation_at = now + timedelta(minutes=ACTIVATION_DELAY_MINUTES)
        payload.update({"password_hash": password_hash, "approved_at": _iso(now), "activate_at": _iso(activation_at)})
    rows = _request("PATCH", params={token_field: f"eq.{_hash_token(token)}", "status": "eq.pending",
        "token_expires_at": f"gt.{_iso(now)}",
        "select": "id,email,status,created_at,decided_at,approved_at,activate_at"},
        payload=payload, prefer="return=representation")
    if not rows:
        raise AccountProvisioningError("此審核連結已失效、已逾期，或已被使用。")
    result = dict(rows[0])
    result["decision"] = decision
    if decision == "approve":
        result.update({"password": password, "password_hash": password_hash, "activate_at": _iso(activation_at)})
    return result

def account_login_state(email: str) -> dict[str, Any]:
    """回報帳號是否受資料庫管控及是否已到啟用時間；舊帳號維持原本行為。"""
    if not is_configured():
        return {"found": False, "configured": False, "allowed": None}
    clean_email = str(email or "").strip().lower()
    try:
        rows = _request("GET", params={"select": "email,status,activate_at", "email": f"eq.{clean_email}", "limit": "1"})
    except AccountProvisioningError as exc:
        return {"found": False, "configured": True, "allowed": False, "error": str(exc)}
    if not rows:
        return {"found": False, "configured": True, "allowed": None}
    row = rows[0]
    if row.get("status") != "approved" or not row.get("activate_at"):
        return {"found": True, "configured": True, "allowed": False, "status": row.get("status")}
    try:
        activate_at = datetime.fromisoformat(str(row["activate_at"]).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return {"found": True, "configured": True, "allowed": False, "error": "啟用時間資料格式錯誤。"}
    return {"found": True, "configured": True, "allowed": _utc_now() >= activate_at,
            "status": row.get("status"), "activate_at": str(row["activate_at"])}

def public_review_url(decision: str, token: str) -> str:
    base = os.getenv("ACCOUNT_PUBLIC_BASE_URL",
        "https://stocktrading-c66hschtsd2qh6vkyv4mtq.streamlit.app/").strip()
    if decision not in {"approve", "reject"}:
        raise AccountProvisioningError("審核動作不正確。")
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}account_review={quote(decision)}&token={quote(token)}"
