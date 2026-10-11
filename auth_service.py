# -*- coding: utf-8 -*-
"""網站登入、簽章登入工作階段與帳號申請服務。

帳號資料只從 AUTH_USERS_JSON 讀取，不寫進 GitHub 程式碼。
密碼使用 PBKDF2-HMAC-SHA256；登入工作階段使用 HMAC 簽章 cookie，
讓使用者切換網站分頁或 Streamlit 重新建立 session 後仍能保持登入。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any

EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")
ITERATIONS = 310_000
ADMIN_EMAIL = "a1113359@mail.nuk.edu.tw"
SESSION_COOKIE_NAME = "stock_trading_auth_v1"
SESSION_TTL_SECONDS = 14 * 24 * 60 * 60


def _users() -> dict[str, Any]:
    raw = os.getenv("AUTH_USERS_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {str(k).strip().lower(): v for k, v in data.items()} if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def hash_password(password: str, salt_hex: str | None = None, iterations: int = ITERATIONS) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return "pbkdf2_sha256$" + str(int(iterations)) + "$" + salt.hex() + "$" + digest.hex()


def verify_password(password: str, record: Any) -> bool:
    if not isinstance(record, str):
        return False
    try:
        algorithm, rounds, salt_hex, expected_hex = record.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hash_password(password, salt_hex=salt_hex, iterations=int(rounds)).split("$", 3)[3]
        return hmac.compare_digest(actual, expected_hex)
    except (ValueError, TypeError):
        return False


def authenticate(email: str, password: str) -> bool:
    email = str(email or "").strip().lower()
    return bool(EMAIL_RE.fullmatch(email) and password and verify_password(password, _users().get(email)))


def _session_signing_key() -> bytes:
    """優先使用獨立的 AUTH_SESSION_SECRET；未設定時以帳號設定導出相容性金鑰。"""
    secret = os.getenv("AUTH_SESSION_SECRET", "").strip()
    if secret:
        return secret.encode("utf-8")
    # 相容尚未加上 AUTH_SESSION_SECRET 的舊 Secrets。更新帳號清單會使舊 cookie 失效。
    source = os.getenv("AUTH_USERS_JSON", "").strip()
    return hashlib.sha256(("stock-trading-session-v1:" + source).encode("utf-8")).digest()


def create_session_token(email: str, now: int | None = None) -> str:
    """建立有效 14 天的簽章工作階段 token，不包含明文密碼。"""
    clean_email = str(email or "").strip().lower()
    users = _users()
    record = users.get(clean_email)
    if not record or not isinstance(record, str):
        raise ValueError("帳號不存在，無法建立登入工作階段。")
    issued_at = int(time.time() if now is None else now)
    payload = {
        "email": clean_email,
        "exp": issued_at + SESSION_TTL_SECONDS,
        "credential_hash": hashlib.sha256(record.encode("utf-8")).hexdigest(),
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_part = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = hmac.new(_session_signing_key(), payload_part.encode("ascii"), hashlib.sha256).hexdigest()
    return payload_part + "." + signature


def verify_session_token(token: str | None, now: int | None = None) -> str | None:
    """驗證 cookie 簽章、期限及帳號密碼雜湊；無效時回傳 None。"""
    if not token or not isinstance(token, str) or "." not in token:
        return None
    try:
        payload_part, supplied_signature = token.rsplit(".", 1)
        expected_signature = hmac.new(
            _session_signing_key(), payload_part.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None
        padded = payload_part + "=" * (-len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        email = str(payload.get("email") or "").strip().lower()
        expires_at = int(payload.get("exp") or 0)
        current = int(time.time() if now is None else now)
        record = _users().get(email)
        if current >= expires_at or not record or not isinstance(record, str):
            return None
        current_credential_hash = hashlib.sha256(record.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(str(payload.get("credential_hash") or ""), current_credential_hash):
            return None
        return email if EMAIL_RE.fullmatch(email) else None
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def request_account(email: str, base_dir: str | Path) -> dict[str, Any]:
    """寄送帳號申請通知；不會自動建立帳號或密碼。"""
    email = str(email or "").strip().lower()
    if not EMAIL_RE.fullmatch(email):
        return {"sent": False, "error": "請輸入有效的電子郵件地址。"}
    previous_admin = os.environ.get("EMAIL_ADMIN_RECIPIENT")
    try:
        from email_agent import EmailAgent
        agent = EmailAgent(base_dir)
        if not agent.status().get("configured"):
            return {
                "sent": False,
                "error": "管理員通知 Email 尚未設定，請檢查 EMAIL_SENDER、EMAIL_APP_PASSWORD 與 EMAIL_RECIPIENT。",
            }
        os.environ["EMAIL_ADMIN_RECIPIENT"] = ADMIN_EMAIL
        result = agent.send_subscription_notice(
            name="網站帳號申請",
            email=email,
            message="使用者申請網站登入帳號。請管理員確認後，手動將帳號與密碼雜湊加入 Streamlit Secrets 的 AUTH_USERS_JSON；系統不會自動建立帳號。",
            subscribed=False,
        )
        if result.get("sent"):
            return {"sent": True}
        return {"sent": False, "error": str(result.get("error") or "管理員通知寄送失敗。")}
    except Exception as exc:
        return {"sent": False, "error": "寄送申請通知失敗：" + type(exc).__name__ + ": " + str(exc)}
    finally:
        if previous_admin is None:
            os.environ.pop("EMAIL_ADMIN_RECIPIENT", None)
        else:
            os.environ["EMAIL_ADMIN_RECIPIENT"] = previous_admin


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "hash-password":
    email = sys.argv[2].strip().lower()
    if not EMAIL_RE.fullmatch(email):
        raise SystemExit("Email 格式不正確")
    import getpass
    p1 = getpass.getpass("請輸入要設定的密碼：")
    p2 = getpass.getpass("請再次輸入密碼：")
    if p1 != p2:
        raise SystemExit("兩次密碼不一致")
    if len(p1) < 10:
        raise SystemExit("密碼至少需要 10 個字元")
    print(json.dumps({email: hash_password(p1)}, ensure_ascii=False))
