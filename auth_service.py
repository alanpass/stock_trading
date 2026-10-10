# -*- coding: utf-8 -*-
"""網站登入與帳號申請服務。帳號資料只從 AUTH_USERS_JSON 讀取，不寫進 GitHub 程式碼。
格式：{"email@example.com":"pbkdf2_sha256$迭代次數$salt_hex$digest_hex"}
使用 python auth_service.py hash-password user@example.com 產生密碼雜湊。
"""
from __future__ import annotations
import hashlib, hmac, json, os, re, secrets, sys
from pathlib import Path
from typing import Any

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ITERATIONS = 310_000
ADMIN_EMAIL = "a1113359@mail.nuk.edu.tw"

def _users() -> dict[str, Any]:
    raw = os.getenv("AUTH_USERS_JSON", "").strip()
    if not raw: return {}
    try:
        data = json.loads(raw)
        return {str(k).strip().lower(): v for k, v in data.items()} if isinstance(data, dict) else {}
    except (TypeError, ValueError): return {}

def hash_password(password: str, salt_hex: str | None = None, iterations: int = ITERATIONS) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return "pbkdf2_sha256$" + str(int(iterations)) + "$" + salt.hex() + "$" + digest.hex()

def verify_password(password: str, record: Any) -> bool:
    if not isinstance(record, str): return False
    try:
        algorithm, rounds, salt_hex, expected_hex = record.split("$", 3)
        if algorithm != "pbkdf2_sha256": return False
        actual = hash_password(password, salt_hex=salt_hex, iterations=int(rounds)).split("$", 3)[3]
        return hmac.compare_digest(actual, expected_hex)
    except (ValueError, TypeError): return False

def authenticate(email: str, password: str) -> bool:
    email = str(email or "").strip().lower()
    return bool(EMAIL_RE.fullmatch(email) and password and verify_password(password, _users().get(email)))

def request_account(email: str, base_dir: str | Path) -> dict[str, Any]:
    """寄送帳號申請通知；不會自動建立帳號或密碼。"""
    email = str(email or "").strip().lower()
    if not EMAIL_RE.fullmatch(email): return {"sent": False, "error": "請輸入有效的電子郵件地址。"}
    try:
        from email_agent import EmailAgent
        agent = EmailAgent(base_dir)
        if not agent.status().get("configured"):
            return {"sent": False, "error": "管理員通知 Email 尚未設定，請檢查 EMAIL_SENDER、EMAIL_APP_PASSWORD 與 EMAIL_RECIPIENT。"}
        previous_admin = os.environ.get("EMAIL_ADMIN_RECIPIENT")
        os.environ["EMAIL_ADMIN_RECIPIENT"] = ADMIN_EMAIL
        result = agent.send_subscription_notice(name="網站帳號申請", email=email, message="使用者申請網站登入帳號。請管理員確認後，手動將帳號與密碼雜湊加入 Streamlit Secrets 的 AUTH_USERS_JSON；系統不會自動建立帳號。", subscribed=False)
        if previous_admin is None:
            os.environ.pop("EMAIL_ADMIN_RECIPIENT", None)
        else:
            os.environ["EMAIL_ADMIN_RECIPIENT"] = previous_admin
        return {"sent": True} if result.get("sent") else {"sent": False, "error": str(result.get("error") or "管理員通知寄送失敗。")}
    except Exception as exc:
        return {"sent": False, "error": "寄送申請通知失敗：" + type(exc).__name__ + ": " + str(exc)}

if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "hash-password":
    email = sys.argv[2].strip().lower()
    if not EMAIL_RE.fullmatch(email): raise SystemExit("Email 格式不正確")
    import getpass
    p1 = getpass.getpass("請輸入要設定的密碼：")
    p2 = getpass.getpass("請再次輸入密碼：")
    if p1 != p2: raise SystemExit("兩次密碼不一致")
    if len(p1) < 10: raise SystemExit("密碼至少需要 10 個字元")
    print(json.dumps({email: hash_password(p1)}, ensure_ascii=False))
