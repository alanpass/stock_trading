# -*- coding: utf-8 -*-
"""唯讀測試本機 stock_dashboard 專案能否連線至 Supabase account_requests。

用法：
  cd <專案根目錄>
  python test_local_supabase_connection.py
  python test_local_supabase_connection.py --project-dir "C:\\...\\stock_dashboard"

不修改資料、不列印 service-role key、Email 清單或 password_hash。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def normalize_project_url(raw_url: str) -> str:
    value = str(raw_url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"https", "http"} or not parsed.netloc:
        raise ValueError("ACCOUNT_DB_URL 不是有效的完整網址。")
    if parsed.username or parsed.password:
        raise ValueError("ACCOUNT_DB_URL 不應包含帳號或密碼。")
    # 與本專案 account_provisioning.py 的 URL 處理邏輯一致：
    # 若誤貼 /rest/v1 或 /rest/v1/account_requests，自動去除 REST 路徑。
    parts = [x for x in parsed.path.split("/") if x]
    rest_index = next(
        (i for i in range(len(parts) - 1)
         if parts[i].lower() == "rest" and parts[i + 1].lower() == "v1"),
        None,
    )
    if rest_index is not None:
        parts = parts[:rest_index]
    elif parts:
        raise ValueError(
            "ACCOUNT_DB_URL 含有非預期的路徑。請填 Supabase Project URL，"
            "不要貼 Table Editor 網址。"
        )
    base_path = "/" + "/".join(parts) if parts else ""
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, base_path, "", "")).rstrip("/")


def safe_error_body(response) -> str:
    """只顯示 API 錯誤訊息；絕不顯示任何 request header/key。"""
    try:
        body = response.json()
        if isinstance(body, dict):
            allowed = {k: body[k] for k in ("code", "message", "details", "hint") if body.get(k)}
            return json.dumps(allowed or {"response": "Supabase 回傳 JSON 錯誤"}, ensure_ascii=False)[:1200]
        return json.dumps({"response": "Supabase 回傳非預期 JSON 格式"}, ensure_ascii=False)
    except Exception:
        # 不列印 HTML 內文，避免輸出意外包含敏感資訊。
        return "伺服器未回傳可解析的 JSON 錯誤內容。"


def main() -> int:
    parser = argparse.ArgumentParser(description="唯讀測試本機專案與 Supabase 的連線")
    parser.add_argument(
        "--project-dir", default=str(Path.cwd()),
        help="本機 stock_dashboard 專案根目錄；預設為目前 PowerShell 所在目錄",
    )
    args = parser.parse_args()
    root = Path(args.project_dir).expanduser().resolve()
    secrets_path = root / ".streamlit" / "secrets.toml"

    print("=" * 68)
    print("本機 Supabase 連線測試（唯讀）")
    print(f"專案目錄：{root}")
    print(f"Secrets 檔案：{secrets_path}")

    if not root.exists() or not root.is_dir():
        print("結果：失敗 — 專案目錄不存在，請檢查 --project-dir 路徑。")
        return 2
    if not secrets_path.is_file():
        print("結果：失敗 — 找不到 .streamlit/secrets.toml。")
        print("請確認 Secrets 放在這個專案目錄底下的 .streamlit 資料夾。")
        return 2

    try:
        import tomllib  # Python 3.11+
        with secrets_path.open("rb") as f:
            secrets = tomllib.load(f)
    except ModuleNotFoundError:
        print("結果：失敗 — 目前 Python 版本低於 3.11，缺少內建 tomllib。")
        print("請使用專案的 Python 3.11+ 執行本工具。")
        return 2
    except Exception as exc:
        print(f"結果：失敗 — secrets.toml 解析錯誤：{type(exc).__name__}: {exc}")
        return 2

    # 先取本專案 secrets.toml；如果不存在才以同名環境變數作為備援。
    db_url_raw = str(secrets.get("ACCOUNT_DB_URL") or os.getenv("ACCOUNT_DB_URL", "")).strip()
    service_key = str(
        secrets.get("ACCOUNT_DB_SERVICE_ROLE_KEY")
        or os.getenv("ACCOUNT_DB_SERVICE_ROLE_KEY", "")
    ).strip()

    print(f"ACCOUNT_DB_URL：{'已設定' if db_url_raw else '未設定'}")
    print(f"ACCOUNT_DB_SERVICE_ROLE_KEY：{'已設定' if service_key else '未設定'}")
    if not db_url_raw or not service_key:
        print("結果：失敗 — 缺少必要設定。請檢查 TOML key 名稱是否完全一致。")
        return 2

    try:
        base_url = normalize_project_url(db_url_raw)
    except Exception as exc:
        print(f"結果：失敗 — URL 格式不正確：{exc}")
        print("提示：ACCOUNT_DB_URL 應是 https://<project-ref>.supabase.co，不是 REST API 或資料表頁面網址。")
        return 2

    endpoint = f"{base_url}/rest/v1/account_requests"
    print(f"測試端點：{base_url}/rest/v1/account_requests")
    print("正在發出 GET 請求（不會新增、更新或刪除資料）……")

    try:
        import requests
    except ImportError:
        print("結果：失敗 — 目前 Python 環境沒有 requests。請執行 python -m pip install requests。")
        return 2

    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Accept": "application/json",
    }
    # 查詢登入程式會用到的欄位，但不輸出資料列內容或 password_hash。
    params = {
        "select": "email,status,password_hash,activate_at,created_at",
        "limit": "1",
    }
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=15)
    except requests.exceptions.Timeout:
        print("結果：失敗 — 連線逾時（15 秒）。請檢查網路、DNS 或 Supabase 服務狀態。")
        return 3
    except requests.exceptions.SSLError:
        print("結果：失敗 — TLS/SSL 憑證驗證失敗。請檢查系統憑證、代理或網路安全軟體。")
        return 3
    except requests.exceptions.ConnectionError as exc:
        print(f"結果：失敗 — 無法建立 HTTPS 連線：{type(exc).__name__}")
        print("請檢查網路、網址、DNS 或防火牆。沒有輸出 URL 中的任何金鑰資訊。")
        return 3
    except requests.exceptions.RequestException as exc:
        print(f"結果：失敗 — HTTP 請求發生問題：{type(exc).__name__}")
        return 3

    print(f"HTTP 狀態碼：{response.status_code}")
    if response.status_code == 200:
        try:
            rows = response.json()
        except ValueError:
            print("結果：部分成功 — HTTP 200，但回應不是 JSON。")
            return 4
        if not isinstance(rows, list):
            print("結果：部分成功 — API 已回應，但格式不是預期的資料列清單。")
            return 4
        print("結果：成功 — 本機設定可連到 Supabase，API 金鑰有效，account_requests 資料表及登入所需欄位可讀取。")
        print(f"查詢回傳資料列數：{len(rows)}（僅顯示數量，不顯示帳號或雜湊）")
        print("注意：這證明本機這份 secrets.toml 可連線；若網站仍登入失敗，下一步要查實際登入的帳號狀態／密碼驗證。")
        return 0

    print("API 錯誤摘要：" + safe_error_body(response))
    if response.status_code in (401, 403):
        print("判讀：金鑰無效、金鑰屬於另一個 Supabase 專案，或目前金鑰權限不足。")
    elif response.status_code == 404:
        print("判讀：可能是 Project URL 錯誤、REST 路徑錯誤，或 account_requests 資料表／API 路由不存在。")
    elif response.status_code == 400:
        print("判讀：常見原因是資料表欄位與登入程式預期不一致，請依上方 code/message 檢查 schema。")
    elif response.status_code == 429:
        print("判讀：請求頻率受到限制，稍後再試並確認是否有程式重複輪詢。")
    else:
        print("判讀：請依 HTTP 狀態與上方錯誤摘要檢查 Supabase 設定或服務狀態。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
