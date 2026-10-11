# -*- coding: utf-8 -*-
"""只修正本機 stock_trading 專案的登入／帳號申請功能，不會寫入 GitHub。

用法：
  python repair_local_login.py --project-dir "C:\\Users\\10501\\OneDrive\\Desktop\\stock_trading\\stock_dashboard"

此腳本只會對 GitHub 使用 HTTPS GET 下載指定 commit 的檔案，
並在本機備份後修補 stock_dashboard.py、email_agent.py、requirements.txt。
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 鎖定已檢查過、含自動 Supabase 帳號憑證功能的 main commit；避免下載未來變動版本。
REPO = "alanpass/stock_trading"
COMMIT = "a7f0e7a00b15fab6a26efa6557638503243172b3"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"

SUPPORT_FILES = (
    "auth_service.py",
    "account_provisioning.py",
    "account_provisioning_schema.sql",
    "ACCOUNT_LOGIN_SETUP.md",
    "ACCOUNT_PROVISIONING_SETUP.md",
)

AUTH_SECRET_KEYS = (
    "AUTH_USERS_JSON",
    "AUTH_SESSION_SECRET",
    "ACCOUNT_DB_URL",
    "ACCOUNT_DB_SERVICE_ROLE_KEY",
    "ACCOUNT_PUBLIC_BASE_URL",
)


def say(message: str) -> None:
    print(message, flush=True)


def download_text(path: str) -> str:
    url = RAW_BASE + path
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "stock-trading-local-login-repair/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"下載 {path} 失敗。請確認本機可連線 raw.githubusercontent.com。\n詳細資訊：{exc}"
        ) from exc
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"GitHub 檔案 {path} 不是可讀取的 UTF-8 文字檔。") from exc


def extract_between(source: str, start_marker: str, end_marker: str, label: str) -> str:
    start = source.find(start_marker)
    if start < 0:
        raise RuntimeError(f"在參考版本中找不到 {label} 的起始標記：{start_marker!r}")
    end = source.find(end_marker, start + len(start_marker))
    if end < 0:
        raise RuntimeError(f"在參考版本中找不到 {label} 的結束標記：{end_marker!r}")
    return source[start:end].rstrip() + "\n\n"


def add_cloud_secret_keys(text: str) -> str:
    if "def _load_cloud_secret_env()" not in text or "secret_keys = [" not in text:
        raise RuntimeError("stock_dashboard.py 找不到預期的 _load_cloud_secret_env()/secret_keys 區塊；已停止，未覆寫檔案。")
    match = re.search(r"(?s)(secret_keys\s*=\s*\[)(.*?)(\])", text)
    if not match:
        raise RuntimeError("無法解析 stock_dashboard.py 的 secret_keys 清單；已停止，未覆寫檔案。")
    body = match.group(2)
    additions = [key for key in AUTH_SECRET_KEYS if not re.search(rf"['\"]{re.escape(key)}['\"]", body)]
    if not additions:
        return text
    # 與專案現有清單相同，使用 12 格縮排；不要碰觸實際 Secrets 值。
    body = body.rstrip() + "\n" + "\n".join(f'            "{key}",' for key in additions) + "\n        "
    return text[:match.start(2)] + body + text[match.end(2):]


def add_imports(text: str) -> str:
    if "import extra_streamlit_components as stx" not in text:
        anchor = "import streamlit as st\n"
        if anchor not in text:
            raise RuntimeError("找不到 `import streamlit as st`，無法加入 CookieManager 依賴。")
        text = text.replace(anchor, anchor + "import extra_streamlit_components as stx\n", 1)

    # CookieManager helper 會使用 timedelta 設定 Cookie 有效期。
    if "from datetime import datetime, time as dt_time" in text:
        text = text.replace(
            "from datetime import datetime, time as dt_time",
            "from datetime import datetime, timedelta, time as dt_time",
            1,
        )
    elif "from datetime import datetime, timedelta, time as dt_time" not in text and "timedelta" not in text:
        raise RuntimeError("無法安全調整 datetime 匯入；已停止，未覆寫檔案。")

    auth_import = (
        "from auth_service import (authenticate, request_account, ADMIN_EMAIL, create_session_token,\n"
        "                         verify_session_token, SESSION_COOKIE_NAME, SESSION_TTL_SECONDS)\n"
    )
    if "from auth_service import" not in text:
        sub_import = "from subscriber_service import register_subscriber, subscription_status\n"
        if sub_import not in text:
            raise RuntimeError("找不到訂閱服務匯入列，無法安全加入登入服務匯入。")
        text = text.replace(sub_import, sub_import + auth_import, 1)
    return text


def guard_add_watchlist(text: str) -> str:
    old = '''        if st.button("加入自選股", key=f"add_watch_site_{code}"):
            st.session_state.watchlist.append(code)
            st.session_state.watchlist = list(dict.fromkeys(st.session_state.watchlist))
            save_watchlist(st.session_state.watchlist)
            st.rerun()
'''
    new = '''        if is_authenticated():
            if st.button("加入自選股", key=f"add_watch_site_{code}"):
                st.session_state.watchlist.append(code)
                st.session_state.watchlist = list(dict.fromkeys(st.session_state.watchlist))
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        else:
            st.caption("登入後才能新增自選股。")
'''
    if "登入後才能新增自選股。" in text and "if is_authenticated():\n            if st.button(\"加入自選股\"" in text:
        return text
    if old not in text:
        raise RuntimeError("找不到新增自選股的預期程式區塊；已停止，避免套用不完整的權限修正。")
    return text.replace(old, new, 1)


def guard_remove_watchlist(text: str) -> str:
    marker = "def _watch_remove(code: str) -> None:\n"
    guard = (
        marker
        + '    if not is_authenticated():\n'
        + '        st.warning("請先登入，才能刪除自選股。")\n'
        + '        return\n'
    )
    if 'st.warning("請先登入，才能刪除自選股。")' in text:
        return text
    if marker not in text:
        raise RuntimeError("找不到自選股刪除函式 _watch_remove；已停止，未覆寫檔案。")
    return text.replace(marker, guard, 1)


def guard_future_page(text: str) -> str:
    old = '''elif current_section == "future":
    render_future_workspace()
'''
    new = '''elif current_section == "future":
    if is_authenticated():
        render_future_workspace()
    else:
        st.markdown('<div class="content-heading"><h1>未來分析</h1><p>此功能僅限登入使用者。</p></div>', unsafe_allow_html=True)
        st.info("請點擊頁面右上角的「登入」或「申請帳號」，開啟左側選單。")
'''
    if 'elif current_section == "future":\n    if is_authenticated():' in text:
        return text
    if old not in text:
        raise RuntimeError("找不到未來分析頁面的預期入口；已停止，未覆寫檔案。")
    return text.replace(old, new, 1)


def integrate_auth_ui(text: str, auth_block: str, sidebar_block: str) -> str:
    if "def is_authenticated() -> bool:" not in text:
        marker = "# ============================================================\n# App entry point\n"
        if marker not in text:
            raise RuntimeError("找不到 App entry point，無法安全插入登入 UI。")
        text = text.replace(marker, auth_block + sidebar_block + marker, 1)

    old_entry = "render_header_and_search()\ncurrent_section = render_site_navigation()"
    desired_entry = (
        "render_account_review_action()\n"
        'auth_cookie_manager = stx.CookieManager(key="stock_trading_auth_cookie_manager")\n'
        "_restore_auth_from_cookie(auth_cookie_manager)\n"
        "_flush_auth_cookie_op(auth_cookie_manager)\n"
        "render_header_and_search()\n"
        "render_auth_header_actions()\n"
        "current_section = render_site_navigation()\n"
        "render_auth_panel()\n"
        "sync_sidebar_visibility(current_section)"
    )
    if "render_account_review_action()" not in text:
        if old_entry not in text:
            raise RuntimeError("找不到首頁入口呼叫列，無法接上登入、登出與帳號申請 UI。")
        text = text.replace(old_entry, desired_entry, 1)
    else:
        required = (
            "_restore_auth_from_cookie(auth_cookie_manager)",
            "_flush_auth_cookie_op(auth_cookie_manager)",
            "render_auth_header_actions()",
            "render_auth_panel()",
            "sync_sidebar_visibility(current_section)",
        )
        missing = [item for item in required if item not in text]
        if missing:
            raise RuntimeError("偵測到不完整的登入入口整合，缺少：" + ", ".join(missing) + "。已停止，避免重複插入。")
    return text


def patch_email_agent(target: str, source: str) -> str:
    if "def send_account_review_request(" in target and "def send_account_request_confirmation(" in target:
        return target
    block = extract_between(
        source,
        "    def send_account_review_request(\n",
        "    def send_subscription_welcome(",
        "帳號審核 Email 方法",
    )
    marker = "    def send_subscription_welcome("
    if marker not in target:
        raise RuntimeError("email_agent.py 找不到 send_subscription_welcome() 插入點；已停止，未覆寫檔案。")
    return target.replace(marker, block + marker, 1)


def ensure_requirement(text: str) -> str:
    requirement = "extra-streamlit-components>=0.1.71,<0.2.0"
    if requirement in text:
        return text
    return text.rstrip() + "\n" + requirement + "\n"


def syntax_check(files: dict[str, str]) -> None:
    for name in ("stock_dashboard.py", "email_agent.py", "auth_service.py", "account_provisioning.py"):
        content = files.get(name)
        if content is None:
            continue
        try:
            ast.parse(content, filename=name)
        except SyntaxError as exc:
            raise RuntimeError(
                f"修正後 {name} 未通過 Python 語法檢查（第 {exc.lineno} 行）：{exc.msg}。未寫入任何檔案。"
            ) from exc


def write_atomically(path: Path, content: str) -> None:
    temp = path.with_name(path.name + ".local-login-repair.tmp")
    temp.write_text(content, encoding="utf-8", newline="\n")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="修復本地 stock_trading 專案的登入與自動帳號申請功能")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent),
                        help="本機 stock_dashboard 專案資料夾；預設為此腳本所在資料夾")
    args = parser.parse_args()
    root = Path(args.project_dir).expanduser().resolve()

    for required in ("stock_dashboard.py", "email_agent.py", "requirements.txt"):
        if not (root / required).is_file():
            say(f"錯誤：{root} 找不到 {required}。")
            say("請指定包含 stock_dashboard.py、email_agent.py、requirements.txt 的專案根目錄。")
            return 2

    say(f"本機專案：{root}")
    say(f"參考來源：GitHub {REPO} commit {COMMIT}（只讀下載，不會寫入 GitHub）")
    say("第 1/4 步：以 HTTPS GET 下載登入模組與參考程式碼……")
    remote: dict[str, str] = {}
    try:
        for name in (*SUPPORT_FILES, "stock_dashboard.py", "email_agent.py"):
            remote[name] = download_text(name)
            say(f"  已取得 {name}")
    except Exception as exc:
        say("下載失敗：" + str(exc))
        say("尚未修改本機任何檔案。")
        return 3

    try:
        dashboard = (root / "stock_dashboard.py").read_text(encoding="utf-8-sig")
        email_agent = (root / "email_agent.py").read_text(encoding="utf-8-sig")
        requirements = (root / "requirements.txt").read_text(encoding="utf-8-sig")

        # 保護措施：限定對本專案預期架構進行移植，不以 main 的整份 Dashboard 覆蓋 finance-data。
        for anchor in ("def render_header_and_search()", "def render_site_navigation()", "def render_future_workspace()"):
            if anchor not in dashboard:
                raise RuntimeError(f"本機 stock_dashboard.py 不符合預期的 finance-data 架構，缺少 {anchor!r}。未寫入任何檔案。")
        if "def _watch_remove(code: str) -> None:" not in dashboard:
            raise RuntimeError("本機 Dashboard 找不到 _watch_remove()；未寫入任何檔案。")

        main_dashboard = remote["stock_dashboard.py"]
        auth_block = extract_between(
            main_dashboard,
            "def is_authenticated() -> bool:",
            "def sync_sidebar_visibility(section: str) -> None:",
            "登入、Cookie、帳號申請與審核 UI",
        )
        sidebar_start = "def sync_sidebar_visibility(section: str) -> None:"
        app_marker = "# ============================================================\n# App entry point\n"
        sidebar_block = extract_between(main_dashboard, sidebar_start, app_marker, "側邊欄同步函式")

        patched_dashboard = add_cloud_secret_keys(dashboard)
        patched_dashboard = add_imports(patched_dashboard)
        patched_dashboard = guard_add_watchlist(patched_dashboard)
        patched_dashboard = guard_remove_watchlist(patched_dashboard)
        patched_dashboard = integrate_auth_ui(patched_dashboard, auth_block, sidebar_block)
        patched_dashboard = guard_future_page(patched_dashboard)

        patched_email_agent = patch_email_agent(email_agent, remote["email_agent.py"])
        patched_requirements = ensure_requirement(requirements)
        outputs = {
            "stock_dashboard.py": patched_dashboard,
            "email_agent.py": patched_email_agent,
            "requirements.txt": patched_requirements,
            **{name: remote[name] for name in SUPPORT_FILES},
        }
        syntax_check(outputs)

        say("第 2/4 步：所有修正內容已通過 Python 語法檢查。")
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = root / f"_local_login_repair_backup_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        targets = list(outputs.keys())
        for name in targets:
            target_path = root / name
            if target_path.exists():
                shutil.copy2(target_path, backup_dir / name)
        say(f"第 3/4 步：已備份原始檔案至 {backup_dir}")

        # 所有遠端下載、錨點檢查及語法檢查均完成後，才開始寫入本機。
        for name, content in outputs.items():
            write_atomically(root / name, content)
        say("第 4/4 步：本機登入修正已完成。")
        say("")
        say("已修正／補回：")
        for name in outputs:
            say(f"  - {name}")
        say("沒有修改 streamlit_app.py、沒有修改 .streamlit/secrets.toml，也沒有執行 git commit/push。")
        say("下一步：安裝 requirements.txt，依 ACCOUNT_LOGIN_SETUP.md 與 ACCOUNT_PROVISIONING_SETUP.md 檢查本機 Secrets，再啟動 Streamlit。")
        return 0
    except Exception as exc:
        say("修正中止：" + str(exc))
        say("若尚未建立備份資料夾，代表目標檔案尚未寫入；若已建立備份，原始檔案位於該資料夾，可用來還原。")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
