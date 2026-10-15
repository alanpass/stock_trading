# -*- coding: utf-8 -*-
"""把公開 JSON 發布到 GitHub finance-data 分支（Contents API，不需要本機 git）。

必要環境變數：GITHUB_TOKEN（只需 Contents: write 權限）。
可選：GITHUB_REPOSITORY、FINANCE_DATA_BRANCH。
"""
from __future__ import annotations

import base64
import os

from publish_research_data import BRANCH, REPO, api_request


def publish_json(remote_path: str, content: str, message: str) -> dict:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        return {"ok": False, "skipped": True, "reason": "GITHUB_TOKEN 未設定"}
    api_url = f"https://api.github.com/repos/{REPO}/contents/{remote_path}"
    sha = None
    try:
        current = api_request("GET", f"{api_url}?ref={BRANCH}", token)
        sha = current.get("sha")
        old = base64.b64decode(current.get("content", "").encode()).decode("utf-8", errors="replace")
        if old.strip() == content.strip():
            return {"ok": True, "no_change": True, "path": remote_path}
    except Exception as exc:
        if "HTTP 404" not in str(exc):
            raise
    body = {"message": message, "content": base64.b64encode(content.encode("utf-8")).decode("ascii"), "branch": BRANCH}
    if sha:
        body["sha"] = sha
    result = api_request("PUT", api_url, token, body)
    return {"ok": True, "repo": REPO, "branch": BRANCH, "path": remote_path, "commit": (result.get("commit") or {}).get("sha")}
