# -*- coding: utf-8 -*-
"""Email Agent v55 簡易設定檢查工具。"""
from pathlib import Path
import json
from email_agent import EmailAgent

base = Path(__file__).resolve().parent
agent = EmailAgent(base)
print(json.dumps(agent.status(), ensure_ascii=False, indent=2))
if not agent.status().get("configured"):
    raise SystemExit("Email Agent 尚未設定完成，請先建立 .env。")
print("Email Agent 設定完整；請用 python email_agent.py --send-latest 測試寄送。")
